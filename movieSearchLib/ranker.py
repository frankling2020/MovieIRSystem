import pyterrier as pt
import torch
import math
import numpy as np
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util


class Ranker:
    def __init__(self, index, tokenizer):
        self.index = index
        self.tokenizer = tokenizer

        def fetch_my_feat(keyFreq, postings, entryStats, collStats):
            return postings.getFrequency()

        def fetch_my_tfidf(keyFreq, postings, entryStats, collStats):
            # Term Frequency (TF)
            tf = postings.getFrequency()
            # Number of documents in the collection
            total_docs = collStats.getNumberOfDocuments()
            # Number of documents containing the term
            doc_freq = entryStats.getDocumentFrequency()
            # Inverse Document Frequency (IDF)
            idf = math.log(total_docs / doc_freq)
            # TF-IDF calculation
            tfidf = tf * idf
            return tfidf
   
        self.query_toks = pt.rewrite.tokenise(tokeniser=lambda x: tokenizer.tokenize(x), matchop=True)

        # compare for different query expansion methods
        self.bm25 = self.query_toks >> pt.BatchRetrieve(index, wmodel="BM25")
        self.bm25_qe = self.query_toks >> pt.BatchRetrieve(index, wmodel="BM25", controls={"qe": "on", "qemodel": "Bo1"})

        # get the text of the document
        self.get_title = pt.text.get_text(index, 'title')
        self.get_category = pt.text.get_text(index, 'category')

        # compare for different retrieval models
        self.tf_title = pt.text.scorer(body_attr="title", wmodel=fetch_my_feat)
        self.tf_category = pt.text.scorer(body_attr="category", wmodel=fetch_my_feat, controls={"qe": "on", "qemodel": "Bo1"})
        self.tf_text = pt.BatchRetrieve(index, wmodel=fetch_my_feat)
        self.tf_idf_title = self.query_toks >> pt.BatchRetrieve(index, wmodel="TF_IDF")
        self.tf_idf_text = self.query_toks >> pt.BatchRetrieve(index, wmodel="TF_IDF")
        self.pl2 = self.query_toks >> pt.BatchRetrieve(index, wmodel="PL2")
        self.dlm = self.query_toks >> pt.BatchRetrieve(index, wmodel="DirichletLM")
        self.cm = self.query_toks >> pt.BatchRetrieve(index, wmodel="CoordinateMatch")
    
    def base_rankers(self):
        rankers = [self.bm25, self.bm25_qe, self.tf_idf_title, self.tf_idf_text, self.pl2, self.dlm, self.cm]
        ranker_names = ["bm25", "bm25_qe", "tf_idf_title", "tf_idf_text", "pl2", "dlm", "cm"]
        return rankers, ranker_names

    def pipeline(self, fe, enable_fe=True):
        RANK_CUTOFF = 25
        if enable_fe:
            pipeline = (self.bm25_qe % RANK_CUTOFF) >> self.get_title >> self.get_category >> (
                # 
                pt.apply.doc_features(fe.add_features)
                **
                # BM25
                self.bm25
                **
                # TF_TITLE
                self.tf_title 
                ** 
                # TF_DOC
                self.tf_text
                ** 
                # TF_IDF_TITLE
                self.tf_idf_title
                **
                # TF_IDF_DOC
                self.tf_idf_text
                **
                self.tf_category
            )
        else:
            pipeline = (self.bm25_qe % RANK_CUTOFF) >> self.get_title >> self.get_category >> (
                # BM25
                self.bm25
                **
                # TF_TITLE
                self.tf_title 
                ** 
                # TF_DOC
                self.tf_text
                ** 
                # TF_IDF_TITLE
                self.tf_idf_title
                **
                # TF_IDF_DOC
                self.tf_idf_text
                **
                self.tf_category
            )
        return pipeline
    
    def bm25_dlm_pipeline(self, cut_off=15):
        bm25_dlm_pipe = (self.bm25_qe % cut_off) >> self.dlm
        return bm25_dlm_pipe

    def pipeline_dlm(self, pipeline, cut_off=15):
        return (pipeline % cut_off) >> self.dlm


class FeatureExtractor:
    def __init__(self, model_name, index, document_ids, encoded_docs, doc_category_info, 
                encoded_genres, recognized_categories, tokenizer):
        self.tokenizer = tokenizer
        self.index = index
        self.doc_ids = document_ids
        self.encoded_docs = encoded_docs
        self.encoder = SentenceTransformer(model_name)
        self.query_emb = {}
        self.doc_category_info = doc_category_info
        self.category_to_id = {k: v for v, k in enumerate(recognized_categories)}
        self.encoded_genres = encoded_genres
        self.recognized_genres = recognized_categories

        self.inv_index = self.index.getInvertedIndex()
        self.lex_index = self.index.getLexicon()
        self.meta_index = self.index.getMetaIndex()

        self.query_doc_word_freq = defaultdict(dict)

    def get_document_categories(self, docid, query_emb):
        doc_categories = np.zeros(len(self.category_to_id))
        for category in self.doc_category_info[str(docid)]:
            if category in self.category_to_id:
                doc_categories[self.category_to_id[category]] = 1
        values = (query_emb @ self.encoded_genres.T) * doc_categories
        max_prob = values.max().item()
        return max(max_prob, 0), values.tolist()

    def doc_word_freq(self, docno, qid, query):
        if qid not in self.query_doc_word_freq:
            query_terms = self.tokenizer.tokenize(query)
            doc_word_freq = defaultdict(list)
            for term, qFreq in Counter(query_terms).items():
                le = self.lex_index.getLexiconEntry(term)
                if le is not None:
                    for posting in self.inv_index.getPostings(le):
                        docno = posting.getId()
                        doc_word_freq[docno].append(posting.getFrequency())
            self.query_doc_word_freq[qid] = doc_word_freq
        return self.query_doc_word_freq[qid].get(docno, [0])

    def weighted_unigram_score(self, word_freq):
        sorted_word_counts = sorted(word_freq, reverse=True)
        num_tokens = len(sorted_word_counts)
        weighted_sum = 0
        weighted_sum = np.sum([sorted_word_counts[i] * math.log(i + 2) for i in range(len(sorted_word_counts))])
        weighted_score = weighted_sum / num_tokens
        return weighted_score

    @torch.no_grad()
    def add_features(self, row):
        docid = int(row["docid"])
        docno = int(row["docno"])
        query = row["query"]
        qid = row["qid"]
        query_emb = None
        if qid not in self.query_emb:
            query_emb = self.encoder.encode(query, normalize_embeddings=True)
            self.query_emb[qid] = query_emb
        else:
            query_emb = self.query_emb[qid]
        doc_emb = self.encoded_docs[self.doc_ids.index(docno)]
        f1 = util.dot_score(query_emb, doc_emb).item()
        f2 = len(self.tokenizer.tokenize(query))
        content = row["title"]
        f3 = len(self.tokenizer.tokenize(content))
        f4 = self.index.getDocumentIndex().getDocumentLength(docid)
        doc_word_count = self.doc_word_freq(docno, qid, query)
        f5 = self.weighted_unigram_score(doc_word_count)
        f6, f7 = self.get_document_categories(docno, query_emb)
        return np.array([f1, f2, f3, f4, f5, f6] + list(f7)).clip(min=0)
    
    @property
    def feature_names(self):
        return ["dot_score", "query_length", "doc_length", "doc_length", "weighted_unigram_score", "category_similarity"]
