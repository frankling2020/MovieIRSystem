import pyterrier as pt
import pandas as pd
import os
import json
import gzip
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import re
import math
import lightgbm as lgb

import streamlit as st


import torch

class FeatureExtractor:
    def __init__(self, model_name, index, document_ids, encoded_docs, 
            network_features, recognized_categories, doc_category_info, tokenizer):
        self.tokenizer = tokenizer
        self.index = index
        self.document_ids = document_ids
        self.encoded_docs = encoded_docs
        self.encoder = SentenceTransformer(model_name)
        self.query_emb = {}
        self.network_features = network_features
        self.doc_category_info = doc_category_info
        self.category_to_id = {k: v for v, k in enumerate(recognized_categories)}

    def get_document_categories(self, docid):
        doc_categories = [0 for _ in range(len(self.category_to_id))]
        for category in self.doc_category_info[str(docid)]:
            if category in self.category_to_id:
                doc_categories[self.category_to_id[category]] = 1
        return doc_categories
    
    @torch.no_grad()
    def add_features(self, row):
        docid = int(row["docid"])
        docno = int(row["docno"])
        content = row["query"]
        qid = row["qid"]
        f1 = len(self.tokenizer.tokenize(content))
        query_emb = None
        if qid not in self.query_emb:
            query_emb = self.encoder.encode(content, normalize_embeddings=True)
            self.query_emb[qid] = query_emb
        else:
            query_emb = self.query_emb[qid]
        doc_emb = self.encoded_docs[self.document_ids.index(docno)]
        f2 = util.dot_score(query_emb, doc_emb).item()
        content = row["title"]
        f3 = len(self.tokenizer.tokenize(content))
        f4 = self.index.getDocumentIndex().getDocumentLength(docid)
        return np.array([f1, f2, f3, f4, *self.network_features[docno].values(), *self.get_document_categories(docno)])


def read_dataset(dataset_path: str, max_docs: int = -1):
    open_func = lambda x: gzip.open(x, 'rb') if x.endswith('.gz') else open(x, 'r')
    with open_func(dataset_path) as f:
        for i, line in tqdm(enumerate(f)):
            if max_docs != -1 and i >= max_docs:
                break
            x = json.loads(line)
            # x["docid"] = str(x["docid"])
            x["docno"] = str(x["docid"])
            yield x

def load_data(dataset_path, qcol):
    df = pd.read_csv(dataset_path).dropna()
    queries = df[qcol].astype(str).unique()
    query_to_id = {q: str(i) for i, q in enumerate(queries)}
    df["qid"] = df[qcol].apply(lambda x: query_to_id[x])
    df["docno"] = df["docid"].apply(lambda x: str(x))
    df["label"] = df["rel"]
    topics = df[[qcol, "qid"]].copy().drop_duplicates()
    topics.columns = ["query", "qid"]
    topics = topics.astype({"qid": str})
    topics["query"] = topics["query"].apply(lambda x: re.sub(r"'", " ", x))
    qrels = df[["qid", "docno", "label"]].copy()
    return topics, qrels

def fetch_my_feat(keyFreq, postings, entryStats, collStats):
    return postings.getFrequency()


@st.cache_data
def data_helper():
    DATASET_PATH = "wikipedia_200k_dataset.jsonl.gz"
    TRAIN_DATA_PATH = "hw3_relevance.train.csv"
    DEV_DATA_PATH = "hw3_relevance.dev.csv"
    TEST_DATA_PATH = "hw3_relevance.test.csv"
    NETWORK_STATS_PATH = 'network_stats.csv'
    INDEX_PATH = "./main_index"
    DOC_CATEGORY_INFO_PATH = 'doc_category_info.json'
    RECOGNIZED_CATEGORY_PATH = 'recognized_categories.txt'

    tokenizer = RegexpTokenizer(r'\w+')

    train_topics, train_qrels = load_data(TRAIN_DATA_PATH, "query")
    validation_topics, validation_qrels = load_data(DEV_DATA_PATH, "query")
    test_topics, test_qrels = load_data(TEST_DATA_PATH, "query")

    ENCODED_DOCUMENT_EMBEDDINGS_NPY_DATA = 'wiki-200k-vecs.msmarco-MiniLM-L12-cos-v5.npy'
    DOCUMENT_ID_TEXT = 'document-ids.txt'

    encoded_docs = None
    with open(ENCODED_DOCUMENT_EMBEDDINGS_NPY_DATA, 'rb') as file:
        encoded_docs = np.load(file)

    document_ids = None
    with open(DOCUMENT_ID_TEXT, 'r') as f:
        document_ids = f.read().splitlines()
    document_ids = [int(x) for x in document_ids]

    recognized_categories = None
    with open(RECOGNIZED_CATEGORY_PATH, 'r') as f:
        recognized_categories = f.read().splitlines()

    doc_category_info = None
    with open(DOC_CATEGORY_INFO_PATH, 'r') as f:
        doc_category_info = json.load(f)

    doc_category_info = None
    with open(DOC_CATEGORY_INFO_PATH, 'r') as f:
        doc_category_info = json.load(f)

    network_features = {}
    networks_stats = pd.read_csv(NETWORK_STATS_PATH, index_col=0)
    for row in tqdm(networks_stats.iterrows()):
        network_features[row[1]['docid']] = row[1][1:].to_dict()
    return document_ids, encoded_docs, network_features, recognized_categories, doc_category_info, tokenizer


def obtain_pipeline(document_ids, encoded_docs, network_features, recognized_categories, doc_category_info, tokenizer):
    pt_index_path = "./main_index"
    if not os.path.exists(pt_index_path + "/data.properties"):
        # create the index, using the IterDictIndexer indexer 
        indexer = pt.index.IterDictIndexer(pt_index_path, tokenizer=tokenizer, blocks=True, verbose=True)
        NUM_DOCS = -1
        doc_iter = read_dataset(DATASET_PATH, NUM_DOCS)
        doc_iter = read_dataset(DATASET_PATH, NUM_DOCS)
        index_ref = indexer.index(doc_iter, fields=['text'], meta=['docno', 'title'])
    else:
        # if you already have the index, use it.
        index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")

    index = pt.IndexFactory.of(index_ref)
    
    query_toks = pt.rewrite.tokenise(tokeniser=lambda x: tokenizer.tokenize(x), matchop=True)
    bm25 = query_toks >> pt.BatchRetrieve(index, wmodel="BM25")
    get_title = pt.text.get_text(index, 'title')
    tf_title = pt.text.scorer(body_attr="title", wmodel=fetch_my_feat)
    tf_text = pt.BatchRetrieve(index, wmodel=fetch_my_feat)
    tf_idf_title = query_toks >> pt.BatchRetrieve(index, wmodel="TF_IDF")
    tf_idf_text = query_toks >> pt.BatchRetrieve(index, wmodel="TF_IDF")
    pl2 = query_toks >> pt.BatchRetrieve(index, wmodel="PL2")
    dlm = query_toks >> pt.BatchRetrieve(index, wmodel="DirichletLM")
    sdm = pt.rewrite.SDM()
    qe = pt.rewrite.Bo1QueryExpansion(index)
    cm = query_toks >> pt.BatchRetrieve(index, wmodel="CoordinateMatch")

    fe = FeatureExtractor('sentence-transformers/msmarco-MiniLM-L12-cos-v5', 
                index, document_ids, encoded_docs, network_features, recognized_categories, doc_category_info, tokenizer)

    RANK_CUTOFF = 100
    pipeline = (bm25 % RANK_CUTOFF) >> get_title >> (
        # 
        pt.apply.doc_features(fe.add_features)
        **
        # BM25
        bm25
        **
        # TF_TITLE
        tf_title 
        ** 
        # TF_DOC
        tf_text
        ** 
        # TF_IDF_TITLE
        tf_idf_title
        **
        # TF_IDF_DOC
        tf_idf_text
    )


    # this configures LightGBM as LambdaMART
    file_name = "./lmart_l.txt"

    lmart_l = lgb.Booster(model_file=file_name)
    lmart_l_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_l, form="ltr")
    return lmart_l_pipe



if __name__ == "__main__":
    if not pt.started():
        pt.init()
    lmart_l_pipe = obtain_pipeline(*data_helper())
    st.title('Wikipedia Search')
    search = st.text_input('Enter your search:')
    if search:
        retrieved_docnos = lmart_l_pipe.search(search)[['docno', 'title']].head(10).reset_index(drop=True)
        retrieved_docnos['url'] = retrieved_docnos['docno'].apply(lambda x: f'https://en.wikipedia.org/wiki?curid={x}')
        st.write(retrieved_docnos)