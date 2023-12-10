from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import pyterrier as pt

import os
from .utils import read_dataset_df

class MyTokenizer():
    def __init__(self, stopwords, stem=True):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = stopwords
        self.stem = stem
        self.stemmer = None
        if stem:
            self.stemmer = PorterStemmer()

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        non_stopwords = [token for token in tokens if token not in self.stopwords]
        filtered_tokens = non_stopwords
        if self.stem:
            filtered_tokens = [self.stemmer.stem(token) for token in non_stopwords]
        return filtered_tokens
    
    def unified_preprocess(self, text):
        tokens = self.tokenize(text)
        return " ".join(tokens)


def doc_indexer(pt_index_path, movie_base_df, movie_info, doc_tag_info, doc_category_info, tokenizer, with_tag=True):
    """Index the documents in the dataset using PyTerrier. Assume the pyterrrier is initialized."""
    docids = set()
    if not os.path.exists(pt_index_path + "/data.properties"):
        indexer = pt.index.IterDictIndexer(pt_index_path, tokenizer=tokenizer, stemmer=tokenizer.stemmer)
        doc_iter = read_dataset_df(movie_base_df, movie_info, doc_tag_info, doc_category_info, tokenizer, with_tag)
        index_ref = indexer.index(doc_iter, fields=['text'], meta=['docno', 'title', 'category'])
    else:
        # if you already have the index, use it.
        index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")

    index = pt.IndexFactory.of(index_ref)
    return index

