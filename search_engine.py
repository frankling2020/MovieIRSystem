from movieSearchLib.indexing import MyTokenizer, doc_indexer
from movieSearchLib.ranker import Ranker
from movieSearchLib.ranker import FeatureExtractor
from movieSearchLib.ranker import Ranker
from movieSearchLib.constants import *

import lightgbm as lgb
import pyterrier as pt
import pandas as pd
import numpy as np
import json

from tqdm import tqdm
from collections import defaultdict

if not pt.started():
    pt.init()

stopwords = None
with open(STOPWORDS_PATH, 'r') as f:
    stopwords = f.readlines()
stopwords = [word.strip() for word in stopwords]

tokenizer = MyTokenizer(stopwords, stem=False)

# Reload the index
# Do not run this code in the backend.py file
movie_df = pd.read_csv(DATASET_PATH)
movie_base_df = pd.read_csv(MOVIE_PATH)
print("Loaded movie data")

movie_info = {}
for docid, text in zip(movie_df['id'], movie_df['data']):
    movie_info[docid] = text
print("Loaded movie info: ", len(movie_info))


tags_df = pd.read_csv(TAG_DATA_FILE).dropna()
doc_tag_info = defaultdict(str)
cnt = 0
for userid, docid, tag in tqdm(zip(tags_df['userId'], tags_df['movieId'], tags_df['tag'])):
    doc_tag_info[str(docid)] += tag + ' '

doc_category_info = None
with open(DOC_CATEGORY_INFO_PATH, 'r') as f:
    doc_category_info = json.load(f)
print("Loaded doc category info: ", len(doc_category_info))

index = doc_indexer(pt_index_path, movie_base_df, movie_info, doc_tag_info, doc_category_info, tokenizer)
ranker = Ranker(index, tokenizer)
print("Loaded index")

encoded_docs = None
with open(ENCODED_DOCUMENT_EMBEDDINGS_NPY_DATA, 'rb') as file:
    encoded_docs = np.load(file)
print("Loaded encoded docs")

document_ids = None
with open(DOCUMENT_ID_TEXT, 'r') as f:
    document_ids = f.read().splitlines()
document_ids = [int(x) for x in document_ids]
print("Loaded document ids")

recognized_genres = None
with open(RECOGNIZED_CATEGORY_PATH, 'r') as f:
    recognized_genres = f.read().splitlines()
print("Loaded recognized genres")

encoded_genres = None
with open(ENCODED_GENRE_FILE, 'rb') as file:
    encoded_genres = np.load(file)
print("Loaded encoded genres")

fe = FeatureExtractor(BI_ENCODER_MODEL_NAME, index,
    document_ids, encoded_docs, 
    doc_category_info, encoded_genres, recognized_genres, tokenizer)
print("Loaded feature extractor")

base_pipeline_fe = ranker.pipeline(fe, enable_fe=True)
lmart_l = lgb.Booster(model_file=MODEL_PARAMS)

CUT_OFF = 25
bm25_qe = ranker.bm25_qe % CUT_OFF
pipeline = (base_pipeline_fe >> pt.ltr.apply_learned_model(lmart_l, form="ltr")) ^ bm25_qe
final_pipeline = ranker.pipeline_dlm(pipeline) ^ pipeline
print("Loaded final pipeline")