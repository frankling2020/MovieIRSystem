from tqdm import tqdm
import re

def read_dataset_df(dataset_df, movie_info, doc_tag_info, doc_category_info, tokenizer, with_tag=True):
    for i, row in tqdm(dataset_df.iterrows()):
        x = {}
        docid = row["movieId"]
        x["docno"] = str(docid)
        x["text"] = movie_info.get(docid, "")
        if with_tag: 
            x["text"] += " " + tokenizer.unified_preprocess(doc_tag_info[str(docid)])
        x["title"] = row["title"]
        x["category"] = " ".join(doc_category_info[str(docid)])
        yield x


def load_data(dataset_df, qcol="query"):
    df = dataset_df.copy().dropna()
    queries = df[qcol].astype(str).unique()
    df["docno"] = df["docid"].apply(lambda x: str(x))
    df["label"] = df["rel"]
    df["qid"] = df["qid"].apply(lambda x: str(x))
    topics = df[[qcol, "qid"]].copy().drop_duplicates()
    topics.columns = ["query", "qid"]
    topics = topics.astype({"qid": str})
    topics["query"] = topics["query"].apply(lambda x: re.sub(r"'", " ", x))
    qrels = df[["qid", "docno", "label", "query"]].copy()
    return topics, qrels