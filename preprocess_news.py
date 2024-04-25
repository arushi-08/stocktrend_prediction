
import os
from bs4 import BeautifulSoup
from collections import defaultdict
from dataset import get_path
dataset_name ='miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests'
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import numpy as np
import lzma
import pickle
import torch

import warnings
warnings.filterwarnings('ignore')


ticker_dir = "datasets/ticker_data/"
os.makedirs(ticker_dir, exist_ok=True)
embeddings_dir = "datasets/ticker_data/embeddings/"
os.makedirs(embeddings_dir, exist_ok=True)
sentiments_dir = "datasets/ticker_data/sentiments/"
os.makedirs(sentiments_dir, exist_ok=True)

processed_data_dir = "datasets/processed/"
os.makedirs(processed_data_dir, exist_ok=True)
processed_count_file = os.path.join(processed_data_dir, "processed.count")

companies = ['AAPL','META', 'GOOGL','AMZN', 'MSFT', "FB", "TSLA", "NFLX"]
companies = []
companies = {c:1 for c in companies}
def is_valid_ticker(t):
    return t in companies

device='cuda:0'

csvs = [os.path.join(processed_data_dir, f) for f in sorted(os.listdir(processed_data_dir)) if f.endswith('.csv')]

import ast

from sentence_transformers import SentenceTransformer

model_id = 'paraphrase-MiniLM-L6-v2'

sentence_model = SentenceTransformer(model_id)
sentence_model.to(device)
sentence_model.eval()

task = "text-classification"
model_id = "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis"
classifier = pipeline(task, model_id, device=device)
classifier.model.eval()

sentiments = {c: {} for c in companies}
embeddings = {c: {} for c in companies}


count = 0

with torch.no_grad():
    for csv in csvs:
        print(csv)
        df = pd.read_csv(csv)
        
        # display(df)
        for row in tqdm(df.iterrows()):
            ticker = row[1]['Ticker']
            if not is_valid_ticker(ticker):
                continue
            date = row[1]['Date']
            news = row[1]['News']
            # news = json.loads(news)
            news = " ".join(ast.literal_eval(news))
            embedding = sentence_model.encode(news)
            try:
                result = classifier(news[:16384])[0]
            except Exception as e:
                print(len(news), len(row[1]['News']))
                raise e
            sentiments[ticker][date] = result
            embeddings[ticker][date] = embedding

for ticker in sentiments:
    if not sentiments[ticker]:
        continue
    with lzma.open(f"{sentiments_dir}/{ticker}.xz", "wb") as f:
        pickle.dump(sentiments[ticker], f)

    with lzma.open(f"{embeddings_dir}/{ticker}.xz", "wb") as f:
        pickle.dump(embeddings[ticker], f)
