import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from time import sleep
from tqdm import tqdm
from collections import defaultdict
from itertools import islice
import os
import json
from requests.adapters import HTTPAdapter, Retry
import traceback

from dataset import load_dataset,get_path

dataset_name ='miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests'
file_name = 'raw_analyst_ratings.csv'

raw_data_dir = os.path.join(get_path(dataset_name), 'raw_data')
os.makedirs(raw_data_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0'
}

out_file = os.path.join(raw_data_dir, f"data.txt")
lock_file = os.path.join(raw_data_dir, f"run.lock")
track_file = os.path.join(raw_data_dir, "track.num")
if os.path.isfile(lock_file):
    print("Lock file found exiting")
    exit(0)

with open(lock_file, "w") as wf:
    wf.write("locked")

raw_analyst_ratings_df = load_dataset(dataset_name, file_name)
# raw_analyst_ratings_df = pd.read_csv('raw_analyst_ratings.csv')
raw_analyst_ratings_df.sort_values(by=['date'], ascending=False, inplace=True)
num_rows = len(raw_analyst_ratings_df)
start = 0
end = num_rows//10

if os.path.isfile(track_file):
    with open(track_file, 'r') as tf:
        try:
            x = int(tf.read())
            print(x)
            if x > start and x <= end:
                start = x
        except:
            print(tf.read())
            exit(0)
            pass

print(start, end)

retries = 3
backoff = 0.5
backoff_factor = [backoff * (2 ** i) for i in range(retries)]

news_body = {}
next_start = start
try:
    for _idx, row in tqdm(islice(raw_analyst_ratings_df.iterrows(), start, end)):
        
        url = row['url']
        date = row['date']
        idx = row['Unnamed: 0']
        
        html_content = None

        for r in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    html_content = response.text
                    break
                elif response.status_code == 429:
                    sleep(5)
                elif response.status_code == 404:
                    print(f"Failed with {response.status_code} while retrieving: {url}")
                    break
                else:
                    print(f"Failed with {response.status_code} while retrieving: {url}")
                    sleep(backoff_factor[r])
            except KeyboardInterrupt as kbe:
                raise kbe
            except Exception as e:
                sleep(5)
                print(e)

        if html_content is not None:
            soup = BeautifulSoup(html_content, 'html.parser')
            raw_main_content = str(soup.find("div", {"class": "main-content-container"})).replace("\n", "").replace("\t", "")
            news_body[idx] = raw_main_content
        next_start = _idx + 1
        if _idx % 100 == 99:
            with open(out_file, 'a', encoding="utf-8") as wf:
                for k, v in news_body.items():
                    wf.write(f"{k: <10d}{v}\n")
                news_body.clear()
            with open(track_file, 'w') as wtf:
                wtf.write(f"{next_start}")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Keyboard interrupt, exiting safely")
except Exception as e:
    print(e)
    print(traceback.format_exc())
finally:
    with open(track_file, 'w') as wtf:
        wtf.write(f"{next_start}")
    os.remove(lock_file)
    with open(out_file, 'a', encoding="utf-8") as wf:
        for k, v in news_body.items():
            wf.write(f"{k: <10d}{v}\n")
        news_body.clear()
