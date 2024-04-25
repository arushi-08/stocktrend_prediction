# Stock price prediction

### Course: CS2756 Principles of Datamining (Spring 2024)
### Group 12: Arushi Sharma, Lokesh Paturu, Norah Almousa

<hr />

### Environment setup

`pip install -r requirements.txt`

### Download generated embeddings and sentiments

Use this [link](https://pitt-my.sharepoint.com/:u:/g/personal/vep52_pitt_edu/EVvgnXBBHEFOjKFZoXTZhtQBnX_-5ciEBjqTBlkI32id9Q?e=J4MNqj) to download and extract it to `datasets/ticker_data`.

### Run experiments with models

LSTM vs Transformer: `python exps.py`

Model generalization experiment:
`python multi_stock_model.py`

Predicting multiple days into the future:
`python multi_day_pred.py`

### Generate embeddings and sentiments for news data

Download extracted news using [link](https://pitt-my.sharepoint.com/:u:/g/personal/vep52_pitt_edu/EQWRaswzU2xIjAtyID8kyPkBmOmcp9Te-k5gf-SChU8NCA?e=bkDx1s) and extract it to `datasets/processed`. Then run the script:

`python preprocess_news.py`

### Download raw news data

(Not recommended) Downloads more than 10GiB of data by scraping financial website. Can be run using `download_news_data.py`
