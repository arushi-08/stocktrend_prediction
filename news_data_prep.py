import pandas as pd

raw_analyst_ratings_df = pd.read_csv('raw_analyst_ratings.csv')
nasdaq_df = pd.read_csv('nasdaq_screener_1711229230838.csv')

import requests
from bs4 import BeautifulSoup

news = {}
for idx, row in raw_analyst_ratings_df.iterrows():
    
    url = row['url']
    date = row['date']

    response = requests.get(url)

    if response.status_code == 200:
        html_content = response.text
#         print(html_content)
    else:
        print("Failed to retrieve the content")

    soup = BeautifulSoup(html_content, 'html.parser')
    
    text = soup.find('body').get_text(separator="\n", strip=True).split('\n-\n')[-1]


    ticker_news = {}

    text_list = text.split('\n')
    for i in range(len(text_list)):
        if not nasdaq_df[nasdaq_df['Symbol']==text_list[i]].empty:
            ticker_news[text_list[i]] = text_list[i+1]

    news[date] = ticker_news


news_df = pd.DataFrame.from_dict(news).reset_index().rename(columns={'index':'Ticker'})
news_df_processed = pd.melt(news_df, id_vars=['Ticker'], var_name='Date', value_name='News')
news_df_processed = news_df_processed[~news_df_processed['News'].isna()]
news_df_processed.to_csv('news_df_processed.csv', index=False)

dates_to_be_processed = list(set(raw_analyst_ratings_df['date']) - set(news_df.columns))

