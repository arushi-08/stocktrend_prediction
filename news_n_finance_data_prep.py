import pandas as pd
import bs4 as bs
import pickle
import requests
import yfinance as yf
import numpy as np
import ast
from sentence_transformers import SentenceTransformer

df = pd.read_csv('news_df_processed_2.csv')

df['date_processed'] = df['Date'].str.split(" ").apply(lambda x: x[0])

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.strip())
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers

sp500_company_tickers = save_sp500_tickers()

sp500_company_tickers_in_kaggle_df = ' '.join(
    df[df['Ticker'].isin(sp500_company_tickers)]['Ticker'].unique().tolist()
)

numerical_df = yf.download(sp500_company_tickers_in_kaggle_df, "2018-01-01", "2020-12-31")

def calculate_rsi(data, period=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()[:period+1]
    avg_loss = loss.rolling(window=period, min_periods=period).mean()[:period+1]
    
    for i in range(period+1, len(data)):
        avg_gain = pd.concat(
            [avg_gain, 
             pd.Series([(avg_gain.iloc[-1] * (period - 1) + gain.iloc[i]) / period], index=[gain.index[i]])
            ]
        )
        avg_loss = pd.concat(
            [avg_loss,
                pd.Series([(avg_loss.iloc[-1] * (period - 1) + loss.iloc[i]) / period], index=[loss.index[i]])
            ]
        )

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

n_days = 50
sp500_tickers = sorted(sp500_company_tickers_in_kaggle_df.split(" "))

for comp in sp500_tickers:
    numerical_df[('RSI',    comp)] = calculate_rsi(numerical_df[('Close',    comp)], 50)
    numerical_df[('EMA',    comp)] = numerical_df[('Close',    comp)].ewm(span=n_days, adjust=False).mean()
    numerical_df[('SMA',    comp)] = numerical_df[('Close',    comp)].rolling(window=14).mean()
    temp_12 = df[('Close',    comp)].ewm(span=12, adjust=False).mean()
    temp_26 = df[('Close',    comp)].ewm(span=26, adjust=False).mean()
    numerical_df[('MACD',    comp)] = temp_12 - temp_26



with open('sp500_tickers.txt', 'w') as file:
    for item in sp500_tickers:
        file.write(f"{item}\n")  # Add a newline character after each item for formatting


news_df = df[df['Ticker'].isin(sp500_company_tickers)]
news_df['date_processed'] = news_df['Date'].str.split(" ").apply(lambda x: x[0]) 
news_df['News_processed'] = news_df['News'].apply(lambda x: ". ".join(ast.literal_eval(x)))

news_df_og = news_df.copy()
news_df = news_df[['date_processed', 'Ticker', 'News_processed']]
news_df.columns = ['Date', 'Ticker', 'News']
news_df = news_df.groupby(['Date', 'Ticker'])['News'].apply('; '.join).reset_index()
news_df['Date'] = pd.to_datetime(news_df['Date'])
news_df.set_index(['Date', 'Ticker'], inplace=True)
df_unstacked = news_df.unstack(level='Ticker')
df_unstacked.columns = pd.MultiIndex.from_tuples([('News', x[1]) for x in df_unstacked.columns])


df_unstacked = df_unstacked.reset_index().sort_values(by= 'Date',ascending=False)
df_unstacked = df_unstacked[df_unstacked['Date'] > '2018-01-01']
news_df = df_unstacked.set_index('Date')
final_df = numerical_df.reset_index().merge(news_df.reset_index(), on='Date')
final_df.to_csv('news_n_finance_data.csv', index=False)


# if we have news_n_finance_data and sp500_tickers.txt, just run below code:

df = pd.read_csv('news_n_finance_data.csv', header=[0, 1], skipinitialspace=True)

with open('sp500_tickers.txt', 'r') as file:
    sp500_tickers = file.read()

sp500_tickers = list(filter(None, sp500_tickers.split('\n')))

for comp in sp500_tickers:
    temp_12 = df[('Close',    comp)].ewm(span=12, adjust=False).mean()
    temp_26 = df[('Close',    comp)].ewm(span=26, adjust=False).mean()
    df[('MACD',    comp)] = temp_12 - temp_26

# df=df.fillna(method='ffill')

numerical_cols = ['Close', 'Open', 'Adj Close', 'High', 'Low', 'Volume','RSI', 'EMA', 'SMA', 'MACD']

columns_to_normalize = []
for col in numerical_cols:
    for comp in sp500_tickers:
        columns_to_normalize.append((col, comp))
        

companies = ['AAPL','META', 'GOOGL','AMZN', 'MSFT']

company = 'AMZN'

company_df = pd.DataFrame(columns=numerical_cols)

for col in numerical_cols:
    company_df[col] = df[(col, company)]

company_df['News'] = df[('News', company)]
company_df['Date'] = df[('Date', 'Unnamed: 0_level_1')]
company_df = company_df.dropna(subset='News')


model_id = "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis"

model = SentenceTransformer(model_id)

news_list = company_df['News'].tolist()
embeddings = model.encode(news_list)

price_features = company_df[numerical_cols].to_numpy().reshape(-1, len(numerical_cols))
news_list = company_df['News'].tolist()

data = np.concatenate((price_features, embeddings), axis=1)
data.to_csv('news_n_finance_data_prepared.csv', index=False)
        
