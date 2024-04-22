
from tensorflow import keras
from keras import layers
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import lzma
import pickle
import numpy as np
import tensorflow as tf
import os

import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        # apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


# Step 1: Fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data


def calculate_rsi(data, period=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, span=50):
    return data.ewm(span=span, adjust=False).mean()

def calculate_sma(data, window=14):
    return data.rolling(window=window).mean()

def calculate_macd(data, span_short=12, span_long=26):
    ema_short = calculate_ema(data, span=span_short)
    ema_long = calculate_ema(data, span=span_long)
    return ema_short - ema_long


def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test, verbose=1)
    return loss


def compute_technical_indicators(data):
    data['RSI'] = calculate_rsi(data['Close'], period=14)
    data['EMA'] = calculate_ema(data['Close'], span=50)
    data['SMA'] = calculate_sma(data['Close'], window=14)
    data['MACD'] = calculate_macd(data['Close'], span_short=12, span_long=26)
    data_filled = data.fillna(data.mean())
    return data_filled

# Step 3: Prepare data
def prepare_data(data, news_columns=[], decay_factor=0, n_context_days = 3):

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data[['Close', 'RSI', 'EMA', 'SMA', 'MACD']])
    scaled_data = data[['Close', 'RSI', 'EMA', 'SMA', 'MACD']+news_columns].values
    X, y = [], []
    for i in range(n_context_days, len(data)):
        X.append(scaled_data[i-n_context_days:i])
        y.append(scaled_data[i, 0])  # Closing price
    X, y = np.array(X), np.array(y)
    return X, y,# scaler


def prepare_news_embed_data(symbol, data_with_technical_indicators):
    if symbol == 'META':
        symbol = 'FB'
    with lzma.open(f'datasets/ticker_data/embeddings/{symbol}.xz') as rf:
        data = pickle.load(rf)
    embedding_df = pd.DataFrame.from_dict(data, orient='index')
    embedding_df.index = pd.to_datetime(embedding_df.index)
    data_w_news = data_with_technical_indicators.join(embedding_df)
    data_w_news = data_w_news.fillna(0)
    return data_w_news

def prepare_news_sentiment_data(symbol, data_with_technical_indicators):
    if symbol == 'META':
        symbol = 'FB'
    with lzma.open(f'datasets/ticker_data/sentiments/{symbol}.xz') as rf:
        sentiment_data = pickle.load(rf)

    sentiment_df = pd.DataFrame.from_dict(sentiment_data, orient='index')
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    data = data_with_technical_indicators.join(sentiment_df)
    data['decayed_score'] = apply_decay(data, decay_factor)
    data['label'] = data['label'].ffill()
    data['decayed_score'] = np.where(
        data['label']=='neutral', 0, 
        data['decayed_score']
    )
    data['decayed_score'] = np.where(
        data['label']=='negative', 
        -data['decayed_score'], 
        data['decayed_score']
    )
    data = data.drop(columns=['score', 'label']).dropna()

    return data


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
#     inputs = tf.expand_dims(inputs, axis=1)
#     print(inputs.shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res


class Time2Vector(keras.Layer):
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic = self.add_weight(name='weight_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic = self.add_weight(name='bias_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

  def call(self, x):
    x = tf.math.reduce_mean(x[:,:,:4], axis=-1) # Convert (batch, seq_len, 5) to (batch, seq_len)
    time_linear = self.weights_linear * x + self.bias_linear
    time_linear = tf.expand_dims(time_linear, axis=-1) # (batch, seq_len, 1)
    
    time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1) # (batch, seq_len, 1)
    return tf.concat([time_linear, time_periodic], axis=-1) # (batch, seq_len, 2)
  

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    with_pos_embed=False
):
    
    inputs = keras.Input(shape=(input_shape))

    if with_pos_embed:
        # Create positional encoding
        positional_encoding_layer = PositionalEncoding(input_shape[0], input_shape[1])
        x = positional_encoding_layer(inputs)

    else:
        x = inputs

    # time_embedding = Time2Vector(input_shape[0])
    # x = time_embedding(inputs)
    # x = layers.Concatenate(axis=-1)([inputs, x])

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    return keras.Model(inputs, outputs)

def train_model(x_train, y_train, with_pos_embed):
    input_shape = x_train.shape[1:]

    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=3,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.35,
        with_pos_embed=with_pos_embed
    )

    model.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["mean_squared_error", "mean_absolute_error", "mape"],
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(
        x_train,
        y_train,
        validation_split=0.25,
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
    )
    return model


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=200, activation='relu', input_shape=input_shape),
#         Dropout(0.2),
        Dense(units=50, activation='relu'),
#         Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mae", "mape"])
    return model



def train_lstm_model(x_train, y_train):
    # print("check x_train", x_train.shape[1:])
    input_shape = x_train.shape[1:]

    model = build_lstm_model(input_shape)
    
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=25,
        batch_size=128,
        callbacks=callbacks,
    )
    return model


# Step 7: Predict next day's closing price
def predict_next_day_price(model, last_data_point, scaler=None):
    last_data_point = last_data_point.reshape((1, last_data_point.shape[0], last_data_point.shape[1]))
    predicted_scaled_price = model.predict(last_data_point)
    if scaler:
        predicted_price = scaler.inverse_transform([[predicted_scaled_price[0][0], 0, 0, 0, 0]])[0][0]
    else:
        predicted_price = predicted_scaled_price
        
    return predicted_price

def apply_decay(df, decay_factor):
    series = df['score']
    mask = series.isna()
    # Calculate the distance since the last non-NaN value
    distance = mask.groupby((mask != mask.shift()).cumsum()).cumcount() + 1
    # Apply decay factor to the filled values
    decayed_values = series.ffill() * (decay_factor ** distance)
    df['decayed_score'] = decayed_values
    return np.where(df['score'].isna(), df['decayed_score'], df['score'])
    

def main(symbol, start_date, end_date, decay_factor, n_context_days, dtype, model_type, with_pos_embed):

    data = fetch_stock_data(symbol, start_date, end_date)
    data_with_technical_indicators = compute_technical_indicators(data)

    if dtype == 'numerical_n_news_embed':
        final_df = prepare_news_embed_data(symbol, data_with_technical_indicators)
        news_columns = list(final_df.drop(columns=data_with_technical_indicators.columns).columns)
        print("check news_columns", news_columns)

    elif dtype=='numerical_n_news_sentiment':
        final_df = prepare_news_sentiment_data(symbol, data_with_technical_indicators)
        news_columns=['decayed_score']
    else:
        final_df = data_with_technical_indicators
        news_columns=[]
        
    # X, y, scaler = prepare_data(data_w_sentiment)
    X, y = prepare_data(final_df, news_columns, 
                        decay_factor=decay_factor, 
                        n_context_days=n_context_days)


    # # AMZN
    # symbol = 'AMZN'
    # data = fetch_stock_data(symbol, "2018-01-01", end_date)
    # data_with_technical_indicators = compute_technical_indicators(data)

    # with lzma.open(f'datasets/ticker_data/sentiments/{symbol}.xz') as rf:
    #     sentiment_data = pickle.load(rf)

    # sentiment_df = pd.DataFrame.from_dict(sentiment_data, orient='index')
    # sentiment_df.index = pd.to_datetime(sentiment_df.index)
    # data_w_sentiment = data_with_technical_indicators.join(sentiment_df)

    # X, y, scaler = prepare_data(data_w_sentiment)
    # X_test, y_test = prepare_data(data_w_sentiment, news_columns=['decayed_score'], decay_factor=decay_factor)


    # Split data into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if model_type == 'lstm':
        model = train_lstm_model(X_train, y_train)
    elif model_type == 'transformer':
        model = train_model(X_train, y_train, with_pos_embed)
    loss = evaluate_model(model, X_test, y_test)
    print("Test Loss:", loss)  # Test loss: Represents the average loss (error) between the predicted values and the actual values. Lower values indicate better performance.
    
    # Predict next day's closing price
    last_data_point = X_test[-1]
    next_day_price = predict_next_day_price(model, last_data_point)
    print("Predicted Next Day's Closing Price:", next_day_price)

    # Calculate additional evaluation metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)

    # print("Mean Absolute Error (MAE):", mae)  # Mean Absolute Error (MAE): Average magnitude of the errors in the predictions. Lower values indicate better performance.
    # print("Mean Squared Error (MSE):", mse)  # Mean Squared Error (MSE): Average of the squared differences between the predicted values and the actual values. Lower values indicate better performance.
    # print("Root Mean Squared Error (RMSE):", rmse)  # Root Mean Squared Error (RMSE): Standard deviation of the residuals (prediction errors). Lower values indicate better performance.


    data = [[f"{model_type}_{symbol}_{n_context_days}_{dtype}_pos_embed_{with_pos_embed}", mae, mse, rmse, mape]]

    df = pd.DataFrame(data, 
                      columns=["Evaluation_type", 
                               "Mean Absolute Error", 
                               "Mean Squared Error", 
                               "Root Mean Squared Error", 
                               "Mean Absolute Percentage Error"
                               ]
                               )

    if os.path.exists('eval_results.csv'):
        results_df = pd.read_csv('eval_results.csv')
        pd.concat([results_df, df]).to_csv('eval_results.csv', index=False)
    else:
        df.to_csv('eval_results.csv', index=False)

    return y_test, y_pred


start_date = '2015-01-01'
end_date = '2020-01-01'
decay_factor = 0.9 
n_context_days = 5

data_types = ['numerical', 'numerical_n_news_embed', 'numerical_n_news_sentiment']

if __name__ == "__main__":
    # Fetch data
    symbol = 'AAPL'  # Example symbol
    dtype = 'numerical_n_news_sentiment' # [numerical, numerical_n_news_embed, numerical_n_news_sentiment]
    with_pos_embed = True
    model_type = 'lstm'

    for dtype in data_types:

        y_test, y_pred = main(
            symbol, 
            start_date, 
            end_date, 
            decay_factor, 
            n_context_days, 
            dtype,
            model_type,
            with_pos_embed
            )

        # Visualize model predictions
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual Stock Prices')
        plt.plot(y_pred, label='Predicted Stock Prices')
        plt.title('Actual vs Predicted Stock Prices')
        plt.xlabel('Time')
        plt.ylabel(f'Stock Price {model_type.upper()} Model')
        plt.legend()
        # plt.show()
        plt.savefig(f'plots/{model_type}_{symbol}_{n_context_days}_{dtype}_pos_embed_{with_pos_embed}.png')


# aapl_3_numerical_pos_embed_false
# Mean Absolute Error (MAE): 1.1314397257162756
# Mean Squared Error (MSE): 1.9890221097088925
# Root Mean Squared Error (RMSE): 1.4103269513516687


# aapl_3_numerical_n_news_embed_pos_embed_false




# ------------------------------------------------------------------------------------------------------------------------

# Numerical + News Sentiment Data

# without positional encoding

# context_days = 5
# Mean Absolute Error (MAE): 2.972464000274989
# Mean Squared Error (MSE): 13.309710675209596
# Root Mean Squared Error (RMSE): 3.648247617036102

# context_days = 10
# Mean Absolute Error (MAE): 6.889542538186778
# Mean Squared Error (MSE): 62.022623332968344
# Root Mean Squared Error (RMSE): 7.875444326066202



# time2vec layer

# Mean Absolute Error (MAE): 7.229265481556365
# Mean Squared Error (MSE): 71.16695418183684
# Root Mean Squared Error (RMSE): 8.436050864109156


# with positional encoding

# context_days = 3
# Mean Absolute Error (MAE): 2.6753567306573642
# Mean Squared Error (MSE): 11.545681378771157
# Root Mean Squared Error (RMSE): 3.3978936679612497

# context_days = 5
# Mean Absolute Error (MAE): 3.1986416427667392
# Mean Squared Error (MSE): 16.535260759628358
# Root Mean Squared Error (RMSE): 4.066357185446005

# context_days = 10
# Mean Absolute Error (MAE): 4.263220358585966
# Mean Squared Error (MSE): 28.1058884893803
# Root Mean Squared Error (RMSE): 5.301498702195475


# Training on 5 companies data
# Testing on 2 companies data
# show that this model will do well for companies without much historical data

# technicality:
# LSTM
# Dataset level - 3 exps - numerical / numerical + news / numerical + sentiment

# Transformers
# Dataset level - 3 exps - numerical / numerical + news / numerical + sentiment
# use best of above - 2 exps - with positional encoding / without positional encoding
# context days - 3 exps - days = 3, 5, 10 (hopefully with 10th day with positional encoding does better)
