
from enum import Enum


from exps import *


def get_tickers(train_symbols, dtype, decay_factor, n_context_days):
    X_train = []
    y_train = []

    for symbol in train_symbols:
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
        X_train.append(X)
        y_train.append(y)
        print(X.shape, y.shape)
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    print(X_train.shape, y_train.shape)
    return X_train, y_train


def multi_main(train_symbols, eval_symbols, start_date, end_date, decay_factor, n_context_days, dtype, model_type, with_pos_embed):

    # Split data into training and testing sets
    # split_index = int(len(X) * 0.8)
    # X_train, X_test = X[:split_index], X[split_index:]
    # y_train, y_test = y[:split_index], y[split_index:]

    X_train, y_train = get_tickers(train_symbols, dtype, decay_factor, n_context_days)
    X_test, y_test = get_tickers(eval_symbols, dtype, decay_factor, n_context_days)

    if model_type == 'LSTM':#ModelType.LSTM:
        model = train_lstm_model(X_train, y_train)
    elif model_type == 'TRANSFORMER':#ModelType.TRANSFORMER:
        model = train_model(X_train, y_train, with_pos_embed)
    else:
        raise NotImplementedError()
    
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


    data = [[f"{model_type}_multi_{n_context_days}_{dtype}_pos_embed_{with_pos_embed}", mae, mse, rmse, mape]]

    df = pd.DataFrame(data, 
                      columns=["Evaluation_type", 
                               "Mean Absolute Error", 
                               "Mean Squared Error", 
                               "Root Mean Squared Error", 
                               "Mean Absolute Percentage Error"
                               ]
                               )

    if os.path.exists('multi_eval_results.csv'):
        results_df = pd.read_csv('multi_eval_results.csv')
        pd.concat([results_df, df]).to_csv('multi_eval_results.csv', index=False)
    else:
        df.to_csv('multi_eval_results.csv', index=False)

    return y_test, y_pred






class ModelType(Enum):
    LSTM = "LSTM"
    TRANSFORMER = "TRANSFORMER"




# AMZN
# FB
# AAPL
# TSLA
# GOOGL
# MSFT
# NFLX
# META


train_symbols = ['AMZN', 'META', 'AAPL', 'TSLA', 'MSFT']
eval_symbols = ['GOOGL', 'NFLX']

decay_factor = 0.9
n_context_days = 5

model_type = "TRANSFORMER"
with_pos_embed = False

if __name__ == '__main__':

    for dtype in data_types:

        y_test, y_pred = multi_main(
            train_symbols,
            eval_symbols,
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
        plt.savefig(f'plots/{model_type}_multi_{n_context_days}_{dtype}_pos_embed_{with_pos_embed}.png')

