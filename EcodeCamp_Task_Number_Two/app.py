from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

app = Flask(__name__)

def download_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data


def preprocess_data(data):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    return scaled_data, scaler


def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def build_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
  
    stock_data = download_stock_data(ticker, '2020-01-01', '2023-01-01')
    scaled_data, scaler = preprocess_data(stock_data)
    
  
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[:training_data_len, :]
    X_train, y_train = create_dataset(train_data, 60)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = build_model(X_train)
    model.fit(X_train, y_train, batch_size=64, epochs=10)

    real_time_price = yf.download(ticker, period='1d', interval='1m')['Close'].iloc[-1]
    prediction = model.predict(np.reshape(scaler.transform([[real_time_price]]), (1, 1, 1)))
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    return render_template('index.html', predicted_price=predicted_price, ticker=ticker)

if __name__ == '__main__':
    app.run(debug=True)
