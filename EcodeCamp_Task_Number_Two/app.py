import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
import requests
import math
from datetime import datetime
import os

# Polygon.io API key
API_KEY = '27DzQ9ENavxTqAh6IaO4kFVZZQnc3uDk'

# Function to fetch stock data from Polygon.io
def get_polygon_data(symbol):
    if symbol == "BTC-USD":
        symbol = "X:BTCUSD"
    elif symbol == "ETH-USD":
        symbol = "X:ETHUSD"

    url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2004-01-01/{datetime.now().date()}?adjusted=true&sort=asc&limit=5000&apiKey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    if 'results' not in data:
        raise ValueError(f"Failed to fetch data for {symbol}. Check the symbol or API response.")
    
    df = pd.DataFrame(data['results'])
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('date', inplace=True)
    df['adj_close'] = df['c']
    return df[['adj_close']]

# Function to fetch stock data from Yahoo Finance
def get_yfinance_data(symbol):
    df = yf.download(symbol, start="2004-01-01")
    return df[['Close']]

# Streamlit UI
st.title("Real-Time Stock Price Prediction")

stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, BTC-USD, ETH-USD):", "AAPL").upper()

# List of popular stocks for easy access
st.sidebar.header("Popular Stocks")
popular_stocks = {
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'Microsoft': 'MSFT',
    'Netflix': 'NFLX',
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD'
}
for name, symbol in popular_stocks.items():
    st.sidebar.write(f"{name}: {symbol}")

try:
    # Fetch data from Polygon.io
    stock_data_polygon = get_polygon_data(stock)
    st.success(f"Successfully fetched data from Polygon.io for {stock}")
    st.write(stock_data_polygon.tail())
except Exception as e:
    st.error(f"Error fetching data from Polygon.io: {e}")
    stock_data_polygon = None

try:
    # Fetch data from Yahoo Finance
    yfinance_data = get_yfinance_data(stock)
    st.success(f"Successfully fetched data from Yahoo Finance for {stock}")
    st.write(yfinance_data.tail())
except Exception as e:
    st.error(f"Error fetching data from Yahoo Finance: {e}")
    yfinance_data = None

if stock_data_polygon is not None and yfinance_data is not None:
    # Align the data to the same date range
    start_date = stock_data_polygon.index.min()
    end_date = stock_data_polygon.index.max()
    yfinance_data_filtered = yfinance_data.loc[start_date:end_date]

    # Merge the data on date
    merged_data = pd.merge(stock_data_polygon, yfinance_data_filtered, left_index=True, right_index=True, how='outer')
    merged_data.ffill(inplace=True)

    # Plot Adjusted Close Price
    st.subheader('Adjusted Close Price History')
    st.line_chart(merged_data['adj_close'])

    # Preprocessing for prediction
    merged_data = merged_data.dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(merged_data['adj_close'].values.reshape(-1, 1))

    # Train/Test split
    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - 60:]

    # Create training data
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # Create testing data
    x_test, y_test = [], []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        y_test.append(test_data[i, 0])

    # Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Reshape the data for the LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Check for pre-trained model
model_filename = f"{stock}_lstm_model.h5"
model = None  # Initialize model as None

if os.path.exists(model_filename):
    model = load_model(model_filename)
else:
    if st.checkbox("No pre-trained model found. Train a model?"):
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Training the model
        with st.spinner('Training the model...'):
            history = model.fit(x_train, y_train, batch_size=2, epochs=100)
            st.success("Model training completed.")
        
        # Save the model
        model.save(model_filename)
        st.success(f"Model saved as {model_filename}")

# Ensure the model is defined before making predictions
if model is not None:
    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot the prediction results
    st.subheader('Predicted vs Actual Stock Price')
    plt.figure(figsize=(14, 6))
    plt.plot(merged_data.index[train_len:], merged_data['adj_close'][train_len:], label='Actual Price')
    plt.plot(merged_data.index[train_len:], predictions, label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Show predicted stock price for next day
    st.subheader(f"Predicted Stock Price for {stock} (next day):")
    st.write(f"${predictions[-1][0]:.2f}")
else:
    st.error("The model could not be created or loaded.")