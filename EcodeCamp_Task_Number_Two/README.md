# Stock Price Prediction System Report

## Table of Contents
1. Introduction
2. Libraries Used
3. Project Structure
4. Data Collection
5. Data Preprocessing
6. Model Building
7. Real-Time Prediction
8. Flask Web Application
9. HTML Interface
10. Command-Line Deployment
11. Conclusion
12. References

## 1. Introduction
The objective of this project is to develop a real-time stock price prediction system using historical stock data. This system utilizes Long Short-Term Memory (LSTM) networks to forecast future stock prices based on past trends. The application is built using Python, Flask, and HTML, providing a user-friendly interface for real-time predictions.

## 2. Libraries Used
The project employs several key libraries:

- **yfinance**: This library allows for the easy download of historical market data from Yahoo Finance, enabling the retrieval of stock prices.
- **numpy**: A fundamental package for numerical computations in Python, especially useful for handling arrays and mathematical operations.
- **pandas**: A powerful data manipulation library that simplifies data analysis and manipulation through data frames.
- **scikit-learn**: A widely-used machine learning library that provides tools for data preprocessing, including scaling techniques.
- **keras**: A high-level neural networks API that facilitates the building and training of LSTM models.
- **Flask**: A lightweight web framework for Python that is used to create web applications.

## 3. Project Structure
The project is organized into a straightforward directory structure, with the main application file (app.py) responsible for the core functionality, a requirements file listing the necessary libraries, and a templates folder containing the HTML interface.

## 4. Data Collection
Historical stock data is collected using the `yfinance` library. A function is defined to download stock data for a specified ticker and date range. This step ensures that the application has access to the necessary data for analysis and prediction.

## 5. Data Preprocessing
Data preprocessing is a crucial step that prepares the raw stock data for modeling. This involves several tasks:

- **Scaling the Data**: The 'Close' prices of the stock are extracted and scaled to a range between 0 and 1 using the MinMaxScaler from scikit-learn. This scaling is essential for LSTM models, as they perform better with normalized input data.
  
- **Creating the Dataset**: The scaled data is structured into input-output pairs, where a sequence of past prices serves as input and the next price to predict acts as the output. This is done by defining a specific time step, which represents how many past observations to use for predicting the future price.

## 6. Model Building
An LSTM model is constructed using the Keras library. The architecture includes multiple layers, specifically two LSTM layers followed by dense layers. The model is compiled with the Adam optimizer and mean squared error as the loss function. This design allows the model to learn complex patterns in time-series data.

## 7. Real-Time Prediction
To facilitate real-time predictions, the application retrieves the most recent stock price and uses the trained LSTM model to predict the next price. Functions are defined to handle both the retrieval of real-time data and the prediction process. The model takes the latest price, scales it, and generates a prediction that is then transformed back to the original scale.

## 8. Flask Web Application
The web application is built using Flask, which provides a framework for handling HTTP requests and rendering HTML templates. The main route serves the homepage, while a dedicated route processes the stock ticker input from users and triggers the prediction workflow. The integration of model training and real-time prediction within the Flask application ensures that users can obtain immediate results.

## 9. HTML Interface
The user interface is created using HTML. It consists of a simple form that prompts users to enter a stock ticker symbol. Upon submission, the form data is sent to the Flask application for processing. If a prediction is available, it is displayed on the same page, providing users with immediate feedback on their input.

## 10. Command-Line Deployment
To deploy the Flask application, the command line interface (CLI) is utilized. Users navigate to the project directory using the command prompt and execute the command `python app.py`. This command runs the Flask server, allowing the application to be accessed via a web browser at the specified local address (usually http://127.0.0.1:5000). This straightforward deployment process enables users to launch the application quickly and efficiently.

## 11. Conclusion
This project successfully implements a real-time stock price prediction system utilizing LSTM networks. By integrating Flask for web development and HTML for the user interface, users can easily input stock tickers and receive predictions. Future enhancements could include the addition of alternative forecasting models, such as ARIMA, for comparative analysis, as well as the ability to visualize predictions over time.

## 12. References
- yfinance Documentation
- Keras Documentation
- Flask Documentation
- Scikit-learn Documentation

