Stock Price Prediction Using LSTM

This project involves building a Stock Price Prediction model using Long Short-Term Memory (LSTM) networks. The model predicts stock prices based on historical data and displays trends, moving averages, and predictions on an interactive web application powered by Flask. The project specifically focuses on HDFC Bank (HDFCBANK.NS) stock data but supports other stock tickers as well.

Key Files:
Stock_Price_Prediction.ipynb: Contains the Jupyter Notebook used for data preprocessing, visualization, and model building.
app.py: Flask application for deploying the web interface. It allows users to select stock tickers and visualize data trends and predictions.
templates/index.html:Frontend interface for user interaction.
static/:Contains generated plots and datasets for easy access and download.
stock_dl_model.h5: Pretrained LSTM model for stock price prediction.
requirements.txt: Lists the Python dependencies required for the project.

Setup Instructions
Prerequisites
Python 3.8+
Virtual Environment (recommended)
pip (Python package installer)
Steps:
1. Clone the Repository:
   git clone <repository-link>
   cd Major_Project
2. Set up a Virtual Environment: Create a virtual environment to isolate dependencies.
   python -m venv .venv
   .venv\Scripts\activate
3. Install Dependencies: Use pip to install the required packages.
   pip install -r requirements.txt
4. Run the Flask Application: Start the web application locally.
   python app.py

Features
1. Historical Data Visualization:
Displays trends in opening, closing, high, and low prices.
Candlestick chart for better pattern recognition.

2. Moving Averages:
Simple Moving Average (SMA) for 100 and 200 days.
Exponential Moving Average (EMA) for 100 and 200 days.

3. LSTM Model for Prediction:
Trained on historical stock data.
Predicts stock prices and compares with original trends.

4. Interactive Web Interface:
User-friendly input for stock tickers.
Real-time visualization of data trends and predictions.

5. Downloadable Datasets:
Provides CSV download links for analyzed datasets.
