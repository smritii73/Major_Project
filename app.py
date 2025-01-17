import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
plt.style.use("fivethirtyeight")

app = Flask(__name__)
model = load_model('stock_dl_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'HDFCBANK.NS'
        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2024, 12, 30)
        # Download stock data
        df = yf.download(stock, start=start, end=end)
        
        # Descriptive Data
        data_desc = df.describe()
        
        # Moving Averages (Using rolling for MA)
        ma100 = df['Close'].rolling(window=100).mean()
        ma200 = df['Close'].rolling(window=200).mean()

        # Exponential Moving Averages (Using ewm for EMA)
        ema100 = df['Close'].ewm(span=100, adjust=False).mean()
        ema200 = df['Close'].ewm(span=200, adjust=False).mean()
        
        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
        
        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        
        # Prepare data for prediction
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)
        
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)
        
        # Inverse scaling for predictions
        scaler = scaler.scale_
        scale_factor = 1 / scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor
        
        # Plot 1: Closing Price vs Time Chart with 100 & 200 Days MA
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df['Close'], 'y', label='Closing Price')
        ax1.plot(ma100, 'r', label='MA 100', linewidth=1)
        ax1.plot(ma200, 'b', label='MA 200', linewidth=1)
        ax1.set_title("Closing Price vs Row Indices (100 & 200 Days MA)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ma_chart_path = "static/ma.png"
        fig1.savefig(ma_chart_path)
        plt.close(fig1)
        
        # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df['Close'], 'y', label='Closing Price')
        ax2.plot(ema100, 'r', label='EMA 100', linewidth=1)
        ax2.plot(ema200, 'b', label='EMA 200', linewidth=1)
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)
        
        # Plot 3: Prediction vs Original Trend
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'b', label="Original Price", linewidth=1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)
        # Save dataset as CSV
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        # Return the rendered template with charts and dataset
        return render_template('index.html', 
                               plot_path_ma_100_200=ma_chart_path, 
                               plot_path_ema_100_200=ema_chart_path_100_200, 
                               plot_path_prediction=prediction_chart_path, 
                               data_desc=data_desc.to_html(classes='table table-bordered'),
                               dataset_link=csv_file_path)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
