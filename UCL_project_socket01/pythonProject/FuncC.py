import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
import random

# ** S&P 500 predictor**
def pred_c(query_date):
    # load Scaler and model
    model = load_model('models/Oracle_model_S&P500_prediction.h5', custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load('models/S&P500_scaler.joblib')

    # read data
    path = 'data/S&P 500 Historical Data.csv'
    df = pd.read_csv(path, parse_dates=['Date'], dayfirst=True)  # ensure formatting
    df = df.sort_values(by='Date')
    df['Price'] = df['Price'].str.replace(',', '').astype(float)  # process data

    # calculate date to process
    query_date = pd.to_datetime(query_date).normalize()
    last_date = df['Date'].max()  # extract newest date
    days_to_predict = (query_date - last_date).days  # calculate estimation length
    if days_to_predict <= 0:
        print("Error: The date must be in the future!")
        return

    print(f"Predicting for {days_to_predict} days from {last_date.date()} to {query_date.date()}.")

    # get data
    seq_length = 10
    last_10_days = df['Price'].values[-seq_length:]  # get new data
    last_10_days_scaled = scaler.fit_transform(last_10_days.reshape(-1, 1))  # normalize
    last_10_days_scaled = last_10_days_scaled.reshape(1, seq_length, 1).astype('float32')

    # prediction iteration
    predicted_prices = []
    for _ in range(days_to_predict):
        predicted_price = model.predict(last_10_days_scaled, verbose=0)  # prediction
        real_price = scaler.inverse_transform(predicted_price)[0][0]  # inverse normalization
        predicted_prices.append(real_price)

        # renew data

        last_10_days_scaled = np.append(last_10_days_scaled[:, 1:, :], predicted_price.reshape(1, 1, 1)/seq_length, axis=1)

    # print result
    #for i, price in enumerate(predicted_prices, 1):
        #print(f"Predicted S&P 500 price for {last_date.date() + timedelta(days=i)}: {price:.2f}")
    #get last day estimation
    final_price = predicted_prices[-1]
    prev_day_price = predicted_prices[-2] if len(predicted_prices) > 1 else df['Price'].values[-1]
    change_percentage = ((final_price - prev_day_price) / prev_day_price) * 100

    return final_price, change_percentage

