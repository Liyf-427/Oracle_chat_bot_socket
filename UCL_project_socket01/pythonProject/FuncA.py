import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load the model
def load_model(name):
    model_path = f"models/{name}_lstm_model.h5"
    return tf.keras.models.load_model(model_path)

# Predict weather for multiple days iteratively
def predict_weather_iterative(model, df, start_date, time_steps=20):
    """
    Predict the weather iteratively for multiple days using LSTM model.

    Parameters:
    - model: trained LSTM model
    - df: preprocessed weather data
    - start_date: the target date (first day of prediction)
    - time_steps: number of days used for prediction (default 20)
    - days_to_predict: how many days in the future to predict

    Returns:
    - predicted_temperature, predicted_humidity
    """

    # Ensure "Date" is in string format for comparison
    df["Date"] = df["Date"].astype(str)
    # Normalize the temperature and humidity data (use the same scaler as before)
    scaler = MinMaxScaler()
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

    # Check if the start date is in the dataset
    if start_date in df["Date"].values:
        real_temp = df.loc[df["Date"] == start_date, "Temperature"].values[0]
        real_humidity = df.loc[df["Date"] == start_date, "Humidity"].values[0]
        return real_temp, real_humidity

    # Calculate days difference from the last date in the dataset
    last_date = pd.to_datetime(df["Date"].max())
    target_date = pd.to_datetime(start_date)
    days_diff = (target_date - last_date).days

    # Get the most recent data (the last `time_steps` days)
    last_known_data = df.iloc[-time_steps:, 1:].values  # (temperature, humidity)

    # Initialize future data and iteratively predict
    future_data = last_known_data.flatten().reshape(1, time_steps, -1)

    predictions = []
    for day in range(days_diff):
        # Predict temperature and humidity for the next day
        temp_pred = model.predict(future_data)[0][0]
        humidity_pred = model.predict(future_data)[0][1]

        # Update future_data with the new predictions for the next iteration + np.random.normal(0, 0.01)

        future_data = np.hstack([future_data[:, 1:, :], np.array([[temp_pred, humidity_pred]]).reshape(1, 1, -1)])

        # Inverse transform the predictions (temperature and humidity)
        temp_pred, humidity_pred = scaler.inverse_transform([[temp_pred, humidity_pred]])[0]

        # Store predictions for this day
        predictions.append((temp_pred, humidity_pred))

    last_prediction = predictions[-1]  # get prediction
    temp, humidity = last_prediction  # unpack
    # trans into str
    temp = f"{temp:.2f}"
    humidity = f" {humidity:.2f}"
    print(temp, humidity)

    return temp, humidity  # Returns a list of (temperature, humidity) tuples

# Main function for input and predictions
def pred_a(name, target_date):
    output_file = f"data/{name}.csv"

    # Read the dataset
    df = pd.read_csv(output_file)

    # Load the trained model
    model = load_model(name)

    # Predict the weather for the given date
    t,h = predict_weather_iterative(model, df, target_date, time_steps=20)
    return t,h