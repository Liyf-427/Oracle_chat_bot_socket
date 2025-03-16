import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib


def predict_sp500_price():
    # Load the model and scaler
    model = load_model('models/Oracle_model_S&P500_prediction.h5', custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load('models/S&P500_scaler.joblib')

    # process the newest data for last 10 days to enter to the model
    seq_length = 10
    new_data = [5983.25, 6013.13, 6117.52, 6144.15, 6129.58, 6114.63, 6115.07, 6051.97, 6068.50, 6066.44]
    new_data = np.array(new_data)
    new_array = new_data.reshape(-1, 1)
    scaled_array = scaler.fit_transform(new_array)
    reshaped_array = scaled_array.reshape(1, seq_length, 1).astype('float32')

    # Call the model
    max_attempt = 2

    for i in range(max_attempt):
        query = input("Enter your inquiry: ")

        if query == 'Can you predict the S&P 500 price for tomorrow?':
            predicted_price = model.predict(reshaped_array, verbose=0)
            prediction = scaler.inverse_transform(predicted_price)

            print(f"Predicted S&P 500 price for tomorrow: {prediction[0][0]:.2f}")
            return prediction[0][0]

        else:
            print("Please ask: 'Can you predict the S&P 500 price for tomorrow?' ")


if __name__ == "__main__":
    predicted_price_tomorrow = predict_sp500_price()
