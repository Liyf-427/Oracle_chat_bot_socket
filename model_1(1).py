import pandas as pd
import numpy as np
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib

path = 'data/S&P 500 Historical Data.csv'

try:
    loaded_data = pd.read_csv(path)
    prices = loaded_data['Price'].str.replace(',', '').values
    print("Shape of prices: ", prices.shape)

    # Define the input value and target value as x&y
    def create_arr(data, seq_len):
        x, y = [], []
        for i in range(len(data) - seq_len):
            x.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(x), np.array(y)

    seq_len = 10
    train_size = int(len(prices) * 0.9)
    train_data = prices[:train_size]
    test_data = prices[train_size:]

    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data.reshape((-1, 1)))
    scaled_test_data = scaler.fit_transform(test_data.reshape((-1, 1)))

    joblib.dump(scaler, 'models/S&P500_scaler.joblib')

    x_train, y_train = create_arr(scaled_train_data, seq_len)
    x_test, y_test = create_arr(scaled_test_data, seq_len)
    # 确保形状正确
    print(f"x_train shape after create_arr: {x_train.shape}")  # 应该是 (样本数, 10, 1)
    print(f"x_test shape after create_arr: {x_test.shape}")  # 应该是 (样本数, 10, 1)

    print(type(x_train))
    x_train = y_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    print(type(y_train))


    #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    # 确保形状正确
    print(f"x_train shape after create_arr: {x_train.shape}")  # 应该是 (样本数, 10, 1)
    print(f"x_test shape after create_arr: {x_test.shape}")  # 应该是 (样本数, 10, 1)

    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_len, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    model.fit(x_train, y_train, epochs=30,
                batch_size=32, validation_split=0.1, verbose=1)

    model.save('models/Oracle_model_S&P500_prediction.h5')

except FileNotFoundError:
    print(f"Error! \"{path}\" cannot be found.")



