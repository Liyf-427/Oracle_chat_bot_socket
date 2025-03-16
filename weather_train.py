import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import os
from sklearn.preprocessing import MinMaxScaler
import joblib


def prepare_lstm_data(df,city, time_steps=20):

    # 归一化数据（Temperature 和 Humidity）
    scaler = MinMaxScaler()
    print("原始温度范围:", df["Temperature"].min(), "-", df["Temperature"].max())
    print("原始湿度范围:", df["Humidity"].min(), "-", df["Humidity"].max())

    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])  # 只归一化数值列
    print("Scaler min values:", scaler.data_min_)
    print("Scaler scale values:", scaler.scale_)
    print("归一化后温度范围:", df["Temperature"].min(), "-", df["Temperature"].max())
    print("归一化后湿度范围:", df["Humidity"].min(), "-", df["Humidity"].max())
    joblib.dump(scaler, f"models/{city}_scaler.pkl")  # 保存归一化器

    X, y = [], []
    for i in range(time_steps, len(df)):
        X.append(df.iloc[i-time_steps:i, 1:].values)  # 使用 'Temperature' 和 'Humidity' 作为特征
        y.append(df.iloc[i, 1:].values)  # 输出当天的 (Temperature, Humidity)
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    return X, y

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),  # 增加神经元数，提高学习能力
        Dropout(0.2),

        LSTM(64, return_sequences=True),  # 额外增加 LSTM 层
        Dropout(0.1),

        LSTM(64, return_sequences=False),
        Dropout(0.1),

        Dense(2)  # 预测温度和湿度
    ])
    model.compile(optimizer= Adam(learning_rate=0.001), loss='mean_squared_error')

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    #  修正：转换 y 为 float32 避免类型问题
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))
    return model


def save_model(model, filename=r"models\lstm_model.h5"):
    # 确保路径格式正确
    filename = os.path.abspath(filename)

    # 确保目录存在
    model_dir = os.path.dirname(filename
                                )
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # 保存模型为 HDF5 格式
    model.save(filename, save_format="h5")
    print(f"✅ Model successfully saved to: {filename}")

def main(name):

    df = pd.read_csv(fr"data\{name}.csv")

    X, y = prepare_lstm_data(df,name)
    model = train_model(X, y)
    loc = fr"models\{name}_lstm_model.h5"
    save_model(model,loc)

if __name__ == "__main__":
    name = "Berlin"#change different city by name
    main(name)
