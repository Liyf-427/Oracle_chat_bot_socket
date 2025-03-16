import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import os
import joblib  # 用於儲存 Scaler

# 1. 讀取並預處理數據
df = pd.read_csv('data/California_Air_Quality_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', inplace=True)
df = df[['Date', 'PM2.5']].dropna()

df['Date'] = df['Date'].dt.normalize()

# 2. 創建並儲存 MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
aqi_values = df['PM2.5'].values.reshape(-1, 1)
scaled_values = scaler.fit_transform(aqi_values)

scaler_path = 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler has been saved to {scaler_path}")

# 3. 創建數據集
window_size = 7

def create_dataset(series, window_size=7):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

train_size = len(scaled_values) - 30
train_data = scaled_values[:train_size]
test_data = scaled_values[train_size:]

X_train, y_train = create_dataset(train_data, window_size)
X_test, y_test = create_dataset(test_data, window_size)

# 4. 定義 Transformer Encoder Block
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": 64, "num_heads": 4, "dff": 128, "rate": 0.1})
        return config

# 5. 訓練模型並保存
model_path = 'models/transformer_aqi_model.h5'

if not os.path.exists(model_path):
    inputs = tf.keras.layers.Input(shape=(None, 1))  # None 允许可变时间步长

    #inputs = tf.keras.layers.Input()#shape=(window_size, 1)
    embed = tf.keras.layers.Dense(64)(inputs)
    transformer_block = TransformerEncoder(64, 4, 128, 0.1)
    x = transformer_block(embed, training=True)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("No trained model found. Training a new model...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

    model.save(model_path)
    print("Model has been trained and saved.")
else:
    print("A trained model already exists. No training required.")
