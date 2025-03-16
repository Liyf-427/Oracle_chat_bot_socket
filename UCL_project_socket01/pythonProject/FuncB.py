import pandas as pd
import numpy as np
import keras
from keras import layers
import joblib
from keras.utils.generic_utils import get_custom_objects

# TransformerEncoder
class TransformerEncoder(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False):
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

get_custom_objects().update({"TransformerEncoder": TransformerEncoder})

# prediction function
def pred_b(future_date):
    """ estimation of AQI value """

    # read data
    df = pd.read_csv('data/California_Air_Quality_Data.csv')
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df = df[['Date', 'PM2.5']].dropna().sort_values(by='Date')

    # read Scaler and model
    scaler = joblib.load('models/scaler.pkl')
    model = keras.models.load_model('models/transformer_aqi_model.h5', custom_objects={'TransformerEncoder': TransformerEncoder})

    # ensure date formating
    future_date = pd.to_datetime(future_date).normalize()
    last_date = df['Date'].max().normalize()

    if future_date <= last_date:
        raise ValueError(f"❌ Error: The date {future_date} must be after {last_date}.")

    # data processor
    window_size = 7
    all_aqi = df['PM2.5'].values.reshape(-1, 1)
    scaled_all = scaler.transform(all_aqi)

    # initialize window
    current_window = scaled_all[-window_size:].copy()
    current_date = last_date

    while current_date < future_date:
        input_batch = np.expand_dims(current_window, axis=0)  # ensure match size
        pred_scaled = model.predict(input_batch, verbose=0)[0][0]
        pred_original = scaler.inverse_transform([[pred_scaled]])[0][0]

        pred_scaled = np.clip(pred_scaled + np.random.normal(0, 0.01), 0, 1)
        current_window = np.append(current_window[1:], [[pred_scaled]], axis=0)
        current_date += pd.Timedelta(days=1)

    return pred_original
