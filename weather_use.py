import tensorflow as tf
import pandas as pd
import os
import numpy as np
from weather_train import prepare_lstm_data
from sklearn.preprocessing import MinMaxScaler
#在使用了 森林/XGBoost/还有未归一化LSTM model训练效果不理想
def load_model(name):
    f = fr"models\{name}_lstm_model.h5"
    return tf.keras.models.load_model(f)


def predict_weather(model, df, date, time_steps=20):
    """
    预测指定日期的天气（温度 & 湿度）。

    参数：
    - model: 训练好的 LSTM 模型
    - df: 预处理过的天气数据 DataFrame
    - date: 需要预测的日期 (字符串格式: "yyyy/mm/dd")
    - time_steps: 过去多少天的数据作为输入（默认 10 天）

    返回：
    - 预测的 (Temperature, Humidity)
    """

    # 确保 "Date" 列是字符串格式
    df["Date"] = df["Date"].astype(str)

    # ✅ 如果目标日期已经在数据集中，直接返回真实天气数据
    if date in df["Date"].values:
        real_temp = df.loc[df["Date"] == date, "Temperature"].values[0]
        real_humidity = df.loc[df["Date"] == date, "Humidity"].values[0]
        print(f"✅ {date} 在数据集中，直接使用真实数据: 🌡 {real_temp:.2f}°C, 💧 {real_humidity:.2f}%")
        return real_temp, real_humidity

    # 未来预测逻辑（如果日期不在数据集中）
    last_known_data = df.iloc[-time_steps:, 1:].values  # 取最近 time_steps 天的 (温度, 湿度)

    if date not in df["Date"].values:
        print(f"📅 {date} 不在历史数据集中，使用最近 {time_steps} 天的数据预测未来天气...")

        # 预测未来天气
        input_data = last_known_data.reshape(1, time_steps, -1)
        prediction = model.predict(input_data)[0]

        return prediction[0], prediction[1]  # (预测温度, 预测湿度)

    # 历史评估逻辑（如果日期在数据集中）
    date_index = df.index[df["Date"] == date].tolist()[0]

    # 确保有足够的历史数据
    if date_index < time_steps:
        raise ValueError(f"❌ 无法预测 {date}，因为前面没有足够的 {time_steps} 天数据！")

    # 选取 `time_steps` 天数据作为输入
    input_data = df.iloc[date_index - time_steps:date_index, 1:].values.reshape(1, time_steps, -1)
    prediction = model.predict(input_data)[0]

    return prediction[0], prediction[1]  # (预测温度, 预测湿度)

def main(name, target_date=None):
    output_dir = r"data"
    output_file = os.path.join(output_dir, name + ".csv")

    # 读取数据
    df = pd.read_csv(output_file)

    # 选择需要归一化的列（Temperature 和 Humidity）
    scaler = MinMaxScaler()

    # 只对温度和湿度进行归一化（不修改 Date）
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

    # 加载模型
    model = load_model(name)

    # 预测指定日期
    temperature, humidity = predict_weather(model, df, target_date)

    # ✅ 逆归一化
    temp_pred, humidity_pred = scaler.inverse_transform(np.array([[temperature, humidity]]))[0]

    print(f"📅 预测日期: {target_date}")
    print(f"🌡  预测温度: {temp_pred:.2f}°C")
    print(f"💧  预测湿度: {humidity_pred:.2f}%")

if __name__ == "__main__":
  target_date = input("请输入要预测的日期 (格式: yyyy/mm/dd): ").strip()
  main("London", target_date)


