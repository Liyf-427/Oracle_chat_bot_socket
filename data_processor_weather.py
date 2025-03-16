import pandas as pd
import re
import os
# 读取原始CSV文件（跳过前8行的说明）
name = "Berlin"
file_path = fr"raw data\{name}.csv"
df = pd.read_csv(file_path, skiprows=8, delimiter=";", header=0, encoding="utf-8", on_bad_lines='skip')
print(df.head())

# 提取所需列（日期、气温 T、湿度 U）
df_filtered = df.iloc[:, [0, 1, 4]].copy()  # 确保是副本
print(df_filtered.head())
# 重命名列
df_filtered.columns = ["DateTime", "Temperature", "Humidity"]

# 解析日期并转换格式
def convert_date(date_str):
    match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", date_str)
    if match:
        return f"{match.group(3)}/{match.group(2)}/{match.group(1)}"  # 转换为 yyyy/mm/dd
    return date_str  # 如果匹配失败，保持原格式

df_filtered.loc[:, "Date"] = df_filtered["DateTime"].apply(convert_date)

# 选择最终列（去掉原始时间）
df_final = df_filtered[["Date", "Temperature", "Humidity"]]
# 按日期分组并计算均值
df_daily = df_final.groupby("Date", as_index=False).mean()

output_dir = r"data"
output_file = os.path.join(output_dir, f"{name}.csv")

# 如果目录不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 保存 CSV
df_daily.to_csv(output_file, index=False, float_format="%.3f", encoding="utf-8")
print(f"✅ 文件已成功保存至: {output_file}")
