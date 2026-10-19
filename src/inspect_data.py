from pathlib import Path

import pandas as pd


column_names = (
    ["engine_id", "cycle"]
    + [f"setting{i}" for i in range(1, 4)]
    + [f"sensor{i}" for i in range(1, 22)]
)
dataset_path = Path(__file__).resolve().parent.parent / "data" / "train_FD001.txt"
data = pd.read_csv(dataset_path, sep=r"\s+", header=None, names=column_names)

print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nMissing-value counts:")
print(data.isna().sum())

cycles_per_engine = data.groupby("engine_id")["cycle"].count()

print("\nNumber of unique engines:")
print(data["engine_id"].nunique())

print("\nMinimum cycles per engine:")
print(cycles_per_engine.min())

print("\nMaximum cycles per engine:")
print(cycles_per_engine.max())

print("\n5 engines with the most operating cycles:")
print(cycles_per_engine.sort_values(ascending=False).head(5))

sensor_columns = [f"sensor{i}" for i in range(1, 22)]
print("\nDescriptive statistics for sensors:")
print(data[sensor_columns].describe())
