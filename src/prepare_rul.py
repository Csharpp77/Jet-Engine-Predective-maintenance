from pathlib import Path

import pandas as pd


column_names = (
    ["engine_id", "cycle"]
    + [f"setting{i}" for i in range(1, 4)]
    + [f"sensor{i}" for i in range(1, 22)]
)
dataset_path = Path(__file__).resolve().parent.parent / "data" / "train_FD001.txt"
data = pd.read_csv(dataset_path, sep=r"\s+", header=None, names=column_names)
data = data.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

max_cycle_per_engine = data.groupby("engine_id")["cycle"].transform("max")
data["RUL"] = max_cycle_per_engine - data["cycle"]

final_cycle_rows = data["cycle"].eq(max_cycle_per_engine)
final_cycle_rul = data.loc[final_cycle_rows, ["engine_id", "RUL"]]
if (
    final_cycle_rul["engine_id"].nunique() != 100
    or not final_cycle_rul["RUL"].eq(0).all()
):
    raise AssertionError("All 100 engines must end with uncapped RUL 0.")
print("Verified: all 100 engines end with uncapped RUL 0.")

data["RUL"] = data["RUL"].clip(upper=125)

print("\nFirst 10 rows with engine_id, cycle, and capped RUL:")
print(data[["engine_id", "cycle", "RUL"]].head(10))

print("\nMinimum capped RUL:")
print(data["RUL"].min())

print("\nMaximum capped RUL:")
print(data["RUL"].max())

print("\nNumber of rows with capped RUL = 125:")
print(data["RUL"].eq(125).sum())

engine_1 = data.loc[data["engine_id"].eq(1)].sort_values("cycle")
print("\nFirst 5 rows of Engine 1 (cycle and capped RUL):")
print(engine_1[["cycle", "RUL"]].head(5))

print("\nLast 5 rows of Engine 1 (cycle and capped RUL):")
print(engine_1[["cycle", "RUL"]].tail(5))

sensor_columns = [f"sensor{i}" for i in range(1, 22)]
for sensor in sensor_columns:
    grouped_sensor = data.groupby("engine_id")[sensor]
    data[f"{sensor}_rolling_mean_15"] = grouped_sensor.transform(
        lambda values: values.rolling(window=15, min_periods=1).mean()
    )
    data[f"{sensor}_rolling_std_15"] = grouped_sensor.transform(
        lambda values: values.rolling(window=15, min_periods=1).std()
    ).fillna(0)

output_path = dataset_path.parent / "processed_train.csv"
data.to_csv(output_path, index=False)
print(f"\nSaved prepared training dataset to: {output_path}")
