from pathlib import Path

import pandas as pd

try:
    from lightgbm import Booster
except ImportError as error:
    raise SystemExit(
        "LightGBM is required to read the trained model. "
        "Install it with `python -m pip install lightgbm`."
    ) from error


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
model_path = project_dir / "outputs" / "lightgbm_model.txt"

column_names = (
    ["engine_id", "cycle"]
    + [f"setting{i}" for i in range(1, 4)]
    + [f"sensor{i}" for i in range(1, 22)]
)
test_data = pd.read_csv(
    data_dir / "test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=column_names,
).sort_values(["engine_id", "cycle"])

sensor_columns = [f"sensor{i}" for i in range(1, 22)]
for sensor in sensor_columns:
    grouped_sensor = test_data.groupby("engine_id")[sensor]
    test_data[f"{sensor}_rolling_mean_15"] = grouped_sensor.transform(
        lambda values: values.rolling(window=15, min_periods=1).mean()
    )
    test_data[f"{sensor}_rolling_std_15"] = grouped_sensor.transform(
        lambda values: values.rolling(window=15, min_periods=1).std()
    ).fillna(0)

model = Booster(model_file=str(model_path))
feature_columns = model.feature_name()
missing_features = [column for column in feature_columns if column not in test_data]
if missing_features:
    raise ValueError(f"Test data is missing model features: {missing_features}")

# Retain engine_id for identification, while keeping model inputs in trained order.
final_rows = test_data.groupby("engine_id", sort=False).tail(1)
processed_test = final_rows[["engine_id", *feature_columns]].reset_index(drop=True)

if processed_test["engine_id"].nunique() != test_data["engine_id"].nunique():
    raise AssertionError("Expected exactly one final row for each test engine.")

output_path = data_dir / "processed_test.csv"
processed_test.to_csv(output_path, index=False)

print(f"Processed test shape: {processed_test.shape}")
print(
    "Confirmed: "
    f"{processed_test['engine_id'].nunique()} rows, one final recorded row "
    "per test engine."
)
