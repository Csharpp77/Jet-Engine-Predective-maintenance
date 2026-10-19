from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from lightgbm import Booster
except ImportError as error:
    raise SystemExit(
        "LightGBM is required to load the trained model. "
        "Install it with `python -m pip install lightgbm`."
    ) from error


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
outputs_dir = project_dir / "outputs"
model_path = outputs_dir / "lightgbm_uncapped_model_v2.txt"
prediction_path = outputs_dir / "test_predictions_uncapped_v2.csv"

if prediction_path.exists():
    raise FileExistsError(f"Refusing to overwrite existing predictions: {prediction_path}")

model = Booster(model_file=str(model_path))
test_data = pd.read_csv(data_dir / "processed_test.csv")
labels = pd.read_csv(
    data_dir / "RUL_FD001.txt",
    sep=r"\s+",
    header=None,
    names=["actual_RUL"],
)

engine_ids = sorted(test_data["engine_id"].unique())
if len(engine_ids) != 100:
    raise ValueError(f"Expected 100 test engines, found {len(engine_ids)}.")
if len(labels) != len(engine_ids):
    raise ValueError(
        f"Found {len(engine_ids)} test engines but {len(labels)} official RUL labels."
    )
if test_data["engine_id"].duplicated().any():
    raise ValueError("Processed test data must contain one row per engine.")

feature_columns = model.feature_name()
if {"engine_id", "RUL", "actual_RUL", "predicted_RUL"}.intersection(feature_columns):
    raise ValueError("Model feature names must exclude engine IDs and target columns.")
missing_features = [column for column in feature_columns if column not in test_data]
if missing_features:
    raise ValueError(f"Processed test data is missing model features: {missing_features}")

# Keep input columns in the exact order stored in the trained LightGBM model.
test_features = test_data.set_index("engine_id").loc[engine_ids, feature_columns]
predictions = model.predict(test_features)
results = pd.DataFrame(
    {
        "engine_id": engine_ids,
        "actual_RUL": labels["actual_RUL"].to_numpy(),
        "predicted_RUL": predictions,
    }
)

rmse = mean_squared_error(results["actual_RUL"], results["predicted_RUL"]) ** 0.5
mae = mean_absolute_error(results["actual_RUL"], results["predicted_RUL"])
print(f"Official test RMSE (uncapped-target model): {rmse:.4f}")
print(f"Official test MAE (uncapped-target model): {mae:.4f}")

outputs_dir.mkdir(parents=True, exist_ok=True)
# Exclusive creation ensures a file created after the initial check is not replaced.
results.to_csv(prediction_path, index=False, mode="x")
print(f"Saved predictions to: {prediction_path}")
