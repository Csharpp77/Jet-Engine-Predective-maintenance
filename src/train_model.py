from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

try:
    from lightgbm import LGBMRegressor
except ImportError as error:
    raise SystemExit(
        "LightGBM is not installed. Install it with `python -m pip install lightgbm`."
    ) from error


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
training_data = pd.read_csv(data_dir / "train_split.csv")
validation_data = pd.read_csv(data_dir / "validation_split.csv")

# Exclude engine identity and the target from model inputs.
training_features = training_data.drop(columns=["engine_id", "RUL"])
validation_features = validation_data.drop(columns=["engine_id", "RUL"])
if not training_features.columns.equals(validation_features.columns):
    raise ValueError("Training and validation input features must match.")

training_target = training_data["RUL"]
validation_target = validation_data["RUL"]

model = LGBMRegressor(random_state=42)
model.fit(training_features, training_target)
validation_predictions = model.predict(validation_features)
lightgbm_rmse = mean_squared_error(validation_target, validation_predictions) ** 0.5

# Report the previously measured mean-RUL baseline alongside this model's score.
baseline_rmse = 41.5590
print(f"Validation RMSE (mean training RUL baseline): {baseline_rmse:.4f}")
print(f"Validation RMSE (LightGBM): {lightgbm_rmse:.4f}")

output_path = project_dir / "outputs" / "lightgbm_model.txt"
output_path.parent.mkdir(parents=True, exist_ok=True)
model.booster_.save_model(str(output_path))
print(f"Saved trained model to: {output_path}")
