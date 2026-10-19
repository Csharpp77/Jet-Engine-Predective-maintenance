from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from lightgbm import Booster, LGBMRegressor
except ImportError as error:
    raise SystemExit(
        "LightGBM is not installed. Install it with `python -m pip install lightgbm`."
    ) from error


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
outputs_dir = project_dir / "outputs"
current_model_path = outputs_dir / "lightgbm_model.txt"
experimental_model_path = outputs_dir / "lightgbm_uncapped_model_v2.txt"

if experimental_model_path.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing experimental model: {experimental_model_path}"
    )

data = pd.read_csv(data_dir / "processed_train.csv")
engine_ids = data["engine_id"].drop_duplicates()
if len(engine_ids) != 100:
    raise ValueError(f"Expected 100 unique engines, found {len(engine_ids)}.")

# Match the existing engine-level 80/20 split, including its seed and ID ordering.
training_ids = set(engine_ids.sample(frac=0.8, random_state=42))
validation_ids = set(engine_ids) - training_ids
training_data = data.loc[data["engine_id"].isin(training_ids)].copy()
validation_data = data.loc[data["engine_id"].isin(validation_ids)].copy()

# The existing model's feature list preserves the exact training feature order.
capped_model = Booster(model_file=str(current_model_path))
feature_columns = capped_model.feature_name()
missing_features = [column for column in feature_columns if column not in data.columns]
if missing_features:
    raise ValueError(f"Processed training data is missing model features: {missing_features}")
if {"engine_id", "RUL"}.intersection(feature_columns):
    raise ValueError("Model features must exclude engine_id and target columns.")

training_features = training_data.loc[:, feature_columns]
validation_features = validation_data.loc[:, feature_columns]
uncapped_target = (
    data.groupby("engine_id")["cycle"].transform("max") - data["cycle"]
)
training_uncapped_target = uncapped_target.loc[training_data.index]
validation_uncapped_target = uncapped_target.loc[validation_data.index]

# Evaluate the existing capped-target model against uncapped RUL so both models
# are compared on the same validation rows and the same target definition.
capped_predictions = capped_model.predict(validation_features)
capped_rmse = mean_squared_error(
    validation_uncapped_target, capped_predictions
) ** 0.5
capped_mae = mean_absolute_error(validation_uncapped_target, capped_predictions)

uncapped_model = LGBMRegressor(random_state=42)
uncapped_model.fit(training_features, training_uncapped_target)
uncapped_predictions = uncapped_model.predict(validation_features)
uncapped_rmse = mean_squared_error(
    validation_uncapped_target, uncapped_predictions
) ** 0.5
uncapped_mae = mean_absolute_error(validation_uncapped_target, uncapped_predictions)

print(f"Training engines: {len(training_ids)}")
print(f"Validation engines: {len(validation_ids)}")
print("Validation metrics (actual target: uncapped RUL):")
print(f"Current capped-target model - RMSE: {capped_rmse:.4f}, MAE: {capped_mae:.4f}")
print(f"Uncapped-target model       - RMSE: {uncapped_rmse:.4f}, MAE: {uncapped_mae:.4f}")

outputs_dir.mkdir(parents=True, exist_ok=True)
uncapped_model.booster_.save_model(str(experimental_model_path))
print(f"Saved uncapped-target model to: {experimental_model_path}")
