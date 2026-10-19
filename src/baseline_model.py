from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error


data_dir = Path(__file__).resolve().parent.parent / "data"
training_data = pd.read_csv(data_dir / "train_split.csv")
validation_data = pd.read_csv(data_dir / "validation_split.csv")

training_features = training_data.drop(columns=["engine_id", "RUL"])
validation_features = validation_data.drop(columns=["engine_id", "RUL"])
if not training_features.columns.equals(validation_features.columns):
    raise ValueError("Training and validation input features must match.")

mean_training_rul = training_data["RUL"].mean()
predictions = pd.Series(mean_training_rul, index=validation_features.index)
rmse = mean_squared_error(validation_data["RUL"], predictions) ** 0.5

print(f"Validation RMSE (mean training RUL baseline): {rmse:.4f}")
