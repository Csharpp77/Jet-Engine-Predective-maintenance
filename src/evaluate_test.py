from pathlib import Path

import matplotlib.pyplot as plt
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

model = Booster(model_file=str(outputs_dir / "lightgbm_model.txt"))
test_data = pd.read_csv(data_dir / "processed_test.csv")
labels = pd.read_csv(
    data_dir / "RUL_FD001.txt",
    sep=r"\s+",
    header=None,
    names=["actual_RUL"],
)

# The official RUL file is ordered by test engine ID; align it to sorted IDs.
engine_ids = sorted(test_data["engine_id"].unique())
if len(engine_ids) != len(labels):
    raise ValueError(
        f"Found {len(engine_ids)} test engines but {len(labels)} official RUL labels."
    )
if test_data["engine_id"].duplicated().any():
    raise ValueError("Processed test data must contain one row per engine.")

feature_columns = model.feature_name()
excluded_columns = {"engine_id", "RUL", "actual_RUL", "predicted_RUL"}
if excluded_columns.intersection(feature_columns):
    raise ValueError("Model feature names contain an engine ID or target column.")
missing_features = [column for column in feature_columns if column not in test_data]
if missing_features:
    raise ValueError(f"Processed test data is missing model features: {missing_features}")

# Keep model inputs to the exact feature set and order used during training.
test_features = test_data.set_index("engine_id").loc[engine_ids, feature_columns]
predicted_rul = model.predict(test_features)
results = pd.DataFrame(
    {
        "engine_id": engine_ids,
        "actual_RUL": labels["actual_RUL"].to_numpy(),
        "predicted_RUL": predicted_rul,
    }
)

rmse = mean_squared_error(results["actual_RUL"], results["predicted_RUL"]) ** 0.5
mae = mean_absolute_error(results["actual_RUL"], results["predicted_RUL"])
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

outputs_dir.mkdir(parents=True, exist_ok=True)
predictions_path = outputs_dir / "test_predictions.csv"
results.to_csv(predictions_path, index=False)

figure, axis = plt.subplots(figsize=(8, 8))
axis.scatter(results["actual_RUL"], results["predicted_RUL"], alpha=0.75)
plot_min = min(results["actual_RUL"].min(), results["predicted_RUL"].min())
plot_max = max(results["actual_RUL"].max(), results["predicted_RUL"].max())
axis.plot([plot_min, plot_max], [plot_min, plot_max], "r--", label="Ideal prediction")
axis.set_title("Actual vs. Predicted RUL on FD001 Test Set")
axis.set_xlabel("Actual RUL (cycles)")
axis.set_ylabel("Predicted RUL (cycles)")
axis.legend()
figure.tight_layout()

plot_path = outputs_dir / "test_predictions.png"
figure.savefig(plot_path, dpi=150)
plt.close(figure)

print(f"Saved predictions to: {predictions_path}")
print(f"Saved actual-vs-predicted plot to: {plot_path}")
