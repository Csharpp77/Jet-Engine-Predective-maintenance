from pathlib import Path

import pandas as pd


predictions_path = (
    Path(__file__).resolve().parent.parent / "outputs" / "test_predictions.csv"
)
data = pd.read_csv(predictions_path)

required_columns = {"engine_id", "actual_RUL", "predicted_RUL"}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
    raise ValueError(f"Prediction file is missing required columns: {sorted(missing_columns)}")

data["absolute_error"] = (data["predicted_RUL"] - data["actual_RUL"]).abs()
data["signed_error"] = data["predicted_RUL"] - data["actual_RUL"]

largest_errors = data.nlargest(10, "absolute_error")
print("10 engines with the largest absolute prediction errors:")
print(
    largest_errors[
        ["engine_id", "absolute_error", "actual_RUL", "predicted_RUL", "signed_error"]
    ].to_string(index=False)
)

mean_signed_error = data["signed_error"].mean()
if mean_signed_error > 0:
    tendency = "overestimate"
elif mean_signed_error < 0:
    tendency = "underestimate"
else:
    tendency = "neither overestimate nor underestimate"

print(f"\nMean signed error: {mean_signed_error:.4f}")
print(f"Average tendency: {tendency} RUL.")

rul_ranges = [
    ("0-25", data["actual_RUL"].between(0, 25)),
    ("26-50", data["actual_RUL"].between(26, 50)),
    ("51-75", data["actual_RUL"].between(51, 75)),
    ("76-100", data["actual_RUL"].between(76, 100)),
    (">100", data["actual_RUL"].gt(100)),
]
if sum(mask.sum() for _, mask in rul_ranges) != len(data):
    raise ValueError("Actual RUL values must be non-negative to fit the requested ranges.")

print("\nError metrics by actual RUL range:")
print(f"{'Actual RUL':>12} {'Engines':>8} {'RMSE':>10} {'MAE':>10} {'Mean signed error':>20}")
for range_label, mask in rul_ranges:
    range_data = data.loc[mask]
    signed_errors = range_data["signed_error"]
    rmse = (signed_errors.pow(2).mean()) ** 0.5
    mae = range_data["absolute_error"].mean()
    mean_range_signed_error = signed_errors.mean()
    print(
        f"{range_label:>12} {len(range_data):>8d} "
        f"{rmse:>10.4f} {mae:>10.4f} {mean_range_signed_error:>20.4f}"
    )
