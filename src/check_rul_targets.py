from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
training_data = pd.read_csv(data_dir / "train_split.csv")
test_labels = pd.read_csv(
    data_dir / "RUL_FD001.txt",
    sep=r"\s+",
    header=None,
    names=["RUL"],
)

training_rul = training_data["RUL"]
official_test_rul = test_labels["RUL"]

print("Training split RUL:")
print(f"Minimum: {training_rul.min()}")
print(f"Maximum: {training_rul.max()}")
print(f"Mean: {training_rul.mean():.4f}")
print(f"Rows equal to 125: {training_rul.eq(125).sum()}")

print("\nOfficial test RUL labels:")
print(f"Minimum: {official_test_rul.min()}")
print(f"Maximum: {official_test_rul.max()}")
print(f"Mean: {official_test_rul.mean():.4f}")
print(f"Labels above 125: {official_test_rul.gt(125).sum()}")
