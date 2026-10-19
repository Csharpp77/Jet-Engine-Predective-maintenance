from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
data = pd.read_csv(data_dir / "processed_train.csv")

engine_ids = data["engine_id"].drop_duplicates()
if len(engine_ids) != 100:
    raise ValueError(f"Expected 100 unique engines, found {len(engine_ids)}.")

training_engine_ids = engine_ids.sample(frac=0.8, random_state=42)
training_ids = set(training_engine_ids)
validation_ids = set(engine_ids) - training_ids

training_data = data.loc[data["engine_id"].isin(training_ids)].copy()
validation_data = data.loc[data["engine_id"].isin(validation_ids)].copy()

training_data.to_csv(data_dir / "train_split.csv", index=False)
validation_data.to_csv(data_dir / "validation_split.csv", index=False)

overlap = training_ids & validation_ids
if overlap:
    raise AssertionError(f"Engines appear in both splits: {sorted(overlap)}")

print(f"Training engines: {len(training_ids)}")
print(f"Training rows: {len(training_data)}")
print(f"Validation engines: {len(validation_ids)}")
print(f"Validation rows: {len(validation_data)}")
print("Verified: no engine overlap between training and validation splits.")
