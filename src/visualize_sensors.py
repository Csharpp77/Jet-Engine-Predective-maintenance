from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


column_names = (
    ["engine_id", "cycle"]
    + [f"setting{i}" for i in range(1, 4)]
    + [f"sensor{i}" for i in range(1, 22)]
)
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "data" / "train_FD001.txt"
data = pd.read_csv(dataset_path, sep=r"\s+", header=None, names=column_names)

engine_ids = [1, 10, 50]
sensor_columns = ["sensor2", "sensor3", "sensor4"]
figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

for axis, sensor in zip(axes, sensor_columns):
    for engine_id in engine_ids:
        engine_data = data.loc[data["engine_id"] == engine_id].sort_values("cycle")
        axis.plot(
            engine_data["cycle"],
            engine_data[sensor],
            label=f"Engine {engine_id}",
        )

    axis.set_title(f"{sensor} vs. Cycle")
    axis.set_xlabel("Cycle")
    axis.set_ylabel(sensor)
    axis.legend()

figure.tight_layout()

output_path = project_dir / "outputs" / "sensor_trends.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output_path)
plt.close(figure)
