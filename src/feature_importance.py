from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import Booster


project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "outputs" / "lightgbm_model.txt"
model = Booster(model_file=str(model_path))

importance = pd.Series(
    model.feature_importance(importance_type="gain"),
    index=model.feature_name(),
    name="gain",
).sort_values(ascending=False).head(15)

print("Top 15 features ranked by LightGBM gain:")
print(importance.to_string())

figure, axis = plt.subplots(figsize=(10, 8))
importance.sort_values().plot(kind="barh", ax=axis)
axis.set_title("Top 15 Feature Importances (LightGBM Gain)")
axis.set_xlabel("Gain")
axis.set_ylabel("Feature")
figure.tight_layout()

output_path = project_dir / "outputs" / "feature_importance.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output_path, dpi=150)
plt.close(figure)
print(f"\nSaved feature importance chart to: {output_path}")
