# Streamlit Dashboard Plan

## Goal

Build a small FD001 predictive-maintenance dashboard that shows an engine's
predicted remaining useful life (RUL), its sensor history, and the model's
current evaluation metrics. The dashboard will use existing artifacts only;
it will not retrain the model.

## Proposed dashboard

1. **Engine selector**
   - Use a dropdown of engine IDs present in `Data/processed_test.csv`.
   - Keep the selection consistent across the RUL summary and sensor charts.

2. **Predicted RUL**
   - Load `outputs/lightgbm_model.txt` as a LightGBM `Booster`.
   - Select the chosen engine's final-cycle row from `Data/processed_test.csv`.
   - Use the feature names and order stored in the model, excluding
     `engine_id` and any target columns, to predict RUL in cycles.
   - Optionally compare the generated value with that engine's
     `predicted_RUL` in `outputs/test_predictions.csv` as a consistency check.

3. **Remaining-life status**
   - Display the prediction prominently with units (cycles).
   - Use proposed, clearly labeled thresholds: **Critical** for 0-25 cycles,
     **Warning** for 26-75 cycles, and **Monitoring** for more than 75 cycles.
     Keep thresholds together in one place so they can be adjusted later.
   - Explain that these are dashboard categories, not NASA-provided diagnoses.

4. **Sensor trends**
   - Read `Data/test_FD001.txt` with the 26 FD001 column names.
   - Filter the selected engine and plot sensor readings by cycle.
   - Provide a sensor selector (default to a small set such as sensors 2, 3,
     and 4) and clearly label axes and the engine ID.
   - Plot the complete observed history, not only the final-cycle feature row.

5. **Model evaluation**
   - Load `outputs/test_predictions.csv` and calculate RMSE and MAE from
     `actual_RUL` and `predicted_RUL`.
   - Show both metrics and the number of evaluated engines.
   - Label these as FD001 official test-set metrics; actual labels are for
     evaluation display only and must never be model inputs.

## Required files and data

- `src/dashboard.py` — Streamlit entry point and dashboard UI.
- `requirements.txt` — declare the dashboard/runtime dependencies if the
  project does not already have a dependency manifest. Expected packages:
  `streamlit`, `pandas`, `lightgbm`, and a plotting library (`matplotlib` or
  `plotly`).
- `outputs/lightgbm_model.txt` — existing trained LightGBM model.
- `outputs/test_predictions.csv` — existing per-engine predictions and
  official labels used for evaluation metrics.
- `Data/processed_test.csv` — final-cycle features in the trained model's
  feature order.
- `Data/test_FD001.txt` — raw cycle-by-cycle sensor history for plots.

Use paths relative to the dashboard file's project root rather than relying
on the current working directory. The workspace folder is displayed as
`Data`; Windows file lookup is case-insensitive.

## Implementation sequence

1. Add the dashboard dependency declarations without changing model artifacts.
2. Implement cached loading for the saved model, processed test features,
   prediction/evaluation CSV, and raw FD001 test data.
3. Validate required columns, engine coverage, and model feature names/order;
   keep `engine_id` and RUL labels out of inference features.
4. Add the engine selector, model prediction, status indicator, sensor
   selector/chart, and RMSE/MAE summary.
5. Launch the Streamlit app and verify several engine selections, matching
   prediction rows, sensor histories, and displayed evaluation metrics.

## Scope and safeguards

- Do not retrain or overwrite the existing LightGBM model.
- Do not use `actual_RUL` or `RUL_FD001.txt` as input features.
- Do not modify the source dataset or precomputed prediction CSV.
- This plan does not create the dashboard itself.
