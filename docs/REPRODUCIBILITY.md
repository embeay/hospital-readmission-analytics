# Reproducing the frozen Version 2 analysis

1. Download `diabetic_data.csv` from the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
2. Place `diabetic_data.csv` in the **repository root** alongside `requirements.txt` (do not commit the dataset).
3. Install dependencies: `python -m pip install -r requirements.txt`.
4. Run `python src/model_logistic_v2.py` and `python src/model_xgboost_v2.py`.
5. The scripts write new results under `src/logistic_v2_results/` and `src/xgboost_v2_results/`. Compare the generated `metrics.json` files with the published frozen summaries under `results/`.

## Scope and limitations
- These are the frozen Version 2 scripts; no additional missing-value or diagnosis regrouping changes were made.
- Three administrative identifiers are treated as categorical; diagnoses remain categorical.
- Patient-grouped train/validation/test partitions are deterministic given the input and dependency versions. Exact floating-point reproducibility may vary by environment.
- XGBoost SHAP is computed on sparse preprocessed data with an additivity check.
- Models were assessed repeatedly on the test set during development, so results are exploratory rather than an untouched final validation.
- The model predicts recorded readmission, not preventability; the financial analysis is hypothetical.
- Do not use these models for patient care without external and prospective validation.
