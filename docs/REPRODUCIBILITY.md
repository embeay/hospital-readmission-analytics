
# Reproducing the Version 2 Analysis

## Environment

The independently verified reference environment uses Python 3.12 and the exact dependency versions listed in `environment_verified.txt`.

## Reproduction Steps

1. Download or clone this repository.
2. Open a terminal in the repository root.
3. Create and activate a Python 3.12 virtual environment.
4. Install dependencies using:

   `python -m pip install -r environment_verified.txt`

5. Download the original UCI dataset:

   `python download_data.py`

6. Execute both models:

   `python src/model_logistic_v2.py`

   `python src/model_xgboost_v2.py`

7. Verify the generated results:

   `python verify_results.py`

Successful verification displays:

`SUCCESS: Both models match the verified benchmarks.`

## Published Results

The `results/` directory distinguishes two sets of outputs.

**Independently verified Python 3.12 results:**
- `logistic_v2_verified_metrics.json`
- `xgboost_v2_verified_metrics.json`

**Original HIMSS research results:**
- `logistic_v2_original_himss_metrics.json`
- `xgboost_v2_original_himss_metrics.json`

The original research results differ slightly from the verified Python 3.12 outputs. The exact cause has not been established. The verified files are the reference benchmarks for the documented environment.

## Methodological Notes

- Both models use patient-grouped training, validation, and testing partitions.
- The Version 2 model specifications remain unchanged.
- Three administrative identifiers and diagnosis codes are treated as categorical variables.
- XGBoost SHAP calculations use sparse preprocessed data and include an additivity validation.
- Generated model outputs are saved under `src/logistic_v2_results/` and `src/xgboost_v2_results/`.

## Research Limitations

- The dataset reflects historical hospital encounters from 1999–2008.
- Model performance is modest and does not establish clinical utility.
- Repeated assessment on the test set during development means the results are exploratory rather than an untouched final validation.
- SHAP explains model predictions, not causal relationships.
- Predicted readmission risk does not establish preventability.
- Financial scenarios are hypothetical, not observed savings.
- External and prospective validation would be necessary before clinical deployment.
