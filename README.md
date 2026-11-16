# Hospital Readmission Analytics
### Predictive Modeling | Explainable AI | Healthcare Resource Optimization

## Project Overview

This independent healthcare analytics project explores how predictive modeling can help hospitals identify patients at elevated risk of 30-day readmission and allocate limited care-management resources more effectively.

The project integrates statistical modeling, machine learning, explainable artificial intelligence, and financial decision analysis to connect predictive insights with practical healthcare operations.

**Research framework:** Predict → Explain → Prioritize → Evaluate

## Research Objectives

1. Develop predictive models to estimate 30-day hospital readmission risk.
2. Compare regularized logistic regression with XGBoost.
3. Use SHAP (SHapley Additive exPlanations) to identify influential predictors.
4. Evaluate risk-based care-management prioritization against random patient selection.
5. Explore staffing requirements and potential financial value under hypothetical intervention assumptions.

## Dataset

**Source:** [UCI Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

- Original dataset: 101,766 hospital encounters
- Eligible modeling population: 99,340 encounters
- Held-out test population: 14,985 encounters
- Recorded 30-day readmissions in test population: 1,752 (11.69%)

The analysis used patient-grouped training, validation, and testing partitions to reduce patient-level data leakage.

## Modeling Approach

**Regularized Logistic Regression**
- L2 regularization
- Numeric standardization
- Categorical one-hot encoding

**XGBoost**
- Gradient-boosted decision trees
- Regularization and early stopping
- SHAP-based model explainability

Both models used consistent patient populations and data partitions.

## Key Findings

The following results were independently reproduced using Python 3.12 and the software versions documented in `environment_verified.txt`.

| Performance Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Area Under ROC Curve (AUC) | 0.6608 | 0.6638 |
| Precision-Recall AUC | 0.2287 | 0.2327 |
| Readmissions identified in highest-risk 20% | 666 | 667 |
| Share of all readmissions identified | 38.0% | 38.1% |

**Primary finding:** Selecting the highest-risk 20% of encounters using XGBoost identified 667 of 1,752 observed readmissions (38.1%), compared with approximately 20% expected through random selection.

Both models demonstrated useful risk stratification. XGBoost modestly outperformed logistic regression in overall discrimination.

### Original HIMSS Research Results

The original research environment produced slightly different XGBoost results:

- ROC-AUC: 0.6643
- Readmissions identified in highest-risk 20%: 675
- Share of all readmissions identified: 38.5%

The difference is associated with a change in the computational environment, although the precise cause has not been established. The GitHub reproducibility benchmarks use the independently verified local results above.

## Explainable AI

SHAP analysis identified the following influential predictors:

1. Previous inpatient encounters
2. Discharge disposition
3. Number of diagnoses
4. Insulin treatment status
5. Hospital length of stay

These findings describe how variables influence model predictions; they do not establish causal relationships or identify which readmissions are preventable.

## Care-Management Resource Optimization

The project proposes a tiered care-management approach:

- Highest-risk 10%: Comprehensive care-management assessment
- Next-highest-risk 10%: Targeted review and intervention assessment
- Remaining patients: Standard discharge planning with clinician-directed escalation

An illustrative financial model evaluates review costs, hypothetical intervention effectiveness, and the potential financial value of avoided readmissions.

Financial scenarios are not estimates of realized hospital savings.

## Reproducing the Analysis

The repository includes Python scripts for both predictive models, automated dataset retrieval, and a verification script.

### Tested Environment

- Python 3.12
- NumPy 2.5.5
- pandas 3.1.1
- scikit-learn 1.9.1
- XGBoost 3.4.1

The complete tested dependency list is available in [environment_verified.txt](environment_verified.txt).

### Setup and Execution

Download the repository and open a terminal in its main directory.

Create a Python 3.12 virtual environment:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Install the tested dependencies:

```cmd
python -m pip install -r environment_verified.txt
```

Download the original UCI dataset:

```cmd
python download_data.py
```

Train and evaluate both models:

```cmd
python src\model_logistic_v2.py
python src\model_xgboost_v2.py
```

Verify the results:

```cmd
python verify_results.py
```

The verification script checks the test population, model performance, risk-decile capture, and XGBoost SHAP reconstruction.

**Expected outcome:** `SUCCESS: Both models match the verified benchmarks.`

### Reproducibility Notes

- Both models use the same patient-grouped train, validation, and test partitions.
- The original UCI dataset is downloaded separately rather than stored in this repository.
- The verified environment was tested on Windows using Python 3.12.
- Results may vary slightly under different library versions or operating systems.
- The automated checks use a small numerical tolerance for ROC-AUC and PR-AUC but require exact matches for readmission counts.

## Technologies

Python | pandas | NumPy | scikit-learn | XGBoost | SHAP | Predictive Analytics | Healthcare Informatics

## Professional Presentation

**Beyond Prediction: Using Explainable AI and Resource Optimization to Improve Hospital Readmission Management**

Selected for presentation at the Indiana HIMSS student research webinar, November 19, 2026.

## Limitations

- Historical hospital data may not reflect current clinical practices.
- Readmissions outside participating hospitals may not be captured.
- Predictive risk does not necessarily indicate preventability.
- Financial estimates depend on hypothetical intervention assumptions.
- Prospective and external validation would be necessary before clinical implementation.

## References

Strack, B., et al. (2014). *Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records.* BioMed Research International, 2014, Article 781670. https://doi.org/10.1155/2014/781670

UCI Machine Learning Repository. *Diabetes 130-US Hospitals for Years 1999–2008.* https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

---

**Project status:** Both predictive models independently reproduced and verified in the documented Python 3.12 environment. Research documentation and portfolio presentation are being finalized.
