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

| Performance Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Area Under ROC Curve (AUC) | 0.661 | 0.664 |
| Readmissions identified in highest-risk 20% | 666 | 675 |
| Share of all readmissions identified | 38.0% | 38.5% |

**Primary finding:** Selecting the highest-risk 20% of encounters using XGBoost identified a group containing approximately 38.5% of observed readmissions, compared with 20% expected through random selection.

XGBoost modestly outperformed logistic regression, although both models demonstrated useful risk stratification.

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

**Project status:** Research models finalized; supporting code, analysis, and documentation being prepared for publication.
