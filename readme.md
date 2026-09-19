# Customer Churn Prediction

A reproducible XGBoost classification experiment on the included BankChurners dataset (10,127 rows). The positive label is explicitly **Attrited Customer = 1**. This predicts customer attrition, not default risk or creditworthiness.

## Corrections to the original experiment

The earlier LabelEncoder assigned Existing Customer to 1, but its probability was incorrectly called churn probability. The former 300–850 score mapping had no credit-risk validation and has been removed. The old 98.9% claim is superseded by the reproducible results below. The repository URL retains its original name to preserve links.

Preprocessing now fits only on training rows, one-hot encodes categorical fields, preserves fractional numeric values, and excludes CLIENTNUM plus both target-derived Naive_Bayes columns. Two engineered transaction features are added. No Optuna, SHAP, or customer-segmentation implementation is claimed.

## Reproduce

Use Python 3.11+ in a virtual environment. On macOS XGBoost also requires `brew install libomp`.

```sh
pip install -r requirements.txt
python app.py
python -m pytest -q
```

The recorded run used Python 3.14, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1 and XGBoost 3.4.1. Full environment: `requirements-lock.txt`.

The split is stratified 60/20/20 (6,075 training, 2,026 validation, 2,026 test rows; seeds 42 and 43). Model parameters are fixed; the decision threshold is selected using validation F1, then test metrics are computed. The model is not refit using validation rows. There is no hyperparameter search on test data.

## Held-out results

| Metric | Result |
|---|---:|
| Churn average precision (AP) | 0.9634 |
| Churn ROC AUC | 0.9922 |
| Churn precision | 88.27% |
| Churn recall | 88.00% |
| Churn F1 | 0.8814 |
| Overall accuracy | 96.20% |
| Majority-class baseline accuracy | 83.96% |
| Constant-score baseline AP | 0.1604 |
| Brier score | 0.0253 |

At the validation-selected threshold of approximately 0.50, the test confusion matrix is TN=1,663, FP=38, FN=39, TP=286. Metrics and the dataset checksum are in `results/evaluation.json`; row-level test outputs are in `results/predictions.csv`.

## Limits

This is one random split of a public dataset, not temporal or external validation. Feature availability at a prospective prediction date has not been established. Class-weighted model probabilities are not calibrated real-world churn rates. The data has no repayment/default outcome, so it cannot support a credit score. No deployed model, business impact, or real users are claimed.
