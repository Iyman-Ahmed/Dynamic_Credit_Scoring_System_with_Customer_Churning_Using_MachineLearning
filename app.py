"""Reproducible customer attrition experiment. No creditworthiness inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

TARGET = {"Existing Customer": 0, "Attrited Customer": 1}


def prepare(data):
    y = data["Attrition_Flag"].map(TARGET)
    if y.isna().any():
        raise ValueError("Missing or unknown attrition label")
    excluded = [c for c in data if c.startswith("Naive_Bayes_") or c in
                ("CLIENTNUM", "Attrition_Flag")]
    X = data.drop(columns=excluded).copy()
    X["Avg_Trans_value"] = X.Total_Trans_Amt / X.Total_Trans_Ct.replace(0, np.nan)
    X["Total_Chng_Q4_Q1"] = (X.Total_Amt_Chng_Q4_Q1 + X.Total_Ct_Chng_Q4_Q1) / 2
    return X.replace([np.inf, -np.inf], np.nan), y.astype(int)


def split(X, y):
    train, test = train_test_split(X.index, test_size=.2, stratify=y, random_state=42)
    train, validation = train_test_split(train, test_size=.25, stratify=y.loc[train], random_state=43)
    return train, validation, test


def build_model(y_train):
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number))])
    return Pipeline([("preprocess", preprocess), ("model", XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=.1, n_jobs=2, random_state=42,
        scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
        eval_metric="logloss"))])


def churn_probability(model, X):
    column = list(model.classes_).index(1)
    return model.predict_proba(X)[:, column]


def run(dataset, output):
    X, y = prepare(pd.read_csv(dataset))
    train, validation, test = split(X, y)
    model = build_model(y.loc[train]).fit(X.loc[train], y.loc[train])
    vp = churn_probability(model, X.loc[validation])
    thresholds = np.linspace(.05, .95, 91)
    threshold = float(max(thresholds, key=lambda t: f1_score(y.loc[validation], vp >= t)))
    p = churn_probability(model, X.loc[test])
    predictions = p >= threshold
    report = {
        "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        "positive_class": "Attrited Customer = 1", "seed_train_test": 42, "seed_train_validation": 43,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "threshold_selected_on_validation_f1": threshold,
        "validation_f1": float(f1_score(y.loc[validation], vp >= threshold)),
        "test": {"average_precision": float(average_precision_score(y.loc[test], p)),
                 "roc_auc": float(roc_auc_score(y.loc[test], p)),
                 "brier_score": float(brier_score_loss(y.loc[test], p)),
                 "confusion_matrix_labels_0_1": confusion_matrix(y.loc[test], predictions).tolist(),
                 "classification_report": classification_report(y.loc[test], predictions, output_dict=True)},
        "baseline": {"majority_accuracy": float((y.loc[test] == 0).mean()),
                     "constant_score_average_precision": float(y.loc[test].mean())},
        "limitations": ["Single random split of a public dataset; not external validation.",
                        "Weighted model probabilities are not calibrated churn rates.",
                        "No default/repayment labels: no credit score or creditworthiness claim."]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame({"row_index": test, "actual_churn": y.loc[test].to_numpy(),
                  "churn_model_probability": p, "predicted_churn": predictions.astype(int)}).to_csv(
                      output / "predictions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("BankChurners.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    run(args.data, args.output)
