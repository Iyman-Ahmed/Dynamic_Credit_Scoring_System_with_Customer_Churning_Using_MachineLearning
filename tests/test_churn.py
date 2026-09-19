import numpy as np
import pandas as pd
import pytest
from app import prepare, split, build_model, churn_probability


def dataset():
    return pd.read_csv('BankChurners.csv')


def test_target_and_leakage_columns():
    raw = dataset()
    X, y = prepare(raw)
    assert y[raw.Attrition_Flag == 'Attrited Customer'].eq(1).all()
    assert y[raw.Attrition_Flag == 'Existing Customer'].eq(0).all()
    assert not any(c.startswith('Naive_Bayes_') for c in X)
    assert 'CLIENTNUM' not in X and 'Attrition_Flag' not in X
    assert X.Total_Amt_Chng_Q4_Q1.equals(raw.Total_Amt_Chng_Q4_Q1)
    raw.loc[0, 'Attrition_Flag'] = 'unexpected'
    with pytest.raises(ValueError):
        prepare(raw)


def test_splits_and_unseen_category():
    X, y = prepare(dataset())
    train, val, test = split(X, y)
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert set(train) | set(val) | set(test) == set(X.index)
    model = build_model(y.loc[train]).fit(X.loc[train], y.loc[train])
    unseen = X.loc[test[:3]].copy()
    unseen['Gender'] = 'UNSEEN'
    probabilities = churn_probability(model, unseen)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    encoder = model['preprocess'].named_transformers_['categorical']['encode']
    assert not any('UNSEEN' in categories for categories in encoder.categories_)


def test_probability_column_is_churn_even_with_reversed_classes():
    class Reversed:
        classes_ = np.array([1, 0])
        def predict_proba(self, X):
            return np.array([[.8, .2]])
    assert churn_probability(Reversed(), None)[0] == .8
