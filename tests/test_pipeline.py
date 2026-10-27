"""Tests for the model pipeline, with emphasis on leakage prevention."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from src import data as d
from src import evaluate as ev
from src.pipeline import build_pipeline, build_preprocessor, candidate_models


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.uniform(20, 120, n),
            "TotalCharges": rng.uniform(0, 8000, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "gender": rng.choice(["Male", "Female"], n),
        }
    )
    y = pd.Series(rng.choice([0, 1], n, p=[0.73, 0.27]))
    return X, y


def test_pipeline_is_a_pipeline(xy):
    X, _ = xy
    numeric, categorical = d.column_groups(X)
    pipe = build_pipeline(candidate_models()["logistic_regression"], numeric, categorical)
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["preprocess", "model"]


def test_preprocessor_unfitted_before_training(xy):
    """Preprocessing must not be fitted until fit() is called on training data.

    This is the guard against the original notebook's bug, where the scaler was
    fitted on the full feature matrix before the train/test split.
    """
    X, _ = xy
    numeric, categorical = d.column_groups(X)
    pre = build_preprocessor(numeric, categorical)
    with pytest.raises(NotFittedError):
        pre.transform(X)


def test_fit_predict_end_to_end(xy):
    X, y = xy
    numeric, categorical = d.column_groups(X)
    pipe = build_pipeline(candidate_models()["random_forest"], numeric, categorical)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(y),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_all_candidate_models_fit(xy):
    X, y = xy
    numeric, categorical = d.column_groups(X)
    for est in candidate_models().values():
        build_pipeline(est, numeric, categorical).fit(X, y)


def test_unseen_category_does_not_crash(xy):
    """handle_unknown='ignore' must tolerate a category absent from training."""
    X, y = xy
    numeric, categorical = d.column_groups(X)
    pipe = build_pipeline(candidate_models()["logistic_regression"], numeric, categorical)
    pipe.fit(X.iloc[:150], y.iloc[:150])
    unseen = X.iloc[150:].copy()
    unseen.loc[unseen.index[0], "Contract"] = "Four year"
    pipe.predict(unseen)


def test_threshold_sweep_shape_and_monotonic_positives(xy):
    _, y = xy
    proba = np.linspace(0, 1, len(y))
    sweep = ev.threshold_sweep(y, proba)
    assert {"threshold", "precision", "recall", "f1"} <= set(sweep.columns)
    # Raising the threshold can never increase the number of positive calls.
    assert sweep["predicted_positive"].is_monotonic_decreasing


def test_baseline_accuracy_is_majority_rate(xy):
    _, y = xy
    assert ev.baseline_accuracy(y) == pytest.approx(max(y.mean(), 1 - y.mean()))
