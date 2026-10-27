"""Tests for data loading and cleaning."""

import pandas as pd
import pytest

from src import data as d


def raw_frame() -> pd.DataFrame:
    """A miniature frame with the quirks of the real CSV."""
    return pd.DataFrame(
        {
            "customerID": ["a1", "b2", "c3", "d4"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0],
            "tenure": [0, 12, 34, 5],
            "MonthlyCharges": [29.85, 56.95, 53.85, 70.70],
            # Blank string is how the real file encodes a tenure-0 customer.
            "TotalCharges": [" ", "1889.5", "108.15", "151.65"],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "Churn": ["No", "No", "Yes", "No"],
        }
    )


def test_total_charges_becomes_numeric():
    cleaned = d.clean(raw_frame())
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_blank_total_charges_filled_with_zero():
    cleaned = d.clean(raw_frame())
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert not cleaned["TotalCharges"].isna().any()


def test_customer_id_dropped():
    assert d.ID_COLUMN not in d.clean(raw_frame()).columns


def test_target_is_binary():
    cleaned = d.clean(raw_frame())
    assert set(cleaned["Churn"].unique()) <= {0, 1}
    assert cleaned["Churn"].tolist() == [0, 0, 1, 0]


def test_column_groups_separate_numeric_from_categorical():
    X, _ = d.split_features_target(d.clean(raw_frame()))
    numeric, categorical = d.column_groups(X)
    assert set(numeric) == {"tenure", "MonthlyCharges", "TotalCharges"}
    # SeniorCitizen must land in categorical, not numeric.
    assert "SeniorCitizen" in categorical
    assert "gender" in categorical


def test_no_column_is_both_numeric_and_categorical():
    X, _ = d.split_features_target(d.clean(raw_frame()))
    numeric, categorical = d.column_groups(X)
    assert set(numeric).isdisjoint(set(categorical))
    assert len(numeric) + len(categorical) == X.shape[1]


def test_class_balance_reports_positive_rate():
    _, y = d.split_features_target(d.clean(raw_frame()))
    balance = d.class_balance(y)
    assert balance["n"] == 4
    assert balance["positive"] == 1
    assert balance["positive_rate"] == pytest.approx(0.25)


def test_load_raw_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        d.load_raw("does/not/exist.csv")
