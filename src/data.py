"""Load and clean the Telco Customer Churn dataset.

The raw CSV has one quirk that matters: ``TotalCharges`` is stored as text and
contains blank strings for customers whose ``tenure`` is 0 (they have not been
billed yet). Pandas therefore reads the column as ``object``. If the frame is
passed straight to ``pd.get_dummies``, that column is treated as categorical and
explodes into thousands of indicator columns, which wrecks downstream models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw Kaggle CSV into a DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the dataset from "
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and place the CSV at this path."
        )
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame: fix dtypes, drop the ID, binarise the target."""
    df = df.copy()

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # TotalCharges arrives as text with blanks for tenure-0 customers.
    # Coerce to numeric, then fill those rows with 0.0 -- a customer who has
    # never been billed has charged nothing, which is more faithful than
    # imputing the median.
    # Coerce unconditionally: the column may arrive as object or as a pandas
    # string dtype depending on the pandas version, so do not branch on dtype.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # SeniorCitizen is already 0/1 but stored as int; treat it as categorical
    # so it is one-hot encoded consistently with the other yes/no fields.
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with the target removed from the feature frame."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(numeric_columns, categorical_columns)`` present in ``X``."""
    numeric = [c for c in NUMERIC_COLUMNS if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def load_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper: read, clean, and split into features and target."""
    return split_features_target(clean(load_raw(csv_path)))


def class_balance(y: pd.Series) -> dict[str, float]:
    """Report the positive rate, for documenting class imbalance."""
    positive_rate = float(np.mean(y))
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "positive_rate": positive_rate,
        "imbalance_ratio": (1 - positive_rate) / positive_rate if positive_rate else float("inf"),
    }
