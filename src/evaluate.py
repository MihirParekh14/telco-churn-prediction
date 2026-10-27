"""Evaluation helpers.

Accuracy is a poor headline metric on this dataset: predicting "no churn" for
everyone scores about 73%. The functions here report precision, recall, F1 and
both ROC-AUC and average precision (PR-AUC), and include a threshold sweep,
because on an imbalanced problem the decision threshold is a modelling choice
rather than a detail.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Scores:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float

    def as_dict(self) -> dict:
        return asdict(self)


def score(y_true, y_pred, y_proba) -> Scores:
    """Compute the full metric set for one set of predictions."""
    return Scores(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba),
        average_precision=average_precision_score(y_true, y_proba),
    )


def baseline_accuracy(y_true) -> float:
    """Accuracy of always predicting the majority class."""
    rate = float(np.mean(y_true))
    return max(rate, 1 - rate)


def threshold_sweep(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Precision, recall and F1 across candidate decision thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "predicted_positive": int(y_pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def best_threshold_for_f1(y_true, y_proba) -> float:
    """Threshold maximising F1 -- a defensible default when both errors cost."""
    sweep = threshold_sweep(y_true, y_proba)
    return float(sweep.loc[sweep["f1"].idxmax(), "threshold"])


def confusion(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix as a labelled frame."""
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["actual_no_churn", "actual_churn"],
        columns=["predicted_no_churn", "predicted_churn"],
    )


def feature_importances(fitted_pipeline, top_n: int = 15) -> pd.DataFrame:
    """Feature importances from a fitted pipeline, with encoded names resolved."""
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{type(model).__name__} exposes no feature_importances_")

    names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    frame = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    return frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
