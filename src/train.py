"""Train and compare churn models.

Usage:
    python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

from src import data as data_mod
from src import evaluate as ev
from src.pipeline import LOGISTIC_PARAM_GRID, RANDOM_STATE, build_pipeline, candidate_models

DEFAULT_DATA = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and compare telco churn models.")
    p.add_argument("--data", default=DEFAULT_DATA, help="Path to the raw Kaggle CSV.")
    p.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction.")
    p.add_argument("--out", default="results", help="Directory for metrics output.")
    p.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable class_weight='balanced' to reproduce the unweighted baseline.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = data_mod.load_dataset(args.data)
    numeric, categorical = data_mod.column_groups(X)

    balance = data_mod.class_balance(y)
    print("Class balance:", json.dumps(balance, indent=2))
    print(f"Majority-class accuracy baseline: {ev.baseline_accuracy(y):.4f}\n")

    # Stratified split so the minority class is represented proportionally.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}

    for name, estimator in candidate_models(balanced=not args.no_class_weight).items():
        pipe = build_pipeline(estimator, numeric, categorical)

        # Cross-validated on the TRAINING split only. Preprocessing is re-fitted
        # inside each fold, so no test information reaches the transformers.
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        scores = ev.score(y_test, pipe.predict(X_test), proba)

        fitted[name] = pipe
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": cv_auc.mean(),
                "cv_roc_auc_std": cv_auc.std(),
                **scores.as_dict(),
            }
        )
        print(
            f"{name:<24} cv_auc={cv_auc.mean():.4f}  "
            f"test_auc={scores.roc_auc:.4f}  recall={scores.recall:.4f}"
        )

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results.to_csv(out_dir / "model_comparison.csv", index=False)
    print("\n", results.to_string(index=False), sep="")

    # Tune the logistic model on the training split only.
    best_lr = GridSearchCV(
        build_pipeline(candidate_models()["logistic_regression"], numeric, categorical),
        LOGISTIC_PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    ).fit(X_train, y_train)
    print(f"\nBest logistic params: {best_lr.best_params_}  (cv roc_auc={best_lr.best_score_:.4f})")

    # Threshold analysis on the strongest model by test AUC.
    top_name = results.iloc[0]["model"]
    top_proba = fitted[top_name].predict_proba(X_test)[:, 1]
    sweep = ev.threshold_sweep(y_test, top_proba)
    sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)

    t = ev.best_threshold_for_f1(y_test, top_proba)
    print(f"\nThreshold sweep for {top_name} (best F1 at t={t}):")
    print(sweep.to_string(index=False))
    print("\nConfusion matrix at default t=0.5:")
    print(ev.confusion(y_test, (top_proba >= 0.5).astype(int)).to_string())
    print(f"\nConfusion matrix at t={t}:")
    print(ev.confusion(y_test, (top_proba >= t).astype(int)).to_string())

    try:
        imp = ev.feature_importances(fitted["random_forest"])
        imp.to_csv(out_dir / "feature_importances.csv", index=False)
        print("\nTop features (random forest):")
        print(imp.to_string(index=False))
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
