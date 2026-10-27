"""Model pipelines.

The important design decision here: scaling and one-hot encoding live *inside*
a scikit-learn ``Pipeline``. In the original notebook the scaler was fitted on
the full feature matrix before the train/test split, which leaks test-set
statistics (means and standard deviations) into training. Wrapping the
preprocessing in a Pipeline means it is re-fitted on the training portion of
every cross-validation fold, so reported scores reflect genuinely unseen data.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical ones."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def candidate_models(balanced: bool = True) -> dict:
    """Return the classifiers to compare.

    ``balanced=True`` sets ``class_weight="balanced"`` where the estimator
    supports it. Churn is roughly a 1:3 minority class, so unweighted models
    tend to score well on accuracy while missing much of the positive class --
    exactly the failure mode this project is meant to expose.
    """
    weight = "balanced" if balanced else None

    logistic = LogisticRegression(max_iter=1000, class_weight=weight, random_state=RANDOM_STATE)

    return {
        "logistic_regression": logistic,
        "decision_tree": DecisionTreeClassifier(class_weight=weight, random_state=RANDOM_STATE),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight=weight, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "logistic_bagging": BaggingClassifier(
            estimator=logistic, n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "logistic_adaboost": AdaBoostClassifier(
            estimator=logistic, n_estimators=50, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(estimator, numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    """Wrap an estimator with its preprocessing so the two are fitted together."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_columns, categorical_columns)),
            ("model", estimator),
        ]
    )


LOGISTIC_PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
}
