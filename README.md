# Telco Customer Churn Prediction

Predicting customer churn for a telecom provider, and — more to the point —
measuring it honestly on an imbalanced dataset where accuracy is a misleading
headline metric.

This repository is a rebuild of a graduate course project. The original
notebook is preserved at [`notebooks/original_term_project.ipynb`](notebooks/original_term_project.ipynb);
the [What changed and why](#what-changed-and-why) section documents the four
defects found on review and how each was fixed.

## The problem

The [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
has 7,043 customers described by 20 features — demographics, subscribed
services, contract type, and billing — with a binary churn label.

About 26.5% of customers churn. That matters more than it first appears:
**a model that predicts "no churn" for every customer scores 73.5% accuracy**
while being completely useless. Any accuracy figure in that neighbourhood is
reporting the base rate, not model skill. This project therefore leads with
ROC-AUC, average precision, and per-class recall, and treats the decision
threshold as a modelling choice rather than a fixed 0.5.

## Quick start

```bash
git clone https://github.com/MihirParekh14/telco-churn-prediction.git
cd telco-churn-prediction

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Download the CSV from Kaggle and place it here (the data/ directory is gitignored):
#   data/WA_Fn-UseC_-Telco-Customer-Churn.csv

python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Run the tests:

```bash
pytest --cov=src --cov-report=term-missing
```

Or with Docker:

```bash
docker build -t telco-churn .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" telco-churn \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## What it does

`python -m src.train` will:

1. Load and clean the raw CSV (see the `TotalCharges` note below).
2. Report the class balance and the majority-class accuracy baseline.
3. Split the data **stratified**, holding out 20% as a test set.
4. Cross-validate six models on the training split only, with preprocessing
   re-fitted inside every fold.
5. Score each model on the held-out test set: accuracy, precision, recall, F1,
   ROC-AUC, and average precision.
6. Grid-search the logistic regression's regularisation strength.
7. Sweep the decision threshold on the strongest model and print confusion
   matrices at t=0.5 and at the F1-optimal threshold.
8. Write feature importances from the random forest.

Outputs land in `results/` as CSV.

## Models compared

| Model | Notes |
|---|---|
| Logistic regression | Interpretable baseline; `class_weight="balanced"` |
| Decision tree | Non-linear, prone to overfitting here |
| Gaussian naive Bayes | Independence assumption is a poor fit for these features |
| Random forest | Ensemble; usually strongest on ROC-AUC |
| Logistic + bagging | Variance reduction over the linear baseline |
| Logistic + AdaBoost | Sequential reweighting of hard cases |

## What changed and why

Reviewing the original notebook turned up four defects. Each is worth stating
plainly, because the fixes are the substance of this rebuild.

### 1. Preprocessing leaked test data into training

The original fitted a `StandardScaler` on the **entire** feature matrix and
only then split into train and test:

```python
X_scaled = scaler.fit_transform(X)                  # sees every row
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, ...)
```

The scaler's means and standard deviations are therefore computed partly from
the test set, so reported scores are optimistic. `cross_val_score` was then
run on `X_scaled` as well, which leaks into every fold.

**Fix:** preprocessing lives inside a `Pipeline`
([`src/pipeline.py`](src/pipeline.py)), so it is re-fitted on the training
portion of each fold. `tests/test_pipeline.py::test_preprocessor_unfitted_before_training`
guards against a regression.

### 2. `TotalCharges` silently exploded into thousands of columns

`TotalCharges` is stored as text and is blank for the 11 customers with
`tenure == 0`. Pandas reads it as `object`, so `pd.get_dummies(data)` treated
it as categorical and created one indicator column per distinct charge value.
This is the likely reason naive Bayes scored 28.8% accuracy in the original
run — far below the 73.5% you get by guessing the majority class.

**Fix:** coerce to numeric and fill the blanks with `0.0` — a customer who has
never been billed has been charged nothing ([`src/data.py`](src/data.py)).

### 3. The validation split was created and never used

`X_val` and `y_val` were carved out and then never referenced; every
evaluation ran against `X_test`.

**Fix:** a single stratified train/test split, with model selection done by
cross-validation *within* the training split. The test set is touched once.

### 4. Accuracy was the headline metric on an imbalanced problem

The original ranked models by accuracy, where the majority-class baseline is
73.5%. A model scoring 78.9% is 5 points above guessing, which is a far less
impressive result than it reads as.

**Fix:** ROC-AUC and average precision for ranking, per-class recall reported
alongside, `class_weight="balanced"` on estimators that support it, and an
explicit threshold sweep ([`src/evaluate.py`](src/evaluate.py)).

## Why threshold choice matters here

Retaining a customer is cheap; losing one is not. A model tuned for F1 at
t=0.5 misses a large share of churners. Lowering the threshold trades
precision for recall — more customers flagged, more of the real churners
caught. The correct threshold depends on the cost of a retention offer versus
the value of a retained customer, which is a business decision, not a
statistical one. The sweep in `results/threshold_sweep.csv` exists so that
decision can be made with the tradeoff visible.

## Repository layout

```
src/
  data.py        loading, cleaning, class-balance reporting
  pipeline.py    ColumnTransformer + Pipeline definitions, model registry
  evaluate.py    metrics, threshold sweep, confusion matrices, importances
  train.py       CLI entry point
tests/           pytest suite (15 tests)
notebooks/       the original course notebook, unmodified
```

## Data

The dataset is not committed. Download it from
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and
place the CSV in `data/`.

## License

MIT
