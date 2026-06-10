"""Modeling pipeline for fraud detection portfolio.

Three models, one engineered Parquet, strict time-based 70/15/15 split:

  1. LogisticRegression  - linear baseline (lift floor)
  2. LightGBM            - production champion (tree-based)
  3. IsolationForest     - unsupervised anomaly companion

LightGBM uses (train -> val) for early stopping; the test set is held
out and touched only once. Sequential-ID and monotonic-time columns
(TransactionID, TransactionDT, Transaction_day) are dropped to prevent
the leakage previously surfaced by SHAP in the notebook track.
"""

import time
from dataclasses import dataclass
from typing import NamedTuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "isFraud"
TIME_COL = "TransactionDT"
LEAKY_COLS = ["TransactionID", "TransactionDT", "TransactionDay"]
SEED = 42


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------

class Split(NamedTuple):
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Split:
    """Strict time-based 70/15/15 split using TransactionDT quantiles.

    Train comes before val comes before test on the time axis. No overlap.
    """
    times = df[TIME_COL].values
    train_cut = float(np.quantile(times, train_frac))
    val_cut = float(np.quantile(times, train_frac + val_frac))

    train = times <= train_cut
    val = (times > train_cut) & (times <= val_cut)
    test = times > val_cut

    # Sanity: temporal ordering preserved across splits
    assert df.loc[train, TIME_COL].max() <= df.loc[val, TIME_COL].min(), \
        "train/val time overlap"
    assert df.loc[val, TIME_COL].max() <= df.loc[test, TIME_COL].min(), \
        "val/test time overlap"

    return Split(train, val, test)


# ----------------------------------------------------------------------
# Feature prep
# ----------------------------------------------------------------------

def feature_columns(df: pd.DataFrame) -> list[str]:
    """Everything except target and leaky time/ID columns."""
    drop = set([TARGET] + LEAKY_COLS)
    return [c for c in df.columns if c not in drop]


def detect_categoricals(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Catch category, object, and the newer pandas string dtype."""
    out = []
    for c in cols:
        dt = df[c].dtype
        if dt.name in ("category", "object") or pd.api.types.is_string_dtype(dt):
            out.append(c)
    return out


def make_lgbm_frame(
    df: pd.DataFrame, feat_cols: list[str], cat_cols: list[str]
) -> pd.DataFrame:
    """Categoricals as pandas category dtype; LGBM handles them natively."""
    X = df[feat_cols].copy()
    for c in cat_cols:
        if X[c].dtype.name != "category":
            X[c] = X[c].astype("category")
    return X


def make_linear_frame(
    df: pd.DataFrame, feat_cols: list[str], cat_cols: list[str]
) -> pd.DataFrame:
    """Categoricals as int codes for LR / IF (need numeric input).

    NaN -> -1 via pandas category codes, so missing is its own bucket.
    """
    X = df[feat_cols].copy()
    for c in cat_cols:
        if X[c].dtype.name != "category":
            X[c] = X[c].astype("category")
        X[c] = X[c].cat.codes.astype(np.int32)
    return X


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_logistic(X_train, y_train, X_val, y_val) -> Pipeline:
    """LR with median imputation + standard scaling. Class-balanced weighting."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, solver="lbfgs", class_weight="balanced",
            n_jobs=-1, random_state=SEED,
        )),
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"  fit time: {time.time() - t0:.1f}s")
    val_score = pipe.predict_proba(X_val)[:, 1]
    print(f"  val PR-AUC: {average_precision_score(y_val, val_score):.4f}")
    return pipe


def train_lightgbm(X_train, y_train, X_val, y_val, cat_cols) -> lgb.LGBMClassifier:
    """LGBM with scale_pos_weight=3 (from prior systematic search).

    Early stopping on validation PR-AUC; n_estimators is the ceiling, not
    the actual trained count. best_iteration_ is what's actually used at
    inference time.
    """
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=3,
        objective="binary",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  fit time: {time.time() - t0:.1f}s  |  best iter: {model.best_iteration_}")
    val_score = model.predict_proba(X_val)[:, 1]
    print(f"  val PR-AUC: {average_precision_score(y_val, val_score):.4f}")
    return model


def train_isolation_forest(X_train, contamination: float) -> Pipeline:
    """IF with contamination set to observed fraud rate.

    Unsupervised - does not see y_train. Pipeline wraps the same numeric
    prep as LR so the two models see identical inputs (fair comparison).
    """
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", IsolationForest(
            n_estimators=200,
            max_samples=256,
            contamination=contamination,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])
    t0 = time.time()
    pipe.fit(X_train)
    print(f"  fit time: {time.time() - t0:.1f}s")
    return pipe


def iforest_anomaly_score(pipe: Pipeline, X) -> np.ndarray:
    """Higher score = more anomalous.

    sklearn IF.decision_function uses the opposite convention (higher =
    more normal), so we negate. The pipeline passes decision_function
    through to the final estimator after preprocessing.
    """
    return -pipe.decision_function(X)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

@dataclass
class EvalResult:
    model: str
    pr_auc: float
    roc_auc: float
    threshold: float
    confusion_matrix: list
    n_samples: int
    n_positive: int


def evaluate(y_true, y_score, name: str, threshold: float = 0.5) -> EvalResult:
    y_pred = (y_score >= threshold).astype(int)
    return EvalResult(
        model=name,
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        threshold=threshold,
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        n_samples=int(len(y_true)),
        n_positive=int(y_true.sum()),
    )


# ----------------------------------------------------------------------
# Disagreement analysis
# ----------------------------------------------------------------------

@dataclass
class DisagreementResult:
    top_pct: float
    k: int
    lgbm_top_k_fraud_rate: float
    if_top_k_fraud_rate: float
    overlap_count: int
    overlap_pct: float


def disagreement_analysis(
    y_true: np.ndarray,
    lgbm_score: np.ndarray,
    if_score: np.ndarray,
    top_pct: float = 0.05,
) -> DisagreementResult:
    """Compare the top-k% highest-scoring transactions from each model.

    Two interview-grade signals come out of this:
      - fraud rate inside each top-k set (precision @ k)
      - overlap between the sets (does IF flag the same things as LGBM?)
    """
    n = len(y_true)
    k = int(n * top_pct)
    lgbm_top = np.argsort(-lgbm_score)[:k]
    if_top = np.argsort(-if_score)[:k]
    intersection = np.intersect1d(lgbm_top, if_top)
    return DisagreementResult(
        top_pct=top_pct,
        k=k,
        lgbm_top_k_fraud_rate=float(y_true[lgbm_top].mean()),
        if_top_k_fraud_rate=float(y_true[if_top].mean()),
        overlap_count=int(len(intersection)),
        overlap_pct=float(len(intersection) / k),
    )