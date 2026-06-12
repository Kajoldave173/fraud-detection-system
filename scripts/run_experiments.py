"""
Run the full Step 8 experiment matrix and log to MLflow on DagsHub.

22 runs total:
  - 3 baselines (LR, IF, LGBM at Step 6 config)
  - 12 LGBM hyperparameter sweep runs (num_leaves, learning_rate,
        scale_pos_weight, min_child_samples)
  - 5 LGBM feature-ablation runs (noUID, noAgg, noV, Vonly, top50)
  - 2 LGBM seed sensitivity runs

Each run logs:
  - tags (model_type, step, git_commit, git_dirty, dataset_version)
  - params (hyperparams + n_train/n_val/n_test/n_features/fraud_rate/seed)
  - metrics (train/val/test x pr_auc/roc_auc/log_loss, plus best_iteration)
  - artifacts (.joblib model + feature_columns.json)
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import lightgbm as lgb  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from modeling import (  # noqa: E402
    SEED,
    TARGET,
    detect_categoricals,
    feature_columns,
    iforest_anomaly_score,
    make_lgbm_frame,
    make_linear_frame,
    time_based_split,
    train_isolation_forest,
    train_lightgbm,
    train_logistic,
)
from tracking import (  # noqa: E402
    log_training_run,
    mlflow_session,
    select_features,
)


# --------------------------------------------------------------------------
# Paths and config
# --------------------------------------------------------------------------
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "model" / "artifacts" / "experiments"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SHAP_TOP50_PATH = PROJECT_ROOT / "model" / "artifacts" / "shap" / "top50_features.json"
EXPERIMENT_NAME = "fraud-detection"


# --------------------------------------------------------------------------
# Connection check
# --------------------------------------------------------------------------

def check_dagshub_connection():
    """Ping DagsHub MLflow before burning 45 min on a matrix that can't log."""
    print("Pinging DagsHub MLflow...", end=" ", flush=True)
    try:
        mlflow.search_experiments()
        print("OK")
    except Exception as e:
        print(f"FAILED: {str(e)[:200]}")
        sys.exit(1)


# --------------------------------------------------------------------------
# Experiment matrix definition
# --------------------------------------------------------------------------

@dataclass
class Experiment:
    model: str               # 'lr' | 'lgbm' | 'if'
    treatment: str           # 'stratified' | 'raw' | 'noUID' | 'noAgg' | 'noV' | 'Vonly' | 'top50'
    focus: str               # 'baseline' | 'leaves_32' | 'lr_001' | etc.
    feature_spec: str = "all"
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = SEED


def build_experiment_matrix(top50_features: list[str]) -> list[Experiment]:
    """Return the 22-run experiment matrix."""
    exps: list[Experiment] = []

    # --- 3 baselines -----------------------------------------------------
    exps.append(Experiment("lr",   "stratified", "baseline"))
    exps.append(Experiment("if",   "raw",        "contamination_034",
                           params={"contamination": 0.0348}))
    exps.append(Experiment("lgbm", "stratified", "baseline"))

    # --- 12 LGBM hyperparameter sweep -----------------------------------
    for nl in (32, 128, 256):
        exps.append(Experiment("lgbm", "stratified", f"leaves_{nl}",
                               params={"num_leaves": nl}))
    for lr_val, tag in [(0.01, "001"), (0.1, "01"), (0.2, "02")]:
        exps.append(Experiment("lgbm", "stratified", f"lr_{tag}",
                               params={"learning_rate": lr_val}))
    for spw in (1, 5, 10):
        exps.append(Experiment("lgbm", "stratified", f"spw_{spw}",
                               params={"scale_pos_weight": spw}))
    for mcs in (20, 100, 500):
        exps.append(Experiment("lgbm", "stratified", f"minchild_{mcs}",
                               params={"min_child_samples": mcs}))

    # --- 5 LGBM ablations -----------------------------------------------
    exps.append(Experiment("lgbm", "noUID", "baseline",
                           feature_spec="exclude:uid_identifiers,uid_aggregates"))
    exps.append(Experiment("lgbm", "noAgg", "baseline",
                           feature_spec="exclude:uid_aggregates"))
    exps.append(Experiment("lgbm", "noV", "baseline",
                           feature_spec="exclude:v_features"))
    exps.append(Experiment("lgbm", "Vonly", "baseline",
                           feature_spec="only:v_features"))
    exps.append(Experiment("lgbm", "top50", "baseline",
                           feature_spec="list:" + ",".join(top50_features)))

    # --- 2 seed sensitivity ---------------------------------------------
    for s, tag in [(123, "123"), (7, "7")]:
        exps.append(Experiment("lgbm", "stratified", f"seed_{tag}", seed=s))

    return exps


# --------------------------------------------------------------------------
# Run executors (one per model type)
# --------------------------------------------------------------------------

def run_lr(df: pd.DataFrame, split, feat_cols, cat_cols, exp: Experiment):
    """Train LR, return (model, scores, params)."""
    X = make_linear_frame(df, feat_cols, cat_cols)
    X_tr, X_val, X_te = X.loc[split.train], X.loc[split.val], X.loc[split.test]
    y_tr = df.loc[split.train, TARGET].values
    y_val = df.loc[split.val, TARGET].values
    y_te = df.loc[split.test, TARGET].values

    model = train_logistic(X_tr, y_tr, X_val, y_val)
    scores = {
        "train": model.predict_proba(X_tr)[:, 1],
        "val":   model.predict_proba(X_val)[:, 1],
        "test":  model.predict_proba(X_te)[:, 1],
    }
    labels = {"train": y_tr, "val": y_val, "test": y_te}

    # Pull params from the fitted estimator
    lr_step = model.named_steps.get("logreg") or model.named_steps.get("logisticregression") or model.steps[-1][1]
    params = {
        "C": getattr(lr_step, "C", "unknown"),
        "penalty": getattr(lr_step, "penalty", "unknown"),
        "class_weight": str(getattr(lr_step, "class_weight", "unknown")),
        "solver": getattr(lr_step, "solver", "unknown"),
        "max_iter": getattr(lr_step, "max_iter", "unknown"),
    }
    return model, scores, labels, params, {}


def run_if(df: pd.DataFrame, split, feat_cols, cat_cols, exp: Experiment):
    """Train IF, return (model, scores, params, extra_metrics)."""
    X = make_linear_frame(df, feat_cols, cat_cols)
    X_tr, X_val, X_te = X.loc[split.train], X.loc[split.val], X.loc[split.test]
    y_tr = df.loc[split.train, TARGET].values
    y_val = df.loc[split.val, TARGET].values
    y_te = df.loc[split.test, TARGET].values

    contamination = exp.params.get("contamination", 0.0348)
    model = train_isolation_forest(X_tr, contamination=contamination)
    scores = {
        "train": iforest_anomaly_score(model, X_tr),
        "val":   iforest_anomaly_score(model, X_val),
        "test":  iforest_anomaly_score(model, X_te),
    }
    labels = {"train": y_tr, "val": y_val, "test": y_te}

    if_step = model.named_steps.get("iforest") or model.steps[-1][1]
    params = {
        "n_estimators": getattr(if_step, "n_estimators", "unknown"),
        "contamination": contamination,
        "max_samples": str(getattr(if_step, "max_samples", "unknown")),
        "random_state": getattr(if_step, "random_state", "unknown"),
    }
    return model, scores, labels, params, {}


def run_lgbm(df: pd.DataFrame, split, feat_cols, cat_cols, exp: Experiment):
    """Train LGBM with the experiment's hyperparameter overrides."""
    X = make_lgbm_frame(df, feat_cols, cat_cols)
    X_tr, X_val, X_te = X.loc[split.train], X.loc[split.val], X.loc[split.test]
    y_tr = df.loc[split.train, TARGET].values
    y_val = df.loc[split.val, TARGET].values
    y_te = df.loc[split.test, TARGET].values

    # train_lightgbm uses Step 6 defaults. To apply per-run overrides we'd
    # need to either parameterize train_lightgbm or build the LGBMClassifier
    # here. We do the latter for the sweep — keeps modeling.py untouched.
    if not exp.params:
        model = train_lightgbm(X_tr, y_tr, X_val, y_val, cat_cols)
    else:
        # Step 6 defaults, overridden by exp.params
        defaults = dict(
            objective="binary",
            metric="average_precision",
            learning_rate=0.05,
            num_leaves=64,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=5,
            min_child_samples=100,
            scale_pos_weight=3,
            n_estimators=3000,
            random_state=exp.seed,
            n_jobs=-1,
            verbose=-1,
        )
        defaults.update(exp.params)
        model = lgb.LGBMClassifier(**defaults)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )

    scores = {
        "train": model.predict_proba(X_tr)[:, 1],
        "val":   model.predict_proba(X_val)[:, 1],
        "test":  model.predict_proba(X_te)[:, 1],
    }
    labels = {"train": y_tr, "val": y_val, "test": y_te}

    params = {
        "learning_rate": model.learning_rate,
        "num_leaves": model.num_leaves,
        "n_estimators": model.n_estimators,
        "scale_pos_weight": model.scale_pos_weight,
        "min_child_samples": model.min_child_samples,
        "feature_fraction": getattr(model, "feature_fraction", "default"),
        "random_state": model.random_state,
    }
    extra_metrics = {}
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None:
        extra_metrics["lgbm_best_iteration"] = float(best_iter)
    return model, scores, labels, params, extra_metrics


# --------------------------------------------------------------------------
# Single-experiment driver
# --------------------------------------------------------------------------

def run_experiment(df: pd.DataFrame, exp: Experiment) -> dict:
    """Run one experiment, log to MLflow, return summary dict."""
    t0 = time.time()
    feat_cols_all = feature_columns(df)
    feat_cols = select_features(feat_cols_all, exp.feature_spec)
    cat_cols = detect_categoricals(df, feat_cols)
    split = time_based_split(df)

    if exp.model == "lr":
        runner = run_lr
    elif exp.model == "if":
        runner = run_if
    elif exp.model == "lgbm":
        runner = run_lgbm
    else:
        raise ValueError(f"Unknown model type: {exp.model}")

    model, scores, labels, model_params, extra_metrics = runner(
        df, split, feat_cols, cat_cols, exp,
    )

    # Pipeline metadata params (added on top of model hyperparams)
    pipeline_params = {
        "n_features": len(feat_cols),
        "n_train": int(split.train.sum()),
        "n_val":   int(split.val.sum()),
        "n_test":  int(split.test.sum()),
        "fraud_rate_train": float(df.loc[split.train, TARGET].mean()),
        "fraud_rate_test":  float(df.loc[split.test,  TARGET].mean()),
        "feature_spec": exp.feature_spec,
        "split_method": "time_based_70_15_15",
        "seed": exp.seed,
    }
    all_params = {**pipeline_params, **model_params}

    # Persist artifact for MLflow to upload, then clean up
    from tracking import run_name as _run_name
    name = _run_name(exp.model, exp.treatment, exp.focus)
    artifact_path = ARTIFACT_DIR / f"{name}.joblib"
    joblib.dump(model, artifact_path)

    try:
        run_id = log_training_run(
            model=model,
            model_type=exp.model,
            data_treatment=exp.treatment,
            hp_focus=exp.focus,
            scores=scores,
            labels=labels,
            params=all_params,
            feature_columns_list=feat_cols,
            artifact_path=artifact_path,
            dataset_path=FEATURES_PATH,
            extra_tags={"feature_spec": exp.feature_spec},
            extra_metrics=extra_metrics,
        )
    finally:
        artifact_path.unlink(missing_ok=True)

    test_pr_auc = float(np.array(  # quick re-eval for the summary print
        [labels["test"], scores["test"]]
    )[0].size > 0)  # placeholder; the real value is in MLflow
    elapsed = time.time() - t0

    return {
        "name": name,
        "run_id": run_id,
        "elapsed_s": elapsed,
        "n_features": len(feat_cols),
        "extra": extra_metrics,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    print("=" * 78)
    print("FRAUD DETECTION — EXPERIMENT MATRIX")
    print("=" * 78)
    print()

    check_dagshub_connection()
    print()

    print(f"[setup] loading features from {FEATURES_PATH.name}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"        shape: {df.shape}")

    if not SHAP_TOP50_PATH.exists():
        raise FileNotFoundError(
            f"{SHAP_TOP50_PATH} not found — run Step 7 SHAP first."
        )
    top50 = json.loads(SHAP_TOP50_PATH.read_text())["features"]
    print(f"        loaded top-50 SHAP features: {len(top50)}")

    experiments = build_experiment_matrix(top50)
    print(f"        experiment matrix: {len(experiments)} runs")
    print()

    with mlflow_session(experiment=EXPERIMENT_NAME):
        summaries = []
        for i, exp in enumerate(experiments, start=1):
            from tracking import run_name as _run_name
            name = _run_name(exp.model, exp.treatment, exp.focus)
            print(f"[{i:02d}/{len(experiments)}] {name} ...", end=" ", flush=True)
            try:
                summary = run_experiment(df, exp)
                summaries.append(summary)
                extra = summary["extra"]
                best_iter = extra.get("lgbm_best_iteration")
                extra_str = f" [best_iter={int(best_iter)}]" if best_iter is not None else ""
                print(f"OK ({summary['elapsed_s']:.1f}s, n_feat={summary['n_features']}){extra_str}")
            except Exception as e:
                print(f"FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                # MLflow run already auto-failed via the context manager
                continue

    print()
    print("=" * 78)
    print(f"All experiments complete in {(time.time() - t_start) / 60:.1f} min")
    print(f"View runs at: https://dagshub.com/kajoldave8/fraud-detection-system.mlflow")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Successful runs: {len(summaries)}/{len(experiments)}")


if __name__ == "__main__":
    main()