"""
MLflow tracking layer for the fraud detection training pipeline.

Wraps the pure training functions in src/modeling.py with MLflow run context.
The training functions themselves remain untouched.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import log_loss


# ---------------------------------------------------------------------------
# Naming convention
# ---------------------------------------------------------------------------

def run_name(model_type: str, data_treatment: str, hp_focus: str) -> str:
    """Build a run name following the convention model-treatment-focus."""
    for part, label in [
        (model_type, "model_type"),
        (data_treatment, "data_treatment"),
        (hp_focus, "hp_focus"),
    ]:
        if not part or " " in part:
            raise ValueError(f"{label!r} must be a non-empty whitespace-free string")
    return f"{model_type}-{data_treatment}-{hp_focus}"


# ---------------------------------------------------------------------------
# Feature group identification (for ablations)
# ---------------------------------------------------------------------------

# Regex patterns based on IEEE-CIS naming conventions + our engineered features.
_GROUP_PATTERNS = {
    "v_features":      re.compile(r"^V\d+$"),
    "c_features":      re.compile(r"^C\d+$"),
    "d_features":      re.compile(r"^D\d+$"),
    "m_features":      re.compile(r"^M\d+$"),
    "id_features":     re.compile(r"^id_\d+$"),
    "card_features":   re.compile(r"^card\d+$"),
    "addr_features":   re.compile(r"^addr\d+$"),
    "email_features":  re.compile(r"^[PR]_emaildomain$"),
    "dist_features":   re.compile(r"^dist\d+$"),
    "uid_identifiers": re.compile(r"^(UID|UID_primary|UID_fallback|D1n|is_new_entity)$"),
    "uid_aggregates":  re.compile(r"^uid_.*$"),
}


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """
    Group column names by IEEE-CIS convention + engineered feature regex.

    Columns not matching any pattern are placed in 'misc'.
    """
    groups: dict[str, list[str]] = {name: [] for name in _GROUP_PATTERNS}
    groups["misc"] = []
    for col in columns:
        matched = False
        for name, pat in _GROUP_PATTERNS.items():
            if pat.match(col):
                groups[name].append(col)
                matched = True
                break
        if not matched:
            groups["misc"].append(col)
    return groups


def select_features(columns: list[str], spec: str) -> list[str]:
    """
    Select a subset of feature columns based on a spec string.

    Spec formats:
      'all'                  -> all columns
      'exclude:group1,group2' -> all except listed groups
      'only:group1,group2'    -> only listed groups
      'list:col1,col2,...'    -> explicit column allow-list (e.g. SHAP top-50)
    """
    if spec == "all":
        return list(columns)

    groups = feature_groups(columns)

    if spec.startswith("exclude:"):
        excluded_groups = spec[len("exclude:"):].split(",")
        excluded_cols = set()
        for g in excluded_groups:
            if g not in groups:
                raise ValueError(f"Unknown feature group: {g!r}. Known: {list(groups)}")
            excluded_cols.update(groups[g])
        return [c for c in columns if c not in excluded_cols]

    if spec.startswith("only:"):
        included_groups = spec[len("only:"):].split(",")
        included_cols: list[str] = []
        for g in included_groups:
            if g not in groups:
                raise ValueError(f"Unknown feature group: {g!r}. Known: {list(groups)}")
            included_cols.extend(groups[g])
        # Preserve original column order
        included_set = set(included_cols)
        return [c for c in columns if c in included_set]

    if spec.startswith("list:"):
        allow_list = spec[len("list:"):].split(",")
        allow_set = set(allow_list)
        return [c for c in columns if c in allow_set]

    raise ValueError(f"Unrecognized feature spec: {spec!r}")


# ---------------------------------------------------------------------------
# Environment + session setup
# ---------------------------------------------------------------------------

def _ensure_credentials() -> None:
    load_dotenv()
    missing = [
        k for k in ("MLFLOW_TRACKING_URI",
                    "MLFLOW_TRACKING_USERNAME",
                    "MLFLOW_TRACKING_PASSWORD")
        if not os.environ.get(k)
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {missing}. "
            f"Check your .env file at the project root."
        )


@contextmanager
def mlflow_session(experiment: str = "fraud-detection"):
    """Set up MLflow client with experiment selected."""
    _ensure_credentials()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment)
    yield


# ---------------------------------------------------------------------------
# Environment metadata capture
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def _dvc_data_version(parquet_path: Path) -> str:
    dvc_file = Path(str(parquet_path) + ".dvc")
    if not dvc_file.exists():
        return "untracked"
    try:
        import yaml
        with open(dvc_file) as f:
            data = yaml.safe_load(f)
        return data["outs"][0]["md5"]
    except Exception:
        return "unreadable"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute log_loss; return NaN if scores aren't valid probabilities."""
    try:
        s = np.asarray(y_score, dtype=float)
        if np.any(np.isnan(s)) or np.any(s < 0) or np.any(s > 1):
            return float("nan")
        eps = 1e-15
        s = np.clip(s, eps, 1 - eps)
        return float(log_loss(y_true, s))
    except Exception:
        return float("nan")


def compute_split_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    split: str,
) -> dict:
    """Return the standard 3 metrics for a split, keyed `{split}_{metric}`."""
    from modeling import evaluate  # local import; sys.path set by runner
    if len(y_true) == 0:
        return {
            f"{split}_pr_auc": float("nan"),
            f"{split}_roc_auc": float("nan"),
            f"{split}_log_loss": float("nan"),
        }
    er = evaluate(y_true, y_score, name=split)
    return {
        f"{split}_pr_auc": er.pr_auc,
        f"{split}_roc_auc": er.roc_auc,
        f"{split}_log_loss": _safe_log_loss(y_true, y_score),
    }


# ---------------------------------------------------------------------------
# Core logging function
# ---------------------------------------------------------------------------

def log_training_run(
    model,
    model_type: str,
    data_treatment: str,
    hp_focus: str,
    scores: dict,
    labels: dict,
    params: dict,
    feature_columns_list: list,
    artifact_path: Path,
    dataset_path: Path,
    extra_tags: Optional[dict] = None,
    extra_metrics: Optional[dict] = None,
) -> str:
    """
    Log a single training run to MLflow under the current experiment.

    Returns the MLflow run UUID.
    """
    name = run_name(model_type, data_treatment, hp_focus)

    with mlflow.start_run(run_name=name) as run:
        # Tags
        tags = {
            "model_type": model_type,
            "step": "8",
            "git_commit": _git_commit(),
            "git_dirty": str(_git_dirty()).lower(),
            "dataset_version": _dvc_data_version(dataset_path),
        }
        if extra_tags:
            tags.update(extra_tags)
        mlflow.set_tags(tags)

        # Params (MLflow caps each at 500 chars)
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

        # Metrics — drop NaN since MLflow rejects them
        metrics: dict = {}
        for split in ("train", "val", "test"):
            if split in scores and split in labels:
                metrics.update(compute_split_metrics(labels[split], scores[split], split))
        if extra_metrics:
            metrics.update(extra_metrics)
        clean = {k: v for k, v in metrics.items()
                 if not (isinstance(v, float) and np.isnan(v))}
        mlflow.log_metrics(clean)
        dropped = sorted(set(metrics) - set(clean))
        if dropped:
            mlflow.set_tag("nan_metrics_skipped", ",".join(dropped))

        # Artifacts
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
        else:
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

        # Feature schema for inference-time parity
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(
                {"feature_columns": feature_columns_list,
                 "n_features": len(feature_columns_list)},
                f, indent=2,
            )
            feat_tmp = f.name
        try:
            mlflow.log_artifact(feat_tmp, artifact_path="schema")
        finally:
            Path(feat_tmp).unlink(missing_ok=True)

        return run.info.run_id