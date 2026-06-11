"""
SHAP explainability for the LightGBM fraud detection champion.

Provides TreeExplainer-based feature attribution, dependence plots,
per-prediction reason codes, and a SHAP-based leakage audit.

Designed for use with sklearn-wrapped LightGBM (lgb.LGBMClassifier).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Core SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: Optional[int] = None,
    stratify_target: Optional[pd.Series] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run TreeExplainer on (optionally stratified-sampled) rows of X.

    Returns
    -------
    sample_idx : np.ndarray
        Positional indices into the original X used for the sample.
    shap_values : np.ndarray, shape (n_sample, n_features)
        SHAP values for the positive (fraud) class.
    expected_value : float
        Base value E[f(x)] for the positive class.
    """
    rng = np.random.default_rng(seed)

    if sample_size is None or sample_size >= len(X):
        sample_idx = np.arange(len(X))
    elif stratify_target is not None:
        y = np.asarray(stratify_target)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        pos_take = min(len(pos), max(1, int(round(sample_size * len(pos) / len(y)))))
        neg_take = max(0, sample_size - pos_take)
        neg_take = min(neg_take, len(neg))
        sel_pos = rng.choice(pos, size=pos_take, replace=False)
        sel_neg = rng.choice(neg, size=neg_take, replace=False)
        sample_idx = np.sort(np.concatenate([sel_pos, sel_neg]))
    else:
        sample_idx = np.sort(rng.choice(len(X), size=sample_size, replace=False))

    X_sample = X.iloc[sample_idx]

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample)

    # Normalize to positive-class ndarray of shape (n_sample, n_features)
    if isinstance(raw, list):
        # Older SHAP returned [neg_class, pos_class]
        shap_arr = raw[1]
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        # Newer SHAP can return (n, p, 2) for binary classifiers
        shap_arr = raw[:, :, 1]
    else:
        shap_arr = raw

    # Expected value can be scalar or 2-element list/array
    ev = explainer.expected_value
    ev_arr = np.atleast_1d(np.asarray(ev))
    expected_value = float(ev_arr[1] if ev_arr.size > 1 else ev_arr[0])

    return sample_idx, np.asarray(shap_arr), expected_value


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_summary_plots(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_dir: Path,
    max_display: int = 25,
) -> dict:
    """Save dot and bar summary plots. Returns a dict of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    plt.figure(figsize=(10, 12))
    shap.summary_plot(
        shap_values, X_sample, max_display=max_display,
        plot_type="dot", show=False,
    )
    plt.title(f"SHAP Summary (dot) — top {max_display} features", fontsize=12, pad=12)
    plt.tight_layout()
    paths["dot"] = output_dir / "summary_dot.png"
    plt.savefig(paths["dot"], dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 12))
    shap.summary_plot(
        shap_values, X_sample, max_display=max_display,
        plot_type="bar", show=False,
    )
    plt.title(f"SHAP Mean |Value| — top {max_display} features", fontsize=12, pad=12)
    plt.tight_layout()
    paths["bar"] = output_dir / "summary_bar.png"
    plt.savefig(paths["bar"], dpi=150, bbox_inches="tight")
    plt.close()

    return paths


def save_dependence_plots(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> list:
    """Dependence plots for the top-N features by mean(|SHAP|)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # shap.dependence_plot sorts values internally; pandas category dtype
    # with NaN causes float-vs-str comparison errors. Cast categoricals
    # to integer codes for plotting (LightGBM treats them this way anyway).
    X_plot = X_sample.copy()
    for col in X_plot.columns:
        if isinstance(X_plot[col].dtype, pd.CategoricalDtype):
            codes = X_plot[col].cat.codes.astype("float32")
            X_plot[col] = codes.where(codes >= 0, np.nan)

    importance = np.abs(shap_values).mean(axis=0)
    feat_order = np.argsort(importance)[::-1][:top_n]
    feature_names = X_sample.columns.tolist()

    saved = []
    for rank, idx in enumerate(feat_order, start=1):
        feat = feature_names[idx]
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in feat)
        out_path = output_dir / f"dep_{rank:02d}_{safe}.png"

        plt.figure(figsize=(8, 6))
        try:
            shap.dependence_plot(
                idx, shap_values, X_plot,
                interaction_index="auto", show=False,
            )
        except Exception as e:
            plt.close()
            print(f"  [skip] dependence plot for {feat}: {type(e).__name__}: {e}")
            continue
        plt.title(f"#{rank}: {feat}", fontsize=11, pad=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out_path)

    return saved


# ---------------------------------------------------------------------------
# Per-prediction reason codes
# ---------------------------------------------------------------------------

def compute_reason_codes(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    expected_value: float,
    predicted_proba: np.ndarray,
    top_k: int = 5,
    filter_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Per-row top-k feature contributors, sorted by signed SHAP value descending.

    If filter_threshold is given, only rows with predicted_proba >= threshold are kept.
    """
    predicted_proba = np.asarray(predicted_proba)
    mask = (predicted_proba >= filter_threshold) if filter_threshold is not None \
        else np.ones(len(X_sample), dtype=bool)

    if mask.sum() == 0:
        return pd.DataFrame()

    X_f = X_sample.iloc[mask]
    shap_f = shap_values[mask]
    proba_f = predicted_proba[mask]
    feature_names = np.array(X_sample.columns.tolist())
    n_rows, n_feat = shap_f.shape
    k = min(top_k, n_feat)

    # Top-k by |SHAP| (unsorted partition), then sort by signed SHAP desc within each row
    abs_shap = np.abs(shap_f)
    topk_unsorted = np.argpartition(-abs_shap, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n_rows)[:, None]
    topk_shap_unsorted = shap_f[rows, topk_unsorted]
    order = np.argsort(-topk_shap_unsorted, axis=1)
    topk_idx = np.take_along_axis(topk_unsorted, order, axis=1)

    # Pull values as object array so categoricals come out as labels
    X_values = X_f.values

    cols = {
        "predicted_proba": proba_f,
        "expected_value": np.full(n_rows, expected_value),
    }
    flat_rows = rows.ravel()
    for rank in range(k):
        feat_idx = topk_idx[:, rank]
        cols[f"reason_{rank + 1}_feature"] = feature_names[feat_idx]
        cols[f"reason_{rank + 1}_shap"] = shap_f[flat_rows, feat_idx]
        cols[f"reason_{rank + 1}_value"] = X_values[flat_rows, feat_idx]

    return pd.DataFrame(cols, index=X_f.index)


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def shap_leakage_audit(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    y_sample: np.ndarray,
    top_n: int = 20,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Formal SHAP-based leakage check on the top-N features by mean(|SHAP|).

    Columns:
      mean_abs_shap    - global importance
      shap_label_corr  - Pearson corr between feature SHAP and isFraud label
      monotonicity     - Spearman rank corr between binned-feature-value order
                          and empirical fraud rate. |rho| ~ 1.0 with high shap_label_corr
                          indicates a suspiciously clean relationship.
      n_unique         - distinct values (sanity for sequential-ID style leaks)
      flag             - "review" if |monotonicity| > 0.95 AND |shap_label_corr| > 0.5
    """
    importance = np.abs(shap_values).mean(axis=0)
    feat_order = np.argsort(importance)[::-1][:top_n]
    feature_names = X_sample.columns.tolist()
    y = np.asarray(y_sample).astype(float)

    rows = []
    for idx in feat_order:
        feat = feature_names[idx]
        col_shap = shap_values[:, idx]
        col_val = X_sample.iloc[:, idx]

        if np.std(col_shap) > 0:
            shap_corr = float(np.corrcoef(col_shap, y)[0, 1])
        else:
            shap_corr = 0.0

        try:
            if pd.api.types.is_numeric_dtype(col_val):
                bins = pd.qcut(col_val, q=n_bins, duplicates="drop")
            else:
                bins = col_val.astype("object")
            tmp = pd.DataFrame({"bin": bins, "y": y})
            grp = tmp.groupby("bin", observed=True)["y"].mean()
            if len(grp) >= 3:
                rho, _ = spearmanr(np.arange(len(grp)), grp.values)
                monotonicity = float(rho) if not np.isnan(rho) else 0.0
            else:
                monotonicity = 0.0
        except Exception:
            monotonicity = 0.0

        n_unique = int(col_val.nunique(dropna=False))
        flag = "review" if abs(monotonicity) > 0.95 and abs(shap_corr) > 0.5 else ""

        rows.append({
            "feature": feat,
            "mean_abs_shap": float(importance[idx]),
            "shap_label_corr": shap_corr,
            "monotonicity": monotonicity,
            "n_unique": n_unique,
            "flag": flag,
        })

    return pd.DataFrame(rows)