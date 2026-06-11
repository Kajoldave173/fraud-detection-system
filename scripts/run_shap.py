"""
Runner for SHAP explainability on the LightGBM champion.

Loads the trained model, reconstructs the time-based test split,
runs TreeExplainer on a stratified 10K sample, and saves:
  - SHAP summary plots (dot + bar)
  - Dependence plots for top 10 features
  - Per-prediction reason codes for model-flagged fraud
  - SHAP-based leakage audit table
to model/artifacts/shap/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from modeling import (  # noqa: E402
    TARGET,
    detect_categoricals,
    feature_columns,
    make_lgbm_frame,
    time_based_split,
)
from explainability import (  # noqa: E402
    compute_reason_codes,
    compute_shap_values,
    save_dependence_plots,
    save_summary_plots,
    shap_leakage_audit,
)

# --------------------------------------------------------------------------
# Paths and config
# --------------------------------------------------------------------------
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH = PROJECT_ROOT / "model" / "artifacts" / "lightgbm.joblib"
OUTPUT_DIR = PROJECT_ROOT / "model" / "artifacts" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 10_000
TOP_N_DEPENDENCE = 10
TOP_N_AUDIT = 20
REASON_PROBA_THRESHOLD = 0.5
SEED = 42


def main() -> None:
    t_start = time.time()

    # --- Load data + reconstruct test split -------------------------------
    print(f"[1/6] Loading features from {FEATURES_PATH.name}")
    df = pd.read_parquet(FEATURES_PATH)
    split = time_based_split(df)

    feat_cols = feature_columns(df)
    cat_cols = detect_categoricals(df, feat_cols)
    X_lgbm = make_lgbm_frame(df, feat_cols, cat_cols)

    X_test = X_lgbm.loc[split.test].reset_index(drop=True)
    y_test = df.loc[split.test, TARGET].reset_index(drop=True).values
    print(f"      test set: {X_test.shape}, fraud rate: {y_test.mean():.4f}")

    # --- Load model -------------------------------------------------------
    print(f"[2/6] Loading model from {MODEL_PATH.name}")
    model = joblib.load(MODEL_PATH)

    # --- Compute SHAP values on stratified sample -------------------------
    print(f"[3/6] Computing SHAP values on stratified sample of {SAMPLE_SIZE:,}")
    t0 = time.time()
    sample_idx, shap_values, expected_value = compute_shap_values(
        model,
        X_test,
        sample_size=SAMPLE_SIZE,
        stratify_target=pd.Series(y_test),
        seed=SEED,
    )
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_test[sample_idx]
    print(f"      done in {time.time() - t0:.1f}s")
    print(f"      shap_values: {shap_values.shape}, expected_value: {expected_value:.4f}")
    print(f"      sample fraud rate: {y_sample.mean():.4f} "
          f"({int(y_sample.sum())} positives)")

    # Cache values for downstream reuse
    np.savez_compressed(
        OUTPUT_DIR / "shap_values.npz",
        shap_values=shap_values,
        sample_idx=sample_idx,
        y_sample=y_sample,
        expected_value=expected_value,
        feature_names=np.array(X_sample.columns.tolist()),
    )

    # --- Summary plots ----------------------------------------------------
    print("[4/6] Saving summary plots")
    summary_paths = save_summary_plots(shap_values, X_sample, OUTPUT_DIR, max_display=25)
    for k, p in summary_paths.items():
        print(f"      {k}: {p.name}")

    # --- Dependence plots -------------------------------------------------
    print(f"[5/6] Saving top-{TOP_N_DEPENDENCE} dependence plots")
    dep_paths = save_dependence_plots(
        shap_values, X_sample, OUTPUT_DIR, top_n=TOP_N_DEPENDENCE,
    )
    print(f"      saved {len(dep_paths)} plots")

    # --- Reason codes + leakage audit -------------------------------------
    print("[6/6] Reason codes + leakage audit")
    proba = model.predict_proba(X_sample)[:, 1]

    rc = compute_reason_codes(
        X_sample, shap_values, expected_value, proba,
        top_k=5, filter_threshold=REASON_PROBA_THRESHOLD,
    )
    if len(rc) == 0:
        print(f"      no rows above proba >= {REASON_PROBA_THRESHOLD}, "
              f"falling back to top-200 by proba")
        order = np.argsort(-proba)[:200]
        rc = compute_reason_codes(
            X_sample.iloc[order].reset_index(drop=True),
            shap_values[order],
            expected_value,
            proba[order],
            top_k=5,
            filter_threshold=None,
        )
    rc_path = OUTPUT_DIR / "reason_codes.csv"
    rc.to_csv(rc_path, index=False)
    print(f"      reason codes: {len(rc):,} rows -> {rc_path.name}")

    audit = shap_leakage_audit(shap_values, X_sample, y_sample, top_n=TOP_N_AUDIT)
    audit_path = OUTPUT_DIR / "leakage_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"      leakage audit: {len(audit)} features -> {audit_path.name}")

    flagged = audit[audit["flag"] == "review"]
    print()
    print("=" * 78)
    print("LEAKAGE AUDIT (top 20 features by mean|SHAP|)")
    print("=" * 78)
    with pd.option_context("display.max_rows", None, "display.width", 140,
                           "display.float_format", "{:.4f}".format):
        print(audit.to_string(index=False))
    print()
    if len(flagged) == 0:
        print("[OK] No features flagged for leakage review.")
    else:
        print(f"[REVIEW] {len(flagged)} feature(s) flagged:")
        print(flagged["feature"].tolist())
    print()
    print(f"All outputs in {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()