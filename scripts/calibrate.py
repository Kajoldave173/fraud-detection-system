"""Step 5-6: Calibrate champion model and select cost-based threshold."""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calibration import calibrate, find_cost_threshold, fit_isotonic
from modeling import (
    TARGET,
    detect_categoricals,
    feature_columns,
    make_lgbm_frame,
    time_based_split,
)

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "model" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

FP_COST = 5.0  # dollars per false block


def main():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)
    feat_cols = feature_columns(df)
    cat_cols = detect_categoricals(df, feat_cols)
    split = time_based_split(df)

    X = make_lgbm_frame(df, feat_cols, cat_cols)
    X_tr = X.loc[split.train]
    X_val = X.loc[split.val]
    X_te = X.loc[split.test]
    y_tr = df.loc[split.train, TARGET].values
    y_val = df.loc[split.val, TARGET].values
    y_te = df.loc[split.test, TARGET].values
    amt_val = df.loc[split.val, "TransactionAmt"].values
    amt_te = df.loc[split.test, "TransactionAmt"].values

    # --- retrain champion (leaves_256) ---
    print("Training champion (leaves_256)...")
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="average_precision",
        learning_rate=0.05,
        num_leaves=256,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_child_samples=100,
        scale_pos_weight=3,
        n_estimators=3000,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    print(f"  best_iteration: {model.best_iteration_}")

    raw_val = model.predict_proba(X_val)[:, 1]
    raw_te = model.predict_proba(X_te)[:, 1]

    # --- Step 5: isotonic calibration ---
    print("Fitting isotonic calibration on val...")
    iso = fit_isotonic(y_val, raw_val)
    cal_val = calibrate(iso, raw_val)
    cal_te = calibrate(iso, raw_te)

    # Quick calibration check
    print(f"  Raw  val — mean pred: {raw_val.mean():.4f}, actual fraud rate: {y_val.mean():.4f}")
    print(f"  Cal  val — mean pred: {cal_val.mean():.4f}, actual fraud rate: {y_val.mean():.4f}")
    print(f"  Cal test — mean pred: {cal_te.mean():.4f}, actual fraud rate: {y_te.mean():.4f}")

    # --- Step 6: cost-based threshold ---
    print(f"Finding cost-optimal threshold (fp_cost=${FP_COST})...")
    result = find_cost_threshold(y_val, cal_val, amt_val, fp_cost=FP_COST)
    t = result["threshold"]
    print(f"  Optimal threshold: {t:.4f}")
    print(f"  Val total cost: ${result['total_cost']:,.0f}")

    # Apply to test
    te_pred = (cal_te >= t).astype(int)
    tp = int(((y_te == 1) & (te_pred == 1)).sum())
    fn = int(((y_te == 1) & (te_pred == 0)).sum())
    fp = int(((y_te == 0) & (te_pred == 1)).sum())
    tn = int(((y_te == 0) & (te_pred == 0)).sum())
    fn_cost = float(amt_te[(y_te == 1) & (te_pred == 0)].sum())
    fp_cost_total = float(fp * FP_COST)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print()
    print("Test set results at optimal threshold:")
    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  FN cost (missed fraud): ${fn_cost:,.0f}")
    print(f"  FP cost (false blocks): ${fp_cost_total:,.0f}")
    print(f"  Total cost:             ${fn_cost + fp_cost_total:,.0f}")

    # --- save artifacts ---
    joblib.dump(model, ARTIFACT_DIR / "champion_lgbm.joblib")
    joblib.dump(iso, ARTIFACT_DIR / "isotonic_calibrator.joblib")

    threshold_config = {
        "threshold": t,
        "fp_cost_per_block": FP_COST,
        "val_total_cost": result["total_cost"],
        "champion": "lgbm_leaves_256",
        "calibration": "isotonic",
    }
    (ARTIFACT_DIR / "threshold_config.json").write_text(
        json.dumps(threshold_config, indent=2)
    )

    print()
    print(f"Saved: champion_lgbm.joblib, isotonic_calibrator.joblib, threshold_config.json")
    print("Done.")


if __name__ == "__main__":
    main()