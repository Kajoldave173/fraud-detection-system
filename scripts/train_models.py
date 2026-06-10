"""Step 6: Train LR + LGBM + IF on engineered Parquet features.

Run from repo root:
    python scripts/train_models.py

Outputs (model/artifacts/):
  - logistic_regression.joblib
  - lightgbm.joblib
  - isolation_forest.joblib
  - test_scores.npz             (y_true + per-model scores for later analysis)
  - training_metadata.json      (split sizes, params, test metrics)
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling import (
    TARGET, TIME_COL, LEAKY_COLS,
    time_based_split,
    feature_columns, detect_categoricals,
    make_lgbm_frame, make_linear_frame,
    train_logistic, train_lightgbm, train_isolation_forest,
    iforest_anomaly_score,
    evaluate, disagreement_analysis,
)

PARQUET_PATH = Path("data/processed/features.parquet")
ARTIFACT_DIR = Path("model/artifacts")


def main() -> None:
    # ----- Load -----
    print("[1] Loading Parquet")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  shape: {df.shape}  fraud rate: {df[TARGET].mean():.4f}")

    # ----- Split -----
    print("\n[2] Time-based 70/15/15 split on TransactionDT")
    split = time_based_split(df)
    for name, mask in [("train", split.train), ("val", split.val), ("test", split.test)]:
        n_fraud = int(df.loc[mask, TARGET].sum())
        print(f"  {name:5s}: {mask.sum():>7,}  fraud {n_fraud:>5,}  "
              f"({df.loc[mask, TARGET].mean():.4f})")

    # ----- Features -----
    print("\n[3] Feature columns")
    feat_cols = feature_columns(df)
    cat_cols = detect_categoricals(df, feat_cols)
    print(f"  total: {len(feat_cols)}  "
          f"(categorical {len(cat_cols)}, numeric {len(feat_cols) - len(cat_cols)})")

    X_lgbm = make_lgbm_frame(df, feat_cols, cat_cols)
    X_lin = make_linear_frame(df, feat_cols, cat_cols)
    y_all = df[TARGET].values

    Xl_tr, Xl_va, Xl_te = X_lgbm.loc[split.train], X_lgbm.loc[split.val], X_lgbm.loc[split.test]
    Xn_tr, Xn_va, Xn_te = X_lin.loc[split.train], X_lin.loc[split.val], X_lin.loc[split.test]
    y_tr, y_va, y_te = y_all[split.train], y_all[split.val], y_all[split.test]

    # ----- LR -----
    print("\n[4a] Logistic Regression (linear baseline)")
    lr = train_logistic(Xn_tr, y_tr, Xn_va, y_va)
    lr_score = lr.predict_proba(Xn_te)[:, 1]
    lr_eval = evaluate(y_te, lr_score, "logistic_regression")
    print(f"  TEST PR-AUC {lr_eval.pr_auc:.4f}  ROC-AUC {lr_eval.roc_auc:.4f}")

    # ----- LGBM -----
    print("\n[4b] LightGBM (production champion)")
    lgbm = train_lightgbm(Xl_tr, y_tr, Xl_va, y_va, cat_cols)
    lgbm_score = lgbm.predict_proba(Xl_te)[:, 1]
    lgbm_eval = evaluate(y_te, lgbm_score, "lightgbm")
    print(f"  TEST PR-AUC {lgbm_eval.pr_auc:.4f}  ROC-AUC {lgbm_eval.roc_auc:.4f}")

    # ----- IF -----
    print("\n[4c] Isolation Forest (unsupervised anomaly)")
    iforest = train_isolation_forest(Xn_tr, contamination=float(y_tr.mean()))
    if_score = iforest_anomaly_score(iforest, Xn_te)
    if_eval = evaluate(y_te, if_score, "isolation_forest")
    print(f"  TEST PR-AUC {if_eval.pr_auc:.4f}  ROC-AUC {if_eval.roc_auc:.4f}")

    # ----- Lift summary -----
    print("\n[5] Lift summary (test PR-AUC)")
    base = lr_eval.pr_auc
    print(f"  LR   {lr_eval.pr_auc:.4f}   (baseline 1.0x)")
    print(f"  LGBM {lgbm_eval.pr_auc:.4f}   ({lgbm_eval.pr_auc / base:.1f}x baseline)")
    print(f"  IF   {if_eval.pr_auc:.4f}   ({if_eval.pr_auc / base:.1f}x baseline)")

    # ----- Disagreement -----
    print("\n[6] Disagreement at top 5% by score")
    dis = disagreement_analysis(y_te, lgbm_score, if_score, top_pct=0.05)
    print(f"  k = {dis.k:,}  ({dis.top_pct:.0%} of test)")
    print(f"  LGBM top-k fraud rate: {dis.lgbm_top_k_fraud_rate:.2%}")
    print(f"  IF   top-k fraud rate: {dis.if_top_k_fraud_rate:.2%}")
    print(f"  overlap: {dis.overlap_count:,} ({dis.overlap_pct:.2%})")

    # ----- Save -----
    print("\n[7] Saving artifacts")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, ARTIFACT_DIR / "logistic_regression.joblib")
    joblib.dump(lgbm, ARTIFACT_DIR / "lightgbm.joblib")
    joblib.dump(iforest, ARTIFACT_DIR / "isolation_forest.joblib")
    np.savez(
        ARTIFACT_DIR / "test_scores.npz",
        y_true=y_te, lr=lr_score, lgbm=lgbm_score, isolation_forest=if_score,
    )

    metadata = {
        "parquet": str(PARQUET_PATH),
        "shape": list(df.shape),
        "split": {
            "train_n": int(split.train.sum()),
            "val_n": int(split.val.sum()),
            "test_n": int(split.test.sum()),
            "train_max_dt": float(df.loc[split.train, TIME_COL].max()),
            "val_max_dt": float(df.loc[split.val, TIME_COL].max()),
            "test_max_dt": float(df.loc[split.test, TIME_COL].max()),
        },
        "features": {
            "total": len(feat_cols),
            "categorical_count": len(cat_cols),
            "dropped_leaky": LEAKY_COLS,
        },
        "results": {
            "logistic_regression": asdict(lr_eval),
            "lightgbm": asdict(lgbm_eval),
            "isolation_forest": asdict(if_eval),
        },
        "lgbm_best_iteration": int(lgbm.best_iteration_),
        "disagreement_top_5pct": asdict(dis),
    }
    with open(ARTIFACT_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[done] -> {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()