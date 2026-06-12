"""Step 5-6: Isotonic calibration and cost-based threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    """Fit isotonic calibrator on val fold raw probabilities."""
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    return iso


def calibrate(iso: IsotonicRegression, y_prob: np.ndarray) -> np.ndarray:
    """Apply fitted calibrator to raw probabilities."""
    return iso.transform(y_prob)


def find_cost_threshold(
    y_true: np.ndarray,
    y_prob_cal: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float = 5.0,
    n_thresholds: int = 500,
) -> dict:
    """Pick threshold minimizing instance-dependent cost on val.

    Cost model:
      - FN (missed fraud): lose the transaction amount
      - FP (false block):  fixed cost per blocked legit txn
        (customer friction, support call, churn risk)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_t, best_cost = 0.5, np.inf
    details = []

    for t in thresholds:
        pred = (y_prob_cal >= t).astype(int)
        fn_mask = (y_true == 1) & (pred == 0)
        fp_mask = (y_true == 0) & (pred == 1)

        fn_total = float(amounts[fn_mask].sum())   # dollars lost
        fp_total = float(fp_mask.sum() * fp_cost)   # friction cost
        total = fn_total + fp_total

        tp = int(((y_true == 1) & (pred == 1)).sum())
        fn = int(fn_mask.sum())
        fp = int(fp_mask.sum())
        tn = int(((y_true == 0) & (pred == 0)).sum())

        details.append({
            "threshold": float(t),
            "total_cost": total,
            "fn_cost": fn_total,
            "fp_cost": fp_total,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        })

        if total < best_cost:
            best_cost = total
            best_t = float(t)

    return {
        "threshold": best_t,
        "total_cost": best_cost,
        "details": details,
    }