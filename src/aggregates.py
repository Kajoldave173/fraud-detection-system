"""Point-in-time UID aggregations.

CRITICAL: Every aggregate uses .shift(1) before rolling, so a transaction
NEVER sees itself or future transactions when its features are computed.
"""

import numpy as np
import pandas as pd


def build_uid_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lagged expanding aggregates per UID on TransactionAmt."""
    df = df.sort_values(["UID", "TransactionDT"]).reset_index(drop=True)

    grouped = df.groupby("UID", sort=False)["TransactionAmt"]
    shifted = grouped.shift(1)
    shifted_by_uid = shifted.groupby(df["UID"], sort=False)

    new_cols = pd.DataFrame({
        "uid_txn_count_prior": shifted_by_uid.expanding().count().reset_index(level=0, drop=True),
        "uid_amt_mean_prior": shifted_by_uid.expanding().mean().reset_index(level=0, drop=True),
        "uid_amt_std_prior": shifted_by_uid.expanding().std().reset_index(level=0, drop=True),
        "uid_amt_max_prior": shifted_by_uid.expanding().max().reset_index(level=0, drop=True),
    }, index=df.index)
    new_cols["uid_amt_ratio_to_prior_mean"] = df["TransactionAmt"] / new_cols["uid_amt_mean_prior"]

    return pd.concat([df, new_cols], axis=1)


def leakage_check(df: pd.DataFrame) -> None:
    """For the first transaction of every UID (no prior history exists):
      - count must equal 0 (zero prior transactions, NOT one)
      - mean/std/max must be NaN (no values to average)
    """
    print("\n--- LEAKAGE CHECK ---")
    df_sorted = df.sort_values(["UID", "TransactionDT"]).reset_index(drop=True)
    firsts = df_sorted.groupby("UID", sort=False).head(1)
    firsts = firsts[firsts["UID"].notna()]

    bad_count = (firsts["uid_txn_count_prior"] != 0).sum()
    if bad_count > 0:
        raise AssertionError(
            f"LEAKAGE: uid_txn_count_prior has {bad_count} non-zero values on first-per-UID rows"
        )
    print(f"  OK  uid_txn_count_prior: all first-per-UID values are 0")

    for col in ["uid_amt_mean_prior", "uid_amt_std_prior", "uid_amt_max_prior"]:
        non_null = firsts[col].notna().sum()
        if non_null > 0:
            raise AssertionError(
                f"LEAKAGE: {col} has {non_null} non-null values on first-per-UID rows"
            )
        print(f"  OK  {col}: all first-per-UID values are NaN")

    print("Leakage check passed.")


if __name__ == "__main__":
    from data_engineering import load_raw, reduce_mem_usage
    from features import build_time_columns, build_uids

    df = load_raw()
    df = reduce_mem_usage(df)
    df = build_time_columns(df)
    df = build_uids(df)
    df = build_uid_aggregates(df)

    leakage_check(df)

    print(f"\nShape: {df.shape}")
    new_cols = [c for c in df.columns if c.startswith("uid_") and c.endswith("_prior") or c == "uid_amt_ratio_to_prior_mean"]
    print(df[new_cols].describe())

    sample_uid = df[df["UID"].notna()]["UID"].value_counts().head(1).index[0]
    sample = df[df["UID"] == sample_uid].sort_values("TransactionDT")
    print(f"\nSample rows from UID with {len(sample)} transactions:")
    print(sample[["TransactionDT", "TransactionAmt"] + new_cols].head(10))