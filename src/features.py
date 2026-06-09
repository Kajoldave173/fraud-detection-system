"""Entity resolution and point-in-time feature engineering for IEEE-CIS."""

import numpy as np
import pandas as pd


def build_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TransactionDT (seconds offset) into day/hour columns."""
    seconds_per_day = 24 * 60 * 60
    new_cols = pd.DataFrame({
        "TransactionDay": (df["TransactionDT"] // seconds_per_day).astype(np.int16),
        "TransactionHour": ((df["TransactionDT"] // 3600) % 24).astype(np.int8),
    }, index=df.index)
    return pd.concat([df, new_cols], axis=1)


def build_uids(df: pd.DataFrame) -> pd.DataFrame:
    """Construct synthetic customer identifiers.

    UID_primary  = card1 + addr1 + D1n  (winner's formula; D1n is card's first-seen day)
    UID_fallback = card1 + addr1 + P_emaildomain  (when D1 is missing)
    is_new_entity = 1 if both UIDs unbuildable (true cold-start)
    """
    d1n = df["TransactionDay"] - df["D1"]

    uid_primary = (
        df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + d1n.astype(str)
    )
    primary_invalid = df["card1"].isna() | df["addr1"].isna() | df["D1"].isna()
    uid_primary = uid_primary.where(~primary_invalid)

    uid_fallback = (
        df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + df["P_emaildomain"].astype(str)
    )
    fallback_invalid = df["card1"].isna() | df["addr1"].isna() | df["P_emaildomain"].isna()
    uid_fallback = uid_fallback.where(~fallback_invalid)

    uid = uid_primary.fillna(uid_fallback)
    is_new_entity = uid.isna().astype(np.int8)

    new_cols = pd.DataFrame({
        "D1n": d1n,
        "UID_primary": uid_primary,
        "UID_fallback": uid_fallback,
        "UID": uid,
        "is_new_entity": is_new_entity,
    }, index=df.index)
    return pd.concat([df, new_cols], axis=1)


if __name__ == "__main__":
    from data_engineering import load_raw, reduce_mem_usage

    df = load_raw()
    df = reduce_mem_usage(df)
    df = build_time_columns(df)
    df = build_uids(df)

    print(f"\nShape: {df.shape}")
    print(f"Day range: {df['TransactionDay'].min()} to {df['TransactionDay'].max()}")
    print(f"\nUID coverage:")
    print(f"  Primary UID built:  {df['UID_primary'].notna().mean():.1%}")
    print(f"  Fallback UID built: {df['UID_fallback'].notna().mean():.1%}")
    print(f"  Final UID built:    {df['UID'].notna().mean():.1%}")
    print(f"  Cold-start rows:    {df['is_new_entity'].mean():.1%}")
    print(f"\nUnique UIDs: {df['UID'].nunique():,}")
    print(f"Avg transactions per UID: {df['UID'].value_counts().mean():.1f}")
    print(f"Max transactions per UID: {df['UID'].value_counts().max():,}")