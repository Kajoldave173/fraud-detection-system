"""Entity resolution and point-in-time feature engineering for IEEE-CIS."""

import numpy as np
import pandas as pd


def build_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TransactionDT (seconds offset) into day/hour columns.

    TransactionDT is seconds from an unknown reference datetime. We treat
    day 0 = first day in the data, so TransactionDay is an integer day index
    from 0 to ~181 across the dataset's ~6-month span.
    """
    seconds_per_day = 24 * 60 * 60
    df["TransactionDay"] = (df["TransactionDT"] // seconds_per_day).astype(np.int16)
    df["TransactionHour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    return df


def build_uids(df: pd.DataFrame) -> pd.DataFrame:
    """Construct synthetic customer identifiers.

    The IEEE-CIS dataset has no customer ID, but we can synthesize one.
    D1 is "days since card was first seen", so (TransactionDay - D1) gives
    the card's first-seen day -- a stable anchor that's identical across
    every transaction from the same card.

    UID_primary = card1 + addr1 + D1n  (the winner's formula)
    UID_fallback = card1 + addr1 + P_emaildomain  (when D1 is missing)
    is_new_entity = 1 if both UIDs couldn't be built (true cold-start)
    """
    # D1n: a stable per-card anchor day
    df["D1n"] = df["TransactionDay"] - df["D1"]

    # Primary UID: card1 + addr1 + D1n
    df["UID_primary"] = (
        df["card1"].astype(str)
        + "_"
        + df["addr1"].astype(str)
        + "_"
        + df["D1n"].astype(str)
    )
    # Mark rows where any component was NaN (these UIDs are unreliable)
    primary_invalid = df["card1"].isna() | df["addr1"].isna() | df["D1"].isna()
    df.loc[primary_invalid, "UID_primary"] = np.nan

    # Fallback UID: card1 + addr1 + P_emaildomain
    df["UID_fallback"] = (
        df["card1"].astype(str)
        + "_"
        + df["addr1"].astype(str)
        + "_"
        + df["P_emaildomain"].astype(str)
    )
    fallback_invalid = (
        df["card1"].isna() | df["addr1"].isna() | df["P_emaildomain"].isna()
    )
    df.loc[fallback_invalid, "UID_fallback"] = np.nan

    # Final UID: prefer primary, fall back to secondary
    df["UID"] = df["UID_primary"].fillna(df["UID_fallback"])

    # Cold-start flag
    df["is_new_entity"] = df["UID"].isna().astype(np.int8)

    return df


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