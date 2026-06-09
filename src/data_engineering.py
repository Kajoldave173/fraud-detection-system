"""Data loading and memory optimization for IEEE-CIS fraud detection."""

from pathlib import Path
import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to the smallest dtype that fits the data."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object or isinstance(col_type, pd.CategoricalDtype):
            continue

        c_min, c_max = df[col].min(), df[col].max()
        # Skip if min/max aren't numeric (e.g., mixed-type column)
        if not (np.issubdtype(type(c_min), np.number) and np.issubdtype(type(c_max), np.number)):
            continue

        if str(col_type).startswith("int"):
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            # float16 is unstable in LightGBM, skip it
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        pct = 100 * (start_mem - end_mem) / start_mem
        print(f"Memory: {start_mem:.1f} MB -> {end_mem:.1f} MB ({pct:.1f}% reduction)")
    return df


def load_raw(
    transaction_path: str | Path = "data/raw/train_transaction.csv",
    identity_path: str | Path = "data/raw/train_identity.csv",
) -> pd.DataFrame:
    """Load raw IEEE-CIS files and left-join transaction with identity."""
    print(f"Loading {transaction_path}...")
    txn = pd.read_csv(transaction_path)
    print(f"  shape: {txn.shape}")

    print(f"Loading {identity_path}...")
    idy = pd.read_csv(identity_path)
    print(f"  shape: {idy.shape}")

    print("Left-joining on TransactionID...")
    df = txn.merge(idy, on="TransactionID", how="left")
    print(f"  merged shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = load_raw()
    df = reduce_mem_usage(df)
    print(f"\nFinal shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.4f}")