"""One-shot pipeline runner: raw CSVs -> engineered Parquet.

Run this once per data update. Downstream code (training, inference,
monitoring) all read from the Parquet output instead of re-running
the 3-minute load every time.
"""

import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_engineering import load_raw, reduce_mem_usage
from features import build_time_columns, build_uids
from aggregates import build_uid_aggregates, leakage_check


OUTPUT_PATH = Path("data/processed/features.parquet")


def main() -> None:
    df = load_raw()
    df = reduce_mem_usage(df)
    df = build_time_columns(df)
    df = build_uids(df)
    df = build_uid_aggregates(df)

    leakage_check(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {OUTPUT_PATH}...")
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy", index=False)

    size_mb = OUTPUT_PATH.stat().st_size / 1024**2
    print(f"  size: {size_mb:.1f} MB")
    print(f"  shape: {df.shape}")
    print(f"Done.")


if __name__ == "__main__":
    main()