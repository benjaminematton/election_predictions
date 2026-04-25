"""OpenSecrets independent-expenditure aggregations (secondary source).

Per the v3 plan, FEC Schedule E is the primary IE source (canonical, snapshot-
filterable). OpenSecrets is used only for value-add aggregations FEC doesn't
expose directly — e.g., dark-money totals by candidate already grouped.

For now this is a thin reader for the legacy CSV format used by the original
project, kept for backward compatibility with cached files in legacy_src/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dark_money_csv(path: Path) -> pd.DataFrame:
    """Load a dark-money CSV staged from OpenSecrets.

    Filters to House races (drops Senate rows where the DISTRICT field
    contains 'S').
    """
    df = pd.read_csv(path)

    # Drop columns that are noise / duplicates of fields we have elsewhere.
    drop = [
        c for c in ("Name", "State/Dist", "For Dems", "Against Dems",
                    "For Repubs", "AgainstRepubs")
        if c in df.columns
    ]
    if drop:
        df = df.drop(columns=drop)

    if "DISTRICT" in df.columns:
        # Senate rows have 'S' in DISTRICT; House rows have integers.
        df = df.loc[~df["DISTRICT"].astype(str).str.contains("S", na=False)].copy()
        df["DISTRICT"] = pd.to_numeric(df["DISTRICT"], errors="coerce")

    for col in ("DARK_FOR", "DARK_AGAINST", "Total"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.rename(columns=str.lower)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load OpenSecrets dark-money CSV.")
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    df = load_dark_money_csv(args.path)
    print(df.head())
    print(f"\n{len(df)} rows.")
