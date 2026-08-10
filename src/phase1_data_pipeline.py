"""
Phase 1: build clean, enriched daily chain snapshots from the raw combined
options parquet. Scope is deliberately narrow -- spot merge, time-to-expiry,
moneyness, mid price, and data-quality flags. The real discount curve
(replacing any rate proxy) is Phase 2's job, not this one's.

Output: data/clean/spy_options_clean_<year>.parquet, partitioned by year so
downstream per-date calibration phases can filter-pushdown instead of loading
23.5M rows every time.
"""
import numpy as np
import pandas as pd

RAW = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Data\spy_options_2010_2025_combined.parquet"
PROJ = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation"
OUT_DIR = f"{PROJ}\\data\\clean"

def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    opts = pd.read_parquet(RAW)
    spy = pd.read_parquet(f"{PROJ}\\data\\spy_underlying_close.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    opts["date"] = pd.to_datetime(opts["date"])
    opts["expiration"] = pd.to_datetime(opts["expiration"])

    n_before = len(opts)
    opts = opts.merge(spy, on="date", how="left")
    n_unmatched_spot = opts["spy_close"].isna().sum()

    opts["T"] = (opts["expiration"] - opts["date"]).dt.days / 365.0
    opts["moneyness"] = opts["strike"] / opts["spy_close"]
    opts["is_call"] = opts["type"] == "call"
    opts["mid"] = np.where(
        (opts["bid"] > 0) & (opts["ask"] > 0), (opts["bid"] + opts["ask"]) / 2, np.nan
    )

    opts["has_valid_quote"] = (opts["bid"] > 0) & (opts["ask"] > 0)
    opts["iv_quality_ok"] = opts["implied_volatility"] > 0.01
    opts["not_expired_yet"] = opts["T"] > (1 / 365)

    qa_rows = []
    for year, g in opts.groupby("year"):
        qa_rows.append(dict(
            year=year, source=g["source"].iloc[0], n_rows=len(g),
            pct_valid_quote=100 * g["has_valid_quote"].mean(),
            pct_iv_quality_ok=100 * g["iv_quality_ok"].mean(),
            pct_expired_same_day=100 * (~g["not_expired_yet"]).mean(),
            pct_missing_spot=100 * g["spy_close"].isna().mean(),
        ))
        out_path = f"{OUT_DIR}\\spy_options_clean_{year}.parquet"
        g.to_parquet(out_path, index=False)

    qa = pd.DataFrame(qa_rows).sort_values("year")
    qa.to_csv(f"{PROJ}\\results\\phase1_qa_summary.csv", index=False)

    print(f"Rows in: {n_before:,}  Rows out: {len(opts):,}  Unmatched spot: {n_unmatched_spot:,}")
    print(qa.to_string(index=False))
    print(f"\nWrote {opts['year'].nunique()} yearly partitions to {OUT_DIR}")

if __name__ == "__main__":
    main()
