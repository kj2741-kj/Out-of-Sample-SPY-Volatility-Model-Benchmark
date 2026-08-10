"""
Phase 10: turn Phase 9's raw OOS scorings into the final result tables.

Produces:
 1. Headline ranking by horizon (RMSE, vega-RMSE, n) -- the main deliverable
 2. Decay curve: how each model degrades from h=5 -> 10 -> 21 days
 3. Era split (OptionsDX 2010-13 vs AlphaVantage 2014-25), kept separate throughout per the
    standing project rule
 4. Regime split: calm vs stressed periods (by realized market vol proxy)
 5. Coverage table (n per model) -- ESSENTIAL for honesty: AH_Surface silently drops contracts
    it can't price (Phase 6/9 finding), so its RMSE is on an easier subset and is NOT directly
    comparable. Any ranking must be read alongside coverage.
"""
import numpy as np
import pandas as pd

PROJ = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation"

MODEL_ORDER = ["BSM", "Black76", "GBM_MC", "CRR_American", "Heston", "SABR",
               "Merton", "AH_Surface", "LocalVol"]


def load():
    df = pd.read_csv(f"{PROJ}\\results\\phase9_oos_scores.csv")
    # AUDIT FIX (2026-08-09): a resumed Phase 9 run appends rows whose dates serialise WITH a
    # " 00:00:00" suffix while the original rows have none. Inferred-format parsing then raises
    # and Phase 10 dies silently behind a redirect, leaving a STALE tables file that looks valid.
    # format="mixed" handles both spellings.
    df["calib_date"] = pd.to_datetime(df["calib_date"], format="mixed")
    df["score_date"] = pd.to_datetime(df["score_date"], format="mixed")
    df["year"] = df["calib_date"].dt.year
    df["era"] = np.where(df["year"] <= 2013, "OptionsDX(2010-13)", "AlphaVantage(2014-25)")
    return df[df["n"] > 0]


def weighted(df, col):
    """n-weighted mean -- rows with more scored contracts should count more, and it makes the
    AH_Surface coverage gap explicit rather than letting a thin row carry equal weight."""
    return df.groupby("model").apply(
        lambda g: np.average(g[col], weights=g["n"]), include_groups=False)


def main():
    df = load()
    print(f"{len(df)} model-scoring rows, {df['calib_date'].nunique()} calibration dates, "
          f"{df['model'].nunique()} models\n")

    # 1. Headline: overall + per horizon
    print("=" * 78)
    print("HEADLINE: out-of-sample accuracy (params frozen at t, scored at t+h)")
    print("=" * 78)
    tbl = pd.DataFrame({
        "RMSE": weighted(df, "rmse"),
        "vega_RMSE": weighted(df, "vega_rmse"),
        "pct_in_spread": weighted(df, "pct_in_spread"),
        "total_contracts_scored": df.groupby("model")["n"].sum(),
        "coverage_%": df.groupby("model")["n"].sum() / df.groupby("model")["n"].sum().max() * 100,
    }).reindex(MODEL_ORDER).dropna(how="all")
    print(tbl.round(3).to_string())

    # 2. Decay curve
    print("\n" + "=" * 78)
    print("DECAY: RMSE by horizon (does the calibration hold up as time passes?)")
    print("=" * 78)
    decay = df.pivot_table(index="model", columns="horizon", values="rmse",
                            aggfunc=lambda x: np.mean(x)).reindex(MODEL_ORDER).dropna(how="all")
    decay.columns = [f"h={c}d" for c in decay.columns]
    if "h=5d" in decay and "h=21d" in decay:
        decay["degradation_%"] = (decay["h=21d"] / decay["h=5d"] - 1) * 100
    print(decay.round(3).to_string())

    # 3. Era split
    print("\n" + "=" * 78)
    print("ERA SPLIT (kept separate throughout the project)")
    print("=" * 78)
    era = df.pivot_table(index="model", columns="era", values="rmse",
                          aggfunc="mean").reindex(MODEL_ORDER).dropna(how="all")
    print(era.round(3).to_string())

    # 4. Regime split by year (proxy for calm vs stressed)
    print("\n" + "=" * 78)
    print("BY YEAR (RMSE) -- watch 2020 (COVID) and 2022 (rate hikes)")
    print("=" * 78)
    byyear = df.pivot_table(index="year", columns="model", values="rmse", aggfunc="mean")
    byyear = byyear.reindex(columns=[m for m in MODEL_ORDER if m in byyear.columns])
    print(byyear.round(3).to_string())

    # 5. Fair-comparison subset: only contracts/rows where EVERY model produced a price
    print("\n" + "=" * 78)
    print("FAIR SUBSET: calib_date+horizon cells where ALL models scored (removes AH's")
    print("survivorship advantage -- this is the honest apples-to-apples ranking)")
    print("=" * 78)
    cell = df.groupby(["calib_date", "horizon"])["model"].nunique()
    full_cells = set(cell[cell == df["model"].nunique()].index)
    fair = df[df.set_index(["calib_date", "horizon"]).index.isin(full_cells)]
    if len(fair):
        fair_tbl = pd.DataFrame({
            "RMSE": weighted(fair, "rmse"),
            "vega_RMSE": weighted(fair, "vega_rmse"),
            "n_cells": fair.groupby("model").size(),
        }).reindex(MODEL_ORDER).dropna(how="all").sort_values("RMSE")
        print(fair_tbl.round(3).to_string())
        print(f"\n({len(full_cells)} of {len(cell)} date-horizon cells have all models present)")
    else:
        print("No cells with all models present.")

    with open(f"{PROJ}\\results\\phase10_summary_tables.txt", "w") as f:
        f.write("HEADLINE\n" + tbl.round(3).to_string())
        f.write("\n\nDECAY\n" + decay.round(3).to_string())
        f.write("\n\nERA\n" + era.round(3).to_string())
        f.write("\n\nBY YEAR\n" + byyear.round(3).to_string())
        if len(fair):
            f.write("\n\nFAIR SUBSET\n" + fair_tbl.round(3).to_string())
    print(f"\nSaved -> results/phase10_summary_tables.txt")


if __name__ == "__main__":
    main()
