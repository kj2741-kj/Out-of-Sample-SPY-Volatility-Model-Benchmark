"""
Phase 10: final result tables from Phase 9's per-contract out-of-sample prices.

AUDIT REWRITE (2026-11-18). The earlier version worked from per-cell aggregates and had three
problems, all fixed here:
 1. Its "RMSE" was an n-weighted MEAN of per-cell RMSEs, not an RMSE. The pooled RMSE,
    sqrt(mean of squared errors over all contracts), is reported now; the old statistic is
    kept in one column only for comparison with earlier write-ups.
 2. Its "fair subset" required each model to be PRESENT in a cell, not to price every contract
    in it. LocalVol skipped 182 contracts inside such cells, mostly in hard cells, which
    flattered it. The like-for-like set is now defined per CONTRACT: priced by all eight
    ranked models.
 3. Decay, era and year tables used an unweighted mean of cell RMSEs while the headline used a
    weighted one. Every table now uses pooled RMSE on the same like-for-like contracts.

Andreasen-Huge stays out of the ranking (it prices about half the grid, the hard half missing)
and is reported on its own coverage only.

Reference row: Oracle_VendorIV = Black-Scholes at each contract's own same-day vendor IV. It is
not a model; its error is the floor every model inherits because calibration targets vendor IVs
while scoring uses mids.
"""
import numpy as np
import pandas as pd

PROJ = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation"
RANKED = ["LocalVol", "SABR", "Heston", "Merton", "GBM_MC", "CRR_American", "BSM", "Black76"]
EXCLUDED = ["AH_Surface"]
REFERENCE = "Oracle_VendorIV"


def pooled_rmse(err):
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan


def load():
    c = pd.read_csv(f"{PROJ}\\results\\phase9_oos_contracts.csv")
    c["calib_date"] = pd.to_datetime(c["calib_date"], format="mixed")
    c["score_date"] = pd.to_datetime(c["score_date"], format="mixed")
    c["year"] = c["calib_date"].dt.year
    c["era"] = np.where(c["year"] <= 2013, "OptionsDX 2010-13", "AlphaVantage 2014-25")
    for m in RANKED + EXCLUDED + [REFERENCE]:
        if m in c:
            c[f"err_{m}"] = c[m] - c["mid"]
    return c


def like_for_like(c):
    """Contracts priced by every ranked model."""
    return c[c[RANKED].notna().all(axis=1)].copy()


def table(c, models, by=None):
    cols = models + [REFERENCE]
    if by is None:
        return pd.Series({m: pooled_rmse(c[f"err_{m}"]) for m in cols})
    return c.groupby(by).apply(lambda g: pd.Series({m: pooled_rmse(g[f"err_{m}"]) for m in cols}),
                               include_groups=False).T


def main():
    c = load()
    n_all = len(c)
    lfl = like_for_like(c)
    out = []

    def emit(title, obj):
        text = f"\n{'=' * 78}\n{title}\n{'=' * 78}\n{obj if isinstance(obj, str) else obj.to_string()}"
        print(text)
        out.append(text)

    emit("SAMPLE",
         f"{n_all:,} contract scorings, {c.groupby(['calib_date', 'horizon']).ngroups} date-horizon cells, "
         f"{c['calib_date'].nunique()} calibration dates.\n"
         f"Like-for-like set (priced by all 8 ranked models): {len(lfl):,} contracts "
         f"({len(lfl) / n_all:.1%}).")

    cov = pd.DataFrame({
        "pooled_RMSE_own_coverage": [pooled_rmse(c[f"err_{m}"]) for m in RANKED + EXCLUDED],
        "contracts_priced": [int(c[m].notna().sum()) for m in RANKED + EXCLUDED],
    }, index=RANKED + EXCLUDED)
    cov["coverage_%"] = cov["contracts_priced"] / n_all * 100
    emit("EACH MODEL ON ITS OWN COVERAGE (not comparable across models with <100% coverage)",
         cov.round(3))

    head = pd.DataFrame({
        "pooled_RMSE": table(lfl, RANKED),
        "median_abs_err": pd.Series({m: float(lfl[f"err_{m}"].abs().median()) for m in RANKED + [REFERENCE]}),
        "mean_abs_err": pd.Series({m: float(lfl[f"err_{m}"].abs().mean()) for m in RANKED + [REFERENCE]}),
        "bias": pd.Series({m: float(lfl[f"err_{m}"].mean()) for m in RANKED + [REFERENCE]}),
        "old_stat_mean_of_cell_RMSE": pd.Series({
            m: float(lfl.groupby(["calib_date", "horizon"]).apply(
                lambda g: np.sqrt(np.mean(g[f"err_{m}"] ** 2)), include_groups=False).mean())
            for m in RANKED + [REFERENCE]}),
    })
    head["vs_BSM_%"] = (head["pooled_RMSE"] / head.loc["BSM", "pooled_RMSE"] - 1) * 100
    emit("HEADLINE: like-for-like contracts, pooled RMSE in $ (Oracle_VendorIV = reference floor, not a model)",
         head.round(3))

    emit("DECAY BY HORIZON (pooled RMSE, like-for-like)", table(lfl, RANKED, by="horizon").round(3))
    emit("ERA SPLIT (pooled RMSE, like-for-like)", table(lfl, RANKED, by="era").round(3))
    emit("BY YEAR (pooled RMSE, like-for-like)", table(lfl, RANKED, by="year").round(3))

    pairs = []
    for a, b in [("LocalVol", "SABR"), ("LocalVol", "Heston"), ("LocalVol", "BSM"),
                 ("SABR", "Heston"), ("Heston", "Merton"), ("SABR", "Merton"), ("Merton", "BSM")]:
        win_c = float((lfl[f"err_{a}"].abs() < lfl[f"err_{b}"].abs()).mean() * 100)
        cells = lfl.groupby(["calib_date", "horizon"]).apply(
            lambda g: pooled_rmse(g[f"err_{a}"]) < pooled_rmse(g[f"err_{b}"]), include_groups=False)
        pairs.append(dict(pair=f"{a} beats {b}", contracts_pct=win_c, cells_pct=float(cells.mean() * 100)))
    emit("HEAD TO HEAD (like-for-like): share of contracts and of date-horizon cells won",
         pd.DataFrame(pairs).set_index("pair").round(1))

    sub = c[c["AH_Surface"].notna()]
    ah = pd.Series({m: pooled_rmse(sub[f"err_{m}"]) for m in ["AH_Surface", "LocalVol", "SABR", "BSM"]})
    emit(f"ANDREASEN-HUGE, excluded: on the {len(sub):,} contracts it priced ({len(sub) / n_all:.1%}), "
         "other models' pooled RMSE on the same contracts", ah.round(3))

    with open(f"{PROJ}\\results\\phase10_summary_tables.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\nSaved -> results/phase10_summary_tables.txt")


if __name__ == "__main__":
    main()
