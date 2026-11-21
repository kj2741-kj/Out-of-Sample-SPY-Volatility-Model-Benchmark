"""
Phase 12 scoring: true Dupire local vol vs the Phase 9 models on the same contract-scorings.

Labels used from here on:
  StickyStrike_Surface = the old Phase 9 "LocalVol" column (frozen SABR-smoothed implied-vol surface
                         read by strike; localVol flag missing)
  LocalVol_Dupire      = true Dupire local vol (Phase 12)
Pooled RMSE in $ on like-for-like contracts (priced by every model in the comparison), plus a
cluster bootstrap over calibration dates (2,000 draws) and the by-horizon decay.
Guard: the script first re-derives the published Phase 9 headline (7,828 contracts, 0.864 / 1.889)
from the old columns and stops if it does not match.
"""
import numpy as np
import pandas as pd
from phase4_heston_calibration import PROJ

IN = f"{PROJ}\\results\\phase12_localvol_oos_contracts.csv"
OUT = f"{PROJ}\\results\\phase12_summary.txt"
OLD = ["BSM", "Black76", "GBM_MC", "CRR_American", "Heston", "SABR", "Merton", "LocalVol"]
RNG = np.random.default_rng(12)


def rmse(x, m):
    return float(np.sqrt(((x[m] - x["mid"]) ** 2).mean()))


def boot(x, m, n=2000):
    cd = x["calib_date"].values
    u = np.unique(cd)
    idx = [np.where(cd == k)[0] for k in u]
    e2 = ((x[m] - x["mid"]) ** 2).values
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in RNG.integers(0, len(u), len(u))])
        out.append(np.sqrt(e2[pick].mean()))
    return np.percentile(out, [2.5, 97.5])


def main():
    c = pd.read_csv(IN)
    for k in ("calib_date", "score_date"):
        c[k] = pd.to_datetime(c[k], format="mixed")
    lines = []
    old_ok = c[OLD].notna().all(axis=1)
    n_old, l_old, b_old = int(old_ok.sum()), rmse(c[old_ok], "LocalVol"), rmse(c[old_ok], "BSM")
    lines.append(f"GUARD: old like-for-like n={n_old}, 'LocalVol' {l_old:.3f}, BSM {b_old:.3f} (published 7828 / 0.864 / 1.889)")
    assert n_old == 7828 and abs(l_old - 0.864) < 0.001 and abs(b_old - 1.889) < 0.001, "guard failed"

    c = c.rename(columns={"LocalVol": "StickyStrike_Surface"})
    models = ["BSM", "Black76", "GBM_MC", "CRR_American", "Heston", "SABR", "Merton", "StickyStrike_Surface", "LocalVol_Dupire"]
    cov = c["LocalVol_Dupire"].notna().mean()
    lines.append(f"Dupire coverage: {c['LocalVol_Dupire'].notna().sum()} of {len(c)} contract-scorings ({cov:.1%}); "
                 f"calibration dates {c['calib_date'].nunique()}; mean guarded-node share {c.groupby('calib_date')['dupire_guarded_share'].first().mean():.2%}")
    ok = c[models].notna().all(axis=1)
    x = c[ok]
    rows = []
    for m in models:
        lo, hi = boot(x, m)
        rows.append(dict(model=m, pooled_rmse=rmse(x, m), ci_lo=lo, ci_hi=hi, vs_BSM=rmse(x, m) / rmse(x, "BSM") - 1))
    t = pd.DataFrame(rows).sort_values("pooled_rmse")
    lines += ["", f"LIKE-FOR-LIKE, all 9 models ({len(x)} contract-scorings), pooled RMSE $", t.round(3).to_string(index=False)]
    dec = x.groupby("horizon").apply(lambda g: pd.Series({m: rmse(g, m) for m in models})).round(3)
    lines += ["", "BY HORIZON (trading days)", dec.to_string()]
    x = x.assign(year=x["score_date"].dt.year)
    yr = x.groupby("year").apply(lambda g: pd.Series({m: rmse(g, m) for m in ["BSM", "SABR", "StickyStrike_Surface", "LocalVol_Dupire"]})).round(3)
    lines += ["", "BY YEAR (selected models)", yr.to_string()]
    h2h = (np.abs(x["LocalVol_Dupire"] - x["mid"]) < np.abs(x["SABR"] - x["mid"])).mean()
    h2s = (np.abs(x["LocalVol_Dupire"] - x["mid"]) < np.abs(x["StickyStrike_Surface"] - x["mid"])).mean()
    lines += ["", f"Head to head: Dupire beats SABR on {h2h:.1%} of contracts, beats StickyStrike_Surface on {h2s:.1%}"]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
