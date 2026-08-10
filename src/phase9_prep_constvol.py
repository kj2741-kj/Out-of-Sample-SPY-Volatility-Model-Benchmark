"""
Phase 9 prep: fit ONE constant volatility per calibration date, for the four BSM-family models
(BSM, Black-76, GBM-MC, CRR-American). Gap found while building Phase 9: Phase 3 built and
cross-validated those pricers but never *calibrated* them -- they need a per-date sigma to be
scorable out-of-sample on the same footing as Heston/SABR/Merton.

Definition: the single sigma minimizing the same vega-weighted IV-space squared error used for
every other model's calibration, over the same select_chain() contract set. Deliberately the
same objective and same chain, so the comparison in Phase 9 differs only by MODEL, not by
calibration methodology.

Fast (1 param, closed-form pricing) -- the whole 192-date run takes well under a minute.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from phase4_heston_batch import monthly_calibration_dates
from phase4_heston_calibration import select_chain, PROJ, Q_DIV


def fit_const_vol(calib_date, df, vas):
    spot = df.loc[df["date"] == calib_date, "spy_close"].iloc[0]
    r3mo = vas.loc[vas["date"] == calib_date, "r3mo"].iloc[0]
    contracts = select_chain(df, calib_date)
    ivs = contracts["implied_volatility"].values
    vegas = np.maximum(contracts["vega"].values, 1e-4)

    # In IV space with a constant-vol model, model_iv == sigma for every contract, so the
    # vega-weighted least-squares solution is just the vega-weighted mean IV (closed form).
    # Solving it numerically anyway would give the identical answer -- doing it in closed form
    # is exact and instant.
    sigma = float(np.sum(vegas * ivs) / np.sum(vegas))
    resid = ivs - sigma
    return dict(date=calib_date, sigma=sigma, spot=spot, r3mo=r3mo,
                n_contracts=len(contracts),
                mean_iv_error=float(np.mean(np.abs(resid))))


def main():
    dates = monthly_calibration_dates()
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    rows, year_cache = [], {}
    for d in dates:
        if d.year not in year_cache:
            year_cache.clear()
            year_cache[d.year] = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{d.year}.parquet")
        rows.append(fit_const_vol(d, year_cache[d.year], vas))
    out = pd.DataFrame(rows)
    out.to_parquet(f"{PROJ}\\data\\constvol_params_by_date.parquet", index=False)
    out.to_csv(f"{PROJ}\\results\\phase9_constvol_params.csv", index=False)
    print(f"{len(out)} dates. sigma range {out['sigma'].min():.4f}-{out['sigma'].max():.4f}, "
          f"mean in-sample |IV err| = {out['mean_iv_error'].mean()*100:.3f}pp "
          f"(expected LARGE -- a single flat vol cannot fit a skewed smile; that's the point of "
          f"the comparison)")


if __name__ == "__main__":
    main()
