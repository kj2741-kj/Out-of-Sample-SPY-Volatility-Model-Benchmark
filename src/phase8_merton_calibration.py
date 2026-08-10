"""
Phase 8 design: Merton Jump-Diffusion calibration. One set of 4 params (sigma, lambda, mu_j,
sigma_j) per date fit to the WHOLE surface at once -- same spirit as Heston (jump risk is a
market-wide parameter, not tenor-specific, standard treatment in the literature), unlike SABR's
per-expiry fits. Reuses Phase 4's exact select_chain() for cross-model comparability.

Design decisions:
1. Objective: vega-weighted IV-space squared error, same as Heston/SABR.
2. Optimizer: coarse grid pre-screen + L-BFGS-B refine, same pattern as Heston -- but expected
   to be far more reliable given merton_price() is a simple closed-form sum, no numerical
   integration/PDE fragility.
3. Bounds: sigma[0.01,2], lambda[0,5] (jumps/year), mu_j[-2,1] (mean log jump size, expect
   negative = crash risk), sigma_j[0.001,2].
4. No warm-starting attempted initially -- test whether it's even needed given expected speed/
   reliability (unlike Heston where warm-starting was load-bearing for the schedule).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from phase4_heston_calibration import select_chain, PROJ, Q_DIV
from pricers import bs_implied_vol
from merton import merton_price, merton_price_batch

BOUNDS = [(0.01, 2.0), (0.0, 5.0), (-2.0, 1.0), (0.001, 2.0)]  # sigma, lambda, mu_j, sigma_j


def objective(x, spot, contracts, r_flat):
    """Prices the whole contract set in ONE vectorized call (merton_price_batch) instead of
    looping per-contract in Python -- this, not just vectorizing the n_terms sum alone, is what
    cut a single date's calibration from ~130s to a few seconds (verified below)."""
    sigma, lam, mu_j, sigma_j = x
    prices = merton_price_batch(spot, contracts["strike"].values, contracts["T"].values,
                                 r_flat, Q_DIV, sigma, lam, mu_j, sigma_j, contracts["is_call"].values)
    err = 0.0
    for price, row in zip(prices, contracts.itertuples(index=False)):
        if not np.isfinite(price) or price <= 0:
            err += 10.0
            continue
        model_iv = bs_implied_vol(price, spot, row.strike, row.T, r_flat, Q_DIV, row.is_call)
        if not np.isfinite(model_iv):
            err += 10.0
            continue
        vega = max(row.vega, 1e-4)
        err += vega * (model_iv - row.implied_volatility) ** 2
    return err


def calibrate_one_date(calib_date, df, vas, verbose=False):
    spot = df.loc[df["date"] == calib_date, "spy_close"].iloc[0]
    r3mo = vas.loc[vas["date"] == calib_date, "r3mo"].iloc[0]
    contracts = select_chain(df, calib_date)
    atm_iv = contracts.loc[(contracts["moneyness"] - 1).abs().idxmin(), "implied_volatility"]

    grid_sigma = [atm_iv * 0.7, atm_iv * 0.9]
    grid_lam = [0.3, 1.0, 2.5]
    grid_mu_j = [-0.3, -0.1]
    grid_sigma_j = [0.1, 0.25]
    seeds = []
    for sigma in grid_sigma:
        for lam in grid_lam:
            for mu_j in grid_mu_j:
                for sigma_j in grid_sigma_j:
                    x = [sigma, lam, mu_j, sigma_j]
                    val = objective(x, spot, contracts, r3mo)
                    seeds.append((val, x))
    seeds.sort(key=lambda t: t[0])

    best = None
    for val, x0 in seeds[:2]:
        res = minimize(lambda x: objective(x, spot, contracts, r3mo), x0, method="L-BFGS-B",
                        bounds=BOUNDS, options=dict(maxiter=60, eps=1e-5))
        if best is None or res.fun < best.fun:
            best = res

    sigma, lam, mu_j, sigma_j = best.x
    prices = merton_price_batch(spot, contracts["strike"].values, contracts["T"].values,
                                 r3mo, Q_DIV, sigma, lam, mu_j, sigma_j, contracts["is_call"].values)
    ivs = [bs_implied_vol(p, spot, row.strike, row.T, r3mo, Q_DIV, row.is_call)
           for p, row in zip(prices, contracts.itertuples(index=False))]
    iv_errors = np.array([abs(iv - row.implied_volatility) if np.isfinite(iv) else np.nan
                           for iv, row in zip(ivs, contracts.itertuples(index=False))])
    mean_iv_error = np.nanmean(iv_errors)

    if verbose:
        print(f"{calib_date.date()}: obj={best.fun:.6f} converged={best.success} sigma={sigma:.4f} "
              f"lambda={lam:.3f} mu_j={mu_j:.4f} sigma_j={sigma_j:.4f} mean|IVerr|={mean_iv_error*100:.3f}pp")

    return dict(date=calib_date, sigma=sigma, lam=lam, mu_j=mu_j, sigma_j=sigma_j,
                objective=best.fun, converged=bool(best.success), n_contracts=len(contracts),
                mean_iv_error=mean_iv_error, spot=spot, r3mo=r3mo)


def main():
    calib_date = pd.Timestamp("2019-06-21")
    df = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_2019.parquet")
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    res = calibrate_one_date(calib_date, df, vas, verbose=True)
    pd.DataFrame([res]).to_csv(f"{PROJ}\\results\\phase8_merton_single_date_fit.csv", index=False)


if __name__ == "__main__":
    main()
