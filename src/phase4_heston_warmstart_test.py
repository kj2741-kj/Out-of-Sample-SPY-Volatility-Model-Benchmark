"""
Phase 4 design decision 5: does warm-starting a date's calibration from the prior
date's converged params (skip the grid, one L-BFGS-B refine) reach essentially the
same fit as a full cold-start (grid + refine top 3), for less compute?

Tests 2019-06-24 (next trading day after the working 2019-06-21 calibration) both ways.
"""
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from phase4_heston_calibration import build_curves, select_chain, objective, BOUNDS, PROJ

def cold_start(calib_date, df, vas, vas_params):
    spot = df.loc[df["date"] == calib_date, "spy_close"].iloc[0]
    r3mo = vas.loc[vas["date"] == calib_date, "r3mo"].iloc[0]
    eval_date, calendar, day_count, r_ts, q_ts = build_curves(calib_date, r3mo, vas_params)
    contracts = select_chain(df, calib_date)
    atm_iv = contracts.loc[(contracts["moneyness"] - 1).abs().idxmin(), "implied_volatility"]
    v0_seed = atm_iv ** 2

    grid_kappa, grid_xi, grid_rho = [0.5, 2.0, 5.0], [0.3, 0.6, 1.0], [-0.7, -0.3]
    grid_theta = [max(v0_seed * 0.7, 0.01), v0_seed, v0_seed * 1.5]
    seeds = []
    for kappa in grid_kappa:
        for theta in grid_theta:
            for xi in grid_xi:
                for rho in grid_rho:
                    x = [kappa, theta, xi, rho, v0_seed]
                    val, _ = objective(x, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts, r3mo)
                    seeds.append((val, x))
    seeds.sort(key=lambda t: t[0])

    bounds_list = [BOUNDS["kappa"], BOUNDS["theta"], BOUNDS["xi"], BOUNDS["rho"], BOUNDS["v0"]]
    best = None
    for val, x0 in seeds[:3]:
        res = minimize(lambda x: objective(x, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts, r3mo)[0],
                        x0, method="L-BFGS-B", bounds=bounds_list, options=dict(maxiter=100, eps=1e-4))
        if best is None or res.fun < best.fun:
            best = res
    return best, contracts, spot, r3mo, eval_date, calendar, day_count, r_ts, q_ts


def warm_start(calib_date, df, vas, vas_params, prior_params):
    spot = df.loc[df["date"] == calib_date, "spy_close"].iloc[0]
    r3mo = vas.loc[vas["date"] == calib_date, "r3mo"].iloc[0]
    eval_date, calendar, day_count, r_ts, q_ts = build_curves(calib_date, r3mo, vas_params)
    contracts = select_chain(df, calib_date)
    bounds_list = [BOUNDS["kappa"], BOUNDS["theta"], BOUNDS["xi"], BOUNDS["rho"], BOUNDS["v0"]]
    res = minimize(lambda x: objective(x, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts, r3mo)[0],
                    prior_params, method="L-BFGS-B", bounds=bounds_list, options=dict(maxiter=100, eps=1e-4))
    return res


def main():
    df = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_2019.parquet")
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]

    d1 = pd.Timestamp("2019-06-21")
    d2 = pd.Timestamp("2019-06-24")

    t0 = time.time()
    res1, *_ = cold_start(d1, df, vas, vas_params)
    print(f"Day 1 ({d1.date()}) cold-start: obj={res1.fun:.6f} params={np.round(res1.x,4)} "
          f"time={time.time()-t0:.1f}s")

    t0 = time.time()
    res2_cold, *_ = cold_start(d2, df, vas, vas_params)
    t_cold = time.time() - t0
    print(f"Day 2 ({d2.date()}) COLD start: obj={res2_cold.fun:.6f} params={np.round(res2_cold.x,4)} "
          f"time={t_cold:.1f}s")

    t0 = time.time()
    res2_warm = warm_start(d2, df, vas, vas_params, prior_params=res1.x)
    t_warm = time.time() - t0
    print(f"Day 2 ({d2.date()}) WARM start (seeded from day1): obj={res2_warm.fun:.6f} "
          f"params={np.round(res2_warm.x,4)} time={t_warm:.1f}s")

    print(f"\nObjective ratio (warm/cold): {res2_warm.fun/res2_cold.fun:.3f} "
          f"(close to 1.0 = warm start reaches an equally good fit)")
    print(f"Speedup: {t_cold/t_warm:.1f}x")
    param_diff = np.abs(res2_warm.x - res2_cold.x)
    print(f"Param diff (warm vs cold): kappa={param_diff[0]:.4f} theta={param_diff[1]:.4f} "
          f"xi={param_diff[2]:.4f} rho={param_diff[3]:.4f} v0={param_diff[4]:.4f}")

if __name__ == "__main__":
    main()
