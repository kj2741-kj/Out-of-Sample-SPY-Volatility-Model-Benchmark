"""
Phase 4 design: Heston calibration framework, tested on a single date before any
batch run. Five decisions locked here (see docs/phase4_handoff.md for the full
rationale of each):

1. Objective: vega-weighted IV-space squared error (not raw price error, which
   overweights deep-ITM contracts that carry little smile information).
2. Optimizer: coarse grid pre-screen (cheap forward pricing only) -> refine the
   best few seeds with L-BFGS-B. Plain single-seed local optimization risks a
   bad local minimum; full global search (differential evolution) is overkill
   given the grid already covers the plausible region.
3. Feller condition (2*kappa*theta > xi^2): soft quadratic penalty, not a hard
   constraint -- violating it doesn't break the model (variance can hit zero),
   real market calibrations often sit close to or across the line.
4. Bounds: kappa[0.01,20], theta[0.001,4.0], xi[0.01,5.0], rho[-0.999,0.999],
   v0[0.0001,4.0] -- wide enough to avoid the Phase 2 Vasicek-MLE-style
   degenerate-boundary trap, tight enough to keep the optimizer out of
   nonsensical regions.
5. Cross-date seeding: not testable with a single date -- flagged for the
   batch-run (Sonnet) phase: warm-start each date from the prior date's
   converged params, falling back to the grid seed if the warm start's
   objective is much worse than a fresh grid search (signals a regime break).
"""
import numpy as np
import pandas as pd
import QuantLib as ql
from pricers import bs_implied_vol

PROJ = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation"
Q_DIV = 0.014
FELLER_PENALTY_WEIGHT = 50.0

BOUNDS = dict(kappa=(0.01, 20.0), theta=(0.001, 4.0), xi=(0.01, 5.0),
              rho=(-0.999, 0.999), v0=(0.0001, 4.0))


def vasicek_zero_rate(a, b, sigma, r0, tau):
    B = (1 - np.exp(-a * tau)) / a
    A = np.exp((b - sigma**2 / (2 * a**2)) * (B - tau) - (sigma**2 / (4 * a)) * B**2)
    P = A * np.exp(-B * r0)
    return -np.log(P) / tau


def build_curves(calib_date, r3mo, vasicek_params):
    a, b, sigma = vasicek_params["a"], vasicek_params["b"], vasicek_params["sigma"]
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()
    eval_date = ql.Date(calib_date.day, calib_date.month, calib_date.year)
    ql.Settings.instance().evaluationDate = eval_date

    pillar_taus = [1/12, 3/12, 0.5, 1.0, 1.5, 2.0, 3.0]
    dates = [eval_date] + [eval_date + ql.Period(int(round(t * 365)), ql.Days) for t in pillar_taus]
    zero_rates = [r3mo] + [vasicek_zero_rate(a, b, sigma, r3mo, t) for t in pillar_taus]
    r_curve = ql.ZeroCurve(dates, zero_rates, day_count)
    r_ts = ql.YieldTermStructureHandle(r_curve)

    q_curve = ql.FlatForward(eval_date, Q_DIV, day_count)
    q_ts = ql.YieldTermStructureHandle(q_curve)
    return eval_date, calendar, day_count, r_ts, q_ts


def select_chain(df, calib_date, n_expiries=5, strikes_per_expiry=9):
    """Balance puts/calls explicitly per expiry -- an earlier version linspaced over the
    combined sorted-by-moneyness set, which silently over-sampled puts whenever put strikes
    were simply denser in the raw data (27 vs 18 in the first pass on 2019-06-21), biasing
    the vega-weighted objective toward the put wing independent of any real model limitation."""
    day = df[(df["date"] == calib_date) & df["iv_quality_ok"] & df["has_valid_quote"]].copy()
    day = day[(day["moneyness"] >= 0.85) & (day["moneyness"] <= 1.15) & (day["T"] >= 0.08) & (day["T"] <= 2.0)]
    day = day[((day["moneyness"] < 1.0) & (~day["is_call"])) | ((day["moneyness"] >= 1.0) & (day["is_call"]))]

    expiries = sorted(day["expiration"].unique())
    step = max(1, len(expiries) // n_expiries)
    chosen_expiries = expiries[::step][:n_expiries]

    half = strikes_per_expiry // 2
    rows = []
    for exp in chosen_expiries:
        sub = day[day["expiration"] == exp]
        puts = sub[~sub["is_call"]].sort_values("moneyness")
        calls = sub[sub["is_call"]].sort_values("moneyness")
        for side, n_take in [(puts, half), (calls, strikes_per_expiry - half)]:
            if len(side) <= n_take:
                rows.append(side)
            else:
                idx = np.linspace(0, len(side) - 1, n_take).astype(int)
                rows.append(side.iloc[idx])
    return pd.concat(rows).reset_index(drop=True)


def heston_model_prices(params, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts):
    kappa, theta, xi, rho, v0 = params
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    process = ql.HestonProcess(r_ts, q_ts, spot_h, v0, kappa, theta, xi, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    prices = []
    for row in contracts.itertuples(index=False):
        exp_date = ql.Date(row.expiration.day, row.expiration.month, row.expiration.year)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if row.is_call else ql.Option.Put, row.strike)
        exercise = ql.EuropeanExercise(exp_date)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)
        try:
            prices.append(option.NPV())
        except RuntimeError:
            prices.append(np.nan)
    return np.array(prices)


def objective(x, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts, r_flat):
    kappa, theta, xi, rho, v0 = x
    model_prices = heston_model_prices(x, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts)

    err = 0.0
    n_valid = 0
    for price, row in zip(model_prices, contracts.itertuples(index=False)):
        if not np.isfinite(price):
            err += 10.0
            continue
        if price <= 0:
            # Found via investigate_phantom_penalty.py on 2013-03-15 (-7.9e-11) and
            # 2017-10-16 (-7.2e-6): AnalyticHestonEngine's numerical quadrature can return a
            # small NEGATIVE price for deep-OTM/short-dated contracts whose true price is
            # essentially zero (market mid $0.02-0.045 in both cases) -- noise, not a real
            # pricing failure. A flat +10 penalty here doesn't affect the argmin (it's
            # parameter-independent once triggered) but corrupts the reported objective
            # value into a misleading ~10x-normal number. Clamp near-zero noise (widened to
            # -1e-4, i.e. a tenth of a cent -- comfortably covers both observed magnitudes
            # while still catching genuinely broken/large-negative prices) to a tiny positive
            # floor so IV solving still works; only penalize truly negative prices.
            if price > -1e-4:
                price = 1e-8
            else:
                err += 10.0
                continue
        model_iv = bs_implied_vol(price, spot, row.strike, row.T, r_flat, Q_DIV, row.is_call)
        if not np.isfinite(model_iv):
            err += 10.0
            continue
        vega = max(row.vega, 1e-4)
        err += vega * (model_iv - row.implied_volatility) ** 2
        n_valid += 1

    feller_gap = xi**2 - 2 * kappa * theta
    penalty = FELLER_PENALTY_WEIGHT * max(0.0, feller_gap) ** 2
    return err + penalty, n_valid


def calibrate_one_date(calib_date, df, vas, vas_params, warm_start_x=None, verbose=False):
    """Shared calibration entry point -- used by the single-date design script, the warm-start
    test, and the batch run, so the grid/refine logic exists in exactly one place.

    If warm_start_x is given: skip the grid, single L-BFGS-B refine from that seed (fast path,
    ~6s, validated in the warm-start test to reach an equally good or better fit than cold-start).
    If warm_start_x is None: full 54-seed grid + refine top 3 (slow path, ~4min, needed for the
    first date of each vendor era or as the regime-break fallback).

    Returns a dict with fitted params, objective, convergence flag, and the contracts+fit-quality
    dataframe (None if verbose fit-quality computation wasn't requested by the caller).
    """
    spot = df.loc[df["date"] == calib_date, "spy_close"].iloc[0]
    r3mo = vas.loc[vas["date"] == calib_date, "r3mo"].iloc[0]
    eval_date, calendar, day_count, r_ts, q_ts = build_curves(calib_date, r3mo, vas_params)
    contracts = select_chain(df, calib_date)

    if len(contracts) < 10:
        return dict(date=calib_date, converged=False, error="too_few_contracts", n_contracts=len(contracts))

    atm_iv = contracts.loc[(contracts["moneyness"] - 1).abs().idxmin(), "implied_volatility"]
    v0_seed = atm_iv ** 2
    bounds_list = [BOUNDS["kappa"], BOUNDS["theta"], BOUNDS["xi"], BOUNDS["rho"], BOUNDS["v0"]]

    from scipy.optimize import minimize

    if warm_start_x is not None:
        res = minimize(
            lambda x: objective(x, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts, r3mo)[0],
            warm_start_x, method="L-BFGS-B", bounds=bounds_list, options=dict(maxiter=100, eps=1e-4))
        best = res
        mode = "warm"
    else:
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
        best = None
        for val, x0 in seeds[:3]:
            res = minimize(
                lambda x: objective(x, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts, r3mo)[0],
                x0, method="L-BFGS-B", bounds=bounds_list, options=dict(maxiter=100, eps=1e-4))
            if best is None or res.fun < best.fun:
                best = res
        mode = "cold"

    kappa, theta, xi, rho, v0 = best.x
    feller = 2 * kappa * theta - xi**2

    model_prices = heston_model_prices(best.x, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts)
    ivs = [bs_implied_vol(p, spot, row.strike, row.T, r3mo, Q_DIV, row.is_call)
           for p, row in zip(model_prices, contracts.itertuples(index=False))]
    iv_errors = np.array([abs(iv - row.implied_volatility) if np.isfinite(iv) else np.nan
                           for iv, row in zip(ivs, contracts.itertuples(index=False))])
    mean_iv_error = np.nanmean(iv_errors)

    if verbose:
        print(f"{calib_date.date()} [{mode}]: obj={best.fun:.6f} converged={best.success} "
              f"kappa={kappa:.3f} theta={theta:.4f} xi={xi:.3f} rho={rho:.3f} v0={v0:.4f} "
              f"feller={feller:.4f} mean|IVerr|={mean_iv_error*100:.2f}pp")

    return dict(date=calib_date, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
                feller=feller, objective=best.fun, converged=bool(best.success), mode=mode,
                n_contracts=len(contracts), mean_iv_error=mean_iv_error, spot=spot, r3mo=r3mo)


def main():
    calib_date = pd.Timestamp("2019-06-21")
    df = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_2019.parquet")
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]

    result = calibrate_one_date(calib_date, df, vas, vas_params, verbose=True)
    pd.DataFrame([result]).to_csv(f"{PROJ}\\results\\phase4_heston_params_test.csv", index=False)


if __name__ == "__main__":
    main()
