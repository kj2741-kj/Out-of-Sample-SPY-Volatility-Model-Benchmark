"""
Phase 7 design: Local Volatility (Dupire), built differently than Phase 6's handoff suggested
retrying Andreasen-Huge with more data (already shown not to help -- see
docs/phase6_svi_completion_notes.md).

Instead: build a smooth ql.BlackVarianceSurface from Phase 5's already-validated SABR fits
(evaluated on a common strike grid per expiry, not raw market points), then wrap in
ql.LocalVolSurface for Dupire's formula via QuantLib's own standard, robust finite-difference
differentiation. Rationale: SABR slices are smooth by construction (Hagan's closed-form formula
doesn't produce the calendar-arbitrage noise that raw market quotes across different
expiries/vendors can have), so feeding SABR-smoothed IVs into the standard
BlackVarianceSurface->LocalVolSurface pipeline should sidestep the instability Andreasen-Huge hit
when fed raw market points directly. This also ties Phase 5 and Phase 7 together coherently
rather than duplicating a second raw-data smile-fitting stage.
"""
import numpy as np
import pandas as pd
import QuantLib as ql
from phase4_heston_calibration import build_curves, select_chain, PROJ
from phase5_sabr_calibration import calibrate_sabr_one_date
from sabr import sabr_iv, BETA

STRIKE_GRID_MONEYNESS = np.linspace(0.85, 1.15, 21)


def build_local_vol_surface(calib_date, df, vas, vas_params, verbose=False):
    spot = df.loc[df["date"] == calib_date, "spy_close"].iloc[0]
    r3mo = vas.loc[vas["date"] == calib_date, "r3mo"].iloc[0]
    eval_date, calendar, day_count, r_ts, q_ts = build_curves(calib_date, r3mo, vas_params)

    sabr_fits = calibrate_sabr_one_date(calib_date, df, vas=vas, verbose=False)
    sabr_fits = sabr_fits.sort_values("T")

    strikes = STRIKE_GRID_MONEYNESS * spot
    ql_strikes = list(strikes)
    ql_dates = [eval_date + ql.Period(int(round(row.T * 365)), ql.Days) for row in sabr_fits.itertuples()]

    # Raw SABR-implied vols, per (strike, expiry). SABR fits each expiry fully independently
    # (Phase 5's whole design point) so nothing enforces cross-expiry consistency -- on ~11% of
    # dates this produces real calendar arbitrage (total variance decreasing in T at some
    # strike), which QuantLib's BlackVarianceSurface correctly rejects outright (loud, specific
    # exception naming the strike/time -- a far better failure mode than Andreasen-Huge's silent
    # garbage output on the same underlying problem). Fix: enforce monotonic total variance
    # w=iv^2*T across expiries at each strike (standard practical remedy, not a hack) before
    # building the surface.
    T_values = [row.T for row in sabr_fits.itertuples()]
    iv_matrix = np.zeros((len(ql_strikes), len(ql_dates)))
    for j, row in enumerate(sabr_fits.itertuples()):
        for i, K in enumerate(ql_strikes):
            iv_matrix[i, j] = sabr_iv(K, row.forward, row.T, row.alpha, row.rho, row.nu, BETA)

    var_matrix = iv_matrix ** 2 * np.array(T_values)[np.newaxis, :]
    for i in range(var_matrix.shape[0]):
        for j in range(1, var_matrix.shape[1]):
            if var_matrix[i, j] < var_matrix[i, j - 1]:
                var_matrix[i, j] = var_matrix[i, j - 1]
    iv_matrix = np.sqrt(var_matrix / np.array(T_values)[np.newaxis, :])

    vol_matrix = ql.Matrix(len(ql_strikes), len(ql_dates))
    for j in range(len(ql_dates)):
        for i in range(len(ql_strikes)):
            vol_matrix[i][j] = iv_matrix[i, j]

    black_var_surface = ql.BlackVarianceSurface(eval_date, calendar, ql_dates, ql_strikes,
                                                 vol_matrix, day_count)
    # NOT setInterpolation("bicubic"): cubic splines can overshoot BETWEEN grid points even when
    # the grid points themselves are monotonic in T (confirmed -- the monotonic-variance fix on
    # grid points alone didn't stop the exceptions; the errors cited sub-day time deltas, e.g.
    # "between time 0.134247 and 0.134347" ~ 1 day apart, meaning the exception was firing on
    # spline-interpolated intermediate points, not the raw grid). Default (bilinear) doesn't
    # overshoot, so a monotonic grid stays monotonic everywhere after interpolation too.
    black_vol_handle = ql.BlackVolTermStructureHandle(black_var_surface)
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    local_vol_surface = ql.LocalVolSurface(black_vol_handle, r_ts, q_ts, spot_handle)

    if verbose:
        print(f"  {calib_date.date()}: built from {len(sabr_fits)} SABR-fitted expiries x "
              f"{len(ql_strikes)} strikes")
    return local_vol_surface, black_var_surface, sabr_fits, spot, r3mo


def local_vol_price(local_vol_surface, black_var_surface, spot, r_ts, q_ts, strike, T, is_call, eval_date):
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    black_vol_handle = ql.BlackVolTermStructureHandle(black_var_surface)
    local_vol_handle = ql.LocalVolTermStructureHandle(local_vol_surface)
    process = ql.GeneralizedBlackScholesProcess(spot_handle, q_ts, r_ts, black_vol_handle, local_vol_handle)
    engine = ql.FdBlackScholesVanillaEngine(process, 200, 200)

    exp_date = eval_date + ql.Period(int(round(T * 365)), ql.Days)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike)
    exercise = ql.EuropeanExercise(exp_date)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return option.NPV()


def main():
    calib_date = pd.Timestamp("2019-06-21")
    df = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_2019.parquet")
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]

    local_vol, black_var, sabr_fits, spot, r3mo = build_local_vol_surface(
        calib_date, df, vas, vas_params, verbose=True)

    print(f"\nspot={spot:.2f}")
    print("\nLocal vol at various (T, K), same points that blew up with Andreasen-Huge:")
    for T in sorted(sabr_fits["T"].unique()):
        for K in [spot * 0.95, spot, spot * 1.05]:
            try:
                lv = local_vol.localVol(T, K)
                print(f"  T={T:.3f} K={K:.1f}: localVol={lv*100:.2f}%")
            except RuntimeError as e:
                print(f"  T={T:.3f} K={K:.1f}: ERROR {e}")

    eval_date, calendar, day_count, r_ts, q_ts = build_curves(calib_date, r3mo, vas_params)
    from pricers import bs_price, bs_implied_vol
    print("\nLocal vol pricer vs BSM (at market IV) vs raw market mid, for the actual calibration contracts:")
    contracts = select_chain(df, calib_date)
    errs = []
    for row in contracts.sample(10, random_state=1).itertuples(index=False):
        lv_price = local_vol_price(local_vol, black_var, spot, r_ts, q_ts, row.strike, row.T, row.is_call, eval_date)
        bs_at_market_iv = bs_price(spot, row.strike, row.T, r3mo, 0.014, row.implied_volatility, row.is_call)
        lv_iv = bs_implied_vol(lv_price, spot, row.strike, row.T, r3mo, 0.014, row.is_call)
        err = abs(lv_iv - row.implied_volatility) if lv_iv == lv_iv else float("nan")
        errs.append(err)
        print(f"  K={row.strike:.0f} T={row.T:.3f} {'C' if row.is_call else 'P'}: "
              f"LV_price={lv_price:.4f} BS@mktIV={bs_at_market_iv:.4f} market_mid={row.mid:.4f} "
              f"LV_iv={lv_iv*100:.2f}% market_iv={row.implied_volatility*100:.2f}%")
    import numpy as np
    print(f"\nmean|LV_iv - market_iv| = {np.nanmean(errs)*100:.3f}pp")


if __name__ == "__main__":
    main()
