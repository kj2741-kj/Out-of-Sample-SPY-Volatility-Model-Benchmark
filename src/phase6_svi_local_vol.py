"""
Phase 6 design: arbitrage-free implied-vol surface construction. Design pivot from the original
plan (documented in docs/phase4_handoff.md / phase4_heston_design_notes.md as an explicitly
flagged option): use QuantLib's AndreasenHugeVolatilityInterpl instead of hand-rolled raw-SVI.

Why the pivot: Phase 5 demonstrated a stark reliability gap in this environment between
QuantLib-native closed-form/production methods (SABR: 960 fits, 18 seconds, zero failures) and
hand-rolled numerical approaches (Heston needed ~19 background-kill recoveries). Hand-rolling raw
SVI's arbitrage constraints (Durrleman conditions for full butterfly-arbitrage-freedom, calendar
constraints across expiries) is exactly the kind of fiddly numerical work that caused Heston's
pain. Andreasen-Huge (2011) is a well-established industry method that calibrates a discrete
local volatility surface guaranteed arbitrage-free by construction (via a one-step implicit
finite-difference calibration) -- it solves Phase 6's actual goal (arbitrage-free surface) AND
directly produces what Phase 7 needs (a local vol surface), in one tested implementation, instead
of two separate hand-built numerical stages (fit SVI, then finite-difference Dupire on top of it).

This departs from the letter of the original phase plan (raw SVI + separate Dupire derivation)
but not the spirit (an arbitrage-free surface feeding a local-vol model) -- documented here
rather than silently substituted.
"""
import numpy as np
import pandas as pd
import QuantLib as ql
from phase4_heston_calibration import build_curves, select_chain, PROJ


def calibrate_andreasen_huge(calib_date, df, vas, vas_params, verbose=False):
    spot = df.loc[df["date"] == calib_date, "spy_close"].iloc[0]
    r3mo = vas.loc[vas["date"] == calib_date, "r3mo"].iloc[0]
    eval_date, calendar, day_count, r_ts, q_ts = build_curves(calib_date, r3mo, vas_params)
    contracts = select_chain(df, calib_date)

    calib_set = ql.CalibrationSet()
    for row in contracts.itertuples(index=False):
        if not np.isfinite(row.implied_volatility) or row.implied_volatility <= 0:
            continue
        exp_date = ql.Date(row.expiration.day, row.expiration.month, row.expiration.year)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if row.is_call else ql.Option.Put, row.strike)
        exercise = ql.EuropeanExercise(exp_date)
        option = ql.VanillaOption(payoff, exercise)
        quote = ql.SimpleQuote(row.implied_volatility)  # AndreasenHugeVolatilityInterpl's
        # CalibrationSet quote is IMPLIED VOL, not price -- confirmed against QuantLib's own
        # test-suite source (andreasenhugevolatilityinterpl.cpp), not assumed.
        calib_set.push_back((option, quote))

    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    surface = ql.AndreasenHugeVolatilityInterpl(calib_set, spot_h, r_ts, q_ts)

    calib_err_tuple = surface.calibrationError()
    calib_err = calib_err_tuple.first()
    if verbose:
        print(f"  {calib_date.date()}: {len(calib_set)} contracts, calibration error tuple="
              f"({calib_err_tuple.first()}, {calib_err_tuple.second()}, {calib_err_tuple.third()})")

    # Fit-quality check: reprice every calibration contract off the fitted surface, convert back
    # to IV (more interpretable / comparable to Heston & SABR's IV-space fit-quality numbers)
    from pricers import bs_implied_vol
    rows = []
    for row in contracts.itertuples(index=False):
        if not np.isfinite(row.implied_volatility) or row.implied_volatility <= 0:
            continue
        model_price = surface.optionPrice(row.T, row.strike, ql.Option.Call if row.is_call else ql.Option.Put)
        model_iv = bs_implied_vol(model_price, spot, row.strike, row.T, r3mo, 0.014, row.is_call)
        rows.append(dict(strike=row.strike, T=row.T, is_call=row.is_call,
                          market_iv=row.implied_volatility, model_price=model_price, model_iv=model_iv,
                          iv_err=(model_iv - row.implied_volatility) if np.isfinite(model_iv) else np.nan))
    fit_df = pd.DataFrame(rows)
    return surface, fit_df, calib_err


def main():
    calib_date = pd.Timestamp("2019-06-21")
    df = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_2019.parquet")
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]

    surface, fit_df, calib_err = calibrate_andreasen_huge(calib_date, df, vas, vas_params, verbose=True)
    print(f"Repricing check (IV space): mean|IV err|={fit_df['iv_err'].abs().mean()*100:.3f}pp, "
          f"max|IV err|={fit_df['iv_err'].abs().max()*100:.3f}pp")

    # Local vol spot-check: query local vol at a few (strike, T) points
    print("\nLocal vol surface sample points:")
    for K, T in [(spot_atm := df.loc[df["date"]==calib_date, "spy_close"].iloc[0], 0.25),
                 (spot_atm * 0.9, 0.25), (spot_atm * 1.1, 0.25), (spot_atm, 1.0)]:
        lv = surface.localVol(T, K)
        print(f"  K={K:.1f} T={T}: local vol = {lv*100:.2f}%")

    fit_df.to_csv(f"{PROJ}\\results\\phase6_andreasenhuge_single_date_fit.csv", index=False)


if __name__ == "__main__":
    main()
