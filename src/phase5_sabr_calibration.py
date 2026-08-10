"""
Phase 5 design: SABR per-expiry calibration, tested on the same 2019-06-21 date used for
Heston's design phase (Phase 4), reusing the SAME select_chain() contract selection so results
are directly comparable across models later (Phase 9 scoring wants apples-to-apples chains).
"""
import numpy as np
import pandas as pd
from phase4_heston_calibration import select_chain, PROJ, Q_DIV
from sabr import fit_sabr_slice


def calibrate_sabr_one_date(calib_date, df, vas=None, verbose=False):
    contracts = select_chain(df, calib_date)
    spot = df.loc[df["date"] == calib_date, "spy_close"].iloc[0]

    # AUDIT FIX (2026-08-09): originally used forward = spot with a note that beta would absorb
    # the scale. That was wrong twice over: (a) SABR's smile is a function of ln(F/K), so using
    # spot instead of the true forward shifts effective moneyness; (b) far worse, Phase 9 scored
    # with F = S*exp((r-q)T) while these params were fitted with F = spot -- a calibrate-with-one-
    # definition / apply-with-another mismatch. Measured impact was $0.58 mean price error (62%
    # of SABR's reported OOS RMSE), max $3.69, and it FLIPPED SIGN across rate regimes (r-q<0 in
    # the ZIRP years, >0 recently). Now uses the true dividend-adjusted forward everywhere.
    if vas is None:
        vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    r3mo = vas.loc[vas["date"] == calib_date, "r3mo"].iloc[0]

    results = []
    for exp, group in contracts.groupby("expiration"):
        group = group.sort_values("strike")
        T = group["T"].iloc[0]
        forward = spot * np.exp((r3mo - Q_DIV) * T)
        fit = fit_sabr_slice(group["strike"].values, forward, T, group["implied_volatility"].values, group["vega"].values)
        fit.update(date=calib_date, expiration=exp, T=T, forward=forward)
        results.append(fit)
        if verbose:
            print(f"  {exp.date()} (T={T:.3f}, n={fit['n_strikes']}): alpha={fit['alpha']:.4f} "
                  f"rho={fit['rho']:.4f} nu={fit['nu']:.4f} obj={fit['objective']:.6f} "
                  f"conv={fit['converged']} mean|err|={fit['mean_iv_error']*100:.3f}pp "
                  f"max|err|={fit['max_iv_error']*100:.3f}pp")
    return pd.DataFrame(results)


def main():
    calib_date = pd.Timestamp("2019-06-21")
    df = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_2019.parquet")
    print(f"SABR calibration, {calib_date.date()}, beta fixed at 1.0")
    out = calibrate_sabr_one_date(calib_date, df, verbose=True)
    print(f"\nOverall: mean|IV err|={out['mean_iv_error'].mean()*100:.3f}pp, "
          f"max|IV err|={out['max_iv_error'].max()*100:.3f}pp, "
          f"all converged={out['converged'].all()}")
    out.to_csv(f"{PROJ}\\results\\phase5_sabr_single_date_fit.csv", index=False)


if __name__ == "__main__":
    main()
