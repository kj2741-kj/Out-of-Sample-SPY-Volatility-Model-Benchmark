"""Cross-phase consistency audit of Phases 0-3, the counterpart to audit_consistency.py
(which covered Phases 4-10).

The question is the same one: where can two phases disagree about a shared quantity without
either of them raising an error? Seven checks:

  1. Rate SOURCE      Phase 0 used yfinance ^IRX; Phase 2+ use FRED DGS3MO. Same nominal
                      quantity, two vendors, and ^IRX is quoted on a DISCOUNT basis.
  2. Rate SHAPE       Phase 0 used one flat rate for every tenor; Phases 4-10 use the Vasicek
                      term structure. Does re-running Phase 0 on the curve improve the match
                      to vendor IV -- i.e. is "flat rate proxy" really an explanation for the
                      residual gap, as the Phase 0 notes claim?
  3. EXERCISE STYLE   The Phase 0 notes attribute the residual IV gap partly to SPY options
                      being American. Section 6.4 of the final report later concluded the
                      early-exercise premium is negligible. Both cannot be right; this measures
                      it on the actual Phase 0 sample rather than on a synthetic grid.
  4. T CONVENTION     Phase 1 computes T = calendar_days/365 in pandas. Phases 4-7 hand dates
                      to QuantLib and let a day counter decide. Do they agree?
  5. VALIDATION       Phase 3 cross-validated the pricers on a synthetic grid. Do the later
     ENVELOPE         phases actually operate inside that grid, or do they extrapolate?
  6. COMPOUNDING      FRED DGS3MO is a coupon-equivalent yield; every pricer here consumes it
                      as a continuously-compounded rate.
  7. CALIB INVERSION  Phase 4's objective prices with the Vasicek CURVE but inverts the
                      resulting price back to IV with the FLAT r3mo -- the same split-rate
                      issue that audit_consistency.py found in Phase 9 SCORING, which was only
                      ever fixed on the scoring side.

Read-only. Recomputes from committed artifacts; writes nothing except its own report.
"""
import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.optimize import brentq

from pricers import bs_price, bs_implied_vol, crr_binomial_price
from phase4_heston_calibration import vasicek_zero_rate, build_curves, select_chain, PROJ, Q_DIV

SAMPLE_PER_YEAR = 120      # stratified sample for checks 2-3
CRR_SUBSAMPLE = 400        # American repricing is the expensive one
CRR_STEPS = 200
SEED = 42

PHASE3_GRID = dict(moneyness=(0.70, 1.30), T=(1 / 12, 3.0), sigma=(0.10, 0.30), r=(0.01, 0.05))


def hdr(n, title):
    print("\n" + "=" * 78)
    print(f"CHECK {n}: {title}")
    print("=" * 78)


def load_common():
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas["date"] = pd.to_datetime(vas["date"])
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]
    return vas, vas_params


def build_sample(vas):
    """Rebuild a Phase-0-style stratified sample, but from the Phase 1 CLEAN partitions so the
    sample carries spot/mid/T/flags directly and every check below sees identical rows."""
    rng = np.random.RandomState(SEED)
    frames = []
    for year in range(2010, 2026):
        try:
            df = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{year}.parquet")
        except FileNotFoundError:
            continue
        df = df[df["has_valid_quote"] & df["iv_quality_ok"] & df["not_expired_yet"]]
        df = df.dropna(subset=["spy_close", "mid", "implied_volatility", "vega"])
        # same region the project actually prices in
        df = df[(df["moneyness"].between(0.85, 1.15)) & (df["T"].between(0.08, 2.0))]
        if len(df) == 0:
            continue
        take = min(len(df), SAMPLE_PER_YEAR)
        frames.append(df.sample(take, random_state=rng.randint(1 << 30)))
    s = pd.concat(frames, ignore_index=True)
    s = s.merge(vas[["date", "r3mo"]], on="date", how="left").dropna(subset=["r3mo"])
    return s


# ---------------------------------------------------------------- check 1
def check_rate_source(vas):
    hdr(1, "Rate SOURCE -- Phase 0 (^IRX, discount basis) vs Phase 2+ (FRED DGS3MO)")
    irx = pd.read_parquet(f"{PROJ}\\data\\riskfree_irx.parquet")
    irx["date"] = pd.to_datetime(irx["date"])
    m = irx.merge(vas[["date", "y3mo"]], on="date", how="inner").dropna()
    m["diff_bp"] = (m["irx_yield_pct"] - m["y3mo"]) * 100

    print(f"overlapping trading days: {len(m):,}  ({m['date'].min().date()} to {m['date'].max().date()})")
    print(f"mean  IRX - DGS3MO = {m['diff_bp'].mean():+.2f} bp")
    print(f"mean |IRX - DGS3MO|= {m['diff_bp'].abs().mean():.2f} bp   max = {m['diff_bp'].abs().max():.1f} bp")
    print("\nby era (the two vendor eras Phase 1 keeps separate):")
    m["era"] = np.where(m["date"] < "2014-01-01", "OptionsDX 2010-13", "AlphaVantage 2014-25")
    print(m.groupby("era")["diff_bp"].agg(["mean", "std", "min", "max"]).round(2).to_string())

    # translate the typical gap into an IV error on a representative 3-month ATM contract
    S = K = 400.0
    T, sig = 0.25, 0.18
    worst = m.loc[m["diff_bp"].abs().idxmax()]
    r_a, r_b = worst["irx_yield_pct"] / 100, worst["y3mo"] / 100
    p_a = bs_price(S, K, T, r_a, Q_DIV, sig, True)
    p_b = bs_price(S, K, T, r_b, Q_DIV, sig, True)
    iv_a = bs_implied_vol(p_a, S, K, T, r_b, Q_DIV, True)
    print(f"\nworst day {pd.Timestamp(worst['date']).date()}: IRX={r_a*100:.3f}% vs DGS3MO={r_b*100:.3f}%")
    print(f"  -> 3m ATM call price differs by ${abs(p_a-p_b):.4f}; back-solved IV differs by "
          f"{abs(iv_a-sig)*100:.3f}pp")
    return m


# ---------------------------------------------------------------- checks 2 & 3
def check_rate_shape_and_exercise(s, vas_params):
    hdr(2, "Rate SHAPE -- flat 3m rate (Phase 0) vs Vasicek term structure (Phase 2+)")
    a, b, sg = vas_params["a"], vas_params["b"], vas_params["sigma"]

    s = s.copy()
    s["r_curve"] = [vasicek_zero_rate(a, b, sg, r0, T) for r0, T in zip(s["r3mo"], s["T"])]
    s["rate_diff_bp"] = (s["r_curve"] - s["r3mo"]) * 1e4

    s["iv_flat"] = [bs_implied_vol(m_, S, K, T, r, Q_DIV, c)
                    for m_, S, K, T, r, c in zip(s["mid"], s["spy_close"], s["strike"],
                                                 s["T"], s["r3mo"], s["is_call"])]
    s["iv_curve"] = [bs_implied_vol(m_, S, K, T, r, Q_DIV, c)
                     for m_, S, K, T, r, c in zip(s["mid"], s["spy_close"], s["strike"],
                                                  s["T"], s["r_curve"], s["is_call"])]
    v = s.dropna(subset=["iv_flat", "iv_curve"]).copy()
    print(f"sample: {len(v):,} rows, {v['date'].nunique():,} dates, "
          f"{v['date'].dt.year.min()}-{v['date'].dt.year.max()}")
    print(f"rate difference (curve - flat): mean {v['rate_diff_bp'].mean():+.2f} bp, "
          f"max |{v['rate_diff_bp'].abs().max():.1f}| bp")
    print(f"resulting IV difference        : mean |{(v['iv_curve']-v['iv_flat']).abs().mean()*100:.4f}|pp, "
          f"max |{(v['iv_curve']-v['iv_flat']).abs().max()*100:.4f}|pp")

    print("\nmatch to VENDOR implied vol -- does using the real curve improve it?")
    for lbl, col in [("flat r3mo  (Phase 0 style)", "iv_flat"), ("Vasicek curve (Phase 2+)", "iv_curve")]:
        d = v[col] - v["implied_volatility"]
        print(f"  {lbl:28s} corr={v[col].corr(v['implied_volatility']):.4f}  "
              f"mean_diff={d.mean()*100:+.3f}pp  mean|diff|={d.abs().mean()*100:.3f}pp")

    hdr(3, "EXERCISE STYLE -- does American exercise explain Phase 0's residual IV gap?")
    sub = v.sample(min(CRR_SUBSAMPLE, len(v)), random_state=SEED).copy()

    def amer_iv(mid, S, K, T, r, is_call):
        # NOTE: the bracket must start above the vol where CRR's risk-neutral probability
        # leaves [0,1] -- p = (e^{(r-q)dt} - d)/(u - d) blows up as sigma -> 0, and
        # crr_binomial_price raises rather than returning garbage. A 1e-4 lower bracket
        # therefore makes brentq throw on nearly every row instead of solving it.
        lo, hi = 0.02, 3.0
        try:
            f_lo = crr_binomial_price(S, K, T, r, Q_DIV, lo, is_call, True, CRR_STEPS) - mid
            f_hi = crr_binomial_price(S, K, T, r, Q_DIV, hi, is_call, True, CRR_STEPS) - mid
        except ValueError:
            return np.nan
        if f_lo * f_hi > 0:
            return np.nan
        f = lambda sig: crr_binomial_price(S, K, T, r, Q_DIV, sig, is_call, True, CRR_STEPS) - mid
        try:
            return brentq(f, lo, hi, xtol=1e-6, maxiter=100)
        except (ValueError, RuntimeError):
            return np.nan

    sub["iv_amer"] = [amer_iv(m_, S, K, T, r, c)
                      for m_, S, K, T, r, c in zip(sub["mid"], sub["spy_close"], sub["strike"],
                                                   sub["T"], sub["r_curve"], sub["is_call"])]
    sub = sub.dropna(subset=["iv_amer"])
    prem = (sub["iv_curve"] - sub["iv_amer"]) * 100   # European IV overstates if premium is real
    gap = (sub["iv_curve"] - sub["implied_volatility"]).abs() * 100

    print(f"subsample: {len(sub):,} rows repriced with a {CRR_STEPS}-step CRR American tree")
    print(f"European-minus-American IV      : mean {prem.mean():+.4f}pp, "
          f"median {prem.median():+.4f}pp, max {prem.abs().max():.4f}pp")
    print(f"residual gap vs VENDOR IV        : mean {gap.mean():.4f}pp")
    share = prem.abs().mean() / gap.mean() * 100 if gap.mean() else float("nan")
    print(f"-> early exercise explains ~{share:.1f}% of the residual gap")
    print("\nby moneyness (early exercise, if anywhere, shows up in deep ITM puts):")
    sub["m_bucket"] = pd.cut(sub["moneyness"], [0.84, 0.95, 1.0, 1.05, 1.16])
    sub["ee_pp"] = prem
    print(sub.groupby("m_bucket", observed=True)["ee_pp"].agg(["count", "mean", "max"]).round(4).to_string())

    hdr("3b", "So what DOES drive Phase 0's residual gap?")
    v = v.copy()
    v["gap_pp"] = (v["iv_curve"] - v["implied_volatility"]) * 100
    print("signed gap by moneyness (a sign flip across the strikes points at a spot/forward")
    print("mismatch; a flat but noisy gap points at quote timing or a different price field):")
    v["m_bucket"] = pd.cut(v["moneyness"], [0.84, 0.95, 1.0, 1.05, 1.16])
    print(v.groupby("m_bucket", observed=True)["gap_pp"]
          .agg(["count", "mean", "median", "std"]).round(3).to_string())
    print("\nsigned gap by tenor:")
    v["t_bucket"] = pd.cut(v["T"], [0.08, 0.25, 0.5, 1.0, 2.0])
    print(v.groupby("t_bucket", observed=True)["gap_pp"]
          .agg(["count", "mean", "median", "std"]).round(3).to_string())
    print("\nsigned gap by vendor era:")
    v["era"] = np.where(v["date"] < "2014-01-01", "OptionsDX 2010-13", "AlphaVantage 2014-25")
    print(v.groupby("era")["gap_pp"].agg(["count", "mean", "median", "std"]).round(3).to_string())

    print("\ndividend assumption: what flat q best reproduces the vendor's IV?")
    best = None
    for q_try in np.arange(0.0, 0.0301, 0.0025):
        iv_q = np.array([bs_implied_vol(m_, S, K, T, r, q_try, c)
                         for m_, S, K, T, r, c in zip(v["mid"], v["spy_close"], v["strike"],
                                                      v["T"], v["r_curve"], v["is_call"])])
        ok = np.isfinite(iv_q)
        mad = np.abs(iv_q[ok] - v["implied_volatility"].values[ok]).mean() * 100
        flag = ""
        if best is None or mad < best[1]:
            best, flag = (q_try, mad), ""
        print(f"  q={q_try*100:4.2f}%  mean|IV - vendor IV| = {mad:.3f}pp{flag}")
    print(f"-> best flat q = {best[0]*100:.2f}% at {best[1]:.3f}pp; project uses "
          f"{Q_DIV*100:.1f}% giving {v['gap_pp'].abs().mean():.3f}pp")
    return v


# ---------------------------------------------------------------- check 4
def check_t_convention(s):
    hdr(4, "T CONVENTION -- Phase 1 pandas (days/365) vs QuantLib day counter (Phases 4-7)")
    dc = ql.Actual365Fixed()
    sub = s.sample(min(2000, len(s)), random_state=SEED)
    diffs = []
    for row in sub.itertuples(index=False):
        d0 = ql.Date(row.date.day, row.date.month, row.date.year)
        d1 = ql.Date(row.expiration.day, row.expiration.month, row.expiration.year)
        diffs.append(dc.yearFraction(d0, d1) - row.T)
    diffs = np.array(diffs)
    print(f"compared {len(diffs):,} contracts")
    print(f"max |QuantLib T - Phase 1 T| = {np.abs(diffs).max():.3e} years "
          f"({np.abs(diffs).max()*365*24*3600:.2e} seconds)")
    print("VERDICT: conventions agree exactly." if np.abs(diffs).max() < 1e-12
          else "VERDICT: MISMATCH -- investigate.")

    # the curve pillars are built as integer day offsets; check that round-tripping a tenor
    # through ql.Period(int(round(t*365)), Days) doesn't shift the rate a model actually sees
    err = max(abs(int(round(t * 365)) / 365 - t) for t in [1/12, 3/12, 0.5, 1.0, 1.5, 2.0, 3.0])
    print(f"curve-pillar rounding (build_curves): max tenor shift {err*365:.2f} days "
          f"({err:.5f} yr) -- {'immaterial' if err < 0.002 else 'CHECK'}")


# ---------------------------------------------------------------- check 5
def check_validation_envelope(s):
    hdr(5, "VALIDATION ENVELOPE -- does live usage stay inside Phase 3's cross-validated grid?")
    cv = pd.read_csv(f"{PROJ}\\results\\phase9_constvol_params.csv")
    print("Phase 3 synthetic grid: " + ", ".join(f"{k} {v[0]}-{v[1]}" for k, v in PHASE3_GRID.items()))
    print()
    rows = []
    for name, series, lo, hi in [
        ("moneyness (calibration chain)", s["moneyness"], *PHASE3_GRID["moneyness"]),
        ("T years    (calibration chain)", s["T"], *PHASE3_GRID["T"]),
        ("sigma      (fitted const vol)", cv["sigma"], *PHASE3_GRID["sigma"]),
        ("r          (3m rate, all dates)", cv["r3mo"], *PHASE3_GRID["r"]),
    ]:
        out = ((series < lo) | (series > hi)).mean() * 100
        rows.append(dict(quantity=name, actual_min=series.min(), actual_max=series.max(),
                         grid_lo=lo, grid_hi=hi, pct_outside=out))
    r = pd.DataFrame(rows)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    bad = r[r["pct_outside"] > 0]
    if len(bad):
        print("\nEXTRAPOLATION: the pricers were never cross-validated at these values:")
        for _, x in bad.iterrows():
            print(f"  - {x['quantity'].strip()}: {x['pct_outside']:.1f}% of live usage outside "
                  f"[{x['grid_lo']}, {x['grid_hi']}] (actual range "
                  f"{x['actual_min']:.4f} to {x['actual_max']:.4f})")
    return r


# ---------------------------------------------------------------- check 6
def check_compounding(vas):
    hdr(6, "COMPOUNDING -- FRED DGS3MO is coupon-equivalent; pricers consume it as continuous")
    y = vas["y3mo"].dropna() / 100
    # coupon-equivalent (semiannual) -> continuously compounded
    cont = 2 * np.log1p(y / 2)
    d = (y - cont) * 1e4
    print(f"{len(y):,} daily observations, y3mo range {y.min()*100:.3f}% - {y.max()*100:.3f}%")
    print(f"quoted-minus-continuous: mean {d.mean():.2f} bp, max {d.max():.2f} bp "
          f"(at the {y.max()*100:.2f}% peak)")
    S = K = 400.0
    for T in [0.25, 0.5, 1.0]:
        p1 = bs_price(S, K, T, y.max(), Q_DIV, 0.18, True)
        p2 = bs_price(S, K, T, 2 * np.log1p(y.max() / 2), Q_DIV, 0.18, True)
        print(f"  worst-case ATM call T={T:.2f}: price impact ${abs(p1-p2):.4f}")
    print("Applied identically to every model, so the RANKING is unaffected; absolute price")
    print("levels carry it, same category as the flat dividend proxy.")


# ---------------------------------------------------------------- check 7
def check_calibration_inversion(vas, vas_params):
    hdr(7, "CALIB INVERSION -- Phase 4 prices on the CURVE but inverts to IV on FLAT r3mo")
    a, b, sg = vas_params["a"], vas_params["b"], vas_params["sigma"]
    rows = []
    for ds in ["2011-03-15", "2016-06-15", "2023-06-15"]:
        d = pd.Timestamp(ds)
        df = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{d.year}.parquet")
        spot = df.loc[df["date"] == d, "spy_close"].iloc[0]
        r3 = vas.loc[vas["date"] == d, "r3mo"].iloc[0]
        chain = select_chain(df, d)
        for row in chain.itertuples(index=False):
            rc = vasicek_zero_rate(a, b, sg, r3, row.T)
            # a price generated under the curve, then read back as IV under flat r3mo
            px = bs_price(spot, row.strike, row.T, rc, Q_DIV, row.implied_volatility, row.is_call)
            iv_flat = bs_implied_vol(px, spot, row.strike, row.T, r3, Q_DIV, row.is_call)
            if np.isfinite(iv_flat):
                rows.append(dict(date=ds, T=row.T, iv_bias_pp=(iv_flat - row.implied_volatility) * 100))
    z = pd.DataFrame(rows)
    print(f"{len(z):,} calibration contracts across 3 representative dates")
    print(f"IV inversion bias: mean {z['iv_bias_pp'].mean():+.4f}pp, "
          f"mean |bias| {z['iv_bias_pp'].abs().mean():.4f}pp, max |bias| {z['iv_bias_pp'].abs().max():.4f}pp")
    print("\nby date (sign tracks the rate regime, as in audit_consistency.py issue 1):")
    print(z.groupby("date")["iv_bias_pp"].agg(["mean", "max"]).round(4).to_string())
    print("\nContext: calibration fit errors are 0.29-4.03pp, so this is a systematic but")
    print("second-order distortion of the objective, applied identically to every model.")


def main():
    vas, vas_params = load_common()
    check_rate_source(vas)
    s = build_sample(vas)
    check_rate_shape_and_exercise(s, vas_params)
    check_t_convention(s)
    check_validation_envelope(s)
    check_compounding(vas)
    check_calibration_inversion(vas, vas_params)
    print("\n" + "=" * 78)
    print("Audit complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
