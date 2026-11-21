"""
Phase 11: model Greeks, daily delta-hedging test, and Greeks-based P&L explain (2020-2021).

Scope (decided 2026-11-20): the 24 monthly calibration dates in 2020 and 2021 only, one stressed
year (COVID crash) and one calm year. Same FROZEN parameters as Phase 9: each model is fitted at
calibration date t and never refitted while its contracts are tracked.

Panel: at each t, the Phase 9 scoring grid (5 moneyness x 3 tenor buckets, ~15 liquid contracts)
is selected ON t. Each contract is then followed on every trading day d = t, t+1, ... up to 21
trading days (about one month, i.e. until the next calibration). A contract leaves the panel when
its quote is missing or fails the Phase 9 quality filters, or when fewer than 7 calendar days
remain to expiry (expiry-week gamma would dominate every statistic).

Deltas/gammas on day d use ONLY day-d information (spot, curve, tenor) plus parameters fitted at
t <= d. No look-ahead.

Models (Heston deliberately left out of this phase):
  BS_market  Black-Scholes at the contract's OWN same-day vendor IV (practitioner sticky-strike
             delta). The benchmark every model is measured against, as in Hull & White (2017).
  BS_calib   Black-Scholes at the single sigma fitted at t (flat vol, frozen).
  SABR       nearest calibrated expiry slice (as Phase 9); delta by spot bump with the forward
             moving and SABR parameters fixed (so the smile moves with the forward).
  Merton     frozen (sigma, lambda, mu_j, sigma_j); delta by spot bump.
  LocalVol   TRUE Dupire (Phase 12 surface from the frozen SABR slices, localVol=True), the same
             function of (S, calendar time) as at t; delta and gamma from the FD engine.

Part 2 hedge error for one day, long 1 option, short delta shares, cash account at r:
  e = (V1 - V0) - delta0 * (S1 - S0 + q*S0*dt) - r0 * (V0 - delta0*S0) * dt
with V = market mid, dt = calendar days / 365. Metric: Gain = 1 - SSE(model) / SSE(BS_market),
plus the unhedged variance reduction for scale.

Part 3 P&L explain (risk-based, the shape of a bank's daily P&L explain / FRTB PLA):
  predicted = Delta*dS + 0.5*Gamma*dS^2 + Vega*dIV + Theta*dt  (BS Greeks at day-0 vendor IV,
  dIV = the contract's own vendor-IV change). Reported step by step. Unexplained share =
  mean|actual - predicted| / mean|actual|; Spearman rank correlation of actual vs predicted.
"""
import os
import time
import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.stats import norm, spearmanr

from phase4_heston_batch import monthly_calibration_dates
from phase4_heston_calibration import build_curves, PROJ, Q_DIV
from phase9_oos_scoring import build_scoring_grid, arbitrage_ok
from phase9_oos_runner import tenor_rate, load_all_params
from pricers import bs_price
from sabr import sabr_iv, BETA
from merton import merton_price_batch
from phase12_dupire import DupireSurface, price_lv

YEARS = (2020, 2021)
TRACK_DAYS = 21
MIN_DAYS_TO_EXPIRY = 7
BUMP = 0.005
PDE_GRID = 200
OUT_ROWS = f"{PROJ}\\results\\phase11_contract_days.csv"
OUT_SUMMARY = f"{PROJ}\\results\\phase11_summary.txt"
MODELS = ["BS_market", "BS_calib", "SABR", "Merton", "LocalVol"]
# History: Phase 7/9 never passed localVol=True, so the old 'LocalVol' was a frozen sticky-strike
# surface. Phase 11 first ran without LocalVol; it now uses the Phase 12 Dupire build.
USE_LOCALVOL = True   # Phase 12 Dupire (phase12_dupire.py), localVol=True, frozen in calendar time


def bs_greeks(S, K, T, r, q, sig, is_call):
    """Analytic BS delta, gamma, vega (per 1.00 vol), theta (per year, calendar)."""
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * sq)
    d2 = d1 - sig * sq
    eq, er = np.exp(-q * T), np.exp(-r * T)
    delta = eq * norm.cdf(d1) if is_call else -eq * norm.cdf(-d1)
    gamma = eq * norm.pdf(d1) / (S * sig * sq)
    vega = S * eq * norm.pdf(d1) * sq
    if is_call:
        theta = -S * eq * norm.pdf(d1) * sig / (2 * sq) + q * S * eq * norm.cdf(d1) - r * K * er * norm.cdf(d2)
    else:
        theta = -S * eq * norm.pdf(d1) * sig / (2 * sq) - q * S * eq * norm.cdf(-d1) + r * K * er * norm.cdf(-d2)
    return delta, gamma, vega, theta


def bump_greeks(price_fn, S):
    h = BUMP * S
    up, mid, dn = price_fn(S + h), price_fn(S), price_fn(S - h)
    return mid, (up - dn) / (2 * h), (up - 2 * mid + dn) / (h * h)


def main(limit=None):
    t0 = time.time()
    cv, he, sa, me, lv_bad = load_all_params()
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]
    data = {}
    for y in (*YEARS, YEARS[-1] + 1):
        fp = f"{PROJ}\\data\\clean\\spy_options_clean_{y}.parquet"
        if os.path.exists(fp):
            data[y] = pd.read_parquet(fp)
    full = pd.concat(data.values(), ignore_index=True)
    ok = full["has_valid_quote"] & full["iv_quality_ok"] & full["mid"].notna()
    full = full[ok]
    full = full[arbitrage_ok(full)]
    full = full[(full["implied_volatility"] > 0.01) & (full["implied_volatility"] < 3.0)]
    full = full.set_index(["contractID", "date"]).sort_index()
    trade_dates = np.array(sorted(pd.concat(data.values())["date"].unique()))

    cal_dates = [d for d in monthly_calibration_dates() if d.year in YEARS]
    if limit:
        cal_dates = cal_dates[:limit]
    rows = []
    for t in cal_dates:
        df_t = data[t.year]
        grid = build_scoring_grid(df_t, t)
        p_cv = cv[cv["date"] == t].iloc[0]
        p_me = me[me["date"] == t].iloc[0]
        p_sa = sa[sa["date"] == t]
        lv_ok = USE_LOCALVOL
        if lv_ok:
            try:
                _, _, _, rts_t, _ = build_curves(t, vas.loc[vas["date"] == t, "r3mo"].iloc[0], vas_params)
                spot_t = float(df_t.loc[df_t["date"] == t, "spy_close"].iloc[0])
                dup = DupireSurface(spot_t, lambda T, rr=rts_t: tenor_rate(rr, T), Q_DIV, p_sa.to_dict("records"))
                mesh_v = dup.mesh_vol()
            except Exception as ex:
                print(f"{t.date()} Dupire build failed: {ex}", flush=True)
                lv_ok = False
        days = trade_dates[trade_dates >= np.datetime64(t)][: TRACK_DAYS + 1]

        for d in days:
            d = pd.Timestamp(d)
            r3 = vas.loc[vas["date"] == d, "r3mo"]
            if r3.empty:
                continue
            eval_d, _, _, rts, qts = build_curves(d, r3.iloc[0], vas_params)
            if lv_ok:
                lv_surf_d = dup.ql_surface(eval_d, shift=(d - t).days / 365.0)
            for c in grid.itertuples(index=False):
                key = (c.contractID, d)
                if key not in full.index:
                    continue
                q = full.loc[key]
                if isinstance(q, pd.DataFrame):
                    q = q.iloc[0]
                T = (pd.Timestamp(q["expiration"]) - d).days / 365.0
                if T * 365 < MIN_DAYS_TO_EXPIRY:
                    continue
                S, K, is_call = float(q["spy_close"]), float(c.strike), bool(c.is_call)
                r = tenor_rate(rts, T)
                rec = dict(calib_date=t, date=d, contractID=c.contractID, strike=K, is_call=is_call,
                           tenor_bucket=float(c.T), grid_moneyness=float(c.grid_moneyness),
                           S=S, T=T, r=r, mid=float(q["mid"]), iv=float(q["implied_volatility"]))
                dm, gm, vm, tm = bs_greeks(S, K, T, r, Q_DIV, rec["iv"], is_call)
                rec.update(delta_BS_market=dm, gamma_BS_market=gm, vega_mkt=vm, theta_mkt=tm,
                           price_BS_market=bs_price(S, K, T, r, Q_DIV, rec["iv"], is_call))
                dc, gc, _, _ = bs_greeks(S, K, T, r, Q_DIV, p_cv["sigma"], is_call)
                rec.update(delta_BS_calib=dc, gamma_BS_calib=gc,
                           price_BS_calib=bs_price(S, K, T, r, Q_DIV, p_cv["sigma"], is_call))
                # SABR: nearest calibrated slice to the CURRENT tenor, forward moves with spot
                s = p_sa.loc[(p_sa["T"] - T).abs().idxmin()]

                def f_sabr(x):
                    F = x * np.exp((r - Q_DIV) * T)
                    iv = sabr_iv(K, F, T, s["alpha"], s["rho"], s["nu"], BETA)
                    return bs_price(x, K, T, r, Q_DIV, iv, is_call)
                try:
                    p, dl, gm_ = bump_greeks(f_sabr, S)
                    rec.update(price_SABR=p, delta_SABR=dl, gamma_SABR=gm_)
                except RuntimeError:
                    rec.update(price_SABR=np.nan, delta_SABR=np.nan, gamma_SABR=np.nan)

                def f_mer(x):
                    return float(merton_price_batch(x, [K], [T], r, Q_DIV, p_me["sigma"], p_me["lam"],
                                                    p_me["mu_j"], p_me["sigma_j"], [is_call])[0])
                p, dl, gm_ = bump_greeks(f_mer, S)
                rec.update(price_Merton=p, delta_Merton=dl, gamma_Merton=gm_)
                if lv_ok:
                    try:
                        p, dl, gm_ = price_lv(lv_surf_d, S, rts, qts, eval_d, K, T, is_call, grid=PDE_GRID,
                                              greeks=True, mesh_vol=mesh_v)
                        rec.update(price_LocalVol=p, delta_LocalVol=dl, gamma_LocalVol=gm_)
                    except RuntimeError:
                        pass
                rows.append(rec)
        print(f"{t.date()} done, rows={len(rows)}, {(time.time()-t0)/60:.1f} min", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROWS if not limit else OUT_ROWS.replace(".csv", "_test.csv"), index=False)
    print(f"saved {len(out)} contract-days to {OUT_ROWS}", flush=True)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
