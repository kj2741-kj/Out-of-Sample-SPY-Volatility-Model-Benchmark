"""
Phase 9 runner: score all 9 models out-of-sample.

Per calibration date t: load each model's params (fitted at t, Phases 4-8 + phase9_prep_constvol),
then price a fixed grid of real contracts at t+h for h in {5,10,21} trading days using those
FROZEN params, and compare to actual market quotes at t+h.

Model list and how each is priced OOS (params from t, market state from t+h):
 1 BSM              constant sigma_t, spot_{t+h}
 2 Black-76         same sigma_t on the forward F = S_{t+h} e^{(r-q)T}
 3 GBM-MC           same sigma_t, Monte Carlo (20k paths -- enough given we report RMSE over
                    thousands of scorings; MC noise averages out and is ~1e-2 per price)
 4 CRR-American     same sigma_t, American exercise (the only model here pricing early exercise)
 5 Heston           (kappa,theta,xi,rho,v0)_t via QuantLib AnalyticHestonEngine
 6 SABR             (alpha,rho,nu) of the calibrated expiry NEAREST the scoring contract's tenor
 7 AH-surface       local-vol PDE under the Andreasen-Huge surface built at t
 8 LocalVol         local-vol PDE under the SABR-smoothed surface built at t
 9 Merton           (sigma,lambda,mu_j,sigma_j)_t, closed-form Poisson-BS sum

Note on 7 vs 8: both are local-vol PDE models, differing ONLY in how the surface was
constructed (raw market quotes vs SABR-smoothed). That's a deliberate, informative pairing.
They are priced by PDE at the new spot rather than by reading an implied vol off the t-surface,
because the latter would make model 8 mathematically collapse into model 6 (SABR) -- its surface
IS SABR -- and the comparison would be circular.
"""
import time
import numpy as np
import pandas as pd
import QuantLib as ql

from phase4_heston_batch import monthly_calibration_dates
from phase4_heston_calibration import build_curves, PROJ, Q_DIV
from phase9_oos_scoring import build_scoring_grid, score_metrics, HORIZONS
from pricers import bs_price, black76_price, gbm_mc_price, crr_binomial_price
from sabr import sabr_iv, BETA
from merton import merton_price_batch

PDE_GRID = 100
MC_PATHS = 20_000
PROGRESS_CSV = f"{PROJ}\\results\\phase9_oos_scores.csv"


def tenor_rate(r_ts, T):
    """AUDIT FIX (2026-08-09): tenor-matched continuously-compounded zero rate off the Vasicek
    curve. Previously the scalar-r models (BSM family, SABR, Merton) were scored with a FLAT
    r3mo while the curve-based models (Heston, AH, LocalVol) used the full term structure --
    i.e. models were being compared under different rate assumptions. Measured gap is 0.7-5.6bp
    at scored tenors (~$0.02-0.10 per option); small, but the SABR-vs-Heston RMSE gap is only
    0.013, so it was large enough to flip a ranking. Now every model sees the same curve."""
    return r_ts.zeroRate(max(T, 1e-6), ql.Continuous).rate()


def price_bsm_family(grid, spot, r_ts, sigma):
    bsm, b76, mc, crr = [], [], [], []
    for row in grid.itertuples(index=False):
        r = tenor_rate(r_ts, row.T)
        F = spot * np.exp((r - Q_DIV) * row.T)
        bsm.append(bs_price(spot, row.strike, row.T, r, Q_DIV, sigma, row.is_call))
        b76.append(black76_price(F, row.strike, row.T, r, sigma, row.is_call))
        mc.append(gbm_mc_price(spot, row.strike, row.T, r, Q_DIV, sigma, row.is_call,
                                n_paths=MC_PATHS, seed=7)[0])
        crr.append(crr_binomial_price(spot, row.strike, row.T, r, Q_DIV, sigma, row.is_call,
                                       is_american=True, n_steps=200))
    return bsm, b76, mc, crr


def price_heston(grid, spot, r_ts, q_ts, eval_date, hp):
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    process = ql.HestonProcess(r_ts, q_ts, spot_h, hp["v0"], hp["kappa"], hp["theta"], hp["xi"], hp["rho"])
    engine = ql.AnalyticHestonEngine(ql.HestonModel(process))
    out = []
    for row in grid.itertuples(index=False):
        exp_date = eval_date + ql.Period(int(round(row.T * 365)), ql.Days)
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if row.is_call else ql.Option.Put, row.strike),
                                ql.EuropeanExercise(exp_date))
        opt.setPricingEngine(engine)
        try:
            p = opt.NPV()
            out.append(p if np.isfinite(p) and p > -1e-4 else np.nan)
        except RuntimeError:
            out.append(np.nan)
    return out


def price_sabr(grid, spot, r_ts, sabr_slices):
    """sabr_slices: DataFrame of that date's per-expiry fits. Use the slice whose calibrated T
    is nearest the scoring contract's tenor (the standard way to use a per-expiry model at a
    tenor between calibrated pillars)."""
    out = []
    for row in grid.itertuples(index=False):
        i = (sabr_slices["T"] - row.T).abs().idxmin()
        s = sabr_slices.loc[i]
        r = tenor_rate(r_ts, row.T)
        F = spot * np.exp((r - Q_DIV) * row.T)
        try:
            iv = sabr_iv(row.strike, F, row.T, s["alpha"], s["rho"], s["nu"], BETA)
            out.append(bs_price(spot, row.strike, row.T, r, Q_DIV, iv, row.is_call) if iv > 0 else np.nan)
        except RuntimeError:
            out.append(np.nan)
    return out


def price_localvol_pde(grid, spot, r_ts, q_ts, eval_date, black_vol_handle, local_vol_handle):
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    process = ql.GeneralizedBlackScholesProcess(spot_h, q_ts, r_ts, black_vol_handle, local_vol_handle)
    engine = ql.FdBlackScholesVanillaEngine(process, PDE_GRID, PDE_GRID)
    out = []
    for row in grid.itertuples(index=False):
        exp_date = eval_date + ql.Period(int(round(row.T * 365)), ql.Days)
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if row.is_call else ql.Option.Put, row.strike),
                                ql.EuropeanExercise(exp_date))
        opt.setPricingEngine(engine)
        try:
            p = opt.NPV()
            out.append(p if np.isfinite(p) else np.nan)
        except RuntimeError:
            out.append(np.nan)
    return out


def price_merton(grid, spot, r_ts, mp):
    r = np.array([tenor_rate(r_ts, t) for t in grid["T"].values])
    return merton_price_batch(spot, grid["strike"].values, grid["T"].values, r, Q_DIV,
                               mp["sigma"], mp["lam"], mp["mu_j"], mp["sigma_j"],
                               grid["is_call"].values)


def load_all_params():
    cv = pd.read_parquet(f"{PROJ}\\data\\constvol_params_by_date.parquet")
    he = pd.read_parquet(f"{PROJ}\\data\\heston_params_by_date.parquet")
    sa = pd.read_parquet(f"{PROJ}\\data\\sabr_params_by_date_expiry.parquet")
    try:
        me = pd.read_parquet(f"{PROJ}\\data\\merton_params_by_date.parquet")
    except FileNotFoundError:
        # Merton's final parquet is only written when its batch COMPLETES; fall back to the
        # in-progress checkpoint CSV so Phase 9 can be developed/run on however many dates are
        # done so far (and so a partially-complete Merton run never blocks the whole harness).
        me = pd.read_csv(f"{PROJ}\\results\\phase8_merton_batch_progress.csv")
        print(f"NOTE: merton parquet absent, using in-progress CSV ({len(me)} dates)", flush=True)
    for d in (cv, he, sa, me):
        d["date"] = pd.to_datetime(d["date"])
    lv_val = pd.read_csv(f"{PROJ}\\results\\phase7_local_vol_batch_validation.csv")
    lv_val["date"] = pd.to_datetime(lv_val["date"])
    lv_bad = set(lv_val.loc[lv_val["error"].notna(), "date"]) if "error" in lv_val else set()
    return cv, he, sa, me, lv_bad


def main():
    dates = monthly_calibration_dates()
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]
    cv, he, sa, me, lv_bad = load_all_params()

    rows = []
    try:
        prior = pd.read_csv(PROGRESS_CSV)
        prior["calib_date"] = pd.to_datetime(prior["calib_date"])
        done = set(prior["calib_date"].unique())
        dates = [d for d in dates if d not in done]
        rows = prior.to_dict("records")
        print(f"RESUMING: {len(done)} calib dates scored, {len(dates)} remaining", flush=True)
    except FileNotFoundError:
        print("No checkpoint, starting fresh", flush=True)

    from phase6_svi_local_vol import calibrate_andreasen_huge
    from phase7_local_vol import build_local_vol_surface

    year_cache = {}
    t_start = time.time()
    for i, t in enumerate(dates):
        # AUDIT FIX (2026-08-09): the t+45d lookahead resolves to year 2026 for Dec-2025
        # calibration dates, and that partition does not exist (data ends 2025-12-31). The
        # unguarded read raised FileNotFoundError and killed the run -- which is why the first
        # Phase 9 pass reported 188/192 dates rather than 192. Those 4 dates were LOST, not
        # legitimately skipped. Guard the preload; dates with no future partition simply have
        # fewer scorable horizons, handled below by the `len(fut) < h` check.
        import os
        for y in {t.year, (t + pd.Timedelta(days=45)).year}:
            if y not in year_cache:
                fp = f"{PROJ}\\data\\clean\\spy_options_clean_{y}.parquet"
                if os.path.exists(fp):
                    year_cache[y] = pd.read_parquet(fp)
        if len(year_cache) > 3:
            for k in list(year_cache)[:-3]:
                del year_cache[k]
        df_t = year_cache[t.year]

        p_cv = cv[cv["date"] == t]
        p_he = he[he["date"] == t]
        p_me = me[me["date"] == t]
        p_sa = sa[sa["date"] == t]
        if p_cv.empty or p_he.empty or p_me.empty or p_sa.empty:
            print(f"{t.date()}: missing params (cv={len(p_cv)} he={len(p_he)} me={len(p_me)} sa={len(p_sa)}), skipping", flush=True)
            continue
        p_cv, p_he, p_me = p_cv.iloc[0], p_he.iloc[0], p_me.iloc[0]

        # Surfaces built ONCE per calibration date, reused across all horizons
        ah_ok, lv_ok = True, (t not in lv_bad)
        try:
            ah_surface, _, _ = calibrate_andreasen_huge(t, df_t, vas, vas_params)
            ah_lv_handle = ql.LocalVolTermStructureHandle(ql.AndreasenHugeLocalVolAdapter(ah_surface)) \
                if hasattr(ql, "AndreasenHugeLocalVolAdapter") else None
            ah_bv_handle = ql.BlackVolTermStructureHandle(ql.AndreasenHugeVolatilityAdapter(ah_surface)) \
                if hasattr(ql, "AndreasenHugeVolatilityAdapter") else None
            if ah_lv_handle is None or ah_bv_handle is None:
                ah_ok = False
        except Exception:
            ah_ok = False
        if lv_ok:
            try:
                lv_surface, lv_black_var, _, _, _ = build_local_vol_surface(t, df_t, vas, vas_params)
                lv_bv_handle = ql.BlackVolTermStructureHandle(lv_black_var)
                lv_lv_handle = ql.LocalVolTermStructureHandle(lv_surface)
            except Exception:
                lv_ok = False

        all_dates_t = np.array(sorted(year_cache[t.year]["date"].unique()))
        for h in HORIZONS:
            fut = all_dates_t[all_dates_t > np.datetime64(t)]
            if len(fut) < h:
                continue
            score_date = pd.Timestamp(fut[h - 1])
            df_s = year_cache.get(score_date.year)
            if df_s is None:
                continue
            grid = build_scoring_grid(df_s, score_date)
            if len(grid) == 0:
                continue
            spot_s = grid["spy_close"].iloc[0]
            r_s = vas.loc[vas["date"] == score_date, "r3mo"]
            if r_s.empty:
                continue
            r_s = r_s.iloc[0]
            eval_s, cal_s, dc_s, rts_s, qts_s = build_curves(score_date, r_s, vas_params)

            preds = {}
            b, b76, mc, crr = price_bsm_family(grid, spot_s, rts_s, p_cv["sigma"])
            preds["BSM"], preds["Black76"], preds["GBM_MC"], preds["CRR_American"] = b, b76, mc, crr
            preds["Heston"] = price_heston(grid, spot_s, rts_s, qts_s, eval_s, p_he)
            preds["SABR"] = price_sabr(grid, spot_s, rts_s, p_sa)
            preds["Merton"] = price_merton(grid, spot_s, rts_s, p_me)
            if ah_ok:
                try:
                    preds["AH_Surface"] = price_localvol_pde(grid, spot_s, rts_s, qts_s, eval_s, ah_bv_handle, ah_lv_handle)
                except Exception:
                    pass
            if lv_ok:
                try:
                    preds["LocalVol"] = price_localvol_pde(grid, spot_s, rts_s, qts_s, eval_s, lv_bv_handle, lv_lv_handle)
                except Exception:
                    pass

            for model, prices in preds.items():
                m = score_metrics(prices, grid)
                m.update(calib_date=t, score_date=score_date, horizon=h, model=model)
                rows.append(m)

        if (i + 1) % 4 == 0:
            pd.DataFrame(rows).to_csv(PROGRESS_CSV, index=False)
            print(f"[{i+1}/{len(dates)}] {t.date()} elapsed={(time.time()-t_start)/60:.1f}min", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(PROGRESS_CSV, index=False)
    print(f"\nDone. {len(out)} model-scorings, {out['calib_date'].nunique()} calib dates, "
          f"{(time.time()-t_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
