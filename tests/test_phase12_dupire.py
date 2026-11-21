"""Model-identity tests for the Phase 12 Dupire local vol (python tests/test_phase12_dupire.py).

T1 flat smile: SABR with nu ~ 0 and rho = 0 (beta = 1) is a flat implied vol alpha, so local vol
   must equal alpha everywhere and the local-vol PDE price must equal Black-Scholes.
T2 repricing: on real calibration dates, the local-vol price must reproduce the input surface
   (BS at the surface implied vol) within 0.5 implied-vol points on every scoring-grid contract.
T3 distinguishing: under SPY's negative skew, the local-vol delta must sit BELOW the sticky-strike
   BS delta at the money (the defining property that the localVol=False engine failed).
T4 the OLD call (localVol=False) must FAIL T3, proving the test can tell the two apart.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import QuantLib as ql
from phase4_heston_calibration import build_curves, PROJ, Q_DIV
from phase9_oos_scoring import build_scoring_grid
from phase9_oos_runner import tenor_rate
from pricers import bs_price
from phase12_dupire import DupireSurface, price_lv, effective_T

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    fails += 0 if ok else 1


vas = pd.read_parquet(os.path.join(PROJ, "data", "vasicek_discount_curve.parquet"))
vp = pd.read_csv(os.path.join(PROJ, "results", "phase2_vasicek_params.csv")).iloc[0]
sa = pd.read_parquet(os.path.join(PROJ, "data", "sabr_params_by_date_expiry.parquet"))
sa["date"] = pd.to_datetime(sa["date"])

# ---- T1 flat smile
t = pd.Timestamp("2020-01-15")
ev, _, _, rts, qts = build_curves(t, vas.loc[vas.date == t, "r3mo"].iloc[0], vp)
r_of_T = lambda T: tenor_rate(rts, T)
flat = [dict(T=T, alpha=0.2, rho=0.0, nu=1e-4) for T in (0.1, 0.3, 0.6, 1.0)]
S = 300.0
ds = DupireSurface(S, r_of_T, Q_DIV, flat)
core = ds.lv[(ds.strikes > S * 0.6) & (ds.strikes < S * 1.5)]
check("T1 flat smile -> local vol = 0.20", np.allclose(core, 0.2, atol=2e-3), f"range {core.min():.4f}-{core.max():.4f}")
surf = ds.ql_surface(ev)
errs = []
for K in (240, 280, 300, 320, 360):
    for T in (0.1, 0.4):
        lvp = price_lv(surf, S, rts, qts, ev, K, T, K >= S, mesh_vol=ds.mesh_vol())
        Te = effective_T(ev, T)
        bsp = bs_price(S, K, Te, r_of_T(Te), Q_DIV, 0.2, K >= S)
        errs.append(abs(lvp - bsp))
check("T1 flat smile -> LV PDE price = BS", max(errs) < 0.02, f"max abs err {max(errs):.4f}")

# ---- T2 / T3 / T4 on real dates
for ds_name in ["2020-01-15", "2020-03-16", "2021-06-15"]:
    t = pd.Timestamp(ds_name)
    df = pd.read_parquet(os.path.join(PROJ, "data", "clean", f"spy_options_clean_{t.year}.parquet"))
    ev, _, _, rts, qts = build_curves(t, vas.loc[vas.date == t, "r3mo"].iloc[0], vp)
    r_of_T = lambda T, rts=rts: tenor_rate(rts, T)
    S = float(df.loc[df.date == t, "spy_close"].iloc[0])
    slices = sa[sa.date == t].to_dict("records")
    dup = DupireSurface(S, r_of_T, Q_DIV, slices)
    surf = dup.ql_surface(ev)
    g = build_scoring_grid(df, t)
    e_surf, e_mid_lv, e_mid_ss, e_volpts = [], [], [], []
    for c in g.itertuples(index=False):
        Te = effective_T(ev, c.T)
        iv = dup.implied_vol(c.strike, Te)
        ss = bs_price(S, c.strike, Te, r_of_T(Te), Q_DIV, iv, c.is_call)
        lvp = price_lv(surf, S, rts, qts, ev, c.strike, c.T, c.is_call, mesh_vol=dup.mesh_vol())
        vega = S * np.exp(-Q_DIV * Te) * np.sqrt(Te) * 0.3989 * np.exp(-0.5 * ((np.log(S / c.strike) + (r_of_T(Te) - Q_DIV + 0.5 * iv * iv) * Te) / (iv * np.sqrt(Te))) ** 2)
        e_surf.append(lvp - ss); e_mid_lv.append(lvp - c.mid); e_mid_ss.append(ss - c.mid)
        e_volpts.append(abs(lvp - ss) / vega * 100)
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
    check(f"T2 {ds_name} LV reprices its input surface (every contract within 0.5 vol pts)", max(e_volpts) < 0.5,
          f"max {max(e_volpts):.2f} vol pts; $RMSE LV vs surface {rm(e_surf):.3f}; vs mid LV {rm(e_mid_lv):.3f}, surface {rm(e_mid_ss):.3f}; guarded nodes {dup.guarded_share:.1%}")
    # T3 delta at the money, 3 months
    K, T = round(S), effective_T(ev, 0.25)
    _, d_lv, _ = price_lv(surf, S, rts, qts, ev, K, T, True, greeks=True, mesh_vol=dup.mesh_vol())
    iv = dup.implied_vol(K, T)
    h = 0.005 * S
    d_ss = (bs_price(S + h, K, T, r_of_T(T), Q_DIV, iv, True) - bs_price(S - h, K, T, r_of_T(T), Q_DIV, iv, True)) / (2 * h)
    check(f"T3 {ds_name} LV delta < sticky-strike delta (ATM 3m)", d_lv < d_ss - 0.01, f"LV {d_lv:.3f} vs SS {d_ss:.3f}")

# ---- T4 the old engine call must fail the distinguishing test
from phase7_local_vol import build_local_vol_surface
t = pd.Timestamp("2020-01-15")
df = pd.read_parquet(os.path.join(PROJ, "data", "clean", "spy_options_clean_2020.parquet"))
lv_old, bv_old, _, S, _ = build_local_vol_surface(t, df, vas, vp)
ev, _, _, rts, qts = build_curves(t, vas.loc[vas.date == t, "r3mo"].iloc[0], vp)
proc = ql.GeneralizedBlackScholesProcess(ql.QuoteHandle(ql.SimpleQuote(S)), qts, rts,
                                         ql.BlackVolTermStructureHandle(bv_old), ql.LocalVolTermStructureHandle(lv_old))
exp_old = ev + ql.Period(91, ql.Days)
o = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, round(S)), ql.EuropeanExercise(exp_old))
o.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, 200, 200))   # the Phase 9 call, no localVol flag
Te = ql.Actual365Fixed().yearFraction(ev, exp_old)
iv = bv_old.blackVol(Te, round(S))
h = 0.005 * S
r = tenor_rate(rts, Te)
d_ss = (bs_price(S + h, round(S), Te, r, Q_DIV, iv, True) - bs_price(S - h, round(S), Te, r, Q_DIV, iv, True)) / (2 * h)
check("T4 old Phase 9 engine call fails T3 (its delta equals sticky strike)", abs(o.delta() - d_ss) < 0.01,
      f"old delta {o.delta():.3f} vs SS {d_ss:.3f}")

print(f"\n{fails} failure(s)")
sys.exit(1 if fails else 0)
