"""Quantify the three cross-phase consistency issues found in the audit, so severity is
measured rather than guessed."""
import numpy as np, pandas as pd
from phase4_heston_calibration import build_curves, select_chain, PROJ, Q_DIV
from sabr import sabr_iv, BETA
from pricers import bs_price

vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]
sa = pd.read_parquet(f"{PROJ}\\data\\sabr_params_by_date_expiry.parquet")
sa["date"] = pd.to_datetime(sa["date"])

print("=" * 70)
print("ISSUE 1: SABR forward mismatch (calibrated F=spot, scored F=spot*e^((r-q)T))")
print("=" * 70)
rows = []
for ds in ["2011-03-15", "2016-06-15", "2023-06-15"]:
    d = pd.Timestamp(ds)
    df = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{d.year}.parquet")
    spot = df.loc[df["date"] == d, "spy_close"].iloc[0]
    r = vas.loc[vas["date"] == d, "r3mo"].iloc[0]
    sl = sa[sa["date"] == d]
    for _, s in sl.iterrows():
        T = s["T"]
        F_wrong = spot * np.exp((r - Q_DIV) * T)   # what phase9 used
        F_right = spot                              # what phase5 calibrated with
        for m in [0.95, 1.00, 1.05]:
            K = spot * m
            iv_w = sabr_iv(K, F_wrong, T, s["alpha"], s["rho"], s["nu"], BETA)
            iv_r = sabr_iv(K, F_right, T, s["alpha"], s["rho"], s["nu"], BETA)
            p_w = bs_price(spot, K, T, r, Q_DIV, iv_w, m >= 1)
            p_r = bs_price(spot, K, T, r, Q_DIV, iv_r, m >= 1)
            rows.append(dict(date=ds, T=round(T, 3), m=m, r=round(r, 4),
                             iv_diff_pp=(iv_w - iv_r) * 100, price_diff=p_w - p_r))
a = pd.DataFrame(rows)
print(f"IV impact  : mean |diff| = {a['iv_diff_pp'].abs().mean():.3f}pp, max = {a['iv_diff_pp'].abs().max():.3f}pp")
print(f"Price impact: mean |diff| = ${a['price_diff'].abs().mean():.4f}, max = ${a['price_diff'].abs().max():.4f}")
print(f"vs SABR's reported OOS RMSE of 0.937 -> mean impact is {a['price_diff'].abs().mean()/0.937*100:.1f}% of RMSE")
print("\nsign flips across rate regimes (r-q<0 early, >0 late):")
print(a.groupby("date")["price_diff"].mean().round(4).to_string())

print("\n" + "=" * 70)
print("ISSUE 2/3: flat r3mo vs Vasicek term-structure zero rate, by tenor")
print("=" * 70)
rows = []
for ds in ["2011-03-15", "2016-06-15", "2023-06-15"]:
    d = pd.Timestamp(ds)
    r3 = vas.loc[vas["date"] == d, "r3mo"].iloc[0]
    ed, cal, dc, r_ts, q_ts = build_curves(d, r3, vas_params)
    for T in [1/12, 3/12, 6/12, 1.0]:
        z = r_ts.zeroRate(T, ql_comp := __import__("QuantLib").Continuous).rate()
        rows.append(dict(date=ds, T=round(T, 3), flat_r3mo=r3, curve_zero=z, diff_bp=(z - r3) * 1e4))
b = pd.DataFrame(rows)
print(b.round(5).to_string(index=False))
print(f"\nmax |rate diff| = {b['diff_bp'].abs().max():.1f} bp")

# translate the worst rate diff into a price error on a representative ATM option
worst = b.loc[b["diff_bp"].abs().idxmax()]
S, K, T = 400.0, 400.0, float(worst["T"])
p1 = bs_price(S, K, T, float(worst["flat_r3mo"]), Q_DIV, 0.18, True)
p2 = bs_price(S, K, T, float(worst["curve_zero"]), Q_DIV, 0.18, True)
print(f"worst-case ATM call price impact (S=K=400, T={T:.2f}, vol=18%): ${abs(p1-p2):.4f}")
print(f"  -> vs typical model RMSE ~0.58-1.71, that is {abs(p1-p2)/0.937*100:.1f}% of SABR's RMSE")
