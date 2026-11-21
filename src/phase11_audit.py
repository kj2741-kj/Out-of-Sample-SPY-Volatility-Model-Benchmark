"""Phase 11 logic-audit checks (run after phase11_analysis.py)."""
import numpy as np
import pandas as pd
from phase4_heston_calibration import PROJ
from phase11_analysis import make_pairs, MODELS

d = pd.read_csv(f"{PROJ}\\results\\phase11_contract_days.csv", parse_dates=["calib_date", "date"])
p = make_pairs(d)
p = p[p["year"].isin([2020, 2021])]
out = []

# 1. Minimum-variance check (Hull-White 2017 style): regress the BS_market hedge error on dS.
#    A negative slope means the variance-minimising delta is BELOW the BS sticky-strike delta.
for lab, x in [("All", p), ("2020", p[p.year == 2020]), ("2021", p[p.year == 2021])] + \
              [(s, p[p.side == s]) for s in ["OTM puts", "ATM", "OTM calls"]]:
    b = np.polyfit(x["dS"], x["e_BS_market"], 1)[0]
    # residual variance cut if the hedge used delta + b
    e_mv = x["e_BS_market"] - b * x["dS"]
    out.append(f"MV slope {lab:10s}: extra delta {b:+.4f} per share; SSE cut vs BS_market {1 - (e_mv**2).sum() / (x['e_BS_market']**2).sum():.3%} (in-sample, n={len(x)})")

# 2. Delta gap sign/size of the frozen smile models vs BS_calib (both frozen at t)
for m in ["SABR", "Merton", "LocalVol"]:
    out.append(f"mean delta {m} - BS_calib: {(p['delta_' + m] - p['delta_BS_calib']).mean():+.4f}")

# 3. SABR rho at the -0.999 bound
sa = pd.read_parquet(f"{PROJ}\\data\\sabr_params_by_date_expiry.parquet")
sa["date"] = pd.to_datetime(sa["date"])
s = sa[sa["date"].dt.year.isin([2020, 2021])]
out.append(f"SABR slices 2020-21: {len(s)}, rho at -0.999 bound: {(s['rho'] <= -0.998).mean():.1%}")
cd_bound = s.groupby("date")["rho"].apply(lambda r: (r <= -0.998).mean())
bad = set(cd_bound[cd_bound > 0.5].index)
x = p[~p["calib_date"].isin(bad)]
g = 1 - (x["e_SABR"] ** 2).sum() / (x["e_BS_market"] ** 2).sum()
out.append(f"SABR gain vs BS_market excluding {len(bad)} calib dates with rho mostly at bound: {g:+.3f} (n={len(x)})")

# 4. Staleness: does the SABR/Merton penalty grow with days since calibration?
p["age"] = (p["date"] - p["calib_date"]).dt.days
for lo, hi in [(0, 7), (8, 15), (16, 35)]:
    x = p[(p.age >= lo) & (p.age <= hi)]
    gs = {m: 1 - (x[f"e_{m}"] ** 2).sum() / (x["e_BS_market"] ** 2).sum() for m in ["BS_calib", "SABR", "Merton", "LocalVol"]}
    out.append(f"age {lo:2d}-{hi:2d}d n={len(x)}: " + ", ".join(f"{k} {v:+.3f}" for k, v in gs.items()))

# 5. Bid-ask noise scale vs hedge error
hs = (d["mid"] * 0).astype(float)
out.append(f"hedge RMSE BS_market {np.sqrt((p['e_BS_market']**2).mean()):.3f}; mean |dV| {p['dV'].abs().mean():.3f}")
# 6. Blend: hedge delta = BS_market + w*(LocalVol - BS_market); least-squares w (in sample)
for lab, x in [("All", p), ("2020", p[p.year == 2020]), ("2021", p[p.year == 2021]), ("Stress", p[p.stress]), ("Ex-stress", p[~p.stress])]:
    x = x.dropna(subset=["e_LocalVol"])
    z = (x["delta_LocalVol"] - x["delta_BS_market"]) * x["dS"]
    w = (x["e_BS_market"] * z).sum() / (z * z).sum()
    e = x["e_BS_market"] - w * z
    out.append(f"blend {lab:9s}: weight on local-vol delta {w:.2f}; SSE cut vs BS_market {1 - (e**2).sum() / (x['e_BS_market']**2).sum():.1%} (in-sample, n={len(x)})")
print("\n".join(out))
open(f"{PROJ}\\results\\phase11_audit.txt", "w").write("\n".join(out))
