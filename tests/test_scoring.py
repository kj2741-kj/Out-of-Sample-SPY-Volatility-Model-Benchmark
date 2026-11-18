#!/usr/bin/env python
# coding: utf-8
"""
Checks for the Phase 9 scoring filters and the Phase 10 statistics (audit 2026-11-18).
Every check compares against a hand-computed value or an independent construction.

    python tests/test_scoring.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phase9_oos_scoring import arbitrage_ok, monotone_ok  # noqa: E402
from phase10_analysis import pooled_rmse, like_for_like, RANKED, PROJ  # noqa: E402
from pricers import bs_price, bs_implied_vol  # noqa: E402

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def chain(S=100.0, T=0.25, r=0.03, q=0.014, vol=0.2, strikes=np.arange(80, 121, 2.5)):
    rows = []
    for K in strikes:
        for is_call in (True, False):
            p = bs_price(S, K, T, r, q, vol + 0.1 * max(0, (100 - K) / 100), is_call)
            rows.append(dict(expiration="E1", strike=float(K), is_call=is_call, T=T, spy_close=S,
                             mid=p, bid=p - 0.02, ask=p + 0.02))
    return pd.DataFrame(rows)


print("\n[1] American no-arbitrage bounds")
d = pd.DataFrame([
    dict(spy_close=100.0, strike=90.0, is_call=True, mid=10.20, bid=10.1, ask=10.3),   # valid ITM call
    dict(spy_close=100.0, strike=90.0, is_call=True, mid=9.00, bid=8.9, ask=9.1),      # below intrinsic 10
    dict(spy_close=100.0, strike=90.0, is_call=True, mid=9.95, bid=9.8, ask=10.1),     # within half spread
    dict(spy_close=100.0, strike=110.0, is_call=False, mid=0.30, bid=0.2, ask=0.4),    # put worth >= 10
    dict(spy_close=100.0, strike=50.0, is_call=True, mid=101.0, bid=100.9, ask=101.1), # call above spot
    dict(spy_close=100.0, strike=95.0, is_call=False, mid=1.10, bid=1.0, ask=1.2),     # valid OTM put
])
ok = arbitrage_ok(d).tolist()
check("keeps valid quotes and quotes within half a spread of intrinsic", ok[0] and ok[2] and ok[5])
check("drops mid below intrinsic, ITM put priced like OTM, call above spot",
      (not ok[1]) and (not ok[3]) and (not ok[4]), str(ok))

print("\n[2] Vertical-spread monotonicity")
c = chain()
check("a clean Black-Scholes chain loses nothing", monotone_ok(c).all())
bad = c.copy()
i = bad.index[(bad.strike == 90.0) & (~bad.is_call)][0]
call_px = bad.loc[(bad.strike == 90.0) & bad.is_call, "mid"].iloc[0]
bad.loc[i, ["mid", "bid", "ask"]] = [call_px, call_px - 0.05, call_px + 0.05]
k = monotone_ok(bad)
check("a put row carrying the call price is removed, and only that row",
      (not k[i]) and k.drop(i).all(), f"removed {int((~k).sum())}")
bad2 = c.copy()
rows = bad2.index[(bad2.strike.isin([85.0, 87.5, 90.0, 92.5])) & (~bad2.is_call)]
for r_ in rows:
    K = bad2.loc[r_, "strike"]
    px = bad2.loc[(bad2.strike == K) & bad2.is_call, "mid"].iloc[0]
    bad2.loc[r_, ["mid", "bid", "ask"]] = [px, px - 0.05, px + 0.05]
k2 = monotone_ok(bad2)
check("a cluster of four mislabelled puts is removed, every good row kept",
      (~k2[rows]).all() and k2.drop(rows).all(), f"removed {int((~k2).sum())}")
bad3 = c.copy()
j = bad3.index[(bad3.strike == 115.0) & bad3.is_call][0]
bad3.loc[j, ["mid", "bid", "ask"]] = [12.0, 11.9, 12.1]
k3 = monotone_ok(bad3)
check("an OTM call priced like an ITM put is removed (call direction)", (not k3[j]) and k3.drop(j).all())

print("\n[3] Statistics")
e = np.array([1.0, -2.0, 2.0, np.nan])
check("pooled RMSE = sqrt(mean squared error), NaN ignored", np.isclose(pooled_rmse(e), np.sqrt(3.0)))
cells = [np.array([1.0, 1.0, 1.0, 1.0]), np.array([4.0])]
mean_of_cells = np.average([pooled_rmse(x) for x in cells], weights=[4, 1])
check("mean of per-cell RMSE differs from pooled RMSE (why the headline changed)",
      not np.isclose(mean_of_cells, pooled_rmse(np.concatenate(cells))),
      f"{mean_of_cells:.3f} vs {pooled_rmse(np.concatenate(cells)):.3f}")
df = pd.DataFrame({m: [1.0, 2.0, 3.0] for m in RANKED})
df.loc[1, "LocalVol"] = np.nan
check("like-for-like keeps only contracts every ranked model priced", list(like_for_like(df).index) == [0, 2])

print("\n[4] Pricer round trip (oracle reference sanity)")
p = bs_price(400, 380, 0.3, 0.04, 0.014, 0.23, False)
iv = bs_implied_vol(p, 400, 380, 0.3, 0.04, 0.014, False)
check("Black-Scholes at the IV implied from a price reproduces that price",
      np.isclose(bs_price(400, 380, 0.3, 0.04, 0.014, iv, False), p, atol=1e-6))

print("\n[5] Results file")
path = os.path.join(PROJ, "results", "phase9_oos_contracts.csv")
if os.path.exists(path):
    res = pd.read_csv(path)
    res["calib_date"] = pd.to_datetime(res["calib_date"], format="mixed")
    dec21 = res[(res.calib_date.dt.month == 12) & (res.horizon == 21)].calib_date.nunique()
    check("December 21-day horizons are present (year-boundary fix)", dec21 >= 15, f"{dec21} Decembers")
    check("no contract scored twice in a cell",
          not res.duplicated(["calib_date", "horizon", "contractID"]).any())
    feb14 = res[pd.to_datetime(res.score_date, format="mixed") == "2019-02-14"]
    check("2019-02-14 grid has no put-rows carrying call prices",
          not ((~feb14.is_call) & (feb14.strike < 0.95 * feb14.spot)
               & (feb14.mid > 0.8 * (feb14.spot - feb14.strike))).any(),
          f"{len(feb14)} contracts")
else:
    check("results file present", False, path)

print("\n" + "=" * 62)
print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
for f_ in FAILED:
    print(f"    FAILED: {f_}")
print("=" * 62)
sys.exit(1 if FAILED else 0)
