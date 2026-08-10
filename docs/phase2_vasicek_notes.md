# Phase 2: Vasicek Discount Curve

## What it does
`src/fetch_fred_yields.py` pulls real FRED Treasury constant-maturity yields (DGS3MO, DGS1,
DGS2, DGS3; 2010-2025) via the public `fredgraph.csv` endpoint — no API key needed. Extended to
3yr (not stopped at 2yr) after confirming SPY LEAPS in the options dataset run out to a full 3
years (`data/clean/*.parquet` max `T` = 3.0), with 2.5-5%/year of rows beyond the 2yr point —
material, not a rounding error.

`src/phase2_vasicek_calibration.py` fits `dr = a(b-r)dt + sigma*dW` and writes a daily discount
curve (`data/vasicek_discount_curve.parquet`) with `df_1mo/3mo/1yr/2yr/3yr` factors for every
date 2010-2025, usable directly by Phase 3+.

## Two real findings, both handled rather than papered over

**1. Time-series MLE for (a,b) is degenerate on this series.** Daily 3mo-yield autocorrelation
is 0.9999 — essentially unit-root persistence. With only ~1.5 full rate cycles in the 15-year
window (ZIRP 2010-15 → 2022-23 hikes → plateau), pure historical maximum likelihood can't
distinguish "no mean reversion" from "slow mean reversion": an unbounded fit pushed `a -> 0` with
`b` diverging to nonsense (14,145%); a bounded retry still hit `a` near its floor with `b` pinned
to its ceiling. This is a well-known, real limitation of fitting Vasicek's speed parameter from
history alone — practitioners typically calibrate `a` to the market curve shape or derivatives
instead. Applied that fix here: kept `sigma` from the time series (not degenerate), fit `(a,b)`
to match the sample's **average** 1yr/2yr/3yr term-structure shape instead of the noisy daily
path (3 target points, 2 unknowns, overdetermined least squares).

**2. Even the curve-shape fit only pins down the combination `a*b`, not `a` and `b`
individually.** Verified directly: refitting with the upper bound on `b` relaxed from 0.20 to
0.50 to 5.0 gives wildly different `(a, b)` — (0.0114, 0.20), (0.0043, 0.50), (0.0009, 2.29) —
but the fit residual barely moves (cost ~2.6-2.8e-8 in every case) and `a*b` stays ~0.0021-0.0023
throughout. This is the classic Vasicek short-end identifiability limit: for `a*tau` small (max
here is `a*3yr` ≈ 0.03), bond prices depend on `a` and `b` almost entirely through their product,
not separately — you'd need much longer-dated instruments (10yr+) to split them apart. **Reported
`a`/`b` individually should not be read as real economic mean-reversion-speed/long-run-rate
estimates** — they're one representative member of a nearly-flat family. What IS well-identified
and validated is the resulting discount curve itself, which is the only thing downstream pricing
phases actually consume.

## Discount curve validation (out-of-sample by construction — fit to the AVERAGE curve, checked
day-by-day against the actual daily path)
| Tenor | Corr vs real yield | RMSE |
|---|---|---|
| 1yr | 0.990 | 0.29pp |
| 2yr | 0.973 | 0.50pp |
| 3yr | 0.954 | 0.66pp |

Accuracy degrades gracefully with tenor (expected — single-factor Vasicek is a simplification),
but stays tight enough across the full 1-3yr range the options data actually uses. Final params:
`results/phase2_vasicek_params.csv`.

## Next
Phase 3: production BSM + Black-76 + GBM-MC + American CRR/BAW pricers, cross-validated against
each other, now with a real discount curve instead of the flat-rate proxy from Phase 0.
