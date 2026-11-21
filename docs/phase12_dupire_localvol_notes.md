# Phase 12: True Dupire Local Volatility (fix for the Phase 7/9 labelling bug)

Code: `src/phase12_dupire.py` (surface + pricer), `src/phase12_localvol_oos.py` (OOS run),
`src/phase12_analysis.py` (tables). Tests: `tests/test_phase12_dupire.py`. Results:
`results/phase12_localvol_oos_contracts.csv`, `results/phase12_summary.txt`.

## The bug
Phases 7 and 9 priced "LocalVol" (and "AH_Surface") with `ql.FdBlackScholesVanillaEngine(process, N, N)`.
In QuantLib 1.43 the `localVol` argument defaults to False, and the engine then uses the Black vol at the
strike, not the local vol. So "LocalVol" was Black-Scholes at the frozen SABR-smoothed implied-vol surface,
read by strike: a **sticky-strike surface**. Evidence: on 2020-01-15 its price equals BS at the surface vol to
four decimals, and its delta equals the sticky-strike delta (test T4). Phase 7's repricing check used the same
engine call, so it passed by construction. Two audits checked outputs, not model identity, and missed it.
With the flag on, the old Dupire surface (QuantLib `LocalVolSurface` on a bilinear `BlackVarianceSurface`)
failed on 3-4 of 15 grid contracts and could not reprice its own calibration date.

## The fix
1. Per calibrated expiry, the stored Phase 5 SABR fit as total implied variance w_i(y) in log-moneyness y
   against that expiry's forward.
2. Calendar consistency: w non-decreasing in T at each y. Linear in w between expiries, constant implied vol
   before the first and after the last.
3. Gatheral's form of Dupire: sigma_loc^2 = w_T / (1 - (y/w) w_y + 1/4(-1/4 - 1/w + y^2/w^2) w_y^2 + 1/2 w_yy),
   with w_y and w_yy from the closed-form SABR smile (no surface-interpolation noise).
4. Guard rails: denominator floor 1e-3 (a negative denominator means butterfly arbitrage in the input smile),
   local vol clipped to [0.02, 2.0]. Guarded share of near-the-money nodes: 0.54% on average across 192 dates
   (1.8% on 2020-03-16, when SABR rho sat at its -0.999 bound).
5. `FixedLocalVolSurface` + `FdBlackScholesVanillaEngine(..., localVol=True)`, mesh sized from the surface's own
   vol level. Out of sample, the same function of (S, calendar time) is used: the surface time axis is shifted by
   d - t, so the model is frozen, not re-anchored.

## Model-identity tests (all pass)
- T1 flat smile (SABR nu ~ 0, rho = 0): local vol = 0.2000 everywhere; LV PDE price = BS within $0.002.
- T2 reprices its own input surface on 2020-01-15, 2020-03-16, 2021-06-15: every grid contract within 0.5
  implied-vol points (max 0.33, 0.45, 0.50; $RMSE 0.028, 0.169, 0.059).
- T3 under SPY's skew the local-vol ATM 3m delta sits below the sticky-strike delta (0.351 vs 0.517;
  0.454 vs 0.554; 0.287 vs 0.501).
- T4 the old Phase 9 engine call fails T3 (its delta 0.517 equals the sticky-strike 0.517).
- `tests/test_scoring.py`: 13 passed.

## Out-of-sample result (same 8,583 Phase 9 contract-scorings; guard re-derived 7,828 / 0.864 / 1.889 first)
Dupire priced 8,583 of 8,583. On the 7,828-contract like-for-like set, pooled RMSE ($, 95% cluster bootstrap):

| Model | RMSE | vs BS | 95% range |
|---|---|---|---|
| Sticky-strike surface (old "LocalVol") | 0.864 | -54% | 0.66-1.12 |
| Merton | 1.233 | -35% | 1.05-1.41 |
| Heston | 1.256 | -34% | 1.03-1.52 |
| SABR | 1.290 | -32% | 1.02-1.61 |
| **Local vol, true Dupire** | **1.359** | **-28%** | 1.17-1.57 |
| Black-Scholes | 1.889 | | 1.74-2.05 |

Median absolute error: Dupire 0.346 (SABR 0.342, sticky-strike 0.254, BS 0.961). By horizon Dupire decays fastest:
1.152 at 5 days, 1.275 at 10, 1.621 at 21 (sticky-strike 0.779 / 0.813 / 0.992). Head to head, Dupire beats SABR on
47.8% of contracts and the sticky-strike surface on 36.6%.

## Reading
Frozen local vol implies the smile moves against spot (sticky local vol); SABR with beta 1 and Merton move it
with the forward (sticky moneyness); the sticky-strike surface keeps it fixed by strike. Out of sample SPY's smile
stayed closest to fixed by strike, so the sticky-strike surface wins and true local vol lands with the other
smile models. Phase 11's hedging test says the same from the Greeks side: the least-squares best hedge delta sits
about a third of the way from sticky-strike BS toward local vol (0.32 in both 2020 and 2021, 0.50 in the crash).

## Limits
- The Dupire surface is only as arbitrage-free as the SABR slices it is built from (guard rails on 0.54% of nodes).
- One time convention (calendar-time frozen); a re-anchored variant was not tested.
- European pricing for American SPY options, as in every other model here.
