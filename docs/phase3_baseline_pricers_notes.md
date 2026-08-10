# Phase 3: Baseline Vanilla Pricers

## What it builds
`src/pricers.py` — four reusable pricers shared by every later phase:
- `bs_price` — European Black-Scholes-Merton (spot-based, continuous dividend yield)
- `black76_price` — Black-76 (forward-based); mathematically the same model as BSM
  reparametrized on the forward, included as its own implementation specifically so
  cross-validating it against `bs_price` is a real check of both, not one formula pasted twice
- `gbm_mc_price` — GBM Monte Carlo, exact terminal-value sampling + antithetic variates
- `crr_binomial_price` — Cox-Ross-Rubinstein tree, `is_american` toggle for early exercise
- `baw_american_price` — Barone-Adesi-Whaley (1987) closed-form American approximation

`src/phase3_baseline_pricers_crossvalidation.py` runs 7 checks across a synthetic grid (1,008
call/put combos: 7 moneyness x 6 tenors [1mo-3yr, matching the real data's range] x 4 vols x 3
rates) — deliberately BEFORE touching any real SPY quotes, per the phase plan.

## Results
| # | Check | Result |
|---|---|---|
| 1 | Black-76 == BSM (same forward) | max abs diff 5e-14 — machine precision |
| 2 | Put-call parity (BSM) | max abs diff 2e-14 — machine precision |
| 3 | GBM-MC vs BSM (in MC std errors) | mean\|z\|=0.44, max\|z\|=1.48, 0/982 rows exceed \|z\|>4 |
| 4 | CRR-European vs BSM | median rel diff 0.035%, max 1.29% (material prices only) |
| 5 | CRR-American >= CRR-European | 0 violations / 1008 |
| 6 | CRR-American >= intrinsic | 0 violations / 1008 |
| 7 | BAW vs CRR-American | median 0.086%, 90th pct 0.94%, max 51.85% (material prices only) |

Checks 1-2 confirm the formula implementations are algebraically correct. Check 3 confirms the
Monte Carlo engine is statistically unbiased. Checks 5-6 confirm the American-exercise logic in
the binomial tree is correct (no-arbitrage bounds hold everywhere, not just on average).

## One real finding, not a bug
Raw (unfiltered) relative-error stats for checks 4 and 7 were misleadingly huge (up to 4001% for
BAW) purely because deep-OTM/near-worthless option prices (sub-cent on $100 notional) make any
relative-error metric blow up on tiny denominators — verified directly: the worst CRR-vs-BSM row
was $0.0000013 vs $0.0000012, and the worst raw BAW row was $0.0000006 vs $0.00004, both
economically meaningless. Re-ran with a materiality floor (price > $0.01) to get an honest
headline number.

Even after that floor, one real BAW weakness survives: **51.85% relative error for deep-OTM
(K=$70 vs S=$100), long-dated (3yr), low-vol (10%) puts** (CRR=$0.0168, BAW=$0.0255). This is a
known, documented limitation of BAW's quadratic approximation in exactly that corner (far OTM +
long-dated + low vol, where the early-exercise boundary estimate is least accurate) — not an
implementation bug, confirmed by CRR (the more exact reference) landing far from BAW only there.
Doesn't block anything: CRR is the primary American pricer here, BAW is a secondary fast
cross-check. Worth remembering if BAW is ever used standalone for speed at scale later — flag/
exclude that parameter corner rather than trust it blindly.

## Next
Phase 4: Heston stochastic-vol calibration at t0, using QuantLib's `HestonModel` +
`HestonModelHelper` (confirmed installed and available, see Phase 1/2 notes).
