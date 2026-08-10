# Phase 5: SABR Calibration — Completion Report

Design + full batch, both completed in a single session 2026-08-08 (continuing the autonomous
run authorized while the user was asleep). Unlike Heston, this phase needed no dedicated design
doc — the design succeeded cleanly on the first attempt, no bugs to chase, no background
infrastructure needed at all.

## Design decisions
1. **beta fixed at 1.0** (lognormal SABR) — not jointly fitted with alpha. Same remedy as
   Phase 2's Vasicek a/b and Phase 4's Heston kappa: alpha/beta trade off and are poorly
   identified jointly from one smile slice. Standard practice (including Hagan's own paper).
   beta=1 fits an equity index better than beta=0 (more common for rates).
2. **Formula**: `ql.sabrVolatility(strike, forward, T, alpha, beta, nu, rho)` — QuantLib's own
   Hagan-approximation implementation, used directly rather than hand-rolled, to avoid
   transcription-error risk in a formula with this many terms.
3. **Objective**: vega-weighted squared IV error, same rationale as Heston.
4. **Per-expiry independent fits** (alpha, rho, nu) — SABR's whole point, unlike Heston's single
   full-surface fit.
5. **No warm-starting needed** — each fit is a small, fast, well-conditioned 3-parameter
   closed-form optimization, nothing like Heston's numerical-integration fragility.
6. Reused Phase 4's exact `select_chain()` (same 5 expiries, same balanced put/call selection)
   so results are directly comparable to Heston later in Phase 9's scoring.

## Batch results
**192 dates x 5 expiries = 960 fits, completed in 18 seconds, entirely in the foreground** — no
resumable-checkpoint apparatus, no kills, no background process at all. A striking contrast with
Phase 4's Heston batch (~19 kills, hours of wall-clock across the design+batch+cleanup cycle):
closed-form pricing with no numerical integration is categorically more robust than a
characteristic-function/quadrature-based engine, both computationally and operationally.

- **959/960 converged (99.9%)**. The one non-convergence (2023-08-15, the 1.1yr expiry) has a
  small objective (0.0018) and reasonable fit (0.84pp) — not investigated further, negligible.
- **Mean |IV error| = 0.505pp overall** — meaningfully tighter than Heston's ~1.6-2pp. Expected,
  not a sign SABR is "better": SABR spends 15 free parameters per date (5 expiries x 3) fitting
  each expiry's smile independently, vs Heston's 5 parameters fitting the whole surface with
  cross-expiry consistency built in. SABR's tighter in-sample fit doesn't imply better
  out-of-sample predictive power — that trade-off is exactly what Phase 9's OOS scoring will
  test, not something to assume now.

## Sanity checks (all passed, unprompted economic sense)
- **Era split matches Heston's pattern**: OptionsDX 0.17pp vs AlphaVantage 0.62pp mean IV error
  — same relative ordering as Heston, reinforcing that the 2014+ era is genuinely harder to fit
  well (more expiries/strikes, different vendor conventions), not an artifact of one model.
- **nu (vol-of-vol) decreases with maturity**: 2.66 (T<0.15) -> 1.83 -> 1.41 -> 1.19 (T=0.6-1.0),
  the textbook SABR term structure (short-dated smiles curve more sharply). Slight uptick at the
  longest bucket (1.24, T>1yr) — plausible given sparser long-dated data, not investigated
  further given the overall pattern is clearly right.
- **rho consistently negative every year** (-0.67 to -0.84) — correct equity skew sign,
  stable throughout the full 15-year sample.

## Scripts
`src/sabr.py` (Hagan formula wrapper + per-slice fitter), `src/phase5_sabr_calibration.py`
(single-date design/test), `src/phase5_sabr_batch.py` (full batch, resumable though never
actually needed to resume). Output: `data/sabr_params_by_date_expiry.parquet`,
`results/phase5_sabr_batch_progress.csv`.

## Next
Phase 6: SVI arbitrage-free surface construction — the other smile-parametrization approach,
to contrast against SABR (different functional form, same per-expiry-fit spirit) before Phase 7
builds Local Vol on top of it.
