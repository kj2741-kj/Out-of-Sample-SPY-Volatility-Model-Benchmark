# Phase 4 Handoff: Heston Calibration Design

Read this file first, before exploring the repo. It's written so a cold session doesn't need to
re-derive anything below by scanning files. Full history is in `docs/phase0-3_*_notes.md` if
deeper context is ever needed, but nothing in this handoff requires reading those.

## Your job this phase
Design the Heston calibration framework (this is the Fable 5 half of Phase 4; batch-running it
across ~180 dates once designed is a separate, later Sonnet pass — not your job here). Success
criterion: a single date's calibration runs and produces stable, sensible parameters. Don't move
to batch execution until that's true.

## Five decisions to make, in this order
1. **Objective function** — price-space RMSE, IV-space RMSE, or vega-weighted residuals. (Lean
   IV-space or vega-weighted — price-space overweights deep ITM contracts, which dominate raw
   price magnitude but carry little information about the smile.)
2. **Optimizer strategy** — plain local Levenberg-Marquardt will land in bad local minima from a
   poor seed. Needs either a global stage (differential evolution / multi-start grid) or a
   carefully chosen analytic seed (e.g. long-run variance from realized vol, mean-reversion speed
   from a term-structure heuristic).
3. **Feller condition** (`2*kappa*theta > xi^2`) — enforce as a hard constraint, penalize softly,
   or allow violation (violating it doesn't break the model, just means variance can hit zero).
4. **Parameter bounds** — kappa, theta, xi, rho, v0 all need sane bounds to keep the optimizer
   out of degenerate regions (same failure mode as Phase 2's Vasicek MLE — check
   `docs/phase2_vasicek_notes.md` if unsure what "degenerate" looked like there).
5. **Cross-date seeding** — cold-start every date independently, or warm-start each date from the
   prior date's converged parameters (faster, and Heston params shouldn't jump wildly day to day
   in a genuine fit — if they do, that's itself a diagnostic signal).

## What's already built (exact locations/schemas, no need to re-derive)

**QuantLib 1.43 is installed** (`python -m pip install QuantLib` if a fresh env needs it — NOT
plain `pip install`, that resolves to a different interpreter on this machine, see
`docs/phase1_data_pipeline_notes.md`). Confirmed available: `ql.HestonModel`,
`ql.HestonModelHelper`, `ql.AnalyticHestonEngine` — use these rather than hand-rolling a
characteristic-function pricer.

**Clean options data**: `data/clean/spy_options_clean_<year>.parquet`, 16 files, 2010-2025.
Columns: `contractID, symbol, expiration, strike, type, last, mark, bid, bid_size, ask, ask_size,
volume, open_interest, date, implied_volatility, delta, gamma, theta, vega, rho, year, source,
spy_close, T, moneyness, is_call, mid, has_valid_quote, iv_quality_ok, not_expired_yet`.
- `T` = year-fraction to expiry (calendar days/365), `moneyness` = strike/spy_close, `mid` =
  (bid+ask)/2 (NaN if no valid quote).
- Filter rule: drop `iv_quality_ok == False` before trusting IV-based objective functions
  (~8-10% of OptionsDX-era 2010-13 rows, negligible in AlphaVantage 2014-25).
- `source` column: `"OptionsDX"` (2010-13) vs `"AlphaVantage"` (2014-25) — keep separate in any
  cross-era comparison, different vendor quirks.

**Discount curve**: `data/vasicek_discount_curve.parquet`. Columns: `date, y3mo, y1yr, y2yr, y3yr,
r3mo, df_1mo, df_3mo, df_1yr, df_2yr, df_3yr, vasicek_yield_1yr, vasicek_yield_2yr,
vasicek_yield_3yr`. `df_*` are discount factors P(t,T) for a given date's row. For an arbitrary
tenor not in that list, use `vasicek_zero_price(a, b, sigma, r0, tau)` from
`src/phase2_vasicek_calibration.py` (params in `results/phase2_vasicek_params.csv`) — but note
individual `a`/`b` are NOT independently meaningful (see that script's docstring / phase2 notes),
only the discount curve output is validated.

**Pricers**: `src/pricers.py` — `bs_price(S,K,T,r,q,sigma,is_call)`,
`black76_price(F,K,T,r,sigma,is_call)`, `gbm_mc_price(S,K,T,r,q,sigma,is_call,n_paths,seed)`
returns `(price, stderr)`, `crr_binomial_price(S,K,T,r,q,sigma,is_call,is_american,n_steps)`,
`baw_american_price(S,K,T,r,q,sigma,is_call)` (known to degrade for deep-OTM+long-dated+low-vol,
see `docs/phase3_baseline_pricers_notes.md` — not relevant to Heston work but don't reuse BAW
blindly elsewhere without remembering this).

**Dividend yield**: flat 1.4% proxy used through Phase 3 (`Q_DIV_YIELD` in
`src/phase0_iv_greeks_replication.py`). No real dividend schedule built yet — fine to keep using
the flat proxy for Heston calibration unless it becomes a clear source of fit error.

**SPY spot**: `data/spy_underlying_close.parquet` (date, spy_close), already merged into the
clean chain files above as the `spy_close` column — don't re-fetch or re-merge.

## Not your job here
Batch-running the design across all dates, persisting fitted parameters, and QA-ing convergence
across the 15-year sample is the Sonnet half of Phase 4. Stop once one date works well and hand
back.
