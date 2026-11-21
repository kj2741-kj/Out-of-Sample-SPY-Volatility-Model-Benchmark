# Phase 11: Model Greeks, Daily Delta Hedging and P&L Explain (2020-2021)

Scripts: `src/phase11_greeks_hedging.py` (panel), `src/phase11_analysis.py` (tables), `src/phase11_audit.py` (checks).
Results: `results/phase11_contract_days.csv`, `phase11_part1_greeks.csv`, `phase11_part2_hedging.csv`,
`phase11_part3_pl_explain.csv`, `phase11_summary.txt`, `phase11_audit.txt`.

## Design
- 24 monthly calibration dates in 2020 (COVID stress) and 2021 (calm). Parameters frozen at each date t, as in Phase 9.
- At each t, the Phase 9 scoring grid (5 moneyness x 3 tenor buckets) is selected on t and every contract is followed
  daily for up to 21 trading days. A contract leaves when its quote fails the Phase 9 filters or under 7 days remain.
- Panel: 7,196 contract-days, 355 contracts, 501 trading days; 6,818 consecutive-day hedge pairs (6,712 in 2020-21,
  the rest are early-January 2022 days of the December 2021 cohort).
- Models: BS at the contract's own same-day vendor IV (BS_market, sticky-strike benchmark), BS at the frozen calibrated
  sigma (BS_calib), SABR (beta 1, nearest slice) and Merton, both by spot bump-and-reprice (0.5%).
- Hedge error, long one option, short delta shares, cash at r: e = dV - delta*(dS + qS dt) - r(V - delta S)dt, V = mid.
  Gain vs BS_market = 1 - SSE(model)/SSE(BS_market). 95% intervals: cluster bootstrap over calibration dates.
- P&L explain: day-0 BS Greeks at vendor IV; steps delta, +gamma, +vega (own IV change), +theta.

## Local volatility excluded (bug found in Phases 7 and 9)
`FdBlackScholesVanillaEngine` was built without `localVol=True`. QuantLib then prices with the Black vol at the strike,
so the "LocalVol" and "AH_Surface" models were frozen implied-vol surfaces read by strike (sticky strike), not Dupire.
On 2020-01-15 the "LocalVol" price equals BS at the surface IV to four decimals. With the flag on, the Dupire surface
as built fails on 3-4 of 15 grid contracts and misprices its own calibration date (RMSE 0.40 vs 0.18; 1.23 vs 0.36 on
2021-06-15). Phase 9's numbers are valid for a frozen sticky-strike surface; the label is wrong. Not yet fixed.

## Results
**Part 1, Greeks.** SABR and Merton deltas sit above the market BS delta: ATM by +0.11/+0.08 (2020) and +0.13/+0.12
(2021); OTM puts less negative by about 0.09-0.13. Both are scale-invariant models, so their deltas follow sticky
moneyness. Their gammas on OTM puts are about half the market BS gamma (median ratio 0.30-0.60).

**Part 2, hedging (6,818 pairs).** BS_market hedges best. Gain vs BS_market: BS_calib -0.14 (CI -0.43 to -0.02),
SABR -1.00 (CI -1.95 to -0.53), Merton -0.85 (CI -1.75 to -0.36), so the smile models roughly double the squared
hedging error. Every hedge removes most variance vs unhedged (BS_market 91%, SABR 82%). The penalty is largest in calm
2021 (SABR -1.99, Merton -2.10) and smallest in the March-April 2020 stress (SABR -0.39, Merton -0.20, BS_calib +0.06,
intervals overlapping zero for Merton and BS_calib).

**Audit.** The variance-minimising adjustment to BS_market delta is negative in every slice (-0.017 to -0.050 per share;
in-sample regression), as Hull and White (2017) report for index options, so the right correction is downward while
SABR and Merton move delta up. The penalty is present in the first week after calibration (SABR -0.62, Merton -0.45,
BS_calib -0.05), so it is structural, not only stale parameters. SABR rho sits at its -0.999 bound on 4% of slices;
excluding that date does not change the result.

**Part 3, P&L explain.** Unexplained share of daily option P&L (mean |actual - predicted| / mean |actual|):
delta 31.1%, +gamma 29.1%, +vega 11.7%, +theta 4.9% (Spearman 0.943 to 0.996). Vega is the largest step. Gamma helps
mainly in the stress window (30.3% to 25.3%) and barely outside it (31.4% to 30.8%).

## Limits
- SPY options are American; Greeks are European BS. The grid uses OTM puts and calls, where early exercise is small.
- Mids carry bid-ask noise; it inflates every model's error equally and does not favour one model.
- The minimum-variance slope is fitted in sample; only its sign and consistency are claimed.
- Continuous dividend yield (1.4%), so ex-dividend days add error to all models.
- Two years, about 15 contracts per date: results describe this sample, not every regime.
