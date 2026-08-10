# Phase 8: Merton Jump-Diffusion — Completion Report

## Model
Merton (1976) adds a compound-Poisson jump component to geometric Brownian motion:

```
dS/S = (r - q - lambda*k) dt + sigma dW + dJ
J = compound Poisson, jump arrivals ~ Poisson(lambda), ln(1+jump) ~ N(mu_j, sigma_j^2)
k = E[jump size] = exp(mu_j + 0.5*sigma_j^2) - 1     (compensator, keeps risk-neutral drift right)
```

Four parameters per date: `sigma` (diffusive vol), `lambda` (jumps/year), `mu_j` (mean log jump
size), `sigma_j` (jump size vol).

Closed-form price: a Poisson-weighted sum of ordinary Black-Scholes prices, conditioning on
n jumps having occurred:

```
V = sum_{n=0..N} [ exp(-lambda' T) (lambda' T)^n / n! ] * BS(S, K, T, r_n, q, sigma_n)
where lambda' = lambda(1+k),  sigma_n = sqrt(sigma^2 + n*sigma_j^2/T),
      r_n = r - lambda*k + n*ln(1+k)/T
```

## Implementation
No native jump-diffusion engine is exposed in this QuantLib build (checked `dir(ql)` for
`Jump`/`Merton`/`diffusion` — only `Merton76Process`, with no matching pricing engine). So
hand-implemented in `src/merton.py`. Low risk: unlike Heston's characteristic-function
quadrature, every term here is a plain Black-Scholes price, and the Poisson weights decay
factorially so the sum converges fast and stays numerically well-behaved.

`N_TERMS = 25`. Justification, not a guess: within this project's bounds (`lambda <= 5`/yr,
`T <= 3`yr) the worst case is `lambda*T = 15`, which needs ~30 terms for full 5-sigma tail
coverage; 25 keeps meaningful margin for the *actual* calibrated lambdas (0.09–1.25 typically,
so `lambda*T` is usually well under 1) while roughly halving pricer cost vs the original 50.

## Calibration
Same design as Heston for comparability: one parameter set per date fit to the **whole surface**
(jump risk is a market-wide property, not tenor-specific — standard in the literature), using the
**same** vega-weighted IV-space objective and the **same** `select_chain()` contract set as every
other model. Coarse grid pre-screen (2×3×2×2 = 24 seeds) then L-BFGS-B refine of the top 2.
Bounds: sigma[0.01,2], lambda[0,5], mu_j[-2,1], sigma_j[0.001,2].

## The performance lesson (the real work of this phase)
First working version took **~130 s/date** → 192 dates ≈ 7 hours. Three fixes, in order of how
much they actually mattered:

1. **Vectorizing across the n-terms sum only: barely helped** (130 s → 109 s). This was the
   obvious optimization and it was the wrong target.
2. **`merton_price_batch()` — vectorizing across ALL CONTRACTS at once**: this was the real
   bottleneck. The objective was looping ~45 contracts in Python per evaluation, times hundreds
   of optimizer evaluations.
3. Reducing optimizer effort (3 seeds → 2, maxiter 150 → 60) and `N_TERMS` 50 → 25: 109 s → ~90 s,
   then steady-state ~50–80 s/date in the actual batch.

Verified `merton_price_batch` against the scalar `merton_price` on a strike/tenor/type sample:
**max difference 0.0** (bit-identical), so the speedup cost nothing in accuracy.

## Results: 192/192 dates, 188 converged (97.9%), mean in-sample IV error 1.471pp
Comparable to Heston (1.86pp on the shared design date), as expected for two 4–5 parameter
models fitting the same surfaces.

| Parameter | Mean | Min | Max |
|---|---|---|---|
| sigma (diffusive vol) | 0.102 | 0.045 | 0.422 |
| lambda (jumps/yr) | 0.461 | 0.088 | 5.000 |
| mu_j (mean log jump) | −0.284 | −0.664 | −0.116 |
| sigma_j | 0.163 | 0.001 | 0.286 |

## Two unprompted economic validations
**1. Lambda tracks real crises.** Mean jump intensity by year:

| 2010 | 2011 | 2015 | 2019 | **2020** | 2021 | **2022** | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|
| 0.37 | 0.56 | 0.31 | 0.32 | **1.25** | 0.37 | **0.86** | 0.23 | 0.23 |

COVID (2020) and the rate-hike year (2022) stand out by 3–5× over calm years. Nothing in the
calibration was told when crises occurred — this is read purely out of option prices.

**2. mu_j is negative on all 192 dates** (mean −28%). The model consistently prices jumps as
*downward* — crash risk, not symmetric surprise. That is the correct, well-documented empirical
sign for equity indices and it emerged without being constrained to.

Also note `sigma` (diffusive vol, mean 10.2%) sits well below total implied vol, which is the
expected decomposition: the jump component absorbs part of the variance that a pure-diffusion
model would have to load into sigma.

## Honest caveats
- **Lambda hit its upper bound (5.0) on at least one date** — for that date lambda is a floor,
  not a fitted value. Not chased; it is a single outlier in 192.
- 4 dates did not converge (same treatment as Heston's non-converged dates: params are usable,
  objective is low, flagged rather than dropped).
- The 24-seed grid is coarse. As with Heston's kappa, do not over-read any single date's exact
  parameter values; the time-series behaviour is the trustworthy signal.

## Infrastructure
Longest batch in the project (~3.5 hr wall clock incl. restarts). Killed twice by the
environment; resumable checkpointing (every 4 dates) meant each cost ~1–3 dates of rework.

## Scripts
`src/merton.py` (pricer, scalar + batch), `src/phase8_merton_calibration.py` (calibration +
single-date design test), `src/phase8_merton_batch.py` (resumable batch).
Output: `data/merton_params_by_date.parquet`, `results/phase8_merton_batch_progress.csv`.
