# Phase 0: Vendor IV/Greeks Methodology Check

## Question
Neither OptionsDX nor Alpha Vantage publish an exact spec for how `implied_volatility`,
`delta`, `gamma`, `theta`, `vega`, `rho` are computed in the combined SPY options dataset.
Third-party docs only say "industry-standard option pricing model, usually Black-Scholes" —
no disclosed dividend yield, risk-free rate source, or American vs. European treatment.

## Method
Independently recomputed European Black-Scholes IV (Brent's method solve) and closed-form
Greeks from `(S, K, T, r, q, mid-price)` and diffed against the vendor-supplied fields, on a
stratified sample (~1500 rows/vendor, spread across all years, liquid bid/ask rows only).

- `S` (SPY close) and `r` (13-week T-bill, ^IRX) pulled live via yfinance — not in the options
  dataset itself, and small (~4k rows) vs. the 23.5M-row options file, so pulling live doesn't
  conflict with using the already-sourced local options data.
- `q` = flat 1.4% dividend yield proxy (SPY trailing average). Will be replaced by an actual
  dividend schedule and a real Vasicek-fit rate curve in later phases — this pass is a sanity
  check, not the production pricing engine.
- Script: `src/phase0_iv_greeks_replication.py`. Sample output: `results/phase0_replication_sample.csv`.

## Result
| Vendor | Metric | Corr | Median abs diff |
|---|---|---|---|
| OptionsDX (2010-13) | IV | 0.895 | 0.0065 |
| OptionsDX | delta | 0.999 | 0.0050 |
| OptionsDX | vega (after filtering artifacts, see below) | 0.983 | 0.0025 |
| AlphaVantage (2014-25) | IV | 0.779 | 0.0108 |
| AlphaVantage | delta | 0.999 | 0.0120 |
| AlphaVantage | vega | 0.979 | 0.0094 |

Delta/gamma correlations >0.99 across both vendors — near-exact match to plain European BSM.
IV and theta/rho correlations 0.78-0.90, with small systematic offsets. This is consistent with
(a) SPY options being American-style (early-exercise premium a European model can't capture),
and (b) our flat rate/dividend proxy vs. whatever real curve the vendor used — not a sign the
vendor data is fabricated. Overall: **confirms the vendor greeks/IV are genuine, internally
consistent, standard-BSM-family outputs**, not placeholder or randomly generated numbers.

## Data-quality finding (new, not in prior data-acquisition notes)
50 OptionsDX-era rows (out of 1439 sampled) had `vendor_vega` in the hundreds (e.g. -903, -777)
alongside `vendor_iv` near zero (0.0001-0.0006) — a vendor-side solver artifact on illiquid,
deep-ITM, stale-quote contracts (near-zero IV blows up the vega formula's denominator). None of
these are usable rows for greeks. **Filter rule for modeling phases: drop any row with
`implied_volatility <= 0.01` before trusting `vega`/`theta`/`rho`** (delta/gamma stayed sane
even on these rows, only vega/theta/rho blew up). Affects ~3.5% of the OptionsDX-era sample;
negligible in AlphaVantage era (5/1344). Add this to the existing caveat list in
[[project_spy_options_dataset]] (open_interest NaN 2010-13, 23 missing trading days).
