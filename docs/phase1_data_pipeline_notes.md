# Phase 1: Data Pipeline

## What it does
`src/phase1_data_pipeline.py` reads the raw 23.5M-row combined options parquet, merges SPY spot
close per date, adds `T` (year-fraction to expiry), `moneyness` (strike/spot), `mid`
((bid+ask)/2), and three quality flags (`has_valid_quote`, `iv_quality_ok`, `not_expired_yet`).
Writes 16 yearly partitions to `data/clean/spy_options_clean_<year>.parquet` so later phases can
filter-pushdown by year instead of loading the full file each time. No rows dropped — flags only,
so downstream phases choose what to filter.

## QA summary (full population, not a sample)
| Year | Source | Rows | % valid quote | % IV quality ok | % same-day expiry | % missing spot |
|---|---|---|---|---|---|---|
| 2010-2013 | OptionsDX | ~525K-723K/yr | 87-90% | 90-92% | 1.5-2.3% | 0% (except 2012) |
| 2014-2025 | AlphaVantage | ~839K-2.6M/yr | 90-96% | 98.5-100% | 2.0-6.5% | 0% |

- **OptionsDX-era IV-quality rate (90-92%) confirms the Phase 0 sample estimate** (~96.5% in that
  smaller liquid-only sample) was in the right range — full population shows a somewhat larger
  share of near-zero-IV artifact rows than the 1,500-row sample suggested, expected since the
  sample was pre-filtered to liquid bid/ask rows only.
- **Same-day-expiry rows rise steadily over time** (1.5% in 2010 to 6.5% in 2023) — tracks SPY's
  real move toward daily/weekly expiration listings and the 0DTE boom, not a data defect.
- **2012-10-29 and 2012-10-30 have options quotes but no SPY close** — those are the two days
  NYSE was closed for Hurricane Sandy. OptionsDX still carries stale/leftover quote rows for
  those dates (830 rows, 0.83% of 2012). Same category of artifact as the Jimmy Carter mourning
  day already noted in [[project_spy_options_dataset]] — a real market closure, not a vendor gap.
  Rule: these 2 dates have `spy_close = NaN` after the Phase 1 merge and will naturally drop out
  of anything requiring a spot price; no special handling needed beyond that.

## Next
Phase 2: Vasicek short-rate fit to FRED Treasury yields → real discount curve, replacing the
flat-rate/IRX proxy used only for the Phase 0 diagnostic.
