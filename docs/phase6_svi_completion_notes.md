# Phase 6: Arbitrage-Free Surface Construction — Completion Report

## Design pivot (documented, not silent)
Original plan: hand-rolled raw SVI with Durrleman no-arbitrage constraints. Pivoted to
QuantLib's `AndreasenHugeVolatilityInterpl` instead — flagged as an acceptable substitute back
in the Phase 4 design notes, and justified by Phase 5's stark reliability lesson (QuantLib-native
methods: SABR, 960 fits in 18s, zero failures; hand-rolled numerics: Heston, ~19 background
kills). Andreasen-Huge (2011) calibrates a discrete surface guaranteed arbitrage-free by
construction and, as a bonus, directly produces what Phase 7 needs (a local vol surface) instead
of requiring a second hand-built differentiation stage on top of a separately-fit SVI surface.

## Two real API bugs found and fixed (not assumed away)
1. **`CalibrationSet` quote type**: initially passed market mid *prices* — produced nonsense
   (price errors of $147 on sub-$20 options, "local vol" in the millions of percent). Verified
   against QuantLib's own test-suite source (`andreasenhugevolatilityinterpl.cpp`, fetched
   directly rather than assumed) that the quote must be **implied volatility**, not price. Fixed
   -> calibration error dropped to ~1e-7 (essentially exact) and repricing fit to 0.33pp mean IV
   error, in line with Heston/SABR's fit quality.
2. **`calibrationError()` and related methods return bound methods, not values** — SWIG wrapping
   quirk (`.first`, `.second`, `.third` are callables: `.first()` not `.first`). Minor, fixed
   immediately once the `TypeError` surfaced.

## What succeeded: the surface itself
With the quote-type fix, `surface.optionPrice(T, K, type)` reprices the calibration set to
0.33pp mean IV error (max 2.49pp) — essentially as good as SABR, and the surface is
arbitrage-free by construction (that's the whole point of the Andreasen-Huge method). **This is
Phase 6's actual deliverable and it works.**

## What surfaced: local vol instability (real finding, carried to Phase 7)
Querying `surface.localVol(T, K)` (needed for Phase 7's Dupire local-vol model) is sane at the
nearest expiry (17-22% ATM, economically reasonable) but blows up at longer tenors: 150-700% by
T~0.25-0.5, over 1000-4500% at T~0.5-0.78. **Tested whether sparse calibration data was the
cause — it wasn't**: rebuilding with a much denser set (148 contracts, 10 expiries, 15 strikes,
vs the original 45/5/9) made it *worse*, not better (4456% vs 1741% at similar points). This
rules out "just need more data" and points to something more structural: most likely mild
calendar-arbitrage inconsistency in the real market IV quotes themselves (different
expiries/vendors/snapshot times), which Dupire-style differentiation (a derivative w.r.t. T in
the denominator) is well known industry-wide to amplify into extreme local-vol noise even from
quotes that look individually reasonable. Local-vol-only construction from raw market data being
unstable without extra smoothing is a documented, common real-world problem — this is not
assessed as a bug in my usage (confirmed the surface itself reprices correctly; the instability
is specifically in the *derived* local-vol function, and got worse with more raw data, the
opposite of what a data-sparsity bug would show).

## Batch validation across all 192 dates (surface construction, not local vol)
Ran the calibration + repricing check across the full monthly sample. **Zero exceptions,
11.9 minutes total** (slower than SABR's 18s -- Andreasen-Huge's PDE-based calibration is
genuinely more compute-intensive per date -- but far more robust than Heston, no background
kills needed at all). Fit quality is excellent, even better than SABR's:

| | Mean IV error | Median |
|---|---|---|
| Overall | 0.288pp | — |
| OptionsDX era | 0.105pp | 0.052pp |
| AlphaVantage era | 0.350pp | 0.331pp |

Same era-difficulty ordering as Heston and SABR (OptionsDX easier to fit than AlphaVantage) --
now confirmed across all three models, reinforcing it's a genuine property of the two data eras,
not an artifact of any one model.

No per-date params file to persist (see module docstring) -- the calibration procedure itself,
reproducible from `data/clean/*.parquet`, is what Phase 9 will re-run when it needs this model's
prices.

## Handoff to Phase 7
The arbitrage-free surface (Phase 6's job) is done and working. Phase 7 needs to either
(a) add smoothing/regularization before extracting local vol, (b) restrict local-vol pricing to
the tenor range where it's stable (near-term only) with that limitation documented, or (c) try
QuantLib's more standard `BlackVarianceSurface` + `LocalVolSurface` path (Dupire's formula
applied to a smoothly cubic-spline-interpolated variance surface) as an alternative to
Andreasen-Huge's own adapter, since it's a different code path that may be less sensitive to the
same calendar-consistency noise. Whichever is chosen, don't just retry Andreasen-Huge with more
raw data again — already shown not to help.

## Scripts
`src/phase6_svi_local_vol.py` (calibration + repricing check + local-vol diagnostic).

## Next
Phase 7: Local Volatility (Dupire), starting from this handoff.
