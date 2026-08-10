"""
Phase 4 batch execution (Sonnet tier -- design is locked, see docs/phase4_heston_design_notes.md).
Runs the calibrate_one_date framework across ~1 date/month, 2010-2025, warm-started from the
prior date's converged params except at the two vendor-era boundaries (don't warm-start across
the OptionsDX->AlphaVantage data-quirk boundary, per the design doc).

Regime-break fallback (untested in the design phase, applied here): if a warm-started date's
mean IV error or non-convergence looks bad relative to a rolling baseline, redo that date cold.

Checkpoints every 12 dates to results/phase4_heston_batch_progress.csv so a crash mid-run doesn't
lose completed work, and so progress is visible without waiting for the full ~20-40min run.

Resumable: if a checkpoint CSV already exists, picks up from the date after the last row instead
of re-running from scratch (skips the expensive cold-starts already paid for). Learned the hard
way -- the first run got killed by something outside this script's control after 24/192 dates
(~1hr wall clock, well past the compute-only ETA), and re-running cold from date 1 would have
wasted the 2 cold-starts and ~24 warm calibrations already completed and checkpointed.
"""
import sys
import time
import numpy as np
import pandas as pd
from phase4_heston_calibration import calibrate_one_date, PROJ

PROGRESS_CSV = f"{PROJ}\\results\\phase4_heston_batch_progress.csv"

WARM_FALLBACK_IV_ERROR_PP = 8.0  # if warm-start mean|IV err| exceeds this, redo cold
CHECKPOINT_EVERY = 6  # shortened from 12 -- background runs in this environment appear to get
                       # killed after ~45-70min (2 occurrences so far), so checkpoint more often
                       # to lose less work per restart

def monthly_calibration_dates():
    dates = []
    for year in range(2010, 2026):
        df = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{year}.parquet", columns=["date"])
        unique_dates = pd.to_datetime(df["date"].unique())
        for month in range(1, 13):
            in_month = unique_dates[(unique_dates.year == year) & (unique_dates.month == month)]
            if len(in_month) == 0:
                continue
            target = pd.Timestamp(year=year, month=month, day=15)
            closest = in_month[np.argmin(np.abs((in_month - target).days))]
            dates.append(pd.Timestamp(closest))
    return sorted(dates)


def main():
    dates = monthly_calibration_dates()
    print(f"{len(dates)} monthly calibration dates, {dates[0].date()} to {dates[-1].date()}", flush=True)

    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]

    era_boundaries = set()  # dates that must cold-start regardless of prior date
    prev_year = None
    for d in dates:
        era = "OptionsDX" if d.year <= 2013 else "AlphaVantage"
        prev_era = "OptionsDX" if prev_year is not None and prev_year <= 2013 else "AlphaVantage"
        if prev_year is None or era != prev_era:
            era_boundaries.add(d)
        prev_year = d.year

    results = []
    prior_params = None

    try:
        prior_progress = pd.read_csv(PROGRESS_CSV)
        prior_progress["date"] = pd.to_datetime(prior_progress["date"])
        last_done = prior_progress["date"].max()
        dates = [d for d in dates if d > last_done]
        results = prior_progress.to_dict("records")
        converged_rows = prior_progress[prior_progress.get("kappa").notna()] if "kappa" in prior_progress else prior_progress.iloc[0:0]
        if len(converged_rows) > 0:
            prior_params = converged_rows.iloc[-1].to_dict()
        print(f"RESUMING from checkpoint: {len(prior_progress)} dates already done (last={last_done.date()}), "
              f"{len(dates)} remaining, warm-start seed {'found' if prior_params else 'NOT found'}", flush=True)
    except FileNotFoundError:
        print("No existing checkpoint -- starting fresh from date 1.", flush=True)

    year_cache = {}
    t_start = time.time()
    n_already_done = len(results)
    total_dates = n_already_done + len(dates)

    for i, d in enumerate(dates):
        if d.year not in year_cache:
            year_cache.clear()  # keep memory bounded -- only hold the current year's file
            year_cache[d.year] = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{d.year}.parquet")
        df = year_cache[d.year]

        warm_start_x = None
        if prior_params is not None and d not in era_boundaries:
            warm_start_x = [prior_params["kappa"], prior_params["theta"], prior_params["xi"],
                             prior_params["rho"], prior_params["v0"]]

        try:
            res = calibrate_one_date(d, df, vas, vas_params, warm_start_x=warm_start_x)
        except Exception as e:
            print(f"{d.date()}: EXCEPTION {e}", flush=True)
            results.append(dict(date=d, converged=False, error=str(e)))
            continue

        if res.get("error") == "too_few_contracts":
            print(f"{d.date()}: SKIPPED, only {res['n_contracts']} contracts available", flush=True)
            results.append(res)
            continue

        # Regime-break fallback: warm start looked bad -> redo cold. Fixed 2026-08-08: raw
        # objective values aren't comparable across a non-converged vs converged result (a
        # non-converged optimizer can report ANY objective, including a low one, since it
        # stopped before actually reaching a local optimum) -- a converged cold result must
        # always beat a non-converged warm one, not just a lower-objective one. The original
        # "only swap if strictly lower objective" version left several dates (2013-12-16,
        # 2014-02-14, 2014-08-15, 2014-09-15) stuck with converged=False final results even
        # though the fallback ran, because the non-converged warm objective happened to look
        # numerically lower. Those dates get patched in the post-run cleanup pass.
        if warm_start_x is not None and (not res["converged"] or res["mean_iv_error"] > WARM_FALLBACK_IV_ERROR_PP / 100):
            print(f"{d.date()}: warm-start fallback triggered (mean_iv_error="
                  f"{res['mean_iv_error']*100:.2f}pp, converged={res['converged']}) -- redoing cold", flush=True)
            res_cold = calibrate_one_date(d, df, vas, vas_params, warm_start_x=None)
            cold_is_better = (res_cold["converged"] and not res["converged"]) or (
                res_cold["converged"] == res["converged"] and res_cold.get("objective", np.inf) < res.get("objective", np.inf))
            if cold_is_better:
                res = res_cold
                res["fallback_triggered"] = True

        res.setdefault("fallback_triggered", False)
        results.append(res)
        if "kappa" in res:
            prior_params = res  # keep warm-starting from the latest successful calibration

        elapsed = time.time() - t_start
        print(f"[{n_already_done+i+1}/{total_dates}] {d.date()} mode={res.get('mode')} obj={res.get('objective', np.nan):.5f} "
              f"conv={res.get('converged')} meanIVerr={res.get('mean_iv_error', np.nan)*100:.2f}pp "
              f"elapsed={elapsed/60:.1f}min", flush=True)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(PROGRESS_CSV, index=False)

    out = pd.DataFrame(results)
    out.to_csv(PROGRESS_CSV, index=False)
    out.to_parquet(f"{PROJ}\\data\\heston_params_by_date.parquet", index=False)

    n_conv = out["converged"].sum() if "converged" in out else 0
    n_fallback = out["fallback_triggered"].sum() if "fallback_triggered" in out else 0
    print(f"\nDone. {len(out)} dates, {n_conv} converged, {n_fallback} fallback-to-cold triggered.", flush=True)
    print(f"Total time: {(time.time()-t_start)/60:.1f} minutes")
    if "mean_iv_error" in out:
        print(out["mean_iv_error"].describe() * 100)

if __name__ == "__main__":
    main()
