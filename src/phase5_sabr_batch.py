"""
Phase 5 batch: SABR per-expiry calibration across the same 192 monthly dates as Phase 4's
Heston batch (reusing monthly_calibration_dates() for consistency). Each date/expiry fit is a
small closed-form optimization -- expected to be fast, no warm-start/fallback infrastructure
needed unlike Heston. Still checkpoints periodically given this environment's demonstrated
background-kill pattern.
"""
import time
import pandas as pd
from phase4_heston_batch import monthly_calibration_dates
from phase5_sabr_calibration import calibrate_sabr_one_date
from phase4_heston_calibration import PROJ

CHECKPOINT_EVERY = 24  # dates (not expiry-rows) -- SABR fits are fast, checkpoint less often

def main():
    dates = monthly_calibration_dates()
    print(f"{len(dates)} dates to calibrate SABR for", flush=True)

    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    all_results = []
    try:
        prior = pd.read_csv(f"{PROJ}\\results\\phase5_sabr_batch_progress.csv")
        prior["date"] = pd.to_datetime(prior["date"])
        done_dates = set(prior["date"].unique())
        dates = [d for d in dates if d not in done_dates]
        all_results = prior.to_dict("records")
        print(f"RESUMING: {len(done_dates)} dates already done, {len(dates)} remaining", flush=True)
    except FileNotFoundError:
        print("No checkpoint, starting fresh", flush=True)

    year_cache = {}
    t_start = time.time()
    for i, d in enumerate(dates):
        if d.year not in year_cache:
            year_cache.clear()
            year_cache[d.year] = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{d.year}.parquet")
        df = year_cache[d.year]
        try:
            out = calibrate_sabr_one_date(d, df, vas=vas)
        except Exception as e:
            print(f"{d.date()}: EXCEPTION {e}", flush=True)
            continue
        all_results.extend(out.to_dict("records"))
        elapsed = time.time() - t_start
        print(f"[{i+1}/{len(dates)}] {d.date()}: {len(out)} expiries, "
              f"mean|err|={out['mean_iv_error'].mean()*100:.3f}pp, "
              f"all_converged={out['converged'].all()}, elapsed={elapsed/60:.1f}min", flush=True)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(all_results).to_csv(f"{PROJ}\\results\\phase5_sabr_batch_progress.csv", index=False)

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(f"{PROJ}\\results\\phase5_sabr_batch_progress.csv", index=False)
    out_df.to_parquet(f"{PROJ}\\data\\sabr_params_by_date_expiry.parquet", index=False)

    print(f"\nDone. {out_df['date'].nunique()} dates, {len(out_df)} date/expiry fits, "
          f"{out_df['converged'].mean()*100:.1f}% converged, "
          f"mean|IVerr|={out_df['mean_iv_error'].mean()*100:.3f}pp, "
          f"total time={(time.time()-t_start)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
