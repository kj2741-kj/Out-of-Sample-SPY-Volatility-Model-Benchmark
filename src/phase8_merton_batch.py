"""
Phase 8 batch: Merton calibration across all 192 dates. Same resumable-checkpoint pattern
proven reliable in Phase 4's Heston batch (19 recoveries, never lost more than ~14min of work).
At ~90s/date this is a genuinely long run (~4.8hr) -- accepted after diminishing returns from
further optimization (vectorizing pricing only cut ~130s->109s; reducing optimizer effort/
n_terms got to ~90s). Reliable recovery matters more here than raw speed.
"""
import time
import pandas as pd
from phase4_heston_batch import monthly_calibration_dates
from phase4_heston_calibration import PROJ
from phase8_merton_calibration import calibrate_one_date

CHECKPOINT_EVERY = 4  # short interval given the slow per-date cost -- minimize lost work per kill
PROGRESS_CSV = f"{PROJ}\\results\\phase8_merton_batch_progress.csv"

def main():
    dates = monthly_calibration_dates()
    print(f"{len(dates)} dates to calibrate Merton for", flush=True)

    results = []
    try:
        prior = pd.read_csv(PROGRESS_CSV)
        prior["date"] = pd.to_datetime(prior["date"])
        done_dates = set(prior["date"].unique())
        dates = [d for d in dates if d not in done_dates]
        results = prior.to_dict("records")
        print(f"RESUMING: {len(done_dates)} dates already done, {len(dates)} remaining", flush=True)
    except FileNotFoundError:
        print("No checkpoint, starting fresh", flush=True)

    year_cache = {}
    t_start = time.time()
    n_already_done = len(results)
    total = n_already_done + len(dates)
    for i, d in enumerate(dates):
        if d.year not in year_cache:
            year_cache.clear()
            year_cache[d.year] = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{d.year}.parquet")
        df = year_cache[d.year]
        vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
        try:
            res = calibrate_one_date(d, df, vas)
        except Exception as e:
            print(f"{d.date()}: EXCEPTION {e}", flush=True)
            results.append(dict(date=d, converged=False, error=str(e)))
            continue
        results.append(res)
        elapsed = time.time() - t_start
        print(f"[{n_already_done+i+1}/{total}] {d.date()} obj={res['objective']:.5f} "
              f"conv={res['converged']} meanIVerr={res['mean_iv_error']*100:.2f}pp "
              f"lambda={res['lam']:.3f} elapsed={elapsed/60:.1f}min", flush=True)
        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(PROGRESS_CSV, index=False)

    out = pd.DataFrame(results)
    out.to_csv(PROGRESS_CSV, index=False)
    out.to_parquet(f"{PROJ}\\data\\merton_params_by_date.parquet", index=False)
    print(f"\nDone. {len(out)} dates, {out['converged'].sum()} converged, "
          f"mean|IVerr|={out['mean_iv_error'].mean()*100:.3f}pp, "
          f"total time={(time.time()-t_start)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
