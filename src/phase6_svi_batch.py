"""
Phase 6 batch: validate the Andreasen-Huge arbitrage-free surface construction succeeds
reliably across all 192 dates (repricing fit quality). Unlike Heston (5 params/date) or SABR
(3 params/expiry/date), Andreasen-Huge has no simple closed-form parametrization to persist --
the "model" is the calibration procedure itself, reproducible on demand from the raw chain data
already on disk. So this batch validates viability at scale rather than producing a params file
Phase 9 will load; Phase 9 will re-run this same calibration per date when it needs Model 7's
(SVI/arb-free-surface) prices.
"""
import time
import pandas as pd
from phase4_heston_batch import monthly_calibration_dates
from phase4_heston_calibration import PROJ
from phase6_svi_local_vol import calibrate_andreasen_huge

def main():
    dates = monthly_calibration_dates()
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]

    results = []
    year_cache = {}
    t_start = time.time()
    for i, d in enumerate(dates):
        if d.year not in year_cache:
            year_cache.clear()
            year_cache[d.year] = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{d.year}.parquet")
        df = year_cache[d.year]
        try:
            surface, fit_df, calib_err = calibrate_andreasen_huge(d, df, vas, vas_params)
            n_valid = fit_df["iv_err"].notna().sum()
            results.append(dict(date=d, n_contracts=len(fit_df), n_valid_iv=n_valid,
                                 mean_iv_error=fit_df["iv_err"].abs().mean(),
                                 max_iv_error=fit_df["iv_err"].abs().max(),
                                 calib_err_avg=calib_err))
        except Exception as e:
            print(f"{d.date()}: EXCEPTION {e}", flush=True)
            results.append(dict(date=d, error=str(e)))
            continue
        if (i + 1) % 24 == 0:
            print(f"[{i+1}/{len(dates)}] {d.date()} mean|IVerr|={results[-1].get('mean_iv_error', float('nan'))*100:.3f}pp "
                  f"elapsed={(time.time()-t_start)/60:.1f}min", flush=True)

    out = pd.DataFrame(results)
    out.to_csv(f"{PROJ}\\results\\phase6_andreasenhuge_batch_validation.csv", index=False)
    print(f"\nDone. {len(out)} dates, {out['error'].notna().sum() if 'error' in out else 0} exceptions.")
    if "mean_iv_error" in out:
        print(f"mean|IVerr| overall: {out['mean_iv_error'].mean()*100:.3f}pp, "
              f"max: {out['mean_iv_error'].max()*100:.3f}pp")
    print(f"Total time: {(time.time()-t_start)/60:.1f}min")

if __name__ == "__main__":
    main()
