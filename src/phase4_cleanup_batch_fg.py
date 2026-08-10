"""Process a specific list of dates' cleanup re-calibration in the FOREGROUND (no
backgrounding), called a few dates at a time from Bash directly. Background runs were
getting killed by something outside this script's control (confirmed via a direct
foreground test of 2014-02-14 completing cleanly in 234s with no crash) -- foreground
avoids that entirely, at the cost of needing several shorter Bash calls instead of one
long background one.
"""
import sys
import pandas as pd
from phase4_heston_calibration import calibrate_one_date, PROJ

def main(date_strs):
    prog = pd.read_csv(f"{PROJ}\\results\\phase4_heston_batch_progress.csv")
    prog["date"] = pd.to_datetime(prog["date"])
    if "cleanup_done" not in prog.columns:
        prog["cleanup_done"] = False
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]

    year_cache = {}
    for date_str in date_strs:
        d = pd.Timestamp(date_str)
        row = prog[prog["date"] == d].iloc[0]
        if row.get("cleanup_done") is True:
            print(f"{d.date()}: already done, skipping")
            continue
        if d.year not in year_cache:
            year_cache.clear()
            year_cache[d.year] = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{d.year}.parquet")
        df_year = year_cache[d.year]

        print(f"{d.date()}: re-calibrating cold (was converged={row['converged']}, obj={row['objective']:.5f})...")
        res = calibrate_one_date(d, df_year, vas, vas_params, warm_start_x=None, verbose=True)
        old_obj, old_conv = row["objective"], row["converged"]
        if res.get("converged") and (not old_conv or res["objective"] < old_obj):
            for col in ["kappa", "theta", "xi", "rho", "v0", "feller", "objective",
                        "converged", "mode", "mean_iv_error"]:
                prog.loc[prog["date"] == d, col] = res.get(col)
            prog.loc[prog["date"] == d, "fallback_triggered"] = True
            print(f"  IMPROVED: converged={res['converged']} obj={res['objective']:.5f} "
                  f"meanIVerr={res['mean_iv_error']*100:.2f}pp")
        else:
            print(f"  NO IMPROVEMENT (new obj={res.get('objective')} conv={res.get('converged')}) "
                  f"-- keeping original, genuinely hard to fit")
        prog.loc[prog["date"] == d, "cleanup_done"] = True
        prog.to_csv(f"{PROJ}\\results\\phase4_heston_batch_progress.csv", index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
