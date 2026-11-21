"""
Phase 11 analysis: reads results/phase11_contract_days.csv (built by phase11_greeks_hedging.py)
and writes results/phase11_summary.txt plus three CSV tables.

Part 1  Model Greeks: delta of each model minus the market (BS at own vendor IV) delta, and the
        gamma ratio, by year and moneyness bucket.
Part 2  Daily delta hedge, long 1 option short delta shares, cash at r:
          e = dV - delta0*(dS + q*S0*dt) - r0*(V0 - delta0*S0)*dt
        Gain vs BS_market = 1 - SSE(model)/SSE(BS_market). Unhedged reduction = 1 - SSE/SSE(no hedge).
        95% intervals from a cluster bootstrap over calibration dates (24 clusters, 2,000 draws):
        contracts selected on the same date share the same market path, so they are not independent.
Part 3  Risk-based P&L explain with day-0 BS Greeks at the contract's own vendor IV:
          D   = delta*dS
          DG  = D + 0.5*gamma*dS^2
          DGV = DG + vega*dIV
          DGVT= DGV + theta*dt
        Unexplained share = mean|actual - predicted| / mean|actual|; Spearman rank correlation.
Only consecutive trading days in the panel are paired (a missing quote breaks the chain).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from phase4_heston_calibration import PROJ, Q_DIV

IN = f"{PROJ}\\results\\phase11_contract_days.csv"
OUT = f"{PROJ}\\results\\phase11_summary.txt"
MODELS = ["BS_market", "BS_calib", "SABR", "Merton", "LocalVol"]
STRESS = (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"))
RNG = np.random.default_rng(11)
NBOOT = 2000


def make_pairs(d):
    d = d.sort_values(["calib_date", "contractID", "date"]).copy()
    cal = np.array(sorted(d["date"].unique()))
    pos = {x: i for i, x in enumerate(cal)}
    d["pos"] = d["date"].map(pos)
    g = d.groupby(["calib_date", "contractID"], sort=False)
    nxt = g.shift(-1)
    keep = (nxt["pos"] - d["pos"]) == 1
    a, b = d[keep].reset_index(drop=True), nxt[keep].reset_index(drop=True)
    p = a.copy()
    p["date1"] = b["date"].values
    p["dt"] = (b["date"].values - a["date"].values) / np.timedelta64(1, "D") / 365.0
    p["dS"] = b["S"].values - a["S"].values
    p["dV"] = b["mid"].values - a["mid"].values
    p["dIV"] = b["iv"].values - a["iv"].values
    carry_cash = p["r"] * p["mid"] * p["dt"]
    p["e_none"] = p["dV"] - carry_cash
    for m in MODELS:
        dl = p[f"delta_{m}"]
        p[f"e_{m}"] = p["dV"] - dl * (p["dS"] + Q_DIV * p["S"] * p["dt"]) - p["r"] * (p["mid"] - dl * p["S"]) * p["dt"]
    p["pl_D"] = p["delta_BS_market"] * p["dS"]
    p["pl_DG"] = p["pl_D"] + 0.5 * p["gamma_BS_market"] * p["dS"] ** 2
    p["pl_DGV"] = p["pl_DG"] + p["vega_mkt"] * p["dIV"]
    p["pl_DGVT"] = p["pl_DGV"] + p["theta_mkt"] * p["dt"]
    p["year"] = pd.to_datetime(p["date"]).dt.year
    p["side"] = np.where(p["grid_moneyness"] < 0.975, "OTM puts", np.where(p["grid_moneyness"] > 1.025, "OTM calls", "ATM"))
    p["stress"] = (pd.to_datetime(p["date"]) >= STRESS[0]) & (pd.to_datetime(p["date"]) <= STRESS[1])
    return p.dropna(subset=[f"e_{m}" for m in MODELS])


def gain_table(p, label):
    rows = []
    cd = p["calib_date"].values
    clusters = np.unique(cd)
    idx = [np.where(cd == c)[0] for c in clusters]
    sse = {m: (p[f"e_{m}"] ** 2).values for m in MODELS + ["none"]}
    for m in MODELS:
        g = 1 - sse[m].sum() / sse["BS_market"].sum()
        u = 1 - sse[m].sum() / sse["none"].sum()
        boots = []
        for _ in range(NBOOT):
            pick = np.concatenate([idx[k] for k in RNG.integers(0, len(clusters), len(clusters))])
            boots.append(1 - sse[m][pick].sum() / sse["BS_market"][pick].sum())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(sample=label, model=m, n_pairs=len(p), gain_vs_BS_market=g, ci_lo=lo, ci_hi=hi,
                         var_reduction_vs_unhedged=u, rmse_hedge_err=np.sqrt(sse[m].mean())))
    return rows


def explain_table(p, label):
    rows = []
    act = p["dV"].values
    for step in ["pl_D", "pl_DG", "pl_DGV", "pl_DGVT"]:
        pred = p[step].values
        rows.append(dict(sample=label, step=step.replace("pl_", ""), n_pairs=len(p),
                         unexplained_share=np.mean(np.abs(act - pred)) / np.mean(np.abs(act)),
                         spearman=spearmanr(act, pred).statistic,
                         mean_abs_unexplained=np.mean(np.abs(act - pred))))
    return rows


def main():
    d = pd.read_csv(IN, parse_dates=["calib_date", "date"])
    d["year"] = d["date"].dt.year
    d["side"] = np.where(d["grid_moneyness"] < 0.975, "OTM puts", np.where(d["grid_moneyness"] > 1.025, "OTM calls", "ATM"))
    lines = [f"Phase 11 summary. Contract-days: {len(d)}, contracts: {d['contractID'].nunique()}, "
             f"calibration dates: {d['calib_date'].nunique()}, trading days: {d['date'].nunique()}"]

    # Part 1
    g1 = []
    for (y, s), x in d.groupby(["year", "side"]):
        for m in MODELS[1:]:
            g1.append(dict(year=y, side=s, model=m, n=len(x),
                           mean_delta_minus_market=(x[f"delta_{m}"] - x["delta_BS_market"]).mean(),
                           mean_abs_delta_diff=(x[f"delta_{m}"] - x["delta_BS_market"]).abs().mean(),
                           median_gamma_ratio=(x[f"gamma_{m}"] / x["gamma_BS_market"]).median()))
    t1 = pd.DataFrame(g1)
    t1.to_csv(f"{PROJ}\\results\\phase11_part1_greeks.csv", index=False)

    # Part 2 and 3
    p = make_pairs(d)
    subsets = [("All 2020-2021", p), ("2020", p[p["year"] == 2020]), ("2021", p[p["year"] == 2021]),
               ("Stress 2020-02-20..04-30", p[p["stress"]]), ("Ex-stress", p[~p["stress"]])]
    subsets += [(f"Side: {s}", p[p["side"] == s]) for s in ["OTM puts", "ATM", "OTM calls"]]
    t2 = pd.DataFrame([r for lab, x in subsets if len(x) > 50 for r in gain_table(x, lab)])
    t2.to_csv(f"{PROJ}\\results\\phase11_part2_hedging.csv", index=False)
    t3 = pd.DataFrame([r for lab, x in subsets[:5] if len(x) > 50 for r in explain_table(x, lab)])
    t3.to_csv(f"{PROJ}\\results\\phase11_part3_pl_explain.csv", index=False)

    pd.set_option("display.width", 200)
    lines += ["", "PART 1: delta vs market BS delta (model minus market), gamma ratio",
              t1.round(4).to_string(index=False),
              "", f"PART 2: daily delta hedge, {len(p)} contract-day pairs",
              t2.round(4).to_string(index=False),
              "", "PART 3: P&L explain with market BS Greeks",
              t3.round(4).to_string(index=False)]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
