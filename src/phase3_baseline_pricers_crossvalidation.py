"""
Phase 3 cross-validation: check the four baseline pricers (BSM, Black-76, GBM-MC,
CRR binomial + BAW) agree with each other on a synthetic parameter grid, BEFORE
touching any real SPY quotes. Real-quote scoring is Phase 9's job, not this one's.

Checks:
  1. Black-76 == BSM when F = S*exp((r-q)T)          -- should match to numerical precision
  2. Put-call parity holds for BSM                     -- C - P = S*exp(-qT) - K*exp(-rT)
  3. GBM-MC converges to BSM within its own std error   -- statistical, not exact-match
  4. CRR European converges to BSM as n_steps grows
  5. CRR American >= CRR European (early-exercise value is never negative)
  6. CRR American >= intrinsic value (no option is worth less than exercising now)
  7. BAW approximates CRR American                      -- known approximation, not exact
"""
import numpy as np
import pandas as pd
from pricers import bs_price, black76_price, gbm_mc_price, crr_binomial_price, baw_american_price

RESULTS = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation\results"

MONEYNESS = [0.70, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30]
TENORS = [1/12, 3/12, 0.5, 1.0, 2.0, 3.0]
SIGMAS = [0.10, 0.15, 0.20, 0.30]
RATES = [0.01, 0.03, 0.05]
Q = 0.014
S0 = 100.0
CRR_STEPS = 400
MC_PATHS = 100_000

def main():
    rows = []
    for m in MONEYNESS:
        K = S0 * m
        for T in TENORS:
            for sigma in SIGMAS:
                for r in RATES:
                    F = S0 * np.exp((r - Q) * T)
                    for is_call in [True, False]:
                        bs = bs_price(S0, K, T, r, Q, sigma, is_call)
                        b76 = black76_price(F, K, T, r, sigma, is_call)
                        mc_price, mc_se = gbm_mc_price(S0, K, T, r, Q, sigma, is_call, n_paths=MC_PATHS, seed=42)
                        crr_e = crr_binomial_price(S0, K, T, r, Q, sigma, is_call, is_american=False, n_steps=CRR_STEPS)
                        crr_a = crr_binomial_price(S0, K, T, r, Q, sigma, is_call, is_american=True, n_steps=CRR_STEPS)
                        baw = baw_american_price(S0, K, T, r, Q, sigma, is_call)
                        intrinsic = max(S0 - K, 0.0) if is_call else max(K - S0, 0.0)
                        rows.append(dict(
                            moneyness=m, K=K, T=T, sigma=sigma, r=r, is_call=is_call,
                            bs=bs, black76=b76, mc=mc_price, mc_se=mc_se,
                            crr_euro=crr_e, crr_amer=crr_a, baw=baw, intrinsic=intrinsic,
                        ))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}\\phase3_crossvalidation_grid.csv", index=False)

    print(f"Grid size: {len(df)} (call+put combos)\n")

    # 1. Black-76 vs BSM
    diff = df["black76"] - df["bs"]
    print(f"[1] Black-76 vs BSM: max abs diff={diff.abs().max():.2e}, mean abs diff={diff.abs().mean():.2e}")

    # 2. Put-call parity on BSM prices
    calls = df[df["is_call"]].set_index(["moneyness", "T", "sigma", "r"])
    puts = df[~df["is_call"]].set_index(["moneyness", "T", "sigma", "r"])
    joined = calls[["bs", "K"]].join(puts[["bs"]], lsuffix="_c", rsuffix="_p")
    joined = joined.reset_index()
    lhs = joined["bs_c"] - joined["bs_p"]
    rhs = S0 * np.exp(-Q * joined["T"]) - joined["K"] * np.exp(-joined["r"] * joined["T"])
    pcp_diff = (lhs - rhs).abs()
    print(f"[2] Put-call parity (BSM): max abs diff={pcp_diff.max():.2e}, mean abs diff={pcp_diff.mean():.2e}")

    # 3. GBM-MC vs BSM, in units of MC standard errors
    z = (df["mc"] - df["bs"]) / df["mc_se"].replace(0, np.nan)
    n_flag = (z.abs() > 4).sum()
    print(f"[3] GBM-MC vs BSM: mean |z|={z.abs().mean():.3f}, max |z|={z.abs().max():.3f}, "
          f"rows with |z|>4: {n_flag}/{z.notna().sum()} (expect ~0 by chance at this sample size)")

    # 4. CRR European vs BSM. Relative error is meaningless for near-worthless deep-OTM
    # options (a $0.000001 price vs $0.0000009 is a "10% error" that matters to nobody) --
    # report both raw and a materiality-floored version (price > $0.01 on S0=$100) so a
    # division-by-near-zero artifact can't masquerade as a real convergence problem.
    MATERIAL = 0.01
    diff = (df["crr_euro"] - df["bs"]).abs()
    rel = (diff / df["bs"].abs().clip(lower=1e-6))
    material = df["bs"] > MATERIAL
    rel_mat = rel[material]
    print(f"[4] CRR-European (n={CRR_STEPS}) vs BSM: mean abs diff={diff.mean():.4f}, "
          f"median rel diff={rel.median()*100:.3f}%, max rel diff={rel.max()*100:.2f}% (raw, incl. sub-cent prices)")
    print(f"    materiality-floored (price>${MATERIAL}, {material.sum()}/{len(df)} rows): "
          f"median rel diff={rel_mat.median()*100:.3f}%, max rel diff={rel_mat.max()*100:.2f}%")

    # 5. Early-exercise premium >= 0
    premium = df["crr_amer"] - df["crr_euro"]
    n_neg = (premium < -1e-6).sum()
    print(f"[5] CRR American >= European: {n_neg} violations out of {len(df)} "
          f"(min premium={premium.min():.6f}, should be >= 0)")

    # 6. American >= intrinsic
    slack = df["crr_amer"] - df["intrinsic"]
    n_neg2 = (slack < -1e-6).sum()
    print(f"[6] CRR American >= intrinsic: {n_neg2} violations out of {len(df)} "
          f"(min slack={slack.min():.6f}, should be >= 0)")

    # 7. BAW vs CRR American -- same materiality floor as check 4. The unfiltered max here
    # is a real, known BAW limitation (quadratic approximation degrades for near-worthless
    # deep-OTM options), not an implementation bug -- but it's economically meaningless
    # (fractions of a cent on $100 notional), so don't let it drive the headline number.
    diff_baw = (df["baw"] - df["crr_amer"]).abs()
    rel_baw = diff_baw / df["crr_amer"].abs().clip(lower=1e-6)
    material_amer = df["crr_amer"] > MATERIAL
    rel_baw_mat = rel_baw[material_amer]
    print(f"[7] BAW vs CRR-American: mean abs diff={diff_baw.mean():.4f}, "
          f"median rel diff={rel_baw.median()*100:.3f}%, max rel diff={rel_baw.max()*100:.2f}% "
          f"(raw, incl. sub-cent prices)")
    print(f"    materiality-floored (price>${MATERIAL}, {material_amer.sum()}/{len(df)} rows): "
          f"median rel diff={rel_baw_mat.median()*100:.3f}%, max rel diff={rel_baw_mat.max()*100:.2f}%, "
          f"90th pct rel diff={rel_baw_mat.quantile(0.9)*100:.2f}%")
    worst = df.loc[rel_baw_mat.idxmax()]
    print(f"    worst material case: moneyness={worst['moneyness']}, T={worst['T']}, sigma={worst['sigma']}, "
          f"r={worst['r']}, is_call={worst['is_call']}, crr_amer={worst['crr_amer']:.4f}, baw={worst['baw']:.4f}")

    summary = pd.DataFrame([dict(
        black76_vs_bs_max_abs=(df["black76"] - df["bs"]).abs().max(),
        pcp_max_abs=pcp_diff.max(),
        mc_max_abs_z=z.abs().max(),
        crr_euro_vs_bs_median_rel_pct=rel.median() * 100,
        crr_euro_vs_bs_material_max_rel_pct=rel_mat.max() * 100,
        crr_amer_violations=n_neg,
        intrinsic_violations=n_neg2,
        baw_vs_crr_median_rel_pct=rel_baw.median() * 100,
        baw_vs_crr_material_max_rel_pct=rel_baw_mat.max() * 100,
    )])
    summary.to_csv(f"{RESULTS}\\phase3_crossvalidation_summary.csv", index=False)

if __name__ == "__main__":
    main()
