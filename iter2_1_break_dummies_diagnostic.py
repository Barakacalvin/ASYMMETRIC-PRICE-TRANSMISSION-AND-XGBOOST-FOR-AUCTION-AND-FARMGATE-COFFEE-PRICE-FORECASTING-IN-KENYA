"""
iter2_1_break_dummies_diagnostic.py
-----------------------------------
Iteration 2.1 — I used this script for Diagnostics only.
The coding was done in 2 main iterations 1 and 2 with refinements done continuouslu to iteration 2.

Question: does the Stage 1 NARDL residual serial correlation (BG(12) p ≈ 0.0003 in the iteration-2 NY+FX baseline) reflect regime shifts that the asymmetric-cumulative-shock decomposition does not absorb?

Approach: re-fit Stage 1 NARDL (NY+FX, BG-augmented (1,2,0)) with five Binseg-derived break-period dummies entered as exogenous controls in BOTH the level equation and the short-run dynamics. Compare BG(12) p, bounds F, BDM t, and asymmetric long-run multipliers against the iteration-2 baseline.

I used this script for Diagnostics only. It does not replace the iteration-2 baseline or change any iteration-2 outputs. It writes a single CSV to _outputs/tables/nardl_stage1_break_dummies_diagnostic.csv and appends a log entry to the iteration-2 master log.

Run from this folder:
    python iter2_1_break_dummies_diagnostic.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

import config as C
from logging_setup import get_logger
from step3a_nardl_stage1 import (
    cumulative_partial_sums,
    PSS_CRIT_K3_CASE3, PSS_CRIT_K3_CASE2,
)

warnings.filterwarnings("ignore")
log = get_logger("iter2_1_diagnostic")


def _coffee_year_month_to_date(cy_str: str, cy_month: int) -> tuple[int, int]:
    """Convert ('1999/2000', 9) -> (calendar_year=2000, calendar_month=6).
    coffee_year_month: 1=Oct, 2=Nov, 3=Dec, 4=Jan, ... 12=Sep."""
    cy_start = int(cy_str.split("/")[0])
    if cy_month <= 3:
        return cy_start, cy_month + 9          # Oct, Nov, Dec
    return cy_start + 1, cy_month - 3          # Jan..Sep of cy_start+1


def load_breaks(monthly: pd.DataFrame) -> list[tuple[str, int, int]]:
    """Read the Binseg-detected break dates and convert to (label, year, month) tuples."""
    breaks = pd.read_csv(C.TAB_DIR / "breaks_binseg.csv")
    out = []
    for _, row in breaks.iterrows():
        tag = row["coffee_year_month"]
        cy, mtag = tag.split("-m")
        cy_month = int(mtag)
        cyear, cmonth = _coffee_year_month_to_date(cy, cy_month)
        out.append((tag, cyear, cmonth))
    return out


def add_break_dummies(df: pd.DataFrame, breaks: list[tuple[str, int, int]]) -> pd.DataFrame:
    """For each break (label, year, month), add a step-dummy column equal to 1
    from that calendar month onwards, 0 before."""
    df = df.copy()
    for tag, year, month in breaks:
        col = f"D_{tag.replace('/', '').replace('-', '_')}"
        is_after = ((df["calendar_year"] > year) |
                    ((df["calendar_year"] == year) & (df["calendar_month"] >= month)))
        df[col] = is_after.astype(int)
    dummy_cols = [f"D_{tag.replace('/', '').replace('-', '_')}" for tag, _, _ in breaks]
    return df, dummy_cols


def fit_stage1_with_dummies(monthly: pd.DataFrame,
                             dummy_cols: list[str],
                             p: int, q: int, r: int) -> dict:
    """
    Fit Stage 1 NARDL (NY+FX, given (p,q,r)) with break dummies as exogenous
    controls in the level equation. Returns dict with model, BG/BP/JB
    diagnostics, and the asymmetric long-run multipliers.
    """
    df = monthly.dropna(subset=["ny_usdkg", "auc_usdkg", "fx_usdkes"]).copy()
    df = df.reset_index(drop=True)
    df["lnNY"]   = np.log(df["ny_usdkg"])
    df["lnAUC"]  = np.log(df["auc_usdkg"])
    df["lnFX"]   = np.log(df["fx_usdkes"])
    df["dlnNY"]  = df["lnNY"].diff()
    df["dlnAUC"] = df["lnAUC"].diff()
    df["dlnFX"]  = df["lnFX"].diff()
    df["NY_pos_cum"], df["NY_neg_cum"] = cumulative_partial_sums(df["dlnNY"])

    # build ARDL design — same logic as _stage1_design but with dummies appended
    cols_X = ["lnAUC_lag1", "NY_pos_lag1", "NY_neg_lag1", "lnFX_lag1"]
    df["lnAUC_lag1"]  = df["lnAUC"].shift(1)
    df["NY_pos_lag1"] = df["NY_pos_cum"].shift(1)
    df["NY_neg_lag1"] = df["NY_neg_cum"].shift(1)
    df["lnFX_lag1"]   = df["lnFX"].shift(1)
    for L in range(1, p + 1):
        col = f"dlnAUC_lag{L}"
        df[col] = df["dlnAUC"].shift(L); cols_X.append(col)
    df["dNY_pos"] = df["NY_pos_cum"].diff()
    df["dNY_neg"] = df["NY_neg_cum"].diff()
    for L in range(0, q + 1):
        for s in ("pos", "neg"):
            col = f"dNY_{s}_lag{L}"
            df[col] = df[f"dNY_{s}"].shift(L); cols_X.append(col)
    for L in range(0, r + 1):
        col = f"dlnFX_lag{L}"
        df[col] = df["dlnFX"].shift(L); cols_X.append(col)

    # ITER2.1: append break dummies to the design
    cols_X = cols_X + dummy_cols

    est = df[["dlnAUC"] + cols_X].dropna()
    y = est["dlnAUC"]
    X = sm.add_constant(est[cols_X])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": C.HAC_MAXLAGS})

    theta1 = model.params["lnAUC_lag1"]
    theta_pos = model.params["NY_pos_lag1"]
    theta_neg = model.params["NY_neg_lag1"]
    L_pos = -theta_pos / theta1
    L_neg = -theta_neg / theta1

    names = list(model.params.index)
    R = np.zeros(len(names))
    R[names.index("NY_pos_lag1")] = 1
    R[names.index("NY_neg_lag1")] = -1
    wald_lr = model.wald_test(R.reshape(1, -1), use_f=False)

    R3 = np.zeros((3, len(names)))
    R3[0, names.index("lnAUC_lag1")] = 1
    R3[1, names.index("NY_pos_lag1")] = 1
    R3[2, names.index("NY_neg_lag1")] = 1
    bounds_F = float(np.squeeze(model.wald_test(R3, use_f=True).statistic))
    bdm_t = model.tvalues["lnAUC_lag1"]

    bg = acorr_breusch_godfrey(model, nlags=12)
    bp = het_breuschpagan(model.resid, model.model.exog)
    jb = jarque_bera(model.resid)
    half_life = (np.log(0.5) / np.log(1 + theta1)) if -1 < theta1 < 0 else np.nan

    return {
        "model": model,
        "n_obs": int(model.nobs),
        "p": p, "q": q, "r": r,
        "n_dummies": len(dummy_cols),
        "theta1": float(theta1),
        "L_pos": float(L_pos),
        "L_neg": float(L_neg),
        "wald_LR_p": float(np.squeeze(wald_lr.pvalue)),
        "bounds_F": float(bounds_F),
        "bdm_t": float(bdm_t),
        "half_life_m": float(half_life) if np.isfinite(half_life) else None,
        "BG12_stat": float(bg[0]), "BG12_p": float(bg[1]),
        "BP_stat":   float(bp[0]), "BP_p":   float(bp[1]),
        "JB_stat":   float(jb[0]), "JB_p":   float(jb[1]),
        "R2": float(model.rsquared),
        "BIC": float(model.bic),
    }


def main():
    monthly_path = C.DATA_OUT / "monthly_clean.csv"
    iter2_summary = pd.read_csv(C.TAB_DIR / "nardl_stage1_baselines.csv")
    iter2_diags   = pd.read_csv(C.TAB_DIR / "nardl_stage1_diagnostics.csv")

    # NY + FX baseline (BG-augmented (1, 2, 0)) is iter-2 row index 0
    iter2_ny = iter2_summary[iter2_summary["spec"] == "NY + FX"].iloc[0]
    iter2_ny_diag = iter2_diags[iter2_diags["spec"] == "NY + FX"].iloc[0]
    p, q, r = int(iter2_ny["p"]), int(iter2_ny["q"]), int(iter2_ny["r"])

    log.info("Iteration 2.1 — Stage 1 NARDL break-dummy diagnostic")
    log.info("  Reference: iter-2 NY+FX baseline (p=%d, q=%d, r=%d), BG12 p = %.4f",
             p, q, r, iter2_ny_diag["BG12_p"])

    monthly = pd.read_csv(monthly_path)
    breaks = load_breaks(monthly)
    log.info("  Binseg-derived break dates loaded: %d breaks", len(breaks))
    for tag, yr, mo in breaks:
        log.info("    %-18s -> %d-%02d", tag, yr, mo)

    monthly_d, dummy_cols = add_break_dummies(monthly, breaks)
    log.info("  Generated dummy columns: %s", dummy_cols)

    res = fit_stage1_with_dummies(monthly_d, dummy_cols, p, q, r)

    # Build the comparison row vs iter-2 baseline
    diag_table = pd.DataFrame([
        {"variant": "iter-2 baseline (NY+FX, no dummies)",
         "n_obs":     int(iter2_ny["n_obs"]),
         "p":         p, "q": q, "r": r, "n_dummies": 0,
         "theta1":    round(float(iter2_ny["theta1"]), 5),
         "L_pos":     round(float(iter2_ny["L_pos"]), 5),
         "L_neg":     round(float(iter2_ny["L_neg"]), 5),
         "wald_LR_p": round(float(iter2_ny["wald_LR_p"]), 4),
         "bounds_F":  round(float(iter2_ny["bounds_F"]), 4),
         "bdm_t":     round(float(iter2_ny["bdm_t"]), 4),
         "half_life_m": round(float(iter2_ny["half_life_m"]), 3),
         "BG12_p":    round(float(iter2_ny_diag["BG12_p"]), 4),
         "BIC":       round(float(iter2_ny["BIC"]), 3)},
        {"variant": "iter-2.1 diagnostic (NY+FX + 5 break dummies)",
         "n_obs":     res["n_obs"],
         "p":         res["p"], "q": res["q"], "r": res["r"],
         "n_dummies": res["n_dummies"],
         "theta1":    round(res["theta1"], 5),
         "L_pos":     round(res["L_pos"], 5),
         "L_neg":     round(res["L_neg"], 5),
         "wald_LR_p": round(res["wald_LR_p"], 4),
         "bounds_F":  round(res["bounds_F"], 4),
         "bdm_t":     round(res["bdm_t"], 4),
         "half_life_m": round(res["half_life_m"], 3) if res["half_life_m"] else None,
         "BG12_p":    round(res["BG12_p"], 4),
         "BIC":       round(res["BIC"], 3)},
    ])

    out_path = C.TAB_DIR / "nardl_stage1_break_dummies_diagnostic.csv"
    diag_table.to_csv(out_path, index=False)
    log.info("\n=== Stage 1 break-dummy diagnostic ===\n%s",
             diag_table.to_string(index=False))
    log.info("Wrote: %s", out_path)

    # Headline interpretation banner
    delta_bg = res["BG12_p"] - float(iter2_ny_diag["BG12_p"])
    delta_F  = res["bounds_F"] - float(iter2_ny["bounds_F"])
    log.info("\nVerdict:")
    log.info("  BG(12) p:  %.4f -> %.4f   (Δ = %+.4f)",
             float(iter2_ny_diag["BG12_p"]), res["BG12_p"], delta_bg)
    if res["BG12_p"] >= 0.05:
        log.info("  BG(12) NOW NON-REJECTING at 5%% — break dummies absorbed the residual SC.")
    elif res["BG12_p"] >= 0.01:
        log.info("  BG(12) STILL REJECTS at 5%% but no longer at 1%% — partial absorption.")
    else:
        log.info("  BG(12) STILL REJECTS at 1%% — residual SC is not regime-driven.")
    log.info("  Bounds F:  %.3f -> %.3f   (Δ = %+.3f)",
             float(iter2_ny["bounds_F"]), res["bounds_F"], delta_F)
    log.info("  L⁺:        %.4f -> %.4f", float(iter2_ny["L_pos"]), res["L_pos"])
    log.info("  L⁻:        %.4f -> %.4f", float(iter2_ny["L_neg"]), res["L_neg"])
    log.info("  Wald LR p: %.4f -> %.4f", float(iter2_ny["wald_LR_p"]), res["wald_LR_p"])

if __name__ == "__main__":
    main()
