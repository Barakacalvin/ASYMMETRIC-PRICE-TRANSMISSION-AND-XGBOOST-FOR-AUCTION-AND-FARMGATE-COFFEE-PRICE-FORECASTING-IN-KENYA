"""
step3b_nardl_stage2.py -> Downstream Transmission execution
----------------------
Stage 2 NARDL: annual farmgate clean-equivalent price (USD/kg) on the within-coffee-year mean of monthly auction (USD/kg).

Implements Chapter 3 v3 sections 3.5.2, 3.5.3, 3.5.4.

Outputs
-------
Codes/_outputs/tables/nardl_stage2_coefficients.csv
Codes/_outputs/tables/nardl_stage2_summary.csv
Codes/_outputs/tables/nardl_stage2_cointegration.csv
Codes/_outputs/tables/nardl_stage2_diagnostics.csv
Codes/_outputs/data/stage2_features_for_xgb.csv
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey, het_breuschpagan,
)
from statsmodels.stats.stattools import jarque_bera

import config as C
from logging_setup import get_logger
log = get_logger(__name__)

warnings.filterwarnings("ignore")

# Narayan (2005) small-sample, Case III, k=2, n~25 (intercept, no trend)
PSS_CRIT_K2_ANNUAL_CASE3 = {
    "5pct"  : (4.094, 4.954),
    "10pct" : (3.395, 4.247),
    "1pct"  : (5.604, 6.620),
}
# ITER2 change 3.4: Narayan (2005) small-sample, Case II, k=2, n~25 (restricted intercept)
PSS_CRIT_K2_ANNUAL_CASE2 = {
    "5pct"  : (3.708, 4.371),
    "10pct" : (3.090, 3.738),
    "1pct"  : (5.018, 5.823),
}
# Asymptotic Pesaran-Shin-Smith Case III for cross-comparison with Narayan
PSS_CRIT_K2_ANNUAL_PESARAN_CASE3 = {
    "5pct"  : (3.79, 4.85),
    "10pct" : (3.17, 4.14),
    "1pct"  : (5.15, 6.36),
}
PSS_CRIT_K2_ANNUAL = PSS_CRIT_K2_ANNUAL_CASE3  # backward-compat alias


def cumulative_partial_sums(dx: pd.Series) -> tuple[pd.Series, pd.Series]:
    pos = dx.clip(lower=0).cumsum()
    neg = dx.clip(upper=0).cumsum()
    return pos, neg


def fit_stage2(annual: pd.DataFrame,
               max_lag: int = None,
               return_features: bool = True,
               force_lags: tuple | None = None) -> dict:
    """
    Estimate Stage 2 NARDL.

    force_lags : optional (m, n) tuple. If provided, skips SBC search and fits
                 exactly the (m, n) specification. Used by ITER2 change 1.2 to
                 produce a richer-dynamics sensitivity alongside the SBC pick.
    """
    if max_lag is None:
        max_lag = C.NARDL_S2_MAX_LAG
    a = annual.dropna(subset=["fg_usdkg_clean", "auc_usdkg_mean"]).copy()
    a = a.sort_values("coffee_year").reset_index(drop=True)
    a["lnFG"]  = np.log(a["fg_usdkg_clean"])
    a["lnAUC"] = np.log(a["auc_usdkg_mean"])
    a["dlnFG"] = a["lnFG"].diff()
    a["dlnAUC"] = a["lnAUC"].diff()
    a["AUC_pos_cum"], a["AUC_neg_cum"] = cumulative_partial_sums(a["dlnAUC"])

    if force_lags is not None:
        # ITER2 change 1.2: forced-dynamics specification (e.g., m=1, n=1)
        m_, n_ = int(force_lags[0]), int(force_lags[1])
        cand = _stage2_design(a, m_, n_)
        if cand is None:
            raise RuntimeError(f"Stage 2 forced specification ({m_},{n_}) infeasible.")
        best = {"bic": cand["model"].bic, "m": m_, "n": n_,
                "design": cand, "model": cand["model"]}
    else:
        best = {"bic": np.inf}
        for m in range(0, max_lag + 1):
            for n in range(0, max_lag + 1):
                cand = _stage2_design(a, m, n)
                if cand is None:
                    continue
                if cand["model"].bic < best["bic"]:
                    best = {"bic": cand["model"].bic, "m": m, "n": n,
                            "design": cand, "model": cand["model"]}
        if "design" not in best:
            raise RuntimeError("Stage 2 lag search returned no valid model.")

    m_, n_ = best["m"], best["n"]
    model = best["model"]
    names = list(model.params.index)

    theta1 = model.params["lnFG_lag1"]
    theta_pos = model.params["AUC_pos_lag1"]
    theta_neg = model.params["AUC_neg_lag1"]
    L_pos = -theta_pos / theta1
    L_neg = -theta_neg / theta1

    R = np.zeros(len(names))
    R[names.index("AUC_pos_lag1")] = 1
    R[names.index("AUC_neg_lag1")] = -1
    wald_lr = model.wald_test(R.reshape(1, -1), use_f=False)

    R3 = np.zeros((3, len(names)))
    R3[0, names.index("lnFG_lag1")] = 1
    R3[1, names.index("AUC_pos_lag1")] = 1
    R3[2, names.index("AUC_neg_lag1")] = 1
    bounds_F = float(np.squeeze(model.wald_test(R3, use_f=True).statistic))
    bdm_t = model.tvalues["lnFG_lag1"]

    bg_p = bp_p = jb_p = None
    try:
        bg = acorr_breusch_godfrey(model, nlags=2)
        bp = het_breuschpagan(model.resid, model.model.exog)
        jb = jarque_bera(model.resid)
        bg_p = bg[1]; bp_p = bp[1]; jb_p = jb[1]
    except Exception:
        pass

    half_life = (np.log(0.5) / np.log(1 + theta1)) if -1 < theta1 < 0 else np.nan

    coef = pd.DataFrame({
        "coef": model.params.round(5),
        "se"  : model.bse.round(5),
        "t"   : model.tvalues.round(3),
        "p"   : model.pvalues.round(4),
    })

    summary = pd.DataFrame([{
        "specification": "forced" if force_lags is not None else "SBC-selected",
        "n_obs": int(model.nobs), "m": m_, "n": n_,
        "theta1": round(theta1, 5),
        "L_pos": round(L_pos, 5), "L_neg": round(L_neg, 5),
        "wald_LR_p": round(float(np.squeeze(wald_lr.pvalue)), 4),
        "bounds_F": round(bounds_F, 4),
        "bdm_t": round(bdm_t, 4),
        "half_life_y": round(half_life, 3) if np.isfinite(half_life) else None,
        "R2": round(model.rsquared, 4),
        "BIC": round(model.bic, 3),
    }])

    # Build a single aligned frame, dropna() jointly, then map residuals
    # back to the full annual index so feat keeps all 26 rows (with NaN
    # for ECT in pre-decomposition years).
    static_df = a[["lnFG", "coffee_year", "AUC_pos_cum", "AUC_neg_cum"]].dropna()
    static = sm.OLS(static_df["lnFG"],
                    sm.add_constant(static_df[["AUC_pos_cum", "AUC_neg_cum"]])).fit()
    ect_full = pd.Series(static.resid.values, index=static_df.index).reindex(a.index)
    feat = a[["coffee_year", "AUC_pos_cum", "AUC_neg_cum"]].copy()
    feat["ECT"] = ect_full.values
    feat["ECT_lag1"] = feat["ECT"].shift(1)
    feat["lnFG_lag1"] = a["lnFG"].shift(1)
    feat["lnAUC_y"]   = a["lnAUC"]
    feat["lnAUC_y_lag1"] = a["lnAUC"].shift(1)

    return {
        "model": model, "coef": coef, "summary": summary,
        "bounds_F": bounds_F, "bdm_t": bdm_t,
        "L_pos": L_pos, "L_neg": L_neg,
        "diagnostics": {"BG2_p": bg_p, "BP_p": bp_p, "JB_p": jb_p},
        "ect_features": feat if return_features else None,
        "lags": (m_, n_),
    }


def _stage2_design(a: pd.DataFrame, m: int, n: int):
    d = a.copy()
    cols = ["lnFG_lag1", "AUC_pos_lag1", "AUC_neg_lag1"]
    d["lnFG_lag1"]    = d["lnFG"].shift(1)
    d["AUC_pos_lag1"] = d["AUC_pos_cum"].shift(1)
    d["AUC_neg_lag1"] = d["AUC_neg_cum"].shift(1)
    for L in range(1, m + 1):
        col = f"dlnFG_lag{L}"
        d[col] = d["dlnFG"].shift(L)
        cols.append(col)
    d["dAUC_pos"] = d["AUC_pos_cum"].diff()
    d["dAUC_neg"] = d["AUC_neg_cum"].diff()
    for L in range(0, n + 1):
        for s in ("pos", "neg"):
            col = f"dAUC_{s}_lag{L}"
            d[col] = d[f"dAUC_{s}"].shift(L)
            cols.append(col)
    est = d[["dlnFG"] + cols].dropna()
    if len(est) < len(cols) + 3:
        return None
    y = est["dlnFG"]
    X = sm.add_constant(est[cols])
    model = sm.OLS(y, X).fit()
    return {"model": model, "df_used": est, "y": y, "X": X}


def main():
    annual = pd.read_csv(C.DATA_OUT / "annual_clean.csv")

    # SBC-selected baseline
    res = fit_stage2(annual)
    res["coef"].reset_index().to_csv(C.TAB_DIR / "nardl_stage2_coefficients.csv", index=False)

    # ITER2 change 1.2: forced (m=1, n=1) richer-dynamics sensitivity
    try:
        res_rich = fit_stage2(annual, force_lags=(1, 1), return_features=False)
        rich_summary = res_rich["summary"]
        rich_coef    = res_rich["coef"].reset_index()
        rich_coef.to_csv(C.TAB_DIR / "nardl_stage2_coefficients_m1n1.csv", index=False)
        # combine summaries side-by-side for the report
        combined = pd.concat([res["summary"], rich_summary], ignore_index=True)
        combined.to_csv(C.TAB_DIR / "nardl_stage2_summary.csv", index=False)
        log.info("Stage 2 forced (m=1, n=1) sensitivity: bounds F=%.3f, BDM t=%.3f, BIC=%.2f",
                 res_rich["bounds_F"], res_rich["bdm_t"], rich_summary.iloc[0]["BIC"])
    except Exception as e:
        log.warning("Stage 2 (m=1, n=1) sensitivity failed: %s; reporting SBC baseline only", e)
        res["summary"].to_csv(C.TAB_DIR / "nardl_stage2_summary.csv", index=False)
    # ITER2 change 3.4: report Narayan Case III, Narayan Case II and Pesaran asymptotic Case III
    F = res["bounds_F"]
    def _verdict(F, lo, hi):
        if F > hi: return "cointegrated"
        if F < lo: return "no_cointegration"
        return "inconclusive"
    pd.DataFrame([
        {"test":   "Narayan small-sample Case III (k=2, n~25) — intercept, no trend",
         "F_stat": round(F, 4),
         "I0_5pct": PSS_CRIT_K2_ANNUAL_CASE3["5pct"][0],
         "I1_5pct": PSS_CRIT_K2_ANNUAL_CASE3["5pct"][1],
         "BDM_t":  round(res["bdm_t"], 4),
         "verdict": _verdict(F, *PSS_CRIT_K2_ANNUAL_CASE3["5pct"])},
        {"test":   "Narayan small-sample Case II (k=2, n~25) — restricted intercept",
         "F_stat": round(F, 4),
         "I0_5pct": PSS_CRIT_K2_ANNUAL_CASE2["5pct"][0],
         "I1_5pct": PSS_CRIT_K2_ANNUAL_CASE2["5pct"][1],
         "BDM_t":  round(res["bdm_t"], 4),
         "verdict": _verdict(F, *PSS_CRIT_K2_ANNUAL_CASE2["5pct"])},
        {"test":   "Pesaran asymptotic Case III (k=2) — intercept, no trend",
         "F_stat": round(F, 4),
         "I0_5pct": PSS_CRIT_K2_ANNUAL_PESARAN_CASE3["5pct"][0],
         "I1_5pct": PSS_CRIT_K2_ANNUAL_PESARAN_CASE3["5pct"][1],
         "BDM_t":  round(res["bdm_t"], 4),
         "verdict": _verdict(F, *PSS_CRIT_K2_ANNUAL_PESARAN_CASE3["5pct"])},
    ]).to_csv(C.TAB_DIR / "nardl_stage2_cointegration.csv", index=False)
    pd.DataFrame([res["diagnostics"]]).to_csv(C.TAB_DIR / "nardl_stage2_diagnostics.csv", index=False)
    if res["ect_features"] is not None:
        res["ect_features"].to_csv(C.DATA_OUT / "stage2_features_for_xgb.csv", index=False)
    log.info("\n=== Stage 2 NARDL ===")
    log.info(res["summary"].T)


if __name__ == "__main__":
    main()
