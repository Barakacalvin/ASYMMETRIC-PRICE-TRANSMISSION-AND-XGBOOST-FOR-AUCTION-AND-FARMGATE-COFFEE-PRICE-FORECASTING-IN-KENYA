"""
step3a_nardl_stage1.py
----------------------
Stage 1 NARDL: monthly auction price (USD/kg) on New York 'C' Futures (USD/kg)
and the USD/KES exchange rate.

Implements Chapter 3 v3 sections 3.5.1, 3.5.3, 3.5.4 and 3.7.6 (FX-free variant).

Outputs
-------
Codes/_outputs/tables/nardl_stage1_coefficients.csv
Codes/_outputs/tables/nardl_stage1_summary.csv          (theta1, L+, L-, half-life)
Codes/_outputs/tables/nardl_stage1_diagnostics.csv      (BG, BP, JB, CUSUM)
Codes/_outputs/tables/nardl_stage1_cointegration.csv    (PSS bounds F, BDM t)
Codes/_outputs/data/stage1_features_for_xgb.csv         (ECT_lag1, partial sums)

Function `fit_stage1(monthly, include_fx=True, max_lag=...)` is reused
inside the rolling-origin CV in step4a so that the NARDL is re-estimated
within every training window without leakage.
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey, het_breuschpagan,
)
from statsmodels.stats.stattools import jarque_bera
from scipy import stats

import config as C
from logging_setup import get_logger
log = get_logger(__name__)

warnings.filterwarnings("ignore")
np.random.seed(C.RANDOM_SEED)

# Pesaran, Shin and Smith (2001) Case III, intercept no trend, k = 2 regressors.
# These are asymptotic critical values. Narayan (2005) tables give finite-sample
# values closer to our n; Chapter 4 reports both bands.
# Pesaran-Shin-Smith (2001) Table CI(iii) Case III: unrestricted intercept, no trend
PSS_CRIT_K2_CASE3 = {  # k = number of long-run regressors (NY+, NY-)
    "5pct"  : (3.79, 4.85),  # I(0) lower, I(1) upper
    "10pct" : (3.17, 4.14),
    "1pct"  : (5.15, 6.36),
}
PSS_CRIT_K3_CASE3 = {  # k = NY+, NY-, FX
    "5pct"  : (3.23, 4.35),
    "10pct" : (2.72, 3.77),
    "1pct"  : (4.29, 5.61),
}
# ITER2 change 3.4: PSS Case II (restricted intercept, no trend) -- Table CI(ii)
PSS_CRIT_K2_CASE2 = {
    "5pct"  : (3.62, 4.16),
    "10pct" : (3.02, 3.51),
    "1pct"  : (4.94, 5.58),
}
PSS_CRIT_K3_CASE2 = {
    "5pct"  : (3.10, 3.87),
    "10pct" : (2.63, 3.35),
    "1pct"  : (4.13, 5.00),
}
# Aliases for backward-compatibility with existing function bodies
PSS_CRIT_K2 = PSS_CRIT_K2_CASE3
PSS_CRIT_K3 = PSS_CRIT_K3_CASE3


# ============================================================ shock series
def cumulative_partial_sums(dx: pd.Series) -> tuple[pd.Series, pd.Series]:
    pos = dx.clip(lower=0).cumsum()
    neg = dx.clip(upper=0).cumsum()
    return pos, neg


def select_lags_sbc(y: pd.Series, X: pd.DataFrame,
                    p_max: int = 12, q_max: int = 12,
                    short_run_cols: list[str] | None = None) -> int:
    """
    Coarse SBC-based lag selection: returns the number of own-lags p that
    minimises the Schwarz Bayesian Criterion. Short-run lags on the regressors
    are searched separately by a one-pass general-to-specific reduction.
    """
    best_p, best_bic = None, np.inf
    for p in range(0, p_max + 1):
        Xp = X.copy()
        for L in range(1, p + 1):
            Xp[f"d_y_lag{L}"] = y.shift(L)
        df = pd.concat([y.rename("__y__"), Xp], axis=1).dropna()
        if len(df) < Xp.shape[1] + 5:
            continue
        m = sm.OLS(df["__y__"], sm.add_constant(df.drop(columns="__y__"))).fit()
        if m.bic < best_bic:
            best_bic, best_p = m.bic, p
    return int(best_p) if best_p is not None else 0


# ============================================================ Stage 1 fit
def fit_stage1(monthly: pd.DataFrame,
               include_fx: bool = True,
               max_lag: int = None,
               international_col: str = "ny_usdkg",
               return_features: bool = True) -> dict:
    """
    Estimate the Stage 1 NARDL ECM on the monthly panel.

    Parameters
    ----------
    monthly : DataFrame from step1.
    include_fx : if False, runs the §3.7.6 sensitivity (FX-free baseline).
    international_col : which international reference; baseline 'ny_usdkg',
                        robustness 'ico_usdkg' or 'col_usdkg' (§3.7.2).
    """
    if max_lag is None:
        max_lag = C.NARDL_S1_MAX_LAG
    df = monthly.dropna(subset=[international_col, "auc_usdkg", "fx_usdkes"]).copy()
    # ITER2 change 1.1: BG-aware lag-selection toggle (default on for iteration 2)
    bg_aware = bool(getattr(C, "STAGE1_BG_AWARE", True))
    df = df.reset_index(drop=True)
    df["lnNY"]  = np.log(df[international_col])
    df["lnAUC"] = np.log(df["auc_usdkg"])
    df["lnFX"]  = np.log(df["fx_usdkes"])
    df["dlnNY"] = df["lnNY"].diff()
    df["dlnAUC"] = df["lnAUC"].diff()
    df["dlnFX"]  = df["lnFX"].diff()
    df["NY_pos_cum"], df["NY_neg_cum"] = cumulative_partial_sums(df["dlnNY"])

    # short-run terms (currents and lags)
    sr_cols = []
    p, q, r = 1, 1, 1                   # placeholder; refined by SBC pass below
    # Build a candidate full design matrix; SBC selects parsimonious specification.
    # Search over (p, q, r) with a tractable upper bound; tied to max_lag.
    # General-to-specific via t-stat pruning is applied after the SBC pick.

    best = {"bic": np.inf}
    for p in range(1, min(max_lag, 6) + 1):
        for q in range(0, min(max_lag, 6) + 1):
            for r in range(0, min(max_lag, 4) + 1) if include_fx else [0]:
                cand = _stage1_design(df, p, q, r, include_fx)
                if cand is None:
                    continue
                bic = cand["model"].bic
                if bic < best["bic"]:
                    best = {"bic": bic, "p": p, "q": q, "r": r,
                            "design": cand, "model": cand["model"]}

    if "design" not in best:
        raise RuntimeError("Stage 1 lag search returned no valid model.")
    p, q, r = best["p"], best["q"], best["r"]
    sbc_pq_r = (p, q, r)
    dgn = best["design"]
    model = best["model"]

    # ITER2 change 1.1: BG-aware extension. If BG(12) rejects at 5% on the
    # SBC-selected residuals, greedily extend lags (p first, then q, then r)
    # one at a time, refitting and re-testing, until BG(12) fails to reject
    # OR we hit caps (12, 6, 4).
    if bg_aware:
        try:
            bg_p_init = acorr_breusch_godfrey(model, nlags=12)[1]
        except Exception:
            bg_p_init = np.nan
        if np.isfinite(bg_p_init) and bg_p_init < 0.05:
            cap_p, cap_q, cap_r = 12, 6, 4
            best_bg = {"p": p, "q": q, "r": r,
                       "model": model, "design": dgn,
                       "bg12_p": bg_p_init}
            improved = True
            while improved and best_bg["bg12_p"] < 0.05:
                improved = False
                for which in ("p", "q", "r"):
                    cur = best_bg[which]
                    cap = {"p": cap_p, "q": cap_q, "r": cap_r}[which]
                    if cur >= cap:
                        continue
                    new_pqr = {"p": best_bg["p"], "q": best_bg["q"], "r": best_bg["r"]}
                    new_pqr[which] = cur + 1
                    cand = _stage1_design(df, new_pqr["p"], new_pqr["q"], new_pqr["r"], include_fx)
                    if cand is None:
                        continue
                    try:
                        bg_p_new = acorr_breusch_godfrey(cand["model"], nlags=12)[1]
                    except Exception:
                        continue
                    if not np.isfinite(bg_p_new):
                        continue
                    # Accept if BG p-value strictly improves toward non-rejection
                    if bg_p_new > best_bg["bg12_p"]:
                        best_bg.update({"p": new_pqr["p"], "q": new_pqr["q"],
                                        "r": new_pqr["r"],
                                        "model": cand["model"], "design": cand,
                                        "bg12_p": bg_p_new})
                        improved = True
            # If extension produced a non-rejecting BG, replace the model used
            # downstream. Otherwise keep the SBC pick (with its known BG flag).
            if best_bg["bg12_p"] >= 0.05 or (best_bg["bg12_p"] > bg_p_init):
                p, q, r = best_bg["p"], best_bg["q"], best_bg["r"]
                model = best_bg["model"]
                dgn   = best_bg["design"]

    # ---------------- coefficient table ----------------
    coef = pd.DataFrame({
        "coef"   : model.params.round(5),
        "hac_se" : model.bse.round(5),
        "t"      : model.tvalues.round(3),
        "p"      : model.pvalues.round(4),
    })

    # ---------------- long-run multipliers ----------------
    theta1 = model.params["lnAUC_lag1"]
    theta_pos = model.params["NY_pos_lag1"]
    theta_neg = model.params["NY_neg_lag1"]
    L_pos = -theta_pos / theta1
    L_neg = -theta_neg / theta1

    # Wald test on recovered long-run coefficients (L+ == L-)  via theta+ == theta-
    names = list(model.params.index)
    R = np.zeros(len(names))
    R[names.index("NY_pos_lag1")] = 1
    R[names.index("NY_neg_lag1")] = -1
    wald_lr = model.wald_test(R.reshape(1, -1), use_f=False)

    # short-run asymmetry: sum of pos coefs == sum of neg coefs
    pos_sr = [c for c in names if c.startswith("dNY_pos_lag")]
    neg_sr = [c for c in names if c.startswith("dNY_neg_lag")]
    R2 = np.zeros(len(names))
    for c in pos_sr: R2[names.index(c)] = 1
    for c in neg_sr: R2[names.index(c)] = -1
    wald_sr = model.wald_test(R2.reshape(1, -1), use_f=False)

    # ---------------- PSS bounds + BDM ----------------
    R3 = np.zeros((3 if include_fx else 3, len(names)))
    R3[0, names.index("lnAUC_lag1")] = 1
    R3[1, names.index("NY_pos_lag1")] = 1
    R3[2, names.index("NY_neg_lag1")] = 1
    bounds_F = float(np.squeeze(model.wald_test(R3, use_f=True).statistic))
    # BDM t-statistic (Banerjee, Dolado & Mestre 1998) on lnAUC_lag1
    bdm_t = model.tvalues["lnAUC_lag1"]

    # ---------------- diagnostics ----------------
    bg = acorr_breusch_godfrey(model, nlags=12)
    bp = het_breuschpagan(model.resid, model.model.exog)
    jb = jarque_bera(model.resid)

    # ---------------- residual-based ECT (confirmatory) ----------------
    # Build a single aligned frame, then dropna() jointly so endog and
    # exog share the same row index.
    static_cols = ["NY_pos_cum", "NY_neg_cum"] + (["lnFX"] if include_fx else [])
    keep_cols = ["lnAUC", "coffee_year", "coffee_year_month"] + static_cols
    static_df = df[keep_cols].dropna().reset_index(drop=True)
    static = sm.OLS(static_df["lnAUC"],
                    sm.add_constant(static_df[static_cols])).fit()
    df_ect = static_df[["coffee_year", "coffee_year_month"]].copy()
    df_ect["ECT"] = static.resid.values
    df_ect["ECT_lag1"] = df_ect["ECT"].shift(1)

    # ---------------- half-life from conditional theta1 ----------------
    half_life = (np.log(0.5) / np.log(1 + theta1)) if -1 < theta1 < 0 else np.nan

    # ---------------- assemble outputs ----------------
    summary = pd.DataFrame([{
        "n_obs"           : int(model.nobs),
        "p"               : p,  "q": q, "r": r,
        "p_sbc"           : sbc_pq_r[0], "q_sbc": sbc_pq_r[1], "r_sbc": sbc_pq_r[2],
        "selection"       : "BG-augmented" if bg_aware and (p, q, r) != sbc_pq_r else "SBC-only",
        "include_fx"      : include_fx,
        "international"   : international_col,
        "theta1"          : round(theta1, 5),
        "L_pos"           : round(L_pos,  5),
        "L_neg"           : round(L_neg,  5),
        "wald_LR_p"       : round(float(np.squeeze(wald_lr.pvalue)), 4),
        "wald_SR_p"       : round(float(np.squeeze(wald_sr.pvalue)), 4),
        "bounds_F"        : round(bounds_F, 4),
        "bdm_t"           : round(bdm_t, 4),
        "half_life_m"     : round(half_life, 3) if np.isfinite(half_life) else None,
        "R2"              : round(model.rsquared, 4),
        "BIC"             : round(model.bic, 3),
    }])
    diag = pd.DataFrame([{
        "BG12_stat" : bg[0], "BG12_p" : bg[1],
        "BP_stat"   : bp[0], "BP_p"   : bp[1],
        "JB_stat"   : jb[0], "JB_p"   : jb[1],
    }]).round(4)

    out = {
        "model": model,
        "coef": coef,
        "summary": summary,
        "diagnostics": diag,
        "bounds_F": bounds_F,
        "bdm_t": bdm_t,
        "L_pos": L_pos, "L_neg": L_neg,
        "ect_features": df_ect if return_features else None,
        "lags": (p, q, r),
    }
    return out


def _stage1_design(df: pd.DataFrame, p: int, q: int, r: int,
                   include_fx: bool):
    """Build design matrix for given (p, q, r) and fit OLS with HAC SEs."""
    d = df.copy()
    cols_X = ["lnAUC_lag1", "NY_pos_lag1", "NY_neg_lag1"]
    d["lnAUC_lag1"]  = d["lnAUC"].shift(1)
    d["NY_pos_lag1"] = d["NY_pos_cum"].shift(1)
    d["NY_neg_lag1"] = d["NY_neg_cum"].shift(1)
    if include_fx:
        d["lnFX_lag1"] = d["lnFX"].shift(1)
        cols_X.append("lnFX_lag1")
    for L in range(1, p + 1):
        col = f"dlnAUC_lag{L}"
        d[col] = d["dlnAUC"].shift(L)
        cols_X.append(col)
    d["dNY_pos"] = d["NY_pos_cum"].diff()
    d["dNY_neg"] = d["NY_neg_cum"].diff()
    for L in range(0, q + 1):
        for s in ("pos", "neg"):
            col = f"dNY_{s}_lag{L}"
            d[col] = d[f"dNY_{s}"].shift(L)
            cols_X.append(col)
    if include_fx:
        for L in range(0, r + 1):
            col = f"dlnFX_lag{L}"
            d[col] = d["dlnFX"].shift(L)
            cols_X.append(col)
    est = d[["dlnAUC"] + cols_X].dropna()
    if len(est) < len(cols_X) + 10:
        return None
    y = est["dlnAUC"]
    X = sm.add_constant(est[cols_X])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": C.HAC_MAXLAGS})
    return {"model": model, "y": y, "X": X, "df_used": est}


# ============================================================ main
def main():
    monthly = pd.read_csv(C.DATA_OUT / "monthly_clean.csv")

    # ITER2 changes 1.3 & 1.4: run four Stage 1 specifications side-by-side
    specs = [
        # (label, international_col, include_fx)
        ("NY + FX",         "ny_usdkg",  True),   # iteration-1 baseline
        ("NY (FX-free)",    "ny_usdkg",  False),  # FX-free sensitivity
        ("Colombian + FX",  "col_usdkg", True),   # ITER2 1.3 co-baseline
        ("Colombian (FX-free)", "col_usdkg", False),
    ]

    all_summaries, all_diags = [], []
    primary = None  # NY+FX is the canonical baseline for downstream features

    for label, intl, incfx in specs:
        try:
            res = fit_stage1(monthly, include_fx=incfx, international_col=intl)
        except Exception as e:
            log.warning("Spec %s failed: %s", label, e)
            continue
        s = res["summary"].copy()
        s.insert(0, "spec", label)
        d = res["diagnostics"].copy()
        d.insert(0, "spec", label)
        all_summaries.append(s)
        all_diags.append(d)

        # write per-spec coefficient table for traceability
        slug = label.replace(" + ", "_plus_").replace(" ", "_").replace("(", "").replace(")", "")
        res["coef"].reset_index().to_csv(
            C.TAB_DIR / f"nardl_stage1_coefficients_{slug}.csv", index=False)

        if label == "NY + FX":
            primary = res
            res["coef"].reset_index().to_csv(C.TAB_DIR / "nardl_stage1_coefficients.csv", index=False)

    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(C.TAB_DIR / "nardl_stage1_baselines.csv", index=False)
    if all_diags:
        pd.concat(all_diags, ignore_index=True).to_csv(C.TAB_DIR / "nardl_stage1_diagnostics.csv", index=False)

    # Backward-compatibility: NY+FX summary remains in the original filename
    if primary is not None:
        primary["summary"].to_csv(C.TAB_DIR / "nardl_stage1_summary.csv", index=False)
        if primary["ect_features"] is not None:
            primary["ect_features"].to_csv(C.DATA_OUT / "stage1_features_for_xgb.csv", index=False)
        # backward-compat: NY-only FX-free summary kept in its own file
        for s in all_summaries:
            if s.iloc[0]["spec"] == "NY (FX-free)":
                s.to_csv(C.TAB_DIR / "nardl_stage1_summary_nofx.csv", index=False)

    # Cointegration table now reports both Case III and Case II per change 3.4,
    # using the NY+FX baseline F-statistic (the canonical reference).
    F = primary["bounds_F"] if primary is not None else 0.0
    bdm_t = primary["bdm_t"] if primary is not None else 0.0
    def _verdict(F, lo, hi):
        if F > hi: return "cointegrated"
        if F < lo: return "no_cointegration"
        return "inconclusive"
    pd.DataFrame([
        {"test":   "PSS bounds Case III (k=3) — unrestricted intercept, no trend",
         "F_stat": round(F, 4),
         "I0_5pct": PSS_CRIT_K3_CASE3["5pct"][0],
         "I1_5pct": PSS_CRIT_K3_CASE3["5pct"][1],
         "BDM_t":  round(bdm_t, 4),
         "verdict": _verdict(F, *PSS_CRIT_K3_CASE3["5pct"])},
        {"test":   "PSS bounds Case II (k=3) — restricted intercept, no trend",
         "F_stat": round(F, 4),
         "I0_5pct": PSS_CRIT_K3_CASE2["5pct"][0],
         "I1_5pct": PSS_CRIT_K3_CASE2["5pct"][1],
         "BDM_t":  round(bdm_t, 4),
         "verdict": _verdict(F, *PSS_CRIT_K3_CASE2["5pct"])},
    ]).to_csv(C.TAB_DIR / "nardl_stage1_cointegration.csv", index=False)

    log.info("\n=== Stage 1 NARDL — four-spec baseline matrix ===")
    log.info(summary.to_string(index=False))
    log.info("\nDiagnostics across specs:")
    if all_diags:
        log.info(pd.concat(all_diags, ignore_index=True).to_string(index=False))
    log.info("Wrote Stage 1 outputs to %s", C.TAB_DIR)


if __name__ == "__main__":
    main()
