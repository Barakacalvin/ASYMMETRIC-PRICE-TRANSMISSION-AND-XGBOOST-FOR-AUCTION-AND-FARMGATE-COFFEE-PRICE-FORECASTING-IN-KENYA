"""
step5_robustness.py
-------------------
Robustness, sensitivity and ablation suite (Chapter 3 v3 §3.7.2 to §3.7.5).

  3.7.2  Alternative international benchmarks (ICO, Colombian) re-fit Stage 1
  3.7.3  Stage 1 feature ablations
  3.7.4  Stage 2 feature ablations
  3.7.5  Cherry-payment descriptive comparison (5-year window)
  3.7.6  Stage 1 without exchange rate (already produced in step3a)
  3.7.7  Stage 2 pre-season variant (already produced in step4b)

Outputs
-------
Codes/_outputs/tables/robustness_alt_intl.csv
Codes/_outputs/tables/robustness_stage1_ablation.csv
Codes/_outputs/tables/robustness_stage2_ablation.csv
Codes/_outputs/tables/robustness_cherry.csv
"""
from __future__ import annotations
import warnings
import json
import numpy as np
import pandas as pd
import xgboost as xgb

import config as C
from logging_setup import get_logger
log = get_logger(__name__)
from step3a_nardl_stage1 import fit_stage1
from step3b_nardl_stage2 import fit_stage2
from step4a_xgb_stage1_monthly import build_stage1_features as build_s1_feats
from step4b_xgb_stage2_annual  import build_stage2_features as build_s2_feats, FEAT_BASELINE

warnings.filterwarnings("ignore")
np.random.seed(C.RANDOM_SEED)


# ============================================================ 3.7.2
def alt_international_benchmarks(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ref in ("ny_usdkg", "ico_usdkg", "col_usdkg"):
        try:
            r = fit_stage1(monthly, include_fx=True, international_col=ref,
                            return_features=False)
            s = r["summary"].iloc[0].to_dict()
            s["benchmark"] = ref
            rows.append(s)
        except Exception as e:
            rows.append({"benchmark": ref, "error": str(e)})
    return pd.DataFrame(rows)


# ============================================================ 3.7.3 / 3.7.4
def stage1_ablation(monthly: pd.DataFrame) -> pd.DataFrame:
    """Ablation on the in-sample Stage 1 XGBoost, full-sample fit, log-target.

    Three variants:
      a) drop volatility features (NY_vol3_lag1, NY_vol12_lag1)
      b) drop cumulative shocks    (NY_pos_lag1, NY_neg_lag1)
      c) drop ECT_lag1             (Stage1_ECT_lag1)
    """
    nardl = fit_stage1(monthly, include_fx=True, return_features=True)
    feats = build_s1_feats(monthly, nardl["ect_features"]).dropna()
    full_cols = ["NY_lag1","NY_lag2","NY_lag3","NY_lag12",
                  "FX_lag1","FX_lag2",
                  "NY_vol3_lag1","NY_vol12_lag1",
                  "NY_pos_lag1","NY_neg_lag1",
                  "Stage1_ECT_lag1","AUC_lag1"]
    variants = {
        "baseline":          full_cols,
        "drop_volatility":   [c for c in full_cols if "vol" not in c],
        "drop_shocks":       [c for c in full_cols if "_pos_" not in c and "_neg_" not in c],
        "drop_ECT":          [c for c in full_cols if c != "Stage1_ECT_lag1"],
    }
    rows = []
    y = feats["target"].values
    for tag, cols in variants.items():
        X = feats[cols].values
        m = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=300,
                             subsample=0.85, colsample_bytree=0.85,
                             objective="reg:squarederror", verbosity=0,
                             random_state=C.RANDOM_SEED)
        m.fit(X, y, verbose=False)
        pred = m.predict(X)
        rows.append({"variant": tag, "n_features": len(cols),
                     "in_sample_RMSE_log": float(np.sqrt(np.mean((pred - y)**2)))})
    return pd.DataFrame(rows)


def stage2_ablation(annual: pd.DataFrame) -> pd.DataFrame:
    """v3 §3.7.4: Stage 2 ablation removing ECT and cumulative shocks."""
    ect = fit_stage2(annual, return_features=True)["ect_features"]
    feats = build_s2_feats(annual, ect, variant="baseline").dropna()
    variants = {
        "baseline": FEAT_BASELINE,
        "drop_ECT_and_shocks": [c for c in FEAT_BASELINE
                                  if c not in ("Stage2_ECT_lag1",
                                               "AUC_Shock_Pos_cum",
                                               "AUC_Shock_Neg_cum")],
    }
    rows = []
    y = feats["target"].values
    for tag, cols in variants.items():
        X = feats[cols].values
        m = xgb.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=200,
                             subsample=0.85, colsample_bytree=0.85,
                             objective="reg:squarederror", verbosity=0,
                             random_state=C.RANDOM_SEED)
        m.fit(X, y, verbose=False)
        pred = m.predict(X)
        rows.append({"variant": tag, "n_features": len(cols),
                     "in_sample_RMSE_log": float(np.sqrt(np.mean((pred - y)**2)))})
    return pd.DataFrame(rows)


# ============================================================ 3.7.5
def cherry_descriptive(annual: pd.DataFrame,
                       payout_assumption: float = 0.80,
                       cherry_to_clean: float = 7.0) -> pd.DataFrame:
    """
    Five-year descriptive comparison: cherry-derived clean-equivalent vs
    AFA-published clean-equivalent. Per v3 §3.7.5 NO formal NARDL or XGBoost
    re-fit is undertaken on this sub-sample.

    cherry_to_clean ratio of 7.0 is a placeholder; replace with
    AFA-Yearbook published ratio when finalising Chapter 4.
    """
    a = annual.dropna(subset=["cherry_keskg", "fg_usdkg_clean", "fx_year_avg"]).copy()
    if len(a) == 0:
        return pd.DataFrame([{"note": "no cherry observations available"}])
    # cherry KES/kg cherry -> KES/kg clean -> USD/kg clean
    a["cherry_clean_keskg"] = a["cherry_keskg"] * cherry_to_clean / payout_assumption
    a["cherry_clean_usdkg"] = a["cherry_clean_keskg"] / a["fx_year_avg"]
    a["abs_diff_usdkg"]     = (a["cherry_clean_usdkg"] - a["fg_usdkg_clean"]).abs()
    a["pct_diff"]           = ((a["cherry_clean_usdkg"] - a["fg_usdkg_clean"])
                                / a["fg_usdkg_clean"]) * 100.0
    return a[["coffee_year", "cherry_keskg", "fg_usdkg_clean",
              "cherry_clean_usdkg", "abs_diff_usdkg", "pct_diff"]].round(4)


# ============================================================ main
def main():
    monthly = pd.read_csv(C.DATA_OUT / "monthly_clean.csv")
    annual  = pd.read_csv(C.DATA_OUT / "annual_clean.csv")

    alt = alt_international_benchmarks(monthly)
    alt.to_csv(C.TAB_DIR / "robustness_alt_intl.csv", index=False)

    s1 = stage1_ablation(monthly)
    s1.to_csv(C.TAB_DIR / "robustness_stage1_ablation.csv", index=False)

    s2 = stage2_ablation(annual)
    s2.to_csv(C.TAB_DIR / "robustness_stage2_ablation.csv", index=False)

    cherry = cherry_descriptive(annual)
    cherry.to_csv(C.TAB_DIR / "robustness_cherry.csv", index=False)

    log.info("\n=== 3.7.2 alt international benchmarks ===")
    log.info(alt.to_string(index=False))
    log.info("\n=== 3.7.3 Stage 1 ablation ===")
    log.info(s1.to_string(index=False))
    log.info("\n=== 3.7.4 Stage 2 ablation ===")
    log.info(s2.to_string(index=False))
    log.info("\n=== 3.7.5 cherry descriptive (5-year window) ===")
    log.info(cherry.to_string(index=False))


if __name__ == "__main__":
    main()
