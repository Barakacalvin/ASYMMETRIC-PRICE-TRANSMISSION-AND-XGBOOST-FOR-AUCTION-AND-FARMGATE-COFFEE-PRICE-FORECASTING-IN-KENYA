"""
step4b_xgb_stage2_annual.py
---------------------------
Stage 2 XGBoost: annual farmgate clean-equivalent (USD/kg).
Six features per Chapter 3 v3 §3.6.4.

Two forecast origins are estimated:

  Baseline (end-of-coffee-year)        — uses fully realised auction-year
                                          information set (features 1, 2 are
                                          contemporaneous-year means).
  Pre-season (policy-oriented variant) — features 1, 2 replaced by lag-1; ECT
                                          and cumulative shocks built using
                                          information up to coffee year y-1
                                          only. (v3 §3.7.7)

Both origins:
- Re-estimate Stage 2 NARDL inside every training window (no leakage in ECT
  or cumulative shocks).
- Use Optuna TPE with 50 trials.
- Compute SHAP on validation predictions.
- Are paired against a NARDL forecast in DM with HLN correction.

Outputs
-------
Codes/_outputs/tables/xgb_stage2_metrics.csv      (baseline + pre-season)
Codes/_outputs/tables/xgb_stage2_dm.csv
Codes/_outputs/tables/xgb_stage2_shap_top.csv
Codes/_outputs/tables/xgb_stage2_predictions.csv
Codes/_outputs/models/xgb_stage2_best_params.json
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
from scipy import stats

import config as C
from logging_setup import get_logger
log = get_logger(__name__)
from step3b_nardl_stage2 import fit_stage2, cumulative_partial_sums

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(C.RANDOM_SEED)


# ============================================================ feature builder
FEAT_BASELINE = [
    "NY_Futures_USDkg",        # coffee-year mean (contemporaneous)
    "Auction_USDkg",           # coffee-year mean (contemporaneous)
    "Stage2_ECT_lag1",
    "AUC_Shock_Pos_cum",
    "AUC_Shock_Neg_cum",
    "Farmgate_USDkg_lag1",
]
FEAT_PRESEASON = [
    "NY_Futures_USDkg_lag1",
    "Auction_USDkg_lag1",
    "Stage2_ECT_lag1_lag1",        # ECT only up to y-1
    "AUC_Shock_Pos_cum_lag1",      # cumulative pos sums up to y-1
    "AUC_Shock_Neg_cum_lag1",      # cumulative neg sums up to y-1
    "Farmgate_USDkg_lag1",
]


def build_stage2_features(annual_train: pd.DataFrame,
                          ect_features: pd.DataFrame,
                          variant: str = "baseline") -> pd.DataFrame:
    """
    Build Stage 2 feature matrix for either 'baseline' (end-of-coffee-year)
    or 'preseason' (policy variant). 'ect_features' must come from a NARDL
    refit on data up to (and not including) the forecast year.
    """
    a = annual_train.copy().sort_values("coffee_year").reset_index(drop=True)
    feats = pd.DataFrame(index=a.index)
    feats["coffee_year"] = a["coffee_year"]
    feats["target"]      = np.log(a["fg_usdkg_clean"])

    if variant == "baseline":
        feats["NY_Futures_USDkg"]    = a["ny_usdkg_mean"]
        feats["Auction_USDkg"]       = a["auc_usdkg_mean"]
        feats["Stage2_ECT_lag1"]     = ect_features.set_index("coffee_year")["ECT_lag1"].reindex(a["coffee_year"]).values
        feats["AUC_Shock_Pos_cum"]   = ect_features.set_index("coffee_year")["AUC_pos_cum"].reindex(a["coffee_year"]).values
        feats["AUC_Shock_Neg_cum"]   = ect_features.set_index("coffee_year")["AUC_neg_cum"].reindex(a["coffee_year"]).values
        feats["Farmgate_USDkg_lag1"] = a["fg_usdkg_clean"].shift(1)
    elif variant == "preseason":
        feats["NY_Futures_USDkg_lag1"] = a["ny_usdkg_mean"].shift(1)
        feats["Auction_USDkg_lag1"]    = a["auc_usdkg_mean"].shift(1)
        feats["Stage2_ECT_lag1_lag1"]  = ect_features.set_index("coffee_year")["ECT_lag1"].shift(1).reindex(a["coffee_year"]).values
        feats["AUC_Shock_Pos_cum_lag1"] = ect_features.set_index("coffee_year")["AUC_pos_cum"].shift(1).reindex(a["coffee_year"]).values
        feats["AUC_Shock_Neg_cum_lag1"] = ect_features.set_index("coffee_year")["AUC_neg_cum"].shift(1).reindex(a["coffee_year"]).values
        feats["Farmgate_USDkg_lag1"]   = a["fg_usdkg_clean"].shift(1)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")
    return feats


def tune_xgb(X, y, search_space, n_trials, early_stopping):
    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            if spec[0] == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif spec[0] == "loguniform":
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif spec[0] == "uniform":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
        params.update({"objective": "reg:squarederror",
                       "verbosity": 0,
                       "random_state": C.RANDOM_SEED})
        n = len(X); cut = max(int(0.75 * n), n - 2)
        Xt, Xv = X[:cut], X[cut:]; yt, yv = y[:cut], y[cut:]
        if len(Xv) == 0:
            return 1e6
        m = xgb.XGBRegressor(**params, early_stopping_rounds=early_stopping)
        try:
            m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        except Exception:
            return 1e6
        pred = m.predict(Xv)
        return float(np.sqrt(np.mean((pred - yv) ** 2)))

    sampler = optuna.samplers.TPESampler(seed=C.RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def rolling_origin_cv(annual: pd.DataFrame, variant: str) -> dict:
    annual = annual.sort_values("coffee_year").reset_index(drop=True)
    n_init = C.XGB_STAGE2["initial_train_years"]
    feat_names = FEAT_BASELINE if variant == "baseline" else FEAT_PRESEASON
    pred_xgb, pred_nardl, actual, keys = [], [], [], []
    best_params_history = []
    best_params = None

    total_folds = len(annual) - n_init
    log.info("Stage 2 rolling-origin CV (variant=%s): %d folds planned",
             variant, total_folds)
    for t in range(n_init, len(annual)):
        train = annual.iloc[:t].copy()
        test  = annual.iloc[t:t+1].copy()

        # NARDL refit on train only
        try:
            nardl = fit_stage2(train, return_features=True)
        except Exception:
            continue
        ect = nardl["ect_features"]

        # Build features for train + test
        block = pd.concat([train, test], ignore_index=True)
        # Need to compute partial sums and ECT for the test row WITHOUT using its FG.
        # Trick: for the test row, the cumulative sums use auction info (already
        # observed in 'baseline' or up to y-1 in 'preseason'), but ECT comes from
        # the in-training static cointegrating regression.
        block_lnAUC = np.log(block["auc_usdkg_mean"])
        block_dlnAUC = block_lnAUC.diff()
        bpos, bneg = cumulative_partial_sums(block_dlnAUC)
        block_ect = ect.copy()
        if test["coffee_year"].iloc[0] not in block_ect["coffee_year"].values:
            # carry-forward last ECT and append cumulative sums for the test year
            new_row = {"coffee_year": test["coffee_year"].iloc[0],
                       "AUC_pos_cum": bpos.iloc[len(block) - 1],
                       "AUC_neg_cum": bneg.iloc[len(block) - 1],
                       "ECT": np.nan,
                       "ECT_lag1": ect["ECT"].iloc[-1] if len(ect) > 0 else np.nan}
            block_ect = pd.concat([block_ect, pd.DataFrame([new_row])], ignore_index=True)
        feats_block = build_stage2_features(block, block_ect, variant=variant)

        train_feat = feats_block.iloc[:-1].dropna(subset=feat_names + ["target"])
        test_feat  = feats_block.iloc[-1:].dropna(subset=feat_names)
        if len(train_feat) < 8 or len(test_feat) == 0:
            continue
        X_tr = train_feat[feat_names].values
        y_tr = train_feat["target"].values
        X_te = test_feat[feat_names].values
        y_te = float(np.log(test["fg_usdkg_clean"].iloc[0]))

        # Optuna tuning per fold (small n -> fast enough)
        best_params = tune_xgb(X_tr, y_tr,
                               C.XGB_STAGE2["search_space"],
                               C.XGB_STAGE2["optuna_trials"],
                               C.XGB_STAGE2["early_stopping_rounds"])
        best_params_history.append({"t": int(t),
                                     "year": test["coffee_year"].iloc[0],
                                     **best_params})
        m = xgb.XGBRegressor(**best_params,
                              objective="reg:squarederror",
                              verbosity=0,
                              random_state=C.RANDOM_SEED)
        m.fit(X_tr, y_tr, verbose=False)
        p_xgb = float(m.predict(X_te)[0])

        # NARDL one-step-ahead point forecast (level lnFG_y)
        p_nardl = _nardl_one_step_ahead_s2(nardl, train, test, variant)

        pred_xgb.append(p_xgb)
        pred_nardl.append(p_nardl)
        actual.append(y_te)
        keys.append(test["coffee_year"].iloc[0])
        log.info("  [%s fold %d/%d] year=%s  xgb=%.4f  nardl=%.4f  actual=%.4f",
                 variant, t - n_init + 1, total_folds,
                 test["coffee_year"].iloc[0], p_xgb, p_nardl, y_te)

    return {
        "pred_xgb": np.asarray(pred_xgb),
        "pred_nardl": np.asarray(pred_nardl),
        "actual": np.asarray(actual),
        "keys": keys,
        "best_params_history": best_params_history,
        "feat_names": feat_names,
    }


def _nardl_one_step_ahead_s2(nardl_result, train_annual, test_row, variant):
    """Stage 2 NARDL forecast for the test coffee year."""
    model = nardl_result["model"]
    m_lag, n_lag = nardl_result["lags"]
    block = pd.concat([train_annual, test_row], ignore_index=True)
    block = block.sort_values("coffee_year").reset_index(drop=True)
    block["lnFG"] = np.log(block["fg_usdkg_clean"])
    block["lnAUC"] = np.log(block["auc_usdkg_mean"])
    block["dlnFG"] = block["lnFG"].diff()
    block["dlnAUC"] = block["lnAUC"].diff()
    pos, neg = cumulative_partial_sums(block["dlnAUC"])
    block["AUC_pos_cum"] = pos; block["AUC_neg_cum"] = neg
    block["dAUC_pos"] = block["AUC_pos_cum"].diff()
    block["dAUC_neg"] = block["AUC_neg_cum"].diff()
    last = len(block) - 1
    feat = {"const": 1.0,
            "lnFG_lag1": block["lnFG"].iloc[last - 1],
            "AUC_pos_lag1": block["AUC_pos_cum"].iloc[last - 1],
            "AUC_neg_lag1": block["AUC_neg_cum"].iloc[last - 1]}
    for L in range(1, m_lag + 1):
        feat[f"dlnFG_lag{L}"] = block["dlnFG"].iloc[last - L]
    for L in range(0, n_lag + 1):
        if variant == "baseline":
            feat[f"dAUC_pos_lag{L}"] = block["dAUC_pos"].iloc[last - L]
            feat[f"dAUC_neg_lag{L}"] = block["dAUC_neg"].iloc[last - L]
        else:
            # pre-season: shift one extra period back
            feat[f"dAUC_pos_lag{L}"] = block["dAUC_pos"].iloc[last - L - 1]
            feat[f"dAUC_neg_lag{L}"] = block["dAUC_neg"].iloc[last - L - 1]
    x = pd.Series({k: feat.get(k, 0.0) for k in model.params.index})
    pred_d = float(np.dot(x.values, model.params.values))
    pred_lnFG = block["lnFG"].iloc[last - 1] + pred_d
    return pred_lnFG


def diebold_mariano_hln(y, p1, p2, h=1):
    e1 = np.asarray(y) - np.asarray(p1)
    e2 = np.asarray(y) - np.asarray(p2)
    d = e1**2 - e2**2
    n = len(d)
    if n < 3: return np.nan, np.nan, np.nan
    md = d.mean(); var = np.var(d, ddof=1)
    if var <= 0: return np.nan, np.nan, np.nan
    dm = md / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n) * dm
    p = 2 * (1 - stats.norm.cdf(abs(hln)))
    return dm, hln, p


# ============================================================ main
def main():
    annual = pd.read_csv(C.DATA_OUT / "annual_clean.csv")
    metrics_rows, dm_rows, pred_rows = [], [], []
    shap_tables = {}

    for variant in ("baseline", "preseason"):
        cv = rolling_origin_cv(annual, variant=variant)
        a = cv["actual"]; px = cv["pred_xgb"]; pn = cv["pred_nardl"]
        if len(a) == 0:
            log.info(f"[{variant}] no folds produced — skipping")
            continue
        rmse_x = float(np.sqrt(np.mean((a - px) ** 2)))
        rmse_n = float(np.sqrt(np.mean((a - pn) ** 2)))
        mape_x = float(np.mean(np.abs((np.exp(a) - np.exp(px)) / np.exp(a))) * 100)
        mape_n = float(np.mean(np.abs((np.exp(a) - np.exp(pn)) / np.exp(a))) * 100)
        metrics_rows += [
            {"variant": variant, "model": "XGBoost",
             "RMSE_log": round(rmse_x, 4), "MAPE_level_%": round(mape_x, 2)},
            {"variant": variant, "model": "NARDL",
             "RMSE_log": round(rmse_n, 4), "MAPE_level_%": round(mape_n, 2)},
        ]
        dm, hln, p = diebold_mariano_hln(a, px, pn)
        dm_rows.append({"variant": variant,
                        "comparison": "XGBoost vs NARDL (Stage 2)",
                        "DM": round(dm, 4) if dm == dm else None,
                        "HLN": round(hln, 4) if hln == hln else None,
                        "p_value": round(p, 4) if p == p else None,
                        "n_pairs": int(len(a))})
        for i, k in enumerate(cv["keys"]):
            pred_rows.append({"variant": variant, "coffee_year": k,
                              "actual_lnFG": a[i], "xgb_lnFG": px[i],
                              "nardl_lnFG": pn[i]})

        # SHAP on validation predictions
        ect_full = fit_stage2(annual, return_features=True)["ect_features"]
        feats_all = build_stage2_features(annual, ect_full, variant=variant
                                           ).dropna(subset=cv["feat_names"] + ["target"])
        X_val = feats_all[cv["feat_names"]].values
        last_params = cv["best_params_history"][-1].copy() if cv["best_params_history"] else {}
        for k in ("t", "year"):
            last_params.pop(k, None)
        final = xgb.XGBRegressor(**last_params,
                                  objective="reg:squarederror",
                                  verbosity=0,
                                  random_state=C.RANDOM_SEED)
        final.fit(X_val, feats_all["target"].values, verbose=False)
        explainer = shap.TreeExplainer(final)
        shap_vals = explainer.shap_values(X_val)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        shap_tables[variant] = pd.DataFrame({
            "variant": variant, "feature": cv["feat_names"],
            "mean_abs_shap": mean_abs.round(4),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    pd.DataFrame(metrics_rows).to_csv(C.TAB_DIR / "xgb_stage2_metrics.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(C.TAB_DIR / "xgb_stage2_dm.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(C.TAB_DIR / "xgb_stage2_predictions.csv", index=False)
    if shap_tables:
        pd.concat(shap_tables.values(), ignore_index=True).to_csv(
            C.TAB_DIR / "xgb_stage2_shap_top.csv", index=False)
    log.info("\n=== Stage 2 XGBoost ===")
    log.info(pd.DataFrame(metrics_rows).to_string(index=False))
    log.info(pd.DataFrame(dm_rows).to_string(index=False))


if __name__ == "__main__":
    main()
