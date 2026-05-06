"""
step4a_xgb_stage1_monthly.py -> Models the upstream prediction i.e the Auction Prices with reference to Global indicators (ICIP, NY Futures and the Columbian Milds)
----------------------------
Stage 1 XGBoost predictive comparator: monthly auction price (USD/kg).
Twelve features per Chapter 3 v3 §3.6.3 (one-month-ahead horizon, all features lagged).

Implements:
- v3 §3.6.2: gradient-boosting objective
- v3 §3.6.3: 12-feature catalogue
- v3 §3.6.5: rolling-origin CV with NARDL re-estimation INSIDE every training
              window (no look-ahead leakage in ECT or partial sums)
- v3 §3.6.6: TreeSHAP on validation predictions (not full-sample fits)
- v3 §3.7.1: DM (Harvey-Leybourne-Newbold corrected) vs NARDL forecast

Outputs
-------
Codes/_outputs/tables/xgb_stage1_metrics.csv
Codes/_outputs/tables/xgb_stage1_dm.csv
Codes/_outputs/tables/xgb_stage1_shap_top.csv
Codes/_outputs/tables/xgb_stage1_predictions.csv
Codes/_outputs/models/xgb_stage1_best_params.json
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
from step3a_nardl_stage1 import fit_stage1, cumulative_partial_sums

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(C.RANDOM_SEED)


# -> feature builder
def build_stage1_features(monthly_train: pd.DataFrame,
                          ect_features: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 12-feature catalogue using ONLY data observed up to each row's
    forecast origin (one-month-ahead prediction, all features lagged).
    """
    df = monthly_train.dropna(subset=["ny_usdkg", "auc_usdkg", "fx_usdkes"]).copy()
    df = df.sort_values(["coffee_year", "coffee_year_month"]).reset_index(drop=True)
    df["lnNY"]  = np.log(df["ny_usdkg"])
    df["dlnNY"] = df["lnNY"].diff()
    df["NY_pos_cum"], df["NY_neg_cum"] = cumulative_partial_sums(df["dlnNY"])
    feats = pd.DataFrame(index=df.index)
    feats["NY_lag1"]     = df["ny_usdkg"].shift(1)
    feats["NY_lag2"]     = df["ny_usdkg"].shift(2)
    feats["NY_lag3"]     = df["ny_usdkg"].shift(3)
    feats["NY_lag12"]    = df["ny_usdkg"].shift(12)
    feats["FX_lag1"]     = df["fx_usdkes"].shift(1)
    feats["FX_lag2"]     = df["fx_usdkes"].shift(2)
    feats["NY_vol3_lag1"]  = df["ny_usdkg"].rolling(3).std().shift(1)
    feats["NY_vol12_lag1"] = df["ny_usdkg"].rolling(12).std().shift(1)
    # cumulative shocks aligned by (coffee_year, coffee_year_month)
    cs = df[["coffee_year", "coffee_year_month",
             "NY_pos_cum", "NY_neg_cum"]].copy()
    cs["NY_pos_lag1"] = cs["NY_pos_cum"].shift(1)
    cs["NY_neg_lag1"] = cs["NY_neg_cum"].shift(1)
    feats = feats.join(cs[["NY_pos_lag1", "NY_neg_lag1"]])
    # ECT lag from NARDL re-estimation.
    # We merge the un-shifted ECT (training-only) and compute lag-1 AFTER
    # the merge so that the test row's Stage1_ECT_lag1 = last training
    # row's ECT — which is the correct value because the NARDL is refit
    # on training data and the test row's lag-1 sits inside that window.
    # ffill before shift handles any internal gaps where the static
    # cointegrating regression dropped a row (e.g., FX-gap months).
    if ect_features is not None and len(ect_features) > 0 and "ECT" in ect_features.columns:
        ect = ect_features[["coffee_year", "coffee_year_month", "ECT"]]
        df_keys = df[["coffee_year", "coffee_year_month"]].reset_index(drop=True)
        merged = df_keys.merge(ect, on=["coffee_year", "coffee_year_month"], how="left")
        feats["Stage1_ECT_lag1"] = merged["ECT"].ffill().shift(1).values
    else:
        feats["Stage1_ECT_lag1"] = np.nan
    feats["AUC_lag1"] = df["auc_usdkg"].shift(1)
    feats["target"]   = np.log(df["auc_usdkg"])      # log-target as in v3
    feats["coffee_year"] = df["coffee_year"]
    feats["coffee_year_month"] = df["coffee_year_month"]
    return feats


# -> Optuna tuner
def tune_xgb(X: np.ndarray, y: np.ndarray,
             search_space: dict, n_trials: int,
             early_stopping: int) -> dict:
    """Bayesian hyperparameter search using TPE sampler."""
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
        # internal split on the training window only
        n = len(X)
        cut = int(0.8 * n)
        Xt, Xv = X[:cut], X[cut:]
        yt, yv = y[:cut], y[cut:]
        m = xgb.XGBRegressor(**params, early_stopping_rounds=early_stopping)
        m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        pred = m.predict(Xv)
        return float(np.sqrt(np.mean((pred - yv) ** 2)))

    sampler = optuna.samplers.TPESampler(seed=C.RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# -> rolling CV
def rolling_origin_cv(monthly: pd.DataFrame) -> dict:
    """
    Expanding-window rolling-origin CV. NARDL is re-estimated inside every
    training window before features are built, ensuring no look-ahead leakage
    in ECT or partial-sum features.
    """
    monthly = monthly.sort_values(["coffee_year", "coffee_year_month"]).reset_index(drop=True)
    n_init  = C.XGB_STAGE1["initial_train_months"]
    n_total = len(monthly)
    pred_xgb, pred_nardl, actual = [], [], []
    keys = []
    best_params_history = []

    feat_names = ["NY_lag1","NY_lag2","NY_lag3","NY_lag12",
                  "FX_lag1","FX_lag2",
                  "NY_vol3_lag1","NY_vol12_lag1",
                  "NY_pos_lag1","NY_neg_lag1",
                  "Stage1_ECT_lag1","AUC_lag1",
                  "nardl_pred_t"]   # ITER2 change 2.2: hybrid feature

    # only retune Optuna once per K folds to keep runtime reasonable
    retune_every = max(1, (n_total - n_init) // 5)
    best_params = None
    total_folds = n_total - n_init
    log.info("Stage 1 rolling-origin CV: %d folds planned (initial train=%d, total months=%d)",
             total_folds, n_init, n_total)
    t_cv0 = __import__("time").perf_counter()
    skip_reasons = {}
    for t in range(n_init, n_total):
        train = monthly.iloc[:t].copy()
        test  = monthly.iloc[t:t+1].copy()

        # 1) re-estimate NARDL on the training window
        try:
            nardl = fit_stage1(train, include_fx=True, return_features=True)
        except Exception as e:
            reason = f"NARDL fit failed: {type(e).__name__}: {e}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if skip_reasons[reason] <= 3:
                log.warning("  fold %d skipped (NARDL fit): %s", t - n_init + 1, e)
            continue
        ect_train = nardl["ect_features"]

        # 2) build features for train + test together
        block = pd.concat([train, test], ignore_index=True)
        feats_block = build_stage1_features(block, ect_train)

        # ITER2 change 2.2: attach the NARDL fitted/predicted lnAUC for every
        # row of `block`. For training rows this is the in-sample fitted level
        # implied by NARDL; for the test row it is the one-step-ahead forecast.
        # All values use NARDL coefficients fit on `train` only — no leakage.
        nardl_pred_block = _nardl_fitted_levels_for_block(nardl, block)
        feats_block["nardl_pred_t"] = nardl_pred_block

        train_feat = feats_block.iloc[:-1].dropna(subset=feat_names + ["target"])
        test_feat  = feats_block.iloc[-1:].dropna(subset=feat_names)
        if len(train_feat) < 20 or len(test_feat) == 0:
            reason = f"insufficient features (train_feat={len(train_feat)}, test_feat={len(test_feat)})"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if skip_reasons[reason] <= 3:
                log.warning("  fold %d skipped: %s", t - n_init + 1, reason)
            continue
        X_tr = train_feat[feat_names].values
        y_tr = train_feat["target"].values
        X_te = test_feat[feat_names].values
        y_te = float(np.log(test["auc_usdkg"].iloc[0])) if pd.notna(test["auc_usdkg"].iloc[0]) else None
        if y_te is None:
            reason = "target NaN (e.g., recess month)"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        # 3) Optuna retune at intervals
        if best_params is None or (t - n_init) % retune_every == 0:
            log.info("  [fold %d/%d] Optuna retune (TPE, %d trials)",
                     t - n_init + 1, total_folds,
                     C.XGB_STAGE1["optuna_trials"])
            best_params = tune_xgb(X_tr, y_tr,
                                   C.XGB_STAGE1["search_space"],
                                   C.XGB_STAGE1["optuna_trials"],
                                   C.XGB_STAGE1["early_stopping_rounds"])
            best_params_history.append({"t": int(t), **best_params})

        m = xgb.XGBRegressor(**best_params,
                              objective="reg:squarederror",
                              verbosity=0,
                              random_state=C.RANDOM_SEED)
        m.fit(X_tr, y_tr, verbose=False)
        p_xgb = float(m.predict(X_te)[0])

        # 4) NARDL one-step-ahead point forecast (level form, then take log)
        try:
            p_nardl = _nardl_one_step_ahead(nardl, train, test)
        except Exception as e:
            reason = f"NARDL forecast failed: {type(e).__name__}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if skip_reasons[reason] <= 3:
                log.warning("  fold %d skipped (NARDL fcst): %s", t - n_init + 1, e)
            continue

        pred_xgb.append(p_xgb)
        pred_nardl.append(p_nardl)
        actual.append(y_te)
        keys.append((test["coffee_year"].iloc[0],
                     int(test["coffee_year_month"].iloc[0])))
        if (t - n_init + 1) % 25 == 0 or (t + 1) == n_total:
            elapsed = __import__("time").perf_counter() - t_cv0
            log.info("  progress: fold %d/%d (%.0f%%) elapsed %.1fs",
                     t - n_init + 1, total_folds,
                     100.0 * (t - n_init + 1) / total_folds, elapsed)

    n_done = len(actual)
    log.info("Stage 1 CV finished: %d successful / %d planned (%d skipped)",
             n_done, total_folds, total_folds - n_done)
    if skip_reasons:
        log.info("  skip-reason tally:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            log.info("    %4d x %s", count, reason)
    return {
        "pred_xgb": np.asarray(pred_xgb),
        "pred_nardl": np.asarray(pred_nardl),
        "actual": np.asarray(actual),
        "keys": keys,
        "best_params_history": best_params_history,
        "feat_names": feat_names,
    }


def _nardl_one_step_ahead(nardl_result: dict,
                          train_monthly: pd.DataFrame,
                          test_row: pd.DataFrame) -> float:
    """
    Compute the NARDL Stage 1 one-step-ahead point forecast for the held-out
    month using the conditional ECM with coefficients from the training-window
    fit. Returns the predicted log auction price.
    """
    model = nardl_result["model"]
    p, q, r = nardl_result["lags"]
    full = pd.concat([train_monthly, test_row], ignore_index=True)
    full = full.dropna(subset=["ny_usdkg", "auc_usdkg", "fx_usdkes"]).reset_index(drop=True)
    full["lnNY"]  = np.log(full["ny_usdkg"])
    full["lnAUC"] = np.log(full["auc_usdkg"])
    full["lnFX"]  = np.log(full["fx_usdkes"])
    full["dlnNY"] = full["lnNY"].diff()
    full["dlnAUC"] = full["lnAUC"].diff()
    full["dlnFX"]  = full["lnFX"].diff()
    pos, neg = cumulative_partial_sums(full["dlnNY"])
    full["NY_pos_cum"] = pos; full["NY_neg_cum"] = neg
    last = len(full) - 1                # the test row position in `full`
    # Build feature row for the model
    feat = {"const": 1.0,
            "lnAUC_lag1" : full["lnAUC"].iloc[last - 1],
            "NY_pos_lag1": full["NY_pos_cum"].iloc[last - 1],
            "NY_neg_lag1": full["NY_neg_cum"].iloc[last - 1]}
    # FX optional (we built baseline with FX)
    if "lnFX_lag1" in model.params.index:
        feat["lnFX_lag1"] = full["lnFX"].iloc[last - 1]
    for L in range(1, p + 1):
        feat[f"dlnAUC_lag{L}"] = full["dlnAUC"].iloc[last - L]
    full["dNY_pos"] = full["NY_pos_cum"].diff()
    full["dNY_neg"] = full["NY_neg_cum"].diff()
    for L in range(0, q + 1):
        feat[f"dNY_pos_lag{L}"] = full["dNY_pos"].iloc[last - L]
        feat[f"dNY_neg_lag{L}"] = full["dNY_neg"].iloc[last - L]
    if "lnFX_lag1" in model.params.index:
        for L in range(0, r + 1):
            feat[f"dlnFX_lag{L}"] = full["dlnFX"].iloc[last - L]

    # Predict d(lnAUC) then add to lnAUC_{t-1}
    x = pd.Series({k: feat.get(k, 0.0) for k in model.params.index})
    pred_d = float(np.dot(x.values, model.params.values))
    pred_lnAUC = full["lnAUC"].iloc[last - 1] + pred_d
    return pred_lnAUC


# -> DM test
def _nardl_fitted_levels_for_block(nardl_result, block: pd.DataFrame) -> np.ndarray:
    """
    ITER2 change 2.2 helper (fixed alignment).

    Returns one fitted-value per row of `block.dropna(subset=[ny, auc, fx])`,
    matching the index that `build_stage1_features()` produces. The consumer
    assigns `feats_block["nardl_pred_t"] = result`, so the result must be
    sized to feats_block, not to the full unfiltered `block`.

    For each row t (after the dropna), we compute lnAUC[t] = lnAUC[t-1] +
    predicted ΔlnAUC[t] using NARDL coefficients fit on the training window.
    For training rows this is in-sample fitted; for the test row at the end
    it is the one-step-ahead forecast — same value _nardl_one_step_ahead
    returns. No leakage: NARDL coefficients are fixed for the whole call.
    """
    model = nardl_result["model"]
    p_lag, q_lag, r_lag = nardl_result["lags"]

    full = block.dropna(subset=["ny_usdkg", "auc_usdkg", "fx_usdkes"]).reset_index(drop=True)
    full["lnNY"]   = np.log(full["ny_usdkg"])
    full["lnAUC"]  = np.log(full["auc_usdkg"])
    full["lnFX"]   = np.log(full["fx_usdkes"])
    full["dlnAUC"] = full["lnAUC"].diff()
    full["dlnNY"]  = full["lnNY"].diff()
    full["dlnFX"]  = full["lnFX"].diff()
    pos, neg = cumulative_partial_sums(full["dlnNY"])
    full["NY_pos_cum"] = pos
    full["NY_neg_cum"] = neg
    full["dNY_pos"]    = full["NY_pos_cum"].diff()
    full["dNY_neg"]    = full["NY_neg_cum"].diff()

    out = np.full(len(full), np.nan)        # aligned to feats_block, NOT block
    for i in range(1, len(full)):
        feat = {"const": 1.0,
                "lnAUC_lag1":  full["lnAUC"].iloc[i - 1],
                "NY_pos_lag1": full["NY_pos_cum"].iloc[i - 1],
                "NY_neg_lag1": full["NY_neg_cum"].iloc[i - 1]}
        if "lnFX_lag1" in model.params.index:
            feat["lnFX_lag1"] = full["lnFX"].iloc[i - 1]
        ok = True
        for L in range(1, p_lag + 1):
            if i - L < 0:
                ok = False; break
            feat[f"dlnAUC_lag{L}"] = full["dlnAUC"].iloc[i - L]
        if not ok: continue
        for L in range(0, q_lag + 1):
            if i - L < 0:
                ok = False; break
            feat[f"dNY_pos_lag{L}"] = full["dNY_pos"].iloc[i - L]
            feat[f"dNY_neg_lag{L}"] = full["dNY_neg"].iloc[i - L]
        if not ok: continue
        if "lnFX_lag1" in model.params.index:
            for L in range(0, r_lag + 1):
                if i - L < 0:
                    ok = False; break
                feat[f"dlnFX_lag{L}"] = full["dlnFX"].iloc[i - L]
            if not ok: continue
        x = pd.Series({k: feat.get(k, 0.0) for k in model.params.index})
        pred_d = float(np.dot(x.values, model.params.values))
        out[i] = full["lnAUC"].iloc[i - 1] + pred_d
    return out


def diebold_mariano_hln(y, p1, p2, h=1):
    e1 = np.asarray(y) - np.asarray(p1)
    e2 = np.asarray(y) - np.asarray(p2)
    d = e1**2 - e2**2
    n = len(d)
    if n < 3:                      # cannot run DM on fewer than 3 pairs
        return np.nan, np.nan, np.nan
    md = d.mean()
    var = np.var(d, ddof=1)
    for k in range(1, h):
        gk = np.cov(d[:-k], d[k:], ddof=1)[0, 1]
        var += 2 * gk
    if var <= 0:
        return np.nan, np.nan, np.nan
    dm = md / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n) * dm
    p = 2 * (1 - stats.norm.cdf(abs(hln)))
    return dm, hln, p


# --> main
def main():
    monthly = pd.read_csv(C.DATA_OUT / "monthly_clean.csv")
    cv = rolling_origin_cv(monthly)

    a = cv["actual"]; px = cv["pred_xgb"]; pn = cv["pred_nardl"]
    if len(a) == 0:
        log.error("Stage 1 CV produced no successful folds. See skip-reason\n"
                  "tally above. Writing empty stub tables and exiting cleanly.")
        for fn in ("xgb_stage1_metrics", "xgb_stage1_dm",
                   "xgb_stage1_predictions", "xgb_stage1_shap_top"):
            pd.DataFrame().to_csv(C.TAB_DIR / f"{fn}.csv", index=False)
        return

    def metrics(y, p):
        rmse = np.sqrt(np.mean((y - p) ** 2))
        mape = np.mean(np.abs((np.exp(y) - np.exp(p)) / np.exp(y))) * 100
        return float(rmse), float(mape)

    rmse_x, mape_x = metrics(a, px)
    rmse_n, mape_n = metrics(a, pn)

    pd.DataFrame([
        {"model": "XGBoost",  "RMSE_log": round(rmse_x, 4), "MAPE_level_%": round(mape_x, 2)},
        {"model": "NARDL",    "RMSE_log": round(rmse_n, 4), "MAPE_level_%": round(mape_n, 2)},
    ]).to_csv(C.TAB_DIR / "xgb_stage1_metrics.csv", index=False)

    dm, hln, pval = diebold_mariano_hln(a, px, pn)
    pd.DataFrame([{"comparison": "XGBoost vs NARDL (Stage 1)",
                    "DM": round(dm, 4) if dm == dm else None,
                    "HLN": round(hln, 4) if hln == hln else None,
                    "p_value": round(pval, 4) if pval == pval else None,
                    "n_pairs": int(len(a))}]
                 ).to_csv(C.TAB_DIR / "xgb_stage1_dm.csv", index=False)

    pd.DataFrame({
        "coffee_year": [k[0] for k in cv["keys"]],
        "coffee_year_month": [k[1] for k in cv["keys"]],
        "actual_lnAUC": a, "xgb_lnAUC": px, "nardl_lnAUC": pn,
    }).to_csv(C.TAB_DIR / "xgb_stage1_predictions.csv", index=False)

    # SHAP on validation predictions: refit on full sample then explain ALL
    # validation rows. (Per v3, SHAP is interpreted as supporting interpretability,
    # not as a full-sample fit artefact.)
    # ITER2 change 2.2: SHAP runs on the same hybrid feature set as the CV.
    # We refit NARDL on the FULL sample for SHAP attribution (acceptable here
    # because SHAP is interpreting the ML decision surface, not the forecast).
    nardl_full = fit_stage1(monthly, include_fx=True, return_features=True)
    feats_all  = build_stage1_features(monthly, nardl_full["ect_features"])
    feats_all["nardl_pred_t"] = _nardl_fitted_levels_for_block(nardl_full, monthly)
    feats_all  = feats_all.dropna(subset=cv["feat_names"] + ["target"])
    X_val = feats_all[cv["feat_names"]].values
    last_params = cv["best_params_history"][-1] if cv["best_params_history"] else {}
    last_params = {k: v for k, v in last_params.items() if k != "t"}
    final = xgb.XGBRegressor(**last_params,
                              objective="reg:squarederror",
                              verbosity=0,
                              random_state=C.RANDOM_SEED)
    final.fit(X_val, feats_all["target"].values, verbose=False)
    explainer = shap.TreeExplainer(final)
    shap_vals = explainer.shap_values(X_val)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    shap_table = pd.DataFrame({
        "feature": cv["feat_names"],
        "mean_abs_shap": mean_abs.round(4),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_table.to_csv(C.TAB_DIR / "xgb_stage1_shap_top.csv", index=False)

    with open(C.MODEL_DIR / "xgb_stage1_best_params.json", "w") as f:
        json.dump(cv["best_params_history"], f, indent=2)

    log.info("\n=== Stage 1 XGBoost rolling-origin CV ===")
    log.info(f"Validation pairs : {len(a)}")
    log.info(f"RMSE (log)  XGBoost: {rmse_x:.4f}    NARDL: {rmse_n:.4f}")
    log.info(f"DM HLN p-value (XGB vs NARDL): {pval:.4f}")


if __name__ == "__main__":
    main()
