"""
step6_figures.py
----------------
This script is used for the publication-quality PNG figures for Chapter 4. Reads the CSV outputs producedby steps 1–5. 
All plots are saved at 300 dpi to Codes/_outputs/figures/.

Figures
-------
fig_1_overview.png             4-panel overview of the four core series
fig_2_partial_sums.png         NY positive/negative cumulative shock partial sums
fig_3_breaks.png               Bai–Perron-style breaks (Binseg) on lnAUC
fig_4_stage1_predictions.png   Stage 1 actual vs XGBoost vs NARDL (rolling-origin)
fig_5_stage2_predictions.png   Stage 2 actual vs XGBoost vs NARDL (both origins)
fig_6_shap_stage1.png          Stage 1 mean |SHAP| bar chart
fig_7_shap_stage2.png          Stage 2 mean |SHAP| bar chart per variant
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from logging_setup import get_logger
log = get_logger(__name__)

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.titlesize"  : 12,
    "axes.labelsize"  : 11,
    "xtick.labelsize" : 9,
    "ytick.labelsize" : 9,
    "legend.fontsize" : 9,
    "figure.dpi"      : 300,
})


def fig_overview():
    monthly = pd.read_csv(C.DATA_OUT / "monthly_clean.csv")
    monthly["t"] = pd.to_datetime(monthly["calendar_year"].astype(str) + "-" +
                                   monthly["calendar_month"].astype(str) + "-01")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0,0].plot(monthly["t"], monthly["ny_usdkg"]);  ax[0,0].set_title("New York 'C' Futures (USD/kg)")
    ax[0,1].plot(monthly["t"], monthly["auc_usdkg"]); ax[0,1].set_title("Nairobi auction average (USD/kg)")
    ax[1,0].plot(monthly["t"], monthly["fx_usdkes"]); ax[1,0].set_title("USD/KES rate")
    ax[1,1].plot(monthly["t"], monthly["ico_usdkg"]); ax[1,1].plot(monthly["t"], monthly["col_usdkg"])
    ax[1,1].set_title("ICO Composite & Colombian Milds (USD/kg)")
    ax[1,1].legend(["ICO Composite", "Colombian Milds"])
    for a in ax.ravel(): a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_1_overview.png"); plt.close(fig)


def fig_partial_sums():
    monthly = pd.read_csv(C.DATA_OUT / "monthly_clean.csv")
    monthly = monthly.dropna(subset=["ny_usdkg"]).reset_index(drop=True)
    dlnNY = np.log(monthly["ny_usdkg"]).diff()
    pos = dlnNY.clip(lower=0).cumsum()
    neg = dlnNY.clip(upper=0).cumsum()
    t = pd.to_datetime(monthly["calendar_year"].astype(str) + "-" +
                        monthly["calendar_month"].astype(str) + "-01")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, pos, label="cumulative positive (NY⁺)", color="#1f77b4")
    ax.plot(t, neg, label="cumulative negative (NY⁻)", color="#d62728")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_title("NARDL cumulative shock partial sums (lnNY)")
    ax.set_ylabel("cumulative log-change"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig_2_partial_sums.png"); plt.close(fig)


def fig_breaks():
    monthly = pd.read_csv(C.DATA_OUT / "monthly_clean.csv")
    monthly = monthly.dropna(subset=["auc_usdkg"]).reset_index(drop=True)
    t = pd.to_datetime(monthly["calendar_year"].astype(str) + "-" +
                        monthly["calendar_month"].astype(str) + "-01")
    try:
        breaks = pd.read_csv(C.TAB_DIR / "breaks_binseg.csv")
    except FileNotFoundError:
        breaks = pd.DataFrame(columns=["coffee_year_month"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, np.log(monthly["auc_usdkg"]), color="#333", linewidth=1)
    for tag in breaks["coffee_year_month"]:
        # tag like '2008/2009-m04': map back to a calendar date
        try:
            cy, mtag = tag.split("-m")
            cy_start = int(cy.split("/")[0])
            m = int(mtag)
            cm = m + 9 if m <= 3 else m - 3
            cy_year = cy_start if m <= 3 else cy_start + 1
            ax.axvline(pd.Timestamp(year=cy_year, month=cm, day=1),
                        color="#d62728", linewidth=0.8, alpha=0.6)
        except Exception:
            pass
    ax.set_title("ln(Auction USD/kg) with Binseg breakpoints")
    ax.set_ylabel("ln(USD/kg)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig_3_breaks.png"); plt.close(fig)


def fig_stage1_predictions():
    try:
        p = pd.read_csv(C.TAB_DIR / "xgb_stage1_predictions.csv")
    except FileNotFoundError:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    idx = range(len(p))
    ax.plot(idx, p["actual_lnAUC"], label="Actual", color="#333", linewidth=1.2)
    ax.plot(idx, p["xgb_lnAUC"],   label="XGBoost",  color="#1f77b4", alpha=0.85)
    ax.plot(idx, p["nardl_lnAUC"], label="NARDL",    color="#2ca02c", alpha=0.85)
    ax.set_title("Stage 1 rolling-origin one-step-ahead forecasts (log auction USD/kg)")
    ax.set_xlabel("validation step"); ax.set_ylabel("ln(USD/kg)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig_4_stage1_predictions.png"); plt.close(fig)


def fig_stage2_predictions():
    try:
        p = pd.read_csv(C.TAB_DIR / "xgb_stage2_predictions.csv")
    except FileNotFoundError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for i, variant in enumerate(["baseline", "preseason"]):
        sub = p[p["variant"] == variant].reset_index(drop=True)
        if sub.empty: continue
        ax[i].plot(sub["coffee_year"], sub["actual_lnFG"], label="Actual", color="#333")
        ax[i].plot(sub["coffee_year"], sub["xgb_lnFG"],   label="XGBoost", color="#1f77b4")
        ax[i].plot(sub["coffee_year"], sub["nardl_lnFG"], label="NARDL",   color="#2ca02c")
        ax[i].set_title(f"Stage 2 — {variant}")
        ax[i].set_xlabel("coffee year"); ax[i].grid(alpha=0.3)
        ax[i].tick_params(axis="x", rotation=45)
        ax[i].legend()
    ax[0].set_ylabel("ln(USD/kg)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig_5_stage2_predictions.png"); plt.close(fig)


def _shap_bar(df: pd.DataFrame, title: str, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    df = df.sort_values("mean_abs_shap", ascending=True)
    ax.barh(df["feature"], df["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_shap():
    try:
        s1 = pd.read_csv(C.TAB_DIR / "xgb_stage1_shap_top.csv")
        _shap_bar(s1, "Stage 1 XGBoost feature importance (mean |SHAP|)",
                  C.FIG_DIR / "fig_6_shap_stage1.png")
    except FileNotFoundError:
        pass
    try:
        s2 = pd.read_csv(C.TAB_DIR / "xgb_stage2_shap_top.csv")
        for variant in ("baseline", "preseason"):
            sub = s2[s2["variant"] == variant]
            if sub.empty: continue
            _shap_bar(sub.drop(columns="variant"),
                      f"Stage 2 XGBoost ({variant}) feature importance",
                      C.FIG_DIR / f"fig_7_shap_stage2_{variant}.png")
    except FileNotFoundError:
        pass


def main():
    fig_overview()
    fig_partial_sums()
    fig_breaks()
    fig_stage1_predictions()
    fig_stage2_predictions()
    fig_shap()
    log.info(f"Wrote figures to {C.FIG_DIR}")


if __name__ == "__main__":
    main()
