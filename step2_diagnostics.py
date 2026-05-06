"""
step2_diagnostics.py
--------------------
Stationarity battery (ADF + Phillips-Perron + KPSS) and Bai-Perron-style multiple-break detection via the `ruptures` library (Binseg primary, PELT robustness).
Implements Chapter 3 soecifically sections 3.4.1 and 3.4.2. This executes after the running of step1_load_and_prepare.py for earlier sections of the chapter.

Implementation Notes
-----
*The code refers to the Bai-Perron-style*: There are no maintained Python packages that implement the full Bai-Perron sequential supF testing protocol. 
We use binary segmentation through `ruptures` as a computational approximation; the resulting break dates are treated as diagnostic regime indicators rather than as formal Bai-Perron test outcomes.

Outputs
-------
Codes/_outputs/tables/unit_root.csv
Codes/_outputs/tables/breaks_binseg.csv
Codes/_outputs/tables/breaks_pelt.csv
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
import ruptures as rpt

import config as C
from logging_setup import get_logger
log = get_logger(__name__)

warnings.filterwarnings("ignore")
np.random.seed(C.RANDOM_SEED)

# -> unit roots
def _adf(s: np.ndarray, regression: str = "c") -> dict:
    s = s[~np.isnan(s)]
    out = adfuller(s, regression=regression, autolag="AIC")
    return {"stat": out[0], "p_value": out[1], "lags": out[2], "n": out[3]}


def _pp(s: np.ndarray, trend: str = "c") -> dict:
    s = s[~np.isnan(s)]
    pp = PhillipsPerron(s, trend=trend)
    return {"stat": pp.stat, "p_value": pp.pvalue, "lags": pp.lags, "n": pp.nobs}


def _kpss(s: np.ndarray, regression: str = "c") -> dict:
    s = s[~np.isnan(s)]
    stat, pval, lags, _ = kpss(s, regression=regression, nlags="auto")
    return {"stat": stat, "p_value": pval, "lags": lags, "n": len(s)}


def unit_root_battery(series_dict: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, s in series_dict.items():
        s_arr = np.asarray(s, dtype=float)
        for tag, x in [("level", s_arr), ("d1", np.diff(s_arr))]:
            adf = _adf(x)
            pp_ = _pp(x)
            ks  = _kpss(x)
            rows.append({
                "series"        : name,
                "transformation": tag,
                "ADF_stat"      : round(adf["stat"], 4),
                "ADF_p"         : round(adf["p_value"], 4),
                "PP_stat"       : round(pp_["stat"], 4),
                "PP_p"          : round(pp_["p_value"], 4),
                "KPSS_stat"     : round(ks["stat"], 4),
                "KPSS_p"        : round(ks["p_value"], 4),
                "n"             : int(adf["n"]),
                "verdict"       : _verdict(adf["p_value"], pp_["p_value"], ks["p_value"]),
            })
    return pd.DataFrame(rows)


def _verdict(adf_p: float, pp_p: float, kpss_p: float) -> str:
    """ADF/PP reject => stationary; KPSS reject => non-stationary."""
    adf_stationary  = adf_p  < 0.05
    pp_stationary   = pp_p   < 0.05
    kpss_stationary = kpss_p > 0.05  # KPSS null = stationary
    votes = [adf_stationary, pp_stationary, kpss_stationary]
    n_yes = sum(votes)
    if n_yes >= 2:
        return "I(0)"
    if n_yes == 0:
        return "I(1) candidate"
    return "ambiguous"


# -> breaks
def detect_breaks_binseg(signal: np.ndarray, n_max: int) -> list[int]:
    """Binary segmentation with at most n_max breaks. Returns break indices."""
    algo = rpt.Binseg(model=C.BAI_PERRON["binseg_model"]).fit(signal)
    cps = algo.predict(n_bkps=n_max)        # includes endpoint == len(signal)
    return [c for c in cps if c < len(signal)]


def detect_breaks_pelt(signal: np.ndarray, pen: float = None) -> list[int]:
    """PELT with linear penalty. Returns break indices."""
    if pen is None:
        pen = C.BAI_PERRON["pelt_pen"]
    algo = rpt.Pelt(model=C.BAI_PERRON["pelt_model"],
                    min_size=C.BAI_PERRON["min_segment_length_months"]).fit(signal)
    cps = algo.predict(pen=pen)
    return [c for c in cps if c < len(signal)]


def break_dates_from_indices(monthly: pd.DataFrame, idx: list[int]) -> list[str]:
    """Translate monthly-row indices to coffee-year-month tags."""
    out = []
    for i in idx:
        if 0 <= i < len(monthly):
            r = monthly.iloc[i]
            out.append(f"{r['coffee_year']}-m{int(r['coffee_year_month']):02d}")
    return out


# -> main
def main():
    monthly = pd.read_csv(C.DATA_OUT / "monthly_clean.csv")
    annual  = pd.read_csv(C.DATA_OUT / "annual_clean.csv")

    # -> unit-root battery <-
    series = {
        # monthly, in logs
        "lnNY"      : np.log(monthly["ny_usdkg"].dropna().values),
        "lnAUC"     : np.log(monthly["auc_usdkg"].dropna().values),
        "lnICO"     : np.log(monthly["ico_usdkg"].dropna().values),
        "lnCOL"     : np.log(monthly["col_usdkg"].dropna().values),
        "lnFX"      : np.log(monthly["fx_usdkes"].dropna().values),
        # annual, in logs
        "lnFG_y"    : np.log(annual["fg_usdkg_clean"].dropna().values),
        "lnAUC_y"   : np.log(annual["auc_usdkg_mean"].dropna().values),
        "lnNY_y"    : np.log(annual["ny_usdkg_mean"].dropna().values),
    }
    ur = unit_root_battery(series)
    log.info("\n=== Unit-root battery ===")
    log.info(ur.to_string(index=False))
    ur.to_csv(C.TAB_DIR / "unit_root.csv", index=False)

    # -> structural breaks (monthly auction series) <- Baraka Calvin
    auc = monthly["auc_usdkg"].dropna().reset_index(drop=True)
    sig = np.log(auc.values).reshape(-1, 1)

    # Binseg primary
    bs_idx  = detect_breaks_binseg(sig, n_max=C.BAI_PERRON["n_breaks_max"])
    bs_dates = break_dates_from_indices(monthly.dropna(subset=["auc_usdkg"]).reset_index(drop=True), bs_idx)

    # PELT robustness
    pe_idx  = detect_breaks_pelt(sig)
    pe_dates = break_dates_from_indices(monthly.dropna(subset=["auc_usdkg"]).reset_index(drop=True), pe_idx)

    bs_table = pd.DataFrame({"break_no": range(1, len(bs_dates) + 1),
                              "coffee_year_month": bs_dates,
                              "method": "Binseg (primary)"})
    pe_table = pd.DataFrame({"break_no": range(1, len(pe_dates) + 1),
                              "coffee_year_month": pe_dates,
                              "method": "PELT (robustness)"})
    log.info("\n=== Breaks (Binseg, primary) ===")
    log.info(bs_table.to_string(index=False))
    log.info("\n=== Breaks (PELT, robustness) ===")
    log.info(pe_table.to_string(index=False))

    bs_table.to_csv(C.TAB_DIR / "breaks_binseg.csv", index=False)
    pe_table.to_csv(C.TAB_DIR / "breaks_pelt.csv",   index=False)

    # combined
    combined = pd.concat([bs_table, pe_table], ignore_index=True)
    combined.to_csv(C.TAB_DIR / "breaks_combined.csv", index=False)
    log.info(f"\nWrote {C.TAB_DIR/'unit_root.csv'}, "
          f"{C.TAB_DIR/'breaks_binseg.csv'}, {C.TAB_DIR/'breaks_pelt.csv'}.")


if __name__ == "__main__":
    main()
