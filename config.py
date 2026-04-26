"""
config.py — central configuration for the Kenya coffee NARDL/XGBoost pipeline.

All scripts import from this module so that paths, sample windows, random seeds
and hyperparameter grids are defined in exactly one place.

Aligned with Methodology Register v2 and Chapter 3 v3.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
# In the iteration-folder layout each Iteration_N/ is self-contained:
#   Iteration_N/
#       Data/           -- snapshot of the data this iteration was built on
#       _outputs/       -- live results of this iteration
#       _outputs_iter1/ -- frozen first-pass snapshot (iteration 1 only)
#       *.py            -- pipeline modules
ITER_ROOT  = Path(__file__).resolve().parent             # .../Codes/Iteration_N
ROOT       = ITER_ROOT.parent.parent                     # .../Dissertation
DATA_DIR   = ITER_ROOT / "Data"                          # iteration-local data snapshot
WORKBOOK   = DATA_DIR / "Datasets for Modelling  Updated Final.xlsx"

# Output locations (created on first run by step1) — relative to this iteration
OUT_DIR    = ITER_ROOT / "_outputs"
TAB_DIR    = OUT_DIR / "tables"
FIG_DIR    = OUT_DIR / "figures"
DATA_OUT   = OUT_DIR / "data"
MODEL_DIR  = OUT_DIR / "models"
for p in (OUT_DIR, TAB_DIR, FIG_DIR, DATA_OUT, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- sample
SAMPLE_START_COFFEE_YEAR = "1998/1999"   # October 1998
SAMPLE_END_COFFEE_YEAR   = "2023/2024"   # September 2024
N_COFFEE_YEARS_EXPECTED  = 26
N_MONTHS_EXPECTED        = 312           # 26 * 12

# ---------------------------------------------------------------- seeds
RANDOM_SEED = 42

# ---------------------------------------------------------------- NARDL
NARDL_S1_MAX_LAG = 12          # monthly stage
NARDL_S2_MAX_LAG = 2           # annual stage; SBC selects 1 or 2
LAG_CRITERION    = "bic"       # statsmodels SBC == BIC
HAC_MAXLAGS      = 6           # Newey-West for Stage 1
PSS_CASE         = 3           # intercept, no trend (Pesaran et al. 2001 Case III)

# ---------------------------------------------------------------- XGBoost
# Stage 1 monthly module
XGB_STAGE1 = {
    "initial_train_months": 60,        # 5 coffee years
    "n_splits_minimum"   : 50,         # rolling-origin one-step-ahead folds
    "optuna_trials"      : 50,
    "early_stopping_rounds": 25,
    "search_space": {
        "max_depth"        : ("int",   3, 6),
        "learning_rate"    : ("loguniform", 1e-2, 0.2),
        "n_estimators"     : ("int",   100, 800),
        "subsample"        : ("uniform", 0.6, 1.0),
        "colsample_bytree" : ("uniform", 0.6, 1.0),
        "reg_alpha"        : ("uniform", 0.0, 1.0),
        "reg_lambda"       : ("uniform", 0.0, 1.0),
    },
}

# Stage 2 annual module
XGB_STAGE2 = {
    "initial_train_years": 10,         # first 10 coffee years
    "n_splits_minimum"   : 5,
    "optuna_trials"      : 50,
    "early_stopping_rounds": 15,
    "search_space": {
        "max_depth"        : ("int",   2, 4),       # tighter ceiling for n=26
        "learning_rate"    : ("loguniform", 1e-2, 0.2),
        "n_estimators"     : ("int",   50, 400),
        "subsample"        : ("uniform", 0.6, 1.0),
        "colsample_bytree" : ("uniform", 0.6, 1.0),
        "reg_alpha"        : ("uniform", 0.0, 1.0),
        "reg_lambda"       : ("uniform", 0.0, 1.0),
    },
}

# ---------------------------------------------------------------- breaks
BAI_PERRON = {
    "min_segment_length_months": 24,    # ~2 coffee years
    "n_breaks_max"             : 5,
    "binseg_model"             : "rbf", # ruptures
    "pelt_model"               : "rbf",
    "pelt_pen"                 : 10.0,  # tuned later inside step2
}

# ---------------------------------------------------------------- unit conversions
CENTS_PER_LB_TO_USD_PER_KG = 1.0 / 100.0 / 0.45359237   # (cents/lb) -> USD/kg
USD_PER_50KG_TO_USD_PER_KG = 1.0 / 50.0                 # USD/50kg   -> USD/kg


# ITER2 change 1.1: BG-aware Stage 1 lag selection
STAGE1_BG_AWARE = True
