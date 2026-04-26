# Kenya Coffee Asymmetric Price Transmission — Analysis Pipeline

MSc Data Science Dissertation — Calvin Baraka Rimba, UEL DS-7010
Aligned with **Methodology Register v2** and **Chapter 3 v3**.

This pipeline implements the two-stage NARDL specification with the parallel
XGBoost predictive comparator and produces every table and figure that
appears in Chapter 4 of the dissertation.

## What the pipeline does

| Step | Script | Implements |
|------|--------|------------|
| 1 | `step1_load_and_prepare.py`  | Load `Datasets for Modelling Updated Final.xlsx`, harmonise to USD/kg, coffee-year index (Oct = month 1), exclude recess months from annual auction means |
| 2 | `step2_diagnostics.py`       | ADF + Phillips–Perron + KPSS unit-root battery; Bai–Perron-style multiple-break detection via `ruptures` (Binseg primary, PELT robustness) |
| 3a | `step3a_nardl_stage1.py`    | Stage 1 NARDL (monthly auction ← NY Futures + FX), SBC lag selection, HAC SEs, PSS bounds + BDM, baseline + FX-free sensitivity |
| 3b | `step3b_nardl_stage2.py`    | Stage 2 NARDL (annual farmgate ← annual auction), Narayan small-sample bounds critical values |
| 4a | `step4a_xgb_stage1_monthly.py` | Stage 1 XGBoost — 12 features per v3 §3.6.3, rolling-origin CV with NARDL refit inside every training window, Optuna TPE 50-trial cap, SHAP on validation predictions, DM (HLN-corrected) vs NARDL |
| 4b | `step4b_xgb_stage2_annual.py`  | Stage 2 XGBoost — 6 features per v3 §3.6.4, end-of-coffee-year baseline + pre-season variant, NARDL refit inside fold |
| 5  | `step5_robustness.py`         | §3.7.2 alternative international benchmarks; §3.7.3 Stage 1 ablations; §3.7.4 Stage 2 ablation; §3.7.5 cherry descriptive comparison |
| 6  | `step6_figures.py`            | All publication figures |

## Methodology choices documented elsewhere

> Hyperparameter optimisation was performed using Optuna's Tree-structured
> Parzen Estimator sampler, capped at 50 trials to limit small-sample
> overfitting and preserve computational reproducibility.

> Structural breaks were detected using a Bai–Perron-style multiple-break
> procedure implemented through the `ruptures` library. Because no maintained
> Python package implements the full Bai–Perron sequential supF testing
> protocol, the study uses binary segmentation as a computational
> approximation and treats the resulting break dates as diagnostic regime
> indicators rather than as formal Bai–Perron test outcomes. PELT is reported
> as a robustness comparison.

## Requirements

```bash
pip install -r requirements.txt
```

`config.py` centralises every path, seed, sample window, lag bound and
hyperparameter range. To re-target a different workbook or output folder,
edit `config.py` only.

## How to run

End-to-end:

```bash
python run_all.py
```

Or step-by-step (each script is independently runnable once step 1 has been
executed once):

```bash
python step1_load_and_prepare.py
python step2_diagnostics.py
python step3a_nardl_stage1.py
python step3b_nardl_stage2.py
python step4a_xgb_stage1_monthly.py
python step4b_xgb_stage2_annual.py
python step5_robustness.py
python step6_figures.py
```

## Outputs

All outputs land under `Codes/_outputs/`:

```
_outputs/
├── data/
│   ├── monthly_clean.csv           # 312 monthly rows, USD/kg, coffee-year indexed
│   ├── annual_clean.csv            # 26 annual rows, USD/kg
│   ├── stage1_features_for_xgb.csv # ECT and partial sums (Stage 1)
│   ├── stage2_features_for_xgb.csv # ECT and partial sums (Stage 2)
│   └── data_audit.json             # provenance + recess-month counts n_y
├── tables/                         # CSVs feeding Chapter 4
└── figures/                        # PNGs feeding Chapter 4
```

## Reproducibility

- `random_state = 42` everywhere (numpy, XGBoost, Optuna sampler).
- No browser or filesystem dependencies beyond `Data/Datasets for Modelling Updated Final.xlsx`.
- Every coefficient, p-value and figure cited in Chapter 4 traces back to a
  CSV under `_outputs/tables/` or a PNG under `_outputs/figures/`.

## Citation

> Rimba, C. B. (2026). *Asymmetric Price Transmission and Farmgate Coffee
> Price Forecasting in Kenya: A Two-Stage NARDL Approach with Explainable
> Gradient Boosting.* MSc Dissertation, University of East London.

## Logging

Every script writes to both the console and a timestamped log file at
`Codes/_outputs/logs/pipeline_<UTC-timestamp>.log`.

- When you run `python run_all.py`, one master log file is created and every
  subprocess step appends to it (via the `DISSO_LOG_FILE` environment variable
  that `run_all.py` exports before each subprocess call).
- When you run an individual step script directly, it opens its own
  timestamped log file — use this for debugging one step in isolation.

Each log line is formatted as:

```
YYYY-MM-DD HH:MM:SS LEVEL   module.name      | message
```

Progress checkpoints are emitted inside the two long-running rolling-origin
loops. Stage 1 XGBoost logs Optuna retune events and a progress heartbeat
every 25 folds; Stage 2 XGBoost logs every fold (the annual loop has few
folds). `run_all.py` additionally reports elapsed seconds per step and
total pipeline time.

To tail the master log live from a second PowerShell window while the
pipeline runs:

```powershell
Get-Content .\_outputs\logs\pipeline_*.log -Wait -Tail 30
```
