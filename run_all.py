"""
run_all.py
----------
End-to-end runner for the final pipeline.

- Opens ONE master log file and exports its path via DISSO_LOG_FILE so every subprocess step writes into the same file.
- Reports elapsed time per step and for the full run.
- Aborts on first failing step and prints the offending script name for ease of follow up and verification incase of process breaks
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

from logging_setup import export_log_path_to_env, get_logger

HERE = Path(__file__).resolve().parent
STEPS = [
    "step1_load_and_prepare.py",
    "step2_diagnostics.py",
    "step3a_nardl_stage1.py",
    "step3b_nardl_stage2.py",
    "step4a_xgb_stage1_monthly.py",
    "step4b_xgb_stage2_annual.py",
    "step5_robustness.py",
    "step6_figures.py",
]


def main() -> int:
    log_path = export_log_path_to_env()
    log = get_logger("dissertation.run_all")
    log.info("Master log file : %s", log_path)
    log.info("Working dir     : %s", HERE)
    log.info("Python interpreter: %s", sys.executable)
    log.info("Running %d steps in sequence", len(STEPS))

    t_all = time.perf_counter()
    for i, step in enumerate(STEPS, start=1):
        log.info("-" * 78)
        log.info("[%d/%d] START: %s", i, len(STEPS), step)
        t0 = time.perf_counter()
        result = subprocess.run([sys.executable, step], cwd=str(HERE))
        dt = time.perf_counter() - t0
        if result.returncode != 0:
            log.error("[%d/%d] FAIL : %s (rc=%d) in %.1fs -- aborting",
                      i, len(STEPS), step, result.returncode, dt)
            return result.returncode
        log.info("[%d/%d] DONE : %s in %.1fs", i, len(STEPS), step, dt)

    total = time.perf_counter() - t_all
    log.info("=" * 78)
    log.info("Pipeline completed successfully in %.1fs (~%.1f minutes)",
             total, total / 60.0)
    log.info("Tables : %s", HERE / "_outputs" / "tables")
    log.info("Figures: %s", HERE / "_outputs" / "figures")
    log.info("Log    : %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
