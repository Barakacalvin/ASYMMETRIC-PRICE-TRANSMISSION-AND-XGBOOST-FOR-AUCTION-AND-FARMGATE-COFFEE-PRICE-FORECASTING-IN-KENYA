"""
logging_setup.py - An important file to monitor the process as it ran. The Logging file was printed in terminal upon running. 
I suggest will be grat for larger file executions.
----------------
Single, shared logger configuration for the entire pipeline.

Expected Outputs
---------
- Writes everything to Codes/_outputs/logs/pipeline_<UTC-timestamp>.log
- Mirrors the same lines to stdout so you can watch progress live.
- When run_all.py launches a subprocess, the env var DISSO_LOG_FILE is set so
  every child step appends to the same master log file.

Usage in each script
--------------------
    from logging_setup import get_logger
    log = get_logger(__name__)
    log.info("starting step ...")
    log.warning("skipped fold %d (insufficient data)", t)
    log.error("model fit failed: %s", e)
"""
from __future__ import annotations
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import config as C

_LOG_FORMAT  = "%(asctime)s %(levelname)-7s %(name)-32s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ENV_VAR     = "DISSO_LOG_FILE"

_initialised: dict[str, logging.Logger] = {}


def _resolve_log_path() -> Path:
    """If a parent runner has set DISSO_LOG_FILE, append to that. Else open
    a fresh timestamped file."""
    env_path = os.environ.get(_ENV_VAR)
    if env_path:
        return Path(env_path)
    log_dir = C.OUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"pipeline_{stamp}.log"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the calling module."""
    if name in _initialised:
        return _initialised[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers in interactive sessions
    if not logger.handlers:
        log_path = _resolve_log_path()
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # Banner only on the first logger to be created in this process
        if not _initialised:
            banner_logger = logging.getLogger("dissertation.runner")
            banner_logger.setLevel(logging.INFO)
            banner_logger.addHandler(fh)
            banner_logger.addHandler(sh)
            banner_logger.info("=" * 78)
            banner_logger.info("Pipeline log: %s", log_path)
            banner_logger.info("=" * 78)

    _initialised[name] = logger
    return logger


def export_log_path_to_env() -> str:
    """Used by run_all.py to fix the master log file once and propagate it
    to every subprocess via the environment."""
    path = _resolve_log_path()
    os.environ[_ENV_VAR] = str(path)
    return str(path)
