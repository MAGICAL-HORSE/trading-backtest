"""
utils.py — Shared helpers (config loader, logger factory, time utilities)
=========================================================================
Every module imports from here. Keeping this thin avoids circular deps.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, time as dtime
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ── Structured JSON Logger ───────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra keys passed via `extra={...}`
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    cfg = get_config()
    log_cfg = cfg.get("logging", {})
    level_str: str = os.environ.get("LOG_LEVEL", log_cfg.get("level", "INFO"))
    level = getattr(logging, level_str.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    formatter = _JsonFormatter()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (rotating)
    log_file = Path(log_cfg.get("file", "logs/bot.jsonl"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        log_file,
        maxBytes=log_cfg.get("max_bytes", 10_485_760),
        backupCount=log_cfg.get("backup_count", 5),
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


# ── Market Time Utilities ─────────────────────────────────────────────────────

ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(tz=ET)


def market_open_et() -> datetime:
    """Returns today's 09:30 ET as an aware datetime."""
    d = now_et().date()
    return datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)


def parse_time_et(t_str: str) -> dtime:
    """Parse 'HH:MM' string to a time object (no tz, for local comparison)."""
    h, m = t_str.split(":")
    return dtime(int(h), int(m))


def is_within_trading_window() -> bool:
    """Return True if current ET time is between 09:30 and 16:00."""
    cfg = get_config()
    risk = cfg["risk"]
    now = now_et().time()
    open_time = parse_time_et("09:30")
    close_time = parse_time_et("16:00")
    return open_time <= now <= close_time


def can_enter_new_trade() -> tuple[bool, str]:
    """
    Returns (allowed, reason).
    Checks: no-trade window at open, no-new-entry cutoff.
    """
    cfg = get_config()
    risk = cfg["risk"]
    now = now_et().time()
    no_entry_before = parse_time_et(risk["no_entry_before"])
    no_new_entry_after = parse_time_et(risk["no_new_entry_after"])

    if now < parse_time_et("09:30"):
        return False, "pre_market"
    if now < no_entry_before:
        return False, "opening_blackout"
    if now > no_new_entry_after:
        return False, "after_cutoff"
    return True, "ok"


def must_close_all() -> bool:
    """Return True if we've passed the force-close time."""
    cfg = get_config()
    force_close = parse_time_et(cfg["risk"]["force_close_by"])
    return now_et().time() >= force_close


# ── Misc ─────────────────────────────────────────────────────────────────────

def env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


def env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default
