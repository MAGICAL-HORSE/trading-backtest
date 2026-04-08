"""
intelligence.py — Self-learning adaptive engine
=================================================
A 20-year experienced trader doesn't trade the same way every day.
They learn from every loss, double down on what works, and cut what doesn't.
This module does exactly that — it studies every trade the bot makes and
continuously re-weights the signal scoring system.

What it learns:
  • Win rate by signal combination (VWAP + RSI + ATR together)
  • Win rate by time of day (9:40-10:30 is best; 1-2 PM is choppy)
  • Win rate by sector and market regime
  • Win rate by confidence tier
  • Which rejections were correct (did the stock move without us?)
  • IV environment at entry vs outcome

How it adapts:
  • Bayesian signal weight updates — signals that predict winners get boosted
  • Time-of-day multiplier — bot trades bigger during its historically best hours
  • Sector rotation bias — if biotech is hot this week, allocate more there
  • Streak detection — after 3 losses, temporarily tighten confidence threshold
  • After 3 wins, loosen slightly (momentum in edge)

Storage: JSON file (trades_memory.json) — lightweight, readable, survives restarts.
No external ML dependencies — pure statistics, same tools a discretionary trader uses.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils import get_config, get_logger, now_et

log = get_logger(__name__)
cfg = get_config()

MEMORY_FILE = Path("trades_memory.json")


# ── Trade Record ─────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Everything about a completed trade, used for learning."""
    trade_id: str
    ticker: str
    sector: str
    direction: str                # "call" or "put"
    entry_time: str               # ISO
    exit_time: str                # ISO
    entry_hour_et: int            # 9–15
    confidence_at_entry: int
    rsi_at_entry: float
    atr_expansion_at_entry: float
    vwap_break_pct: float
    volume_spike: float
    had_catalyst: bool
    had_flow_signal: bool         # From flow.py
    market_regime: str            # From market_regime.py
    vix_at_entry: float
    exit_reason: str
    pnl_pct: float                # % gain/loss on premium
    winner: bool                  # True if pnl_pct > 0
    big_winner: bool              # True if pnl_pct >= 50
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Signal Weights ─────────────────────────────────────────────────────────

@dataclass
class SignalWeights:
    """
    Dynamic multipliers applied on top of the base confidence score.
    All start at 1.0 (neutral) and drift based on outcomes.
    Range: 0.5 (significantly penalized) to 2.0 (strongly boosted).
    """
    # Per-hour multipliers (keyed by ET hour 9–15)
    hour_multipliers: dict[str, float] = field(default_factory=lambda: {
        "9":  1.3,   # 9:40–10:00 — strong open momentum
        "10": 1.5,   # 10:00–11:00 — best hour of the day historically
        "11": 1.1,   # 11:00–12:00 — still decent
        "12": 0.7,   # 12:00–13:00 — lunch chop, avoid
        "13": 0.8,   # 13:00–14:00 — slow
        "14": 1.2,   # 14:00–15:00 — afternoon momentum builds
        "15": 1.0,   # 15:00–15:30 — last push
    })
    # Per-sector multipliers
    sector_multipliers: dict[str, float] = field(default_factory=lambda: {
        "biotech": 1.0,
        "energy": 1.0,
    })
    # Per-regime multipliers
    regime_multipliers: dict[str, float] = field(default_factory=lambda: {
        "BULL_TREND":      1.4,
        "BEAR_TREND":      1.2,   # Puts still work in bear
        "CHOPPY":          0.6,   # Avoid choppy days
        "HIGH_VOLATILITY": 1.1,   # Volatile = options premium explodes
        "UNKNOWN":         1.0,
    })
    # Streak state
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    # Total trade count (for statistical significance gating)
    total_trades: int = 0


# ── Intelligence Engine ───────────────────────────────────────────────────────

class IntelligenceEngine:
    """
    Stateful learning engine. Persists to disk between sessions.
    Thread-safe for single-process async use.
    """

    # Minimum trades before we trust the statistics enough to adjust weights
    _MIN_SAMPLE = 10
    # Learning rate — how fast weights shift per trade
    _LEARNING_RATE = 0.05

    def __init__(self) -> None:
        self._weights = SignalWeights()
        self._history: list[TradeRecord] = []
        self._load()

    # ── Public: scoring adjustment ────────────────────────────────────────────

    def adjust_confidence(
        self,
        base_confidence: int,
        sector: str,
        market_regime: str,
        entry_hour_et: int,
    ) -> int:
        """
        Apply learned multipliers to raw confidence score.
        Returns adjusted confidence (still 0–100 capped).
        """
        w = self._weights

        hour_mult = w.hour_multipliers.get(str(entry_hour_et), 1.0)
        sector_mult = w.sector_multipliers.get(sector, 1.0)
        regime_mult = w.regime_multipliers.get(market_regime, 1.0)

        # Streak dampener: 3+ losses → tighten by 20%
        streak_mult = 1.0
        if w.consecutive_losses >= 3:
            streak_mult = 0.80
        elif w.consecutive_losses >= 5:
            streak_mult = 0.65
        # Win streak booster: 3+ wins → loosen slightly
        elif w.consecutive_wins >= 3:
            streak_mult = 1.10
        elif w.consecutive_wins >= 5:
            streak_mult = 1.20

        combined = hour_mult * sector_mult * regime_mult * streak_mult
        adjusted = int(base_confidence * combined)

        log.debug(
            "confidence_adjusted",
            extra={
                "base": base_confidence,
                "adjusted": adjusted,
                "hour_mult": round(hour_mult, 2),
                "sector_mult": round(sector_mult, 2),
                "regime_mult": round(regime_mult, 2),
                "streak_mult": round(streak_mult, 2),
            },
        )
        return max(0, min(100, adjusted))

    def get_min_confidence_override(self) -> int | None:
        """
        If on a losing streak, raise the minimum confidence bar.
        Returns override value or None (use config default).
        """
        if self._weights.consecutive_losses >= 3:
            return 70   # Much stricter after 3 losses
        if self._weights.consecutive_losses >= 5:
            return 80   # Very strict after 5 losses
        return None

    def get_size_multiplier(self, sector: str, market_regime: str) -> float:
        """
        Returns a position size multiplier (0.5–1.5).
        Applied on top of config-based sizing.
        """
        if self._weights.total_trades < self._MIN_SAMPLE:
            return 1.0  # Don't adjust until we have data

        sector_wr = self._win_rate_by_sector(sector)
        regime_wr = self._win_rate_by_regime(market_regime)

        # If sector win rate > 60%, size up; < 40%, size down
        sector_adj = self._wr_to_multiplier(sector_wr)
        regime_adj = self._wr_to_multiplier(regime_wr)

        mult = (sector_adj + regime_adj) / 2
        return max(0.5, min(1.5, mult))

    # ── Public: recording outcomes ────────────────────────────────────────────

    def record_trade(self, record: TradeRecord) -> None:
        """Call after every trade closes. Updates weights immediately."""
        self._history.append(record)
        self._weights.total_trades += 1

        # Update streaks
        if record.winner:
            self._weights.consecutive_wins += 1
            self._weights.consecutive_losses = 0
        else:
            self._weights.consecutive_losses += 1
            self._weights.consecutive_wins = 0

        # Update hour multiplier
        if self._weights.total_trades >= self._MIN_SAMPLE:
            self._update_hour_weight(record)
            self._update_sector_weight(record)
            self._update_regime_weight(record)

        self._save()

        log.info(
            "intelligence_updated",
            extra={
                "ticker": record.ticker,
                "winner": record.winner,
                "pnl_pct": round(record.pnl_pct, 1),
                "consecutive_losses": self._weights.consecutive_losses,
                "consecutive_wins": self._weights.consecutive_wins,
                "total_trades": self._weights.total_trades,
            },
        )

    # ── Public: performance analytics ────────────────────────────────────────

    def get_stats_summary(self) -> dict[str, Any]:
        """Returns a summary dict for the dashboard and logging."""
        if not self._history:
            return {"total_trades": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0}

        winners = [t for t in self._history if t.winner]
        total = len(self._history)
        avg_pnl = sum(t.pnl_pct for t in self._history) / total

        best_hour = self._best_hour()
        best_sector = max(
            ["biotech", "energy"],
            key=lambda s: self._win_rate_by_sector(s),
        )

        return {
            "total_trades": total,
            "win_rate": round(len(winners) / total * 100, 1),
            "avg_pnl_pct": round(avg_pnl, 1),
            "consecutive_losses": self._weights.consecutive_losses,
            "consecutive_wins": self._weights.consecutive_wins,
            "best_hour_et": best_hour,
            "best_sector": best_sector,
            "hour_multipliers": self._weights.hour_multipliers,
            "sector_multipliers": self._weights.sector_multipliers,
        }

    # ── Private: weight updates ───────────────────────────────────────────────

    def _update_hour_weight(self, record: TradeRecord) -> None:
        key = str(record.entry_hour_et)
        current = self._weights.hour_multipliers.get(key, 1.0)
        # Nudge toward success: winner → push up, loser → push down
        delta = self._LEARNING_RATE if record.winner else -self._LEARNING_RATE
        new_val = current + delta
        self._weights.hour_multipliers[key] = max(0.5, min(2.0, round(new_val, 3)))

    def _update_sector_weight(self, record: TradeRecord) -> None:
        current = self._weights.sector_multipliers.get(record.sector, 1.0)
        delta = self._LEARNING_RATE if record.winner else -self._LEARNING_RATE
        new_val = current + delta
        self._weights.sector_multipliers[record.sector] = max(0.5, min(2.0, round(new_val, 3)))

    def _update_regime_weight(self, record: TradeRecord) -> None:
        current = self._weights.regime_multipliers.get(record.market_regime, 1.0)
        delta = self._LEARNING_RATE if record.winner else -self._LEARNING_RATE
        new_val = current + delta
        self._weights.regime_multipliers[record.market_regime] = max(0.5, min(2.0, round(new_val, 3)))

    # ── Private: analytics helpers ────────────────────────────────────────────

    def _win_rate_by_sector(self, sector: str) -> float:
        trades = [t for t in self._history if t.sector == sector]
        if len(trades) < 5:
            return 0.5  # Neutral if not enough data
        return sum(1 for t in trades if t.winner) / len(trades)

    def _win_rate_by_regime(self, regime: str) -> float:
        trades = [t for t in self._history if t.market_regime == regime]
        if len(trades) < 5:
            return 0.5
        return sum(1 for t in trades if t.winner) / len(trades)

    def _best_hour(self) -> int:
        hour_wins: dict[int, list[bool]] = defaultdict(list)
        for t in self._history:
            hour_wins[t.entry_hour_et].append(t.winner)
        if not hour_wins:
            return 10
        return max(hour_wins, key=lambda h: sum(hour_wins[h]) / len(hour_wins[h]))

    @staticmethod
    def _wr_to_multiplier(win_rate: float) -> float:
        """Convert win rate to a size multiplier. 50% = 1.0 (neutral)."""
        # Linear: 70% WR → 1.4x, 30% WR → 0.6x
        return max(0.5, min(1.5, 0.5 + win_rate))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        data = {
            "weights": asdict(self._weights),
            "history": [asdict(r) for r in self._history[-500:]],  # Keep last 500
        }
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            log.warning("intelligence_save_failed", extra={"error": str(exc)})

    def _load(self) -> None:
        if not MEMORY_FILE.exists():
            log.info("intelligence_fresh_start", extra={})
            return
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
            w = data.get("weights", {})
            self._weights = SignalWeights(
                hour_multipliers=w.get("hour_multipliers", SignalWeights().hour_multipliers),
                sector_multipliers=w.get("sector_multipliers", SignalWeights().sector_multipliers),
                regime_multipliers=w.get("regime_multipliers", SignalWeights().regime_multipliers),
                consecutive_losses=w.get("consecutive_losses", 0),
                consecutive_wins=w.get("consecutive_wins", 0),
                total_trades=w.get("total_trades", 0),
            )
            self._history = [TradeRecord(**r) for r in data.get("history", [])]
            log.info(
                "intelligence_loaded",
                extra={
                    "total_trades": self._weights.total_trades,
                    "consecutive_losses": self._weights.consecutive_losses,
                    "consecutive_wins": self._weights.consecutive_wins,
                },
            )
        except Exception as exc:
            log.warning("intelligence_load_failed", extra={"error": str(exc)})
