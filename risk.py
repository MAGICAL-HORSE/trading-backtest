"""
risk.py — Position sizing and kill switches
============================================
All hard risk rules live exclusively here. No other module bypasses these.

Rules enforced:
  1. Daily loss limit (4% of account) — halt all trading
  2. Per-sector cap (max 2 open in biotech, max 2 in energy)
  3. Trading time windows (no entries 9:30–9:40, no new after 3:15, close all by 3:45)
  4. Stop-loss monitoring: 35% of premium paid → market exit signal
  5. Profit target: 30–100% on premium paid (configurable)
  6. Max hold time: 90 minutes hard exit
  7. Liquidity guard: bid/ask spread > $0.20 at entry → skip
  8. Gap risk: stock gaps > 8% against position direction → immediate exit
  9. Position sizing: 5/8/12% of account based on confidence score

Design decisions:
  • RiskManager is a stateful singleton that main.py holds for the session.
  • All mutating operations (add_position, record_pnl, etc.) are synchronous
    because they only touch in-memory state; no I/O.
  • Exit signals are returned as RiskAction objects; the caller (main.py)
    is responsible for actually executing the orders through broker.py.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from utils import (
    can_enter_new_trade,
    env_float,
    get_config,
    get_logger,
    must_close_all,
    now_et,
    parse_time_et,
)

log = get_logger(__name__)
cfg = get_config()

ExitReason = Literal[
    "stop_loss",
    "profit_target",
    "max_hold_time",
    "force_close_eod",
    "gap_risk",
    "daily_limit_hit",
    "manual",
]


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Position:
    """Tracks a single open option position."""
    position_id: str             # UUID or order ID from broker
    ticker: str
    option_ticker: str
    direction: Literal["call", "put"]
    sector: str
    quantity: int                # Number of contracts
    entry_premium: float         # Per-contract mid-price at fill
    entry_price: float           # Underlying stock price at entry
    entry_time: datetime
    stop_price: float            # 35% loss level, computed at entry
    profit_target_min: float     # 30% gain level
    profit_target_max: float     # 100% gain level
    expiration: str              # ISO date

    # Updated in real-time by monitor loop
    last_premium: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    @property
    def max_exit_time(self) -> datetime:
        max_hold: int = cfg["risk"]["max_hold_minutes"]
        return self.entry_time + timedelta(minutes=max_hold)

    @property
    def cost_basis(self) -> float:
        """Total dollars at risk (entry_premium × 100 × contracts)."""
        return self.entry_premium * 100 * self.quantity


@dataclass
class RiskAction:
    """Instruction emitted by RiskManager to the execution layer."""
    position_id: str
    action: Literal["exit", "hold"]
    reason: ExitReason | None = None
    urgency: Literal["market", "limit"] = "market"


@dataclass
class SizingResult:
    """Output of position sizing calculation."""
    allowed: bool
    quantity: int                # Contracts to buy
    dollar_risk: float           # Entry premium × 100 × qty
    size_pct: float              # % of account used
    rejection_reason: str = ""


# ── Risk Manager ─────────────────────────────────────────────────────────────

class RiskManager:

    def __init__(self) -> None:
        self._lock = threading.Lock()  # Protect shared state in async context
        self._positions: dict[str, Position] = {}
        self._daily_realized_pnl: float = 0.0
        self._halted: bool = False
        self._halt_reason: str = ""

        rc = cfg["risk"]
        self.account_capital: float = env_float("ACCOUNT_CAPITAL", 25_000.0)
        self.stop_loss_pct: float = rc["stop_loss_pct"]         # 35.0
        self.profit_target_min_pct: float = rc["profit_target_min_pct"]  # 30.0
        self.profit_target_max_pct: float = rc["profit_target_max_pct"]  # 100.0
        self.daily_loss_limit_pct: float = rc["daily_loss_limit_pct"]    # 4.0
        self.max_hold_minutes: int = rc["max_hold_minutes"]
        self.max_per_sector: dict[str, int] = rc["max_open_per_sector"]
        self.gap_exit_threshold_pct: float = rc["gap_exit_threshold_pct"]
        self.max_spread_entry: float = cfg["options"]["max_spread_entry"]

        # Confidence → position size tiers
        self._size_tiers: list[dict] = cfg["strategy"]["confidence"]["tiers"]

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def daily_pnl(self) -> float:
        with self._lock:
            return self._daily_realized_pnl + sum(
                p.unrealized_pnl for p in self._positions.values()
            )

    @property
    def open_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    @property
    def open_count(self) -> int:
        with self._lock:
            return len(self._positions)

    # ── Pre-trade Checks ─────────────────────────────────────────────────────

    def pre_trade_check(
        self,
        sector: str,
        confidence: int,
        option_mid: float,
        option_spread: float,
    ) -> SizingResult:
        """
        Validate all pre-trade conditions and compute position size.
        Returns SizingResult; caller must check `.allowed` before ordering.
        """
        # 1. Halt check
        if self._halted:
            return SizingResult(False, 0, 0, 0, f"bot_halted:{self._halt_reason}")

        # 2. Time window
        allowed, reason = can_enter_new_trade()
        if not allowed:
            return SizingResult(False, 0, 0, 0, f"time_window:{reason}")

        # 3. Daily loss limit
        daily_loss_pct = (self.daily_pnl / self.account_capital) * 100
        if daily_loss_pct <= -self.daily_loss_limit_pct:
            self._halt(f"daily_loss_limit:{daily_loss_pct:.2f}%")
            return SizingResult(False, 0, 0, 0, "daily_loss_limit_hit")

        # 4. Sector cap
        sector_count = sum(
            1 for p in self._positions.values() if p.sector == sector
        )
        max_sector = self.max_per_sector.get(sector, 2)
        if sector_count >= max_sector:
            return SizingResult(False, 0, 0, 0, f"sector_cap:{sector}={sector_count}")

        # 5. Liquidity guard
        if option_spread > self.max_spread_entry:
            return SizingResult(
                False, 0, 0, 0,
                f"spread_too_wide:{option_spread:.3f}>{self.max_spread_entry}"
            )

        # 6. Confidence threshold
        if confidence < cfg["strategy"]["confidence"]["min_to_trade"]:
            return SizingResult(False, 0, 0, 0, f"confidence_below_min:{confidence}")

        # 7. Position sizing
        size_pct = self._get_size_pct(confidence)
        dollar_allocation = self.account_capital * (size_pct / 100)

        if option_mid <= 0:
            return SizingResult(False, 0, 0, 0, "option_mid_zero")

        # 1 contract = 100 shares
        max_contracts = int(dollar_allocation / (option_mid * 100))
        if max_contracts < 1:
            return SizingResult(
                False, 0, 0, 0,
                f"allocation_too_small:{dollar_allocation:.2f}<{option_mid*100:.2f}"
            )

        dollar_risk = option_mid * 100 * max_contracts
        return SizingResult(
            allowed=True,
            quantity=max_contracts,
            dollar_risk=dollar_risk,
            size_pct=size_pct,
        )

    # ── Position Management ──────────────────────────────────────────────────

    def add_position(self, position: Position) -> None:
        with self._lock:
            self._positions[position.position_id] = position
        log.info(
            "position_opened",
            extra={
                "position_id": position.position_id,
                "ticker": position.ticker,
                "direction": position.direction,
                "qty": position.quantity,
                "entry_premium": position.entry_premium,
                "sector": position.sector,
                "stop_price": position.stop_price,
            },
        )

    def remove_position(self, position_id: str, realized_pnl: float) -> None:
        with self._lock:
            pos = self._positions.pop(position_id, None)
            if pos is not None:
                self._daily_realized_pnl += realized_pnl

        if pos:
            log.info(
                "position_closed",
                extra={
                    "position_id": position_id,
                    "ticker": pos.ticker,
                    "realized_pnl": round(realized_pnl, 2),
                    "daily_pnl": round(self._daily_realized_pnl, 2),
                },
            )

        # Recheck daily limit after each close
        self._check_daily_limit()

    def update_position_premium(self, position_id: str, current_premium: float) -> None:
        """Called by the monitoring loop with the latest mid-price."""
        with self._lock:
            pos = self._positions.get(position_id)
            if pos is None:
                return
            pos.last_premium = current_premium
            dollar_pnl = (current_premium - pos.entry_premium) * 100 * pos.quantity
            pct_pnl = ((current_premium - pos.entry_premium) / pos.entry_premium) * 100
            pos.unrealized_pnl = dollar_pnl
            pos.unrealized_pnl_pct = pct_pnl

    # ── Monitor (called each tick) ────────────────────────────────────────────

    def check_exits(self) -> list[RiskAction]:
        """
        Evaluate all open positions for exit triggers.
        Returns a list of RiskActions; may be empty.
        Call this frequently (e.g., every second in the main loop).
        """
        actions: list[RiskAction] = []
        now = datetime.now(timezone.utc)
        force_close = must_close_all()

        with self._lock:
            for pid, pos in list(self._positions.items()):
                action = self._evaluate_position(pos, now, force_close)
                if action is not None:
                    actions.append(action)

        return actions

    def check_gap_risk(self, position_id: str, current_stock_price: float) -> RiskAction | None:
        """
        Check whether the underlying has gapped adversely against the position.
        Call this after receiving a fresh stock quote.
        """
        with self._lock:
            pos = self._positions.get(position_id)
            if pos is None:
                return None

        entry_price = pos.entry_price
        if entry_price == 0:
            return None

        change_pct = ((current_stock_price - entry_price) / entry_price) * 100
        adverse = (pos.direction == "call" and change_pct <= -self.gap_exit_threshold_pct) or \
                  (pos.direction == "put" and change_pct >= self.gap_exit_threshold_pct)

        if adverse:
            log.warning(
                "gap_risk_exit",
                extra={
                    "position_id": position_id,
                    "ticker": pos.ticker,
                    "entry_price": entry_price,
                    "current_price": current_stock_price,
                    "change_pct": round(change_pct, 2),
                },
            )
            return RiskAction(position_id, "exit", "gap_risk", "market")
        return None

    # ── Halt Controls ─────────────────────────────────────────────────────────

    def manual_halt(self, reason: str = "manual") -> None:
        self._halt(reason)

    def resume(self) -> None:
        """Reset halt state (use with caution — only after reviewing conditions)."""
        with self._lock:
            self._halted = False
            self._halt_reason = ""
        log.warning("trading_resumed", extra={})

    def reset_daily(self) -> None:
        """Call at start of each trading day to reset P&L counters."""
        with self._lock:
            self._daily_realized_pnl = 0.0
            self._halted = False
            self._halt_reason = ""
            self._positions.clear()
        log.info("daily_reset_complete", extra={})

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _evaluate_position(
        self, pos: Position, now: datetime, force_close: bool
    ) -> RiskAction | None:
        """Return exit action if any trigger fires, else None."""
        # Force close (3:45 PM ET)
        if force_close:
            return RiskAction(pos.position_id, "exit", "force_close_eod", "market")

        # Max hold time (90 min hard exit)
        if now >= pos.max_exit_time:
            log.info(
                "max_hold_exit",
                extra={"position_id": pos.position_id, "ticker": pos.ticker},
            )
            return RiskAction(pos.position_id, "exit", "max_hold_time", "market")

        # Stop loss: 35% loss on premium
        if pos.entry_premium > 0 and pos.last_premium > 0:
            loss_pct = ((pos.last_premium - pos.entry_premium) / pos.entry_premium) * 100
            if loss_pct <= -self.stop_loss_pct:
                log.warning(
                    "stop_loss_triggered",
                    extra={
                        "position_id": pos.position_id,
                        "ticker": pos.ticker,
                        "loss_pct": round(loss_pct, 1),
                    },
                )
                return RiskAction(pos.position_id, "exit", "stop_loss", "market")

            # Profit target (take at 100%, or let run from 30% with trailing)
            if loss_pct >= self.profit_target_max_pct:
                log.info(
                    "profit_target_max_hit",
                    extra={
                        "position_id": pos.position_id,
                        "ticker": pos.ticker,
                        "gain_pct": round(loss_pct, 1),
                    },
                )
                return RiskAction(pos.position_id, "exit", "profit_target", "limit")

        return None

    def _get_size_pct(self, confidence: int) -> float:
        """Map confidence score to position size percentage."""
        for tier in reversed(self._size_tiers):
            if confidence >= tier["min"]:
                return float(tier["size_pct"])
        return float(self._size_tiers[0]["size_pct"])

    def _halt(self, reason: str) -> None:
        with self._lock:
            self._halted = True
            self._halt_reason = reason
        log.critical(
            "trading_halted",
            extra={"reason": reason, "daily_pnl": round(self._daily_realized_pnl, 2)},
        )

    def _check_daily_limit(self) -> None:
        pnl_pct = (self._daily_realized_pnl / self.account_capital) * 100
        if pnl_pct <= -self.daily_loss_limit_pct:
            self._halt(f"daily_loss_limit:{pnl_pct:.2f}%")

    # ── Factory for Position object ───────────────────────────────────────────

    @staticmethod
    def build_position(
        position_id: str,
        ticker: str,
        option_ticker: str,
        direction: Literal["call", "put"],
        sector: str,
        quantity: int,
        entry_premium: float,
        entry_stock_price: float,
        expiration: str,
    ) -> Position:
        rc = cfg["risk"]
        stop_pct = rc["stop_loss_pct"] / 100
        profit_min_pct = rc["profit_target_min_pct"] / 100
        profit_max_pct = rc["profit_target_max_pct"] / 100

        stop_price = entry_premium * (1 - stop_pct)
        profit_min = entry_premium * (1 + profit_min_pct)
        profit_max = entry_premium * (1 + profit_max_pct)

        return Position(
            position_id=position_id,
            ticker=ticker,
            option_ticker=option_ticker,
            direction=direction,
            sector=sector,
            quantity=quantity,
            entry_premium=entry_premium,
            entry_price=entry_stock_price,
            entry_time=datetime.now(timezone.utc),
            stop_price=stop_price,
            profit_target_min=profit_min,
            profit_target_max=profit_max,
            expiration=expiration,
            last_premium=entry_premium,
        )
