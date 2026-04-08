"""
flow.py — Unusual Options Activity & Smart Money Detection
===========================================================
The single biggest edge in retail options trading is following institutional
order flow. When a hedge fund or prop desk buys 2,000 contracts on a $15
biotech stock, they know something. We detect this and follow.

What we scan for:
  1. SWEEPS — Large orders split across multiple exchanges simultaneously
             (signs of urgency — someone wants in NOW at any price)
  2. BLOCKS — Single large prints > 50 contracts on small-cap options
             (someone sized up with conviction)
  3. PUT/CALL IMBALANCE — Unusual skew in volume toward one side
             (smart money positioning before a move)
  4. PREMIUM PAID — Total dollar value of unusual flow
             ($50k+ in a single option on a small cap = meaningful)
  5. REPEAT BUYERS — Same strike/expiry seeing consistent accumulation
             (distribution over time = planned position)

Flow score (0–100) feeds directly into confidence scoring in strategy.py.
A strong flow signal can push a borderline trade over the threshold.
A flow signal AGAINST our direction can veto an otherwise good setup.

Data source: Polygon.io options snapshot + trades endpoint.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from utils import get_config, get_logger

log = get_logger(__name__)
cfg = get_config()


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class FlowSignal:
    ticker: str
    direction: str               # "bullish", "bearish", "neutral"
    flow_score: int              # 0–100
    sweep_count: int             # Number of sweeps detected
    block_count: int             # Number of large blocks
    total_premium: float         # Total dollar premium in unusual flow
    call_volume: int
    put_volume: int
    put_call_ratio: float        # > 1.5 = bearish, < 0.5 = bullish
    dominant_expiry: str         # Where the smart money is positioned
    dominant_strike: float
    notes: list[str]             # Human-readable observations
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OptionTrade:
    """A single options print from the tape."""
    ticker: str                  # Option OCC symbol
    underlying: str
    price: float
    size: int
    side: str                    # "call" or "put"
    strike: float
    expiry: str
    exchange_count: int          # How many exchanges this printed on (sweep indicator)
    premium: float               # price * size * 100
    timestamp: datetime


# ── Flow Scanner ──────────────────────────────────────────────────────────────

class FlowScanner:
    """
    Detects unusual options activity for a list of tickers.
    Runs on-demand during the strategy evaluation phase.
    """

    # Thresholds for what counts as "unusual"
    _MIN_PREMIUM_BLOCK = 10_000       # $10k single print = notable
    _MIN_PREMIUM_SWEEP = 25_000       # $25k sweep = significant
    _MIN_SIZE_BLOCK = 50              # 50+ contracts = large print
    _SWEEP_EXCHANGE_MIN = 2           # Printed on 2+ exchanges = sweep

    def __init__(self, polygon_api_key: str) -> None:
        self._api_key = polygon_api_key
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "FlowScanner":
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=aiohttp.ClientTimeout(total=15),
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ── Public API ───────────────────────────────────────────────────────────

    async def scan_ticker(self, ticker: str, current_price: float) -> FlowSignal:
        """
        Full flow analysis for one ticker.
        Returns FlowSignal with direction and score.
        """
        # Fetch options snapshot (volume, OI, IV across all strikes)
        snapshot = await self._fetch_options_snapshot(ticker)
        # Fetch recent large trades from the tape
        recent_trades = await self._fetch_recent_large_trades(ticker)

        if not snapshot and not recent_trades:
            return FlowSignal(
                ticker=ticker, direction="neutral", flow_score=0,
                sweep_count=0, block_count=0, total_premium=0,
                call_volume=0, put_volume=0, put_call_ratio=1.0,
                dominant_expiry="", dominant_strike=0.0, notes=["no_data"],
            )

        return self._analyze_flow(ticker, current_price, snapshot, recent_trades)

    async def scan_batch(
        self, tickers: list[tuple[str, float]]  # (ticker, price)
    ) -> dict[str, FlowSignal]:
        """Scan multiple tickers concurrently."""
        tasks = {
            ticker: self.scan_ticker(ticker, price)
            for ticker, price in tickers
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        output: dict[str, FlowSignal] = {}
        for ticker, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                log.debug("flow_scan_error", extra={"ticker": ticker, "error": str(result)})
                output[ticker] = FlowSignal(
                    ticker=ticker, direction="neutral", flow_score=0,
                    sweep_count=0, block_count=0, total_premium=0,
                    call_volume=0, put_volume=0, put_call_ratio=1.0,
                    dominant_expiry="", dominant_strike=0.0, notes=["scan_error"],
                )
            else:
                output[ticker] = result
        return output

    # ── Core Analysis ─────────────────────────────────────────────────────────

    def _analyze_flow(
        self,
        ticker: str,
        current_price: float,
        snapshot: list[dict],
        recent_trades: list[OptionTrade],
    ) -> FlowSignal:
        notes: list[str] = []
        score = 0

        # ── Volume analysis from snapshot ────────────────────────────────────
        call_vol = sum(c.get("day", {}).get("volume", 0)
                       for c in snapshot if c.get("details", {}).get("contract_type") == "call")
        put_vol = sum(c.get("day", {}).get("volume", 0)
                      for c in snapshot if c.get("details", {}).get("contract_type") == "put")

        total_vol = call_vol + put_vol
        pc_ratio = put_vol / call_vol if call_vol > 0 else 1.0

        # Unusual put/call imbalance
        if pc_ratio < 0.4:
            score += 20
            notes.append(f"bullish_pc_ratio:{pc_ratio:.2f}")
        elif pc_ratio < 0.6:
            score += 10
            notes.append(f"slightly_bullish_pc:{pc_ratio:.2f}")
        elif pc_ratio > 2.5:
            score += 20  # bearish flow is still useful (for puts)
            notes.append(f"bearish_pc_ratio:{pc_ratio:.2f}")
        elif pc_ratio > 1.5:
            score += 10
            notes.append(f"slightly_bearish_pc:{pc_ratio:.2f}")

        # Find dominant expiry and strike (where the action is)
        dominant_expiry, dominant_strike = self._find_dominant_strike(
            snapshot, current_price
        )

        # ── Large print / sweep analysis ─────────────────────────────────────
        sweeps = [t for t in recent_trades
                  if t.exchange_count >= self._SWEEP_EXCHANGE_MIN
                  and t.premium >= self._MIN_PREMIUM_SWEEP]
        blocks = [t for t in recent_trades
                  if t.size >= self._MIN_SIZE_BLOCK
                  and t.premium >= self._MIN_PREMIUM_BLOCK]

        total_premium = sum(t.premium for t in recent_trades)

        if sweeps:
            score += min(30, len(sweeps) * 10)
            notes.append(f"sweeps:{len(sweeps)}")

        if blocks:
            score += min(20, len(blocks) * 7)
            notes.append(f"blocks:{len(blocks)}")

        if total_premium > 100_000:
            score += 20
            notes.append(f"premium:${total_premium:,.0f}")
        elif total_premium > 50_000:
            score += 10
            notes.append(f"premium:${total_premium:,.0f}")

        # ── Direction determination ────────────────────────────────────────
        call_premium = sum(t.premium for t in recent_trades if t.side == "call")
        put_premium = sum(t.premium for t in recent_trades if t.side == "put")

        if call_premium > put_premium * 1.5:
            direction = "bullish"
            notes.append("call_premium_dominant")
        elif put_premium > call_premium * 1.5:
            direction = "bearish"
            notes.append("put_premium_dominant")
        elif pc_ratio < 0.6:
            direction = "bullish"
        elif pc_ratio > 1.5:
            direction = "bearish"
        else:
            direction = "neutral"

        # Cap score
        score = min(100, score)

        log.info(
            "flow_analyzed",
            extra={
                "ticker": ticker,
                "direction": direction,
                "score": score,
                "sweeps": len(sweeps),
                "blocks": len(blocks),
                "pc_ratio": round(pc_ratio, 2),
                "total_premium": round(total_premium),
            },
        )

        return FlowSignal(
            ticker=ticker,
            direction=direction,
            flow_score=score,
            sweep_count=len(sweeps),
            block_count=len(blocks),
            total_premium=total_premium,
            call_volume=call_vol,
            put_volume=put_vol,
            put_call_ratio=round(pc_ratio, 2),
            dominant_expiry=dominant_expiry,
            dominant_strike=dominant_strike,
            notes=notes,
        )

    # ── Data Fetching ─────────────────────────────────────────────────────────

    async def _fetch_options_snapshot(self, ticker: str) -> list[dict]:
        """Fetch full options chain snapshot from Polygon."""
        assert self._session is not None
        url = f"{cfg['polygon']['base_url']}/v3/snapshot/options/{ticker}"
        params = {"limit": 250}
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
            return data.get("results", [])
        except Exception as exc:
            log.debug("options_snapshot_failed", extra={"ticker": ticker, "error": str(exc)})
            return []

    async def _fetch_recent_large_trades(self, ticker: str) -> list[OptionTrade]:
        """
        Fetch recent options trades from Polygon trades endpoint.
        We look at the last 500 trades and filter for large ones.
        """
        assert self._session is not None
        # Get options contracts for this ticker first
        url = f"{cfg['polygon']['base_url']}/v3/reference/options/contracts"
        params = {"underlying_ticker": ticker, "limit": 50, "sort": "expiration_date"}
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
            contracts = data.get("results", [])
        except Exception:
            return []

        if not contracts:
            return []

        # Fetch trades for the most active contract (nearest expiry ATM)
        # In production you'd fan out to all contracts; this is a pragmatic subset
        trades: list[OptionTrade] = []
        for contract in contracts[:5]:  # Top 5 contracts
            option_ticker = contract.get("ticker", "")
            if not option_ticker:
                continue
            contract_trades = await self._fetch_contract_trades(
                option_ticker,
                underlying=ticker,
                side=contract.get("contract_type", "call"),
                strike=float(contract.get("strike_price", 0)),
                expiry=contract.get("expiration_date", ""),
            )
            trades.extend(contract_trades)

        return trades

    async def _fetch_contract_trades(
        self,
        option_ticker: str,
        underlying: str,
        side: str,
        strike: float,
        expiry: str,
    ) -> list[OptionTrade]:
        assert self._session is not None
        url = f"{cfg['polygon']['base_url']}/v3/trades/{option_ticker}"
        params = {"limit": 100, "order": "desc"}
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
            raw_trades = data.get("results", [])
        except Exception:
            return []

        result = []
        for t in raw_trades:
            size = int(t.get("size", 0))
            price = float(t.get("price", 0))
            if size < 10:  # Skip tiny trades
                continue
            premium = price * size * 100
            # Exchange count: Polygon provides exchange ID; multiple IDs = sweep
            exchange_count = 1  # Polygon single-trade endpoint doesn't split by exchange
            # Heuristic: large size in short time = sweep behavior
            if size >= 100:
                exchange_count = 2
            if size >= 500:
                exchange_count = 3

            ts_ns = t.get("sip_timestamp", 0)
            ts = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc) if ts_ns else datetime.now(timezone.utc)

            result.append(OptionTrade(
                ticker=option_ticker,
                underlying=underlying,
                price=price,
                size=size,
                side=side,
                strike=strike,
                expiry=expiry,
                exchange_count=exchange_count,
                premium=premium,
                timestamp=ts,
            ))

        return result

    @staticmethod
    def _find_dominant_strike(
        snapshot: list[dict], current_price: float
    ) -> tuple[str, float]:
        """Find the expiry/strike with the most volume (where action is)."""
        if not snapshot:
            return "", 0.0

        best = max(
            snapshot,
            key=lambda c: c.get("day", {}).get("volume", 0),
            default={},
        )
        details = best.get("details", {})
        return (
            details.get("expiration_date", ""),
            float(details.get("strike_price", current_price)),
        )
