"""
market_regime.py — Market context engine
=========================================
A 20-year trader never trades in a vacuum. They always know:
  • Is the market trending or chopping?
  • Is volatility expanding or contracting?
  • Are small caps leading or lagging?
  • What's the tape saying at the open?

This module answers those questions every 5 minutes and returns a
MarketContext that the strategy engine uses to:
  1. Filter out trades in unfavorable conditions (choppy, low-volatility)
  2. Bias direction (don't buy calls on a -1.5% SPY day)
  3. Adjust position sizing (bigger in trending markets)
  4. Time entries (best setups cluster around specific market conditions)

Regime classifications:
  BULL_TREND       — SPY/QQQ both up >0.5%, IWM leading, VIX < 20
  BEAR_TREND       — SPY/QQQ both down >0.5%, puts favored
  HIGH_VOLATILITY  — VIX > 25, large intraday swings — explosive options moves
  CHOPPY           — SPY/QQQ < 0.3% move, no clear direction — avoid
  UNKNOWN          — Can't determine (pre-market, data missing)

Additional context signals:
  • Opening gap direction (first 5 min vs prior close)
  • TICK extremes (breadth of market)
  • IWM vs SPY relative strength (small caps leading = risk-on)
  • Sector ETF rotation (XLK, XLF, XLE, XBI relative performance)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import aiohttp

from utils import get_config, get_logger, now_et

log = get_logger(__name__)
cfg = get_config()

Regime = Literal["BULL_TREND", "BEAR_TREND", "HIGH_VOLATILITY", "CHOPPY", "UNKNOWN"]


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class MarketContext:
    regime: Regime
    spy_change_pct: float        # SPY intraday %
    qqq_change_pct: float        # QQQ intraday %
    iwm_change_pct: float        # IWM intraday % (small cap pulse)
    vix_level: float             # VIX spot
    vix_trending: str            # "up", "down", "flat"
    small_cap_leading: bool      # IWM outperforming SPY
    opening_gap_direction: int   # +1 gap up, -1 gap down, 0 flat
    sector_bias: str             # "biotech", "energy", "mixed", "defensive"
    trade_bias: Literal["calls", "puts", "neutral"]
    confidence_modifier: float   # Multiplier for confidence scores (0.5–1.5)
    notes: list[str]
    refreshed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Regime Engine ─────────────────────────────────────────────────────────────

class MarketRegimeEngine:
    """
    Fetches macro market data and classifies the current regime.
    Cached for 5 minutes between refreshes.
    """

    _CACHE_TTL = 300  # 5 minutes

    def __init__(self, polygon_api_key: str) -> None:
        self._api_key = polygon_api_key
        self._session: aiohttp.ClientSession | None = None
        self._cache: MarketContext | None = None
        self._cache_time: float = 0.0

    async def __aenter__(self) -> "MarketRegimeEngine":
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=aiohttp.ClientTimeout(total=10),
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ── Public API ───────────────────────────────────────────────────────────

    async def get_context(self, force_refresh: bool = False) -> MarketContext:
        """Get current market context. Cached for 5 minutes."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._cache is not None
            and now - self._cache_time < self._CACHE_TTL
        ):
            return self._cache

        ctx = await self._build_context()
        self._cache = ctx
        self._cache_time = now
        return ctx

    # ── Context Builder ───────────────────────────────────────────────────────

    async def _build_context(self) -> MarketContext:
        # Fetch all macro symbols in parallel
        symbols = ["SPY", "QQQ", "IWM", "VIX", "XBI", "XLE", "XLK", "XLF"]
        snapshots = await self._fetch_snapshots(symbols)

        spy = self._get_change(snapshots, "SPY")
        qqq = self._get_change(snapshots, "QQQ")
        iwm = self._get_change(snapshots, "IWM")
        vix = self._get_price(snapshots, "VIX")
        xbi = self._get_change(snapshots, "XBI")
        xle = self._get_change(snapshots, "XLE")

        notes: list[str] = []

        # ── VIX analysis ──────────────────────────────────────────────────────
        vix_level = vix if vix > 0 else 20.0
        vix_trending = self._vix_trend(snapshots)

        # ── Regime classification ─────────────────────────────────────────────
        avg_index = (spy + qqq) / 2

        if vix_level > 30:
            regime: Regime = "HIGH_VOLATILITY"
            notes.append(f"vix_extreme:{vix_level:.1f}")
        elif avg_index > 0.5 and iwm > 0:
            regime = "BULL_TREND"
            notes.append("indices_green")
        elif avg_index < -0.5 and iwm < 0:
            regime = "BEAR_TREND"
            notes.append("indices_red")
        elif abs(avg_index) < 0.3:
            regime = "CHOPPY"
            notes.append("low_range_chop")
        elif vix_level > 20:
            regime = "HIGH_VOLATILITY"
            notes.append(f"elevated_vix:{vix_level:.1f}")
        else:
            regime = "BULL_TREND" if avg_index > 0 else "BEAR_TREND"

        # ── Trade bias ────────────────────────────────────────────────────────
        if regime == "BULL_TREND":
            trade_bias: Literal["calls", "puts", "neutral"] = "calls"
        elif regime == "BEAR_TREND":
            trade_bias = "puts"
        else:
            trade_bias = "neutral"

        # ── Small cap leadership ──────────────────────────────────────────────
        small_cap_leading = iwm > spy + 0.2
        if small_cap_leading:
            notes.append("small_caps_leading")

        # ── Opening gap ───────────────────────────────────────────────────────
        opening_gap = self._detect_opening_gap(snapshots)
        if opening_gap > 0:
            notes.append("gap_up_open")
        elif opening_gap < 0:
            notes.append("gap_down_open")

        # ── Sector bias ──────────────────────────────────────────────────────
        if xbi > 1.0 and xle < 0.5:
            sector_bias = "biotech"
            notes.append("biotech_hot")
        elif xle > 1.0 and xbi < 0.5:
            sector_bias = "energy"
            notes.append("energy_hot")
        elif xbi > 0.5 and xle > 0.5:
            sector_bias = "mixed"
        else:
            sector_bias = "defensive"

        # ── Confidence modifier ───────────────────────────────────────────────
        conf_mod = self._compute_confidence_modifier(
            regime, vix_level, small_cap_leading, avg_index
        )

        ctx = MarketContext(
            regime=regime,
            spy_change_pct=round(spy, 2),
            qqq_change_pct=round(qqq, 2),
            iwm_change_pct=round(iwm, 2),
            vix_level=round(vix_level, 1),
            vix_trending=vix_trending,
            small_cap_leading=small_cap_leading,
            opening_gap_direction=opening_gap,
            sector_bias=sector_bias,
            trade_bias=trade_bias,
            confidence_modifier=conf_mod,
            notes=notes,
        )

        log.info(
            "market_regime_updated",
            extra={
                "regime": regime,
                "spy": round(spy, 2),
                "qqq": round(qqq, 2),
                "vix": round(vix_level, 1),
                "bias": trade_bias,
                "conf_mod": round(conf_mod, 2),
                "notes": notes,
            },
        )
        return ctx

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _fetch_snapshots(self, symbols: list[str]) -> dict[str, dict]:
        assert self._session is not None
        tickers_str = ",".join(symbols)
        url = (
            f"{cfg['polygon']['base_url']}/v2/snapshot/locale/us"
            f"/markets/stocks/tickers"
        )
        params = {"tickers": tickers_str}
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
            tickers = data.get("tickers", [])
            return {t["ticker"]: t for t in tickers}
        except Exception as exc:
            log.warning("macro_snapshot_failed", extra={"error": str(exc)})
            return {}

    @staticmethod
    def _get_change(snapshots: dict, symbol: str) -> float:
        snap = snapshots.get(symbol, {})
        day = snap.get("day", {})
        prev = snap.get("prevDay", {}).get("c", 0)
        last = day.get("c", prev)
        if prev and prev > 0:
            return ((last - prev) / prev) * 100
        return 0.0

    @staticmethod
    def _get_price(snapshots: dict, symbol: str) -> float:
        snap = snapshots.get(symbol, {})
        return float(snap.get("day", {}).get("c", 0) or snap.get("lastTrade", {}).get("p", 0))

    @staticmethod
    def _vix_trend(snapshots: dict) -> str:
        snap = snapshots.get("VIX", {})
        day = snap.get("day", {})
        open_price = day.get("o", 0)
        close_price = day.get("c", open_price)
        if close_price > open_price * 1.03:
            return "up"
        elif close_price < open_price * 0.97:
            return "down"
        return "flat"

    @staticmethod
    def _detect_opening_gap(snapshots: dict) -> int:
        """Compare SPY first 5-min open to prior close."""
        snap = snapshots.get("SPY", {})
        day_open = snap.get("day", {}).get("o", 0)
        prev_close = snap.get("prevDay", {}).get("c", 0)
        if not day_open or not prev_close:
            return 0
        gap_pct = ((day_open - prev_close) / prev_close) * 100
        if gap_pct > 0.3:
            return 1
        if gap_pct < -0.3:
            return -1
        return 0

    @staticmethod
    def _compute_confidence_modifier(
        regime: Regime,
        vix: float,
        small_cap_leading: bool,
        avg_index_change: float,
    ) -> float:
        """
        Returns a multiplier for confidence scores.
        Strong trending market with small caps leading = boost.
        Choppy = penalize.
        """
        base = {
            "BULL_TREND": 1.3,
            "BEAR_TREND": 1.2,
            "HIGH_VOLATILITY": 1.1,
            "CHOPPY": 0.5,
            "UNKNOWN": 0.8,
        }.get(regime, 1.0)

        # Small caps leading = risk-on, boost small cap options
        if small_cap_leading:
            base *= 1.1

        # Strong trend day (>1% move) = bigger setups
        if abs(avg_index_change) > 1.0:
            base *= 1.15

        # VIX sweet spot: 15–25 = options are moving but not too expensive
        if 15 <= vix <= 25:
            base *= 1.05
        elif vix > 35:
            base *= 0.9  # Too much fear = whipsaw risk

        return max(0.4, min(1.6, round(base, 2)))
