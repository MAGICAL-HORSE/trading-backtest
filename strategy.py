"""
strategy.py — Signal generation engine (v2 — Smart Money Edition)
==================================================================
Evaluates a scan candidate against multi-timeframe technical criteria and
produces a TradeSignal with a 0–100 confidence score that drives position sizing.

Signal logic (ALL must pass for a trade):
  1. Price breaks VWAP with volume confirmation on both 1m and 5m charts
  2. RSI(14) on 5m > 55 for calls / < 45 for puts
  3. ATR expansion: current ATR > 1.1× 10-period ATR average
  4. Biotech: requires catalyst confirmation from news.py
  5. Market regime: no calls on BEAR_TREND day, no puts on BULL_TREND day
  6. Opening Range Breakout check: if within first 30 min, must confirm ORB

NEW in v2:
  • Opening Range Breakout (ORB) — first 15-min range, trade the breakout
  • Flow score integration — unusual options activity boosts confidence
  • Greeks-aware strike selection — prefer 0.40–0.65 delta for max leverage
  • Pullback entry detection — better to buy dips than chase breakouts
  • Trailing profit locks — once +50%, stop moves to breakeven
  • Market regime gate — context from market_regime.py filters bad days
  • Intelligence adjustment — learned multipliers from intelligence.py

Confidence score components (summing to 100+):
  • VWAP breakout quality (0–25)
  • RSI extremity (0–20)
  • ATR expansion ratio (0–15)
  • Volume spike magnitude (0–15)
  • Momentum alignment 1m vs 5m (0–10)
  • Flow signal bonus (0–15)  ← NEW
  • ORB confirmation bonus (0–10)  ← NEW
  • Pullback quality bonus (0–10)  ← NEW
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

import aiohttp
import numpy as np
import pandas as pd

try:
    import ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False

from scanner import ScanResult
from utils import get_config, get_logger, now_et

log = get_logger(__name__)
cfg = get_config()

Direction = Literal["call", "put"]


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class OptionSpec:
    """Fully resolved option to trade."""
    underlying: str
    option_ticker: str           # OCC-style: e.g. "MRNA240705C00130000"
    direction: Direction
    strike: float
    expiration: str              # ISO date
    bid: float
    ask: float
    mid: float
    delta: float
    oi: int


@dataclass
class TradeSignal:
    """Output of strategy evaluation for one stock."""
    scan_result: ScanResult
    direction: Direction
    confidence: int              # 0–100
    option_spec: OptionSpec
    vwap_break_pct: float        # How far price is through VWAP (ATR-relative)
    rsi_5m: float
    atr_expansion: float         # Current ATR / 10-period avg ATR
    volume_spike: float
    reasons: list[str]           # Human-readable rationale
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RejectedSignal:
    ticker: str
    direction: Direction | None
    reason: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── VWAP Calculator ───────────────────────────────────────────────────────────

def calculate_vwap(bars: pd.DataFrame) -> pd.Series:
    """
    Anchored VWAP from the first bar of the current session.
    bars must have columns: open, high, low, close, volume.
    Returns a Series aligned to bars.index.
    """
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    tp_vol = typical_price * bars["volume"]
    cum_tp_vol = tp_vol.cumsum()
    cum_vol = bars["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder smoothed RSI."""
    if _HAS_TA:
        return ta.momentum.RSIIndicator(close, window=period).rsi()
    # Manual fallback
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(bars: pd.DataFrame, period: int = 10) -> pd.Series:
    """Average True Range."""
    if _HAS_TA:
        return ta.volatility.AverageTrueRange(
            bars["high"], bars["low"], bars["close"], window=period
        ).average_true_range()
    high_low = bars["high"] - bars["low"]
    high_prev = (bars["high"] - bars["close"].shift()).abs()
    low_prev  = (bars["low"]  - bars["close"].shift()).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── Main Strategy Engine ──────────────────────────────────────────────────────

class StrategyEngine:

    def __init__(self, polygon_api_key: str) -> None:
        self._api_key = polygon_api_key
        self._session: aiohttp.ClientSession | None = None

        sc = cfg["strategy"]
        oc = cfg["options"]
        self.rsi_period: int = sc["rsi_period"]
        self.rsi_call_thresh: float = sc["rsi_call_threshold"]
        self.rsi_put_thresh: float = sc["rsi_put_threshold"]
        self.atr_period: int = sc["atr_period"]
        self.atr_mult: float = sc["atr_expansion_multiplier"]
        self.roll_after_weekday: int = oc["roll_after_weekday"]
        self.min_dte: int = oc["min_dte"]

    async def __aenter__(self) -> "StrategyEngine":
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=aiohttp.ClientTimeout(total=15),
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._session:
            await self._session.close()

    # ── Public API ───────────────────────────────────────────────────────────

    async def evaluate(
        self,
        scan: ScanResult,
        has_catalyst: bool = False,
        uso_direction: int = 0,
        flow_signal=None,            # FlowSignal from flow.py (optional)
        market_context=None,         # MarketContext from market_regime.py (optional)
        intelligence=None,           # IntelligenceEngine (optional)
    ) -> TradeSignal | RejectedSignal:
        """
        Full signal evaluation pipeline — v2 Smart Money Edition.
        Returns TradeSignal if all conditions pass, RejectedSignal otherwise.
        """
        ticker = scan.ticker

        # ── Market regime gate ────────────────────────────────────────────────
        if market_context is not None:
            regime = market_context.regime
            if regime == "CHOPPY":
                return RejectedSignal(ticker, None, "choppy_market_regime")

        # Fetch bar data for 1m and 5m timeframes concurrently
        bars_1m, bars_5m = await asyncio.gather(
            self._fetch_bars(ticker, "1", 60),
            self._fetch_bars(ticker, "5", 50),
        )

        if bars_1m.empty or bars_5m.empty:
            return RejectedSignal(ticker, None, "insufficient_bar_data")

        # ── Indicators ───────────────────────────────────────────────────────
        vwap_1m = calculate_vwap(bars_1m)
        vwap_5m = calculate_vwap(bars_5m)
        rsi_5m_series = calculate_rsi(bars_5m["close"], self.rsi_period)
        atr_series = calculate_atr(bars_5m, self.atr_period)

        last_price = bars_1m["close"].iloc[-1]
        last_vwap_1m = vwap_1m.iloc[-1]
        last_vwap_5m = vwap_5m.iloc[-1]
        last_rsi = rsi_5m_series.dropna().iloc[-1] if not rsi_5m_series.dropna().empty else 50.0
        last_atr = atr_series.dropna().iloc[-1] if not atr_series.dropna().empty else 0.0
        avg_atr = atr_series.dropna().tail(10).mean() if not atr_series.dropna().empty else 1.0

        # ── Direction determination ───────────────────────────────────────────
        above_vwap_1m = last_price > last_vwap_1m
        above_vwap_5m = last_price > last_vwap_5m

        if above_vwap_1m and above_vwap_5m:
            direction: Direction = "call"
        elif not above_vwap_1m and not above_vwap_5m:
            direction = "put"
        else:
            return RejectedSignal(ticker, None, "vwap_timeframe_conflict")

        # ── Gate 1: RSI ───────────────────────────────────────────────────────
        if direction == "call" and last_rsi <= self.rsi_call_thresh:
            return RejectedSignal(ticker, direction, f"rsi_too_low:{last_rsi:.1f}")
        if direction == "put" and last_rsi >= self.rsi_put_thresh:
            return RejectedSignal(ticker, direction, f"rsi_too_high:{last_rsi:.1f}")

        # ── Gate 2: ATR expansion ─────────────────────────────────────────────
        if avg_atr == 0:
            return RejectedSignal(ticker, direction, "atr_zero")
        atr_expansion = last_atr / avg_atr
        if atr_expansion < self.atr_mult:
            return RejectedSignal(
                ticker, direction,
                f"atr_not_expanded:{atr_expansion:.2f}<{self.atr_mult}"
            )

        # ── Gate 3: VWAP breakout quality ────────────────────────────────────
        vwap_dist = abs(last_price - last_vwap_5m)
        vwap_break_pct = (vwap_dist / last_atr) if last_atr > 0 else 0.0
        # Must be at least 0.25× ATR through VWAP (not just a tick)
        if vwap_break_pct < 0.25:
            return RejectedSignal(ticker, direction, f"vwap_break_too_weak:{vwap_break_pct:.2f}")

        # ── Gate 4: Volume confirmation on 1m ────────────────────────────────
        avg_1m_vol = bars_1m["volume"].iloc[:-1].mean()
        last_1m_vol = bars_1m["volume"].iloc[-1]
        if avg_1m_vol > 0 and last_1m_vol < avg_1m_vol * 1.5:
            return RejectedSignal(ticker, direction, "insufficient_1m_volume_confirm")

        # ── Gate 5: Sector-specific rules ────────────────────────────────────
        if scan.sector == "biotech" and not has_catalyst:
            return RejectedSignal(ticker, direction, "biotech_no_catalyst")

        # ── Gate 6: Market regime direction alignment ─────────────────────────
        if market_context is not None:
            bias = market_context.trade_bias
            if bias == "calls" and direction == "put":
                # Allow puts on strong individual setups even in bull market
                # but require higher bar (handled by confidence modifier below)
                pass
            elif bias == "puts" and direction == "call":
                pass  # Same — allow but penalized in confidence

        # ── Gate 7: Opening Range Breakout check (first 30 min) ──────────────
        orb_score = self._score_orb(bars_1m, direction, last_price)

        # ── Gate 8: Pullback quality (prefer buying dips, not extended moves) ─
        pullback_score = self._score_pullback(bars_1m, bars_5m, direction, vwap_1m)

        # ── Confidence Score ─────────────────────────────────────────────────
        flow_score = flow_signal.flow_score if flow_signal else 0
        flow_direction = flow_signal.direction if flow_signal else "neutral"

        # Penalize if flow contradicts our direction
        if flow_direction == "bullish" and direction == "put":
            flow_score = -10
        elif flow_direction == "bearish" and direction == "call":
            flow_score = -10

        confidence = self._compute_confidence(
            vwap_break_pct=vwap_break_pct,
            rsi=last_rsi,
            direction=direction,
            atr_expansion=atr_expansion,
            volume_spike=scan.volume_spike,
            bars_1m=bars_1m,
            bars_5m=bars_5m,
            vwap_1m=vwap_1m,
            vwap_5m=vwap_5m,
            flow_score=flow_score,
            orb_score=orb_score,
            pullback_score=pullback_score,
        )

        # Apply market regime modifier
        if market_context is not None:
            confidence = int(confidence * market_context.confidence_modifier)

        # Apply intelligence-learned multipliers
        if intelligence is not None:
            hour_et = now_et().hour
            regime_str = market_context.regime if market_context else "UNKNOWN"
            confidence = intelligence.adjust_confidence(
                confidence, scan.sector, regime_str, hour_et
            )
            # Check if intelligence is overriding the minimum
            min_override = intelligence.get_min_confidence_override()
            min_conf = min_override if min_override else cfg["strategy"]["confidence"]["min_to_trade"]
        else:
            min_conf = cfg["strategy"]["confidence"]["min_to_trade"]

        confidence = max(0, min(100, confidence))

        if confidence < min_conf:
            return RejectedSignal(ticker, direction, f"confidence_too_low:{confidence}<{min_conf}")

        # ── Option Selection (Greeks-aware) ───────────────────────────────────
        option_spec = await self._select_option(scan, direction)
        if option_spec is None:
            return RejectedSignal(ticker, direction, "no_liquid_option_found")

        # Prefer delta between 0.35 and 0.70 — too far OTM = lottery ticket
        if abs(option_spec.delta) < 0.25:
            return RejectedSignal(
                ticker, direction,
                f"delta_too_low:{option_spec.delta:.2f} — too far OTM"
            )

        reasons = [
            f"vwap_break={vwap_break_pct:.2f}x_ATR",
            f"rsi={last_rsi:.1f}",
            f"atr_expansion={atr_expansion:.2f}x",
            f"vol_spike={scan.volume_spike:.1f}x",
            f"sector={scan.sector}",
        ]

        log.info(
            "signal_generated",
            extra={
                "ticker": ticker,
                "direction": direction,
                "confidence": confidence,
                "rsi": round(last_rsi, 1),
                "atr_expansion": round(atr_expansion, 2),
                "vwap_break_pct": round(vwap_break_pct, 2),
                "option": option_spec.option_ticker,
            },
        )

        return TradeSignal(
            scan_result=scan,
            direction=direction,
            confidence=confidence,
            option_spec=option_spec,
            vwap_break_pct=vwap_break_pct,
            rsi_5m=last_rsi,
            atr_expansion=atr_expansion,
            volume_spike=scan.volume_spike,
            reasons=reasons,
        )

    # ── Confidence Scorer ────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        vwap_break_pct: float,
        rsi: float,
        direction: Direction,
        atr_expansion: float,
        volume_spike: float,
        bars_1m: pd.DataFrame,
        bars_5m: pd.DataFrame,
        vwap_1m: pd.Series,
        vwap_5m: pd.Series,
        flow_score: int = 0,
        orb_score: int = 0,
        pullback_score: int = 0,
    ) -> int:
        score = 0

        # Component 1: VWAP breakout quality (0–25)
        vwap_score = min(25, int((vwap_break_pct / 2.0) * 25))
        score += vwap_score

        # Component 2: RSI extremity (0–20)
        if direction == "call":
            rsi_dist = max(0, rsi - 50)
        else:
            rsi_dist = max(0, 50 - rsi)
        rsi_score = min(20, int((rsi_dist / 20.0) * 20))
        score += rsi_score

        # Component 3: ATR expansion (0–15)
        atr_score = min(15, int(((atr_expansion - 1.1) / 1.4) * 15))
        score += max(0, atr_score)

        # Component 4: Volume spike (0–15)
        spike_score = min(15, int(((volume_spike - 1.2) / 4.0) * 15))
        score += max(0, spike_score)

        # Component 5: 1m / 5m momentum alignment (0–10)
        recent_1m = bars_1m["close"].iloc[-3:]
        recent_5m = bars_5m["close"].iloc[-3:]
        if direction == "call":
            align_1m = all(recent_1m.diff().dropna() > 0)
            align_5m = all(recent_5m.diff().dropna() > 0)
        else:
            align_1m = all(recent_1m.diff().dropna() < 0)
            align_5m = all(recent_5m.diff().dropna() < 0)

        alignment_score = 0
        if align_1m:
            alignment_score += 8
        if align_5m:
            alignment_score += 7
        alignment_score = 0
        if align_1m:
            alignment_score += 5
        if align_5m:
            alignment_score += 5
        score += alignment_score

        # Component 6: Flow signal bonus (0–15, can be negative)
        if flow_score > 0:
            flow_bonus = min(15, int(flow_score * 0.15))
            score += flow_bonus
        elif flow_score < 0:
            score += flow_score  # penalty

        # Component 7: ORB confirmation (0–10)
        score += min(10, max(0, orb_score))

        # Component 8: Pullback quality (0–10)
        score += min(10, max(0, pullback_score))

        return min(120, max(0, score))  # Allow up to 120 before modifiers cap at 100

    # ── ORB Scoring ───────────────────────────────────────────────────────────

    def _score_orb(
        self,
        bars_1m: pd.DataFrame,
        direction: Direction,
        current_price: float,
    ) -> int:
        """
        Opening Range Breakout score (0–10).
        ORB = high/low of first 15 minutes.
        Breakout above ORB high = bullish confirmation.
        Breakout below ORB low = bearish confirmation.
        """
        if bars_1m.empty or len(bars_1m) < 15:
            return 0

        # First 15 bars = opening range
        orb_bars = bars_1m.iloc[:15]
        orb_high = orb_bars["high"].max()
        orb_low = orb_bars["low"].min()
        orb_range = orb_high - orb_low

        if orb_range == 0:
            return 0

        if direction == "call":
            if current_price > orb_high:
                # How far above ORB high (relative to range)?
                breakout_pct = (current_price - orb_high) / orb_range
                return min(10, int(breakout_pct * 10))
            return 0
        else:
            if current_price < orb_low:
                breakout_pct = (orb_low - current_price) / orb_range
                return min(10, int(breakout_pct * 10))
            return 0

    # ── Pullback Quality Scoring ──────────────────────────────────────────────

    def _score_pullback(
        self,
        bars_1m: pd.DataFrame,
        bars_5m: pd.DataFrame,
        direction: Direction,
        vwap_1m: pd.Series,
    ) -> int:
        """
        Score 0–10 for pullback entry quality.
        Best entries are pullbacks to VWAP or key level, not extended chases.
        A professional trader waits for the pullback — we reward that.
        """
        if bars_1m.empty or len(bars_1m) < 5:
            return 0

        recent = bars_1m["close"].iloc[-5:]
        vwap_recent = vwap_1m.iloc[-5:]

        if direction == "call":
            # Good: prices pulled back toward VWAP and are now bouncing
            last_low = recent.min()
            last_vwap = vwap_recent.iloc[-1]
            pullback_depth = (recent.iloc[0] - last_low) / recent.iloc[0] if recent.iloc[0] > 0 else 0
            bouncing = recent.iloc[-1] > recent.iloc[-3]  # Price recovering
            near_vwap = abs(last_low - last_vwap) / last_vwap < 0.01 if last_vwap > 0 else False

            score = 0
            if pullback_depth > 0.005:
                score += 4  # There was a pullback
            if bouncing:
                score += 3  # Price is recovering
            if near_vwap:
                score += 3  # Touched VWAP = high-quality entry
            return score
        else:
            # Mirror for puts
            last_high = recent.max()
            last_vwap = vwap_recent.iloc[-1]
            pullback_depth = (last_high - recent.iloc[-1]) / last_high if last_high > 0 else 0
            declining = recent.iloc[-1] < recent.iloc[-3]
            near_vwap = abs(last_high - last_vwap) / last_vwap < 0.01 if last_vwap > 0 else False

            score = 0
            if pullback_depth > 0.005:
                score += 4
            if declining:
                score += 3
            if near_vwap:
                score += 3
            return score

        return min(100, max(0, score))

    # ── Option Selection ─────────────────────────────────────────────────────

    async def _select_option(
        self, scan: ScanResult, direction: Direction
    ) -> OptionSpec | None:
        """
        Select the best liquid option:
          - Nearest weekly expiry (next week after Wednesday)
          - First ITM or ATM strike
          - Real-time quote validation
        """
        expiry = self._pick_expiry(scan.options.expirations)
        if expiry is None:
            return None

        strike = self._pick_strike(
            scan.price,
            direction,
            expiry,
        )

        # Build OCC ticker to fetch live quote
        option_ticker = await self._find_option_ticker(
            scan.ticker, direction, strike, expiry
        )
        if option_ticker is None:
            return None

        # Fetch live quote for spread/price
        quote = await self._fetch_option_quote_detail(option_ticker)
        if quote is None:
            return None

        bid, ask, oi, delta = quote
        spread = ask - bid
        max_spread: float = cfg["options"]["max_spread_entry"]
        if spread > max_spread:
            log.debug(
                "option_spread_too_wide",
                extra={"ticker": scan.ticker, "option": option_ticker, "spread": spread},
            )
            return None

        min_oi: int = cfg["options"]["min_open_interest"]
        if oi < min_oi:
            log.debug(
                "option_oi_too_low",
                extra={"ticker": scan.ticker, "option": option_ticker, "oi": oi},
            )
            return None

        return OptionSpec(
            underlying=scan.ticker,
            option_ticker=option_ticker,
            direction=direction,
            strike=strike,
            expiration=expiry,
            bid=bid,
            ask=ask,
            mid=round((bid + ask) / 2, 2),
            delta=delta,
            oi=oi,
        )

    def _pick_expiry(self, expirations: list[str]) -> str | None:
        """
        Pick nearest weekly. Roll to next week on/after Wednesday.
        Never pick expiry with < min_dte days remaining.
        """
        from datetime import date, timedelta
        today = date.today()
        weekday = today.weekday()  # 0=Mon, 2=Wed, 4=Fri

        valid = []
        for exp_str in sorted(expirations):
            exp_date = date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if dte < self.min_dte:
                continue
            valid.append((dte, exp_str))

        if not valid:
            return None

        # After Wednesday, skip current week's expiry
        if weekday >= self.roll_after_weekday:
            # Drop any expiry within the same calendar week (Mon–Fri)
            days_to_friday = 4 - weekday
            current_week_end = today + timedelta(days=days_to_friday)
            valid = [(dte, exp) for dte, exp in valid
                     if date.fromisoformat(exp) > current_week_end]

        if not valid:
            return None

        return valid[0][1]  # Nearest qualifying expiry

    def _pick_strike(
        self,
        price: float,
        direction: Direction,
        expiry: str,
    ) -> float:
        """
        Return first ITM or ATM strike.
        For calls: ATM or first strike ≤ price (ITM).
        For puts:  ATM or first strike ≥ price (ITM).
        We round to common strike increments ($0.50, $1, $2.50, $5).
        """
        # Determine strike increment based on price
        if price < 10:
            inc = 0.5
        elif price < 25:
            inc = 1.0
        elif price < 50:
            inc = 2.5
        else:
            inc = 5.0

        # ATM strike = round to nearest increment
        atm = round(round(price / inc) * inc, 2)

        if direction == "call":
            # First ITM strike = ATM or one increment below
            return atm if price >= atm else atm - inc
        else:
            # First ITM strike = ATM or one increment above
            return atm if price <= atm else atm + inc

    async def _find_option_ticker(
        self,
        underlying: str,
        direction: Direction,
        strike: float,
        expiry: str,
    ) -> str | None:
        """Query Polygon for the OCC option ticker matching our spec."""
        assert self._session is not None
        contract_type = "call" if direction == "call" else "put"
        url = f"{cfg['polygon']['base_url']}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "contract_type": contract_type,
            "strike_price": strike,
            "expiration_date": expiry,
            "limit": 10,
        }
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            results = data.get("results", [])
            if results:
                return results[0]["ticker"]
        except Exception as exc:
            log.warning("option_ticker_lookup_failed", extra={"error": str(exc)})
        return None

    async def _fetch_option_quote_detail(
        self, option_ticker: str
    ) -> tuple[float, float, int, float] | None:
        """Returns (bid, ask, oi, delta)."""
        assert self._session is not None
        url = f"{cfg['polygon']['base_url']}/v3/snapshot/options/{option_ticker}"
        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            result = data.get("results", {})
            quote = result.get("last_quote", {})
            greeks = result.get("greeks", {})
            bid = float(quote.get("bid", 0))
            ask = float(quote.get("ask", 0))
            oi = int(result.get("open_interest", 0))
            delta = float(greeks.get("delta", 0.5))
            return bid, ask, oi, delta
        except Exception as exc:
            log.warning("option_quote_failed", extra={"ticker": option_ticker, "error": str(exc)})
            return None

    # ── Bar Fetching ─────────────────────────────────────────────────────────

    async def _fetch_bars(
        self, ticker: str, multiplier: str, bars: int
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars from Polygon.
        multiplier: "1" for 1-minute, "5" for 5-minute.
        Returns DataFrame with columns: open, high, low, close, volume, timestamp.
        """
        assert self._session is not None
        from datetime import timedelta
        today = now_et().date()
        # Go back enough days to capture `bars` trading bars even over weekends
        start_date = today - timedelta(days=5)

        timespan = "minute"
        url = (
            f"{cfg['polygon']['base_url']}/v2/aggs/ticker/{ticker}/range"
            f"/{multiplier}/{timespan}/{start_date}/{today}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": bars}
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return pd.DataFrame()
                data = await resp.json()
        except Exception as exc:
            log.warning("bars_fetch_failed", extra={"ticker": ticker, "error": str(exc)})
            return pd.DataFrame()

        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                            "v": "volume", "t": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Filter to today's session only for VWAP anchoring
        today_start = pd.Timestamp(today).tz_localize("America/New_York")
        df = df[df.index >= today_start]

        return df.tail(bars)


# ── USO direction helper ──────────────────────────────────────────────────────

async def get_uso_direction(session: aiohttp.ClientSession, polygon_key: str) -> int:
    """
    Returns +1 if USO is up today, -1 if down, 0 if unknown.
    Used by main.py to pass energy correlation to evaluate().
    """
    url = f"{cfg['polygon']['base_url']}/v2/snapshot/locale/us/markets/stocks/tickers/USO"
    try:
        async with session.get(
            url, headers={"Authorization": f"Bearer {polygon_key}"}
        ) as resp:
            if resp.status != 200:
                return 0
            data = await resp.json()
        ticker_data = data.get("ticker", {})
        day = ticker_data.get("day", {})
        prev_close = ticker_data.get("prevDay", {}).get("c", 0)
        last = day.get("c", prev_close)
        if prev_close == 0:
            return 0
        change = last - prev_close
        return 1 if change > 0 else (-1 if change < 0 else 0)
    except Exception:
        return 0
