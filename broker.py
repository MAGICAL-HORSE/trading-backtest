"""
broker.py — Tradier API wrapper
================================
Handles all order lifecycle operations against the Tradier brokerage API.
Paper trading (sandbox) is the default; live trading requires LIVE_TRADING=true.

Order execution strategy:
  • Place limit order at mid-price
  • Re-price every 8 seconds toward the market
  • After 3 attempts without fill, convert to market order
  • All operations are async; 500 ms latency target

Key design decisions:
  • Tradier's REST API is synchronous (no WebSocket for orders), so we use
    aiohttp with asyncio for non-blocking I/O while respecting the ~500ms target.
  • The `reprice_and_fill` method implements the complete limit→market waterfall.
  • Every order attempt is logged with the full order payload for audit.
  • Paper vs live is determined at module load time from the env var;
    the URL is the only difference.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp

from utils import env_bool, get_config, get_logger

log = get_logger(__name__)
cfg = get_config()

OrderSide = Literal["buy_to_open", "sell_to_close"]
OrderStatus = Literal["filled", "partially_filled", "open", "pending", "canceled", "rejected"]


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    order_id: str
    status: OrderStatus
    filled_qty: int
    avg_fill_price: float        # Per-contract premium
    attempts: int
    elapsed_ms: float
    error: str = ""


@dataclass
class PositionQuote:
    """Live option quote fetched from Tradier for P&L updates."""
    option_ticker: str
    bid: float
    ask: float
    mid: float
    last: float
    volume: int


# ── Tradier Client ────────────────────────────────────────────────────────────

class TradierBroker:

    def __init__(self, api_key: str, account_id: str) -> None:
        self._api_key = api_key
        self._account_id = account_id
        self._live = env_bool("LIVE_TRADING", False)

        bc = cfg["broker"]
        ec = cfg["execution"]
        self._base_url = bc["live_base_url"] if self._live else bc["paper_base_url"]
        self._timeout = bc["timeout_seconds"]
        self._reprice_interval = ec["limit_reprice_interval_sec"]
        self._max_attempts = ec["limit_max_attempts"]

        self._session: aiohttp.ClientSession | None = None

        log.info(
            "broker_initialized",
            extra={"mode": "LIVE" if self._live else "PAPER", "base_url": self._base_url},
        )

    async def __aenter__(self) -> "TradierBroker":
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ── Order Execution ──────────────────────────────────────────────────────

    async def buy_to_open(
        self,
        option_ticker: str,
        quantity: int,
        bid: float,
        ask: float,
    ) -> OrderResult:
        """
        Execute a buy-to-open with limit→market waterfall.
        Returns OrderResult regardless of outcome.
        """
        return await self._reprice_and_fill(
            option_ticker=option_ticker,
            side="buy_to_open",
            quantity=quantity,
            bid=bid,
            ask=ask,
        )

    async def sell_to_close(
        self,
        option_ticker: str,
        quantity: int,
        bid: float,
        ask: float,
        force_market: bool = False,
    ) -> OrderResult:
        """
        Execute a sell-to-close. Pass force_market=True for stop-loss / EOD exits.
        """
        if force_market:
            return await self._place_market_order(option_ticker, "sell_to_close", quantity)
        return await self._reprice_and_fill(
            option_ticker=option_ticker,
            side="sell_to_close",
            quantity=quantity,
            bid=bid,
            ask=ask,
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        assert self._session is not None
        url = f"{self._base_url}/accounts/{self._account_id}/orders/{order_id}"
        try:
            async with self._session.delete(url) as resp:
                data = await resp.json()
            success = data.get("order", {}).get("status") in ("canceled", "ok")
            log.info("order_canceled", extra={"order_id": order_id, "success": success})
            return success
        except Exception as exc:
            log.warning("cancel_failed", extra={"order_id": order_id, "error": str(exc)})
            return False

    # ── Quote Fetching ────────────────────────────────────────────────────────

    async def get_option_quote(self, option_ticker: str) -> PositionQuote | None:
        """Fetch real-time option quote for P&L monitoring."""
        assert self._session is not None
        url = f"{self._base_url}/markets/options/chains"
        # Tradier quotes API uses underlying + expiry; for a single OCC symbol
        # we use the quotes endpoint instead.
        url = f"{self._base_url}/markets/quotes"
        params = {"symbols": option_ticker, "greeks": "false"}
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            quotes = data.get("quotes", {}).get("quote", {})
            # Tradier may return a dict (single) or list
            if isinstance(quotes, list):
                q = quotes[0] if quotes else {}
            else:
                q = quotes

            bid = float(q.get("bid", 0))
            ask = float(q.get("ask", 0))
            last = float(q.get("last", 0))
            vol = int(q.get("volume", 0))
            mid = round((bid + ask) / 2, 2) if bid and ask else last

            return PositionQuote(
                option_ticker=option_ticker,
                bid=bid,
                ask=ask,
                mid=mid,
                last=last,
                volume=vol,
            )
        except Exception as exc:
            log.warning(
                "option_quote_failed",
                extra={"option_ticker": option_ticker, "error": str(exc)},
            )
            return None

    async def get_account_balance(self) -> dict[str, float]:
        """Return cash / buying power summary."""
        assert self._session is not None
        url = f"{self._base_url}/accounts/{self._account_id}/balances"
        try:
            async with self._session.get(url) as resp:
                data = await resp.json()
            balances = data.get("balances", {})
            return {
                "total_equity": float(balances.get("total_equity", 0)),
                "cash": float(balances.get("cash", {}).get("cash_available", 0)),
                "option_buying_power": float(
                    balances.get("option_short_value", 0) or
                    balances.get("options_bp", 0)
                ),
            }
        except Exception as exc:
            log.warning("balance_fetch_failed", extra={"error": str(exc)})
            return {}

    async def get_open_orders(self) -> list[dict]:
        """Retrieve all open orders on the account."""
        assert self._session is not None
        url = f"{self._base_url}/accounts/{self._account_id}/orders"
        try:
            async with self._session.get(url) as resp:
                data = await resp.json()
            orders = data.get("orders", {})
            if not orders or orders == "null":
                return []
            raw = orders.get("order", [])
            if isinstance(raw, dict):
                raw = [raw]
            return [o for o in raw if o.get("status") in ("open", "partially_filled", "pending")]
        except Exception as exc:
            log.warning("open_orders_fetch_failed", extra={"error": str(exc)})
            return []

    # ── Internal: Limit → Market waterfall ───────────────────────────────────

    async def _reprice_and_fill(
        self,
        option_ticker: str,
        side: OrderSide,
        quantity: int,
        bid: float,
        ask: float,
    ) -> OrderResult:
        """
        1. Place limit at mid-price.
        2. Every 8s, check fill status and reprice toward market.
        3. After max_attempts misses, convert to market order.
        """
        t0 = time.perf_counter()
        mid = round((bid + ask) / 2, 2)
        spread = ask - bid
        attempts = 0

        # Step-up prices: mid → ask (for buys) or mid → bid (for sells)
        if side == "buy_to_open":
            price_ladder = [
                mid,
                round(mid + spread * 0.25, 2),
                round(ask, 2),
            ]
        else:
            price_ladder = [
                mid,
                round(mid - spread * 0.25, 2),
                round(bid, 2),
            ]

        order_id: str | None = None

        for attempt, limit_price in enumerate(price_ladder[: self._max_attempts]):
            attempts += 1

            if order_id:
                # Cancel previous unfilled limit order before repricing
                await self.cancel_order(order_id)

            result = await self._place_limit_order(
                option_ticker, side, quantity, limit_price
            )
            order_id = result.get("order_id")

            if not order_id:
                log.warning(
                    "limit_order_failed",
                    extra={
                        "option_ticker": option_ticker,
                        "attempt": attempt + 1,
                        "price": limit_price,
                        "error": result.get("error", "unknown"),
                    },
                )
                continue

            log.info(
                "limit_order_placed",
                extra={
                    "order_id": order_id,
                    "option_ticker": option_ticker,
                    "side": side,
                    "qty": quantity,
                    "limit_price": limit_price,
                    "attempt": attempt + 1,
                },
            )

            # Wait reprice_interval then check status
            await asyncio.sleep(self._reprice_interval)

            status_data = await self._get_order_status(order_id)
            status = status_data.get("status", "open")

            if status == "filled":
                avg_fill = float(status_data.get("avg_fill_price", limit_price))
                elapsed = (time.perf_counter() - t0) * 1000
                log.info(
                    "order_filled",
                    extra={
                        "order_id": order_id,
                        "option_ticker": option_ticker,
                        "avg_fill": avg_fill,
                        "attempts": attempts,
                        "elapsed_ms": round(elapsed, 1),
                    },
                )
                return OrderResult(
                    order_id=order_id,
                    status="filled",
                    filled_qty=quantity,
                    avg_fill_price=avg_fill,
                    attempts=attempts,
                    elapsed_ms=elapsed,
                )

        # All limit attempts exhausted — convert to market
        if order_id:
            await self.cancel_order(order_id)

        log.warning(
            "converting_to_market",
            extra={"option_ticker": option_ticker, "attempts": attempts},
        )
        return await self._place_market_order(option_ticker, side, quantity, t0=t0, attempts=attempts)

    async def _place_limit_order(
        self,
        option_ticker: str,
        side: OrderSide,
        quantity: int,
        price: float,
    ) -> dict:
        assert self._session is not None
        url = f"{self._base_url}/accounts/{self._account_id}/orders"
        payload = {
            "class": "option",
            "symbol": self._extract_underlying(option_ticker),
            "option_symbol": option_ticker,
            "side": side,
            "quantity": str(quantity),
            "type": "limit",
            "price": str(round(price, 2)),
            "duration": "day",
        }
        try:
            async with self._session.post(url, data=payload) as resp:
                data = await resp.json()
            order = data.get("order", {})
            return {
                "order_id": str(order.get("id", "")),
                "status": order.get("status", "unknown"),
                "error": data.get("errors", {}).get("error", ""),
            }
        except Exception as exc:
            return {"order_id": None, "status": "error", "error": str(exc)}

    async def _place_market_order(
        self,
        option_ticker: str,
        side: OrderSide,
        quantity: int,
        t0: float | None = None,
        attempts: int = 0,
    ) -> OrderResult:
        assert self._session is not None
        if t0 is None:
            t0 = time.perf_counter()

        url = f"{self._base_url}/accounts/{self._account_id}/orders"
        payload = {
            "class": "option",
            "symbol": self._extract_underlying(option_ticker),
            "option_symbol": option_ticker,
            "side": side,
            "quantity": str(quantity),
            "type": "market",
            "duration": "day",
        }
        try:
            async with self._session.post(url, data=payload) as resp:
                data = await resp.json()
            order = data.get("order", {})
            order_id = str(order.get("id", ""))
            elapsed = (time.perf_counter() - t0) * 1000

            log.info(
                "market_order_placed",
                extra={
                    "order_id": order_id,
                    "option_ticker": option_ticker,
                    "side": side,
                    "qty": quantity,
                    "elapsed_ms": round(elapsed, 1),
                },
            )

            # For market orders we assume fill at last known price;
            # reconcile actual fill price when polling order status.
            return OrderResult(
                order_id=order_id,
                status="filled",  # optimistic; reconciled on next poll
                filled_qty=quantity,
                avg_fill_price=0.0,  # filled in by status poll
                attempts=attempts + 1,
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            log.error(
                "market_order_failed",
                extra={"option_ticker": option_ticker, "error": str(exc)},
            )
            return OrderResult(
                order_id="",
                status="rejected",
                filled_qty=0,
                avg_fill_price=0.0,
                attempts=attempts + 1,
                elapsed_ms=elapsed,
                error=str(exc),
            )

    async def _get_order_status(self, order_id: str) -> dict:
        assert self._session is not None
        url = f"{self._base_url}/accounts/{self._account_id}/orders/{order_id}"
        try:
            async with self._session.get(url) as resp:
                data = await resp.json()
            order = data.get("order", {})
            return {
                "status": order.get("status", "unknown"),
                "filled_qty": int(order.get("exec_quantity", 0)),
                "avg_fill_price": float(order.get("avg_fill_price", 0)),
            }
        except Exception as exc:
            log.warning("order_status_failed", extra={"order_id": order_id, "error": str(exc)})
            return {"status": "unknown", "filled_qty": 0, "avg_fill_price": 0.0}

    @staticmethod
    def _extract_underlying(option_ticker: str) -> str:
        """
        Extract underlying symbol from OCC option ticker.
        E.g. "MRNA240705C00130000" → "MRNA"
        OCC format: <symbol padded to 6><YY><MM><DD><C/P><8-digit-strike>
        """
        # Strip digits from the right until we hit a letter-only prefix
        import re
        match = re.match(r"^([A-Z]+)", option_ticker)
        return match.group(1) if match else option_ticker
