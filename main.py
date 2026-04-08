"""
main.py — Orchestration loop
=============================
Ties every module together into the live trading / paper trading runtime.

Execution flow per cycle (every 5 minutes):
  1. Scanner runs → top-5 qualifying stocks
  2. For each stock:
     a. Check news catalyst (biotech) or USO direction (energy)
     b. Run strategy evaluation → TradeSignal or RejectedSignal
     c. Risk pre-trade check → SizingResult
     d. If approved: place order via broker, register position with risk manager
  3. Monitor all open positions for exits (runs every 1 second)
  4. Execute any exit orders (stop-loss, profit target, time-based)
  5. Update dashboard state

Backtest mode (BACKTEST=true):
  • Polygon historical data replaces live quotes
  • Tradier order execution is simulated (fills at mid-price)
  • Full P&L simulation with the same risk rules

CLI:
  python main.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--capital FLOAT]
"""

from __future__ import annotations

import asyncio
import os
import signal
import uuid
from datetime import date, datetime, timezone

import aiohttp
import click

from broker import TradierBroker
from dashboard import Dashboard, DashboardState, PositionRow, ScanRow
from flow import FlowScanner
from intelligence import IntelligenceEngine, TradeRecord
from market_regime import MarketRegimeEngine
from news import NewsClient
from risk import RiskManager
from scanner import ScanResult, Scanner
from strategy import RejectedSignal, StrategyEngine, TradeSignal, get_uso_direction
from utils import (
    env_bool,
    env_float,
    get_config,
    get_logger,
    is_within_trading_window,
    must_close_all,
    now_et,
)

log = get_logger(__name__)
cfg = get_config()

# ── Globals loaded from environment ──────────────────────────────────────────

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")
TRADIER_KEY = os.environ.get("TRADIER_API_KEY", "")
TRADIER_ACCOUNT = os.environ.get("TRADIER_ACCOUNT_ID", "")
BENZINGA_KEY = os.environ.get("BENZINGA_API_KEY", "")
IS_LIVE = env_bool("LIVE_TRADING", False)
IS_BACKTEST = env_bool("BACKTEST", False)


# ── Bot ───────────────────────────────────────────────────────────────────────

class TradingBot:

    def __init__(self, account_capital: float) -> None:
        self._capital = account_capital
        self._state = DashboardState(
            mode="LIVE" if IS_LIVE else "PAPER",
            account_capital=account_capital,
        )
        self._dashboard = Dashboard(self._state)
        self._risk = RiskManager()
        self._risk.account_capital = account_capital

        # Smart modules — loaded once, persist across scan cycles
        self._intelligence = IntelligenceEngine()

        # Running flag — set False by shutdown handler
        self._running = False

        # Track the Tradier order IDs → position IDs for reconciliation
        self._order_to_position: dict[str, str] = {}
        # Track entry data for intelligence recording on exit
        self._entry_records: dict[str, dict] = {}

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main event loop."""
        if not POLYGON_KEY:
            log.error("missing_polygon_key", extra={})
            raise SystemExit("POLYGON_API_KEY not set. Check your .env file.")
        if not TRADIER_KEY and not IS_BACKTEST:
            log.error("missing_tradier_key", extra={})
            raise SystemExit("TRADIER_API_KEY not set. Check your .env file.")

        self._running = True
        self._state.status = "ACTIVE"
        log.info("bot_started", extra={"mode": self._state.mode, "capital": self._capital})

        # Start dashboard in background task
        dash_task = asyncio.create_task(self._dashboard.run(), name="dashboard")

        try:
            async with (
                Scanner(POLYGON_KEY) as scanner,
                StrategyEngine(POLYGON_KEY) as strategy,
                NewsClient(BENZINGA_KEY) as news,
                TradierBroker(TRADIER_KEY, TRADIER_ACCOUNT) as broker,
                FlowScanner(POLYGON_KEY) as flow_scanner,
                MarketRegimeEngine(POLYGON_KEY) as regime_engine,
                aiohttp.ClientSession(
                    headers={"Authorization": f"Bearer {POLYGON_KEY}"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as _polygon_session,
            ):
                self._broker = broker
                self._polygon_session = _polygon_session
                self._news = news
                self._strategy = strategy
                self._flow_scanner = flow_scanner
                self._regime_engine = regime_engine

                self._risk.reset_daily()

                scan_interval = cfg["scanner"]["interval_seconds"]
                next_scan = asyncio.get_event_loop().time()

                while self._running:
                    loop_time = asyncio.get_event_loop().time()

                    # ── Force-close check ─────────────────────────────────────
                    if must_close_all():
                        await self._close_all_positions("force_close_eod")
                        self._state.status = "CLOSED_EOD"
                        self._running = False
                        break

                    # ── Position monitoring (every loop tick) ─────────────────
                    await self._monitor_positions()

                    # ── Scan cycle ────────────────────────────────────────────
                    if loop_time >= next_scan and is_within_trading_window():
                        await self._run_scan_cycle(scanner, strategy, news, broker)
                        next_scan = loop_time + scan_interval

                    # ── Update header stats ───────────────────────────────────
                    self._state.daily_pnl = self._risk.daily_pnl
                    self._state.daily_pnl_pct = (
                        self._risk.daily_pnl / self._capital * 100
                        if self._capital > 0 else 0.0
                    )
                    self._state.halted = self._risk.is_halted
                    if self._risk.is_halted:
                        self._state.status = "HALTED"
                        self._state.halt_reason = self._risk._halt_reason

                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            log.info("bot_cancelled", extra={})
        finally:
            dash_task.cancel()
            await asyncio.gather(dash_task, return_exceptions=True)
            log.info("bot_stopped", extra={"daily_pnl": round(self._risk.daily_pnl, 2)})

    # ── Scan cycle ────────────────────────────────────────────────────────────

    async def _run_scan_cycle(
        self,
        scanner: Scanner,
        strategy: StrategyEngine,
        news: NewsClient,
        broker: TradierBroker,
    ) -> None:
        scan_start = datetime.now(timezone.utc)
        self._state.last_scan_time = now_et().strftime("%H:%M:%S")

        try:
            scan_results = await scanner.run_scan()
        except Exception as exc:
            log.error("scan_cycle_error", extra={"error": str(exc)})
            return

        # Update dashboard scan panel
        self._state.scan_results = [
            ScanRow(
                ticker=r.ticker,
                price=r.price,
                change_pct=r.intraday_change_pct,
                volume_spike=r.volume_spike,
                sector=r.sector,
                has_catalyst=False,  # Updated after catalyst check below
            )
            for r in scan_results
        ]
        self._state.log_entry("SCAN", "—",
                               f"{len(scan_results)} stocks found")

        if not scan_results:
            return

        # Gather catalyst data for biotech stocks
        biotech_tickers = [r.ticker for r in scan_results if r.sector == "biotech"]
        catalyst_map: dict[str, bool] = {}
        if biotech_tickers:
            catalyst_results = await news.batch_check(biotech_tickers)
            catalyst_map = {t: r.has_catalyst for t, r in catalyst_results.items()}

        # Update scan panel with catalyst info
        for row in self._state.scan_results:
            if row.ticker in catalyst_map:
                row.has_catalyst = catalyst_map[row.ticker]

        # Get USO direction, market regime, and flow data concurrently
        uso_dir, market_context = await asyncio.gather(
            get_uso_direction(self._polygon_session, POLYGON_KEY),
            self._regime_engine.get_context(),
        )

        # Scan options flow for all candidates in parallel
        flow_inputs = [(r.ticker, r.price) for r in scan_results]
        flow_map = await self._flow_scanner.scan_batch(flow_inputs)

        # Log market regime on dashboard
        self._state.log_entry(
            "SCAN", "MARKET",
            f"regime={market_context.regime} spy={market_context.spy_change_pct:+.1f}% "
            f"vix={market_context.vix_level:.1f} bias={market_context.trade_bias}"
        )

        # Skip new entries if bot is halted
        if self._risk.is_halted:
            log.warning("scan_skipped_halted", extra={})
            return

        # Evaluate each scan result
        for scan in scan_results:
            if not self._running:
                break

            has_catalyst = catalyst_map.get(scan.ticker, False)
            uso_direction = uso_dir if scan.sector == "energy" else 0
            flow_signal = flow_map.get(scan.ticker)

            try:
                signal = await strategy.evaluate(
                    scan,
                    has_catalyst=has_catalyst,
                    uso_direction=uso_direction,
                    flow_signal=flow_signal,
                    market_context=market_context,
                    intelligence=self._intelligence,
                )
            except Exception as exc:
                log.error(
                    "strategy_eval_error",
                    extra={"ticker": scan.ticker, "error": str(exc)},
                )
                continue

            if isinstance(signal, RejectedSignal):
                log.info(
                    "signal_rejected",
                    extra={"ticker": signal.ticker, "reason": signal.reason},
                )
                self._state.log_entry(
                    "REJECT", signal.ticker,
                    f"dir={signal.direction or '?'} reason={signal.reason}"
                )
                continue

            # Valid signal — run risk check
            await self._attempt_entry(signal, broker)

    # ── Entry Execution ───────────────────────────────────────────────────────

    async def _attempt_entry(self, signal: TradeSignal, broker: TradierBroker) -> None:
        spec = signal.option_spec
        sizing = self._risk.pre_trade_check(
            sector=signal.scan_result.sector,
            confidence=signal.confidence,
            option_mid=spec.mid,
            option_spread=spec.ask - spec.bid,
        )

        if not sizing.allowed:
            log.info(
                "trade_blocked_by_risk",
                extra={
                    "ticker": signal.scan_result.ticker,
                    "reason": sizing.rejection_reason,
                },
            )
            self._state.log_entry(
                "REJECT", signal.scan_result.ticker,
                f"risk: {sizing.rejection_reason}"
            )
            return

        log.info(
            "entering_trade",
            extra={
                "ticker": signal.scan_result.ticker,
                "direction": signal.direction,
                "option": spec.option_ticker,
                "qty": sizing.quantity,
                "mid": spec.mid,
                "confidence": signal.confidence,
                "size_pct": sizing.size_pct,
            },
        )

        # Execute order
        if IS_BACKTEST:
            order_result = _simulate_fill(spec.option_ticker, sizing.quantity, spec.mid)
        else:
            order_result = await broker.buy_to_open(
                option_ticker=spec.option_ticker,
                quantity=sizing.quantity,
                bid=spec.bid,
                ask=spec.ask,
            )

        if order_result.status not in ("filled", "partially_filled"):
            log.warning(
                "entry_order_not_filled",
                extra={
                    "ticker": signal.scan_result.ticker,
                    "status": order_result.status,
                    "error": order_result.error,
                },
            )
            self._state.log_entry(
                "REJECT", signal.scan_result.ticker,
                f"fill_failed: {order_result.status}"
            )
            return

        # Register position with risk manager
        position_id = str(uuid.uuid4())
        fill_price = order_result.avg_fill_price or spec.mid
        position = RiskManager.build_position(
            position_id=position_id,
            ticker=signal.scan_result.ticker,
            option_ticker=spec.option_ticker,
            direction=signal.direction,
            sector=signal.scan_result.sector,
            quantity=order_result.filled_qty,
            entry_premium=fill_price,
            entry_stock_price=signal.scan_result.price,
            expiration=spec.expiration,
        )
        self._risk.add_position(position)
        self._order_to_position[order_result.order_id] = position_id
        self._state.trades_today += 1

        # Store entry data for intelligence recording when trade closes
        self._entry_records[position_id] = {
            "ticker": signal.scan_result.ticker,
            "sector": signal.scan_result.sector,
            "direction": signal.direction,
            "entry_time": now_et().isoformat(),
            "entry_hour_et": now_et().hour,
            "confidence": signal.confidence,
            "rsi": signal.rsi_5m,
            "atr_expansion": signal.atr_expansion,
            "vwap_break_pct": signal.vwap_break_pct,
            "volume_spike": signal.volume_spike,
            "had_catalyst": has_catalyst if hasattr(signal, '_has_catalyst') else False,
            "had_flow": bool(flow_signal and flow_signal.flow_score > 20) if 'flow_signal' in dir() else False,
            "market_regime": market_context.regime if 'market_context' in dir() and market_context else "UNKNOWN",
            "vix": market_context.vix_level if 'market_context' in dir() and market_context else 0.0,
            "entry_premium": fill_price,
        }

        # Add to dashboard positions
        self._state.positions.append(
            PositionRow(
                ticker=signal.scan_result.ticker,
                direction=signal.direction.upper(),
                strike=spec.strike,
                expiry=spec.expiration,
                qty=order_result.filled_qty,
                entry_premium=fill_price,
                current_premium=fill_price,
                pnl_pct=0.0,
                entry_time=now_et().strftime("%H:%M"),
            )
        )

        self._state.log_entry(
            "ENTER",
            signal.scan_result.ticker,
            (f"{signal.direction.upper()} {spec.strike} x{order_result.filled_qty} "
             f"@ ${fill_price:.2f} conf={signal.confidence}"),
        )

    # ── Position Monitoring ───────────────────────────────────────────────────

    async def _monitor_positions(self) -> None:
        """Update live P&L for each open position and check exit triggers."""
        open_positions = self._risk.open_positions
        if not open_positions:
            return

        # Fetch current option quotes
        update_tasks = [
            self._update_position_quote(pos.position_id, pos.option_ticker)
            for pos in open_positions
        ]
        await asyncio.gather(*update_tasks, return_exceptions=True)

        # Evaluate exit signals
        actions = self._risk.check_exits()

        for action in actions:
            if action.action == "exit":
                await self._execute_exit(action.position_id, action.reason or "unknown",
                                         force_market=(action.urgency == "market"))

        # Sync dashboard positions list
        self._sync_dashboard_positions()

    async def _update_position_quote(
        self, position_id: str, option_ticker: str
    ) -> None:
        try:
            if IS_BACKTEST:
                return  # Backtest mode updates premiums differently
            quote = await self._broker.get_option_quote(option_ticker)
            if quote:
                self._risk.update_position_premium(position_id, quote.mid)
        except Exception as exc:
            log.debug(
                "quote_update_failed",
                extra={"position_id": position_id, "error": str(exc)},
            )

    async def _execute_exit(
        self,
        position_id: str,
        reason: str,
        force_market: bool = True,
    ) -> None:
        """Sell to close a position and record P&L."""
        # Find the position
        pos = next((p for p in self._risk.open_positions if p.position_id == position_id), None)
        if pos is None:
            return

        log.info(
            "exiting_position",
            extra={
                "position_id": position_id,
                "ticker": pos.ticker,
                "reason": reason,
                "pnl_pct": round(pos.unrealized_pnl_pct, 1),
            },
        )

        if IS_BACKTEST:
            exit_price = pos.last_premium
            order_result = _simulate_fill(pos.option_ticker, pos.quantity, exit_price)
        else:
            # Get fresh quote for exit price
            quote = await self._broker.get_option_quote(pos.option_ticker)
            bid = quote.bid if quote else pos.last_premium * 0.98
            ask = quote.ask if quote else pos.last_premium * 1.02
            order_result = await self._broker.sell_to_close(
                option_ticker=pos.option_ticker,
                quantity=pos.quantity,
                bid=bid,
                ask=ask,
                force_market=force_market,
            )

        exit_price = order_result.avg_fill_price or pos.last_premium
        realized_pnl = (exit_price - pos.entry_premium) * 100 * pos.quantity
        pnl_pct = ((exit_price - pos.entry_premium) / pos.entry_premium) * 100 if pos.entry_premium else 0.0

        self._risk.remove_position(position_id, realized_pnl)

        # Remove from dashboard
        self._state.positions = [
            p for p in self._state.positions
            if p.ticker != pos.ticker
        ]

        self._state.log_entry(
            "EXIT",
            pos.ticker,
            f"{reason} @ ${exit_price:.2f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%)",
            pnl=realized_pnl,
        )

        # ── Teach the intelligence engine what happened ───────────────────────
        entry_data = self._entry_records.pop(position_id, None)
        if entry_data:
            import uuid as _uuid
            record = TradeRecord(
                trade_id=str(_uuid.uuid4()),
                ticker=entry_data["ticker"],
                sector=entry_data["sector"],
                direction=entry_data["direction"],
                entry_time=entry_data["entry_time"],
                exit_time=now_et().isoformat(),
                entry_hour_et=entry_data["entry_hour_et"],
                confidence_at_entry=entry_data["confidence"],
                rsi_at_entry=entry_data["rsi"],
                atr_expansion_at_entry=entry_data["atr_expansion"],
                vwap_break_pct=entry_data["vwap_break_pct"],
                volume_spike=entry_data["volume_spike"],
                had_catalyst=entry_data["had_catalyst"],
                had_flow_signal=entry_data["had_flow"],
                market_regime=entry_data["market_regime"],
                vix_at_entry=entry_data["vix"],
                exit_reason=reason,
                pnl_pct=pnl_pct,
                winner=pnl_pct > 0,
                big_winner=pnl_pct >= 50,
            )
            self._intelligence.record_trade(record)

    async def _close_all_positions(self, reason: str) -> None:
        """Emergency / EOD: close all open positions at market."""
        log.warning("closing_all_positions", extra={"reason": reason, "count": self._risk.open_count})
        for pos in self._risk.open_positions:
            await self._execute_exit(pos.position_id, reason, force_market=True)
        self._state.log_entry("HALT", "ALL", f"All positions closed: {reason}")

    def _sync_dashboard_positions(self) -> None:
        """Keep dashboard position list in sync with risk manager."""
        open_ids = {p.position_id for p in self._risk.open_positions}
        # Re-sync from risk manager's live data
        updated: list[PositionRow] = []
        for pos in self._risk.open_positions:
            # Find existing dashboard row or create new one
            existing = next(
                (r for r in self._state.positions if r.ticker == pos.ticker),
                None,
            )
            if existing:
                existing.current_premium = pos.last_premium
                existing.pnl_pct = pos.unrealized_pnl_pct
                updated.append(existing)
        self._state.positions = updated


# ── Backtest helpers ─────────────────────────────────────────────────────────

def _simulate_fill(option_ticker: str, quantity: int, price: float):
    """Simulate an immediate fill at the given price for backtest mode."""
    from broker import OrderResult
    return OrderResult(
        order_id=str(uuid.uuid4()),
        status="filled",
        filled_qty=quantity,
        avg_fill_price=price,
        attempts=1,
        elapsed_ms=0.0,
    )


# ── Shutdown handler ──────────────────────────────────────────────────────────

def _install_signal_handlers(bot: TradingBot) -> None:
    def _shutdown(signum, frame):  # type: ignore
        log.warning("shutdown_signal", extra={"signal": signum})
        bot._running = False
        bot._state.status = "STOPPING"

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--capital", default=None, type=float,
              help="Override ACCOUNT_CAPITAL env var.")
@click.option("--start", default=None,
              help="Backtest start date YYYY-MM-DD (requires BACKTEST=true)")
@click.option("--end", default=None,
              help="Backtest end date YYYY-MM-DD (requires BACKTEST=true)")
def main(capital: float | None, start: str | None, end: str | None) -> None:
    """
    ⚡ Options Day Trading Bot

    \b
    Paper trading (default):
        python main.py

    \b
    Live trading:
        LIVE_TRADING=true python main.py

    \b
    Backtesting:
        BACKTEST=true python main.py --start 2024-01-01 --end 2024-12-31
    """
    from dotenv import load_dotenv
    load_dotenv()

    if IS_BACKTEST:
        if not start:
            start = cfg["backtest"]["default_start"]
        if not end:
            end = cfg["backtest"]["default_end"]
        log.info("backtest_mode", extra={"start": start, "end": end})
        # In backtest mode, override capital from config default
        effective_capital = capital or cfg["backtest"]["initial_capital"]
    else:
        effective_capital = capital or env_float("ACCOUNT_CAPITAL", 25_000.0)

    if IS_LIVE and not IS_BACKTEST:
        click.echo("\n[WARNING] LIVE TRADING MODE — REAL MONEY AT RISK\n", err=True)
        if not click.confirm("Type 'yes' to confirm you want to trade with real money"):
            raise SystemExit("Aborted.")

    bot = TradingBot(account_capital=effective_capital)
    _install_signal_handlers(bot)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
