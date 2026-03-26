"""
dashboard.py — Live terminal dashboard (Rich library)
======================================================
Renders a real-time P&L monitor, open positions table, scan results,
and scrolling trade log. Runs in its own asyncio task and refreshes
at the interval defined in config.yaml (default 1 second).

Layout (terminal):
┌──────────────────── OPTIONS DAY TRADING BOT ──────────────────────┐
│  Mode: PAPER │ Account: $25,000 │ Daily P&L: +$342 (+1.37%)       │
│  Status: ACTIVE │ Open Positions: 2 │ Trades Today: 5             │
├──────────────────── OPEN POSITIONS ───────────────────────────────┤
│  MRNA  |  CALL  |  130C  |  2026-04-04  |  Qty:2  |  +45.3%  |  │
│  ...                                                               │
├──────────────────── LATEST SCAN RESULTS ──────────────────────────┤
│  CRSP  $24.50  +5.2%  spike=3.8x  biotech  catalyst:YES          │
│  ...                                                               │
├──────────────────── TRADE LOG ─────────────────────────────────────┤
│  09:47 ENTER MRNA CALL 130C x2 @ $2.45  conf=82                   │
│  ...                                                               │
└───────────────────────────────────────────────────────────────────┘

Design decisions:
  • Dashboard state is a plain dataclass (DashboardState) that main.py writes
    to; the dashboard only reads it. No locks needed because Python's GIL
    protects simple attribute assignments and list replacements.
  • We use Rich's Live + Layout for smooth, flicker-free updates.
  • The dashboard does NOT perform any I/O or trading logic — it is a pure
    display layer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils import env_bool, get_config, get_logger, now_et

log = get_logger(__name__)
cfg = get_config()


# ── State (written by main.py, read by dashboard) ────────────────────────────

@dataclass
class PositionRow:
    ticker: str
    direction: str
    strike: float
    expiry: str
    qty: int
    entry_premium: float
    current_premium: float
    pnl_pct: float
    entry_time: str


@dataclass
class ScanRow:
    ticker: str
    price: float
    change_pct: float
    volume_spike: float
    sector: str
    has_catalyst: bool


@dataclass
class TradeLogEntry:
    timestamp: str
    action: Literal["ENTER", "EXIT", "REJECT", "SCAN", "HALT"]
    ticker: str
    detail: str
    pnl: float | None = None


@dataclass
class DashboardState:
    """Mutable state that main.py populates each cycle."""
    mode: str = "PAPER"
    account_capital: float = 25_000.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    status: str = "STARTING"
    trades_today: int = 0
    positions: list[PositionRow] = field(default_factory=list)
    scan_results: list[ScanRow] = field(default_factory=list)
    trade_log: list[TradeLogEntry] = field(default_factory=list)
    last_scan_time: str = "—"
    halted: bool = False
    halt_reason: str = ""

    def add_log(self, entry: TradeLogEntry) -> None:
        max_rows: int = cfg["dashboard"]["max_trade_log_rows"]
        self.trade_log.insert(0, entry)
        if len(self.trade_log) > max_rows:
            self.trade_log.pop()

    def log_entry(
        self,
        action: Literal["ENTER", "EXIT", "REJECT", "SCAN", "HALT"],
        ticker: str,
        detail: str,
        pnl: float | None = None,
    ) -> None:
        ts = now_et().strftime("%H:%M:%S")
        self.add_log(TradeLogEntry(timestamp=ts, action=action, ticker=ticker,
                                   detail=detail, pnl=pnl))


# ── Renderer ──────────────────────────────────────────────────────────────────

class Dashboard:

    def __init__(self, state: DashboardState) -> None:
        self.state = state
        self._console = Console()
        self._refresh_rate: float = cfg["dashboard"]["refresh_rate"]
        self._running = False

    async def run(self) -> None:
        """Start the dashboard live display. Runs until cancelled."""
        self._running = True
        with Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=int(1 / self._refresh_rate),
            screen=True,
        ) as live:
            while self._running:
                live.update(self._build_layout())
                await asyncio.sleep(self._refresh_rate)

    def stop(self) -> None:
        self._running = False

    # ── Layout Builder ────────────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="positions", ratio=2),
            Layout(name="scan"),
        )
        layout["right"].name = "tradelog"

        layout["header"].update(self._header_panel())
        layout["positions"].update(self._positions_panel())
        layout["scan"].update(self._scan_panel())
        layout["tradelog"].update(self._tradelog_panel())
        layout["footer"].update(self._footer_panel())
        return layout

    # ── Panels ────────────────────────────────────────────────────────────────

    def _header_panel(self) -> Panel:
        s = self.state
        mode_color = "red" if s.mode == "LIVE" else "yellow"
        status_color = "red" if s.halted else ("green" if s.status == "ACTIVE" else "yellow")

        pnl_color = "green" if s.daily_pnl >= 0 else "red"
        pnl_sign = "+" if s.daily_pnl >= 0 else ""
        pnl_pct_sign = "+" if s.daily_pnl_pct >= 0 else ""

        parts = [
            Text.assemble(
                "  MODE: ", (s.mode, mode_color + " bold"),
                "   │   ACCOUNT: ", f"${s.account_capital:,.2f}",
                "   │   DAILY P&L: ",
                (f"{pnl_sign}${abs(s.daily_pnl):,.2f} ({pnl_pct_sign}{s.daily_pnl_pct:.2f}%)",
                 pnl_color + " bold"),
                "   │   TRADES TODAY: ", str(s.trades_today),
                "   │   STATUS: ", (s.status, status_color + " bold"),
            )
        ]
        if s.halted:
            parts.append(Text.assemble(
                "\n  [!] HALT: ", (s.halt_reason, "red bold")
            ))

        return Panel(
            Columns(parts),
            title="[bold cyan]⚡ OPTIONS DAY TRADING BOT[/bold cyan]",
            border_style="cyan",
        )

    def _positions_panel(self) -> Panel:
        table = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            show_footer=False,
            pad_edge=False,
        )
        table.add_column("Ticker", style="bold", width=6)
        table.add_column("Dir", width=4)
        table.add_column("Strike", justify="right", width=7)
        table.add_column("Exp", width=10)
        table.add_column("Qty", justify="right", width=4)
        table.add_column("Entry $", justify="right", width=7)
        table.add_column("Curr $", justify="right", width=7)
        table.add_column("P&L %", justify="right", width=8)
        table.add_column("Open", width=8)

        for pos in self.state.positions:
            pnl_color = "green" if pos.pnl_pct >= 0 else "red"
            dir_color = "green" if pos.direction == "CALL" else "red"
            table.add_row(
                pos.ticker,
                Text(pos.direction, style=dir_color),
                f"${pos.strike:.2f}",
                pos.expiry,
                str(pos.qty),
                f"${pos.entry_premium:.2f}",
                f"${pos.current_premium:.2f}",
                Text(f"{'+' if pos.pnl_pct >= 0 else ''}{pos.pnl_pct:.1f}%", style=pnl_color + " bold"),
                pos.entry_time,
            )

        if not self.state.positions:
            table.add_row("—", "—", "—", "—", "—", "—", "—", "—", "—")

        return Panel(table, title="[bold]OPEN POSITIONS[/bold]", border_style="blue")

    def _scan_panel(self) -> Panel:
        table = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
        table.add_column("Ticker", style="bold", width=6)
        table.add_column("Price", justify="right", width=7)
        table.add_column("Δ%", justify="right", width=7)
        table.add_column("Spike", justify="right", width=6)
        table.add_column("Sector", width=8)
        table.add_column("Cat?", justify="center", width=5)

        for row in self.state.scan_results[:8]:
            chg_color = "green" if row.change_pct >= 0 else "red"
            cat_text = Text("✓", style="green bold") if row.has_catalyst else Text("✗", style="dim")
            table.add_row(
                row.ticker,
                f"${row.price:.2f}",
                Text(f"{'+' if row.change_pct >= 0 else ''}{row.change_pct:.1f}%", style=chg_color),
                f"{row.volume_spike:.1f}x",
                row.sector[:8],
                cat_text,
            )

        if not self.state.scan_results:
            table.add_row("—", "—", "—", "—", "—", "—")

        last_scan = self.state.last_scan_time
        return Panel(
            table,
            title=f"[bold]SCAN RESULTS[/bold] [dim](last: {last_scan})[/dim]",
            border_style="magenta",
        )

    def _tradelog_panel(self) -> Panel:
        table = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False, show_header=True)
        table.add_column("Time", width=8)
        table.add_column("Act", width=6)
        table.add_column("Ticker", width=6)
        table.add_column("Detail", ratio=1)
        table.add_column("P&L", justify="right", width=9)

        ACTION_COLORS = {
            "ENTER": "green",
            "EXIT": "yellow",
            "REJECT": "dim",
            "SCAN": "blue",
            "HALT": "red bold",
        }

        for entry in self.state.trade_log[:cfg["dashboard"]["max_trade_log_rows"]]:
            color = ACTION_COLORS.get(entry.action, "white")
            pnl_str = ""
            if entry.pnl is not None:
                sign = "+" if entry.pnl >= 0 else ""
                pnl_color = "green" if entry.pnl >= 0 else "red"
                pnl_str = f"[{pnl_color}]{sign}${abs(entry.pnl):.2f}[/{pnl_color}]"

            table.add_row(
                entry.timestamp,
                Text(entry.action, style=color),
                entry.ticker,
                entry.detail[:40],
                Text.from_markup(pnl_str) if pnl_str else Text(""),
            )

        return Panel(table, title="[bold]TRADE LOG[/bold]", border_style="green")

    def _footer_panel(self) -> Panel:
        now_str = now_et().strftime("%Y-%m-%d %H:%M:%S ET")
        live_str = "[red bold]LIVE[/red bold]" if env_bool("LIVE_TRADING") else "[yellow]PAPER[/yellow]"
        return Panel(
            Text.from_markup(
                f"  {now_str}   │   {live_str}   │   "
                "[dim]Ctrl+C to stop   │   All times ET[/dim]"
            ),
            border_style="dim",
        )
