"""
Vertical Spread 0DTE Backtest
═══════════════════════════════════════════════════════════════
Underlyings  : SPY, QQQ, NVDA (synthetic GBM data)
Strategy     : Bull call spread  →  9 EMA crosses ABOVE 20 EMA
               Bear put spread   →  9 EMA crosses BELOW  20 EMA
Spread width : $1 wide
Max risk     : $30 per trade  (10% of $300 account)
PDT rule     : Max 3 completed day-trades per week (account-wide)
Entry        : After 9:45 AM only
Exit         : TP at 80% of max gain | SL at 50% of debit paid | 3:30 PM hard stop
"""

import warnings
warnings.filterwarnings("ignore")

import math
import sys
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────────

TICKER_PARAMS = {
    # S0    : synthetic starting price
    # vol   : annualised implied vol used for BOTH GBM simulation AND B-S pricing
    # drift : annualised drift for GBM
    # seed  : per-ticker RNG seed for reproducibility
    "SPY":  {"S0": 572.0,  "vol": 0.14, "drift": 0.10, "seed": 42},
    "QQQ":  {"S0": 488.0,  "vol": 0.18, "drift": 0.12, "seed": 77},
    "NVDA": {"S0": 124.0,  "vol": 0.58, "drift": 0.15, "seed": 13},
}

PERIOD_DAYS    = 180
INITIAL_CAP    = 300.0
MAX_TRADE_COST = 30.0        # max net debit per spread position (dollars)
SPREAD_WIDTH   = 1.0         # $1 wide spread
TP_RATIO       = 0.80        # take profit: 80% of max gain
SL_RATIO       = 0.50        # stop loss : drop to 50% of entry debit
ENTRY_AFTER    = time(9, 45)
EXIT_BY        = time(15, 30)
MAX_DT_WEEK    = 3           # PDT: max day-trades per week across ALL tickers
RISK_FREE      = 0.045       # annualised risk-free rate
TRADING_MINS   = 390         # minutes in a trading day  (9:30–16:00)
TRADING_DAYS   = 252         # trading days per year


# ─────────────────────────────────────────────────────────────
# 2.  SYNTHETIC DATA GENERATOR  (GBM with intraday vol scaling)
# ─────────────────────────────────────────────────────────────

def generate_ohlcv(ticker: str, params: dict, period_days: int = 180) -> pd.DataFrame:
    """
    Simulate realistic 5-minute OHLCV bars using Geometric Brownian Motion.

    Intraday volatility is amplified 1.6× in the first and last 30 minutes
    of the session to mimic the open/close auction microstructure.
    """
    np.random.seed(params["seed"])

    S0     = params["S0"]
    sigma  = params["vol"]
    mu     = params["drift"]
    dt     = 5.0 / (TRADING_DAYS * TRADING_MINS)   # one 5-min bar in years

    per_bar_drift = mu * dt
    per_bar_vol   = sigma * math.sqrt(dt)

    end   = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=period_days)

    rows = []
    price = S0
    day   = start

    while day <= end:
        if day.weekday() >= 5:          # skip weekends
            day += timedelta(days=1)
            continue

        session_open  = datetime.combine(day, time(9, 30))
        session_close = datetime.combine(day, time(16, 0))
        bar_ts        = session_open

        while bar_ts < session_close - timedelta(minutes=5):
            mins_since_open = (bar_ts - session_open).seconds // 60
            mins_to_close   = (session_close - bar_ts).seconds // 60

            # Higher vol at market open and close
            vol_scale = 1.6 if (mins_since_open < 30 or mins_to_close < 30) else 1.0

            ret       = per_bar_drift + per_bar_vol * vol_scale * np.random.randn()
            new_price = max(price * math.exp(ret), 0.01)

            bar_open  = price
            intra_std = per_bar_vol * vol_scale * price
            bar_high  = max(bar_open, new_price) + abs(np.random.randn()) * intra_std * 0.5
            bar_low   = min(bar_open, new_price) - abs(np.random.randn()) * intra_std * 0.5
            bar_vol   = int(np.random.lognormal(mean=12, sigma=0.5))

            rows.append({
                "Open":   round(bar_open,  2),
                "High":   round(bar_high,  2),
                "Low":    round(bar_low,   2),
                "Close":  round(new_price, 2),
                "Volume": bar_vol,
            })

            price  = new_price
            bar_ts += timedelta(minutes=5)

        day += timedelta(days=1)

    idx = pd.DatetimeIndex(
        [pd.Timestamp(r).tz_localize("America/New_York")
         for r in _bar_timestamps(start, end)],
        name="Datetime",
    )
    df = pd.DataFrame(rows, index=idx[: len(rows)])
    return df


def _bar_timestamps(start: datetime, end: datetime):
    """Yield every 5-min bar timestamp for trading days in [start, end]."""
    day = start
    while day <= end:
        if day.weekday() < 5:
            session_open  = datetime.combine(day, time(9, 30))
            session_close = datetime.combine(day, time(16, 0))
            t = session_open
            while t < session_close - timedelta(minutes=5):
                yield t
                t += timedelta(minutes=5)
        day += timedelta(days=1)


# ─────────────────────────────────────────────────────────────
# 3.  EMA SIGNALS
# ─────────────────────────────────────────────────────────────

def add_signals(df: pd.DataFrame, fast: int = 9, slow: int = 20) -> pd.DataFrame:
    """Compute 9/20 EMA crossover signals on the full multi-day price series."""
    df = df.copy()
    df["EMA9"]  = df["Close"].ewm(span=fast,  adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=slow,  adjust=False).mean()
    df["cross_up"]   = (df["EMA9"] >  df["EMA20"]) & (df["EMA9"].shift(1) <= df["EMA20"].shift(1))
    df["cross_down"] = (df["EMA9"] <  df["EMA20"]) & (df["EMA9"].shift(1) >= df["EMA20"].shift(1))
    return df


# ─────────────────────────────────────────────────────────────
# 4.  BLACK-SCHOLES  +  SPREAD PRICING
# ─────────────────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float,
             sigma: float, opt: str = "call") -> float:
    """European option price via Black-Scholes.  Returns ≥ 0.001."""
    if T <= 1e-9:
        iv = max(S - K, 0) if opt == "call" else max(K - S, 0)
        return max(iv, 0.001)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        px = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        px = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(px, 0.001)


def spread_value(spot: float, K_long: float, K_short: float,
                 T: float, sigma: float, opt: str) -> float:
    """
    Current mark-to-market value of a $1-wide vertical spread (per share).

    Bull call spread : opt='call', K_long < K_short  (K_short = K_long + 1)
    Bear put spread  : opt='put',  K_long > K_short  (K_short = K_long - 1)

    Value is bounded [0, SPREAD_WIDTH].
    """
    long_val  = bs_price(spot, K_long,  T, RISK_FREE, sigma, opt)
    short_val = bs_price(spot, K_short, T, RISK_FREE, sigma, opt)
    return max(min(long_val - short_val, SPREAD_WIDTH), 0.0)


def find_entry_spread(spot: float, signal: str, T: float, sigma: float,
                      max_cost_per_contract: float = MAX_TRADE_COST):
    """
    Walk strikes from ATM outward until the net debit per contract fits the budget.

    Bull call spread (signal='call') :  buy K, sell K+1   →  debit = call(K) - call(K+1)
    Bear put spread  (signal='put')  :  buy K, sell K-1   →  debit = put(K)  - put(K-1)

    Returns (K_long, K_short, debit_per_share) or (None, None, None) if no
    affordable strike found within 5% of spot.
    """
    atm            = round(spot)
    max_otm_steps  = max(int(spot * 0.05), 5)   # cap search at 5% OTM
    opt            = signal                       # 'call' or 'put'

    for step in range(max_otm_steps + 1):
        if signal == "call":
            K_long  = atm + step
            K_short = K_long + SPREAD_WIDTH
        else:
            K_long  = atm - step
            K_short = K_long - SPREAD_WIDTH
            K_long  = max(K_long, SPREAD_WIDTH + 0.01)
            K_short = max(K_short, 0.01)

        debit = spread_value(spot, K_long, K_short, T, sigma, opt)
        cost  = debit * 100   # one contract = 100 shares

        if 0.001 < cost <= max_cost_per_contract:
            return K_long, K_short, debit

    return None, None, None   # no affordable strike found


# ─────────────────────────────────────────────────────────────
# 5.  BACKTESTER
# ─────────────────────────────────────────────────────────────

def week_key(ts: pd.Timestamp) -> str:
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def minutes_to_close(ts: pd.Timestamp) -> int:
    """Minutes remaining until 4:00 PM on the same calendar date."""
    eod = datetime.combine(ts.date(), time(16, 0))
    diff = (eod - ts.replace(tzinfo=None)).total_seconds() / 60
    return max(int(diff), 1)


def run_backtest(dfs: dict, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-ticker vertical spread backtest.

    Rules
    ─────
    • One open position per ticker at a time.
    • Capital (cash) is deducted at trade open; restored + P&L at close.
    • Equity = cash + MTM of all open positions at each bar.
    • PDT counter is incremented at trade OPEN (all 0DTE trades will be same-day
      round-trips, so they count as day trades the moment they're entered).
    • Weekly PDT budget is shared across ALL tickers.
    """
    capital       = INITIAL_CAP
    open_trades   = {}          # ticker → trade metadata dict
    all_trades    = []
    equity_curve  = []
    weekly_opens  = {}          # week_key → count of trades OPENED this week

    # ── collect & sort all bar timestamps ─────────────────────
    all_ts = sorted(set().union(*[set(df.index) for df in dfs.values()]))

    for ts in all_ts:
        wk           = week_key(ts)
        used_wk      = weekly_opens.get(wk, 0)
        bar_time     = ts.time()
        mins_left    = minutes_to_close(ts)
        T_now        = mins_left / (TRADING_DAYS * TRADING_MINS)

        for ticker, df in dfs.items():
            if ts not in df.index:
                continue

            bar    = df.loc[ts]
            spot   = float(bar["Close"])
            sigma  = params[ticker]["vol"]
            trade  = open_trades.get(ticker)

            # ── A. MANAGE OPEN POSITION ───────────────────────
            if trade is not None:
                curr_val = spread_value(
                    spot, trade["K_long"], trade["K_short"],
                    T_now, sigma, trade["opt"]
                )
                entry_debit = trade["entry_debit"]
                max_gain    = SPREAD_WIDTH - entry_debit          # per share
                tp_threshold = entry_debit + TP_RATIO * max_gain  # per share
                sl_threshold = entry_debit * (1 - SL_RATIO)       # 50% of debit

                exit_reason = None
                if curr_val >= tp_threshold:
                    exit_reason = "take_profit"
                elif curr_val <= sl_threshold:
                    exit_reason = "stop_loss"
                elif bar_time >= EXIT_BY:
                    exit_reason = "time_exit"

                if exit_reason:
                    pnl_per_share = curr_val - entry_debit
                    pnl           = pnl_per_share * 100 * trade["contracts"]
                    capital      += trade["cost"] + pnl   # return cost + realised P&L
                    capital       = max(capital, 0.0)

                    all_trades.append({
                        "ticker":        ticker,
                        "date":          str(trade["open_date"]),
                        "week":          wk,
                        "spread_type":   trade["opt"],
                        "K_long":        trade["K_long"],
                        "K_short":       trade["K_short"],
                        "entry_time":    trade["open_time"].strftime("%H:%M"),
                        "exit_time":     bar_time.strftime("%H:%M"),
                        "entry_spot":    round(trade["open_spot"], 2),
                        "exit_spot":     round(spot, 2),
                        "entry_debit":   round(entry_debit, 4),
                        "exit_value":    round(curr_val, 4),
                        "contracts":     trade["contracts"],
                        "pnl":           round(pnl, 2),
                        "pnl_pct":       round(pnl_per_share / entry_debit * 100, 1),
                        "exit_reason":   exit_reason,
                        "capital_after": round(capital, 2),
                    })
                    del open_trades[ticker]

            # ── B. LOOK FOR ENTRY ─────────────────────────────
            else:
                if bar_time < ENTRY_AFTER or bar_time >= EXIT_BY:
                    continue
                # PDT check: count opens this week (all 0DTE = guaranteed day trades)
                if weekly_opens.get(wk, 0) >= MAX_DT_WEEK:
                    continue
                if capital < 5.0:
                    continue

                signal = None
                if bar.get("cross_up", False):
                    signal = "call"
                elif bar.get("cross_down", False):
                    signal = "put"

                if signal is None:
                    continue

                # Position sizing: spend up to min(MAX_TRADE_COST, 10% of capital)
                budget = min(MAX_TRADE_COST, capital * 0.10)
                K_long, K_short, debit = find_entry_spread(
                    spot, signal, T_now, sigma, max_cost_per_contract=budget
                )
                if K_long is None:
                    continue                     # no affordable strike

                cost_1c   = debit * 100
                contracts = max(1, int(budget // cost_1c))
                total_cost = contracts * cost_1c

                if total_cost > capital:
                    continue

                capital -= total_cost
                # Increment PDT counter at open — 0DTE always closes same day
                weekly_opens[wk] = weekly_opens.get(wk, 0) + 1

                open_trades[ticker] = {
                    "opt":         signal,
                    "K_long":      K_long,
                    "K_short":     K_short,
                    "entry_debit": debit,
                    "contracts":   contracts,
                    "cost":        total_cost,
                    "open_date":   ts.date(),
                    "open_time":   bar_time,
                    "open_spot":   spot,
                }

        # ── Equity snapshot (cash + MTM of all open positions) ─
        mtm = 0.0
        for ticker, trade in open_trades.items():
            if ts in dfs[ticker].index:
                s    = float(dfs[ticker].loc[ts, "Close"])
                val  = spread_value(s, trade["K_long"], trade["K_short"],
                                    T_now, params[ticker]["vol"], trade["opt"])
                mtm += val * 100 * trade["contracts"]
        equity_curve.append({"ts": ts, "equity": round(capital + mtm, 2)})

    # Force-close any positions still open at the very last bar
    if open_trades:
        last_ts    = all_ts[-1]
        wk         = week_key(last_ts)
        for ticker, trade in list(open_trades.items()):
            if last_ts in dfs[ticker].index:
                spot    = float(dfs[ticker].loc[last_ts, "Close"])
                sigma   = params[ticker]["vol"]
                curr_val = spread_value(spot, trade["K_long"], trade["K_short"],
                                        1 / (TRADING_DAYS * TRADING_MINS),
                                        sigma, trade["opt"])
                pnl = (curr_val - trade["entry_debit"]) * 100 * trade["contracts"]
                capital = max(capital + trade["cost"] + pnl, 0.0)
                all_trades.append({
                    "ticker": ticker, "date": str(trade["open_date"]),
                    "week": wk, "spread_type": trade["opt"],
                    "K_long": trade["K_long"], "K_short": trade["K_short"],
                    "entry_time": trade["open_time"].strftime("%H:%M"),
                    "exit_time": "FINAL", "entry_spot": round(trade["open_spot"], 2),
                    "exit_spot": round(spot, 2),
                    "entry_debit": round(trade["entry_debit"], 4),
                    "exit_value": round(curr_val, 4), "contracts": trade["contracts"],
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((curr_val - trade["entry_debit"]) / trade["entry_debit"] * 100, 1),
                    "exit_reason": "force_close", "capital_after": round(capital, 2),
                })

    return pd.DataFrame(all_trades), pd.DataFrame(equity_curve)


# ─────────────────────────────────────────────────────────────
# 6.  METRICS
# ─────────────────────────────────────────────────────────────

def print_metrics(trades: pd.DataFrame, equity: pd.DataFrame,
                  initial_cap: float = INITIAL_CAP) -> dict:

    if trades.empty:
        print("No trades executed.")
        return {}

    pnls   = trades["pnl"].values
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate  = len(wins) / len(pnls) * 100
    avg_win   = wins.mean()   if len(wins)   else 0.0
    avg_loss  = losses.mean() if len(losses) else 0.0
    pf        = abs(wins.sum() / losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")

    # Sharpe on daily P&L (annualised)
    daily_pnl = trades.groupby("date")["pnl"].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * math.sqrt(TRADING_DAYS)
    else:
        sharpe = 0.0

    # Max drawdown on equity curve
    eq     = equity["equity"].values
    peak   = np.maximum.accumulate(eq)
    peak[peak == 0] = 1e-9
    dd     = (eq - peak) / peak
    max_dd = dd.min() * 100
    final  = eq[-1] if len(eq) else initial_cap + pnls.sum()

    sep = "═" * 60
    print(f"\n{sep}")
    print("  VERTICAL SPREAD BACKTEST  –  6-Month Results")
    print(sep)
    print(f"  Underlyings     : SPY | QQQ | NVDA")
    print(f"  Strategy        : $1-wide Bull Call / Bear Put Spreads")
    print(f"  Period          : last 6 months  (synthetic GBM data)")
    print(sep)
    print(f"  Total Trades    : {len(trades)}")
    print(f"  Winners         : {len(wins)}   Losers: {len(losses)}")
    print(f"  Win Rate        : {win_rate:.1f}%")
    print(f"  Average Win     : ${avg_win:>8.2f}")
    print(f"  Average Loss    : ${avg_loss:>8.2f}")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Sharpe Ratio    : {sharpe:.2f}  (annualised)")
    print(f"  Max Drawdown    : {max_dd:.1f}%")
    print(f"  Net P&L         : ${pnls.sum():>+.2f}")
    print(f"  Starting Bal    : ${initial_cap:.2f}")
    print(f"  Ending Bal      : ${final:.2f}")
    print(f"  Total Return    : {(final / initial_cap - 1) * 100:+.1f}%")
    print(sep)

    # Per-ticker breakdown
    print("\n  ── Per-Ticker Breakdown ──────────────────────────")
    for tkr in trades["ticker"].unique():
        t    = trades[trades["ticker"] == tkr]
        w    = t[t["pnl"] > 0]
        wr   = len(w) / len(t) * 100
        net  = t["pnl"].sum()
        print(f"  {tkr:<5}: {len(t):>3} trades | WR {wr:>5.1f}% | Net ${net:>+7.2f}")

    # Exit-reason breakdown
    print("\n  ── Exit Reasons ──────────────────────────────────")
    for reason, cnt in trades["exit_reason"].value_counts().items():
        print(f"  {reason:<16}: {cnt}")

    # PDT compliance
    over = (trades.groupby("week").size() > MAX_DT_WEEK).sum()
    print(f"\n  PDT violations  : {over} week(s)")
    print(sep)

    return {
        "total_trades": len(trades), "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": pf, "sharpe": sharpe,
        "max_drawdown": max_dd, "final_balance": final,
        "total_return_pct": (final / initial_cap - 1) * 100,
    }


# ─────────────────────────────────────────────────────────────
# 7.  CHARTS + OUTPUT FILES
# ─────────────────────────────────────────────────────────────

def save_outputs(trades: pd.DataFrame, equity: pd.DataFrame) -> None:
    trades.to_csv("trades.csv", index=False)
    print("\n  Trade log  → trades.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Vertical Spread 0DTE Backtest  –  SPY | QQQ | NVDA",
                     fontsize=13, fontweight="bold")
        gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)

        PALETTE = {"SPY": "#2196F3", "QQQ": "#FF9800", "NVDA": "#9C27B0"}

        # ── 1. Equity curve ───────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :2])
        if not equity.empty:
            ax1.plot(equity["ts"], equity["equity"], color="#1a237e", linewidth=1.2)
            ax1.fill_between(equity["ts"], equity["equity"], INITIAL_CAP,
                             where=(equity["equity"] >= INITIAL_CAP),
                             alpha=0.25, color="#4CAF50")
            ax1.fill_between(equity["ts"], equity["equity"], INITIAL_CAP,
                             where=(equity["equity"] < INITIAL_CAP),
                             alpha=0.25, color="#F44336")
            ax1.axhline(INITIAL_CAP, color="gray", linestyle="--",
                        linewidth=0.8, label=f"Start ${INITIAL_CAP:.0f}")
            ax1.set_title("Equity Curve")
            ax1.set_ylabel("Portfolio Value ($)")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.25)

        # ── 2. P&L per trade (color by ticker) ───────────────
        ax2 = fig.add_subplot(gs[0, 2])
        for i, (_, row) in enumerate(trades.iterrows()):
            color = "#4CAF50" if row["pnl"] > 0 else "#F44336"
            ax2.bar(i, row["pnl"], color=color, edgecolor="none", width=0.8)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_title("P&L per Trade ($)")
        ax2.set_xlabel("Trade #")
        ax2.set_ylabel("P&L ($)")
        ax2.grid(True, alpha=0.25, axis="y")

        # ── 3. Cumulative P&L by ticker ───────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        for tkr in trades["ticker"].unique():
            t = trades[trades["ticker"] == tkr].copy()
            t = t.sort_values("date")
            ax3.plot(range(len(t)), t["pnl"].cumsum().values,
                     label=tkr, color=PALETTE.get(tkr, "gray"), linewidth=1.4)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_title("Cumulative P&L by Ticker")
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("Cumul. P&L ($)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.25)

        # ── 4. P&L % distribution ─────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(trades["pnl_pct"], bins=24, color="#5C6BC0",
                 edgecolor="white", alpha=0.85)
        ax4.axvline(0,   color="black", linewidth=1.0)
        ax4.axvline(-50, color="#F44336", linewidth=1.0, linestyle="--", label="SL −50%")
        ax4.axvline( 80, color="#4CAF50", linewidth=1.0, linestyle="--", label="TP +80%")
        ax4.set_title("Return % Distribution")
        ax4.set_xlabel("Return on Debit (%)")
        ax4.set_ylabel("Count")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.25)

        # ── 5. Win rate by ticker (bar) ───────────────────────
        ax5 = fig.add_subplot(gs[1, 2])
        tickers = trades["ticker"].unique()
        wr_vals = [
            len(trades[(trades["ticker"] == t) & (trades["pnl"] > 0)]) /
            max(len(trades[trades["ticker"] == t]), 1) * 100
            for t in tickers
        ]
        bars = ax5.bar(tickers, wr_vals,
                       color=[PALETTE.get(t, "gray") for t in tickers],
                       edgecolor="white")
        ax5.axhline(50, color="black", linestyle="--", linewidth=0.8, label="50%")
        for bar_obj, v in zip(bars, wr_vals):
            ax5.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                     v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
        ax5.set_title("Win Rate by Ticker")
        ax5.set_ylabel("Win Rate (%)")
        ax5.set_ylim(0, 100)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.25, axis="y")

        plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
        print("  Chart      → backtest_results.png")

    except Exception as exc:
        print(f"  [chart skipped: {exc}]")


# ─────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tickers = list(TICKER_PARAMS.keys())

    # ── Generate / fetch data ──────────────────────────────────
    print(f"[1/4] Generating synthetic 5-min data for {', '.join(tickers)} …")
    dfs = {}
    for tkr in tickers:
        df = generate_ohlcv(tkr, TICKER_PARAMS[tkr], period_days=PERIOD_DAYS)
        df = add_signals(df)
        sigs = int(df["cross_up"].sum() + df["cross_down"].sum())
        print(f"      {tkr}: {len(df):,} bars | {sigs} EMA crossover signals")
        dfs[tkr] = df

    # ── Run backtest ───────────────────────────────────────────
    print("[2/4] Running backtest …")
    trades_df, equity_df = run_backtest(dfs, TICKER_PARAMS)
    print(f"      {len(trades_df)} trades executed across all tickers")

    # ── Print metrics ──────────────────────────────────────────
    print("[3/4] Computing metrics …")
    print_metrics(trades_df, equity_df)

    # ── Save outputs ───────────────────────────────────────────
    print("[4/4] Saving outputs …")
    save_outputs(trades_df, equity_df)

    print("\nDone.")
