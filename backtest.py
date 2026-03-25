"""
0DTE SPY/QQQ Options Day Trading Backtest
Strategy: 9 EMA / 20 EMA crossover on 5-minute candles
Platform constraints: Robinhood PDT (max 3 day trades/week)
Capital: $300 | Max per trade: $50 | Stop loss: -40% | Take profit: +80%
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, time, timedelta
import math
import sys

# ── 1. DATA FETCH ──────────────────────────────────────────────────────────────

def generate_synthetic_spy(ticker="SPY", period_days=180, seed=42):
    """
    Generate realistic synthetic 5-minute OHLCV data for SPY.

    Uses geometric Brownian motion (GBM) calibrated to SPY's historical
    parameters: ~15% annual vol, ~10% annual drift. Intraday volatility
    is scaled up in the first 30 minutes (open) and last 30 minutes (close)
    to mimic real market microstructure.
    """
    np.random.seed(seed)

    # SPY parameters (calibrated to 2023-2025 historical data)
    S0        = 500.0   # starting price (roughly SPY in late 2024)
    ann_drift = 0.10    # ~10% annual drift
    ann_vol   = 0.15    # ~15% annual vol
    dt        = 5 / (252 * 390)   # 5-min bar as fraction of trading year

    per_bar_drift = ann_drift * dt
    per_bar_vol   = ann_vol   * math.sqrt(dt)

    end   = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=period_days)

    timestamps = []
    closes     = []
    opens_     = []
    highs      = []
    lows       = []
    volumes    = []

    price = S0
    day   = start

    while day <= end:
        if day.weekday() >= 5:   # skip weekends
            day += timedelta(days=1)
            continue

        # Build 5-min bars for 09:30–15:55 (78 bars per day)
        bar_time = datetime.combine(day, time(9, 30))
        day_close_time = datetime.combine(day, time(16, 0))

        day_open = price
        while bar_time < day_close_time - timedelta(minutes=5):
            # Intraday vol scaling: higher at open/close
            minutes_since_open  = (bar_time - datetime.combine(day, time(9, 30))).seconds // 60
            minutes_to_close    = (day_close_time - bar_time).seconds // 60

            if minutes_since_open < 30 or minutes_to_close < 30:
                vol_scale = 1.6   # open/close premium
            else:
                vol_scale = 1.0

            ret  = per_bar_drift + per_bar_vol * vol_scale * np.random.randn()
            new_price = price * math.exp(ret)

            # Synthesize OHLC within bar
            bar_open  = price
            intra_vol = per_bar_vol * vol_scale * price
            high_wick = abs(np.random.randn()) * intra_vol * 0.5
            low_wick  = abs(np.random.randn()) * intra_vol * 0.5
            bar_high  = max(bar_open, new_price) + high_wick
            bar_low   = min(bar_open, new_price) - low_wick
            bar_vol   = int(np.random.lognormal(mean=12, sigma=0.5))

            timestamps.append(pd.Timestamp(bar_time).tz_localize("America/New_York"))
            opens_.append(round(bar_open,  2))
            highs.append(round(bar_high,   2))
            lows.append(round(bar_low,     2))
            closes.append(round(new_price, 2))
            volumes.append(bar_vol)

            price    = new_price
            bar_time += timedelta(minutes=5)

        day += timedelta(days=1)

    df = pd.DataFrame({
        "Open":   opens_,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    }, index=timestamps)

    return df


def fetch_spy_data(ticker="SPY", period_days=180):
    """
    Try to fetch live 5-minute data from yfinance.
    Falls back to realistic synthetic data if network is unavailable.
    """
    end   = datetime.today()
    start = end - timedelta(days=period_days)

    print(f"[*] Fetching {ticker} 5-min data from {start.date()} to {end.date()} ...")
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, interval="5m",
                         progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        if df.empty:
            raise ValueError("Empty dataframe")

        print(f"    Got {len(df):,} 5-min bars (live) "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df

    except Exception as e:
        print(f"    [!] Live data unavailable ({type(e).__name__}). "
              f"Using calibrated synthetic SPY data.")
        df = generate_synthetic_spy(ticker=ticker, period_days=period_days)
        print(f"    Generated {len(df):,} synthetic 5-min bars "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df


# ── 2. EMA SIGNALS ─────────────────────────────────────────────────────────────

def compute_emas(df, fast=9, slow=20):
    df = df.copy()
    df["EMA9"]  = df["Close"].ewm(span=fast,  adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=slow, adjust=False).mean()
    df["cross_up"]   = (df["EMA9"] > df["EMA20"]) & (df["EMA9"].shift(1) <= df["EMA20"].shift(1))
    df["cross_down"] = (df["EMA9"] < df["EMA20"]) & (df["EMA9"].shift(1) >= df["EMA20"].shift(1))
    return df


# ── 3. BLACK-SCHOLES OPTION PRICING ────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes price for a European option."""
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return max(intrinsic, 0.01)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0.01)


def find_affordable_strike(spot, option_type, minutes_to_close,
                            max_premium_per_share, iv=0.20, r=0.045):
    """
    Walk OTM from ATM to find the cheapest strike whose per-share premium
    fits within max_premium_per_share.

    For 0DTE SPY (~$500), ATM options cost $1-3/share = $100-300/contract —
    far beyond a $50 budget. Going OTM (1-3%) brings premiums to $0.20-0.50.

    Returns (strike, premium_per_share) or (None, None) if unaffordable.
    """
    T = max(minutes_to_close, 1) / (252 * 390)

    # Try strikes in $1 increments from ATM out to 3% OTM (max ~$15 on $500)
    max_otm_dollars = max(int(spot * 0.03), 5)
    atm = round(spot)

    best_strike = None
    best_premium = None

    for delta_k in range(0, max_otm_dollars + 1):
        if option_type == "call":
            K = atm + delta_k
        else:
            K = atm - delta_k

        K = max(K, 1)
        premium = bs_price(spot, K, T, r, sigma=iv, option_type=option_type)

        if premium <= max_premium_per_share:
            best_strike  = K
            best_premium = premium
            break   # first strike within budget (least OTM = highest delta)

    return best_strike, best_premium


def estimate_option_premium(spot, option_type, minutes_to_close,
                             strike=None, iv=0.20, r=0.045):
    """
    Estimate 0DTE option premium. If strike is None, uses ATM.
    minutes_to_close: minutes until 4 PM close.
    IV = 20% baseline (typical SPY 0DTE VIX regime).
    """
    T = max(minutes_to_close, 1) / (252 * 390)
    K = strike if strike is not None else round(spot)
    return bs_price(spot, K, T, r, sigma=iv, option_type=option_type)


# ── 4. BACKTESTER ──────────────────────────────────────────────────────────────

INITIAL_CAPITAL  = 300.0
MAX_TRADE_SIZE   = 50.0       # max dollars per trade
MAX_RISK_PCT     = 0.15       # 15% of account per trade
STOP_LOSS_PCT    = -0.40      # exit if option -40%
TAKE_PROFIT_PCT  =  0.80      # exit if option +80%
ENTRY_AFTER      = time(9, 45)
EXIT_BEFORE      = time(15, 30)
MAX_TRADES_WEEK  = 3          # PDT rule
CONTRACTS_PER_LOT = 1         # 1 contract = 100 shares

# One "day trade" on Robinhood = same-day open+close. We count each round-trip.

def week_key(ts):
    """ISO week string like '2024-W23'."""
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def run_backtest(df):
    capital   = INITIAL_CAPITAL
    trades    = []
    balance_curve = []
    weekly_trades = {}   # week_key -> count of completed day trades

    # Group by trading day
    daily_groups = df.groupby(df.index.date)

    for date, day_df in daily_groups:
        day_df = day_df.sort_index()

        # Only trade regular market hours
        market_bars = day_df.between_time("09:30", "15:55")
        if len(market_bars) < 25:   # need enough bars for EMAs to warm up
            continue

        # How many day trades used this week?
        today_ts = pd.Timestamp(date, tz="America/New_York")
        wk = week_key(today_ts)
        used_this_week = weekly_trades.get(wk, 0)

        in_trade = None   # dict with trade metadata

        for i, (ts, bar) in enumerate(market_bars.iterrows()):
            bar_time = ts.time()

            # ── MANAGE OPEN TRADE ──
            if in_trade is not None:
                spot    = bar["Close"]
                minutes_left = max(1, (datetime.combine(date, time(16, 0))
                                       - ts.replace(tzinfo=None)).seconds // 60)
                current_premium = estimate_option_premium(
                    spot, in_trade["option_type"], minutes_left,
                    strike=in_trade.get("entry_strike"))
                pnl_pct = (current_premium - in_trade["entry_premium"]) / in_trade["entry_premium"]

                exit_reason = None
                if pnl_pct <= STOP_LOSS_PCT:
                    exit_reason = "stop_loss"
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    exit_reason = "take_profit"
                elif bar_time >= EXIT_BEFORE:
                    exit_reason = "time_exit"

                if exit_reason:
                    exit_premium  = current_premium
                    trade_pnl_per_contract = (exit_premium - in_trade["entry_premium"]) * 100
                    trade_pnl     = trade_pnl_per_contract * in_trade["contracts"]
                    capital      += trade_pnl
                    capital       = max(capital, 0)   # no margin / no negative

                    trades.append({
                        "date":           str(date),
                        "week":           wk,
                        "option_type":    in_trade["option_type"],
                        "strike":         in_trade.get("entry_strike"),
                        "entry_time":     in_trade["entry_time"].strftime("%H:%M"),
                        "exit_time":      bar_time.strftime("%H:%M"),
                        "entry_spot":     round(in_trade["entry_spot"], 2),
                        "exit_spot":      round(spot, 2),
                        "entry_premium":  round(in_trade["entry_premium"], 4),
                        "exit_premium":   round(exit_premium, 4),
                        "contracts":      in_trade["contracts"],
                        "pnl":            round(trade_pnl, 2),
                        "pnl_pct":        round(pnl_pct * 100, 1),
                        "exit_reason":    exit_reason,
                        "capital_after":  round(capital, 2),
                    })

                    weekly_trades[wk] = used_this_week + 1
                    used_this_week   += 1
                    in_trade          = None

                balance_curve.append({"ts": ts, "capital": capital})
                continue

            # ── LOOK FOR ENTRY ──
            if bar_time < ENTRY_AFTER or bar_time >= EXIT_BEFORE:
                balance_curve.append({"ts": ts, "capital": capital})
                continue

            if used_this_week >= MAX_TRADES_WEEK:
                balance_curve.append({"ts": ts, "capital": capital})
                continue

            if capital <= 5:   # effectively bust
                balance_curve.append({"ts": ts, "capital": capital})
                continue

            signal = None
            if bar.get("cross_up", False):
                signal = "call"
            elif bar.get("cross_down", False):
                signal = "put"

            if signal is None:
                balance_curve.append({"ts": ts, "capital": capital})
                continue

            # Position sizing
            # Hard cap: min($50, 15% of capital) per trade
            max_trade_dollars = min(MAX_TRADE_SIZE, capital * MAX_RISK_PCT)
            # Per-share budget for a single contract (100 shares)
            max_premium_per_share = max_trade_dollars / 100.0

            spot         = bar["Close"]
            minutes_left = max(1, (datetime.combine(date, time(16, 0))
                                    - ts.replace(tzinfo=None)).seconds // 60)

            # Find the least-OTM strike we can actually afford
            strike, entry_premium = find_affordable_strike(
                spot, signal, minutes_left, max_premium_per_share)

            if strike is None:
                # Even the most OTM allowed strike exceeds budget — skip
                balance_curve.append({"ts": ts, "capital": capital})
                continue

            cost_per_contract = entry_premium * 100   # 1 contract = 100 shares
            contracts  = max(1, int(max_trade_dollars // cost_per_contract))
            total_cost = contracts * cost_per_contract

            if total_cost > capital:
                balance_curve.append({"ts": ts, "capital": capital})
                continue

            capital  -= total_cost   # deduct cost (option purchase)

            in_trade = {
                "option_type":    signal,
                "entry_time":     bar_time,
                "entry_spot":     spot,
                "entry_strike":   strike,
                "entry_premium":  entry_premium,
                "contracts":      contracts,
                "total_cost":     total_cost,
            }

            balance_curve.append({"ts": ts, "capital": capital})

        # Force close any remaining open trade at end of day
        if in_trade is not None and len(market_bars) > 0:
            last_bar   = market_bars.iloc[-1]
            spot       = last_bar["Close"]
            exit_premium = estimate_option_premium(
                spot, in_trade["option_type"], 1,
                strike=in_trade.get("entry_strike"))
            pnl_pct    = (exit_premium - in_trade["entry_premium"]) / in_trade["entry_premium"]
            trade_pnl  = (exit_premium - in_trade["entry_premium"]) * 100 * in_trade["contracts"]
            capital   += trade_pnl
            capital    = max(capital, 0)

            wk = week_key(pd.Timestamp(date, tz="America/New_York"))
            trades.append({
                "date":          str(date),
                "week":          wk,
                "option_type":   in_trade["option_type"],
                "entry_time":    in_trade["entry_time"].strftime("%H:%M"),
                "exit_time":     "EOD",
                "entry_spot":    round(in_trade["entry_spot"], 2),
                "exit_spot":     round(spot, 2),
                "entry_premium": round(in_trade["entry_premium"], 4),
                "exit_premium":  round(exit_premium, 4),
                "contracts":     in_trade["contracts"],
                "pnl":           round(trade_pnl, 2),
                "pnl_pct":       round(pnl_pct * 100, 1),
                "exit_reason":   "eod_close",
                "capital_after": round(capital, 2),
            })
            weekly_trades[wk] = weekly_trades.get(wk, 0) + 1
            in_trade = None

    return pd.DataFrame(trades), pd.DataFrame(balance_curve)


# ── 5. METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(trades_df, balance_df, initial_capital=INITIAL_CAPITAL):
    if trades_df.empty:
        print("No trades executed.")
        return

    pnls      = trades_df["pnl"].values
    pnl_pcts  = trades_df["pnl_pct"].values
    wins      = pnls[pnls > 0]
    losses    = pnls[pnls <= 0]

    win_rate  = len(wins) / len(pnls) * 100
    avg_gain  = wins.mean()  if len(wins)   else 0
    avg_loss  = losses.mean() if len(losses) else 0
    total_pnl = pnls.sum()

    # Sharpe (daily P&L; annualise with sqrt(252))
    if len(pnls) > 1 and pnls.std() > 0:
        sharpe = (pnls.mean() / pnls.std()) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown on balance curve
    if not balance_df.empty:
        bal = balance_df["capital"].values
        peak = np.maximum.accumulate(bal)
        peak[peak == 0] = 1e-9
        dd = (bal - peak) / peak
        max_dd = dd.min() * 100
        final_balance = bal[-1]
    else:
        max_dd = 0.0
        final_balance = initial_capital + total_pnl

    print("\n" + "="*58)
    print("  0DTE SPY OPTIONS BACKTEST RESULTS (Last 6 Months)")
    print("="*58)
    print(f"  Total Trades        : {len(trades_df)}")
    print(f"  Winning Trades      : {len(wins)}  |  Losing: {len(losses)}")
    print(f"  Win Rate            : {win_rate:.1f}%")
    print(f"  Average Gain        : ${avg_gain:>8.2f}")
    print(f"  Average Loss        : ${avg_loss:>8.2f}")
    print(f"  Profit Factor       : {abs(wins.sum()/losses.sum()):.2f}" if len(losses) else "  Profit Factor       : ∞")
    print(f"  Sharpe Ratio        : {sharpe:.2f}")
    print(f"  Max Drawdown        : {max_dd:.1f}%")
    print(f"  Net P&L             : ${total_pnl:>+.2f}")
    print(f"  Starting Balance    : ${initial_capital:.2f}")
    print(f"  Ending Balance      : ${final_balance:.2f}")
    print(f"  Total Return        : {(final_balance/initial_capital - 1)*100:+.1f}%")
    print("="*58)

    # Exit reason breakdown
    reasons = trades_df["exit_reason"].value_counts()
    print("\n  Exit Breakdown:")
    for reason, cnt in reasons.items():
        print(f"    {reason:<16}: {cnt}")

    # Trade type breakdown
    types = trades_df["option_type"].value_counts()
    print("\n  Option Type:")
    for t, cnt in types.items():
        print(f"    {t:<10}: {cnt}")

    # Weekly trade count compliance check
    weekly = trades_df.groupby("week").size()
    over_pdt = (weekly > MAX_TRADES_WEEK).sum()
    print(f"\n  PDT compliance: {over_pdt} week(s) exceeded 3-trade limit")
    print("="*58)

    return {
        "total_trades": len(trades_df),
        "win_rate": win_rate,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_balance": final_balance,
        "total_return_pct": (final_balance/initial_capital - 1)*100,
    }


# ── 6. SAVE CSV + CHART ────────────────────────────────────────────────────────

def save_outputs(trades_df, balance_df):
    trades_df.to_csv("trades.csv", index=False)
    print("\n  Trade log saved → trades.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("0DTE SPY Options Day Trade Backtest", fontsize=14, fontweight="bold")

        # 1) Equity curve
        ax = axes[0, 0]
        if not balance_df.empty:
            ax.plot(balance_df["ts"], balance_df["capital"], color="#2196F3", linewidth=1.2)
            ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.8, label="Start")
            ax.set_title("Equity Curve")
            ax.set_ylabel("Capital ($)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2) P&L per trade bar chart
        ax = axes[0, 1]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in trades_df["pnl"]]
        ax.bar(range(len(trades_df)), trades_df["pnl"], color=colors, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("P&L per Trade ($)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("P&L ($)")
        ax.grid(True, alpha=0.3, axis="y")

        # 3) P&L % distribution
        ax = axes[1, 0]
        ax.hist(trades_df["pnl_pct"], bins=20, color="#9C27B0", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(-40, color="red",   linewidth=1, linestyle="--", label="Stop -40%")
        ax.axvline( 80, color="green", linewidth=1, linestyle="--", label="TP +80%")
        ax.set_title("P&L % Distribution")
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4) Cumulative P&L
        ax = axes[1, 1]
        cumulative = trades_df["pnl"].cumsum()
        ax.plot(cumulative.values, color="#FF9800", linewidth=1.5)
        ax.fill_between(range(len(cumulative)), cumulative.values, 0,
                        where=(cumulative.values >= 0), alpha=0.3, color="#4CAF50")
        ax.fill_between(range(len(cumulative)), cumulative.values, 0,
                        where=(cumulative.values < 0), alpha=0.3, color="#F44336")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Cumulative P&L ($)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
        print("  Chart saved        → backtest_results.png")

    except Exception as e:
        print(f"  [chart skipped: {e}]")


# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"

    # 1. Fetch data
    df = fetch_spy_data(ticker=ticker, period_days=180)

    # 2. Compute EMAs and signals
    print("[*] Computing EMA crossover signals ...")
    df = compute_emas(df, fast=9, slow=20)
    total_signals = int(df["cross_up"].sum() + df["cross_down"].sum())
    print(f"    {total_signals} crossover signals found")

    # 3. Run backtest
    print("[*] Running backtest ...")
    trades_df, balance_df = run_backtest(df)
    print(f"    {len(trades_df)} trades executed")

    # 4. Metrics
    compute_metrics(trades_df, balance_df)

    # 5. Save outputs
    save_outputs(trades_df, balance_df)

    print("\nDone.")
