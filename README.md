# ⚡ Options Day Trading Bot

An aggressive, production-ready options day trading bot targeting small-cap biotech and energy stocks. Built on Polygon.io (data), Tradier (execution), and Benzinga (news).

## Architecture

```
trading-backtest/
├── main.py          # Orchestration loop — start here
├── scanner.py       # Real-time stock screener (Polygon + yfinance)
├── strategy.py      # Signal engine: VWAP, RSI, ATR
├── risk.py          # Position sizing + all hard kill switches
├── broker.py        # Tradier API wrapper (paper & live)
├── news.py          # Benzinga catalyst detection for biotech
├── dashboard.py     # Rich live terminal UI
├── utils.py         # Config loader, logger, time helpers
├── config.yaml      # All tunable parameters (no hardcoded values)
├── requirements.txt
└── .env.example
```

## Requirements

- Python 3.11+
- Polygon.io account (Starter plan minimum for real-time data)
- Tradier account (free paper trading sandbox)
- Benzinga API key (for biotech catalyst detection)

## Setup

### 1. Clone & install dependencies

```bash
cd trading-backtest
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

Required keys:
| Variable | Source |
|---|---|
| `POLYGON_API_KEY` | [polygon.io/dashboard](https://polygon.io/dashboard/api-keys) |
| `TRADIER_API_KEY` | [developer.tradier.com](https://developer.tradier.com/) |
| `TRADIER_ACCOUNT_ID` | Tradier dashboard → account number |
| `BENZINGA_API_KEY` | [benzinga.com/apis](https://www.benzinga.com/apis/) |

### 3. Set your account capital

```bash
# In .env:
ACCOUNT_CAPITAL=25000.00
```

---

## Paper Trading Quickstart

```bash
# Ensure LIVE_TRADING=false in .env (the default)
python main.py
```

The terminal dashboard launches immediately. The bot:
1. Waits until 09:40 ET before scanning (skips opening chaos)
2. Scans every 5 minutes for qualifying small-cap movers
3. Evaluates VWAP/RSI/ATR signals + catalyst/USO correlation
4. Sizes positions at 5–12% of account based on confidence
5. Exits positions at 35% stop-loss, 30–100% profit target, or 90-minute max hold
6. Closes all positions by 3:45 PM ET
7. Halts all trading if daily loss exceeds 4%

---

## Live Trading Checklist

Work through this before switching to live money:

- [ ] Paper traded for at least 5 full sessions with positive expectancy
- [ ] Verified Tradier live API key has options trading permissions enabled
- [ ] Set `LIVE_TRADING=true` in `.env`
- [ ] Set `ACCOUNT_CAPITAL` to your actual funded account balance
- [ ] Confirmed Tradier account has sufficient options buying power
- [ ] Reviewed and accepted all risk parameters in `config.yaml`
- [ ] Tested the emergency shutdown (Ctrl+C) in paper mode
- [ ] Reviewed logs in `logs/bot.jsonl` for any unexpected rejections
- [ ] Confirmed Polygon API plan supports real-time data (not 15-min delayed)
- [ ] Benzinga API key is active and returning news results

To go live:
```bash
LIVE_TRADING=true python main.py
# You will be prompted to confirm before any orders are placed
```

---

## Backtesting

Uses Polygon historical options data. Requires the same API key with historical data access.

```bash
BACKTEST=true python main.py --start 2024-01-01 --end 2024-12-31
```

Override initial capital:
```bash
BACKTEST=true python main.py --start 2024-06-01 --end 2024-08-31 --capital 50000
```

Backtest results are logged to `logs/bot.jsonl`. Each trade entry/exit records the full signal rationale, sizing, and P&L.

---

## Configuration Reference (config.yaml)

All parameters are documented inline in `config.yaml`. Key settings:

| Section | Key | Default | Meaning |
|---|---|---|---|
| scanner | interval_seconds | 300 | Scan frequency |
| scanner | volume_spike_multiplier | 2.5 | Min volume spike required |
| scanner | min_intraday_move_pct | 3.0 | Min ±% price move |
| options | max_spread_entry | 0.20 | Skip if spread > $0.20 |
| options | min_open_interest | 500 | Skip if OI < 500 |
| strategy | rsi_call_threshold | 60 | RSI floor for calls |
| strategy | atr_expansion_multiplier | 1.3 | ATR expansion required |
| risk | stop_loss_pct | 35 | Exit at 35% loss on premium |
| risk | daily_loss_limit_pct | 4.0 | Halt trading at -4% daily |
| risk | max_hold_minutes | 90 | Hard exit regardless of P&L |

---

## Signal Logic

A trade signal requires ALL of the following to align simultaneously:

1. **VWAP breakout** — Price is above (calls) or below (puts) VWAP on both 1m and 5m charts
2. **Volume confirmation** — Current 1m bar volume > 1.5× average 1m volume
3. **RSI(14) on 5m** — > 60 for calls, < 40 for puts
4. **ATR expansion** — Current ATR > 1.3× 10-period ATR average
5. **Sector rule (biotech)** — Benzinga headline confirms catalyst within 24 hours
6. **Sector rule (energy)** — USO moving in the same direction as the trade

**Confidence score** (0–100) determines position size:
- 60–74 → 5% of account
- 75–89 → 8% of account
- 90–100 → 12% of account

---

## Risk Rules (immutable — never bypassed)

| Rule | Trigger | Action |
|---|---|---|
| Stop loss | -35% on premium | Immediate market sell |
| Profit target | +100% on premium | Limit sell |
| Max hold | 90 minutes | Market sell |
| Daily loss limit | -4% of account | Halt all trading |
| EOD force close | 3:45 PM ET | Market sell all |
| No new entries | After 3:15 PM ET | Skip scan cycle |
| Opening blackout | Before 9:40 AM ET | Skip scan cycle |
| Sector cap | 2 positions per sector | Skip additional entries |
| Liquidity guard | Spread > $0.20 | Skip this option |
| Gap risk | Underlying gaps >8% vs position | Immediate market sell |

---

## Logs

All events are logged as JSON to `logs/bot.jsonl`:

```json
{"ts": "2024-07-05T09:47:23.441Z", "level": "INFO", "logger": "main", "msg": "entering_trade",
 "ticker": "MRNA", "direction": "call", "option": "MRNA240705C00130000",
 "qty": 2, "mid": 2.45, "confidence": 82, "size_pct": 8.0}
```

Events logged: every signal generated, every rejection (with reason), every entry/exit, every scan cycle, halt triggers.

---

## Opinionated Design Decisions

**Why Tradier?** Free paper trading sandbox with a REST API that maps directly to live execution. No special paper-mode flag needed — the URL is the only difference.

**Why not WebSocket for everything?** The 5-minute scan interval means REST polling is sufficient for universe selection. We use REST for bar data and option quotes. For a sub-second HFT strategy you'd move to WebSocket, but that's not this bot's mandate.

**Why first ITM/ATM strike?** Pure aggression. Deep OTM options require larger moves to profit; ITM/ATM options have higher delta and move more dollar-for-dollar with the stock.

**Why 35% stop?** Options are volatile. A 20% stop gets hit by normal intraday noise on small-caps. 35% gives the position room to breathe while capping the loss to a recoverable amount per trade.

**Why Benzinga for biotech?** Benzinga has the fastest biotech news coverage of any reasonably-priced API, with structured ticker tagging. Bloomberg/Refinitiv is overkill for this use case.

**Why the `ta` library for indicators?** It wraps TA-Lib without requiring the C build dependency, making pip install work everywhere. If you need more speed, swap with numpy manual implementations already present as fallbacks.
