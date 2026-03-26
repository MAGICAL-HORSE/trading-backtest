"""
Parameter Sweep Optimizer  —  Fast NumPy Edition
═════════════════════════════════════════════════════════════════════
Tests every combination in the grid, finds configs that hit 95%+ IS
return, then runs the best configs on out-of-sample data to expose
overfitting.

IMPORTANT — OVERFITTING WARNING
────────────────────────────────
Optimising parameters on the same data used to evaluate performance
guarantees apparent "good" results with zero forward predictive value.
This script proves that by running an OOS test on a completely
different random seed (= different synthetic market).
"""

import warnings
warnings.filterwarnings("ignore")

import math
import itertools
import time as time_module
from datetime import datetime, time, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

TRADING_MINS = 390
TRADING_DAYS = 252
RISK_FREE    = 0.045
INITIAL_CAP  = 300.0
TARGET_RETURN = 95.0   # % — what the user asked for

BASE_PARAMS = {
    "SPY":  {"S0": 572.0, "vol": 0.14, "drift": 0.10},
    "QQQ":  {"S0": 488.0, "vol": 0.18, "drift": 0.12},
    "NVDA": {"S0": 124.0, "vol": 0.58, "drift": 0.15},
}
IS_SEEDS  = {"SPY": 42,  "QQQ": 77,  "NVDA": 13}   # in-sample
OOS_SEEDS = {"SPY": 99,  "QQQ": 200, "NVDA": 301}  # out-of-sample (different random paths)


# ─────────────────────────────────────────────────────────────
# FAST BLACK-SCHOLES  (A&S polynomial approx for norm.cdf)
# ─────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    """Fast normal CDF via Abramowitz & Stegun. Max error < 7.5e-8."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    p = (0.319381530
         + t * (-0.356563782
         + t * (1.781477937
         + t * (-1.821255978
         + t * 1.330274429))))
    v = 1.0 - math.exp(-0.5 * x * x) / 2.5066282746 * p * t
    return v if x >= 0 else 1.0 - v


def _bs(S: float, K: float, T: float, sigma: float, opt: str) -> float:
    if T <= 1e-9:
        return max((S - K) if opt == "call" else (K - S), 0.001)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    disc = math.exp(-RISK_FREE * T)
    if opt == "call":
        return max(S * _ncdf(d1) - K * disc * _ncdf(d2), 0.001)
    return max(K * disc * _ncdf(-d2) - S * _ncdf(-d1), 0.001)


def _spread(spot: float, Kl: float, Ks: float, T: float,
            sigma: float, opt: str, width: float) -> float:
    """MTM value of a vertical spread. Bounded [0, width]."""
    return max(min(_bs(spot, Kl, T, sigma, opt) - _bs(spot, Ks, T, sigma, opt), width), 0.0)


def _find_entry(spot: float, signal: str, T: float, sigma: float,
                width: float, max_cost: float):
    """Walk OTM until spread debit fits budget. Returns (Kl, Ks, debit) or (0,0,0)."""
    atm   = round(spot)
    steps = max(int(spot * 0.06), 6)
    for step in range(steps + 1):
        if signal == "call":
            Kl = float(atm + step)
            Ks = Kl + width
        else:
            Kl = float(atm - step)
            Ks = Kl - width
            if Ks <= 0:
                break
        debit = _spread(spot, Kl, Ks, T, sigma, signal, width)
        cost  = debit * 100.0
        if 0.001 < cost <= max_cost:
            return Kl, Ks, debit
    return 0.0, 0.0, 0.0


# ─────────────────────────────────────────────────────────────
# DATA GENERATION  (GBM with intraday vol scaling)
# ─────────────────────────────────────────────────────────────

def _gen(S0, vol, drift, seed, period_days):
    rng = np.random.default_rng(seed)
    dt  = 5.0 / (TRADING_DAYS * TRADING_MINS)
    dpb = drift * dt
    vpb = vol * math.sqrt(dt)

    end   = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=period_days)

    rows, tss = [], []
    price = S0
    day   = start
    while day <= end:
        if day.weekday() >= 5:
            day += timedelta(days=1)
            continue
        so = datetime.combine(day, time(9, 30))
        sc = datetime.combine(day, time(16, 0))
        bt = so
        while bt < sc - timedelta(minutes=5):
            ms = (bt - so).seconds // 60
            mc = (sc - bt).seconds // 60
            vs = 1.6 if (ms < 30 or mc < 30) else 1.0
            np_ = max(price * math.exp(dpb + vpb * vs * rng.standard_normal()), 0.01)
            tss.append(bt)
            rows.append(np_)
            price = np_
            bt += timedelta(minutes=5)
        day += timedelta(days=1)

    return np.array(tss, dtype="datetime64[s]"), np.array(rows, dtype=np.float64)


def _add_ema_signals(closes: np.ndarray, fast: int, slow: int):
    """Vectorised EMA cross signals. Returns (cross_up, cross_down) bool arrays."""
    n   = len(closes)
    ef  = np.empty(n); ef[0] = closes[0]
    es  = np.empty(n); es[0] = closes[0]
    af  = 2.0 / (fast + 1)
    as_ = 2.0 / (slow + 1)
    for i in range(1, n):
        ef[i] = closes[i] * af + ef[i-1] * (1 - af)
        es[i] = closes[i] * as_ + es[i-1] * (1 - as_)
    above   = ef > es
    cup  = above & ~np.roll(above, 1);  cup[0]  = False
    cdn  = ~above & np.roll(above, 1);  cdn[0]  = False
    return cup, cdn


def build_dataset(tickers, seeds, ema_fast, ema_slow, period_days=180):
    """Returns arrays dict: {tkr: {ts, close, cup, cdn, t_frac}} + times."""
    out = {}
    for tkr in tickers:
        p = BASE_PARAMS[tkr]
        ts, cl = _gen(p["S0"], p["vol"], p["drift"], seeds[tkr], period_days)
        cup, cdn = _add_ema_signals(cl, ema_fast, ema_slow)

        # Pre-compute T (fraction of year remaining until 4 PM) for each bar
        # ts is np.datetime64[s] — compute minutes-to-close
        # We extract time-of-day in seconds since midnight
        ts_secs = ts.astype("datetime64[s]").astype(np.int64) % 86400  # secs since midnight
        close_secs = 16 * 3600  # 16:00:00
        mins_left = np.maximum((close_secs - ts_secs) / 60.0, 1.0)
        t_frac = mins_left / (TRADING_DAYS * TRADING_MINS)

        # Entry/exit time gates as secs since midnight
        out[tkr] = {
            "ts":    ts,
            "close": cl,
            "cup":   cup,
            "cdn":   cdn,
            "t":     t_frac,
            "tsecs": ts_secs,
        }

    # All tickers share identical timestamps (same synthetic calendar)
    return out


# ─────────────────────────────────────────────────────────────
# FAST BACKTEST ENGINE  (NumPy arrays, no pandas row access)
# ─────────────────────────────────────────────────────────────

@dataclass
class Cfg:
    ema_fast:       int   = 9
    ema_slow:       int   = 20
    width:          float = 1.0
    max_cost:       float = 30.0
    tp:             float = 0.80   # take-profit fraction of max gain
    sl:             float = 0.50   # stop-loss: exit when value <= debit*(1-sl)
    entry_secs:     int   = 9*3600+45*60   # 9:45 AM in secs since midnight
    exit_secs:      int   = 15*3600+30*60  # 3:30 PM
    max_dt_week:    int   = 3


def week_num(ts_s: int) -> int:
    """Convert Unix seconds to ISO week number (fast integer math)."""
    # ts_s is seconds since epoch (numpy int64)
    # ISO week = int(ts_s // 604800) + offset; good enough for counting
    return int(ts_s) // (7 * 86400)


def run_fast(arrs: dict, sigmas: dict, cfg: Cfg) -> tuple:
    """
    Ultra-fast NumPy backtest. No pandas. Returns (final_cap, return_pct, win_rate, n_trades).
    """
    tickers   = list(arrs.keys())
    n_bars    = len(arrs[tickers[0]]["ts"])

    capital   = INITIAL_CAP
    all_pnl   = []

    # Per-ticker open position state
    pos = {t: None for t in tickers}   # None or dict

    # PDT: count opens per ISO week (int week number)
    weekly_opens = {}

    for i in range(n_bars):
        # Time values for this bar (same for all tickers, same calendar)
        ts_s   = int(arrs[tickers[0]]["ts"][i].astype("datetime64[s]").astype(np.int64))
        tsecs  = int(arrs[tickers[0]]["tsecs"][i])
        wk     = week_num(ts_s)
        used   = weekly_opens.get(wk, 0)

        for tkr in tickers:
            arr   = arrs[tkr]
            spot  = arr["close"][i]
            T     = arr["t"][i]
            sigma = sigmas[tkr]
            p     = pos[tkr]

            # A. MANAGE OPEN POSITION
            if p is not None:
                cv    = _spread(spot, p["Kl"], p["Ks"], T, sigma, p["opt"], cfg.width)
                ed    = p["ed"]
                mg    = cfg.width - ed
                tp_th = ed + cfg.tp * mg
                sl_th = ed * (1.0 - cfg.sl)

                hit = (cv >= tp_th) or (cv <= sl_th) or (tsecs >= cfg.exit_secs)
                if hit:
                    pnl = (cv - ed) * 100.0 * p["n"]
                    capital = max(capital + p["cost"] + pnl, 0.0)
                    all_pnl.append(pnl)
                    pos[tkr] = None

            # B. LOOK FOR ENTRY
            else:
                if tsecs < cfg.entry_secs or tsecs >= cfg.exit_secs:
                    continue
                if used >= cfg.max_dt_week:
                    continue
                if capital < 5.0:
                    continue

                cup = arr["cup"][i]
                cdn = arr["cdn"][i]
                if not cup and not cdn:
                    continue
                signal = "call" if cup else "put"

                budget = min(cfg.max_cost, capital * 0.10)
                Kl, Ks, debit = _find_entry(spot, signal, T, sigma, cfg.width, budget)
                if Kl == 0.0:
                    continue

                c1 = debit * 100.0
                n  = max(1, int(budget // c1))
                tc = n * c1
                if tc > capital:
                    continue

                capital -= tc
                weekly_opens[wk] = used + 1
                used += 1
                pos[tkr] = {"opt": signal, "Kl": Kl, "Ks": Ks,
                             "ed": debit, "n": n, "cost": tc}

    # Force-close any open positions at end
    last_i = n_bars - 1
    for tkr, p in pos.items():
        if p is None:
            continue
        spot = arrs[tkr]["close"][last_i]
        cv   = _spread(spot, p["Kl"], p["Ks"], 1e-9, sigmas[tkr], p["opt"], cfg.width)
        pnl  = (cv - p["ed"]) * 100.0 * p["n"]
        capital = max(capital + p["cost"] + pnl, 0.0)
        all_pnl.append(pnl)

    n   = len(all_pnl)
    wr  = sum(1 for x in all_pnl if x > 0) / n * 100 if n else 0.0
    ret = (capital / INITIAL_CAP - 1.0) * 100.0
    return capital, ret, wr, n


# ─────────────────────────────────────────────────────────────
# PARAMETER GRID
# ─────────────────────────────────────────────────────────────

GRID = {
    "ema_fast":   [5, 9, 12],
    "ema_slow":   [15, 20, 26, 30],
    "width":      [1.0, 2.0, 3.0, 5.0],
    "max_cost":   [20.0, 30.0, 50.0],
    "tp":         [0.60, 0.70, 0.80, 0.90],
    "sl":         [0.30, 0.40, 0.50, 0.60],
    "entry_secs": [9*3600+45*60, 10*3600, 10*3600+30*60],
    "exit_secs":  [14*3600+30*60, 15*3600, 15*3600+30*60],
    "max_dt_week":[3],
    "tickers":    [
        ("NVDA",),
        ("SPY", "QQQ", "NVDA"),
        ("QQQ", "NVDA"),
    ],
}

ENTRY_LABELS = {9*3600+45*60: "09:45", 10*3600: "10:00", 10*3600+30*60: "10:30"}
EXIT_LABELS  = {14*3600+30*60: "14:30", 15*3600: "15:00", 15*3600+30*60: "15:30"}


def all_configs():
    keys   = list(GRID.keys())
    vals   = list(GRID.values())
    for combo in itertools.product(*vals):
        d = dict(zip(keys, combo))
        if d["ema_fast"] >= d["ema_slow"]:       continue
        if d["entry_secs"] >= d["exit_secs"]:    continue
        yield d


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PARAMETER SWEEP — 0DTE Vertical Spreads (NumPy fast)  ║")
    print(f"║   Target: ≥{TARGET_RETURN:.0f}% IS return  │  Then OOS overfitting test ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    combos = list(all_configs())
    print(f"[*] Grid size: {len(combos):,} combinations\n")

    # ── Pre-generate all data for all unique (tickers, ema_fast, ema_slow) ──
    print("[1/4] Pre-generating synthetic price data …")
    t0 = time_module.time()

    is_cache  = {}   # (tickers, ef, es) → arrays
    oos_cache = {}

    unique_keys = set()
    for d in combos:
        unique_keys.add((tuple(d["tickers"]), d["ema_fast"], d["ema_slow"]))

    for (tickers, ef, es) in unique_keys:
        is_cache[(tickers, ef, es)]  = build_dataset(tickers, IS_SEEDS,  ef, es)
        oos_cache[(tickers, ef, es)] = build_dataset(tickers, OOS_SEEDS, ef, es)

    print(f"    {len(unique_keys)} unique datasets generated in {time_module.time()-t0:.1f}s\n")

    # ── Run sweep ──────────────────────────────────────────────
    print("[2/4] Running sweep …")
    t1   = time_module.time()
    rows = []
    hits_above_target = []

    for idx, d in enumerate(combos, 1):
        tickers = tuple(d["tickers"])
        key     = (tickers, d["ema_fast"], d["ema_slow"])
        arrs    = is_cache[key]
        sigmas  = {t: BASE_PARAMS[t]["vol"] for t in tickers}

        cfg = Cfg(ema_fast=d["ema_fast"], ema_slow=d["ema_slow"],
                  width=d["width"], max_cost=d["max_cost"],
                  tp=d["tp"], sl=d["sl"],
                  entry_secs=d["entry_secs"], exit_secs=d["exit_secs"],
                  max_dt_week=d["max_dt_week"])

        cap, ret, wr, n = run_fast(arrs, sigmas, cfg)

        row = {
            "tickers":    ",".join(tickers),
            "ema_fast":   d["ema_fast"],
            "ema_slow":   d["ema_slow"],
            "width":      d["width"],
            "max_cost":   d["max_cost"],
            "tp":         d["tp"],
            "sl":         d["sl"],
            "entry":      ENTRY_LABELS[d["entry_secs"]],
            "exit":       EXIT_LABELS[d["exit_secs"]],
            "n_trades":   n,
            "win_rate":   round(wr, 1),
            "final_bal":  round(cap, 2),
            "IS_ret_%":   round(ret, 1),
            "OOS_ret_%":  None,
        }
        rows.append(row)

        if ret >= TARGET_RETURN:
            hits_above_target.append((idx, row, cfg, key))

        if idx % 5000 == 0:
            elapsed = time_module.time() - t1
            best    = max(r["IS_ret_%"] for r in rows)
            print(f"    [{idx:>6}/{len(combos):,}]  {elapsed:.0f}s  "
                  f"best IS: {best:+.1f}%  hits≥{TARGET_RETURN:.0f}%: {len(hits_above_target)}")

    elapsed_sweep = time_module.time() - t1
    print(f"    Done. {len(combos):,} combos in {elapsed_sweep:.1f}s "
          f"({elapsed_sweep/len(combos)*1000:.1f}ms/combo)\n")

    # ── Build results DataFrame ────────────────────────────────
    df = pd.DataFrame(rows).sort_values("IS_ret_%", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df.to_csv("sweep_results.csv", index=False)
    print("    Saved → sweep_results.csv")

    # ── OOS test on top 15 configs ─────────────────────────────
    print("\n[3/4] Out-of-sample test on top 15 configs …")
    oos_rows = []
    for _, row in df.head(15).iterrows():
        tickers = tuple(row["tickers"].split(","))
        ef = int(row["ema_fast"]); es = int(row["ema_slow"])
        key = (tickers, ef, es)

        entry_secs_map = {v: k for k, v in ENTRY_LABELS.items()}
        exit_secs_map  = {v: k for k, v in EXIT_LABELS.items()}

        cfg = Cfg(ema_fast=ef, ema_slow=es,
                  width=float(row["width"]), max_cost=float(row["max_cost"]),
                  tp=float(row["tp"]), sl=float(row["sl"]),
                  entry_secs=entry_secs_map[row["entry"]],
                  exit_secs=exit_secs_map[row["exit"]],
                  max_dt_week=int(row["max_dt_week"]) if "max_dt_week" in row else 3)

        oos_arrs   = oos_cache[key]
        oos_sigmas = {t: BASE_PARAMS[t]["vol"] for t in tickers}
        _, oos_ret, oos_wr, oos_n = run_fast(oos_arrs, oos_sigmas, cfg)

        oos_rows.append({
            "rank":       int(row["rank"]),
            "tickers":    row["tickers"],
            "IS_ret_%":   row["IS_ret_%"],
            "OOS_ret_%":  round(oos_ret, 1),
            "IS_WR%":     row["win_rate"],
            "OOS_WR%":    round(oos_wr, 1),
            "IS_trades":  int(row["n_trades"]),
            "OOS_trades": oos_n,
            "survived":   "✓" if oos_ret > 0 else "✗",
        })

    oos_df = pd.DataFrame(oos_rows)

    # ── Print final report ─────────────────────────────────────
    print("\n[4/4] Results\n")

    sep = "═" * 68
    print(sep)
    print("  TOP 15 IN-SAMPLE CONFIGURATIONS")
    print(sep)
    show_cols = ["rank","tickers","width","max_cost","tp","sl",
                 "entry","n_trades","win_rate","IS_ret_%"]
    print(df[show_cols].head(15).to_string(index=False))

    print(f"\n{sep}")
    print("  IN-SAMPLE  vs  OUT-OF-SAMPLE  (different random seed)")
    print("  If IS 'edge' were real, OOS should also be profitable.")
    print(sep)
    print(oos_df.to_string(index=False))

    n_survived  = (oos_df["OOS_ret_%"] > 0).sum()
    avg_is      = oos_df["IS_ret_%"].mean()
    avg_oos     = oos_df["OOS_ret_%"].mean()
    degradation = (1 - avg_oos / avg_is) * 100 if avg_is != 0 else 100

    print(f"\n  Survived OOS (positive return): {n_survived}/15")
    print(f"  Avg IS return:  {avg_is:+.1f}%")
    print(f"  Avg OOS return: {avg_oos:+.1f}%")
    print(f"  Return degradation IS→OOS: {degradation:.0f}%")

    # ── Best config detail ─────────────────────────────────────
    best_row = df.iloc[0]
    print(f"\n{sep}")
    print("  BEST IN-SAMPLE CONFIG (detailed)")
    print(sep)
    for col in ["tickers","ema_fast","ema_slow","width","max_cost",
                "tp","sl","entry","exit","n_trades","win_rate","IS_ret_%"]:
        if col in best_row.index:
            print(f"    {col:<16}: {best_row[col]}")
    print(f"    Final balance  : ${INITIAL_CAP * (1 + best_row['IS_ret_%']/100):.2f}")

    best_oos_row = oos_df.iloc[0]
    print(f"    OOS return     : {best_oos_row['OOS_ret_%']:+.1f}%  "
          f"({'PROFITABLE' if best_oos_row['OOS_ret_%'] > 0 else 'LOSING'} on unseen data)")

    print(f"\n{sep}")
    print("  VERDICT")
    print(sep)

    total_above = len(hits_above_target)
    best_is_all = df.iloc[0]["IS_ret_%"]

    print(f"  Combinations tested  : {len(combos):,}")
    print(f"  Configs with IS ≥ {TARGET_RETURN:.0f}%: {total_above}")
    print(f"  Best IS return found : {best_is_all:+.1f}%")
    print(f"  Best OOS return      : {oos_df['OOS_ret_%'].max():+.1f}%")
    print()
    if best_is_all >= TARGET_RETURN:
        print(f"  ✓ Reached target {TARGET_RETURN:.0f}%+ in-sample.")
    else:
        print(f"  ✗ Best IS: {best_is_all:.1f}% — target {TARGET_RETURN:.0f}% not reached.")

    if degradation > 80:
        print()
        print("  ⚠  SEVERE OVERFITTING (IS→OOS degradation > 80%).")
        print("     The winning parameters are curve-fitted to noise.")
        print("     They have NO predictive value on real market data.")
        print()
        print("  What would ACTUALLY grow $300:")
        print("    • A strategy with a thesis (earnings surprise, IV crush,")
        print("      sector rotation) — not just EMA crosses on random data")
        print("    • Walk-forward validation: optimise on year 1, test on year 2")
        print("    • IS→OOS degradation < 40% is the minimum bar for a real edge")
    elif degradation < 40:
        print()
        print("  Reasonable IS→OOS persistence. Validate on REAL data next.")
    print(sep)

    total_time = time_module.time() - t0
    print(f"\n  Total runtime: {total_time:.1f}s")
