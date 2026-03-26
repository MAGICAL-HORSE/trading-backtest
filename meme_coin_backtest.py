"""
Meme Coin Safety Analyzer – Backtest Harness
═══════════════════════════════════════════════════════════════
Simulates a stream of 200 synthetic tokens (mix of rugs + safe),
feeds each through the MemeCoinAnalyzer, records outcomes, and
shows how the signal weights and safety threshold evolve over time.

Synthetic token profiles are generated with realistic distributions:
  Rug  (60 % of base population) – dangerous on-chain characteristics
  Safe (40 % of base population) – healthy on-chain characteristics

Run:  python meme_coin_backtest.py
"""

from __future__ import annotations

import json
import math
import os
import random

from meme_coin_analyzer import (
    MemeCoinAnalyzer,
    TokenData,
    SIGNAL_NAMES,
    DB_PATH,
)

# ─────────────────────────────────────────────────────────────
# SIMULATION CONFIG
# ─────────────────────────────────────────────────────────────

SEED         = 42
N_TOKENS     = 200
RUG_FRACTION = 0.60   # 60 % of generated tokens are rugs

# Path for this run's learning DB (isolated from any production db)
BACKTEST_DB  = os.path.join(os.path.dirname(__file__), "learning_db.json")


# ─────────────────────────────────────────────────────────────
# SYNTHETIC TOKEN GENERATOR
# ─────────────────────────────────────────────────────────────

def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def generate_rug_token(rng: random.Random, idx: int) -> tuple[TokenData, str]:
    """Generate a token with rug-pull characteristics."""
    # Deliberate red flags with some noise to challenge the detector
    noise = rng.gauss(0, 1)

    liq_pct   = _clip(rng.gauss(10, 15), 0, 60)        # mostly unlocked
    liq_days  = int(_clip(rng.gauss(15, 20), 0, 180))   # short lock
    renounced = rng.random() < 0.10                      # rarely renounced
    top10_pct = _clip(rng.gauss(75, 15), 30, 99)        # highly concentrated
    has_mint  = rng.random() < 0.80
    has_pause = rng.random() < 0.70
    has_bl    = rng.random() < 0.65
    dev_rug   = rng.random() < 0.55
    dev_sell  = _clip(rng.gauss(70, 25), 0, 100)
    age_days  = int(_clip(rng.gauss(3, 4), 0, 30))
    soc_age   = int(_clip(rng.gauss(20, 30), 0, 200))
    fake_fol  = _clip(rng.gauss(60, 20), 0, 100)
    liq_usd   = _clip(rng.gauss(15_000, 10_000), 1_000, 50_000)
    vol_usd   = liq_usd * _clip(rng.gauss(40, 20), 5, 100)  # wash trading

    return TokenData(
        address               = f"0xRUG{idx:04d}",
        liquidity_locked_pct  = liq_pct,
        lock_duration_days    = liq_days,
        ownership_renounced   = renounced,
        top10_wallet_pct      = top10_pct,
        has_mint_function     = has_mint,
        has_pause_function    = has_pause,
        has_blacklist         = has_bl,
        dev_rugged_before     = dev_rug,
        dev_sell_pct_prior    = dev_sell,
        contract_age_days     = age_days,
        social_account_age_days= soc_age,
        fake_follower_pct     = fake_fol,
        volume_24h_usd        = vol_usd,
        liquidity_usd         = liq_usd,
    ), "rug"


def generate_safe_token(rng: random.Random, idx: int) -> tuple[TokenData, str]:
    """Generate a token with safety characteristics."""
    liq_pct   = _clip(rng.gauss(85, 10), 50, 100)
    liq_days  = int(_clip(rng.gauss(400, 150), 90, 1000))
    renounced = rng.random() < 0.80
    top10_pct = _clip(rng.gauss(25, 10), 5, 55)
    has_mint  = rng.random() < 0.10
    has_pause = rng.random() < 0.10
    has_bl    = rng.random() < 0.08
    dev_rug   = False
    dev_sell  = _clip(rng.gauss(10, 10), 0, 40)
    age_days  = int(_clip(rng.gauss(90, 60), 7, 400))
    soc_age   = int(_clip(rng.gauss(400, 150), 60, 1200))
    fake_fol  = _clip(rng.gauss(10, 8), 0, 35)
    liq_usd   = _clip(rng.gauss(300_000, 200_000), 50_000, 1_000_000)
    vol_usd   = liq_usd * _clip(rng.gauss(1.5, 0.8), 0.1, 8)

    return TokenData(
        address               = f"0xSAFE{idx:04d}",
        liquidity_locked_pct  = liq_pct,
        lock_duration_days    = liq_days,
        ownership_renounced   = renounced,
        top10_wallet_pct      = top10_pct,
        has_mint_function     = has_mint,
        has_pause_function    = has_pause,
        has_blacklist         = has_bl,
        dev_rugged_before     = dev_rug,
        dev_sell_pct_prior    = dev_sell,
        contract_age_days     = age_days,
        social_account_age_days= soc_age,
        fake_follower_pct     = fake_fol,
        volume_24h_usd        = vol_usd,
        liquidity_usd         = liq_usd,
    ), "safe"


def generate_token_stream(n: int, rug_frac: float,
                          seed: int) -> list[tuple[TokenData, str]]:
    rng = random.Random(seed)
    tokens: list[tuple[TokenData, str]] = []
    rug_idx = safe_idx = 0
    for _ in range(n):
        if rng.random() < rug_frac:
            tokens.append(generate_rug_token(rng, rug_idx))
            rug_idx += 1
        else:
            tokens.append(generate_safe_token(rng, safe_idx))
            safe_idx += 1
    return tokens


# ─────────────────────────────────────────────────────────────
# METRICS TRACKING
# ─────────────────────────────────────────────────────────────

class RunMetrics:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def record(self, token: TokenData, true_outcome: str,
               passed: bool, score: float) -> None:
        # Was the decision correct?
        # Correct = (blocked AND rug) OR (passed AND safe)
        blocked    = not passed
        true_rug   = true_outcome == "rug"
        correct    = (blocked and true_rug) or (passed and not true_rug)
        exposure   = passed and true_rug    # we traded it and it rugged

        self.results.append({
            "address":      token.address,
            "true_outcome": true_outcome,
            "passed":       passed,
            "score":        score,
            "correct":      correct,
            "rug_exposure": exposure,
        })

    def summary(self) -> dict:
        n          = len(self.results)
        rugs       = [r for r in self.results if r["true_outcome"] == "rug"]
        safes      = [r for r in self.results if r["true_outcome"] == "safe"]
        blocked_rugs  = [r for r in rugs  if not r["passed"]]
        passed_rugs   = [r for r in rugs  if     r["passed"]]   # rug exposure
        blocked_safes = [r for r in safes if not r["passed"]]   # false negatives
        passed_safes  = [r for r in safes if     r["passed"]]

        tpr = len(blocked_rugs) / len(rugs)   * 100 if rugs   else 0  # rug detection rate
        tnr = len(passed_safes) / len(safes)  * 100 if safes  else 0  # safe pass rate
        acc = sum(r["correct"] for r in self.results) / n     * 100

        return {
            "total":             n,
            "total_rugs":        len(rugs),
            "total_safes":       len(safes),
            "blocked_rugs":      len(blocked_rugs),
            "passed_rugs":       len(passed_rugs),   # rug exposure events
            "blocked_safes":     len(blocked_safes),
            "passed_safes":      len(passed_safes),
            "rug_detection_rate":tpr,
            "safe_pass_rate":    tnr,
            "accuracy":          acc,
            "rug_exposure_pct":  len(passed_rugs) / len(rugs) * 100 if rugs else 0,
        }


# ─────────────────────────────────────────────────────────────
# BACKTEST RUNNER
# ─────────────────────────────────────────────────────────────

def run_backtest() -> None:
    # Fresh DB each run
    if os.path.exists(BACKTEST_DB):
        os.remove(BACKTEST_DB)

    analyzer = MemeCoinAnalyzer(db_path=BACKTEST_DB)
    metrics  = RunMetrics()

    stream = generate_token_stream(N_TOKENS, RUG_FRACTION, SEED)

    print("═" * 62)
    print("  MEME COIN SAFETY ANALYZER  –  BACKTEST")
    print(f"  Tokens: {N_TOKENS}   Rug fraction: {RUG_FRACTION:.0%}   Seed: {SEED}")
    print("═" * 62)

    # Track weight snapshots at key intervals
    weight_snapshots: list[dict] = []

    for i, (token, true_outcome) in enumerate(stream):
        report = analyzer.analyze(token)

        # Only run post-mortem for tokens the system let through
        # (simulates: we only learn from trades we actually took)
        if report.passed:
            analyzer.record_outcome(
                token.address,
                outcome=true_outcome,
                signal_scores=report.signal_scores,
            )

        metrics.record(token, true_outcome, report.passed, report.weighted_score)

        # Progress snapshot every 25 tokens
        if (i + 1) % 25 == 0:
            s = metrics.summary()
            weight_snapshots.append({
                "after_n":   i + 1,
                "weights":   dict(analyzer.weights),
                "threshold": analyzer.min_score_threshold,
                "rug_exposure_pct": s["rug_exposure_pct"],
            })
            print(f"\n  [Token {i+1:3d}]  "
                  f"Rug exposure: {s['rug_exposure_pct']:5.1f}%  "
                  f"Accuracy: {s['accuracy']:5.1f}%  "
                  f"Threshold: {analyzer.min_score_threshold:.1f}")

    # ── Final Report ──────────────────────────────────────────
    s = metrics.summary()
    sep = "═" * 62
    sub = "─" * 62

    print(f"\n{sep}")
    print("  FINAL BACKTEST RESULTS")
    print(sep)
    print(f"  Total tokens analysed : {s['total']}")
    print(f"  True rugs             : {s['total_rugs']}  ({s['total_rugs']/s['total']*100:.0f}%)")
    print(f"  True safe             : {s['total_safes']}  ({s['total_safes']/s['total']*100:.0f}%)")
    print(sub)
    print(f"  Rugs blocked          : {s['blocked_rugs']} / {s['total_rugs']}")
    print(f"  Rugs that slipped thru: {s['passed_rugs']}  ← rug exposure events")
    print(f"  Safe tokens blocked   : {s['blocked_safes']}  ← false negatives")
    print(f"  Safe tokens passed    : {s['passed_safes']}")
    print(sub)
    print(f"  Rug detection rate    : {s['rug_detection_rate']:6.1f}%")
    print(f"  Safe pass rate        : {s['safe_pass_rate']:6.1f}%")
    print(f"  Overall accuracy      : {s['accuracy']:6.1f}%")
    print(f"  Rug exposure          : {s['rug_exposure_pct']:6.1f}%  (goal: 0%)")
    print(sep)

    # ── Weight evolution ──────────────────────────────────────
    print("\n  SIGNAL WEIGHT EVOLUTION  (initial → final)")
    print(sub)
    initial_w = {n: 1.0 for n in SIGNAL_NAMES}
    final_w   = analyzer.weights
    for name in SIGNAL_NAMES:
        iv = initial_w[name]
        fv = final_w[name]
        direction = "↑" if fv > iv else ("↓" if fv < iv else "=")
        bar = "▓" * int(fv * 2)
        print(f"  {name:<24} {iv:.3f} → {fv:.3f}  {direction}  {bar}")
    print(sub)
    print(f"  Min score threshold   : {analyzer.min_score_threshold:.1f}")
    print(sep)

    # ── Weight progression table ──────────────────────────────
    print("\n  RUG EXPOSURE OVER TIME")
    print(sub)
    for snap in weight_snapshots:
        print(f"  After {snap['after_n']:3d} tokens: "
              f"exposure {snap['rug_exposure_pct']:5.1f}%  "
              f"threshold {snap['threshold']:.1f}")
    print(sep)

    analyzer.print_stats()

    # Save final results to JSON for inspection
    results_path = os.path.join(os.path.dirname(__file__), "backtest_meme_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "n_tokens": N_TOKENS,
                "rug_fraction": RUG_FRACTION,
                "seed": SEED,
            },
            "summary": s,
            "weight_snapshots": weight_snapshots,
            "final_weights": final_w,
            "final_threshold": analyzer.min_score_threshold,
        }, f, indent=2)
    print(f"\n  Backtest results → backtest_meme_results.json")
    print(f"  Learning database → learning_db.json")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_backtest()
