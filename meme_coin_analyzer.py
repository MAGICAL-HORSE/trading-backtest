"""
Meme Coin Safety Analyzer
═══════════════════════════════════════════════════════════════
Scores 8 on-chain / social signals for each token (0–10 each).
Learns from every trade outcome via post-mortem weight updates.
Goal: iteratively drive rug-pull exposure toward 0%.

Signals
───────
1.  liquidity_lock       – lock duration + % locked
2.  ownership_renounced  – on-chain renouncement verified
3.  wallet_concentration – top-10 wallets hold < X% of supply
4.  dangerous_functions  – mint / pause / blacklist absent
5.  dev_history          – dev wallet clean on prior tokens
6.  contract_age         – time since deployment
7.  social_authenticity  – account age + follower legitimacy
8.  volume_liq_ratio     – trading volume vs. liquidity (wash-trade guard)

Learning Rules
──────────────
• After a RUG: weight of every signal that *failed to flag danger*
  (scored ≥ 5 yet token rugged) is increased by WEIGHT_BUMP_PCT.
• After a SAFE outcome: weight of every signal that *correctly
  flagged safety* (scored ≥ 5) is reinforced by WEIGHT_REINFORCE_PCT.
• Minimum safety score threshold starts at MIN_SCORE_INITIAL (60).
  – Raised by SCORE_FLOOR_BUMP after every rug slip.
  – Lowered by SCORE_FLOOR_RELAX only when the last-50-token
    safety-detection win rate exceeds WIN_RATE_THRESHOLD (95 %).
• All weight changes and outcomes are appended to learning_db.json.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SIGNAL_NAMES: list[str] = [
    "liquidity_lock",
    "ownership_renounced",
    "wallet_concentration",
    "dangerous_functions",
    "dev_history",
    "contract_age",
    "social_authenticity",
    "volume_liq_ratio",
]

MIN_SCORE_INITIAL: float = 60.0   # minimum weighted safety score to allow a trade
SCORE_FLOOR_BUMP: float  = 2.0    # raise floor by this many points on each rug
SCORE_FLOOR_RELAX: float = 0.5    # lower floor by this many points when win-rate is high

WEIGHT_BUMP_PCT: float      = 15.0  # % increase on missed-signal weights after rug
WEIGHT_REINFORCE_PCT: float =  5.0  # % increase on correct-signal weights after safe
WEIGHT_MAX: float           = 10.0  # cap per-signal weight to prevent runaway
WEIGHT_MIN: float           =  0.5  # floor per-signal weight

WIN_RATE_THRESHOLD: float = 95.0   # % correct safety predictions over last 50 tokens
WINDOW_SIZE: int          = 50     # rolling window for win-rate calculation

DB_PATH: str = os.path.join(os.path.dirname(__file__), "learning_db.json")


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class TokenData:
    """
    All observable data about a token at analysis time.

    Fields
    ──────
    address              – contract address (used as unique key)
    liquidity_locked_pct – % of liquidity locked (0–100)
    lock_duration_days   – how many days the lock lasts
    ownership_renounced  – True if owner() == 0x000...dead/null
    top10_wallet_pct     – % of supply held by top-10 wallets (0–100)
    has_mint_function    – contract exposes mint()
    has_pause_function   – contract exposes pause()
    has_blacklist        – contract exposes blacklist / block functions
    dev_rugged_before    – True if deployer address rugged a prior token
    dev_sell_pct_prior   – % of holdings dev sold on previous tokens (0–100)
    contract_age_days    – days since contract deployment
    social_account_age_days – oldest linked social account age in days
    fake_follower_pct    – estimated % of followers that are bots (0–100)
    volume_24h_usd       – 24-hour trading volume in USD
    liquidity_usd        – current on-chain liquidity in USD
    """
    address: str
    liquidity_locked_pct: float      = 0.0
    lock_duration_days: int          = 0
    ownership_renounced: bool        = False
    top10_wallet_pct: float          = 100.0
    has_mint_function: bool          = True
    has_pause_function: bool         = True
    has_blacklist: bool              = True
    dev_rugged_before: bool          = True
    dev_sell_pct_prior: float        = 100.0
    contract_age_days: int           = 0
    social_account_age_days: int     = 0
    fake_follower_pct: float         = 100.0
    volume_24h_usd: float            = 0.0
    liquidity_usd: float             = 1.0   # avoid /0


@dataclass
class SignalScores:
    liquidity_lock: float       = 0.0
    ownership_renounced: float  = 0.0
    wallet_concentration: float = 0.0
    dangerous_functions: float  = 0.0
    dev_history: float          = 0.0
    contract_age: float         = 0.0
    social_authenticity: float  = 0.0
    volume_liq_ratio: float     = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def values(self) -> list[float]:
        return list(asdict(self).values())


@dataclass
class SafetyReport:
    address: str
    timestamp: str
    signal_scores: dict[str, float]
    signal_weights: dict[str, float]
    weighted_score: float          # 0–100
    min_score_threshold: float
    passed: bool
    rejection_reason: str | None   = None

    def display(self) -> None:
        sep = "═" * 62
        sub = "─" * 62
        print(f"\n{sep}")
        print(f"  MEME COIN SAFETY REPORT  –  {self.address[:20]}…")
        print(sep)
        for name in SIGNAL_NAMES:
            score  = self.signal_scores[name]
            weight = self.signal_weights[name]
            bar    = "█" * int(score) + "░" * (10 - int(score))
            print(f"  {name:<24} [{bar}] {score:4.1f}/10  w={weight:.2f}")
        print(sub)
        print(f"  Weighted Safety Score : {self.weighted_score:.1f} / 100")
        print(f"  Minimum Threshold     : {self.min_score_threshold:.1f}")
        status = "✓ PASSED" if self.passed else "✗ BLOCKED"
        print(f"  Decision              : {status}")
        if self.rejection_reason:
            print(f"  Reason                : {self.rejection_reason}")
        print(sep)


# ─────────────────────────────────────────────────────────────
# SIGNAL SCORING FUNCTIONS  (pure, deterministic)
# ─────────────────────────────────────────────────────────────

def _score_liquidity_lock(pct: float, days: int) -> float:
    """
    Score based on lock percentage AND duration.

    Liquidity % contributes up to 6 points, duration up to 4 points.
    Thresholds:  ≥80 % locked = full pct points;  ≥365 days = full dur points.
    """
    pct_score = min(pct / 80.0, 1.0) * 6.0
    dur_score = min(days / 365.0, 1.0) * 4.0
    return round(pct_score + dur_score, 2)


def _score_ownership_renounced(renounced: bool) -> float:
    """Binary: 10 if renounced, 0 if not."""
    return 10.0 if renounced else 0.0


def _score_wallet_concentration(top10_pct: float) -> float:
    """
    Inverse sigmoid on top-10 wallet concentration.

    <20 % → near 10.   >80 % → near 0.
    Uses a logistic curve centred at 50 % concentration.
    """
    x = (top10_pct - 50.0) / 10.0          # scale so centre=50% is x=0
    score = 10.0 / (1.0 + math.exp(x))
    return round(max(min(score, 10.0), 0.0), 2)


def _score_dangerous_functions(has_mint: bool, has_pause: bool,
                                has_blacklist: bool) -> float:
    """
    Deduct ~3.33 points per dangerous function present.
    All three absent → 10.  All three present → 0.
    """
    danger_count = sum([has_mint, has_pause, has_blacklist])
    return round(10.0 * (1.0 - danger_count / 3.0), 2)


def _score_dev_history(rugged_before: bool, sell_pct_prior: float) -> float:
    """
    0 if dev has a prior rug on record.
    Otherwise inverse linear on % sold on prior tokens:
      0 % sold → 10,   100 % sold → 0.
    """
    if rugged_before:
        return 0.0
    return round(10.0 * (1.0 - sell_pct_prior / 100.0), 2)


def _score_contract_age(age_days: int) -> float:
    """
    Log-scaled score.  Very new contracts (< 1 day) score 0.
    Contracts ≥ 30 days approach 10.
    """
    if age_days <= 0:
        return 0.0
    score = 10.0 * math.log1p(age_days) / math.log1p(30)
    return round(min(score, 10.0), 2)


def _score_social_authenticity(account_age_days: int,
                                fake_follower_pct: float) -> float:
    """
    Account age contributes up to 5 points (≥180 days → full).
    Fake-follower % deducts up to 5 points.
    """
    age_score   = min(account_age_days / 180.0, 1.0) * 5.0
    auth_score  = (1.0 - fake_follower_pct / 100.0) * 5.0
    return round(age_score + auth_score, 2)


def _score_volume_liq_ratio(volume_24h: float, liquidity: float) -> float:
    """
    Wash-trading proxy.  Healthy tokens have volume/liquidity < 5×.
    Ratio > 50× is almost certainly wash-traded.

    Score 10 for ratio < 0.1 (very low volume vs liquidity).
    Score drops as ratio climbs, reaching 0 at ratio ≥ 50.
    """
    if liquidity <= 0:
        return 0.0
    ratio = volume_24h / max(liquidity, 1.0)
    if ratio < 0.1:
        return 10.0
    score = 10.0 * math.exp(-ratio / 10.0)
    return round(max(min(score, 10.0), 0.0), 2)


# ─────────────────────────────────────────────────────────────
# LEARNING DATABASE  (JSON persistence)
# ─────────────────────────────────────────────────────────────

def _default_db() -> dict[str, Any]:
    return {
        "weights": {name: 1.0 for name in SIGNAL_NAMES},
        "min_score_threshold": MIN_SCORE_INITIAL,
        "outcomes": [],           # list of {address, outcome, signal_scores, timestamp}
        "weight_log": [],         # list of {timestamp, event, adjustments}
        "total_analyzed": 0,
        "total_rugged": 0,
        "total_safe": 0,
        "total_blocked": 0,
    }


def load_db(path: str = DB_PATH) -> dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return _default_db()


def save_db(db: dict[str, Any], path: str = DB_PATH) -> None:
    with open(path, "w") as f:
        json.dump(db, f, indent=2)


# ─────────────────────────────────────────────────────────────
# CORE ANALYZER
# ─────────────────────────────────────────────────────────────

class MemeCoinAnalyzer:
    """
    Stateful analyzer that scores tokens and learns from outcomes.

    Usage
    ─────
    analyzer = MemeCoinAnalyzer()
    report   = analyzer.analyze(token_data)
    if report.passed:
        # … trade …
        analyzer.record_outcome(token_data.address, outcome="rug")
        # or
        analyzer.record_outcome(token_data.address, outcome="safe")
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._db     = load_db(db_path)

    # ── Public API ───────────────────────────────────────────

    def analyze(self, token: TokenData) -> SafetyReport:
        """Score all signals, compute weighted safety score, return report."""
        scores  = self._score_all_signals(token)
        weights = dict(self._db["weights"])
        ws      = self._weighted_score(scores.as_dict(), weights)
        thresh  = self._db["min_score_threshold"]
        passed  = ws >= thresh

        rejection = None
        if not passed:
            rejection = f"score {ws:.1f} < threshold {thresh:.1f}"

        self._db["total_analyzed"] += 1
        if not passed:
            self._db["total_blocked"] += 1
        save_db(self._db, self.db_path)

        return SafetyReport(
            address            = token.address,
            timestamp          = _now(),
            signal_scores      = scores.as_dict(),
            signal_weights     = weights,
            weighted_score     = ws,
            min_score_threshold= thresh,
            passed             = passed,
            rejection_reason   = rejection,
        )

    def record_outcome(self, address: str, outcome: str,
                       signal_scores: dict[str, float] | None = None) -> None:
        """
        Record the real-world outcome of a token and run a post-mortem.

        Parameters
        ──────────
        address       – token contract address
        outcome       – "rug" or "safe"
        signal_scores – the scores from the original analysis (looked up from
                        outcomes list if omitted, otherwise pass report.signal_scores)
        """
        if outcome not in ("rug", "safe"):
            raise ValueError(f"outcome must be 'rug' or 'safe', got {outcome!r}")

        # Find stored scores if not supplied
        if signal_scores is None:
            for rec in reversed(self._db["outcomes"]):
                if rec["address"] == address:
                    signal_scores = rec["signal_scores"]
                    break
        if signal_scores is None:
            raise ValueError(f"No stored analysis found for {address}")

        # Record outcome
        self._db["outcomes"].append({
            "address":       address,
            "outcome":       outcome,
            "signal_scores": signal_scores,
            "timestamp":     _now(),
        })
        if outcome == "rug":
            self._db["total_rugged"] += 1
        else:
            self._db["total_safe"] += 1

        # Run post-mortem and adjust weights / threshold
        self._post_mortem(address, outcome, signal_scores)
        self._maybe_adjust_threshold()
        save_db(self._db, self.db_path)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._db["weights"])

    @property
    def min_score_threshold(self) -> float:
        return self._db["min_score_threshold"]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_analyzed":    self._db["total_analyzed"],
            "total_rugged":      self._db["total_rugged"],
            "total_safe":        self._db["total_safe"],
            "total_blocked":     self._db["total_blocked"],
            "min_score_threshold": self._db["min_score_threshold"],
            "win_rate_last_50":  self._win_rate_last_n(WINDOW_SIZE),
        }

    def print_stats(self) -> None:
        s   = self.stats
        sep = "─" * 50
        print(f"\n{sep}")
        print("  ANALYZER STATISTICS")
        print(sep)
        print(f"  Tokens analyzed   : {s['total_analyzed']}")
        print(f"  Tokens blocked    : {s['total_blocked']}")
        print(f"  Rugs recorded     : {s['total_rugged']}")
        print(f"  Safe recorded     : {s['total_safe']}")
        print(f"  Min score threshold: {s['min_score_threshold']:.1f}")
        print(f"  Win rate (last 50) : {s['win_rate_last_50']:.1f}%")
        print(f"\n  Signal Weights:")
        for name, w in self._db["weights"].items():
            bar = "▓" * int(w * 2)
            print(f"    {name:<24} {w:.3f}  {bar}")
        print(sep)

    # ── Internal Scoring ─────────────────────────────────────

    def _score_all_signals(self, t: TokenData) -> SignalScores:
        return SignalScores(
            liquidity_lock      = _score_liquidity_lock(
                                      t.liquidity_locked_pct, t.lock_duration_days),
            ownership_renounced = _score_ownership_renounced(t.ownership_renounced),
            wallet_concentration= _score_wallet_concentration(t.top10_wallet_pct),
            dangerous_functions = _score_dangerous_functions(
                                      t.has_mint_function,
                                      t.has_pause_function,
                                      t.has_blacklist),
            dev_history         = _score_dev_history(
                                      t.dev_rugged_before, t.dev_sell_pct_prior),
            contract_age        = _score_contract_age(t.contract_age_days),
            social_authenticity = _score_social_authenticity(
                                      t.social_account_age_days, t.fake_follower_pct),
            volume_liq_ratio    = _score_volume_liq_ratio(
                                      t.volume_24h_usd, t.liquidity_usd),
        )

    @staticmethod
    def _weighted_score(scores: dict[str, float],
                        weights: dict[str, float]) -> float:
        """
        Compute a 0–100 normalised weighted score.

        weighted_score = Σ(score_i × weight_i) / Σ(10 × weight_i) × 100
        """
        numerator   = sum(scores[n] * weights[n] for n in SIGNAL_NAMES)
        denominator = sum(10.0     * weights[n] for n in SIGNAL_NAMES)
        if denominator == 0:
            return 0.0
        return round(numerator / denominator * 100.0, 2)

    # ── Post-Mortem Learning ─────────────────────────────────

    def _post_mortem(self, address: str, outcome: str,
                     signal_scores: dict[str, float]) -> None:
        adjustments: dict[str, float] = {}

        if outcome == "rug":
            # Signals that *didn't* flag danger (score ≥ 5) while token rugged
            # → they gave a false sense of security → increase their weight
            for name in SIGNAL_NAMES:
                if signal_scores.get(name, 0.0) >= 5.0:
                    old_w = self._db["weights"][name]
                    new_w = min(old_w * (1.0 + WEIGHT_BUMP_PCT / 100.0), WEIGHT_MAX)
                    self._db["weights"][name] = round(new_w, 4)
                    adjustments[name] = round(new_w - old_w, 4)

            # Raise the minimum score floor
            old_thresh = self._db["min_score_threshold"]
            self._db["min_score_threshold"] = round(
                min(old_thresh + SCORE_FLOOR_BUMP, 95.0), 1)
            adjustments["_threshold"] = round(
                self._db["min_score_threshold"] - old_thresh, 1)

            event = f"RUG  {address[:20]}…  →  floor ↑ to {self._db['min_score_threshold']}"

        else:  # safe
            # Signals that correctly predicted safety (score ≥ 5) → reinforce
            for name in SIGNAL_NAMES:
                if signal_scores.get(name, 0.0) >= 5.0:
                    old_w = self._db["weights"][name]
                    new_w = min(old_w * (1.0 + WEIGHT_REINFORCE_PCT / 100.0), WEIGHT_MAX)
                    self._db["weights"][name] = round(new_w, 4)
                    adjustments[name] = round(new_w - old_w, 4)

            event = f"SAFE {address[:20]}…  →  reinforce"

        self._db["weight_log"].append({
            "timestamp":   _now(),
            "event":       event,
            "adjustments": adjustments,
        })

        print(f"\n  [POST-MORTEM] {event}")
        if adjustments:
            for k, delta in adjustments.items():
                if k != "_threshold":
                    direction = "↑" if delta > 0 else "↓"
                    print(f"    {k:<24} {direction} {abs(delta):.4f}")
            if "_threshold" in adjustments:
                print(f"    min_score_threshold  ↑ {adjustments['_threshold']:.1f} "
                      f"→ {self._db['min_score_threshold']:.1f}")

    def _maybe_adjust_threshold(self) -> None:
        """
        Loosen the threshold only when detection win-rate over last 50
        outcomes exceeds WIN_RATE_THRESHOLD (95 %).
        """
        wr = self._win_rate_last_n(WINDOW_SIZE)
        if wr >= WIN_RATE_THRESHOLD:
            old = self._db["min_score_threshold"]
            new = round(max(old - SCORE_FLOOR_RELAX, MIN_SCORE_INITIAL), 1)
            if new < old:
                self._db["min_score_threshold"] = new
                self._db["weight_log"].append({
                    "timestamp": _now(),
                    "event":     f"RELAX threshold {old:.1f} → {new:.1f}  "
                                 f"(win-rate {wr:.1f}% > {WIN_RATE_THRESHOLD}%)",
                    "adjustments": {"_threshold": round(new - old, 1)},
                })
                print(f"\n  [THRESHOLD RELAX] {old:.1f} → {new:.1f}  "
                      f"(win-rate {wr:.1f}%)")

    def _win_rate_last_n(self, n: int) -> float:
        """
        Detection win rate over the last n outcomes.

        A 'win' = outcome was 'safe' (we correctly let it through)
                  OR the token would have been blocked (score < threshold)
                  and it turned out to be a rug.
        Because we don't store whether we blocked it, we approximate:
        win = (# safe outcomes in last n) / n * 100
        This is a conservative measure; blocked tokens are not counted as
        wins even if they would have been rugs.
        """
        recent = self._db["outcomes"][-n:]
        if not recent:
            return 0.0
        safe_count = sum(1 for r in recent if r["outcome"] == "safe")
        return round(safe_count / len(recent) * 100.0, 1)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
