"""
news.py — Biotech catalyst detection via Benzinga
==================================================
The biotech special rule requires a confirmed catalyst before entering any trade.
This module polls the Benzinga News API and scores each headline against a
keyword list defined in config.yaml.

Design decisions:
  • Results are cached per-ticker with a TTL of 10 minutes; Benzinga charges
    per API call so we don't hit it on every strategy evaluation.
  • The cache is a simple in-memory dict; no Redis required for a single-process
    day trading bot.
  • All network calls are async with aiohttp.
  • Catalyst detection is intentionally conservative: any matching keyword in
    the headline OR summary within the last 24 hours counts as a catalyst.
    False positives (trading on weak news) are preferable to false negatives
    (missing a real mover) given our aggressive mandate.
  • Energy stocks do NOT require catalyst confirmation — they use the USO
    correlation check in strategy.py instead.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from utils import get_config, get_logger

log = get_logger(__name__)
cfg = get_config()


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class CatalystResult:
    ticker: str
    has_catalyst: bool
    matched_keywords: list[str]
    headline: str = ""           # Best matching headline
    published_at: str = ""       # ISO timestamp of the article
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Cache ─────────────────────────────────────────────────────────────────────

class _CatalystCache:
    """Simple TTL cache for catalyst results (10-minute default)."""

    _TTL = 600  # seconds

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, CatalystResult]] = {}

    def get(self, ticker: str) -> CatalystResult | None:
        if ticker not in self._store:
            return None
        ts, result = self._store[ticker]
        if time.monotonic() - ts > self._TTL:
            del self._store[ticker]
            return None
        return result

    def set(self, ticker: str, result: CatalystResult) -> None:
        self._store[ticker] = (time.monotonic(), result)

    def invalidate(self, ticker: str) -> None:
        self._store.pop(ticker, None)

    def clear(self) -> None:
        self._store.clear()


_cache = _CatalystCache()


# ── News Client ───────────────────────────────────────────────────────────────

class NewsClient:

    def __init__(self, benzinga_api_key: str) -> None:
        self._api_key = benzinga_api_key
        self._session: aiohttp.ClientSession | None = None

        nc = cfg["benzinga"]
        self._base_url = nc["base_url"]
        self._keywords: list[str] = [kw.lower() for kw in nc["catalyst_keywords"]]
        self._lookback_hours: int = nc["catalyst_lookback_hours"]

    async def __aenter__(self) -> "NewsClient":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ── Public API ───────────────────────────────────────────────────────────

    async def has_catalyst(self, ticker: str) -> CatalystResult:
        """
        Check whether `ticker` has a relevant biotech catalyst in recent news.
        Results are cached for 10 minutes to reduce API calls.
        """
        cached = _cache.get(ticker)
        if cached is not None:
            log.debug("catalyst_cache_hit", extra={"ticker": ticker})
            return cached

        result = await self._fetch_and_analyze(ticker)
        _cache.set(ticker, result)

        log.info(
            "catalyst_checked",
            extra={
                "ticker": ticker,
                "has_catalyst": result.has_catalyst,
                "keywords": result.matched_keywords,
                "headline": result.headline[:80] if result.headline else "",
            },
        )
        return result

    async def batch_check(self, tickers: list[str]) -> dict[str, CatalystResult]:
        """Check multiple tickers concurrently."""
        tasks = {ticker: self.has_catalyst(ticker) for ticker in tickers}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        output: dict[str, CatalystResult] = {}
        for ticker, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                log.warning(
                    "catalyst_check_failed",
                    extra={"ticker": ticker, "error": str(result)},
                )
                output[ticker] = CatalystResult(
                    ticker=ticker,
                    has_catalyst=False,
                    matched_keywords=[],
                    headline="",
                )
            else:
                output[ticker] = result
        return output

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _fetch_and_analyze(self, ticker: str) -> CatalystResult:
        assert self._session is not None

        articles = await self._fetch_articles(ticker)
        if not articles:
            return CatalystResult(ticker=ticker, has_catalyst=False, matched_keywords=[])

        for article in articles:
            matched = self._match_keywords(article)
            if matched:
                return CatalystResult(
                    ticker=ticker,
                    has_catalyst=True,
                    matched_keywords=matched,
                    headline=article.get("title", ""),
                    published_at=article.get("created", ""),
                )

        return CatalystResult(ticker=ticker, has_catalyst=False, matched_keywords=[])

    async def _fetch_articles(self, ticker: str) -> list[dict]:
        """
        Fetch recent Benzinga news for a ticker.
        Benzinga REST API v2: GET /news?tickers=TICKER&pageSize=20
        """
        assert self._session is not None
        from datetime import timedelta

        lookback = datetime.now(timezone.utc) - timedelta(hours=self._lookback_hours)
        lookback_str = lookback.strftime("%Y-%m-%dT%H:%M:%S")

        url = f"{self._base_url}/news"
        params = {
            "token": self._api_key,
            "tickers": ticker,
            "pageSize": "20",
            "dateFrom": lookback_str,
            "sort": "created:desc",
        }

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 401:
                    log.error("benzinga_auth_failed", extra={"ticker": ticker})
                    return []
                if resp.status != 200:
                    log.warning(
                        "benzinga_non_200",
                        extra={"ticker": ticker, "status": resp.status},
                    )
                    return []
                data = await resp.json()

            # Benzinga returns a list at top level or wrapped in "result"
            if isinstance(data, list):
                return data
            return data.get("result", data.get("data", []))

        except asyncio.TimeoutError:
            log.warning("benzinga_timeout", extra={"ticker": ticker})
            return []
        except Exception as exc:
            log.warning("benzinga_error", extra={"ticker": ticker, "error": str(exc)})
            return []

    def _match_keywords(self, article: dict) -> list[str]:
        """
        Return list of matched keywords found in headline + body.
        Empty list → no catalyst.
        """
        title = article.get("title", "").lower()
        body = article.get("teaser", article.get("body", "")).lower()
        text = f"{title} {body}"

        matched = []
        for kw in self._keywords:
            if kw in text:
                matched.append(kw)

        return matched


# ── Standalone test entry point ───────────────────────────────────────────────

async def _demo() -> None:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("BENZINGA_API_KEY", "demo")
    tickers = ["MRNA", "BNTX", "CRSP"]
    async with NewsClient(key) as client:
        results = await client.batch_check(tickers)
        for ticker, result in results.items():
            icon = "✓" if result.has_catalyst else "✗"
            print(f"{icon} {ticker}: catalyst={result.has_catalyst}  "
                  f"keywords={result.matched_keywords}  "
                  f"headline={result.headline[:60]!r}")


if __name__ == "__main__":
    asyncio.run(_demo())
