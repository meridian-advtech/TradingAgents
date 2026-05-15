"""
Kairos Task Router — Milestone 9 / Phase B

Central routing layer. Every task that enters the Kairos pipeline passes
through here. The router decides whether to dispatch to:
  - Ollama (local, fast, free) — routine orchestration
  - Claude API (cloud, deep reasoning) — investment decisions

Routing table:
  ┌─────────────────────────────────────────┬─────────────┐
  │ Task                                    │ Backend     │
  ├─────────────────────────────────────────┼─────────────┤
  │ Data formatting and structuring         │ Ollama      │
  │ News relevance screening (equities)     │ Ollama      │
  │ Stale source detection                  │ Ollama      │
  │ Crypto news relevance screening         │ Ollama      │
  │ Crypto price/volume signal classify     │ Ollama      │
  │ BUY / SELL / HOLD decision (equities)   │ Claude API  │
  │ Tax optimization                        │ Claude API  │
  │ Multi-factor analysis                   │ Claude API  │
  │ Crypto trading decision                 │ Claude API  │
  └─────────────────────────────────────────┴─────────────┘
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Lazy import — only pulled in when actually needed
_ollama = None


def _get_ollama():
    global _ollama
    if _ollama is None:
        import kairos_ollama as _mod
        _ollama = _mod
    return _ollama


# ── Task types ────────────────────────────────────────────────────────

class TaskType(str, Enum):
    # Ollama tasks — equities
    FORMAT_DATA = "format_data"
    NEWS_RELEVANCE = "news_relevance"
    NEWS_SUMMARY = "news_summary"
    STALE_SOURCES = "stale_sources"
    # Ollama tasks — crypto (fast local classification)
    CRYPTO_NEWS_RELEVANCE = "crypto_news_relevance"
    CRYPTO_SIGNAL_CLASSIFY = "crypto_signal_classify"
    # Claude API tasks — equities
    TRADING_DECISION = "trading_decision"
    TAX_OPTIMIZATION = "tax_optimization"
    MULTI_FACTOR_ANALYSIS = "multi_factor_analysis"
    # Claude API tasks — crypto (complex reasoning required)
    CRYPTO_TRADING_DECISION = "crypto_trading_decision"


class Backend(str, Enum):
    OLLAMA = "ollama"
    CLAUDE = "claude"


# Routing table — maps each task type to its backend
ROUTING_TABLE: dict[TaskType, Backend] = {
    # Equities — Ollama
    TaskType.FORMAT_DATA:              Backend.OLLAMA,
    TaskType.NEWS_RELEVANCE:           Backend.OLLAMA,
    TaskType.NEWS_SUMMARY:             Backend.OLLAMA,
    TaskType.STALE_SOURCES:            Backend.OLLAMA,
    # Crypto — Ollama (fast local triage)
    TaskType.CRYPTO_NEWS_RELEVANCE:    Backend.OLLAMA,
    TaskType.CRYPTO_SIGNAL_CLASSIFY:   Backend.OLLAMA,
    # Equities — Claude
    TaskType.TRADING_DECISION:         Backend.CLAUDE,
    TaskType.TAX_OPTIMIZATION:         Backend.CLAUDE,
    TaskType.MULTI_FACTOR_ANALYSIS:    Backend.CLAUDE,
    # Crypto — Claude (buy/sell/hold decision)
    TaskType.CRYPTO_TRADING_DECISION:  Backend.CLAUDE,
}


# ── Result container ──────────────────────────────────────────────────

@dataclass
class RouteResult:
    task_type: TaskType
    backend: Backend
    result: Any
    latency_ms: float
    success: bool
    error: str = ""


# ── Router ───────────────────────────────────────────────────────────

class KairosRouter:
    """Central task router for the Kairos pipeline.

    Usage:
        router = KairosRouter()
        result = router.route(TaskType.NEWS_RELEVANCE,
                              headline="Apple beats earnings",
                              tickers=["AAPL"])
        if result.success:
            print(result.result)  # True/False
    """

    def __init__(self, ollama_available: bool | None = None):
        """
        Args:
            ollama_available: Override for Ollama availability check.
                              None = auto-detect at first use.
        """
        self._ollama_available: bool | None = ollama_available
        self._stats: list[RouteResult] = []

    @property
    def ollama_available(self) -> bool:
        if self._ollama_available is None:
            self._ollama_available = _get_ollama().is_available()
        return self._ollama_available

    def get_backend(self, task_type: TaskType) -> Backend:
        """Return the assigned backend for a task type.

        Falls back to Claude if Ollama is unavailable.
        """
        assigned = ROUTING_TABLE.get(task_type, Backend.CLAUDE)
        if assigned == Backend.OLLAMA and not self.ollama_available:
            return Backend.CLAUDE  # graceful fallback
        return assigned

    def route(self, task_type: TaskType, **kwargs) -> RouteResult:
        """Dispatch a task to the appropriate backend.

        Returns a RouteResult with the backend response.
        """
        backend = self.get_backend(task_type)
        start = time.time()

        try:
            if backend == Backend.OLLAMA:
                result = self._dispatch_ollama(task_type, **kwargs)
            else:
                result = self._dispatch_claude(task_type, **kwargs)
            success = True
            error = ""
        except Exception as e:
            result = None
            success = False
            error = str(e)

        latency_ms = round((time.time() - start) * 1000)
        route_result = RouteResult(
            task_type=task_type,
            backend=backend,
            result=result,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )
        self._stats.append(route_result)
        return route_result

    def _dispatch_ollama(self, task_type: TaskType, **kwargs) -> Any:
        """Execute an Ollama-routed task."""
        ollama = _get_ollama()

        if task_type == TaskType.NEWS_RELEVANCE:
            headline = kwargs["headline"]
            tickers = kwargs.get("tickers", ["AAPL"])
            return ollama.is_news_relevant(headline, tickers)

        if task_type == TaskType.NEWS_SUMMARY:
            headlines = kwargs["headlines"]
            ticker = kwargs.get("ticker", "AAPL")
            return ollama.summarize_news_headlines(headlines, ticker)

        if task_type == TaskType.STALE_SOURCES:
            source_timestamps = kwargs["source_timestamps"]
            max_age_hours = kwargs.get("max_age_hours", 4.0)
            return ollama.check_stale_sources(source_timestamps, max_age_hours)

        if task_type == TaskType.FORMAT_DATA:
            raw_text = kwargs["raw_text"]
            section_name = kwargs.get("section_name", "data")
            return ollama.format_snapshot_section(raw_text, section_name)

        if task_type == TaskType.CRYPTO_NEWS_RELEVANCE:
            headline = kwargs["headline"]
            assets = kwargs.get("assets", ["BTC", "ETH"])
            return ollama.is_crypto_news_relevant(headline, assets)

        if task_type == TaskType.CRYPTO_SIGNAL_CLASSIFY:
            symbol = kwargs["symbol"]
            change_pct = kwargs["change_pct"]
            volume_change_pct = kwargs.get("volume_change_pct", 0.0)
            return ollama.classify_crypto_signal(symbol, change_pct, volume_change_pct)

        raise ValueError(f"No Ollama handler for task type: {task_type}")

    def _dispatch_claude(self, task_type: TaskType, **kwargs) -> Any:
        """Route Claude API tasks using the Anthropic SDK.

        Uses the anthropic Python SDK (Anthropic().messages.create)
        with model "claude-sonnet-4-6", reading ANTHROPIC_API_KEY from environment.
        """
        import os
        import anthropic
        
        # Get prompt file from kwargs or use default based on task type
        prompt_file = kwargs.get("prompt_file")
        if prompt_file is None:
            prompt_file_map = {
                TaskType.TRADING_DECISION: "kairos_prompt.txt",
                TaskType.TAX_OPTIMIZATION: None,  # No default prompt file
                TaskType.MULTI_FACTOR_ANALYSIS: None,  # No default prompt file
                TaskType.CRYPTO_TRADING_DECISION: "kairos_crypto_prompt.txt",
            }
            prompt_file = prompt_file_map.get(task_type)
        
        # Build prompt from file if available
        if prompt_file:
            try:
                with open(prompt_file, "r") as f:
                    prompt_text = f.read()
            except IOError:
                prompt_text = str(kwargs.get("prompt", ""))
        else:
            prompt_text = str(kwargs.get("prompt", ""))
        
        # Ensure we have a prompt
        if not prompt_text:
            raise ValueError(f"No prompt provided for task type: {task_type}")
        
        # Call Anthropic API
        try:
            client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": "Output ONLY the JSON object. No markdown fences, no explanation, no preamble.",
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt_text}],
                timeout=60,
            )
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            # Parse JSON from response
            import json
            # Try to find JSON in the response
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        pass
            # If no JSON found, try the whole response
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"status": "error", "error": "Could not parse JSON from response", "raw_response": text}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def print_stats(self):
        """Print routing statistics for the current session."""
        if not self._stats:
            print("  No tasks routed yet.")
            return

        ollama_tasks = [r for r in self._stats if r.backend == Backend.OLLAMA]
        claude_tasks = [r for r in self._stats if r.backend == Backend.CLAUDE]
        failures = [r for r in self._stats if not r.success]

        print(f"\n  Routing stats ({len(self._stats)} tasks):")
        print(f"    Ollama:  {len(ollama_tasks)} tasks"
              + (f", avg {sum(r.latency_ms for r in ollama_tasks)//max(len(ollama_tasks),1)}ms" if ollama_tasks else ""))
        print(f"    Claude:  {len(claude_tasks)} tasks")
        if failures:
            print(f"    Failures: {len(failures)}")
            for f in failures:
                print(f"      {f.task_type}: {f.error}")


# ── Module-level convenience functions ───────────────────────────────
# These mirror the TaskType enum values for easy import.

_default_router: KairosRouter | None = None


def get_router() -> KairosRouter:
    """Return the module-level default router instance."""
    global _default_router
    if _default_router is None:
        _default_router = KairosRouter()
    return _default_router


def route_news_relevance(headline: str, tickers: list[str]) -> bool:
    return get_router().route(TaskType.NEWS_RELEVANCE,
                               headline=headline, tickers=tickers).result or True


def route_news_summary(headlines: list[str], ticker: str = "AAPL") -> str:
    result = get_router().route(TaskType.NEWS_SUMMARY,
                                 headlines=headlines, ticker=ticker)
    return result.result or "Unable to summarize."


def route_stale_sources(source_timestamps: dict, max_age_hours: float = 4.0) -> list[str]:
    result = get_router().route(TaskType.STALE_SOURCES,
                                 source_timestamps=source_timestamps,
                                 max_age_hours=max_age_hours)
    return result.result or []


def route_format_data(raw_text: str, section_name: str = "data") -> str:
    result = get_router().route(TaskType.FORMAT_DATA,
                                 raw_text=raw_text, section_name=section_name)
    return result.result or raw_text


def route_crypto_news_relevance(headline: str, assets: list[str] | None = None) -> bool:
    result = get_router().route(TaskType.CRYPTO_NEWS_RELEVANCE,
                                 headline=headline,
                                 assets=assets or ["BTC", "ETH"])
    return result.result if result.success else True  # conservative: include on failure


def route_crypto_signal_classify(
    symbol: str, change_pct: float, volume_change_pct: float = 0.0
) -> str:
    result = get_router().route(TaskType.CRYPTO_SIGNAL_CLASSIFY,
                                 symbol=symbol,
                                 change_pct=change_pct,
                                 volume_change_pct=volume_change_pct)
    return result.result or "unknown"


# ── CLI demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Kairos Router — Demo")
    print("=" * 60)

    router = KairosRouter()
    print(f"\n  Ollama available: {router.ollama_available}")
    print("\n  Routing table:")
    for task, backend in ROUTING_TABLE.items():
        actual = router.get_backend(task)
        fallback = " [FALLBACK→Claude]" if actual != backend else ""
        print(f"    {task.value:<30} → {actual.value}{fallback}")

    print("\n  Test: news_relevance")
    r = router.route(TaskType.NEWS_RELEVANCE,
                     headline="Apple reports record Q1 revenue",
                     tickers=["AAPL"])
    print(f"    Result: {r.result}  ({r.latency_ms}ms via {r.backend.value})")

    print("\n  Test: trading_decision (Claude deferred)")
    r2 = router.route(TaskType.TRADING_DECISION)
    print(f"    Result: {r2.result}")

    router.print_stats()
    print()
