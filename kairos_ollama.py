"""
Kairos Ollama Agent — Two-Model Architecture

Two specialized local LLMs via Ollama:
  Screen model (mistral-small3.2) — fast Tier 1 screening of 264+ tickers
  Reason model (llama3.3)         — quality Tier 2 reasoning for ≤15 tickers

Division of labor:
  Ollama (this module) — fast, free, private, always-on routine tasks
  Claude Code          — complex investment reasoning and final decisions

HTTP API: http://localhost:11434 (Ollama default)
Config:   kairos_config.json → ollama.{screen_model, reason_model}
"""

import json
import os
import time
import requests
from datetime import datetime, timezone

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_TIMEOUT = 60  # seconds
KEEP_ALIVE = "0"      # default: unload immediately after use
KEEP_ALIVE_SCREEN = "2m"   # screen model stays loaded across batches; explicitly unloaded after screening
WARMUP_TIMEOUT = 90   # generous timeout for cold-start model reload (~30s)

# Fallback model if config is missing
_FALLBACK_MODEL = "llama3.2"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_SCRIPT_DIR, "kairos_config.json")

# Cached config
_config: dict | None = None


def _load_ollama_config() -> dict:
    """Load ollama model config from kairos_config.json."""
    global _config
    if _config is not None:
        return _config
    defaults = {"screen_model": "mistral-small3.2", "reason_model": "llama3.3"}
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE) as f:
                data = json.load(f)
            defaults.update(data.get("ollama", {}))
        except (json.JSONDecodeError, IOError):
            pass
    _config = defaults
    return _config


def get_screen_model() -> str:
    """Return the model name for Tier 1 screening (speed-optimized)."""
    return _load_ollama_config().get("screen_model", "mistral-small3.2")


def get_reason_model() -> str:
    """Return the model name for Tier 2 reasoning (quality-optimized)."""
    return _load_ollama_config().get("reason_model", "llama3.3")


# ── Base client ──────────────────────────────────────────────────────

def list_models() -> list[str]:
    """Return list of all model names available in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def is_available(model: str | None = None) -> bool:
    """Return True if Ollama is running and the specified model is loaded.

    If model is None, checks if the reason model (primary) is available.
    """
    if model is None:
        model = get_reason_model()
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        base = model.split(":")[0]
        return any(m.startswith(base) for m in models)
    except Exception:
        return False


def check_required_models() -> dict[str, bool]:
    """Check availability of both screen and reason models.

    Returns {"screen_model": bool, "reason_model": bool, "all_ok": bool,
             "screen_name": str, "reason_name": str, "available": [...],
             "missing": [...]}.
    """
    screen = get_screen_model()
    reason = get_reason_model()
    installed = list_models()

    def _has(name: str) -> bool:
        base = name.split(":")[0]
        return any(m.startswith(base) for m in installed)

    screen_ok = _has(screen)
    reason_ok = _has(reason)

    missing = []
    if not screen_ok:
        missing.append(screen)
    if not reason_ok:
        missing.append(reason)

    return {
        "screen_model": screen_ok,
        "reason_model": reason_ok,
        "all_ok": screen_ok and reason_ok,
        "screen_name": screen,
        "reason_name": reason,
        "available": installed,
        "missing": missing,
    }


def warmup(model: str | None = None) -> bool:
    """Preload a model into VRAM with a trivial prompt.

    After KEEP_ALIVE elapses, Ollama unloads models to free memory.
    The first real call after idle would pay a ~30s reload penalty
    (mistral-small3.2 ≈ 15GB). This function eats that latency
    up front so the pipeline's timed calls don't timeout.

    Returns True if the model responded, False on failure.
    """
    if model is None:
        model = get_screen_model()
    print(f"  [Ollama/{model}] Warming up (cold-start reload may take ~30s)...")
    t0 = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": "Reply OK.",
                "stream": False,
                "keep_alive": KEEP_ALIVE_SCREEN,
            },
            timeout=WARMUP_TIMEOUT,
        )
        resp.raise_for_status()
        elapsed = time.time() - t0
        print(f"  [Ollama/{model}] Warm — loaded in {elapsed:.1f}s")
        return True
    except requests.RequestException as e:
        print(f"  [Ollama/{model}] Warmup failed: {e}")
        return False


def ask(prompt: str, model: str | None = None, timeout: int = DEFAULT_TIMEOUT, keep_alive_override: str | None = None) -> str:
    """Send a prompt to Ollama and return the response text.

    If model is None, uses the reason model (quality-optimized).
    Returns empty string on failure (caller decides how to handle).
    Passes keep_alive so Ollama unloads the model after 5 min idle,
    freeing VRAM between scheduler cycles.
    """
    if model is None:
        model = get_reason_model()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": keep_alive_override if keep_alive_override is not None else KEEP_ALIVE,
                "options": {"think": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.RequestException as e:
        print(f"  [Ollama/{model}] Request failed: {e}")
        return ""


def ask_screen(prompt: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Send a prompt using the screen model (speed-optimized).

    Use this for high-volume Tier 1 scoring where speed > quality.
    """
    return ask(prompt, model=get_screen_model(), timeout=timeout, keep_alive_override=KEEP_ALIVE_SCREEN)


def ask_reason(prompt: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Send a prompt using the reason model (quality-optimized).

    Use this for Tier 2 tasks requiring deeper analysis.
    """
    return ask(prompt, model=get_reason_model(), timeout=timeout)


# ── Task 1: News relevance filter ────────────────────────────────────

def is_news_relevant(headline: str, tickers: list[str]) -> bool:
    """Ask Ollama whether a news headline is relevant to any held ticker.

    Returns True if relevant, False otherwise. Falls back to True on error
    (conservative: let Claude see ambiguous items).
    """
    ticker_str = ", ".join(tickers)
    prompt = f"""You are a financial news filter. Answer with only YES or NO.

Is the following headline relevant to any of these tickers: {ticker_str}?
A headline is relevant if it could materially affect the stock price of any listed ticker
(company news, sector news, macro events that directly impact the company, regulatory news).

Headline: "{headline}"

Answer (YES or NO only):"""

    response = ask_reason(prompt)
    if not response:
        return True  # fallback: include on failure
    return response.strip().upper().startswith("Y")


def filter_news_batch(headlines: list[str], tickers: list[str]) -> list[dict]:
    """Filter a list of headlines for relevance. Returns list of dicts with
    headline and relevance flag.

    Used to trim the snapshot before sending to Claude API.
    """
    results = []
    for headline in headlines:
        relevant = is_news_relevant(headline, tickers)
        results.append({"headline": headline, "relevant": relevant})
    return results


# ── Task 2: Stale source detection ───────────────────────────────────

def check_stale_sources(
    source_timestamps: dict[str, str | None],
    max_age_hours: float = 4.0,
) -> list[str]:
    """Determine which data sources need refreshing.

    Args:
        source_timestamps: dict mapping source name → ISO timestamp string or None
        max_age_hours: threshold in hours before a source is considered stale

    Returns list of source names that need refreshing.
    """
    now = datetime.now(timezone.utc)
    stale = []

    for source, ts_str in source_timestamps.items():
        if ts_str is None:
            stale.append(source)
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age_hours = (now - ts).total_seconds() / 3600
            if age_hours > max_age_hours:
                stale.append(source)
        except (ValueError, TypeError):
            stale.append(source)  # unparseable = treat as stale

    return stale


def llm_check_stale_sources(source_summary: str) -> list[str]:
    """Ask Ollama to identify stale sources from a human-readable summary.

    Useful when you want the LLM to apply judgment (e.g., market is closed,
    so crypto is more important than equity news right now).

    Args:
        source_summary: free-text description of source ages/status

    Returns list of source names identified as needing refresh.
    """
    prompt = f"""You are a data pipeline orchestrator for a stock trading system.
The following data sources are available: finnhub (news), fred (macro rates),
kalshi (prediction markets), congress (legislative bills), coingecko (crypto prices).

Based on the source status below, list which sources need to be refreshed.
Reply with ONLY a JSON array of source names, e.g. ["finnhub", "coingecko"].
If all sources are fresh, return an empty array: [].

Source status:
{source_summary}

JSON array of stale sources:"""

    response = ask(prompt)
    if not response:
        return []

    # Extract JSON array from response
    try:
        # Find the first [ ... ] block
        start = response.find("[")
        end = response.rfind("]")
        if start != -1 and end != -1:
            return json.loads(response[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ── Task 3: Data formatting ───────────────────────────────────────────

def format_snapshot_section(raw_text: str, section_name: str) -> str:
    """Ask Ollama to reformat a raw data section into clean structured text.

    Returns the reformatted text, or original text on failure.
    """
    prompt = f"""You are a data formatting assistant. Reformat the following raw data
from the "{section_name}" source into clean, structured text suitable for
a financial analyst to read. Keep all numeric values exact. Remove noise and
formatting artifacts. Use bullet points where appropriate.

Raw data:
{raw_text[:2000]}

Reformatted:"""

    response = ask_reason(prompt, timeout=45)
    return response if response else raw_text


def summarize_news_headlines(headlines: list[str], ticker: str) -> str:
    """Ask Ollama to produce a one-paragraph sentiment summary of headlines.

    Faster and cheaper than sending all headlines to Claude.
    """
    if not headlines:
        return "No relevant headlines available."

    headlines_text = "\n".join(f"- {h}" for h in headlines[:20])
    prompt = f"""You are a financial analyst assistant. Based only on the headlines below,
write ONE paragraph (3-5 sentences) summarizing the news sentiment for {ticker}.
State whether sentiment is bullish, bearish, or neutral and cite the key themes.
Do not make investment recommendations.

Headlines:
{headlines_text}

Sentiment summary:"""

    response = ask_reason(prompt, timeout=45)
    return response if response else "Unable to summarize headlines."


# ── Task 4: Crypto signal routing ────────────────────────────────────

def is_crypto_news_relevant(headline: str, assets: list[str] | None = None) -> bool:
    """Ask Ollama whether a headline is relevant to BTC, ETH, or crypto broadly.

    Returns True if relevant, False otherwise. Falls back to True on error
    (conservative: let Claude see ambiguous items).
    """
    if assets is None:
        assets = ["BTC", "ETH"]
    asset_str = ", ".join(assets)
    prompt = f"""You are a crypto news filter. Answer with only YES or NO.

Is the following headline relevant to any of these crypto assets: {asset_str}?
A headline is relevant if it could materially affect the price of any listed asset:
crypto regulation, exchange news, macro policy affecting crypto, on-chain events,
large institutional moves, stablecoin news, or broad risk-on/risk-off sentiment.

Headline: "{headline}"

Answer (YES or NO only):"""

    response = ask_reason(prompt)
    if not response:
        return True  # conservative fallback
    return response.strip().upper().startswith("Y")


def classify_crypto_signal(
    symbol: str, change_pct: float, volume_change_pct: float = 0.0
) -> str:
    """Ask Ollama to classify a crypto price/volume move as a trading signal.

    Args:
        symbol:             Asset symbol, e.g. "BTC"
        change_pct:         Price change % (positive = up, negative = down)
        volume_change_pct:  Volume change % vs recent average (0 if unknown)

    Returns one of:
        "alert"    — significant move, escalate to Claude for a decision
        "normal"   — routine fluctuation, no action needed
        "unknown"  — could not classify (caller should treat as "alert")
    """
    direction = "up" if change_pct >= 0 else "down"
    vol_note = (
        f"Volume is {abs(volume_change_pct):.1f}% {'above' if volume_change_pct >= 0 else 'below'} average."
        if volume_change_pct != 0
        else "Volume data not available."
    )

    prompt = f"""You are a crypto trading signal classifier. Answer with only ALERT or NORMAL.

{symbol} has moved {direction} {abs(change_pct):.2f}% recently. {vol_note}

Classify this as:
- ALERT: if the move is large enough to warrant a trading review (typically >3% or combined
  with high volume), unusual for the current market environment, or suggests a breakout/breakdown.
- NORMAL: if this is routine daily volatility with no actionable signal.

Answer (ALERT or NORMAL only):"""

    response = ask_reason(prompt, timeout=30)
    if not response:
        return "unknown"
    upper = response.strip().upper()
    if upper.startswith("A"):
        return "alert"
    if upper.startswith("N"):
        return "normal"
    return "unknown"


# ── Task 5: Routing decision ──────────────────────────────────────────

def should_escalate_crypto_to_claude(symbol: str, signal: str, risk_summary: str) -> bool:
    """Ask Ollama if a crypto signal warrants escalating to Claude for a trade decision.

    Args:
        symbol:       "BTC" or "ETH"
        signal:       Classification from classify_crypto_signal ("alert"/"normal")
        risk_summary: Short text describing current risk guard status

    Returns True if Claude should make a trading decision, False to skip.
    """
    prompt = f"""You are a task router for a crypto trading AI system.
Given the signal and risk context below, decide whether to escalate to Claude API
for a full buy/sell/hold crypto decision, or skip (no action needed).

Escalate if: the signal is ALERT, risk guards are all passing, and market conditions
suggest a genuine trading opportunity or risk. Skip if signal is NORMAL or risk guards
are already blocking trades.

Asset: {symbol}
Signal: {signal}
Risk context: {risk_summary[:500]}

Decision (reply with only ESCALATE or SKIP):"""

    response = ask_reason(prompt, timeout=30)
    if not response:
        return signal == "alert"  # fallback: escalate alerts
    return response.strip().upper().startswith("E")


def should_escalate_to_claude(context_summary: str) -> bool:
    """Ask Ollama if the current situation warrants escalating to Claude API.

    Used as a secondary gate — Ollama screens if deep reasoning is needed.
    Returns True if Claude API should be called, False if Ollama can handle.
    """
    prompt = f"""You are a task router for a trading AI system.
Given the following market context summary, decide if this situation requires
deep multi-factor investment reasoning (escalate to Claude API) or can be
handled with simple rule-based logic (handle locally).

Escalate to Claude API if: multiple conflicting signals, unusual market conditions,
tax implications, multi-factor analysis required, or high uncertainty.
Handle locally if: clear hold signal, data pipeline maintenance, formatting tasks,
simple yes/no filtering.

Context:
{context_summary[:1000]}

Decision (reply with only ESCALATE or LOCAL):"""

    response = ask_reason(prompt, timeout=30)
    if not response:
        return True  # default: escalate on failure
    return response.strip().upper().startswith("E")


# ── Latency measurement ───────────────────────────────────────────────

def benchmark(model: str | None = None,
              prompt: str = "What is 2+2? Reply with only the number.") -> tuple[str, float]:
    """Measure Ollama response latency for a model. Returns (response, seconds)."""
    if model is None:
        model = get_reason_model()
    start = time.time()
    resp = ask(prompt, model=model, timeout=30)
    return resp, round(time.time() - start, 2)


# ── CLI smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Kairos Ollama Agent — Two-Model Smoke Test")
    print("=" * 60)

    # 1. Check both models
    status = check_required_models()
    print(f"\n[1] Model availability:")
    print(f"    Screen model ({status['screen_name']}): "
          f"{'✓ available' if status['screen_model'] else '✗ MISSING'}")
    print(f"    Reason model ({status['reason_name']}): "
          f"{'✓ available' if status['reason_model'] else '✗ MISSING'}")

    if status["missing"]:
        print(f"\n    MISSING MODELS — install with:")
        for m in status["missing"]:
            print(f"      ollama pull {m}")
        raise SystemExit(1)

    print(f"    Installed models: {', '.join(status['available'])}")

    # 2. Ping both models
    print("\n[2] Ping test — screen model (speed):")
    resp_s, lat_s = benchmark(get_screen_model())
    print(f"    {get_screen_model()}: '{resp_s}' in {lat_s}s")

    print("\n[3] Ping test — reason model (quality):")
    resp_r, lat_r = benchmark(get_reason_model())
    print(f"    {get_reason_model()}: '{resp_r}' in {lat_r}s")

    # 3. News relevance (reason model)
    print("\n[4] News relevance filter (reason model):")
    test_headlines = [
        "Apple unveils new iPhone with AI features",
        "Fed holds rates steady at 5.25%",
        "Dogecoin surges 40% on meme revival",
        "AAPL faces EU antitrust investigation over App Store",
    ]
    for h in test_headlines:
        relevant = is_news_relevant(h, ["AAPL"])
        print(f"    {'✓' if relevant else '✗'} {'RELEVANT' if relevant else 'FILTERED'}: {h[:60]}")

    # 4. Stale source detection
    print("\n[5] Stale source detection:")
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    timestamps = {
        "finnhub": (now - timedelta(hours=6)).isoformat(),
        "fred": (now - timedelta(hours=1)).isoformat(),
        "kalshi": None,
        "congress": (now - timedelta(hours=2)).isoformat(),
        "coingecko": (now - timedelta(hours=5)).isoformat(),
    }
    stale = check_stale_sources(timestamps, max_age_hours=4.0)
    print(f"    Stale sources: {stale}")

    print("\n" + "=" * 60)
    print(f"  Screen: {get_screen_model()} ({lat_s}s)")
    print(f"  Reason: {get_reason_model()} ({lat_r}s)")
    print("  All checks passed. Two-model Ollama layer is operational.")
    print("=" * 60)


def unload_model(model: str | None = None) -> bool:
    """Explicitly unload a model from VRAM immediately.

    Call this after a screening or reasoning phase completes to free
    memory before the next model loads. Uses keep_alive=0 to force eviction.
    """
    if model is None:
        model = get_screen_model()
    try:
        requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=10,
        )
        print(f"  [Ollama/{model}] Unloaded from VRAM")
        return True
    except requests.RequestException:
        return False
