"""
Kairos News Digest — per-ticker material-news triage for candidates AND holdings.

Replaces the Qwen post-processing path (kairos_run._ollama_postprocess_snapshot),
which (as of 2026-10-09) timed out on 57/58 summaries, filtered headlines against
the wrong ticker set, and wrote a file nothing ever read.

Design (Craft: "Kairos — Full-System Assessment & Design-Intent Action Plan"):
  Sources
    1. SEC 8-K filings (EDGAR submissions API) — material corporate events.
       Free, point-in-time, and the closest thing to the "information synthesis"
       thesis: these are facts, not commentary.
    2. Finnhub company-news — headline + summary (not just titles), for EVERY
       candidate and EVERY held position, cached per ticker so a cycle only
       re-pulls tickers whose cache has aged out.
  Triage
    ONE batched Haiku call per cycle (claude.news_model in config) returning
    structured JSON per ticker: materiality 0-3, direction, catalyst, why.
    A failure is NEVER treated as "relevant": the digest is status=UNAVAILABLE
    and any prompt section built from it says so explicitly.
  Consumers
    - kairos_news_digest.json   (latest digest, read by prompt/thesis review)
    - kairos_news_shadow.jsonl  (append-only log, one row per ticker per cycle)
    - format_news_section()     (prompt text; injected only when
                                 news.inject_prompt is true in config)

Rollout: shadow first (news.inject_prompt=false) — digest is built and logged
every cycle but does not reach the council. Flip the flag after review.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "kairos_config.json")
CACHE_FILE = os.path.join(SCRIPT_DIR, "kairos_news_cache.json")
DIGEST_FILE = os.path.join(SCRIPT_DIR, "kairos_news_digest.json")
SHADOW_LOG = os.path.join(SCRIPT_DIR, "kairos_news_shadow.jsonl")

SEC_UA = {"User-Agent": "Kairos Trading Research jason@meridiangroup.llc"}
TIMEOUT = 10

# 8-K items that are usually material for a hours-to-days horizon.
MATERIAL_8K_ITEMS = {
    "1.01": "material agreement", "1.02": "agreement terminated",
    "1.03": "bankruptcy", "2.01": "acquisition/disposition completed",
    "2.02": "results of operations", "2.03": "new material debt",
    "2.05": "restructuring/exit costs", "2.06": "material impairment",
    "3.01": "delisting notice", "4.01": "auditor change",
    "4.02": "non-reliance on prior financials", "5.01": "change in control",
    "5.02": "officer/director change", "7.01": "Reg FD disclosure",
    "8.01": "other material event",
}

_SEC_MAP: dict[str, int] | None = None


# ── config ──────────────────────────────────────────────────────────

def load_news_config() -> dict:
    defaults = {
        "enabled": True,
        "inject_prompt": False,
        "lookback_days": 3,
        "per_ticker_articles": 5,
        "cache_ttl_minutes": 45,
        "max_tickers": 90,
        "triage_timeout_s": 150,
    }
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        defaults.update(cfg.get("news", {}))
        defaults["_model"] = cfg.get("claude", {}).get("news_model", "claude-haiku-5-5")
    except Exception:
        defaults["_model"] = "claude-haiku-5-5"
    return defaults


# ── cache ───────────────────────────────────────────────────────────

def _load_cache() -> dict:
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_FILE)


def _fresh(entry: dict | None, ttl_min: int) -> bool:
    if not entry or "fetched_at" not in entry:
        return False
    try:
        ts = datetime.fromisoformat(entry["fetched_at"])
    except ValueError:
        return False
    return datetime.now(timezone.utc) - ts < timedelta(minutes=ttl_min)


# ── sources ─────────────────────────────────────────────────────────

def _finnhub_news(ticker: str, lookback_days: int, limit: int) -> dict:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return {"ok": False, "error": "FINNHUB_API_KEY not set", "items": []}
    now = datetime.now(timezone.utc)
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker,
                    "from": (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
                    "to": now.strftime("%Y-%m-%d"), "token": key},
            timeout=TIMEOUT,
        )
        if r.status_code == 429:
            return {"ok": False, "error": "rate limited", "items": []}
        r.raise_for_status()
        arts = sorted(r.json() or [], key=lambda a: a.get("datetime", 0), reverse=True)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120], "items": []}
    items = []
    for a in arts[:limit]:
        items.append({
            "src": "news",
            "ts": datetime.fromtimestamp(a.get("datetime", 0), tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "headline": (a.get("headline") or "")[:200],
            "summary": (a.get("summary") or "")[:400],
            "source": a.get("source", ""),
        })
    return {"ok": True, "items": items}


def _sec_cik(ticker: str) -> str | None:
    global _SEC_MAP
    try:
        if _SEC_MAP is None:
            r = requests.get("https://www.sec.gov/files/company_tickers.json",
                             headers=SEC_UA, timeout=TIMEOUT)
            _SEC_MAP = {str(v["ticker"]).upper(): int(v["cik_str"])
                        for v in r.json().values() if v.get("ticker")}
        cik = _SEC_MAP.get(ticker.upper().replace(".", "-"))
        return f"{cik:010d}" if cik is not None else None
    except Exception:
        return None


def _sec_8k(ticker: str, lookback_days: int) -> dict:
    cik = _sec_cik(ticker)
    if cik is None:
        return {"ok": True, "items": [], "note": "no CIK (ETF/foreign/unmapped)"}
    try:
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                         headers=SEC_UA, timeout=TIMEOUT)
        r.raise_for_status()
        recent = r.json().get("filings", {}).get("recent", {})
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120], "items": []}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    items = []
    forms = recent.get("form", [])
    for i, form in enumerate(forms):
        if form not in ("8-K", "8-K/A"):
            continue
        fdate = recent.get("filingDate", [""])[i]
        if fdate < cutoff:
            continue
        raw_items = str(recent.get("items", [""] * len(forms))[i] or "")
        codes = [c.strip() for c in raw_items.split(",") if c.strip()]
        labels = [f"{c} {MATERIAL_8K_ITEMS[c]}" for c in codes if c in MATERIAL_8K_ITEMS]
        if not labels:
            continue  # exhibits-only / 9.01-only filings are not events
        items.append({
            "src": "8-K",
            "ts": fdate,
            "headline": f"{form} filed: " + "; ".join(labels),
            "summary": "",
            "source": "SEC EDGAR",
        })
    return {"ok": True, "items": items}


def gather_items(tickers: list[str], cfg: dict) -> tuple[dict, dict]:
    """Return ({ticker: [items]}, coverage_stats). Uses the per-ticker cache."""
    cache = _load_cache()
    ttl = int(cfg["cache_ttl_minutes"])
    lookback = int(cfg["lookback_days"])
    limit = int(cfg["per_ticker_articles"])
    stale = [t for t in tickers if not _fresh(cache.get(t), ttl)]

    def _pull(t: str):
        news = _finnhub_news(t, lookback, limit)
        time.sleep(0.25)  # stay well under Finnhub's 60/min across 4 workers
        sec = _sec_8k(t, lookback)
        return t, news, sec

    errors = 0
    if stale:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for t, news, sec in pool.map(_pull, stale):
                if not news["ok"]:
                    errors += 1
                    # keep any previous items rather than blanking the ticker
                    if t in cache:
                        continue
                cache[t] = {
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "items": sec["items"] + news["items"],
                    "news_ok": news["ok"], "sec_ok": sec["ok"],
                }
        _save_cache(cache)

    out = {t: (cache.get(t) or {}).get("items", []) for t in tickers}
    stats = {
        "tickers": len(tickers),
        "refreshed": len(stale),
        "fetch_errors": errors,
        "with_items": sum(1 for v in out.values() if v),
        "items": sum(len(v) for v in out.values()),
        "eight_k": sum(1 for v in out.values() for i in v if i["src"] == "8-K"),
    }
    return out, stats


# ── triage (one Haiku call) ─────────────────────────────────────────

_TRIAGE_SYSTEM = (
    "You triage company news for an equity trading system with an hours-to-days "
    "horizon. For each ticker, judge only the items given — do not use outside "
    "knowledge or assume facts not stated. Output ONLY a JSON object, no prose, "
    "no markdown fences."
)

_TRIAGE_INSTRUCTIONS = """For EVERY ticker below, return an entry keyed by ticker:
{"TICKER": {"materiality": 0-3, "direction": "bullish|bearish|neutral|mixed",
            "catalyst": "<2-5 words, e.g. earnings beat, guidance cut, M&A, exec departure, analyst action, macro/sector, none>",
            "why": "<one sentence, <=25 words, grounded in the items; do not cite item numbers>",
            "items": [<indexes of the items that matter>]}}

Materiality scale:
  0 = nothing company-specific (listicles, market-wrap mentions, generic sector chatter, no items)
  1 = minor/company-specific but unlikely to move the stock (routine analyst note, small product news)
  2 = notable: could plausibly move the stock several % (earnings, guidance, rating change from a major firm, material contract)
  3 = major: thesis-changing event (M&A, CEO exit, restatement/non-reliance, bankruptcy, delisting, big guidance reset, regulatory action)
An 8-K item list is a fact that an event occurred; weigh it by the item type.
A ticker with no items gets materiality 0, direction neutral, catalyst none.

ITEMS:
"""


def _build_triage_prompt(items_by_ticker: dict, roles: dict) -> str:
    blocks = []
    for t, items in items_by_ticker.items():
        role = roles.get(t, "")
        if not items:
            blocks.append(f"[{t}] ({role}) — no items")
            continue
        lines = [f"[{t}] ({role})"]
        for i, it in enumerate(items):
            s = f"  {i}. [{it['src']} {it['ts']}] {it['headline']}"
            if it.get("summary"):
                s += f" — {it['summary'][:300]}"
            lines.append(s)
        blocks.append("\n".join(lines))
    return _TRIAGE_INSTRUCTIONS + "\n\n".join(blocks)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except json.JSONDecodeError:
        return None


def triage(items_by_ticker: dict, roles: dict, cfg: dict) -> tuple[dict | None, dict]:
    """One Haiku call. Returns (per-ticker digest or None, call meta)."""
    meta = {"model": cfg["_model"], "ok": False}
    empty = {"materiality": 0, "direction": "neutral", "catalyst": "none",
             "why": "no items in window", "items": [], "judged": True}
    with_items = {t: v for t, v in items_by_ticker.items() if v}
    if not with_items:
        # nothing to judge — a valid, empty digest (not a failure)
        meta["ok"] = True
        return {t: dict(empty) for t in items_by_ticker}, meta
    try:
        import anthropic
        client = anthropic.Anthropic()
        t0 = time.time()
        resp = client.messages.create(
            model=cfg["_model"],
            max_tokens=16000,
            system=_TRIAGE_SYSTEM,
            messages=[{"role": "user",
                       "content": _build_triage_prompt(with_items, roles)}],
            timeout=int(cfg["triage_timeout_s"]),
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        meta.update({
            "latency_s": round(time.time() - t0, 1),
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        })
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return None, meta
    parsed = _extract_json(text)
    if not isinstance(parsed, dict):
        meta["error"] = "unparseable triage response"
        return None, meta
    digest = {}
    for t in items_by_ticker:
        if t not in with_items:
            digest[t] = dict(empty)
            continue
        d = parsed.get(t) or parsed.get(t.upper()) or {}
        try:
            mat = max(0, min(3, int(d.get("materiality", 0))))
        except (TypeError, ValueError):
            mat = 0
        digest[t] = {
            "materiality": mat,
            "direction": str(d.get("direction", "neutral"))[:10],
            "catalyst": str(d.get("catalyst", "none"))[:40],
            "why": str(d.get("why", ""))[:240],
            "items": d.get("items", []) if isinstance(d.get("items"), list) else [],
            "judged": bool(d),
        }
    meta["ok"] = True
    meta["unjudged"] = sum(1 for v in digest.values() if not v["judged"])
    return digest, meta


# ── orchestration ───────────────────────────────────────────────────

def run_news_digest(candidates: list[str], holdings: list[str]) -> dict:
    """Build the per-cycle digest for candidates + holdings. Never raises."""
    cfg = load_news_config()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    result = {"timestamp": ts, "status": "UNAVAILABLE", "tickers": {},
              "stats": {}, "triage": {}, "inject_prompt": bool(cfg["inject_prompt"])}
    if not cfg.get("enabled", True):
        result["status"] = "DISABLED"
        return result
    try:
        roles = {}
        for t in candidates:
            roles[t.upper()] = "candidate"
        for t in holdings:
            roles[t.upper()] = "held+candidate" if t.upper() in roles else "held"
        tickers = list(roles)[: int(cfg["max_tickers"])]

        items, stats = gather_items(tickers, cfg)
        digest, meta = triage(items, roles, cfg)
        result["stats"], result["triage"] = stats, meta
        if digest is not None:
            result["status"] = "OK"
            for t in tickers:
                d = dict(digest[t])
                d["role"] = roles[t]
                d["n_items"] = len(items[t])
                picked = [items[t][i] for i in d.get("items", [])
                          if isinstance(i, int) and 0 <= i < len(items[t])]
                d["evidence"] = [f"[{p['src']} {p['ts']}] {p['headline']}" for p in picked[:3]]
                d["digest_hash"] = hashlib.sha256(
                    json.dumps(items[t], sort_keys=True).encode()).hexdigest()[:12]
                result["tickers"][t] = d
    except Exception as exc:  # the cycle must never die on news
        result["triage"]["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    try:
        with open(DIGEST_FILE, "w") as f:
            json.dump(result, f, indent=2)
        with open(SHADOW_LOG, "a") as f:
            f.write(json.dumps({"timestamp": ts, "status": result["status"],
                                "stats": result["stats"], "triage": result["triage"],
                                "tickers": result["tickers"]}) + "\n")
    except Exception:
        pass
    return result


def load_digest(max_age_minutes: int = 90) -> dict | None:
    try:
        with open(DIGEST_FILE) as f:
            d = json.load(f)
        ts = datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - ts > timedelta(minutes=max_age_minutes):
            return None
        return d
    except Exception:
        return None


def format_news_section(tickers: list[str], digest: dict | None = None) -> str:
    """Prompt text for the given tickers. States UNAVAILABLE explicitly."""
    digest = digest or load_digest()
    if not digest or digest.get("status") != "OK":
        return ("NEWS DIGEST: UNAVAILABLE this cycle (fetch or triage failed). "
                "Do NOT infer that the absence of news means nothing happened; "
                "weigh signals without news confirmation.")
    lines = ["Per-ticker material news (last ~3 days; SEC 8-K + company news, triaged).",
             "Materiality 0=none 1=minor 2=notable 3=thesis-changing.", ""]
    rows = []
    for t in tickers:
        d = digest["tickers"].get(t.upper())
        if not d:
            rows.append((0, f"  {t:6} not covered this cycle"))
            continue
        head = (f"  {t:6} M{d['materiality']} {d['direction']:<8} "
                f"{d['catalyst']} — {d['why']}")
        ev = "".join(f"\n           {e}" for e in d.get("evidence", []))
        rows.append((d["materiality"], head + ev))
    rows.sort(key=lambda r: -r[0])
    lines += [r[1] for r in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    cands = [a.upper() for a in sys.argv[1:]] or ["AAPL", "MSFT"]
    res = run_news_digest(cands, [])
    print(json.dumps({k: res[k] for k in ("status", "stats", "triage")}, indent=2))
    print(format_news_section(cands, res))
