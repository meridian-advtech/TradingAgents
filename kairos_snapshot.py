"""
Kairos Snapshot — pulls data from five sources and prints a combined summary.

Sources:
  1. Finnhub   — AAPL news headlines
  2. FRED      — Latest Fed funds rate
  3. Kalshi    — Event contract market probabilities
  4. Congress  — Recent finance-related bills
  5. CoinGecko — BTC & ETH price/volume (free, no key)

Env vars expected: FINNHUB_API_KEY, FRED_API_KEY, KALSHI_API_KEY, CONGRESS_API_KEY
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kairos_snapshot.txt")
NOW = datetime.now(timezone.utc)
SECTION_WIDTH = 72
TIMEOUT = 15


# ── helpers ──────────────────────────────────────────────────────────

def heading(title: str) -> str:
    return f"\n{'━' * SECTION_WIDTH}\n  {title}\n{'━' * SECTION_WIDTH}"


def truncate(text: str, length: int = 90) -> str:
    return text[:length] + "…" if len(text) > length else text


def fmt_usd(val: float) -> str:
    if val >= 1_000_000_000:
        return f"${val / 1e9:,.2f}B"
    if val >= 1_000_000:
        return f"${val / 1e6:,.2f}M"
    if val >= 1_000:
        return f"${val / 1e3:,.1f}K"
    return f"${val:,.2f}"


# ── data fetchers (each returns a list of formatted lines) ───────────

def fetch_finnhub() -> list[str]:
    """Finnhub: recent AAPL news headlines."""
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return ["  ⚠ FINNHUB_API_KEY not set — skipped"]

    today = NOW.strftime("%Y-%m-%d")
    week_ago = (NOW - timedelta(days=7)).strftime("%Y-%m-%d")
    resp = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={"symbol": "AAPL", "from": week_ago, "to": today, "token": key},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    articles = resp.json()[:8]

    if not articles:
        return ["  No recent headlines found."]

    lines = []
    for i, a in enumerate(articles, 1):
        ts = datetime.fromtimestamp(a["datetime"], tz=timezone.utc).strftime("%m-%d %H:%M")
        lines.append(f"  {i}. [{ts}] {truncate(a['headline'])}")
        lines.append(f"     Source: {a.get('source', 'N/A')}")
    return lines


def fetch_fred() -> list[str]:
    """FRED: latest effective federal funds rate (DFF series)."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return ["  ⚠ FRED_API_KEY not set — skipped"]

    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": "DFF",
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    obs = resp.json().get("observations", [])

    if not obs:
        return ["  No observations returned."]

    lines = []
    for o in obs:
        val = o["value"]
        display = f"{float(val):.2f}%" if val != "." else "N/A"
        lines.append(f"  {o['date']}  Fed Funds Rate: {display}")
    return lines


def fetch_kalshi() -> list[str]:
    """Kalshi: current event market probabilities (public markets)."""
    key = os.environ.get("KALSHI_API_KEY")

    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        resp = requests.get(
            "https://api.elections.kalshi.com/trade-api/v2/events",
            params={"limit": 6, "status": "open"},
            headers=headers,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            # Try the public trading API endpoint instead
            try:
                resp = requests.get(
                    "https://trading-api.kalshi.com/trade-api/v2/events",
                    params={"limit": 6, "status": "open"},
                    headers={"Accept": "application/json"},
                    timeout=TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                return ["  ⚠ Kalshi API auth failed — check KALSHI_API_KEY"]
        else:
            raise

    events = data.get("events", [])
    if not events:
        return ["  No open events found."]

    lines = []
    for i, ev in enumerate(events, 1):
        title = truncate(ev.get("title", ev.get("ticker", "Unknown")), 65)
        category = ev.get("category", "N/A")
        markets_count = ev.get("markets_count", len(ev.get("markets", [])))
        lines.append(f"  {i}. {title}")
        lines.append(f"     Category: {category} | Markets: {markets_count}")
    return lines


def fetch_congress() -> list[str]:
    """Congress.gov: latest bills matching 'finance'."""
    key = os.environ.get("CONGRESS_API_KEY")
    if not key:
        return ["  ⚠ CONGRESS_API_KEY not set — skipped"]

    resp = requests.get(
        "https://api.congress.gov/v3/bill",
        params={
            "api_key": key,
            "format": "json",
            "limit": 6,
            "sort": "updateDate+desc",
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    bills = resp.json().get("bills", [])

    # Filter to finance-related bills client-side
    finance_kw = {"finance", "financial", "tax", "fiscal", "banking", "securities",
                  "investment", "treasury", "budget", "economic", "monetary", "tariff", "trade"}
    matched = []
    for b in bills:
        title_lower = b.get("title", "").lower()
        if any(kw in title_lower for kw in finance_kw):
            matched.append(b)

    # If keyword filter yields nothing, show all returned bills
    display = matched if matched else bills

    if not display:
        return ["  No recent bills found."]

    lines = []
    for i, b in enumerate(display[:6], 1):
        bill_id = f"{b.get('type', '?')}{b.get('number', '?')}"
        congress = b.get("congress", "?")
        updated = b.get("updateDate", "N/A")[:10]
        title = truncate(b.get("title", "Untitled"), 60)
        lines.append(f"  {i}. [{congress}th] {bill_id} — {title}")
        lines.append(f"     Updated: {updated}")
    return lines


def fetch_coingecko() -> list[str]:
    """CoinGecko: BTC and ETH price & 24h volume (free, no key)."""
    resp = requests.get(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={
            "vs_currency": "usd",
            "ids": "bitcoin,ethereum",
            "order": "market_cap_desc",
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    coins = resp.json()

    if not coins:
        return ["  No data returned."]

    lines = []
    for c in coins:
        symbol = c["symbol"].upper()
        price = f"${c['current_price']:,.2f}"
        change = c.get("price_change_percentage_24h", 0) or 0
        direction = "▲" if change >= 0 else "▼"
        vol = fmt_usd(c.get("total_volume", 0))
        mcap = fmt_usd(c.get("market_cap", 0))
        lines.append(f"  {symbol:>5}  {price:>12}  {direction} {abs(change):.2f}%  Vol: {vol}  MCap: {mcap}")
    return lines


SOURCES = {
    "Finnhub — AAPL News Headlines": fetch_finnhub,
    "FRED — Federal Funds Rate": fetch_fred,
    "Kalshi — Prediction Markets": fetch_kalshi,
    "Congress.gov — Recent Finance Bills": fetch_congress,
    "CoinGecko — Crypto (BTC / ETH)": fetch_coingecko,
}


# ── main ─────────────────────────────────────────────────────────────

def run_snapshot(print_report: bool = True, save_file: bool = True) -> tuple[str, dict[str, list[str]]]:
    """Run all data fetchers and return (report_text, results_dict).

    Can be imported by other scripts to get structured snapshot data
    without printing or saving to disk.
    """
    timestamp = NOW.strftime("%Y-%m-%d %H:%M:%S UTC")

    results: dict[str, list[str]] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fn): name for name, fn in SOURCES.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as exc:
                errors[name] = str(exc)

    # Build report in source-declaration order
    report_lines = [
        "╔" + "═" * SECTION_WIDTH + "╗",
        f"║  KAIROS MARKET SNAPSHOT — {timestamp}".ljust(SECTION_WIDTH + 1) + "║",
        "╚" + "═" * SECTION_WIDTH + "╝",
    ]

    for name in SOURCES:
        report_lines.append(heading(name))
        if name in errors:
            report_lines.append(f"  ✗ Error: {errors[name]}")
        else:
            report_lines.extend(results[name])

    report_lines.append("\n" + "━" * SECTION_WIDTH)
    report_lines.append(f"  Snapshot complete. {len(results)}/{len(SOURCES)} sources OK.")
    if errors:
        report_lines.append(f"  Failed: {', '.join(errors.keys())}")
    report_lines.append("━" * SECTION_WIDTH + "\n")

    report = "\n".join(report_lines)

    if print_report:
        print(report)

    if save_file:
        with open(OUTPUT_FILE, "w") as f:
            f.write(report)
        if print_report:
            print(f"[Saved to {OUTPUT_FILE}]")

    return report, results


def main():
    run_snapshot()


if __name__ == "__main__":
    main()
