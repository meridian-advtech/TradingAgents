"""
Kairos Automated Tier C Intake Engine

Polls three external data sources for actionable insider/congressional trade
signals, validates and deduplicates candidates, and auto-adds qualifying
tickers to Tier C via kairos_tier_c.py.

Sources:
  1. SEC EDGAR Form 4 RSS  — insider purchases >= $500K
  2. OpenInsider RSS        — insider purchases >= $500K
  3. Congressional trades   — House + Senate recent purchases

Runs on the scheduler interval defined in kairos_config.json (default: every
2 cycles = 60 minutes).

Usage:
  python kairos_intake.py           # Full intake run (all 3 sources)
  python kairos_intake.py --dry-run # Fetch + evaluate but don't add
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

INTAKE_LOG = os.path.join(SCRIPT_DIR, "kairos_intake_log.json")
UNIVERSE_FILE = os.path.join(SCRIPT_DIR, "kairos_universe.json")
TIER_C_FILE = os.path.join(SCRIPT_DIR, "kairos_tier_c.json")
STATE_FILE = os.path.join(SCRIPT_DIR, "kairos_state.json")

W = 72

# SEC requires a User-Agent header identifying the requestor
SEC_HEADERS = {
    "User-Agent": "Kairos Trading System jelmore@kairos.local",
    "Accept": "application/atom+xml, application/xml, text/xml",
}
FETCH_TIMEOUT = 30

# Per-run flood control. Fetchers collect *all* qualifying candidates;
# run_intake then ranks by value (descending) and applies these caps:
#   - per source: top N candidates from each source survive
#   - per run:    top M candidates across all sources survive
# Anything that survived validation but lost the ranking is recorded in
# the intake log under `capped_tickers` (with reason per_source / per_run)
# so we can see what was left on the table.
MAX_ADDS_PER_SOURCE = 3
MAX_ADDS_PER_RUN = 5
# Of the per-run slots, reserve N specifically for the top congressional
# candidate by value. STOCK Act dollar amounts are bracketed (typically
# $15K-$100K) and would always lose a pure value-rank tiebreak against
# EDGAR / OpenInsider, so the reservation guarantees Congress always
# has a voice when signals exist. Unused congress slots (when no
# congressional candidate qualifies that cycle) are released back to
# the general EDGAR/OpenInsider pool.
RESERVED_CONGRESS_SLOTS = 1

# Sanity ceiling for ranking. OpenInsider occasionally emits sentinel
# values (e.g. 2^31-1 = $2,147,483,647) for rows where the dollar amount
# couldn't be parsed. Any candidate above this ceiling has its `value`
# clamped to 0 before ranking, dropping it to the bottom of the stack
# so a parsing artifact can never win the cap. The candidate still
# passes through validation and is eligible to be added if it survives
# the cap with real competition — we just refuse to *promote* it.
VALUE_RANK_CEILING = 1_000_000_000  # $1B


def banner(text: str) -> str:
    return f"\n{'━' * W}\n  {text}\n{'━' * W}"


# ── Shared helpers ───────────────────────────────────────────────────

def _existing_tickers() -> set[str]:
    """All tickers already in Tier A, B, or C."""
    tickers = set()
    try:
        with open(UNIVERSE_FILE) as f:
            universe = json.load(f)
        for _cat, syms in universe.get("tier_a", {}).get("equities", {}).items():
            tickers.update(syms)
        for _cat, syms in universe.get("tier_a", {}).get("etfs", {}).items():
            tickers.update(syms)
        for entry in universe.get("tier_b", {}).get("tickers", []):
            sym = entry["symbol"] if isinstance(entry, dict) else entry
            tickers.add(sym)
    except (IOError, json.JSONDecodeError):
        pass

    try:
        with open(TIER_C_FILE) as f:
            tier_c = json.load(f)
        for entry in tier_c:
            tickers.add(entry["ticker"])
    except (IOError, json.JSONDecodeError):
        pass

    return tickers


def _validate_ticker_yf(ticker: str) -> dict | None:
    """Validate ticker via yfinance. Returns info dict or None."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        if info and info.get("symbol"):
            return info
    except Exception:
        pass
    return None


def _add_to_tier_c(ticker: str, name: str, sector: str, reason: str,
                   source: str) -> bool:
    """Add ticker to Tier C and send auto-add Slack alert.

    Returns True if added, False if skipped/failed.
    """
    from kairos_tier_c import add as tier_c_add
    result = tier_c_add(ticker, name, sector, reason)
    if not result["ok"]:
        return False

    entry = result["entry"]
    # Slack alert already sent by kairos_tier_c.add() — skip duplicate
    return True


def _get_cycle() -> int:
    """Read current cycle_count from state file."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f).get("cycle_count", 0)
    except (IOError, json.JSONDecodeError):
        return 0


def _make_stats() -> dict:
    """Per-source stats shape shared across EDGAR / OpenInsider / Congress.

    Semantics (normalized across sources):
      evaluated        candidates that passed type + value threshold
                       (pre-dedup, pre-yfinance)
      added            successfully added to Tier C
      skipped          total rejected post-evaluation = skipped_dedup + skipped_yf
                       (kept for back-compat with kairos_health.py)
      skipped_dedup    rejected because ticker already in universe
      skipped_yf       rejected because yfinance couldn't resolve ticker
      below_threshold  rejected because value < source threshold
                       (always 0 for congress — no value filter)
    """
    return {
        "evaluated": 0,
        "added": 0,
        "skipped": 0,
        "skipped_dedup": 0,
        "skipped_yf": 0,
        "below_threshold": 0,
        "capped": 0,
    }


def _allocate_run_slots(pending: list[tuple[str, dict]],
                        total_cap: int,
                        reserved_congress: int
                        ) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    """Apply the per-run cap with a reserved Congress carve-out.

    `pending` is the post-per-source-cap stream as [(source_name, candidate)].
    Returns (winners, losers). Both are ranked by value desc within
    their pool. Unused congress slots are released to the general pool
    (so a cycle with no congressional candidates still fills all 5
    general slots).
    """
    congress = sorted([sc for sc in pending if sc[0] == "congress"],
                      key=lambda sc: sc[1]["value"], reverse=True)
    other = sorted([sc for sc in pending if sc[0] != "congress"],
                   key=lambda sc: sc[1]["value"], reverse=True)

    other_slots = total_cap - reserved_congress
    congress_winners = congress[:reserved_congress]
    congress_losers = congress[reserved_congress:]
    other_winners = other[:other_slots]
    other_losers = other[other_slots:]

    # Backfill: release unused congress slots to the other pool.
    unused = reserved_congress - len(congress_winners)
    if unused > 0 and other_losers:
        backfill = other_losers[:unused]
        other_winners.extend(backfill)
        other_losers = other_losers[unused:]

    return congress_winners + other_winners, congress_losers + other_losers


def _parse_congress_value(tx: dict) -> float:
    """Best-effort numeric value extraction from a Quiver Congress trade.

    Quiver typically encodes amount as a bracketed range string in the
    `Range` field (e.g. "$1,001 - $15,000"); we use the upper bound.
    Falls back to a numeric `Amount` / `TradeSize` field if present, or 0.
    A value of 0 means the candidate sorts last in the ranking pass —
    intentional, since we should not promote unknown-sized signals over
    sources where we have hard dollar amounts (EDGAR / OpenInsider).
    """
    r = tx.get("Range") or ""
    matches = re.findall(r"\$[\d,]+", str(r))
    if matches:
        try:
            return float(matches[-1].replace("$", "").replace(",", ""))
        except ValueError:
            pass
    for k in ("Amount", "TradeSize", "trade_size_usd"):
        v = tx.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _dbg(enabled: bool, source: str, ticker: str,
         value: float, outcome: str) -> None:
    """Per-candidate trace line; no-op unless `enabled`."""
    if not enabled:
        return
    v_str = f"${value:>12,.0f}" if value else " " * 13
    print(f"    [{source:<11}] {ticker:<8} {v_str}  → {outcome}")


# ── SOURCE 1: SEC EDGAR Form 4 (via EFTS API + ownership XML) ────────

def _fetch_edgar(existing: set[str], added_this_run: set[str],
                 dry_run: bool = False, debug: bool = False) -> dict:
    """Poll SEC EDGAR for recent Form 4 filings with large insider purchases.

    Uses the EFTS full-text search API to find recent Form 4 filings,
    then fetches each filing's ownership.xml to extract transaction details.
    """
    import requests
    import xml.etree.ElementTree as ET

    stats = _make_stats()
    candidates: list[dict] = []
    errors = []

    # EFTS search for Form 4 filings from the last 3 days
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
    url = (f"https://efts.sec.gov/LATEST/search-index"
           f"?forms=4&dateRange=custom&startdt={start}&enddt={today}")

    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        errors.append(f"EDGAR EFTS fetch failed: {exc}")
        print(f"  ERROR: EDGAR EFTS fetch failed: {exc}")
        return {"stats": stats, "candidates": candidates, "errors": errors}

    hits = data.get("hits", {}).get("hits", [])
    print(f"  EDGAR EFTS: {len(hits)} Form 4 filings found")

    for hit in hits:
        try:
            src = hit.get("_source", {})
            filing_id = hit.get("_id", "")  # e.g. "0001193125-26-152002:ownership.xml"
            if ":" not in filing_id:
                continue

            accession, xml_filename = filing_id.split(":", 1)
            ciks = src.get("ciks", [])
            if not ciks:
                continue

            # Build URL to the ownership XML
            accession_nodash = accession.replace("-", "")
            xml_url = (f"https://www.sec.gov/Archives/edgar/data/"
                       f"{ciks[0]}/{accession_nodash}/{xml_filename}")

            time.sleep(0.12)  # SEC rate limit: ~10 req/sec
            try:
                xml_resp = requests.get(xml_url, headers=SEC_HEADERS,
                                        timeout=FETCH_TIMEOUT)
                xml_resp.raise_for_status()
                xml_text = xml_resp.text
            except Exception:
                continue

            # Parse ownership XML
            try:
                doc = ET.fromstring(xml_text)
            except ET.ParseError:
                continue

            # Extract issuer ticker
            ticker_el = doc.find(".//issuerTradingSymbol")
            if ticker_el is None or not ticker_el.text:
                continue
            ticker = ticker_el.text.strip().upper()
            if not re.match(r'^[A-Z]{1,5}$', ticker):
                continue

            # Extract insider name
            owner_el = doc.find(".//rptOwnerName")
            insider_name = owner_el.text.strip() if owner_el is not None and owner_el.text else "Unknown"

            # Walk all nonDerivativeTransaction elements for purchases
            best_purchase_value = 0.0
            best_shares = 0
            best_price = 0.0

            for tx in doc.findall(".//nonDerivativeTransaction"):
                code_el = tx.find(".//transactionCode")
                if code_el is None or (code_el.text or "").strip() != "P":
                    continue

                shares_el = tx.find(".//transactionShares/value")
                price_el = tx.find(".//transactionPricePerShare/value")
                shares = float(shares_el.text.replace(",", "")) if shares_el is not None and shares_el.text else 0
                price = float(price_el.text.replace(",", "")) if price_el is not None and price_el.text else 0
                value = shares * price
                if value > best_purchase_value:
                    best_purchase_value = value
                    best_shares = shares
                    best_price = price

            if best_purchase_value == 0:
                # No purchase transactions in this filing — not counted anywhere
                continue

            # Filter: >= $500K purchase (pre-evaluation)
            if best_purchase_value < 500_000:
                stats["below_threshold"] += 1
                _dbg(debug, "edgar", ticker, best_purchase_value, "below $500K threshold")
                continue

            # Passed threshold → count as evaluated
            stats["evaluated"] += 1

            # Dedup
            if ticker in existing or ticker in added_this_run:
                stats["skipped_dedup"] += 1
                stats["skipped"] += 1
                _dbg(debug, "edgar", ticker, best_purchase_value, "dedup (already in universe)")
                continue

            # yfinance validation
            info = _validate_ticker_yf(ticker)
            if not info:
                stats["skipped_yf"] += 1
                stats["skipped"] += 1
                _dbg(debug, "edgar", ticker, best_purchase_value, "yfinance can't resolve")
                continue

            company_name = info.get("longName") or info.get("shortName") or ticker
            sector = info.get("sector") or "Unknown"
            value_m = best_purchase_value / 1_000_000

            reason = (f"SEC EDGAR Form 4: ${value_m:.1f}m insider purchase "
                      f"by {insider_name} ({int(best_shares)} shares @ ${best_price:.2f})")

            # Candidate passes all per-source filters — queue for the
            # ranking + capping pass in run_intake. Reserve the ticker so
            # later sources don't re-evaluate it (treats "queued" the same
            # as "added" for dedup purposes; minor inconsistency if this
            # candidate is later capped out, but not worth tracking).
            candidates.append({
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "reason": reason,
                "source_label": "SEC EDGAR Form 4",
                "value": best_purchase_value,
            })
            added_this_run.add(ticker)
            existing.add(ticker)
            _dbg(debug, "edgar", ticker, best_purchase_value, "queued (pre-cap)")

        except Exception as exc:
            errors.append(f"EDGAR entry parse error: {exc}")
            continue

    return {"stats": stats, "candidates": candidates, "errors": errors}


# ── SOURCE 2: OpenInsider (HTML table scrape) ────────────────────────

def _fetch_openinsider(existing: set[str], added_this_run: set[str],
                       dry_run: bool = False, debug: bool = False) -> dict:
    """Scrape OpenInsider's insider-purchases page for purchases >= $500K.

    OpenInsider dropped their RSS feed; we parse the HTML table from
    their pre-built insider purchases page instead.
    """
    import requests

    stats = _make_stats()
    candidates: list[dict] = []
    errors = []

    url = "http://openinsider.com/insider-purchases-25k"

    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT,
                            headers={"User-Agent": "Kairos/1.0"})
        resp.raise_for_status()
    except Exception as exc:
        errors.append(f"OpenInsider fetch failed: {exc}")
        print(f"  ERROR: OpenInsider fetch failed: {exc}")
        return {"stats": stats, "candidates": candidates, "errors": errors}

    # Find the data table — it's the large table (>10 rows) after tablewrapper
    # Headers: X, Filing Date, Trade Date, Ticker, Company Name, Insider Name,
    #          Title, Trade Type, Price, Qty, Owned, ΔOwn, Value, 1d, 1w, 1m, 6m
    all_tables = re.findall(r'<table[^>]*>(.*?)</table>', resp.text, re.DOTALL)
    data_table = None
    for t in all_tables:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', t, re.DOTALL)
        if len(rows) > 10:
            # Verify by checking header row for expected columns
            first_row = rows[0] if rows else ""
            if "Ticker" in first_row or "Trade" in first_row:
                data_table = rows
                break

    if not data_table:
        errors.append("OpenInsider: could not find data table in HTML")
        print("  ERROR: Could not locate data table in OpenInsider HTML")
        return {"stats": stats, "candidates": candidates, "errors": errors}

    print(f"  OpenInsider: {len(data_table) - 1} rows in table")

    def _strip_html(s):
        return re.sub(r'<[^>]+>', '', s).strip()

    # Skip header row (index 0)
    for row_html in data_table[1:]:
        try:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
            if len(cells) < 13:
                continue

            # Columns: 0=X, 1=Filing Date, 2=Trade Date, 3=Ticker, 4=Company,
            #          5=Insider Name, 6=Title, 7=Trade Type, 8=Price, 9=Qty,
            #          10=Owned, 11=ΔOwn, 12=Value
            trade_type = _strip_html(cells[7])
            if "P - Purchase" not in trade_type:
                continue

            # Extract ticker — may be wrapped in <a> or JS
            ticker_cell = cells[3]
            ticker_match = re.search(r'>([A-Z]{1,5})<', ticker_cell)
            if not ticker_match:
                # Fallback: extract from href
                ticker_match = re.search(r'/([A-Z]{1,5})', ticker_cell)
            if not ticker_match:
                continue
            ticker = ticker_match.group(1).upper()

            # Extract value — format like "+$105,960" or "+$2,975,919"
            value_str = _strip_html(cells[12])
            value_clean = re.sub(r'[+$,\s]', '', value_str)
            if not value_clean:
                continue
            try:
                value = float(value_clean)
            except ValueError:
                continue

            insider_name = _strip_html(cells[5])

            # Filter: >= $500K (pre-evaluation)
            if value < 500_000:
                stats["below_threshold"] += 1
                _dbg(debug, "openinsider", ticker, value, "below $500K threshold")
                continue

            # Passed threshold → count as evaluated
            stats["evaluated"] += 1

            # Dedup
            if ticker in existing or ticker in added_this_run:
                stats["skipped_dedup"] += 1
                stats["skipped"] += 1
                _dbg(debug, "openinsider", ticker, value, "dedup (already in universe)")
                continue

            info = _validate_ticker_yf(ticker)
            if not info:
                stats["skipped_yf"] += 1
                stats["skipped"] += 1
                _dbg(debug, "openinsider", ticker, value, "yfinance can't resolve")
                continue

            company_name = info.get("longName") or info.get("shortName") or ticker
            sector = info.get("sector") or "Unknown"
            value_fmt = f"${value/1000:.0f}K" if value < 1_000_000 else f"${value/1_000_000:.1f}M"

            reason = f"OpenInsider: {value_fmt} purchase by {insider_name}"

            # Queue for capping pass in run_intake.
            candidates.append({
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "reason": reason,
                "source_label": "OpenInsider",
                "value": value,
            })
            added_this_run.add(ticker)
            existing.add(ticker)
            _dbg(debug, "openinsider", ticker, value, "queued (pre-cap)")

        except Exception as exc:
            errors.append(f"OpenInsider entry parse error: {exc}")
            continue

    return {"stats": stats, "candidates": candidates, "errors": errors}


# ── SOURCE 3: Congressional trades (via QuiverQuant API) ─────────────

def _fetch_congress(existing: set[str], added_this_run: set[str],
                    dry_run: bool = False, debug: bool = False) -> dict:
    """Poll House + Senate stock trade disclosures (last 48h purchases).

    Uses the QuiverQuant live API which aggregates STOCK Act filings.
    Both /housetrading and /senatetrading now require authentication —
    we pass CONGRESS_API_KEY as `Authorization: Token <key>`. The
    scheduler exports this env var from ~/.zshrc (kairos_scheduler.sh).
    If the key is missing we short-circuit without spamming 401 errors.

    Fields: Representative/Senator, Date, Ticker, Transaction, Range, Amount.
    """
    import requests

    stats = _make_stats()
    candidates: list[dict] = []
    errors = []

    api_key = os.environ.get("CONGRESS_API_KEY", "").strip()
    if not api_key:
        msg = ("Congress source disabled: CONGRESS_API_KEY not set "
               "(needed for Quiver house/senate endpoints)")
        errors.append(msg)
        print(f"  NOTE: {msg}")
        return {"stats": stats, "candidates": candidates, "errors": errors}

    urls = {
        "House": ("https://api.quiverquant.com/beta/live/housetrading", "Representative"),
        "Senate": ("https://api.quiverquant.com/beta/live/senatetrading", "Senator"),
    }

    headers = {
        "Authorization": f"Token {api_key}",
        "Accept": "application/json",
        "User-Agent": "Kairos/1.0",
    }

    # 7-day window. STOCK Act filings are often disclosed in small batches
    # so a 48h window caught almost nothing; 7d comfortably spans a weekly
    # disclosure cadence without picking up stale transactions.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    for chamber, (url, member_key) in urls.items():
        try:
            resp = requests.get(url, timeout=FETCH_TIMEOUT, headers=headers)
            resp.raise_for_status()
            transactions = resp.json()
        except Exception as exc:
            errors.append(f"{chamber} fetch failed: {exc}")
            print(f"  ERROR: {chamber} feed failed: {exc}")
            continue

        if not isinstance(transactions, list):
            errors.append(f"{chamber}: unexpected response format")
            continue

        print(f"  {chamber}: {len(transactions)} total transactions")

        # Pre-evaluation tallies — summarized at end of chamber loop when
        # verbose, so we can confirm the type filter is matching what the
        # feed actually contains (rather than flooding 5000 lines per chamber).
        pre_bad_ticker = 0
        pre_old_date = 0
        pre_not_purchase = 0
        type_hist: dict[str, int] = {}

        for tx in transactions:
            try:
                ticker = (tx.get("Ticker") or "").strip().upper()
                if not ticker or ticker in ("--", "N/A", "", "NONE"):
                    pre_bad_ticker += 1
                    continue

                # Check date freshness
                tx_date = tx.get("Date") or ""
                if not tx_date or tx_date < cutoff:
                    pre_old_date += 1
                    continue

                # Check transaction type — must be a purchase. Tally the
                # raw type string (post-date-filter) so debug output shows
                # what the purchase-regex is actually up against.
                tx_type = (tx.get("Transaction") or "").lower()
                type_hist[tx_type] = type_hist.get(tx_type, 0) + 1
                if "purchase" not in tx_type:
                    pre_not_purchase += 1
                    continue

                # Passed inclusion criteria → count as evaluated
                # (no value threshold for this source; below_threshold always 0)
                stats["evaluated"] += 1

                # Extract member info + best-effort dollar value (used by
                # the cross-source ranking pass; 0 if Quiver doesn't
                # expose a parsable Range / Amount).
                member = tx.get(member_key) or "Unknown"
                tx_value = _parse_congress_value(tx)

                # Dedup
                if ticker in existing or ticker in added_this_run:
                    stats["skipped_dedup"] += 1
                    stats["skipped"] += 1
                    _dbg(debug, f"congress-{chamber.lower()}", ticker, tx_value,
                         "dedup (already in universe)")
                    continue

                info = _validate_ticker_yf(ticker)
                if not info:
                    stats["skipped_yf"] += 1
                    stats["skipped"] += 1
                    _dbg(debug, f"congress-{chamber.lower()}", ticker, tx_value,
                         "yfinance can't resolve")
                    continue

                company_name = info.get("longName") or info.get("shortName") or ticker
                sector = info.get("sector") or "Unknown"

                reason = (f"Congressional trade: {member} "
                          f"purchase disclosed {tx_date}")

                # Queue for capping pass in run_intake.
                candidates.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "sector": sector,
                    "reason": reason,
                    "source_label": f"Congressional ({chamber})",
                    "value": tx_value,
                })
                added_this_run.add(ticker)
                existing.add(ticker)
                _dbg(debug, f"congress-{chamber.lower()}", ticker, tx_value,
                     "queued (pre-cap)")

            except Exception as exc:
                errors.append(f"{chamber} entry parse error: {exc}")
                continue

        # End-of-chamber pre-eval summary. Printed unconditionally (cheap,
        # one line) so the log always shows how the date/type filters
        # chewed through the feed. Type histogram is verbose-only because
        # it can be long.
        pre_passed = (len(transactions) - pre_bad_ticker
                      - pre_old_date - pre_not_purchase)
        print(f"    [{chamber} pre-eval] cutoff={cutoff}  "
              f"bad_ticker={pre_bad_ticker}  old_date={pre_old_date}  "
              f"not_purchase={pre_not_purchase}  passed={pre_passed}")
        if debug and type_hist:
            top = sorted(type_hist.items(), key=lambda kv: -kv[1])[:8]
            print(f"    [{chamber} pre-eval] "
                  f"top Transaction types in date window:")
            for t, n in top:
                print(f"      {n:>5} × {t!r}")

    return {"stats": stats, "candidates": candidates, "errors": errors}


# ── Main intake runner ───────────────────────────────────────────────

def run_intake(dry_run: bool = False, verbose: bool | None = None) -> dict:
    """Run the full intake pipeline across all three sources.

    Args:
        dry_run: Evaluate sources but do not add to Tier C.
        verbose: If True, emit per-candidate trace lines from every source.
            If None (default), respects INTAKE_DEBUG env var
            (values "1", "true", "yes" enable it).

    Returns a summary dict for logging.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    cycle = _get_cycle()

    if verbose is None:
        verbose = os.environ.get("INTAKE_DEBUG", "").lower() in ("1", "true", "yes")

    print("╔" + "═" * W + "╗")
    print(f"║  KAIROS TIER C INTAKE ENGINE — cycle {cycle}".ljust(W + 1) + "║")
    print("╚" + "═" * W + "╝")

    if dry_run:
        print("  *** DRY RUN — will evaluate but not add ***")
    if verbose:
        print("  *** VERBOSE — per-candidate trace enabled ***")

    existing = _existing_tickers()
    added_this_run: set[str] = set()
    all_added: list[str] = []
    all_errors: list[str] = []
    source_results: dict[str, dict] = {}
    capped_records: list[dict] = []

    # ── Phase 1: collect candidates from each source ───────────────────
    print(banner("Source 1: SEC EDGAR Form 4"))
    edgar = _fetch_edgar(existing, added_this_run, dry_run=dry_run, debug=verbose)
    source_results["edgar"] = edgar["stats"]
    all_errors.extend(edgar["errors"])

    print(banner("Source 2: OpenInsider"))
    oi = _fetch_openinsider(existing, added_this_run, dry_run=dry_run, debug=verbose)
    source_results["openinsider"] = oi["stats"]
    all_errors.extend(oi["errors"])

    print(banner("Source 3: Congressional Trades"))
    congress = _fetch_congress(existing, added_this_run, dry_run=dry_run, debug=verbose)
    source_results["congress"] = congress["stats"]
    all_errors.extend(congress["errors"])

    # ── Phase 2: per-source cap (top N by value) ───────────────────────
    per_source_results = {
        "edgar": edgar,
        "openinsider": oi,
        "congress": congress,
    }
    # Clamp parsing-artifact values (>$1B) to 0 so they sort last.
    for result in per_source_results.values():
        for c in result["candidates"]:
            if c["value"] > VALUE_RANK_CEILING:
                c["value"] = 0.0
    pending: list[tuple[str, dict]] = []   # [(source_name, candidate), ...]
    for src_name, result in per_source_results.items():
        cands = sorted(result["candidates"],
                       key=lambda c: c["value"], reverse=True)
        kept = cands[:MAX_ADDS_PER_SOURCE]
        dropped = cands[MAX_ADDS_PER_SOURCE:]
        for c in dropped:
            source_results[src_name]["capped"] += 1
            capped_records.append({
                "ticker": c["ticker"],
                "source": src_name,
                "value": c["value"],
                "reason": "per_source",
            })
        for c in kept:
            pending.append((src_name, c))

    # ── Phase 3: per-run cap with reserved Congress carve-out ──────────
    # Top RESERVED_CONGRESS_SLOTS congressional candidate(s) get
    # dedicated slot(s); remaining slots (4 by default) rank EDGAR +
    # OpenInsider candidates strictly by value. Unused congress slots
    # spill to the general pool.
    winners, losers = _allocate_run_slots(
        pending, MAX_ADDS_PER_RUN, RESERVED_CONGRESS_SLOTS,
    )
    for src_name, c in losers:
        source_results[src_name]["capped"] += 1
        capped_records.append({
            "ticker": c["ticker"],
            "source": src_name,
            "value": c["value"],
            "reason": "per_run",
        })

    # ── Phase 4: actually add (or dry-run-add) the survivors ───────────
    print(banner("Add Phase (post-cap)"))
    print(f"  Per-source cap: {MAX_ADDS_PER_SOURCE}   "
          f"Per-run cap: {MAX_ADDS_PER_RUN} "
          f"(reserved Congress: {RESERVED_CONGRESS_SLOTS})   "
          f"Surviving: {len(winners)}   Capped: {len(capped_records)}")
    for src_name, c in winners:
        if dry_run:
            print(f"    DRY RUN — would add {c['ticker']} (${c['value']:,.0f}): {c['reason']}")
            source_results[src_name]["added"] += 1
            all_added.append(c["ticker"])
        else:
            ok = _add_to_tier_c(c["ticker"], c["company_name"], c["sector"],
                                c["reason"], c["source_label"])
            if ok:
                source_results[src_name]["added"] += 1
                all_added.append(c["ticker"])
            else:
                # tier_c.add rejected (e.g. duplicate or quota at that
                # layer); count as dedup so the math still ties out.
                source_results[src_name]["skipped_dedup"] += 1
                source_results[src_name]["skipped"] += 1
                print(f"    tier_c rejected {c['ticker']}")

    # ── Per-source stats summary (after capping is finalized) ──────────
    def _print_stats(label: str, s: dict) -> None:
        print(f"  [{label:<11}] eval={s['evaluated']}  added={s['added']}  "
              f"capped={s.get('capped',0)}  "
              f"skipped={s['skipped']} "
              f"(dedup={s.get('skipped_dedup',0)}, "
              f"yf={s.get('skipped_yf',0)})  "
              f"below_thr={s.get('below_threshold',0)}")
    print()
    for src_name in ("edgar", "openinsider", "congress"):
        _print_stats(src_name, source_results[src_name])

    # ── Summary ─────────────────────────────────────────────────────────
    print(banner("Intake Summary"))
    print(f"  Total added: {len(all_added)} (cap: {MAX_ADDS_PER_RUN})")
    if all_added:
        print(f"  Tickers: {', '.join(all_added)}")
    if capped_records:
        print(f"  Capped (left on table): {len(capped_records)}")
        for c in capped_records:
            print(f"    • {c['ticker']:<6} ${c['value']:>12,.0f}  "
                  f"[{c['source']}, {c['reason']}]")
    if all_errors:
        print(f"  Errors: {len(all_errors)}")
        for e in all_errors[:5]:
            print(f"    • {e}")

    # ── Log to file ─────────────────────────────────────────────────────
    log_entry = {
        "timestamp": timestamp,
        "cycle": cycle,
        "sources": source_results,
        "added_tickers": all_added,
        "capped_tickers": capped_records,
        "errors": all_errors,
    }
    _write_intake_log(log_entry)

    return log_entry


def _write_intake_log(entry: dict) -> None:
    """Append intake run summary to kairos_intake_log.json (one JSON per line)."""
    try:
        with open(INTAKE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  Intake log → {INTAKE_LOG}")
    except IOError as exc:
        print(f"  WARNING: Could not write intake log: {exc}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kairos Tier C Intake Engine")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate sources but don't add to Tier C")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Emit per-candidate trace line (ticker, value, "
                             "filter outcome) for each evaluated row. "
                             "Also enabled by INTAKE_DEBUG=1 env var.")
    args = parser.parse_args()

    run_intake(dry_run=args.dry_run,
               verbose=True if args.verbose else None)


if __name__ == "__main__":
    main()
