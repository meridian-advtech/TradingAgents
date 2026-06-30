"""
Kairos IPO Capital — Reservation, Liberation, and Release Engine

Implements the capital-allocation half of the HOT-IPO architecture
(Craft: "HOT-IPO: IPO Tracking & Capital Allocation Architecture").

  Phase 1 (T-5 trading days → pricing): reserve_capital()
  Phase 2 (pricing executes): convert_reservation_to_position()
  Phase 3 (pulled / score decay): release_reservation()

The pricing of an IPO with conviction score S earns a percentage of
"available capital" (cash + capital liberatable from stale positions):

      S = 7  →  5%
      S = 8  → 10%
      S = 9  → 20%
      S = 10 → 30%

If cash alone covers the target, no positions are touched. If it
doesn't, the Liberation Engine identifies stale positions that meet at
least one liberation criterion AND whose active conviction is at least
REPLACEMENT_DELTA below the IPO's score. The minimum number of such
positions are sold to free the gap. Default behaviour is dry-run; real
sells require dry_run=False so an autopilot accident can't churn the
book.

State lives in kairos.db:

    ipo_reservations  — one row per reservation, lifecycle is the
                        `status` column ('reserved' | 'converted' |
                        'released').
    ipo_liberations   — one row per liquidated lot per reservation,
                        keyed back to ipo_reservations.id.

Note: this module never reads or writes Renaissance Capital payloads
through an LLM; reservation/liberation decisions are pure numeric
logic (per Renaissance ToS §8(q) — see kairos_renaissance.py).
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Reuse the shared kairos.db so reservations live alongside decisions /
# outcomes / holdings.
from kairos_log_db import DB_PATH  # noqa: E402


# ── Tunables ─────────────────────────────────────────────────────────

CONVICTION_TO_ALLOC_PCT: dict[int, float] = {
    7:  0.05,
    8:  0.10,
    9:  0.20,
    10: 0.30,
}
MIN_RESERVATION_SCORE = 7
RESERVATION_LEAD_DAYS = 5
REPLACEMENT_DELTA = 2                # IPO score must exceed position by ≥2
LIBERATION_LOSS_THRESHOLD = -15.0    # %
LIBERATION_MIN_HOLD_DAYS = 5
LIBERATION_STALE_CHECKPOINT_SCORE = 0.5  # below = "declining thesis"
LIBERATION_HOT_REVERSION_TARGET_PCT = 0.0  # closed/positive = reversion done

# Statuses
STATUS_RESERVED  = "reserved"
STATUS_CONVERTED = "converted"
STATUS_RELEASED  = "released"

# Slack channel for reservation/liberation notices
SLACK_CHANNEL = "alerts"

# Used for the optional self-connect helper
IB_HOST = "127.0.0.1"
IB_PORT = 7497


# ─────────────────────────────────────────────────────────────────────
# Schema + connection
# ─────────────────────────────────────────────────────────────────────

SCHEMA_IPO_RESERVATIONS = """
CREATE TABLE IF NOT EXISTS ipo_reservations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                TEXT NOT NULL,
    company_name          TEXT,
    expected_pricing_date TEXT,
    conviction_score      REAL NOT NULL,
    reserved_pct          REAL NOT NULL,
    reserved_usd          REAL NOT NULL,
    cash_at_reservation   REAL,
    nlv_at_reservation    REAL,
    available_at_reservation REAL,
    reserved_at           TEXT NOT NULL DEFAULT (datetime('now')),
    status                TEXT NOT NULL,
    released_at           TEXT,
    release_reason        TEXT,
    converted_at          TEXT,
    converted_shares      INTEGER,
    converted_avg_price   REAL,
    notes                 TEXT
);
"""

SCHEMA_IPO_LIBERATIONS = """
CREATE TABLE IF NOT EXISTS ipo_liberations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    reservation_id      INTEGER NOT NULL REFERENCES ipo_reservations(id),
    liberated_ticker    TEXT NOT NULL,
    rationale           TEXT,
    position_conviction REAL,
    position_pnl_pct    REAL,
    days_held           INTEGER,
    freed_usd           REAL NOT NULL,
    dry_run             INTEGER NOT NULL DEFAULT 1,
    sell_status         TEXT,
    sell_fill_price     REAL,
    liberated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema() -> None:
    """Idempotent — create ipo_reservations and ipo_liberations tables."""
    conn = _get_connection()
    conn.executescript(SCHEMA_IPO_RESERVATIONS + SCHEMA_IPO_LIBERATIONS)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────
# IBKR / portfolio access
# ─────────────────────────────────────────────────────────────────────

def _connect_ib(timeout: int = 10):
    """Best-effort IBKR connect; returns (ib, own_conn) or (None, False)."""
    try:
        import random
        from ib_insync import IB
        ib = IB()
        ib.connect(IB_HOST, IB_PORT, clientId=random.randint(80, 99),
                   timeout=timeout)
        return ib, True
    except Exception as exc:
        print(f"  WARNING: IBKR connect failed: {exc}")
        return None, False


def _disconnect_ib(ib) -> None:
    try:
        ib.disconnect()
    except Exception:
        pass


def fetch_portfolio_snapshot(ib=None) -> Optional[dict]:
    """Return {nlv, cash, positions} from IBKR.

    Self-connects when ib is None; releases the connection only if it
    owned it. Returns None when the broker is unreachable.
    """
    own_conn = False
    if ib is None:
        ib, own_conn = _connect_ib()
        if ib is None:
            return None
    try:
        from kairos_execute import fetch_portfolio_state
        snap = fetch_portfolio_state(ib)
    except Exception as exc:
        print(f"  WARNING: fetch_portfolio_state failed: {exc}")
        snap = None
    finally:
        if own_conn:
            _disconnect_ib(ib)
    return snap


# ─────────────────────────────────────────────────────────────────────
# Position conviction lookup (thesis_predictions in kairos_ml_outcomes.db)
# ─────────────────────────────────────────────────────────────────────

def get_active_position_conviction(ticker: str) -> Optional[float]:
    """Most recent OPEN thesis_predictions.conviction_score for a ticker.

    Returns None if there's no open prediction or the lookup fails.
    """
    if not ticker:
        return None
    try:
        from kairos_ml_thesis import _get_connection as _ml_conn
        from kairos_ml_thesis import _ensure_schema
        _ensure_schema()
    except Exception:
        return None

    try:
        conn = _ml_conn()
        row = conn.execute(
            """SELECT tp.conviction_score, tp.timestamp_entry
                 FROM thesis_predictions AS tp
                 JOIN trade_outcomes
                   ON trade_outcomes.trade_id = tp.decision_id
                WHERE trade_outcomes.outcome_label IS NULL
                  AND UPPER(tp.ticker) = ?
                ORDER BY tp.id DESC
                LIMIT 1""",
            (ticker.upper(),),
        ).fetchone()
        conn.close()
    except Exception as exc:
        print(f"  WARNING: conviction lookup for {ticker} failed: {exc}")
        return None

    if row and row["conviction_score"] is not None:
        try:
            return float(row["conviction_score"])
        except (TypeError, ValueError):
            return None
    return None


def _latest_checkpoint_score(ticker: str) -> Optional[float]:
    """Latest thesis_checkpoints.checkpoint_score for an open trade."""
    try:
        from kairos_ml_thesis import _get_connection as _ml_conn
    except Exception:
        return None
    try:
        conn = _ml_conn()
        row = conn.execute(
            """SELECT tc.checkpoint_score
                 FROM thesis_checkpoints AS tc
                 JOIN trade_outcomes
                   ON trade_outcomes.trade_id = tc.decision_id
                WHERE trade_outcomes.outcome_label IS NULL
                  AND UPPER(tc.ticker) = ?
                ORDER BY tc.id DESC
                LIMIT 1""",
            (ticker.upper(),),
        ).fetchone()
        conn.close()
    except Exception:
        return None
    if row and row["checkpoint_score"] is not None:
        try:
            return float(row["checkpoint_score"])
        except (TypeError, ValueError):
            return None
    return None


def _open_trade_signals(ticker: str) -> list[str]:
    """Latest signals_fired list for an open trade in trade_outcomes."""
    try:
        from kairos_ml_thesis import _get_connection as _ml_conn
    except Exception:
        return []
    try:
        conn = _ml_conn()
        row = conn.execute(
            """SELECT signals_fired
                 FROM trade_outcomes
                WHERE outcome_label IS NULL
                  AND UPPER(ticker) = ?
                ORDER BY id DESC
                LIMIT 1""",
            (ticker.upper(),),
        ).fetchone()
        conn.close()
    except Exception:
        return []
    raw = row["signals_fired"] if row else None
    if not raw:
        return []
    try:
        return [str(s).upper() for s in (json.loads(raw)
                                         if isinstance(raw, str) else raw)]
    except (json.JSONDecodeError, TypeError):
        return []


# ─────────────────────────────────────────────────────────────────────
# Available capital
# ─────────────────────────────────────────────────────────────────────

def _sum_active_reservations() -> float:
    """USD already committed to other un-released reservations."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(reserved_usd), 0) AS total "
            "FROM ipo_reservations WHERE status = ?",
            (STATUS_RESERVED,),
        ).fetchone()
        return float(row["total"] or 0.0)
    finally:
        conn.close()


def get_available_capital(ib=None) -> dict:
    """Return the capital view that reserve_capital() bases sizing on.

    Shape:
        {
          "cash":              7421.55,
          "nlv":              78230.10,
          "reserved":          5000.00,    # sum of active reservations
          "free_cash":         2421.55,    # cash - reserved
          "liberatable":       14820.00,   # sum of stale-position MV
          "available_total":  17241.55,    # free_cash + liberatable
          "positions_total":  70808.55,
        }
    """
    out = {
        "cash":             None,
        "nlv":              None,
        "reserved":         _sum_active_reservations(),
        "free_cash":        None,
        "liberatable":      0.0,
        "available_total":  None,
        "positions_total":  None,
    }
    snap = fetch_portfolio_snapshot(ib)
    if snap is None:
        return out

    cash = float(snap.get("cash") or 0.0)
    nlv  = float(snap.get("nlv")  or 0.0)
    positions = snap.get("positions") or {}
    positions_total = sum(
        float(p.get("market_value") or 0.0) for p in positions.values()
    )

    free_cash = max(0.0, cash - out["reserved"])

    # Build liberation view (criteria-pass only) for the "could free"
    # number. ipo_score=None means we don't apply the replacement rule
    # yet — just identify stale positions.
    liberatable = 0.0
    for cand in _liberation_classify(positions, ipo_score=None):
        if cand["liberatable"]:
            liberatable += float(cand["market_value"] or 0.0)

    available_total = free_cash + liberatable
    out.update({
        "cash":             round(cash, 2),
        "nlv":              round(nlv,  2),
        "free_cash":        round(free_cash, 2),
        "liberatable":      round(liberatable, 2),
        "available_total":  round(available_total, 2),
        "positions_total":  round(positions_total, 2),
    })
    return out


# ─────────────────────────────────────────────────────────────────────
# Liberation Engine
# ─────────────────────────────────────────────────────────────────────

def _holding_pnl_pct(ticker: str, avg_cost: float,
                     current_price: Optional[float]) -> Optional[float]:
    if current_price is None or not avg_cost:
        return None
    return (current_price - avg_cost) / avg_cost * 100.0


def _holding_days(entry_date_iso: str) -> Optional[int]:
    if not entry_date_iso:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(entry_date_iso[:19], fmt)
            return max(0, (datetime.utcnow() - dt).days)
        except (ValueError, TypeError):
            continue
    return None


def _liberation_classify(
    positions: dict,
    *,
    ipo_score: Optional[float],
) -> list[dict]:
    """Score every open position against the 4 liberation criteria.

    Returns a list of dicts with:
        {ticker, market_value, qty, avg_cost, current_price, pnl_pct,
         days_held, conviction, signals, criteria_hit, liberatable,
         rationale}

    A position is `liberatable` iff:
      - it hits at least one of the four criteria, AND
      - (when ipo_score is given) its conviction is <= ipo_score - REPLACEMENT_DELTA
        (or its conviction is unknown).

    The four criteria from the architecture doc:
      C1. pnl <= -15% AND no recovery signals
      C2. days_held >= 5 AND last checkpoint declining AND no new signal
      C3. signal includes HOT-REVERSION AND pnl >= 0 (reversion realised)
      C4. low-conviction displacement (conviction <= ipo_score - delta)
    """
    out: list[dict] = []
    if not positions:
        return out

    # Pull current prices from the same IBKR snapshot if available; we
    # don't make extra calls per ticker — market_value / qty derives an
    # implicit price.
    from kairos_log_db import get_open_holdings

    for ticker, info in positions.items():
        qty       = float(info.get("qty") or 0.0)
        avg_cost  = float(info.get("avg_cost") or 0.0)
        mkt_val   = float(info.get("market_value") or 0.0)
        cur_price = (mkt_val / qty) if qty else None

        # Days held — use the oldest open lot in kairos.db.holdings
        days_held = None
        try:
            lots = get_open_holdings(ticker)
            if lots:
                days_held = max(
                    (l.get("holding_days") for l in lots if l.get("holding_days") is not None),
                    default=None,
                )
                if days_held is None:
                    # Fall back to entry_date parsing
                    days_held = _holding_days(lots[0].get("entry_date", ""))
        except Exception:
            pass

        conviction = get_active_position_conviction(ticker)
        checkpoint = _latest_checkpoint_score(ticker)
        signals    = _open_trade_signals(ticker)
        signals_up = {s.upper() for s in signals}
        pnl_pct    = _holding_pnl_pct(ticker, avg_cost, cur_price)

        hits: list[str] = []
        # C1 — Loss criterion
        if (pnl_pct is not None
                and pnl_pct <= LIBERATION_LOSS_THRESHOLD
                and not _has_recovery_signal(signals_up)):
            hits.append("C1:loss>=15%")

        # C2 — Stale thesis criterion
        if (days_held is not None
                and days_held >= LIBERATION_MIN_HOLD_DAYS
                and checkpoint is not None
                and checkpoint < LIBERATION_STALE_CHECKPOINT_SCORE
                and not _has_recovery_signal(signals_up)):
            hits.append("C2:stale_declining")

        # C3 — HOT-REVERSION exhausted
        if "HOT-REVERSION" in signals_up and (pnl_pct or 0.0) >= LIBERATION_HOT_REVERSION_TARGET_PCT:
            hits.append("C3:reversion_done")

        # C4 — Low conviction displacement
        if (ipo_score is not None
                and conviction is not None
                and conviction <= (ipo_score - REPLACEMENT_DELTA)):
            hits.append("C4:conviction_gap")

        # Replacement rule (gate). When ipo_score is given, a position
        # with conviction NOT comfortably below the IPO is NEVER
        # liberatable, even if it hits other criteria. (Unknown
        # conviction passes through.)
        replacement_ok = (
            ipo_score is None
            or conviction is None
            or conviction <= (ipo_score - REPLACEMENT_DELTA)
        )

        liberatable = bool(hits) and replacement_ok

        out.append({
            "ticker":         ticker,
            "qty":            qty,
            "avg_cost":       round(avg_cost, 2),
            "current_price":  round(cur_price, 2) if cur_price else None,
            "market_value":   round(mkt_val, 2),
            "pnl_pct":        round(pnl_pct, 2) if pnl_pct is not None else None,
            "days_held":      days_held,
            "conviction":     conviction,
            "checkpoint":     checkpoint,
            "signals":        sorted(signals_up),
            "criteria_hit":   hits,
            "liberatable":    liberatable,
            "rationale":      ", ".join(hits) if hits else "no criterion met",
        })

    return out


def _has_recovery_signal(signals_up: set[str]) -> bool:
    """Any currently-firing HOT-* signal counts as 'active recovery'."""
    return any(s.startswith("HOT-") for s in signals_up)


def liberation_candidates(
    *,
    ipo_conviction_score: Optional[float] = None,
    ib=None,
) -> list[dict]:
    """Public preview of liberation candidates, sorted by liberation rank.

    Rank = lowest position-conviction first (per architecture step 4),
    then largest market_value (frees more cash per sell). Non-liberatable
    rows are dropped.
    """
    snap = fetch_portfolio_snapshot(ib)
    if snap is None:
        return []
    rows = _liberation_classify(snap.get("positions") or {},
                                ipo_score=ipo_conviction_score)
    cands = [r for r in rows if r["liberatable"]]
    cands.sort(
        key=lambda r: (
            r["conviction"] if r["conviction"] is not None else 999,
            -float(r["market_value"] or 0.0),
        )
    )
    return cands


def liquidate_for_funding(
    *,
    target_dollar: float,
    ipo_conviction_score: float,
    reservation_id: Optional[int] = None,
    ib=None,
    dry_run: bool = True,
) -> dict:
    """Sell minimum positions to free `target_dollar` against the IPO score.

    Returns:
        {
          "ok": True|False,
          "needed_usd": target_dollar,
          "freed_usd":  float,
          "dry_run":    bool,
          "actions":    [ {ticker, qty, est_freed_usd, ...sell result} ],
          "shortfall":  None | float,   # set if we ran out of candidates
        }

    If dry_run=True (default), no orders are placed — actions describe
    the plan only. If False, kairos_execute.execute_order issues SELL
    orders one at a time.
    """
    result = {
        "ok":         False,
        "needed_usd": round(float(target_dollar), 2),
        "freed_usd":  0.0,
        "dry_run":    bool(dry_run),
        "actions":    [],
        "shortfall":  None,
    }
    if target_dollar is None or target_dollar <= 0:
        result["ok"] = True
        return result

    own_conn = False
    if ib is None:
        ib, own_conn = _connect_ib()
        if ib is None and not dry_run:
            result["shortfall"] = float(target_dollar)
            return result

    try:
        snap = fetch_portfolio_snapshot(ib)
        if snap is None:
            result["shortfall"] = float(target_dollar)
            return result

        cands = _liberation_classify(snap.get("positions") or {},
                                     ipo_score=ipo_conviction_score)
        cands = [c for c in cands if c["liberatable"]]
        cands.sort(
            key=lambda r: (
                r["conviction"] if r["conviction"] is not None else 999,
                -float(r["market_value"] or 0.0),
            )
        )

        freed = 0.0
        for cand in cands:
            if freed >= target_dollar:
                break
            mkt_val = float(cand["market_value"] or 0.0)
            if mkt_val <= 0:
                continue
            qty = int(cand["qty"] or 0)
            if qty <= 0:
                continue

            action = {
                "ticker":        cand["ticker"],
                "qty":           qty,
                "est_freed_usd": round(mkt_val, 2),
                "rationale":     cand["rationale"],
                "conviction":    cand["conviction"],
                "pnl_pct":       cand["pnl_pct"],
                "days_held":     cand["days_held"],
            }

            if dry_run:
                action["sell_status"] = "DRY_RUN"
            else:
                try:
                    from kairos_execute import execute_order
                    exec_res = execute_order(ib, cand["ticker"], "SELL", qty)
                    action["sell_status"]    = exec_res.get("status")
                    action["sell_fill_price"] = exec_res.get("fill_price")
                except Exception as exc:
                    action["sell_status"] = "ERROR"
                    action["sell_error"]  = str(exc)

            _record_liberation(
                reservation_id=reservation_id,
                cand=cand,
                freed_usd=mkt_val,
                dry_run=dry_run,
                sell_status=action.get("sell_status"),
                sell_fill_price=action.get("sell_fill_price"),
            )

            freed += mkt_val
            result["actions"].append(action)
            _slack_liberation(
                ticker=cand["ticker"],
                freed=mkt_val,
                ipo_score=ipo_conviction_score,
                rationale=cand["rationale"],
                dry_run=dry_run,
            )

        result["freed_usd"] = round(freed, 2)
        if freed < target_dollar:
            result["shortfall"] = round(target_dollar - freed, 2)
        result["ok"] = freed >= target_dollar
        return result
    finally:
        if own_conn:
            _disconnect_ib(ib)


def _record_liberation(
    *,
    reservation_id: Optional[int],
    cand: dict,
    freed_usd: float,
    dry_run: bool,
    sell_status: Optional[str],
    sell_fill_price: Optional[float],
) -> None:
    conn = _get_connection()
    try:
        conn.execute(
            """INSERT INTO ipo_liberations
               (reservation_id, liberated_ticker, rationale,
                position_conviction, position_pnl_pct, days_held,
                freed_usd, dry_run, sell_status, sell_fill_price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                reservation_id,
                cand["ticker"],
                cand["rationale"],
                cand["conviction"],
                cand["pnl_pct"],
                cand["days_held"],
                round(float(freed_usd), 2),
                1 if dry_run else 0,
                sell_status,
                sell_fill_price,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# Reservation lifecycle
# ─────────────────────────────────────────────────────────────────────

def _alloc_pct_for_score(score: float) -> Optional[float]:
    """Round to nearest int and look up the % allocation."""
    try:
        rounded = int(round(float(score)))
    except (TypeError, ValueError):
        return None
    return CONVICTION_TO_ALLOC_PCT.get(rounded)


def reserve_capital(
    *,
    ticker: str,
    conviction_score: float,
    expected_pricing_date: Optional[str] = None,
    company_name: Optional[str] = None,
    ib=None,
    dry_run: bool = True,
) -> dict:
    """Reserve capital for an IPO. Defaults to dry_run.

    Returns:
        {
          "ok": bool,
          "reservation_id": int | None,
          "ticker": "...",
          "score": float,
          "alloc_pct": float,
          "target_usd": float,
          "cash": float,
          "available_total": float,
          "reserved_usd": float,
          "liberation": {...} | None,   # only when liberation triggered
          "notes": str,
        }
    """
    init_schema()
    t = (ticker or "").strip().upper()
    score = float(conviction_score or 0.0)

    result: dict = {
        "ok": False,
        "reservation_id": None,
        "ticker": t,
        "score": score,
        "alloc_pct": None,
        "target_usd": None,
        "cash": None,
        "available_total": None,
        "reserved_usd": 0.0,
        "liberation": None,
        "notes": "",
    }

    if not t:
        result["notes"] = "empty ticker"
        return result
    if score < MIN_RESERVATION_SCORE:
        result["notes"] = (f"score {score} below MIN_RESERVATION_SCORE "
                           f"{MIN_RESERVATION_SCORE}")
        return result

    alloc_pct = _alloc_pct_for_score(score)
    if alloc_pct is None:
        result["notes"] = "no allocation table entry for that rounded score"
        return result
    result["alloc_pct"] = alloc_pct

    # Reuse the broker snapshot once for both views.
    own_conn = False
    if ib is None:
        ib, own_conn = _connect_ib()
        if ib is None:
            result["notes"] = "IBKR unreachable; cannot size reservation"
            return result

    try:
        snap = fetch_portfolio_snapshot(ib)
        if snap is None:
            result["notes"] = "fetch_portfolio_snapshot returned None"
            return result

        cash = float(snap.get("cash") or 0.0)
        nlv  = float(snap.get("nlv")  or 0.0)
        already_reserved = _sum_active_reservations()
        free_cash = max(0.0, cash - already_reserved)

        cands = _liberation_classify(snap.get("positions") or {},
                                     ipo_score=score)
        liberatable = sum(
            float(c["market_value"] or 0.0) for c in cands if c["liberatable"]
        )
        available_total = free_cash + liberatable
        target_usd = round(alloc_pct * available_total, 2)

        result.update({
            "cash":            round(cash, 2),
            "available_total": round(available_total, 2),
            "target_usd":      target_usd,
        })

        if target_usd <= 0:
            result["notes"] = "no available capital"
            return result

        # Liberation only if cash short.
        liberation = None
        if target_usd > free_cash:
            gap = target_usd - free_cash
            liberation = liquidate_for_funding(
                target_dollar=gap,
                ipo_conviction_score=score,
                ib=ib,
                dry_run=dry_run,
                reservation_id=None,   # reservation_id stamped later
            )
            result["liberation"] = liberation
            if not liberation.get("ok"):
                result["notes"] = (
                    f"liberation shortfall: needed ${gap:.2f}, "
                    f"freed ${liberation.get('freed_usd', 0):.2f}"
                )
                return result

        if dry_run:
            result["notes"] = "dry_run — no reservation row written"
            result["reserved_usd"] = target_usd
            result["ok"] = True
            return result

        # Persist the reservation.
        conn = _get_connection()
        try:
            cur = conn.execute(
                """INSERT INTO ipo_reservations
                   (ticker, company_name, expected_pricing_date,
                    conviction_score, reserved_pct, reserved_usd,
                    cash_at_reservation, nlv_at_reservation,
                    available_at_reservation, status, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    t, company_name, expected_pricing_date,
                    score, alloc_pct, target_usd,
                    cash, nlv, available_total,
                    STATUS_RESERVED,
                    f"alloc_pct={alloc_pct:.2f}",
                ),
            )
            reservation_id = int(cur.lastrowid)

            # Backfill liberation rows with the new reservation_id.
            if liberation:
                conn.execute(
                    "UPDATE ipo_liberations SET reservation_id = ? "
                    "WHERE reservation_id IS NULL "
                    "  AND liberated_ticker IN (%s)" % ",".join(
                        ["?"] * len(liberation.get("actions") or [])
                    ),
                    [reservation_id]
                    + [a["ticker"] for a in (liberation.get("actions") or [])],
                ) if liberation.get("actions") else None
            conn.commit()
        finally:
            conn.close()

        result["reservation_id"] = reservation_id
        result["reserved_usd"]   = target_usd
        result["ok"] = True

        _slack_reservation(
            ticker=t,
            score=score,
            reserved_usd=target_usd,
            pricing_date=expected_pricing_date,
            liberation=liberation,
        )
        return result
    finally:
        if own_conn:
            _disconnect_ib(ib)


def release_reservation(
    reservation_id_or_ticker,
    reason: str,
) -> dict:
    """Mark a reservation released. Accepts a numeric id or a ticker.

    If a ticker is given and multiple reservations are open, the most
    recent one is released.
    """
    init_schema()
    conn = _get_connection()
    try:
        row = None
        try:
            rid = int(reservation_id_or_ticker)
            row = conn.execute(
                "SELECT * FROM ipo_reservations WHERE id = ?", (rid,)
            ).fetchone()
        except (TypeError, ValueError):
            t = str(reservation_id_or_ticker or "").strip().upper()
            row = conn.execute(
                "SELECT * FROM ipo_reservations "
                "WHERE UPPER(ticker) = ? AND status = ? "
                "ORDER BY id DESC LIMIT 1",
                (t, STATUS_RESERVED),
            ).fetchone()

        if row is None:
            return {"ok": False, "error": "reservation not found"}
        if row["status"] != STATUS_RESERVED:
            return {"ok": False, "error": f"already {row['status']}"}

        conn.execute(
            "UPDATE ipo_reservations SET status = ?, released_at = ?, "
            "release_reason = ? WHERE id = ?",
            (
                STATUS_RELEASED,
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                reason,
                row["id"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    _slack_release(ticker=row["ticker"], reserved_usd=row["reserved_usd"],
                   reason=reason)
    return {"ok": True, "reservation_id": row["id"], "ticker": row["ticker"]}


def convert_reservation_to_position(
    *,
    ticker: str,
    shares: int,
    avg_price: float,
) -> dict:
    """Mark the most recent reserved row for `ticker` as converted."""
    init_schema()
    t = (ticker or "").strip().upper()
    if not t:
        return {"ok": False, "error": "empty ticker"}

    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM ipo_reservations "
            "WHERE UPPER(ticker) = ? AND status = ? "
            "ORDER BY id DESC LIMIT 1",
            (t, STATUS_RESERVED),
        ).fetchone()
        if row is None:
            return {"ok": False, "error": "no active reservation for ticker"}

        conn.execute(
            "UPDATE ipo_reservations SET status = ?, converted_at = ?, "
            "converted_shares = ?, converted_avg_price = ? "
            "WHERE id = ?",
            (
                STATUS_CONVERTED,
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                int(shares),
                round(float(avg_price), 4),
                row["id"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    _slack_convert(ticker=t, shares=shares, avg_price=avg_price,
                   reserved_usd=row["reserved_usd"])
    return {
        "ok": True,
        "reservation_id": row["id"],
        "ticker": t,
        "shares": int(shares),
        "avg_price": round(float(avg_price), 4),
    }


def check_release_conditions(
    *,
    reservation_id: int,
    current_conviction_score: Optional[float] = None,
    ipo_status: Optional[str] = None,
) -> Optional[str]:
    """Decide whether a reservation should auto-release. Returns the
    release reason (str) or None if it should stay reserved.

    Conditions per the architecture doc:
      - IPO postponed / withdrawn → release.
      - Score drops below MIN_RESERVATION_SCORE → release.
    Conversion to a position is NOT handled here — that's
    convert_reservation_to_position().
    """
    if ipo_status and ipo_status.lower() in {"postponed", "withdrawn", "pulled"}:
        return f"ipo_status={ipo_status}"
    if (current_conviction_score is not None
            and float(current_conviction_score) < MIN_RESERVATION_SCORE):
        return (f"score decayed to {current_conviction_score:.1f} "
                f"(<{MIN_RESERVATION_SCORE})")
    return None


# ─────────────────────────────────────────────────────────────────────
# Queries
# ─────────────────────────────────────────────────────────────────────

def get_active_reservations() -> list[dict]:
    init_schema()
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM ipo_reservations WHERE status = ? "
            "ORDER BY reserved_at DESC",
            (STATUS_RESERVED,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_reservation(reservation_id: int) -> Optional[dict]:
    init_schema()
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM ipo_reservations WHERE id = ?", (reservation_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# Slack alerts (best-effort; never raise)
# ─────────────────────────────────────────────────────────────────────

def _slack(msg: str) -> None:
    try:
        from kairos_alerts import alert_pipeline_event
        alert_pipeline_event(msg, channel=SLACK_CHANNEL)
    except Exception as exc:
        print(f"  WARNING: Slack post failed: {exc}")


def _slack_reservation(*, ticker, score, reserved_usd, pricing_date,
                       liberation) -> None:
    bits = [
        f":bookmark: *IPO RESERVATION:* `${ticker}` "
        f"— ${reserved_usd:,.0f} reserved (score {score:.1f})"
    ]
    if pricing_date:
        bits.append(f"  Expected pricing: {pricing_date}")
    if liberation and liberation.get("actions"):
        freed = liberation.get("freed_usd") or 0
        n = len(liberation["actions"])
        dry = " (dry-run)" if liberation.get("dry_run") else ""
        bits.append(f"  Liberation: ${freed:,.0f} freed from {n} positions{dry}")
    _slack("\n".join(bits))


def _slack_liberation(*, ticker, freed, ipo_score, rationale, dry_run) -> None:
    dry = " (dry-run)" if dry_run else ""
    _slack(
        f":scissors: *CAPITAL LIBERATED:* `${ticker}` "
        f"— ${freed:,.0f} freed{dry}\n"
        f"  Reason: {rationale}\n"
        f"  Funding IPO conviction {ipo_score:.1f}"
    )


def _slack_release(*, ticker, reserved_usd, reason) -> None:
    _slack(
        f":unlock: *RESERVATION RELEASED:* `${ticker}` "
        f"— ${reserved_usd:,.0f} freed\n"
        f"  Reason: {reason}"
    )


def _slack_convert(*, ticker, shares, avg_price, reserved_usd) -> None:
    _slack(
        f":white_check_mark: *RESERVATION CONVERTED:* `${ticker}` "
        f"— {shares:,} sh @ ${avg_price:.2f}\n"
        f"  Reserved was ${reserved_usd:,.0f}"
    )


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _print_json(label: str, payload) -> None:
    print(f"\n── {label} ──")
    if payload in (None, [], {}):
        print("  (no data)")
        return
    print(json.dumps(payload, indent=2, default=str))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Kairos IPO capital reservation + liberation engine",
    )
    parser.add_argument("--init-db", action="store_true",
                        help="Create / verify ipo_reservations + ipo_liberations")
    parser.add_argument("--available", action="store_true",
                        help="Show available-capital view")
    parser.add_argument("--list", action="store_true",
                        help="List active reservations")
    parser.add_argument("--reserve", nargs=2, metavar=("TICKER", "SCORE"),
                        help="Reserve capital for TICKER at conviction SCORE")
    parser.add_argument("--pricing-date", metavar="YYYY-MM-DD",
                        help="Expected pricing date for --reserve")
    parser.add_argument("--company", metavar="NAME",
                        help="Company name for --reserve")
    parser.add_argument("--release", nargs=2,
                        metavar=("ID_OR_TICKER", "REASON"),
                        help="Release a reservation")
    parser.add_argument("--convert", nargs=3,
                        metavar=("TICKER", "SHARES", "PRICE"),
                        help="Mark a reservation as converted to a position")
    parser.add_argument("--candidates", metavar="IPO_SCORE",
                        help="Preview liberation candidates against an IPO score")
    parser.add_argument("--liberate", nargs=2,
                        metavar=("TARGET_USD", "IPO_SCORE"),
                        help="Run liberation for TARGET_USD against IPO_SCORE")
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Execute orders for real (default is dry-run)")
    args = parser.parse_args()

    did_any = False

    if args.init_db:
        init_schema()
        print("ipo_reservations + ipo_liberations: ready")
        did_any = True

    if args.available:
        _print_json("AvailableCapital", get_available_capital())
        did_any = True

    if args.list:
        _print_json("ActiveReservations", get_active_reservations())
        did_any = True

    if args.candidates:
        try:
            score = float(args.candidates)
        except ValueError:
            print(f"--candidates expects a numeric IPO score, got {args.candidates!r}")
            return
        _print_json(
            f"LiberationCandidates (vs ipo_score={score})",
            liberation_candidates(ipo_conviction_score=score),
        )
        did_any = True

    if args.liberate:
        target_s, score_s = args.liberate
        result = liquidate_for_funding(
            target_dollar=float(target_s),
            ipo_conviction_score=float(score_s),
            dry_run=not args.no_dry_run,
        )
        _print_json(
            f"LiquidateForFunding ${target_s} vs ipo_score={score_s}",
            result,
        )
        did_any = True

    if args.reserve:
        ticker, score_s = args.reserve
        result = reserve_capital(
            ticker=ticker,
            conviction_score=float(score_s),
            expected_pricing_date=args.pricing_date,
            company_name=args.company,
            dry_run=not args.no_dry_run,
        )
        _print_json(f"ReserveCapital {ticker} @ {score_s}", result)
        did_any = True

    if args.release:
        id_or_ticker, reason = args.release
        result = release_reservation(id_or_ticker, reason)
        _print_json(f"ReleaseReservation {id_or_ticker}", result)
        did_any = True

    if args.convert:
        ticker, shares_s, price_s = args.convert
        result = convert_reservation_to_position(
            ticker=ticker,
            shares=int(shares_s),
            avg_price=float(price_s),
        )
        _print_json(f"ConvertReservation {ticker}", result)
        did_any = True

    if not did_any:
        parser.print_help()


if __name__ == "__main__":
    main()
