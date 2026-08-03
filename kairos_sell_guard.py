"""Never-oversell guard — clamp every SELL to the actual held quantity.

Kairos is LONG-ONLY. A short position is never an acceptable outcome of any
code path. On 2026-07-29 two Council "THESIS-INVALID" exits oversold because
the executor sized SELLs with `compute_position_size()` (an ENTRY sizer, a
function of confluence tier x NLV x price) instead of the held quantity:

    ETN: sold 44 vs 28 held -> -16 short at the broker
    EME: sold 23 vs 15 held ->  -8 short at the broker

The daily reconciler did not catch it because DB and broker agreed — it
validates *sync*, not *sanity*.

This module is the safety net at the order-submission boundary. The real
sizing fix lives in the callers (see kairos_execute: a SELL is sized from the
held position). Both layers are intentional: the sizing fix is the cure, this
is the seatbelt.

Contract
--------
`clamp_sell_quantity` returns (allowed_qty, note):

  * held is KNOWN and requested <= held  -> (requested, None)      pass through
  * held is KNOWN and requested >  held  -> (held, "clamped ...")  clamp + alert
  * held is KNOWN and held <= 0          -> (0, "aborted ...")     ABORT + alert
  * held is UNKNOWN (both sources fail)  -> (requested, "...")     pass + alert

The last case fails OPEN by design. A held quantity is unknown only if BOTH
the broker and the local DB are unreadable; in that state, silently blocking
every exit (including stop-losses) is a larger financial risk than the
oversell it would prevent. It is alerted loudly so it can never pass quietly.
"""

from __future__ import annotations

import math

# Callers pass integer share counts; tolerate float dust from IBKR/SQLite.
_QTY_EPSILON = 1e-6


def _held_from_broker(ticker: str, ib) -> float | None:
    """Summed STK position for `ticker` at IBKR, or None if unavailable."""
    if ib is None:
        return None
    try:
        if not (hasattr(ib, "isConnected") and ib.isConnected()):
            return None
        total = 0.0
        found = False
        for p in ib.positions():
            try:
                if p.contract.secType != "STK":
                    continue
                if p.contract.symbol != ticker:
                    continue
                total += float(p.position)
                found = True
            except Exception:
                continue
        # No row for the ticker means IBKR holds none of it — that is a real
        # answer (0.0), not a missing one.
        return total if found else 0.0
    except Exception:
        return None


def _held_from_db(ticker: str) -> float | None:
    """Summed open-lot quantity for `ticker` in kairos.db, or None on error."""
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(quantity), 0) AS q FROM holdings "
                "WHERE ticker = ? AND sold_date IS NULL",
                (ticker,),
            ).fetchone()
        finally:
            conn.close()
        return float(row["q"]) if row is not None else 0.0
    except Exception:
        return None


def get_held_quantity(ticker: str, ib=None) -> tuple[float | None, str]:
    """Current held share count for `ticker`.

    Broker position when an IBKR connection is live (source of truth), else
    the sum of open lots in the DB. Returns (quantity, source); quantity is
    None only when neither source can be read.
    """
    held = _held_from_broker(ticker, ib)
    if held is not None:
        return held, "broker"

    held = _held_from_db(ticker)
    if held is not None:
        return held, "db"

    return None, "unavailable"


def clamp_sell_quantity(
    ticker: str,
    requested_qty,
    ib=None,
    context: str = "",
) -> tuple[int, str | None]:
    """Clamp a SELL to the held quantity. Returns (allowed_qty, note).

    An allowed_qty of 0 means the caller MUST NOT submit the order.
    `note` is None when the request passed through untouched.
    """
    try:
        requested = int(requested_qty)
    except (TypeError, ValueError):
        requested = 0

    if requested <= 0:
        return max(requested, 0), None

    where = f" [{context}]" if context else ""
    held, source = get_held_quantity(ticker, ib)

    if held is None:
        note = (f"held quantity unavailable for {ticker} (broker and DB both "
                f"unreadable) — SELL {requested} allowed unclamped{where}")
        _warn(f"oversell guard degraded: {note}", ticker)
        return requested, note

    # Floor: never sell a fraction of a share we do not hold.
    held_int = int(math.floor(held + _QTY_EPSILON))

    if held_int <= 0:
        note = (f"oversell prevented: {ticker} requested {requested}, "
                f"held {held:g} (source: {source}){where}")
        _warn(note, ticker)
        return 0, note

    if requested > held_int:
        note = (f"oversell clamped: {ticker} requested {requested}, "
                f"held {held_int} (source: {source}) — selling {held_int}{where}")
        _warn(note, ticker)
        return held_int, note

    return requested, None


def _warn(message: str, ticker: str = "") -> None:
    """Print and post a #kairos-alerts warning. Never raises."""
    print(f"    ⚠ SELL GUARD: {message}")
    try:
        from kairos_alerts import post_message
        post_message("alerts", f":shield: *Sell Guard — {ticker or 'SELL'}*\n{message}")
    except Exception as exc:  # noqa: BLE001 — alerting must never break a trade
        print(f"    sell-guard Slack post failed: {exc}")
