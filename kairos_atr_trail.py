"""
ATR-scaled armed trail — arithmetic, bind-state attribution, and arm context.

Single source of truth for the ATR trail so that three consumers cannot drift:
kairos_exits.py (which acts on it), kairos_ml_outcomes.py (which stamps it on
every close), and kairos_axis_weights.py (which routes learning evidence by it).

    raw   = atr_mult * ATR14/price * 100
    trail = clamp(raw, trail_lo_pct, trail_hi_pct)

BIND STATE — why this module exists at all
------------------------------------------
The trail actually applied to a position is determined by exactly ONE of the
three parameters:

    free     lo <= raw <= hi   -> atr_mult governed this trade
    floor    raw <  lo         -> trail_lo_pct governed it; atr_mult was IRRELEVANT
    ceiling  raw >  hi         -> trail_hi_pct governed it; atr_mult was IRRELEVANT
    no_atr   ATR unavailable   -> trail_pct (the deprecated fallback) governed it

Attributing a clamp-bound close to atr_mult is not a rounding error, it is a
false statement: the multiplier had no influence on that outcome. On the live
book of 2026-09-09, 30 of 49 held positions are clamp-bound, so misattribution
would be the NORM rather than an edge case — the multiplier would be "learned"
mostly from trades it did not govern.

ARM-TIME CAPTURE
----------------
ATR14 is read ONCE, when the trail arms, and persisted — never recomputed per
cycle. Two reasons, and the second is the one that bites:
  * it matches the backtest, which set the trail at the arming bar;
  * a per-cycle ATR makes the trail itself follow volatility DOWN as a position
    calms after its spike, silently tightening the stop underneath a live
    position. The trail would move without any config change and without any
    proposal, which is unauditable.

The context is computed and recorded even while atr_enabled is FALSE. That is
the whole point of the shadow period: closes accrue a bind state from day one,
so each parameter's evidence pool is populated before the behaviour flips.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

ATR_WINDOW = 14

BIND_FREE = "free"
BIND_FLOOR = "floor"
BIND_CEILING = "ceiling"
BIND_NO_ATR = "no_atr"

# Bind states that represent a real ATR measurement. no_atr is NOT one of them:
# the position ran on the fallback trail, so it is evidence about trail_pct, not
# about any of the three ATR parameters.
BIND_STATES_MEASURED = (BIND_FREE, BIND_FLOOR, BIND_CEILING)
BIND_STATES = BIND_STATES_MEASURED + (BIND_NO_ATR,)

_ATR_CACHE: dict = {}


def clear_atr_cache() -> None:
    """Reset the per-run ATR cache (one network fetch per ticker per run)."""
    _ATR_CACHE.clear()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── ATR measurement ──────────────────────────────────────────────────

def atr14_pct(ticker: str, window: int = ATR_WINDOW):
    """ATR14 as a % of price, or None if it cannot be measured.

    True Range = max(H-L, |H-prev_close|, |L-prev_close|), averaged over the
    last `window` CONFIRMED daily bars and divided by the last confirmed close.

    An in-progress session is excluded, matching kairos_exits._recent_daily_
    closes: during market hours yfinance returns a partial bar whose range is
    only the day so far, which would understate ATR and hand out a tighter
    trail purely as a function of what time of day the position armed.

    Returns None — never a default — when the data is not there. A defaulted
    ATR is indistinguishable from a measured one downstream, and would be
    attributed to atr_mult as if it were evidence.
    """
    key = (ticker, window)
    if key in _ATR_CACHE:
        return _ATR_CACHE[key]

    out = None
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d",
                                         auto_adjust=True)
        if hist is not None and not hist.empty and \
                all(c in hist for c in ("High", "Low", "Close")):
            rows = [(idx.date(), float(h), float(l), float(c))
                    for idx, h, l, c in zip(hist.index, hist["High"],
                                            hist["Low"], hist["Close"])
                    if h == h and l == l and c == c and c > 0]
            # Drop an in-progress session (exchange-local date >= today ET).
            try:
                from zoneinfo import ZoneInfo
                today_et = datetime.now(ZoneInfo("America/New_York")).date()
            except Exception:
                today_et = None
            if rows and today_et is not None and rows[-1][0] >= today_et:
                rows = rows[:-1]
            if len(rows) >= window + 1:
                trs = []
                for i in range(1, len(rows)):
                    _d, h, l, _c = rows[i]
                    pc = rows[i - 1][3]
                    trs.append(max(h - l, abs(h - pc), abs(l - pc)))
                last_close = rows[-1][3]
                if trs and last_close > 0:
                    atr = sum(trs[-window:]) / len(trs[-window:])
                    out = atr / last_close * 100.0
    except Exception:
        out = None

    _ATR_CACHE[key] = out
    return out


# ── Trail resolution + bind state (PURE) ─────────────────────────────

def atr_params(cfg: dict | None = None) -> dict:
    """The four ATR keys out of exits.trailing_stop.target_armed."""
    if cfg is None:
        from kairos_exits import _exits_config
        cfg = _exits_config()
    ta = (cfg.get("trailing_stop") or {}).get("target_armed") or {}
    return {
        "atr_enabled": bool(ta.get("atr_enabled", False)),
        "atr_mult": float(ta.get("atr_mult", 0.75)),
        "trail_lo_pct": float(ta.get("trail_lo_pct", 2.0)),
        "trail_hi_pct": float(ta.get("trail_hi_pct", 4.0)),
        "trail_pct": float(ta.get("trail_pct", 8.0)),
    }


def resolve_trail(atr_pct, cfg: dict | None = None, params: dict | None = None) -> dict:
    """The trail this position gets, and WHICH parameter decided it. Pure.

    Returns {atr_pct, raw_trail_pct, trail_pct, bind_state, governed_by,
             atr_mult, trail_lo_pct, trail_hi_pct, fallback_trail_pct}.

    On atr_pct=None the result is bind_state='no_atr' and trail_pct=the
    deprecated fallback. It deliberately does NOT clamp into the band: applying
    trail_lo_pct to a position whose volatility was never measured would record
    a floor-bound trade in trail_lo_pct's evidence pool on the strength of a
    missing data point.

    Inverted bounds (lo > hi) are reported rather than silently ordered — the
    proposal path refuses to cross them (see kairos_axis_weights), so an
    inverted pair on disk means someone hand-edited the config and the engine
    should say so rather than quietly picking one.
    """
    p = params or atr_params(cfg)
    lo, hi = p["trail_lo_pct"], p["trail_hi_pct"]
    out = {
        "atr_pct": None if atr_pct is None else round(float(atr_pct), 4),
        "atr_mult": p["atr_mult"],
        "trail_lo_pct": lo,
        "trail_hi_pct": hi,
        "fallback_trail_pct": p["trail_pct"],
        "bounds_inverted": lo > hi,
    }
    if atr_pct is None:
        out.update({"raw_trail_pct": None, "trail_pct": p["trail_pct"],
                    "bind_state": BIND_NO_ATR,
                    "governed_by": "trail_pct (fallback — ATR unavailable)"})
        return out

    raw = p["atr_mult"] * float(atr_pct)
    if raw < lo:
        trail, bind, gov = lo, BIND_FLOOR, "trail_lo_pct"
    elif raw > hi:
        trail, bind, gov = hi, BIND_CEILING, "trail_hi_pct"
    else:
        trail, bind, gov = raw, BIND_FREE, "atr_mult"
    out.update({"raw_trail_pct": round(raw, 4), "trail_pct": round(trail, 4),
                "bind_state": bind, "governed_by": gov})
    return out


# ── Arm context store (append-only) ──────────────────────────────────
# Its own table, not columns on `holdings`, for two reasons: the arming event is
# a point-in-time fact that must survive the lot being marked sold (the close
# path reads it AFTER the sell), and an append-only row per arming is an audit
# trail — you can see what the engine measured and when, which is the standard
# this project holds every behaviour-changing mechanism to.

def record_arm(ticker: str, ctx: dict, peak_gain_pct=None, target=None,
               atr_enabled=None) -> int | None:
    """Record an arming event. Returns the row id, or None if it could not write.

    Never raises: failing to record an arm context must not be able to break an
    exit evaluation. A missing row degrades to "recompute next cycle", which is
    a measurement wobble, not a trading failure.
    """
    try:
        from kairos_log_db import get_connection
        conn = get_connection()
        try:
            cur = conn.execute(
                "INSERT INTO armed_trail_context "
                "(ticker, armed_at, atr_pct, raw_trail_pct, trail_pct, "
                " bind_state, atr_mult, trail_lo_pct, trail_hi_pct, "
                " fallback_trail_pct, atr_enabled, peak_gain_pct, target_pct, "
                " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ticker, _now(), ctx.get("atr_pct"), ctx.get("raw_trail_pct"),
                 ctx.get("trail_pct"), ctx.get("bind_state"),
                 ctx.get("atr_mult"), ctx.get("trail_lo_pct"),
                 ctx.get("trail_hi_pct"), ctx.get("fallback_trail_pct"),
                 1 if atr_enabled else 0,
                 None if peak_gain_pct is None else float(peak_gain_pct),
                 None if target is None else float(target), _now()))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()
    except Exception:
        return None


def latest_arm(ticker: str, since: str | None = None,
               measured_only: bool = False) -> dict | None:
    """The most recent arming context for this ticker, or None.

    `since` bounds the lookup to arms at or after a timestamp (the position's
    entry date), so a re-entered ticker does not inherit the arm context of a
    previous, already-closed position.

    `measured_only` restricts to rows carrying a real ATR measurement. See
    arm_context for why a no_atr row must not be treated as a measurement.
    """
    try:
        from kairos_log_db import get_connection
        sql = "SELECT * FROM armed_trail_context WHERE ticker = ?"
        args: list = [ticker]
        if since:
            sql += " AND armed_at >= ?"
            args.append(since)
        if measured_only:
            sql += (" AND bind_state IN ("
                    + ",".join("?" * len(BIND_STATES_MEASURED)) + ")")
            args.extend(BIND_STATES_MEASURED)
        sql += " ORDER BY id DESC LIMIT 1"
        conn = get_connection()
        try:
            row = conn.execute(sql, tuple(args)).fetchone()
        finally:
            conn.close()
        return dict(row) if row is not None else None
    except Exception:
        return None


def arm_context(ticker: str, cfg: dict | None = None, since: str | None = None,
                peak_gain_pct=None, target=None, persist: bool = True) -> dict:
    """Arm context for a position: the stored one if it exists, else measure now.

    This is the ARM-TIME semantics in one place. Once a position has armed, the
    stored ATR is reused for the life of the position, so the trail cannot
    drift under it as volatility changes. Only the first arm measures.

    persist=False makes this a pure read/measure with no write — used by the
    dry-run and reporting paths so inspecting the book cannot create arm rows.

    A no_atr row is a FAILED measurement, not a measurement, and is therefore
    never reused: only a row carrying a real ATR is frozen. Found the hard way
    on 2026-09-09 — a burst of yfinance rate-limiting made 14 live positions
    arm as no_atr, and because the store was reused unconditionally, every one
    of them would have carried "ATR unavailable" for the rest of its life and
    fed trail_pct's evidence pool instead of the ATR family's. A transient data
    outage must not become a permanent attribution. The arm-time property is
    preserved as "the first SUCCESSFUL measurement", and the retry is bounded:
    a repeated failure does not insert another row.
    """
    stored = latest_arm(ticker, since=since, measured_only=True)
    if stored is not None:
        p = atr_params(cfg)
        return {
            "atr_pct": stored.get("atr_pct"),
            "raw_trail_pct": stored.get("raw_trail_pct"),
            "trail_pct": stored.get("trail_pct"),
            "bind_state": stored.get("bind_state"),
            "governed_by": {BIND_FREE: "atr_mult", BIND_FLOOR: "trail_lo_pct",
                            BIND_CEILING: "trail_hi_pct"}.get(
                                stored.get("bind_state"),
                                "trail_pct (fallback — ATR unavailable)"),
            "atr_mult": stored.get("atr_mult"),
            "trail_lo_pct": stored.get("trail_lo_pct"),
            "trail_hi_pct": stored.get("trail_hi_pct"),
            "fallback_trail_pct": stored.get("fallback_trail_pct",
                                             p["trail_pct"]),
            "bounds_inverted": (stored.get("trail_lo_pct") or 0)
                               > (stored.get("trail_hi_pct") or 0),
            "armed_at": stored.get("armed_at"),
            "from_store": True,
        }

    p = atr_params(cfg)
    ctx = resolve_trail(atr14_pct(ticker), params=p)
    ctx["armed_at"] = _now()
    ctx["from_store"] = False
    if persist:
        if ctx["bind_state"] != BIND_NO_ATR:
            # A real measurement: freeze it. This is the arm-time value.
            record_arm(ticker, ctx, peak_gain_pct=peak_gain_pct, target=target,
                       atr_enabled=p["atr_enabled"])
        elif latest_arm(ticker, since=since) is None:
            # Measurement failed and nothing is on record yet. Write ONE audit
            # row so the failure is visible, then stop — otherwise a sustained
            # outage would append a row per position per cycle. The next cycle
            # retries and supersedes it with a real measurement.
            record_arm(ticker, ctx, peak_gain_pct=peak_gain_pct, target=target,
                       atr_enabled=p["atr_enabled"])
    return ctx


def snapshot_block(ctx: dict | None, atr_enabled: bool) -> dict | None:
    """The `armed_trail` block stamped onto exit_params_snapshot, or None.

    Kept OUT of snapshot["params"], which is the numeric regime-key namespace
    _snapshot_value reads. bind_state is a string and would never be readable
    as a regime value; putting it there would imply it was one.
    """
    if not ctx:
        return None
    return {
        "bind_state": ctx.get("bind_state"),
        "governed_by": ctx.get("governed_by"),
        "atr_pct": ctx.get("atr_pct"),
        "raw_trail_pct": ctx.get("raw_trail_pct"),
        "trail_pct": ctx.get("trail_pct"),
        "atr_mult": ctx.get("atr_mult"),
        "trail_lo_pct": ctx.get("trail_lo_pct"),
        "trail_hi_pct": ctx.get("trail_hi_pct"),
        "fallback_trail_pct": ctx.get("fallback_trail_pct"),
        "armed_at": ctx.get("armed_at"),
        # False during the shadow period: the bind state was COMPUTED but the
        # trail was not APPLIED. Consumers that care whether a close actually
        # ran under the ATR trail read this, not atr_mult's presence.
        "atr_enabled": bool(atr_enabled),
        "shadow": not bool(atr_enabled),
    }
