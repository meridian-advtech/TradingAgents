# Kairos — B2b done + live-path ops fixes prepared (2026-06-19, ~11:35 ET)

## DONE this session (behavior-neutral, applied)
kairos_axis_weights.py (backup: kairos_axis_weights.py.bak_20260619_113253)
- exit_timing rewired to bidirectional signed score off persisted B2a features:
  net = mean(give_back_pct | mfe>0)  −  mean(max(post_exit_peak_pct,0)).
  Scale recalibrated 5.0 → 10.0 (EXIT_TIMING_SCALE). Live numbers:
  too-late 11.15pp − too-early 5.95pp = net +5.20pp → score +0.5195 (was saturated 1.0).
- NEW compute_reallocation_aggressiveness: too-eager (intact thesis @ exit, post-exit
  run-up, +) vs too-sticky (decayed thesis @ exit, give-back, −).
  Live: eager 5.97pp (n16) − sticky 11.93pp (n9) = net −5.96pp → score −0.596, sample 25.
  Caveat: sticky pole thin; reallocation exits under-logged until #3 lands.
- Proposed exit_timing id 3 (score 0.5195, Δ+0.15) — PENDING, superseded old id 2.
  Nothing applied to axis_weights (all still 0.0/active). Human gate intact.

## DECISIONS (Jason, 11:40 ET): apply ALL of #2, #3, #4, #6 after 16:00 close.
##   #4 retrain APPROVED. #6 guard = ALERT + SKIP-CYCLE on stale marks.
##   Mechanism: scheduled task re-invokes Claude at 16:05 ET today.

## PREPARED — apply AFTER 16:00 close (NOT yet applied; live trading path)

### #2 screener crash guard — kairos_screener.py ~line 288  [SAFE, minimal]
Current: `s = item.get("score", "COLD").upper()` — AttributeError if score is a list;
the surrounding `except (JSONDecodeError, KeyError, TypeError)` does NOT catch it → cycle aborts.
Fix: coerce safely and add AttributeError to the except.
    raw_s = item.get("score", "COLD")
    if isinstance(raw_s, list):
        raw_s = raw_s[0] if raw_s else "COLD"
    s = str(raw_s).strip().upper()
  and: except (json.JSONDecodeError, KeyError, TypeError, AttributeError):

### #3 reallocation write-back — kairos_reallocation.py ~lines 748-786  [SAFE, consistency fix]
Root cause: reallocation calls write_trade_close(decision_id=None, fill_price=, ticker=,
action=, quantity=, price_entry=, price_exit=, pnl_*=, signals_fired=, confluence_score=,
outcome_label=) — NONE of those kwargs exist on the signature
write_trade_close(trade_id, price_exit, timestamp_exit=None). Every call raises TypeError,
swallowed by `except` → reallocation exits never recorded (the silent drift).
Fix: use the same pattern every other exit path uses (execute/exits/stoploss/thesis_review):
    from kairos_ml_outcomes import init_db, write_trade_close, find_open_trade
    init_db()
    open_tid = find_open_trade(exit_ticker, "BUY")
    if open_tid:
        result = write_trade_close(open_tid, sell_price, timestamp_exit=sell_date)
        print(f"    ML Outcomes: recorded {exit_ticker} reallocation exit "
              f"→ {result['outcome_label']} ({result['pnl_pct']:+.2f}%)")
    else:
        print(f"    ML Outcomes: no open {exit_ticker} BUY trade to close")
(removes the bogus kwargs + the manual decisions-table timestamp lookup; write_trade_close
derives entry/duration from the stored open row.)
Follow-up (optional): also write position_exits exit_reason="REALLOCATION" so the
reallocation axis gets a real exit_reason signal later.

### #4 ML feature mismatch — retrain kairos_ml_model.pkl  [MECHANICAL, ML is advisory/non-fatal]
Error: "X has 6 features, but OneHotEncoder is expecting 7." CATEGORICAL_FEATURES is now 6
(news, macro, legis_sentiment, sector, day_of_week, hour_of_day); the saved pickle's encoder
was fit on 7. Fix: retrain on current closed-trade data so the encoder matches (regenerates
the pickle). Low risk — ML output is advisory and currently skipped on every error.
NEEDS: your OK to retrain (it changes the model once ML scoring comes back online).

### #6 frozen-mark detection guard — kairos_reason.py ~lines 95-114  [BEHAVIORAL — design sign-off needed]
Today: reqMarketDataType(4) then per ticker takes first non-NaN of (last, close, bid, ask).
When the feed freezes, `last` is NaN and it silently falls to `close` = prior close, and
trades/reports on it as if live. Proposed guard: track which attr each price came from; if a
high fraction resolve via `close` (frozen fallback) / no live tick, treat marks as STALE →
post a Slack alert AND signal the cycle to SKIP trading rather than act on stale data.
NEEDS: your sign-off on behavior (alert-only vs alert+skip-cycle) before I write it.

## NOT a code fix (operational) — #1 frozen IBKR feed
Check Gateway UI / paper market-data subscription after the ~04:47 auto-restart. Outside Kairos logs.
