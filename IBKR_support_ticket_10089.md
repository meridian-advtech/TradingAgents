# IBKR Support Ticket — API live market data returns error 10089 / 10168 (paper account)

**Category:** Market Data / API
**Account:** Paper trading account DUN786964 (live/parent username: jwelmore81)
**Connection:** IB Gateway 10.37 (IBC 3.19.0), API port 7497, Non-Professional subscriber

---

## Summary

API market-data requests (`reqMktData`) for live (real-time) data return a "not subscribed for API" error — presenting as **error 10089** ("Requested market data requires additional subscription for API. Delayed market data is available.") and, on live-only requests, as **error 10168** ("Requested market data is not subscribed. Delayed market data is not enabled.") — even though:

- The **Market Data API Acknowledgement is signed and shows "enabled"** (signed 2026-06-19).
- "**Share market data with paper trading account**" is enabled on the parent account.
- The subscription "**US Real-Time Non Consolidated Streaming Quotes**" (Fee Waived) is listed as active.
- The **account NLV updates live** during market hours (confirmed re-marking minute-to-minute), so the account itself is receiving real-time data — but the **API is not being served live streaming quotes**.

This worked previously and stopped on the night of 2026-06-18/19. It has now persisted through a full weekend and is still failing on **Monday 2026-06-22 at ~13:53 ET, during regular market hours.**

## Exact error (from the API)

Requesting `reqMarketDataType(1)` then `reqMktData` for liquid SMART-routed symbols returns:

```
Error 10089, reqId N: Requested market data requires additional subscription for API.
See link in 'Market Data Connections' dialog for more details. Delayed market data is available.
  SPY  ARCA/TOP/ALL    (conId 756733)
  AAPL NASDAQ.NMS/TOP/ALL (conId 265598)
```

A live-only request (no delayed fallback) returns instead:

```
Error 10168, reqId N: Requested market data is not subscribed. Delayed market data is not enabled.
```

Under `reqMarketDataType(3)` (delayed) the same symbols return data normally (e.g. SPY ~746.94), so the connection and routing are fine — only the **live API entitlement** is missing. Separately, the **account NLV updates in real time** during the session (sampled e.g. 1,073,385 → 1,074,029 → 1,074,572 over several minutes on 2026-06-22), confirming the account itself receives live data while the API does not.

## Connection / data farm status (healthy)

- Interactive Brokers API Server: **connected**
- Market Data Farm: **ON** — usfarm.nj, usopt, usfarm
- Historical Data Farm: **ON** — ushmds
- Market Data Subscriber status: **Non-Professional**

## Troubleshooting already performed

1. Signed the Market Data API Terms & Conditions Acknowledgement (now shows "access is enabled," signed 2026-06-19).
2. Verified "Share market data with paper trading account" is enabled on the live account.
3. Performed multiple full IB Gateway stop/start cycles (clean re-logins).
4. Waited through the weekend / IBKR's overnight reset and retested at Monday's (2026-06-22) regular-session open — still failing.
5. Identified and fully cleared a competing live session: a Client Portal (web) session had persisted (it survived a machine reboot via a saved login). While it was active the API returned error **10197** ("No market data during competing live session"). After an explicit logout and closing the browser, 10197 no longer appears — yet live API market data **still fails (error 10168)**. This rules out a competing-session conflict and isolates the problem to the API market-data subscription/entitlement.

## Questions for support

1. Why does the account receive live data (NLV updates in real time) while the **API** (`reqMktData`, type 1) returns 10089 for the same symbols?
2. Does live streaming market data **via the API for SMART-routed US equities** require a market-data subscription beyond "US Real-Time Non Consolidated Streaming Quotes" (e.g., the consolidated network bundles / US Securities Snapshot + Streaming bundle)? If so, exactly which subscription is required for the paper account to receive live API quotes?
3. If the existing subscription should be sufficient, can you confirm the API market-data entitlement is correctly provisioned/shared to paper account DUN786964 and re-push it if needed?

## Desired outcome

Live (real-time, type-1) market data returned via the TWS/Gateway API (`reqMktData`) for SMART-routed US equities on paper account DUN786964.
