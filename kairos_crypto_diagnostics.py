#!/usr/bin/env python3
"""
Crypto Signal Diagnostic Logger
Logs comprehensive crypto signal data every cycle for debugging and monitoring.
"""

import json
import os
import requests
from datetime import datetime, timezone
import logging
from typing import Dict, Any
from kairos_crypto_cache import get_simple_price, get_ohlc, get_fear_greed_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kairos_monitor.log'),
        logging.StreamHandler()
    ]
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def log_crypto_diagnostics():
    """
    Gather and log comprehensive crypto signal diagnostics.
    Runs every cycle to provide visibility into signal calculations.
    """
    
    # Initialize diagnostics data
    diagnostics = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'btc_price': None,
        'eth_price': None,
        'btc_4h_change': None,
        'eth_4h_change': None,
        'fear_greed_index': None,
        'spy_daily_change': None,
        'market_open': None,
        'btc_7day_ma': None,
        'signals': {
            'HOT-CRYPTO-MOMENTUM': False,
            'HOT-CRYPTO-MACRO': False,
            'HOT-CRYPTO-BASELINE': False
        },
        'signal_reasons': {
            'HOT-CRYPTO-MOMENTUM': None,
            'HOT-CRYPTO-MACRO': None,
            'HOT-CRYPTO-BASELINE': None
        },
        'simulated_positions': []
    }
    
    try:
        # 1. Get BTC and ETH prices from cache
        try:
            price_data = get_simple_price('bitcoin,ethereum')
            if price_data:
                diagnostics['btc_price'] = price_data.get('bitcoin', {}).get('usd')
                diagnostics['eth_price'] = price_data.get('ethereum', {}).get('usd')
        except Exception as e:
            logging.warning(f"Price cache fetch failed: {e}")

        # 1.5. Get BTC 7-day moving average from cache
        try:
            candles = get_ohlc('bitcoin', 7)
            if candles and len(candles) > 0:
                closes = [c[4] for c in candles if len(c) >= 5 and c[4] is not None]
                if closes:
                    sma = sum(closes) / len(closes)
                    diagnostics['btc_7day_ma'] = round(sma, 2)
        except Exception as e:
            logging.warning(f"BTC 7-day MA cache fetch failed: {e}")

        # 1.7. Get BTC 4-hour price change from cache
        try:
            candles = get_ohlc('bitcoin', 1)
            if candles and len(candles) >= 2:
                recent_price = candles[-1][4]
                four_hours_ago_price = candles[0][4]
                if recent_price and four_hours_ago_price and four_hours_ago_price > 0:
                    pct_change = (recent_price - four_hours_ago_price) / four_hours_ago_price * 100
                    diagnostics['btc_4h_change'] = round(pct_change, 2)
        except Exception as e:
            logging.warning(f"BTC 4-hour change cache fetch failed: {e}")

        # 1.8. Get ETH 4-hour price change from cache
        try:
            candles = get_ohlc('ethereum', 1)
            if candles and len(candles) >= 2:
                recent_price = candles[-1][4]
                four_hours_ago_price = candles[0][4]
                if recent_price and four_hours_ago_price and four_hours_ago_price > 0:
                    pct_change = (recent_price - four_hours_ago_price) / four_hours_ago_price * 100
                    diagnostics['eth_4h_change'] = round(pct_change, 2)
        except Exception as e:
            logging.warning(f"ETH 4-hour change cache fetch failed: {e}")
        
        # 2. Get fear & greed index from cache
        try:
            fg_data = get_fear_greed_index()
            if fg_data and isinstance(fg_data, dict) and 'data' in fg_data and len(fg_data['data']) > 0:
                diagnostics['fear_greed_index'] = fg_data['data'][0]['value']
        except Exception as e:
            logging.warning(f"Fear & Greed index cache fetch failed: {e}")
        
        # 3. Get SPY data (market hours check)
        try:
            now = datetime.now(timezone.utc)
            market_open = 9 <= now.hour <= 16 and now.weekday() < 5
            diagnostics['market_open'] = market_open
        except Exception as e:
            logging.warning(f"Market data fetch failed: {e}")
        
        # 4. Check simulated crypto positions
        try:
            import sqlite3
            conn = sqlite3.connect('kairos.db')
            cursor = conn.cursor()
            cursor.execute("""
                SELECT asset, SUM(quantity) as qty, 
                       ROUND(SUM(entry_price * quantity) / SUM(quantity), 2) as avg_cost
                FROM crypto_holdings 
                WHERE sold_date IS NULL
                GROUP BY asset
            """)
            for row in cursor.fetchall():
                diagnostics['simulated_positions'].append({
                    'asset': row[0],
                    'quantity': row[1],
                    'avg_cost': row[2]
                })
            conn.close()
        except Exception as e:
            logging.warning(f"Crypto positions fetch failed: {e}")

        # 5. Populate signals and signal_reasons from latest signal summary
        try:
            summary_path = os.path.join(SCRIPT_DIR, 'kairos_crypto_signal_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    signal_summary = json.load(f)

                signal_tags = signal_summary.get('signal_tags', {})
                momentum_hits = signal_summary.get('momentum_hits', {})
                macro = signal_summary.get('macro', {})
                baseline = signal_summary.get('baseline', {})

                # HOT-CRYPTO-MOMENTUM
                momentum_assets = [s for s, tags in signal_tags.items()
                                   if 'HOT-CRYPTO-MOMENTUM' in tags]
                if momentum_assets:
                    diagnostics['signals']['HOT-CRYPTO-MOMENTUM'] = True
                    best_sym = next((s for s in momentum_assets if s in momentum_hits), None)
                    if best_sym:
                        hit = momentum_hits[best_sym]
                        diagnostics['signal_reasons']['HOT-CRYPTO-MOMENTUM'] = (
                            f"{best_sym} +{hit['gain_4h']:.1f}% in 4h"
                        )
                    else:
                        diagnostics['signal_reasons']['HOT-CRYPTO-MOMENTUM'] = (
                            f"fired on {', '.join(momentum_assets)}"
                        )
                else:
                    btc_pct = diagnostics.get('btc_4h_change')
                    eth_pct = diagnostics.get('eth_4h_change')
                    btc_str = f"{btc_pct:+.2f}%" if btc_pct is not None else "n/a"
                    eth_str = f"{eth_pct:+.2f}%" if eth_pct is not None else "n/a"
                    diagnostics['signal_reasons']['HOT-CRYPTO-MOMENTUM'] = (
                        f"below 3% threshold (BTC: {btc_str}, ETH: {eth_str})"
                    )

                # HOT-CRYPTO-MACRO
                if macro.get('fired'):
                    diagnostics['signals']['HOT-CRYPTO-MACRO'] = True
                    diagnostics['signal_reasons']['HOT-CRYPTO-MACRO'] = (
                        macro.get('details', 'risk-on conditions met')
                    )
                elif macro:
                    diagnostics['signal_reasons']['HOT-CRYPTO-MACRO'] = (
                        macro.get('details', 'risk-off conditions')
                    )
                else:
                    diagnostics['signal_reasons']['HOT-CRYPTO-MACRO'] = (
                        'fear/greed data unavailable'
                    )

                # HOT-CRYPTO-BASELINE
                if baseline.get('fired'):
                    diagnostics['signals']['HOT-CRYPTO-BASELINE'] = True
                    diagnostics['signal_reasons']['HOT-CRYPTO-BASELINE'] = (
                        f"BTC ${baseline.get('btc_price', 0):,.0f} > "
                        f"7d SMA ${baseline.get('sma_7d', 0):,.0f} "
                        f"(+{baseline.get('pct_above', 0):.1f}%)"
                    )
                elif baseline:
                    diagnostics['signal_reasons']['HOT-CRYPTO-BASELINE'] = (
                        f"BTC ${baseline.get('btc_price', 0):,.0f} < "
                        f"7d SMA ${baseline.get('sma_7d', 0):,.0f}"
                    )
                else:
                    diagnostics['signal_reasons']['HOT-CRYPTO-BASELINE'] = (
                        'BTC baseline data unavailable'
                    )
        except Exception as e:
            logging.warning(f"Signal summary load failed: {e}")
        
        # Log to file
        with open('kairos_monitor.log', 'a') as f:
            f.write('\n' + '='*60 + '\n')
            f.write('CRYPTO DIAGNOSTICS: ' + diagnostics['timestamp'] + '\n')
            f.write(json.dumps(diagnostics, indent=2))
            f.write('\n' + '='*60 + '\n')
        
        # Log to Slack
        try:
            from kairos_alerts import post_message
            slack_message = f"*CRYPTO DIAGNOSTICS*\n```{json.dumps(diagnostics, indent=2)}```"
            success = post_message("log", slack_message)
            if success:
                logging.info("Posted crypto diagnostics to #kairos-log")
            else:
                logging.warning("Slack posting to #kairos-log failed")
        except Exception as e:
            logging.warning(f"Slack posting failed: {e}")
        
        return diagnostics
        
    except Exception as e:
        logging.error(f"Crypto diagnostics failed: {e}")
        return None


def main():
    """Entry point for testing."""
    result = log_crypto_diagnostics()
    if result:
        print("Crypto diagnostics logged successfully")
    else:
        print("Crypto diagnostics failed")


if __name__ == '__main__':
    main()
