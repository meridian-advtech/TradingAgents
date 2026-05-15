#!/usr/bin/env python3
"""
Kairos Intake Sources Audit
Audits the health of all intake data sources.
"""

import sqlite3
import requests
from datetime import datetime, timedelta

def audit_intake_sources():
    print('=' * 60)
    print('KAIROS INTAKE SOURCES AUDIT')
    print('=' * 60)

    # Connect to database
    conn = sqlite3.connect('kairos.db')
    conn.row_factory = sqlite3.Row

    # 1. SEC EDGAR Form 4 Audit
    print('\n1. SEC EDGAR Form 4')
    print('-' * 40)
    try:
        form4_rows = conn.execute('''
            SELECT * FROM decisions 
            WHERE rationale LIKE '%Form 4%' 
            AND timestamp >= datetime('now', '-7 days')
            ORDER BY timestamp DESC
        ''').fetchall()
        
        print(f'Database records (last 7 days): {len(form4_rows)}')
        
        if form4_rows:
            first_row = form4_rows[0]
            print(f'Most recent: {first_row["ticker"]} - {first_row["rationale"][:50]}...')
            print('Status: PASS (database records found)')
        else:
            print('Status: UNKNOWN (no recent records, may not be active)')
            
    except Exception as e:
        print(f'Status: FAIL (database error: {e})')

    # 2. OpenInsider Audit
    print('\n2. OpenInsider')
    print('-' * 40)
    try:
        openinsider_rows = conn.execute('''
            SELECT * FROM decisions 
            WHERE rationale LIKE '%OpenInsider%' 
            AND timestamp >= datetime('now', '-7 days')
            ORDER BY timestamp DESC
        ''').fetchall()
        
        print(f'Database records (last 7 days): {len(openinsider_rows)}')
        
        if openinsider_rows:
            first_row = openinsider_rows[0]
            print(f'Most recent: {first_row["ticker"]} - {first_row["rationale"][:50]}...')
        
        # Try to hit OpenInsider endpoint
        try:
            response = requests.get('http://openinsider.com/screener?s=', timeout=10)
            print(f'HTTP Status: {response.status_code}')
            if response.status_code == 200:
                print('Status: PASS (endpoint responsive)')
            else:
                print('Status: FAIL (endpoint not responsive)')
        except Exception as e:
            print(f'Status: FAIL (connection error: {e})')
            
    except Exception as e:
        print(f'Status: FAIL (database error: {e})')

    # 3. House/Senate Stock Watcher Audit
    print('\n3. House/Senate Stock Watcher')
    print('-' * 40)
    try:
        congress_rows = conn.execute('''
            SELECT * FROM decisions 
            WHERE rationale LIKE '%Congress%' 
            AND timestamp >= datetime('now', '-7 days')
            ORDER BY timestamp DESC
        ''').fetchall()
        
        print(f'Database records (last 7 days): {len(congress_rows)}')
        
        if congress_rows:
            first_row = congress_rows[0]
            print(f'Most recent: {first_row["ticker"]} - {first_row["rationale"][:50]}...')
        
        # Check for congressional data tables
        congress_tables = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE '%congress%'
        """).fetchall()
        
        congress_table_names = [t[0] for t in congress_tables]
        print(f'Congressional tables found: {congress_table_names}')
        
        if congress_tables:
            print('Status: PASS (congressional data tables exist)')
        else:
            print('Status: UNKNOWN (no congressional tables found)')
            
    except Exception as e:
        print(f'Status: FAIL (database error: {e})')

    # 4. Tier C Tickers Added (last 7 days)
    print('\n4. Tier C Tickers Added (Last 7 Days)')
    print('-' * 40)
    try:
        tier_c_rows = conn.execute('''
            SELECT DISTINCT ticker 
            FROM decisions 
            WHERE timestamp >= datetime('now', '-7 days')
            AND (rationale LIKE '%Form 4%' 
                 OR rationale LIKE '%OpenInsider%' 
                 OR rationale LIKE '%Congress%')
        ''').fetchall()
        
        tier_c_tickers = [row[0] for row in tier_c_rows]
        print(f'Unique Tier C tickers added: {len(tier_c_tickers)}')
        
        if tier_c_tickers:
            print(f'Examples: {tier_c_tickers[:5]}')
        
    except Exception as e:
        print(f'Status: FAIL (database error: {e})')

    conn.close()

    print('\n' + '=' * 60)
    print('AUDIT COMPLETE')
    print('=' * 60)

if __name__ == '__main__':
    audit_intake_sources()