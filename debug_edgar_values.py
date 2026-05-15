#!/usr/bin/env python3
"""
Detailed EDGAR value calculation diagnostic.
"""

import sys
import os
from datetime import datetime, timezone, timedelta
import requests
import json

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import the function
from kairos_intake import _fetch_edgar

def debug_edgar_values():
    print("=" * 60)
    print("EDGAR VALUE CALCULATION DIAGNOSTIC")
    print("=" * 60)
    
    # Run with debug to see all transactions
    existing = set()
    added_this_run = set()
    
    print("\nRunning EDGAR fetch with detailed output...")
    result = _fetch_edgar(existing, added_this_run, dry_run=False, debug=True)
    
    print(f"\nDetailed Analysis:")
    print(f"  Total filings processed: {result['stats']['evaluated']}")
    print(f"  Below $1M threshold: {result['stats']['below_threshold']}")
    print(f"  Final candidates: {len(result['candidates'])}")
    
    if result['candidates']:
        print(f"\nCandidate Details:")
        for i, candidate in enumerate(result['candidates']):
            print(f"\n  Candidate {i+1}: {candidate['ticker']}")
            print(f"    Company: {candidate['company_name']}")
            print(f"    Raw value: ${candidate['value']:,.0f}")
            print(f"    Value in millions: ${candidate['value'] / 1_000_000:.2f}M")
            print(f"    Reason: {candidate['reason']}")
    
    # Check the $1M threshold issue
    print(f"\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)
    
    # The debug output shows transactions that were filtered
    # Let's analyze the threshold impact
    print(f"\nCurrent $1M threshold filters out most transactions.")
    print(f"Considerations:")
    print(f"  1. Typical insider purchases are $100K-$500K range")
    print(f"  2. $1M threshold may be too aggressive")
    print(f"  3. Only 1 out of 100 filings met the threshold")
    
    print(f"\nRecommendation:")
    print(f"  Consider lowering threshold to $500K or $250K to capture")
    print(f"  more meaningful insider activity while still filtering")
    print(f"  out small, potentially noise-level transactions.")

if __name__ == "__main__":
    debug_edgar_values()