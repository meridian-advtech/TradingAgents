#!/usr/bin/env python3
"""
Test EDGAR Form 4 fetch function to diagnose why it's returning zero records.
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

def test_edgar_fetch():
    print("=" * 60)
    print("EDGAR FORM 4 FETCH DIAGNOSTIC")
    print("=" * 60)
    
    # Test with empty sets and verbose output
    existing = set()
    added_this_run = set()
    
    print("\nRunning EDGAR fetch with debug output...")
    result = _fetch_edgar(existing, added_this_run, dry_run=False, debug=True)
    
    print(f"\nResults:")
    print(f"  Candidates found: {len(result['candidates'])}")
    print(f"  Errors: {len(result['errors'])}")
    print(f"  Stats: {result['stats']}")
    
    if result['errors']:
        print(f"\nErrors encountered:")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['candidates']:
        print(f"\nCandidates:")
        for candidate in result['candidates'][:3]:  # Show first 3
            print(f"  - {candidate.get('ticker', 'N/A')}: ${candidate.get('value_m', 0):.1f}M")
    else:
        print(f"\nNo candidates found. Investigating why...")
        
        # Let's manually test the EFTS API
        print(f"\nTesting EFTS API directly...")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        url = (f"https://efts.sec.gov/LATEST/search-index"
               f"?forms=4&dateRange=custom&startdt={start}&enddt={today}")
        
        print(f"URL: {url}")
        
        try:
            # Try to get SEC headers if they exist
            try:
                from kairos_intake import SEC_HEADERS, FETCH_TIMEOUT
            except ImportError:
                SEC_HEADERS = {"User-Agent": "Kairos/1.0"}
                FETCH_TIMEOUT = 30
            
            resp = requests.get(url, headers=SEC_HEADERS, timeout=FETCH_TIMEOUT)
            print(f"HTTP Status: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                print(f"EFTS API returned {len(hits)} Form 4 filings")
                
                if hits:
                    print(f"\nFirst filing sample:")
                    first_hit = hits[0]
                    print(f"  ID: {first_hit.get('_id', 'N/A')}")
                    print(f"  CIKs: {first_hit.get('_source', {}).get('ciks', [])}")
                    
                    # Check if we can parse the filing ID
                    filing_id = first_hit.get("_id", "")
                    if ":" in filing_id:
                        accession, xml_filename = filing_id.split(":", 1)
                        print(f"  Accession: {accession}")
                        print(f"  XML filename: {xml_filename}")
                    else:
                        print(f"  WARNING: Filing ID format unexpected: {filing_id}")
                else:
                    print("  No filings returned from EFTS API")
                    print(f"  Full response keys: {list(data.keys())}")
                    if "hits" in data:
                        print(f"  Hits structure: {data['hits'].keys()}")
            else:
                print(f"  Error: HTTP {resp.status_code}")
                print(f"  Response: {resp.text[:200]}")
                
        except Exception as e:
            print(f"  Exception: {e}")

if __name__ == "__main__":
    test_edgar_fetch()