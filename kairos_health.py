#!/usr/bin/env python3
"""
Kairos System Health Check

Comprehensive health monitoring that runs at the start of every scheduler cycle.
Checks critical system components and dependencies before executing trades.
"""

import os
import sys
import socket
import sqlite3
import json
import time
import platform
import requests
from datetime import datetime, timezone, timedelta
import psutil
import shutil
from typing import Dict, List, Tuple

# Add current directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Configuration
HEALTH_LOG = os.path.join(SCRIPT_DIR, "kairos_monitor.log")
DB_PATH = os.path.join(SCRIPT_DIR, "kairos.db")
REQUIRED_TABLES = [
    "decisions", "holdings", "regime_log", "crypto_decisions", 
    "crypto_holdings", "wash_sale_log", "tax_harvest_log", "reallocation_events"
]
REQUIRED_MODELS = ["llama3.2", "mistral-small3.2"]

# Trading hours for cycle completion check (ET)
TRADING_HOURS_START = 8  # 8 AM ET
TRADING_HOURS_END = 22    # 10 PM ET

class HealthStatus:
    """Health check result container."""
    
    def __init__(self):
        self.checks: List[Tuple[str, str, str]] = []  # (name, status, message)
        self.has_failures = False
        self.has_warnings = False
    
    def add_check(self, name: str, status: str, message: str = ""):
        """Add a health check result."""
        self.checks.append((name, status, message))
        if status == "FAIL":
            self.has_failures = True
        elif status == "WARN":
            self.has_warnings = True
    
    def get_summary(self) -> str:
        """Generate a formatted summary of all checks."""
        lines = []
        lines.append("=" * 60)
        lines.append("KAIROS SYSTEM HEALTH CHECK")
        lines.append("=" * 60)
        
        for name, status, message in self.checks:
            status_symbol = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "⊘"}[status]
            lines.append(f"{status_symbol} {name:40} {status}")
            if message:
                lines.append(f"    {message}")
        
        lines.append("=" * 60)
        overall = "FAIL" if self.has_failures else ("WARN" if self.has_warnings else "PASS")
        lines.append(f"Overall System Status: {overall}")
        lines.append("=" * 60)
        return "\n".join(lines)


def log_health_results(status: HealthStatus) -> None:
    """Log health check results to file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    with open(HEALTH_LOG, "a") as f:
        f.write(f"\n{timestamp}\n")
        f.write(status.get_summary())
        f.write("\n\n")


def send_slack_alert_if_needed(status: HealthStatus) -> None:
    """Send Slack alert only if there are FAIL checks."""
    if not status.has_failures:
        return
    
    try:
        from kairos_alerts import post_message
        
        fail_checks = [f"• {name}: {message}" for name, stat, message in status.checks if stat == "FAIL"]
        alert_message = (
            "🚨 *Kairos System Health Alert* 🚨\n"
            "Critical system failures detected:\n"
            "\n".join(fail_checks)
        )
        
        post_message("kairos-alerts", alert_message)
        status.add_check("Slack Alert", "PASS", "Failure notification sent to #kairos-alerts")
        
    except Exception as e:
        status.add_check("Slack Alert", "FAIL", f"Could not send alert: {e}")


def check_ib_gateway_connection(quick: bool = False) -> Tuple[str, str]:
    """Check if IB Gateway is running and responsive."""
    if quick:
        return "SKIP", "Skipped (quick mode)"
    
    try:
        # Try to connect to IB Gateway on port 7497
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)  # 10 second timeout
        
        start_time = time.time()
        result = sock.connect_ex(("localhost", 7497))
        elapsed = time.time() - start_time
        
        sock.close()
        
        if result == 0:
            return "PASS", f"Connected in {elapsed:.2f}s"
        else:
            return "FAIL", f"Connection refused (error {result})"
            
    except socket.timeout:
        return "FAIL", "Connection timed out after 10 seconds"
    except Exception as e:
        return "FAIL", f"Connection error: {e}"


def check_api_keys() -> Tuple[str, str]:
    """Check that required API keys are present in environment."""
    required_keys = [
        "ANTHROPIC_API_KEY",
        "FINNHUB_API_KEY", 
        "KALSHI_API_KEY",
        "FRED_API_KEY",
        "CONGRESS_API_KEY"
    ]
    
    missing_keys = []
    for key in required_keys:
        if not os.environ.get(key):
            missing_keys.append(key)
    
    if missing_keys:
        return "FAIL", f"Missing: {', '.join(missing_keys)}"
    else:
        return "PASS", f"All {len(required_keys)} keys present"


def check_ollama_running(quick: bool = False) -> Tuple[str, str]:
    """Check if Ollama is running and required models are available."""
    if quick:
        return "SKIP", "Skipped (quick mode)"
    
    try:
        # Check if Ollama API is responsive
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code != 200:
            return "FAIL", f"Ollama API returned status {response.status_code}"
        
        # Check for required models
        models_data = response.json()
        available_models = models_data.get("models", [])
        model_names = [m["name"] for m in available_models]
        
        missing_models = []
        for required_model in REQUIRED_MODELS:
            if required_model not in model_names:
                missing_models.append(required_model)
        
        if missing_models:
            return "WARN", f"Missing models: {', '.join(missing_models)}"
        else:
            return "PASS", f"All {len(REQUIRED_MODELS)} models available"
            
    except requests.ConnectionError:
        return "FAIL", "Ollama not running (connection refused)"
    except Exception as e:
        return "FAIL", f"Ollama check error: {e}"


def check_database_accessible() -> Tuple[str, str]:
    """Check if database is accessible and has required tables."""
    try:
        if not os.path.exists(DB_PATH):
            return "FAIL", f"Database file not found at {DB_PATH}"
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        missing_tables = []
        for required_table in REQUIRED_TABLES:
            if required_table not in existing_tables:
                missing_tables.append(required_table)
        
        conn.close()
        
        if missing_tables:
            return "WARN", f"Missing tables: {', '.join(missing_tables)}"
        else:
            return "PASS", f"All {len(REQUIRED_TABLES)} tables present"
            
    except sqlite3.Error as e:
        return "FAIL", f"Database error: {e}"
    except Exception as e:
        return "FAIL", f"Database check failed: {e}"


def check_last_cycle_completed() -> Tuple[str, str]:
    """Check if the last trading cycle completed successfully."""
    try:
        # Check if we're in trading hours
        now_et = datetime.now(timezone.utc)  # Assuming server is in ET, adjust if needed
        current_hour = now_et.hour
        
        if current_hour < TRADING_HOURS_START or current_hour >= TRADING_HOURS_END:
            return "SKIP", "Outside trading hours (8am-10pm ET)"
        
        # Check for recent decisions
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Look for decisions in the last 35 minutes
        thirty_five_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=35)).strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
            SELECT COUNT(*) FROM decisions 
            WHERE created_at >= ?
        """, (thirty_five_min_ago,))
        
        recent_count = cursor.fetchone()[0]
        conn.close()
        
        if recent_count > 0:
            return "PASS", f"Found {recent_count} recent decisions"
        else:
            return "WARN", "No decisions found in last 35 minutes"
            
    except Exception as e:
        return "FAIL", f"Cycle check error: {e}"


def check_disk_space() -> Tuple[str, str]:
    """Check available disk space."""
    try:
        # Get disk usage for the TradingAgents directory volume
        stat = shutil.disk_usage(SCRIPT_DIR)
        free_gb = stat.free / (1024 ** 3)  # Convert to GB
        
        if free_gb < 5:
            return "WARN", f"Only {free_gb:.1f}GB free (recommended: 5GB+)"
        else:
            return "PASS", f"{free_gb:.1f}GB free"
            
    except Exception as e:
        return "FAIL", f"Disk space check failed: {e}"


def check_memory_pressure() -> Tuple[str, str]:
    """Check system memory availability."""
    try:
        if platform.system() == "Windows":
            # Windows memory check
            mem = psutil.virtual_memory()
        else:
            # Unix/Linux/Mac memory check
            mem = psutil.virtual_memory()
        
        free_gb = mem.available / (1024 ** 3)  # Convert to GB
        
        if free_gb < 4:
            return "WARN", f"Only {free_gb:.1f}GB free (recommended: 4GB+)"
        else:
            return "PASS", f"{free_gb:.1f}GB free"
            
    except Exception as e:
        return "FAIL", f"Memory check failed: {e}"


def run_health_check(quick: bool = False) -> HealthStatus:
    """Run comprehensive system health check."""
    status = HealthStatus()
    
    print("Running Kairos system health check...")
    
    # Run all checks
    checks = [
        ("IB Gateway Connection", lambda: check_ib_gateway_connection(quick)),
        ("API Keys Present", check_api_keys),
        ("Ollama Running", lambda: check_ollama_running(quick)),
        ("Database Accessible", check_database_accessible),
        ("Last Cycle Completed", check_last_cycle_completed),
        ("Disk Space", check_disk_space),
        ("Memory Pressure", check_memory_pressure),
    ]
    
    for check_name, check_func in checks:
        try:
            stat, message = check_func()
            status.add_check(check_name, stat, message)
        except Exception as e:
            status.add_check(check_name, "FAIL", f"Check failed: {e}")
    
    return status


def main():
    """Main entry point for health check."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kairos System Health Check")
    parser.add_argument("--quick", action="store_true", 
                       help="Skip IB Gateway and Ollama checks for fast validation")
    
    args = parser.parse_args()
    
    # Run health check
    status = run_health_check(quick=args.quick)
    
    # Output results
    print("\n" + status.get_summary())
    
    # Log results
    log_health_results(status)
    
    # Send alert if needed
    send_slack_alert_if_needed(status)
    
    # Exit with appropriate code
    if status.has_failures:
        sys.exit(1)
    elif status.has_warnings:
        sys.exit(0)  # Warnings don't prevent execution
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()