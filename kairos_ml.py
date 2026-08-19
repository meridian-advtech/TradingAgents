"""
Kairos ML Track 1 Pattern Recognition Module

Provides machine-learning-based signal scoring for Kairos trading pipeline.

Features:
- Training: RandomForestClassifier on closed trades from kairos_ml_outcomes.db
- Scoring: ML confidence and signal strength for candidate tickers
- Integration: run_ml_phase() hooks into kairos_run.py pipeline

Usage:
    from kairos_ml import train_model, score_candidate, run_ml_phase
    
    # Train (or load existing) model
    model_info = train_model()
    
    # Score a candidate
    result = score_candidate("AAPL", {"news": "bullish", "macro": "easing"})
    
    # Run ML phase on screen results
    scored_candidates = run_ml_phase(candidates)
"""

import json
import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "kairos_ml_outcomes.db")
MODEL_PATH = os.path.join(SCRIPT_DIR, "kairos_ml_model.pkl")
DECISIONS_LOG = os.path.join(SCRIPT_DIR, "kairos_decisions.log")
SCREEN_RESULT_FILE = os.path.join(SCRIPT_DIR, "kairos_screen_result.json")

# Minimum rows required for training
MIN_TRAINING_ROWS = 20

# Feature names
CATEGORICAL_FEATURES = ["news", "macro", "legis_sentiment", "sector", "day_of_week", "hour_of_day"]
NUMERICAL_FEATURES = ["hold_days"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Global model cache
_model_cache: dict = {"model": None, "encoder": None, "feature_names": None, "trade_count": 0}


def _get_logger():
    """Get or create the ML module logger."""
    logger = logging.getLogger("kairos_ml")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("  [ML] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _load_db_connection():
    """Create a SQLite connection to the outcomes database."""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _extract_signal_tags_to_features(signals_fired: list[str] | str | None) -> dict:
    """Convert signal tags list to feature dictionary.
    
    Extracts signal type categories from tags like:
    - HOT-NEWS, news=bullish, news_sentiment=positive -> news
    - HOT-MACRO, macro=easing, macro_environment=easing -> macro  
    - HOT-CRYPTO, crypto=risk-on, risk_appetite=risk-on -> crypto
    - HOT-CONGRESS, congress, legislative -> legis_sentiment
    
    Returns dict with signal category values.
    """
    if not signals_fired:
        return {}
    
    if isinstance(signals_fired, str):
        try:
            signals_fired = json.loads(signals_fired)
        except (json.JSONDecodeError, TypeError):
            signals_fired = []
    
    features = {
        "news": "neutral",
        "macro": "stable", 
        "legis_sentiment": "neutral"
    }
    
    tag_str = " ".join(signals_fired).lower()
    
    # News sentiment
    if any(t in tag_str for t in ["news=bullish", "news_sentiment=positive", "news=positive", "hot-news"]):
        features["news"] = "bullish"
    elif any(t in tag_str for t in ["news=bearish", "news_sentiment=negative", "news=negative"]):
        features["news"] = "bearish"
    
    # Macro environment
    if any(t in tag_str for t in ["macro=easing", "macro_environment=easing", "macro=accommodative", 
            "macro=supportive", "macro=lower", "macro=cutting", "hot-macro"]):
        features["macro"] = "easing"
    elif any(t in tag_str for t in ["macro=tightening", "macro_environment=tightening", 
               "macro=hawkish", "macro=rising", "macro=hiking"]):
        features["macro"] = "tightening"
    
    # Legislative sentiment
    if any(t in tag_str for t in ["legislative=risky", "legislative=threat", "legislative=negative",
            "legislative=targeting", "hot-congress", "congress", "legislative"]):
        features["legis_sentiment"] = "risky"
    elif any(t in tag_str for t in ["legislative=safe", "legislative=positive", "legislative=supportive"]):
        features["legis_sentiment"] = "safe"
    
    return features


def _get_day_of_week(ts: str) -> str:
    """Extract day of week from ISO timestamp."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%A")
    except (ValueError, AttributeError):
        return "Unknown"


def _get_hour_of_day(ts: str) -> str:
    """Extract hour of day (bucketed) from ISO timestamp."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        hour = dt.hour
        # Bucket into 4-hour blocks
        if hour < 6:
            return "0-6"
        elif hour < 10:
            return "6-10"
        elif hour < 14:
            return "10-14"
        elif hour < 18:
            return "14-18"
        elif hour < 22:
            return "18-22"
        else:
            return "22-24"
    except (ValueError, AttributeError):
        return "Unknown"


def _compute_hold_days(hold_duration_mins: int | None) -> float:
    """Convert hold duration minutes to days."""
    if hold_duration_mins is None or hold_duration_mins <= 0:
        return 0.0
    return round(hold_duration_mins / (24 * 60), 2)


def _prepare_training_data():
    """Query DB for closed trades and prepare features + labels.
    
    Returns (X, y, row_count) or (None, None, 0) if insufficient data.
    """
    conn = _load_db_connection()
    
    query = """
        SELECT ticker, signals_fired, sector, hold_duration_mins, 
               timestamp_entry, outcome_label, pnl_pct
        FROM trade_outcomes 
        WHERE outcome_label IS NOT NULL 
        ORDER BY timestamp_entry
    """
    
    rows = conn.execute(query).fetchall()
    conn.close()
    
    if len(rows) < MIN_TRAINING_ROWS:
        return None, None, len(rows)
    
    X = []
    y = []  # 1=WIN, 0=LOSS (binary classification)
    
    for row in rows:
        features = {}
        
        # Extract signal features
        signal_features = _extract_signal_tags_to_features(row["signals_fired"])
        features.update(signal_features)
        
        # Sector
        features["sector"] = row["sector"] or "Unknown"
        
        # Hold days
        features["hold_days"] = _compute_hold_days(row["hold_duration_mins"])
        
        # Time features
        features["day_of_week"] = _get_day_of_week(row["timestamp_entry"])
        features["hour_of_day"] = _get_hour_of_day(row["timestamp_entry"])
        
        X.append(features)
        
        # Label: 1 for WIN, 0 for LOSS (SCRATCH treated as LOSS for simplicity)
        label = 1 if row["outcome_label"] == "WIN" else 0
        y.append(label)
    
    return X, y, len(rows)


def _encode_features(X: list[dict], encoder: OneHotEncoder | None = None, 
                      feature_names: list[str] | None = None) -> tuple[np.ndarray, OneHotEncoder, list[str]]:
    """Encode categorical features using OneHotEncoder.
    
    Returns (encoded_X, encoder, feature_names)
    """
    if not X:
        return np.array([]), encoder, feature_names
    
    # Extract categorical and numerical values
    cat_values = []
    num_values = []
    
    for feat in X:
        cat_row = [feat.get(f, "Unknown") for f in CATEGORICAL_FEATURES]
        num_row = [feat.get(f, 0.0) for f in NUMERICAL_FEATURES]
        cat_values.append(cat_row)
        num_values.append(num_row)
    
    cat_array = np.array(cat_values)
    num_array = np.array(num_values)
    
    # Fit or use existing encoder
    if encoder is None or feature_names is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_encoded = encoder.fit_transform(cat_array)
        # Get feature names for categorical
        cat_feat_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        all_feat_names = list(cat_feat_names) + NUMERICAL_FEATURES
    else:
        cat_encoded = encoder.transform(cat_array)
        all_feat_names = feature_names
    
    # Combine categorical and numerical
    encoded_X = np.hstack([cat_encoded, num_array])
    
    return encoded_X, encoder, all_feat_names


# Sequential on purpose. n_jobs=-1 used to be set here, and it was the last
# fork() left inside kairos_run.py's own process.
#
# joblib spins up its parallel backend on the first n_jobs!=1 call, and that
# backend starts multiprocessing's resource_tracker helper via
# _posixsubprocess.fork_exec. Forking THIS process is what crashes: it is
# heavy, multi-threaded and already networked, and on this host Little Snitch
# and Tailscale install NetworkExtension filters, so Network.framework
# registers a pthread_atfork child handler that SIGSEGVs in the forked child
# before exec() replaces it:
#
#     EXC_BAD_ACCESS in os_log_preferences_refresh
#       <- NEFlowDirectorDestroy <- nw_settings_child_has_forked
#       <- _pthread_atfork_child_handlers <- fork
#       <- _posixsubprocess.subprocess_fork_exec
#
# Worse than a one-off: when the tracker dies, multiprocessing relaunches it,
# which forks again, which crashes again. That relaunch loop is what turned
# one fork into the escalating bursts logged on 2026-08-19 (1 crash at 09:37,
# 4 at 09:47, 11 at 11:30), each one accompanied by
# "resource_tracker: process died unexpectedly, relaunching" in
# kairos_scheduler.log. Same crash class as kairos_alerts.py's Slack
# transport, but there is no subprocess call of ours to convert here — the
# fork is inside joblib, so the fix is to never ask for a second process.
#
# Costs nothing: measured on the live model (100 trees, depth 10, 19
# features, 16 candidates) predictions are BITWISE identical and sequential
# is ~6.5x FASTER (105ms vs 681ms per 50 predicts) — this workload is far too
# small to amortize joblib's dispatch overhead. Revisit only if the corpus
# grows by orders of magnitude, and then measure the fork risk again first.
ML_N_JOBS = 1


def _force_sequential(model):
    """Pin a model to n_jobs=1, in place, and return it.

    Needed on the load path as well as the train path: kairos_ml_model.pkl was
    pickled from a model built with n_jobs=-1, and unpickling restores that
    attribute, so a cached model would still fork on its first predict even
    after this module stopped constructing parallel ones.
    """
    if model is not None and getattr(model, "n_jobs", None) != ML_N_JOBS:
        model.n_jobs = ML_N_JOBS
    return model


def train_model(force_retrain: bool = False) -> dict:
    """Train (or load) RandomForestClassifier on closed trade outcomes.
    
    Queries kairos_ml_outcomes.db for all closed trades with known outcomes,
    extracts features from signal tags, sector, hold duration, and entry time.
    
    Returns dict with:
        - model: the trained RandomForestClassifier (or None if <20 rows)
        - encoder: the fitted OneHotEncoder
        - feature_names: list of feature names
        - trade_count: number of closed trades used for training
        - accuracy: cross-validation accuracy (or None)
        - feature_importances: dict of feature importance scores (or None)
    
    Model is saved to kairos_ml_model.pkl for persistence.
    """
    global _model_cache
    
    logger = _get_logger()
    
    # Check if we have a cached model
    if not force_retrain and _model_cache["model"] is not None:
        return _model_cache
    
    # Try to load from disk
    if not force_retrain and os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                cached = pickle.load(f)
            # Older pickles carry n_jobs=-1; see _force_sequential.
            _force_sequential(cached.get("model"))
            _model_cache = cached
            logger.info(f"Model loaded from {MODEL_PATH} (trained on {cached['trade_count']} trades)")
            return _model_cache
        except Exception as e:
            logger.info(f"Failed to load model from disk: {e}")
    
    # Prepare data
    X, y, trade_count = _prepare_training_data()
    
    if trade_count < MIN_TRAINING_ROWS:
        logger.info(f"Insufficient data for training: {trade_count} closed trades (need {MIN_TRAINING_ROWS}+)")
        _model_cache = {
            "model": None,
            "encoder": None,
            "feature_names": None,
            "trade_count": trade_count,
            "accuracy": None,
            "feature_importances": None
        }
        return _model_cache
    
    # Encode features
    X_encoded, encoder, feature_names = _encode_features(X)
    y_array = np.array(y)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=ML_N_JOBS
    )
    model.fit(X_encoded, y_array)
    
    # Cross-validation accuracy
    cv_scores = cross_val_score(model, X_encoded, y_array, cv=5, scoring="accuracy")
    accuracy = float(np.mean(cv_scores))
    
    # Feature importances
    importances = model.feature_importances_
    feature_importances = dict(zip(feature_names, importances))
    
    # Build cache
    _model_cache = {
        "model": model,
        "encoder": encoder,
        "feature_names": feature_names,
        "trade_count": trade_count,
        "accuracy": accuracy,
        "feature_importances": feature_importances
    }
    
    # Save to disk
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(_model_cache, f)
        logger.info(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        logger.info(f"Failed to save model: {e}")
    
    # Log training details
    _log_training_session(trade_count, accuracy, feature_importances)
    
    logger.info(f"Trained on {trade_count} closed trades | CV Accuracy: {accuracy:.3f}")
    
    return _model_cache


def _log_training_session(trade_count: int, accuracy: float, feature_importances: dict) -> None:
    """Log training session details to kairos_decisions.log."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Sort features by importance
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:10]  # Top 10 features
    
    feature_str = ", ".join([f"{name}:{val:.3f}" for name, val in top_features])
    
    log_entry = {
        "type": "ML_TRAINING",
        "timestamp": ts,
        "trade_count": trade_count,
        "model_type": "RandomForestClassifier",
        "cv_accuracy": round(accuracy, 4),
        "n_estimators": 100,
        "features_used": len(feature_importances),
        "top_features": feature_str,
        "min_training_rows": MIN_TRAINING_ROWS
    }
    
    with open(DECISIONS_LOG, "a") as f:
        f.write("\n" + json.dumps(log_entry, indent=2))
        f.write("\n" + "=" * 72 + "\n")


def score_candidate(ticker: str, signals: dict | None = None) -> dict:
    """Score a candidate ticker using the trained ML model.
    
    Args:
        ticker: Stock ticker symbol
        signals: Optional dict with signal tag information.
                If None, tries to load from screen result or returns default.
    
    Returns dict with:
        - ml_confidence: float 0-1 (predicted probability of WIN)
        - ml_signal: "STRONG" (confidence >= 0.7), "WEAK" (confidence <= 0.4), 
                     or "NEUTRAL" (between 0.4 and 0.7)
        - ml_trained_on: int (number of trades model was trained on, or 0)
    
    If model not trained (insufficient data), returns defaults:
        {"ml_confidence": 0.5, "ml_signal": "NEUTRAL", "ml_trained_on": 0}
    """
    # Ensure model is loaded
    model_info = train_model()
    
    model = model_info["model"]
    encoder = model_info["encoder"]
    feature_names = model_info["feature_names"]
    trade_count = model_info["trade_count"]
    
    # If no model trained, return default
    if model is None or encoder is None:
        return {
            "ml_confidence": 0.5,
            "ml_signal": "NEUTRAL",
            "ml_trained_on": trade_count
        }
    
    # Build feature dict for this candidate
    features = _build_candidate_features(ticker, signals)
    
    # Encode features
    X_candidate, _, _ = _encode_features([features], encoder, feature_names)
    
    if X_candidate.size == 0:
        return {
            "ml_confidence": 0.5,
            "ml_signal": "NEUTRAL",
            "ml_trained_on": trade_count
        }
    
    # Predict probability
    try:
        probas = model.predict_proba(X_candidate)
        # Get probability of class 1 (WIN)
        win_prob = float(probas[0][1]) if probas.shape[1] > 1 else 0.5
    except Exception as e:
        _get_logger().info(f"Prediction failed for {ticker}: {e}")
        win_prob = 0.5
    
    # Map confidence to signal strength
    if win_prob >= 0.7:
        signal = "STRONG"
    elif win_prob <= 0.4:
        signal = "WEAK"
    else:
        signal = "NEUTRAL"
    
    return {
        "ml_confidence": round(win_prob, 4),
        "ml_signal": signal,
        "ml_trained_on": trade_count
    }


def _build_candidate_features(ticker: str, signals: dict | None) -> dict:
    """Build feature dictionary for a candidate ticker.
    
    Uses signals dict if provided, otherwise extracts from screen results
    or uses defaults.
    """
    features = {
        "news": "neutral",
        "macro": "stable",
        "legis_sentiment": "neutral",
        "sector": "Unknown",
        "hold_days": 0.0,
        "day_of_week": "Unknown",
        "hour_of_day": "Unknown"
    }
    
    # Try to get sector from universe file
    try:
        universe_file = os.path.join(SCRIPT_DIR, "kairos_universe.json")
        if os.path.exists(universe_file):
            with open(universe_file) as f:
                universe = json.load(f)
            # Search in tier_a equities
            for cat, symbols in universe.get("tier_a", {}).get("equities", {}).items():
                if ticker in symbols:
                    features["sector"] = cat
                    break
    except Exception:
        pass
    
    # Use provided signals if available
    if signals:
        if isinstance(signals, dict):
            # Direct mapping
            for key in ["news", "macro", "legis_sentiment", "sector"]:
                if key in signals:
                    features[key] = signals[key]
        elif isinstance(signals, list):
            # Convert list of tags to feature dict
            signal_features = _extract_signal_tags_to_features(signals)
            features.update(signal_features)
    else:
        # Try to load from screen result
        try:
            if os.path.exists(SCREEN_RESULT_FILE):
                with open(SCREEN_RESULT_FILE) as f:
                    screen_data = json.load(f)
                signal_tags = screen_data.get("signal_tags", {})
                if ticker in signal_tags:
                    signal_features = _extract_signal_tags_to_features(signal_tags[ticker])
                    features.update(signal_features)
        except Exception:
            pass
    
    # Set time features to current time
    now = datetime.now(timezone.utc)
    features["day_of_week"] = now.strftime("%A")
    features["hour_of_day"] = _get_hour_of_day(now.isoformat())
    
    return features


def run_ml_phase(candidates: list[dict]) -> list[dict]:
    """Integration hook: Score all candidates from screening phase.
    
    Takes the shortlisted candidates from kairos_screen_result.json format
    and returns the list with ml_confidence and ml_signal added to each
    candidate dict.
    
    Args:
        candidates: List of candidate dicts. Each dict should have at least:
                    - ticker: str (required)
                    - signals: dict or list (optional, signal information)
                    - sector: str (optional, sector classification)
    
    Returns:
        List of candidate dicts with ml_confidence and ml_signal added.
        If model not trained, adds default values (0.5, "NEUTRAL", 0).
    """
    logger = _get_logger()
    
    if not candidates:
        logger.info("No candidates to score")
        return candidates
    
    logger.info(f"Scoring {len(candidates)} candidates with ML model...")
    
    # Ensure model is loaded
    model_info = train_model()
    model = model_info["model"]
    
    if model is None:
        logger.info("Model not trained (insufficient data) - using default scores")
        for cand in candidates:
            cand["ml_confidence"] = 0.5
            cand["ml_signal"] = "NEUTRAL"
            cand["ml_trained_on"] = model_info["trade_count"]
    else:
        for cand in candidates:
            ticker = cand.get("ticker", "")
            signals = cand.get("signals") or cand.get("signal_tags")
            result = score_candidate(ticker, signals)
            cand["ml_confidence"] = result["ml_confidence"]
            cand["ml_signal"] = result["ml_signal"]
            cand["ml_trained_on"] = result["ml_trained_on"]
            logger.info(f"  {ticker}: confidence={result['ml_confidence']:.3f}, signal={result['ml_signal']}")
    
    return candidates


def get_ml_model_info() -> dict:
    """Return current model information without triggering training."""
    global _model_cache
    return _model_cache.copy()


def reset_model_cache():
    """Reset the in-memory model cache (for testing or model refresh)."""
    global _model_cache
    _model_cache = {"model": None, "encoder": None, "feature_names": None, "trade_count": 0}


# ── Main (standalone test) ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("Kairos ML Track 1 — Pattern Recognition Module")
    print("=" * 72)
    
    # Test training
    print("\n[1] Training model...")
    info = train_model(force_retrain=True)
    print(f"    Model trained on: {info['trade_count']} trades")
    print(f"    CV Accuracy: {info['accuracy']}")
    
    # Test scoring
    print("\n[2] Testing candidate scoring...")
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    for ticker in test_tickers:
        result = score_candidate(ticker, {"news": "bullish", "macro": "easing"})
        print(f"    {ticker}: confidence={result['ml_confidence']:.4f}, "
              f"signal={result['ml_signal']}, trained_on={result['ml_trained_on']}")
    
    # Test run_ml_phase
    print("\n[3] Testing run_ml_phase...")
    candidates = [
        {"ticker": "AAPL", "signals": {"news": "bullish"}},
        {"ticker": "MSFT", "signals": {"news": "neutral", "macro": "easing"}},
        {"ticker": "GOOGL", "signals": {"news": "bearish"}},
    ]
    scored = run_ml_phase(candidates)
    print(f"    Scored {len(scored)} candidates")
    for c in scored:
        print(f"      {c['ticker']}: {c['ml_signal']} ({c['ml_confidence']:.3f})")
    
    print("\n" + "=" * 72)
    print("ML module test complete")
