"""
AgriTech Soil Analyzer — Production Backend API

WHAT CHANGED vs your original app.py:
  1. Environment config (.env file) instead of hardcoded values
  2. Structured logging with rotating file handler
  3. Rate limiting per IP (configurable)
  4. Response caching for repeated predictions
  5. SHAP explainability in prediction responses
  6. Soil Health Score (weighted 0-100 index)
  7. Input validation with detailed error messages
  8. Custom error classes (ValidationError, ModelError)
  9. NEW endpoint: POST /api/compare — compare 2 soil samples
  10. NEW endpoint: GET /api/seasonal-calendar — planting months
  11. NEW endpoint: POST /api/soil-health-score — health index
  12. Response time headers (X-Response-Time)
  13. Model metadata from training (accuracy, date, type)
  14. Seasonal/planting data added to crop database
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import joblib
import os
import json
import time
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from functools import wraps
from werkzeug.middleware.proxy_fix import ProxyFix

from crop_profiles import CROPS, CROP_INFO, FEATURES, all_crop_scores, crop_suitability
from model_utils import CropShapExplainer, feature_importance

# Load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is optional

# Caching library (optional but recommended)
try:
    from cachetools import TTLCache
except ImportError:
    TTLCache = None


# ============================================================================
# CONFIGURATION — reads from .env file, falls back to defaults
# ============================================================================

class AppConfig:
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    MODEL_DIR = os.getenv('MODEL_DIR', 'models')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    RATE_LIMIT = int(os.getenv('RATE_LIMIT_PER_MINUTE', 30))
    CACHE_TTL = int(os.getenv('CACHE_TTL_SECONDS', 300))
    API_KEY_REQUIRED = os.getenv('API_KEY_REQUIRED', 'false').lower() == 'true'
    API_KEY = os.getenv('API_KEY', '')
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///agritech.db').replace('postgres://', 'postgresql://', 1)


# ============================================================================
# CUSTOM ERROR CLASSES
# ============================================================================

class APIError(Exception):
    """Base error with HTTP status code."""
    def __init__(self, message, status_code=500, details=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class ValidationError(APIError):
    """400 Bad Request — invalid input data."""
    def __init__(self, message, details=None):
        super().__init__(message, status_code=400, details=details)

class ModelError(APIError):
    """503 Service Unavailable — model not loaded."""
    def __init__(self, message, details=None):
        super().__init__(message, status_code=503, details=details)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(app):
    """Configure file + console logging with rotation."""
    os.makedirs('logs', exist_ok=True)

    file_handler = RotatingFileHandler(
        'logs/agritech.log', maxBytes=5_000_000, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S'
    ))

    level = getattr(logging, AppConfig.LOG_LEVEL.upper(), logging.INFO)
    app.logger.setLevel(level)
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)

    if AppConfig.FLASK_ENV == 'production':
        logging.getLogger('werkzeug').setLevel(logging.WARNING)


# ============================================================================
# RATE LIMITER — prevents API abuse
# ============================================================================

class RateLimiter:
    def __init__(self, max_per_minute):
        self.max = max_per_minute
        self.window = 60
        self.requests = {}

    def is_allowed(self, ip):
        now = time.time()
        self.requests.setdefault(ip, [])
        self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window]
        if len(self.requests[ip]) >= self.max:
            return False
        self.requests[ip].append(now)
        return True


# ============================================================================
# RESPONSE CACHE
# ============================================================================

prediction_cache = TTLCache(maxsize=256, ttl=AppConfig.CACHE_TTL) if TTLCache else None

def make_cache_key(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

def new_analysis_id(data):
    return hashlib.md5(
        f"{json.dumps(data, sort_keys=True)}{time.time_ns()}".encode()
    ).hexdigest()[:12]


# ============================================================================
# CROP DATABASE — Burkina Faso / Sahel crops, built from crop_profiles.py so
# the rule-based ranking uses exactly the thresholds that label the ML data.
# ============================================================================

CROP_DATABASE = [
    {
        'name': crop['name'],
        'category': CROP_INFO[crop['name']]['category'],
        'icon': CROP_INFO[crop['name']]['icon'],
        'description': CROP_INFO[crop['name']]['description'],
        'seasons': CROP_INFO[crop['name']]['seasons'],
        'harvest_months': CROP_INFO[crop['name']]['cycle_months'],
        'optimal': crop['optimal'],
        'acceptable': crop['acceptable'],
        'weights': crop['weights'],
    }
    for crop in CROPS.values()
]


# ============================================================================
# MODEL CLASS — multi-output crop suitability regressor
# ============================================================================

class CropSuitabilityModel:
    """
    Predicts a suitability score (0-1) for every crop. Trained by
    train_model.py; if no model file is present the rule-based scores from
    crop_profiles.py (the same function that labels the training data) are
    used instead, so the API keeps working.
    """

    def __init__(self):
        self.model = None
        self.shap_explainer = None
        self.metadata = {}
        self.feature_names = list(FEATURES)
        self.crop_names = [c['name'] for c in CROPS.values()]
        self.is_trained = False

    def load(self, model_dir=None):
        model_dir = model_dir or AppConfig.MODEL_DIR
        meta_path = os.path.join(model_dir, 'model_metadata.json')
        if not os.path.exists(meta_path):
            return False
        with open(meta_path, encoding='utf-8') as f:
            metadata = json.load(f)
        if metadata.get('task') != 'crop_suitability':
            return False
        model_path = os.path.join(model_dir, metadata.get('model_file', 'crop_model.pkl'))
        if not os.path.exists(model_path):
            return False

        self.model = joblib.load(model_path)
        self.metadata = metadata
        self.feature_names = metadata.get('features', self.feature_names)
        self.crop_names = metadata.get('crop_names', self.crop_names)
        self.is_trained = True
        try:
            self.shap_explainer = CropShapExplainer(self.model)
        except Exception:
            self.shap_explainer = None
        return True

    def predict(self, soil_data):
        if not self.is_trained:
            scores = all_crop_scores(soil_data)
            return {'scores': scores, 'source': 'rules',
                    'feature_importance': {}, 'shap_explanation': {}}

        X = pd.DataFrame([soil_data], columns=self.feature_names).values
        raw = np.clip(self.model.predict(X)[0], 0, 1)
        scores = {name: float(s) for name, s in zip(self.crop_names, raw)}
        best_idx = int(np.argmax(raw))

        feat_imp = dict(zip(self.feature_names,
                            [round(float(v), 4) for v in feature_importance(self.model)]))

        shap_exp = {}
        if self.shap_explainer is not None:
            try:
                vals = self.shap_explainer.explain(X, best_idx)
                shap_exp = dict(zip(self.feature_names, [round(float(v), 4) for v in vals]))
            except Exception as err:
                app.logger.warning(f"SHAP explanation failed: {err}")

        return {'scores': scores, 'source': 'model',
                'feature_importance': feat_imp, 'shap_explanation': shap_exp}


# ============================================================================
# CROP RECOMMENDATION ENGINE (same logic, cleaner code)
# ============================================================================

PARAM_UNITS = {
    'N': 'mg/kg', 'P': 'mg/kg', 'K': 'mg/kg', 'pH': '', 'EC': 'dS/m', 'OC': '%',
    'S': 'mg/kg', 'Zn': 'mg/kg', 'Fe': 'mg/kg', 'Cu': 'mg/kg', 'Mn': 'mg/kg', 'B': 'mg/kg',
}


def calculate_crop_suitability(soil_data, crop):
    """Weighted suitability (0-100) using the same scoring as the dataset generator."""
    overall = crop_suitability(soil_data, crop) * 100

    # Report on the parameters that matter most for this crop first
    matched, challenges = [], []
    for param in sorted(FEATURES, key=lambda p: crop['weights'][p], reverse=True):
        val, unit = soil_data[param], PARAM_UNITS[param]
        opt_lo, opt_hi = crop['optimal'][param]
        acc_lo, acc_hi = crop['acceptable'][param]
        if opt_lo <= val <= opt_hi:
            matched.append(f"{param} optimal ({val}{' ' + unit if unit else ''})")
        elif val < acc_lo or val > acc_hi:
            direction = 'too low' if val < acc_lo else 'too high'
            challenges.append(f"{param} {direction} ({val} vs {opt_lo}-{opt_hi})")
        else:
            direction = 'below' if val < opt_lo else 'above'
            challenges.append(f"{param} {direction} optimal ({val} vs {opt_lo}-{opt_hi})")

    priority = 'Excellent' if overall >= 85 else 'Good' if overall >= 70 else 'Fair'

    return {
        'name': crop['name'], 'category': crop['category'],
        'suitabilityScore': int(overall),
        'matchedParameters': matched[:3],
        'potentialChallenges': challenges[:2],
        'priority': priority,
        'plantingSeasons': crop.get('seasons', []),
        'harvestMonths': crop.get('harvest_months', 0),
    }


def recommend_crops(soil_data, top_n=10):
    recs = [calculate_crop_suitability(soil_data, c) for c in CROP_DATABASE]
    recs.sort(key=lambda x: x['suitabilityScore'], reverse=True)
    return recs[:top_n]


# ============================================================================
# SOIL HEALTH SCORE — NEW weighted index
# ============================================================================

# Reference "good fertility" ranges for West African upland soils, in the same
# units as the model features. Upper bounds for EC mark salinity risk.
SOIL_OPTIMAL_RANGES = {
    'N': (100, 250), 'P': (8, 25), 'K': (150, 500), 'pH': (5.8, 7.0),
    'EC': (0.0, 0.8), 'OC': (0.8, 2.0), 'S': (8, 25),
    'Zn': (0.5, 2.0), 'Fe': (0.8, 4.0), 'Cu': (0.3, 1.5),
    'Mn': (2, 8), 'B': (0.2, 1.0),
}

SOIL_HEALTH_WEIGHTS = {
    'N': 0.16, 'P': 0.18, 'K': 0.12, 'pH': 0.15,
    'OC': 0.15, 'EC': 0.05, 'S': 0.05,
    'Zn': 0.05, 'Fe': 0.02, 'Cu': 0.02, 'Mn': 0.02, 'B': 0.03,
}


def calculate_soil_health_score(soil_data):
    total, breakdown = 0, {}
    for nutrient, weight in SOIL_HEALTH_WEIGHTS.items():
        val = soil_data.get(nutrient, 0)
        lo, hi = SOIL_OPTIMAL_RANGES[nutrient]
        if lo <= val <= hi:
            score = 100
        elif nutrient == 'pH':
            # Penalise distance from the range in pH units, both directions
            score = max(0, 100 - min(abs(val - lo), abs(val - hi)) * 40)
        elif val < lo:
            # Quadratic: a nutrient at half its target is a serious limitation
            score = max(0, (val / lo) ** 2 * 100)
        else:
            score = max(0, 100 - ((val - hi) / hi) * 50)
        total += score * weight
        breakdown[nutrient] = {'score': round(score, 1), 'weight': weight, 'value': val}

    # Calibrated on soil-type means: Lithosol D, Lixisol C, Luvisol B, Vertisol/Bas-fond A
    grade = 'A' if total >= 90 else 'B' if total >= 75 else 'C' if total >= 60 else 'D'
    return {'overall_score': round(total, 1), 'grade': grade, 'breakdown': breakdown}


# ============================================================================
# FERTILIZER & SOIL MANAGEMENT ADVICE — products available in Burkina Faso
# ============================================================================

def fertilizer_recommendations(soil_data):
    recs = []
    low = {p: soil_data[p] < SOIL_OPTIMAL_RANGES[p][0] for p in ('N', 'P', 'K', 'OC', 'Zn', 'B')}
    if low['OC']:
        recs.append("Add organic matter: compost or manure (2.5-5 t/ha), zaï pits or "
                    "demi-lunes, and keep crop residues on the field")
    if low['P']:
        recs.append("Correct phosphorus: Burkina Phosphate (BP, 200-400 kg/ha as a "
                    "basal dressing) or NPK 14-23-14 at sowing")
    if low['N']:
        recs.append("Apply nitrogen: urea (46-0-0) split in two top-dressings, or "
                    "rotate with niébé/arachide to fix nitrogen")
    if low['K']:
        recs.append("Apply potassium: NPK 15-15-15 or KCl (0-0-60), especially for cotton and maize")
    if soil_data['pH'] < 5.5:
        recs.append("Raise pH: dolomite or Burkina Phosphate plus organic matter to limit aluminium toxicity")
    elif soil_data['pH'] > 7.5:
        recs.append("High pH: use ammonium sulfate as N source and add organic matter; watch Zn and Fe availability")
    if soil_data['EC'] > 0.8:
        recs.append("Salinity risk: improve drainage and avoid KCl and other chloride fertilizers")
    if low['Zn']:
        recs.append("Zinc deficiency: zinc sulfate (5-10 kg/ha) or Zn-enriched NPK, critical for maize and rice")
    if low['B']:
        recs.append("Boron deficiency: borax (5-10 kg/ha), especially for cotton and groundnut")
    if recs:
        recs.append("Use micro-dosing (a few grams of fertilizer per planting hole) "
                    "to get the best return on small fertilizer budgets")
    else:
        recs = ["Maintain current management with organic inputs",
                "Rotate cereals with legumes (niébé, arachide, soja)",
                "Re-test the soil every 2-3 seasons"]
    return recs


# ============================================================================
# FULL ANALYSIS ENGINE
# ============================================================================

def priority_for(score):
    return 'Excellent' if score >= 85 else 'Good' if score >= 70 else 'Fair'


def analyze_soil(soil_data, model):
    prediction = model.predict(soil_data)
    scores = dict(sorted(prediction['scores'].items(), key=lambda x: x[1], reverse=True))
    best_crop, best_score = next(iter(scores.items()))
    best_pct = int(round(best_score * 100))

    # Key factors: SHAP for the recommended crop when available, else importance
    shap_exp = prediction['shap_explanation']
    if shap_exp:
        top = sorted(shap_exp.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
        key_factors = [f"{f} ({soil_data[f]}) {'raises' if v > 0 else 'lowers'} "
                       f"{best_crop} suitability by {abs(v) * 100:.1f} points" for f, v in top]
    else:
        top = sorted(prediction['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:4]
        key_factors = [f"{f} level ({soil_data[f]}) — high influence" for f, _ in top]

    # Strengths & deficiencies
    strengths, deficiencies = [], []
    for p in ['N', 'P', 'K', 'pH', 'OC', 'Zn', 'B']:
        lo, hi = SOIL_OPTIMAL_RANGES[p]
        v, unit = soil_data[p], PARAM_UNITS[p]
        if lo <= v <= hi:
            strengths.append(f"{p} optimal ({v}{' ' + unit if unit else ''})")
        elif v < lo:
            deficiencies.append(f"{p} below optimal ({v}{' ' + unit if unit else ''}, target: {lo}-{hi})")

    recs = fertilizer_recommendations(soil_data)

    # Crop cards: rule-based explanations, ranked and scored by the model
    crops = []
    for crop in CROP_DATABASE:
        card = calculate_crop_suitability(soil_data, crop)
        card['suitabilityScore'] = int(round(scores[crop['name']] * 100))
        card['priority'] = priority_for(card['suitabilityScore'])
        crops.append(card)
    crops.sort(key=lambda c: c['suitabilityScore'], reverse=True)

    alternatives = [f"{name} ({int(round(s * 100))}/100)" for name, s in list(scores.items())[1:3]]
    summary = (f"{best_crop} is the best match for this soil (suitability {best_pct}/100). "
               f"Also suitable: {', '.join(alternatives)}. "
               f"Priority action: {recs[0]}")

    return {
        'analysis_id': new_analysis_id(soil_data),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'suitability': best_crop,
        'recommendedCrop': best_crop,
        'isModelCropRecommendation': True,
        'scoreSource': prediction['source'],
        'confidence': f"{best_pct}%",
        'confidenceScore': best_pct,
        'cropScores': {k: round(v, 4) for k, v in scores.items()},
        # Kept for older frontends that read 'probabilities'
        'probabilities': {k: round(v, 4) for k, v in scores.items()},
        'keyFactors': key_factors,
        'deficiencies': deficiencies,
        'strengths': strengths,
        'recommendations': recs,
        'summary': summary,
        'recommendedCrops': crops,
        'shap_explanation': shap_exp,
        'soil_health_score': calculate_soil_health_score(soil_data),
    }


# ============================================================================
# INPUT VALIDATION
# ============================================================================

FIELD_RANGES = {
    'N':  (0, 400),  'P':  (0, 60),   'K':  (0, 1000),
    'pH': (0, 14),   'EC': (0, 2),     'OC': (0, 5),
    'S':  (0, 50),   'Zn': (0, 2),     'Fe': (0, 5),
    'Cu': (0, 5),    'Mn': (0, 20),    'B':  (0, 5),
}

def validate_soil_input(data):
    if not data:
        raise ValidationError("Request body is empty or not valid JSON")

    errors, clean = {}, {}
    for field, (lo, hi) in FIELD_RANGES.items():
        if field not in data:
            errors[field] = f"Missing required field: {field}"
            continue
        try:
            val = float(data[field])
        except (ValueError, TypeError):
            errors[field] = f"Invalid number for {field}: {data[field]}"
            continue
        if val < lo or val > hi:
            errors[field] = f"{field} must be {lo}-{hi}, got {val}"
            continue
        clean[field] = val

    if errors:
        raise ValidationError("Input validation failed", details=errors)
    return clean


# ============================================================================
# CREATE FLASK APP
# ============================================================================

app = Flask(__name__)
# Render (and most PaaS) sit behind one reverse proxy: trust its X-Forwarded-For
# so rate limiting sees the real client IP instead of the proxy's.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
# Keep dict order in responses: crop scores are sent ranked best-first
app.json.sort_keys = False
app.config['SECRET_KEY'] = AppConfig.SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = AppConfig.DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app, origins=AppConfig.CORS_ORIGINS.split(','))
setup_logging(app)

db = SQLAlchemy(app)


class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(12), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    soil_data = db.Column(db.Text, nullable=False)
    result = db.Column(db.Text, nullable=False)
    suitability = db.Column(db.String(20))
    health_score = db.Column(db.Float)


with app.app_context():
    db.create_all()

rate_limiter = RateLimiter(AppConfig.RATE_LIMIT)
request_counter = {'total': 0, 'predictions': 0}
start_time = time.time()

# Load model
soil_model = CropSuitabilityModel()
if soil_model.load():
    app.logger.info("✓ Model loaded successfully")
    if soil_model.metadata:
        app.logger.info(f"  Type: {soil_model.metadata.get('model_type', 'N/A')}")
        app.logger.info(f"  Test MAE: {soil_model.metadata.get('test_mae', 'N/A')} points")
else:
    app.logger.warning("✗ No model found — using rule-based crop scores. Run: python train_model.py")


# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.before_request
def enforce_api_key():
    if not AppConfig.API_KEY_REQUIRED:
        return
    if request.path == '/api/health':
        return
    key = request.headers.get('X-API-Key', '')
    if not key or key != AppConfig.API_KEY:
        return jsonify({'error': 'Unauthorized', 'message': 'Missing or invalid X-API-Key header'}), 401

@app.before_request
def enforce_rate_limit():
    g.start_time = time.time()
    request_counter['total'] += 1
    if not rate_limiter.is_allowed(request.remote_addr):
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': f'Max {AppConfig.RATE_LIMIT} requests/minute'
        }), 429

@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        ms = round((time.time() - g.start_time) * 1000, 2)
        response.headers['X-Response-Time'] = f"{ms}ms"
    return response


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(APIError)
def handle_api_error(e):
    app.logger.error(f"[{e.status_code}] {e.message}")
    return jsonify({'error': e.message, 'details': e.details}), e.status_code

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f"Internal error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# API ENDPOINTS
# ============================================================================

# --- 1. Health Check (improved) ---
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': soil_model.is_trained,
        'model_type': soil_model.metadata.get('model_type', 'Rule-based'),
        'uptime_seconds': int(time.time() - start_time),
        'total_requests': request_counter['total'],
        'total_predictions': request_counter['predictions'],
        'timestamp': datetime.now().isoformat(),
    })


# --- 2. Prediction (improved with SHAP + health score + caching) ---
@app.route('/api/predict', methods=['POST'])
def predict_suitability():
    try:
        soil_data = validate_soil_input(request.json)

        # Check cache — reuse the analysis but give it a fresh id/timestamp so
        # every request is its own history entry
        key = make_cache_key(soil_data)
        cached = prediction_cache.get(key) if prediction_cache is not None else None
        if cached is not None:
            result = dict(cached, cached=True)
        else:
            result = analyze_soil(soil_data, soil_model)
            request_counter['predictions'] += 1
            if prediction_cache is not None:
                prediction_cache[key] = dict(result)
        result['analysis_id'] = new_analysis_id(soil_data)
        result['timestamp'] = datetime.now(timezone.utc).isoformat()

        try:
            record = PredictionHistory(
                analysis_id=result['analysis_id'],
                soil_data=json.dumps(soil_data),
                result=json.dumps(result),
                suitability=result['suitability'],
                health_score=result.get('soil_health_score', {}).get('overall_score'),
            )
            db.session.add(record)
            db.session.commit()
        except Exception as db_err:
            app.logger.warning(f"DB save failed (non-fatal): {db_err}")
            db.session.rollback()

        app.logger.info(
            f"Prediction: {result['suitability']} ({result['confidence']}) "
            f"N={soil_data['N']} P={soil_data['P']} K={soil_data['K']}"
        )
        return jsonify(result), 200

    except (ValidationError, ModelError):
        raise
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}", exc_info=True)
        raise APIError(f"Analysis failed: {str(e)}")


# --- 3. Crops Database (unchanged) ---
@app.route('/api/crops', methods=['GET'])
def get_crops():
    return jsonify(CROP_DATABASE), 200


# --- 4. Model Info (improved with metadata) ---
@app.route('/api/model/info', methods=['GET'])
def model_info():
    info = {
        'model_type': soil_model.metadata.get('model_type', 'Rule-based'),
        'task': 'crop_suitability',
        'features': soil_model.feature_names,
        'crops': soil_model.crop_names,
        'is_trained': soil_model.is_trained,
        'has_shap': soil_model.shap_explainer is not None,
    }
    for key in ['test_mae', 'test_r2', 'top1_agreement', 'top3_hit_rate', 'spearman', 'mean_regret',
                'cv_mae_mean', 'cv_mae_std', 'n_samples', 'trained_at', 'best_params']:
        if key in soil_model.metadata:
            info[key] = soil_model.metadata[key]
    return jsonify(info), 200


# --- 5. Compare Samples (NEW) ---
@app.route('/api/compare', methods=['POST'])
def compare_samples():
    try:
        data = request.json
        if not data or 'sample_a' not in data or 'sample_b' not in data:
            raise ValidationError("Provide both 'sample_a' and 'sample_b'")

        a = validate_soil_input(data['sample_a'])
        b = validate_soil_input(data['sample_b'])
        res_a = analyze_soil(a, soil_model)
        res_b = analyze_soil(b, soil_model)

        diffs = {}
        for field in FIELD_RANGES:
            va, vb = a[field], b[field]
            diffs[field] = {
                'sample_a': va, 'sample_b': vb,
                'change': round(vb - va, 4),
                'change_pct': round((vb - va) / va * 100, 1) if va != 0 else 0
            }

        ha = calculate_soil_health_score(a)
        hb = calculate_soil_health_score(b)

        return jsonify({
            'sample_a': res_a, 'sample_b': res_b,
            'differences': diffs,
            'health_comparison': {
                'sample_a_score': ha['overall_score'],
                'sample_b_score': hb['overall_score'],
                'improvement': round(hb['overall_score'] - ha['overall_score'], 1)
            }
        }), 200
    except ValidationError:
        raise
    except Exception as e:
        raise APIError(f"Comparison failed: {str(e)}")


# --- 6. Seasonal Calendar (NEW) ---
@app.route('/api/seasonal-calendar', methods=['GET'])
def seasonal_calendar():
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    calendar = []
    for crop in CROP_DATABASE:
        planting = set(crop.get('seasons', []))
        schedule = {m: ('planting' if m in planting else 'inactive') for m in months}
        calendar.append({
            'name': crop['name'], 'category': crop['category'], 'icon': crop['icon'],
            'planting_months': crop.get('seasons', []),
            'harvest_months': crop.get('harvest_months', 0),
            'schedule': schedule,
        })
    return jsonify(calendar), 200


# --- 7. Soil Health Score (NEW) ---
@app.route('/api/soil-health-score', methods=['POST'])
def soil_health_score_endpoint():
    try:
        soil_data = validate_soil_input(request.json)
        return jsonify(calculate_soil_health_score(soil_data)), 200
    except ValidationError:
        raise
    except Exception as e:
        raise APIError(f"Health score failed: {str(e)}")


# --- 8. Analysis History ---
@app.route('/api/history', methods=['GET'])
def get_history():
    limit = min(int(request.args.get('limit', 20)), 100)
    rows = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).limit(limit).all()
    return jsonify([{
        'id': r.id,
        'analysis_id': r.analysis_id,
        'timestamp': r.timestamp.isoformat(),
        'suitability': r.suitability,
        'health_score': r.health_score,
        'soil_data': json.loads(r.soil_data),
        'result': json.loads(r.result),
    } for r in rows]), 200


@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def delete_history(record_id):
    row = db.get_or_404(PredictionHistory, record_id)
    db.session.delete(row)
    db.session.commit()
    return jsonify({'deleted': record_id}), 200


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🌱 AgriTech Soil Analyzer — Production API")
    print("=" * 60)
    print(f"  Environment: {AppConfig.FLASK_ENV}")
    print(f"  Model:       {'✓ Loaded' if soil_model.is_trained else '✗ NOT LOADED'}")
    if soil_model.metadata:
        print(f"  Model Type:  {soil_model.metadata.get('model_type', '?')}")
        print(f"  Test MAE:    {soil_model.metadata.get('test_mae', '?')} points")
    print(f"  SHAP:        {'✓' if soil_model.shap_explainer else '✗'}")
    print(f"  Rate Limit:  {AppConfig.RATE_LIMIT}/min")
    print(f"  Crops:       {len(CROP_DATABASE)}")
    print("─" * 60)
    print("  GET  /api/health")
    print("  POST /api/predict            (enhanced)")
    print("  GET  /api/crops")
    print("  GET  /api/model/info         (enhanced)")
    print("  POST /api/compare            ← NEW")
    print("  GET  /api/seasonal-calendar  ← NEW")
    print("  POST /api/soil-health-score  ← NEW")
    print("=" * 60)

    app.run(host=AppConfig.FLASK_HOST, port=AppConfig.FLASK_PORT,
            debug=(AppConfig.FLASK_ENV == 'development'))