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
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
import time
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from functools import wraps

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


# ============================================================================
# CROP DATABASE — now includes seasonal planting data
# ============================================================================

CROP_DATABASE = [
    # Cereals
    {'name': 'Wheat', 'category': 'Cereal', 'icon': '🌾',
     'N': (150, 250), 'P': (7, 12), 'K': (400, 600), 'pH': (6.0, 7.5), 'OC': (0.8, 2.0),
     'description': 'Staple grain crop, drought-resistant',
     'seasons': ['Oct', 'Nov', 'Dec'], 'harvest_months': 4},
    {'name': 'Rice', 'category': 'Cereal', 'icon': '🌾',
     'N': (180, 280), 'P': (8, 15), 'K': (450, 700), 'pH': (5.5, 7.0), 'OC': (1.0, 2.5),
     'description': 'High-yielding grain, requires water management',
     'seasons': ['Jun', 'Jul'], 'harvest_months': 4},
    {'name': 'Maize (Corn)', 'category': 'Cereal', 'icon': '🌽',
     'N': (200, 300), 'P': (9, 15), 'K': (500, 750), 'pH': (5.8, 7.5), 'OC': (1.0, 2.0),
     'description': 'Versatile crop, high nutrient demand',
     'seasons': ['Mar', 'Apr', 'May'], 'harvest_months': 3},
    {'name': 'Barley', 'category': 'Cereal', 'icon': '🌾',
     'N': (120, 200), 'P': (6, 10), 'K': (350, 550), 'pH': (6.5, 7.8), 'OC': (0.7, 1.5),
     'description': 'Hardy grain, tolerates alkaline soils',
     'seasons': ['Oct', 'Nov'], 'harvest_months': 4},

    # Legumes
    {'name': 'Soybeans', 'category': 'Legume', 'icon': '🫘',
     'N': (80, 150), 'P': (8, 12), 'K': (450, 650), 'pH': (6.0, 7.0), 'OC': (1.0, 2.0),
     'description': 'Nitrogen-fixing, protein-rich',
     'seasons': ['May', 'Jun'], 'harvest_months': 4},
    {'name': 'Chickpeas', 'category': 'Legume', 'icon': '🫘',
     'N': (60, 120), 'P': (7, 11), 'K': (350, 500), 'pH': (6.5, 8.0), 'OC': (0.8, 1.5),
     'description': 'Drought-tolerant, nitrogen-fixing',
     'seasons': ['Oct', 'Nov'], 'harvest_months': 5},
    {'name': 'Peas', 'category': 'Legume', 'icon': '🫛',
     'N': (70, 130), 'P': (8, 12), 'K': (400, 550), 'pH': (6.0, 7.5), 'OC': (1.0, 2.0),
     'description': 'Cool-season crop, soil improver',
     'seasons': ['Feb', 'Mar'], 'harvest_months': 3},

    # Vegetables
    {'name': 'Tomatoes', 'category': 'Vegetable', 'icon': '🍅',
     'N': (180, 250), 'P': (9, 14), 'K': (500, 700), 'pH': (6.0, 7.0), 'OC': (1.2, 2.5),
     'description': 'High-value crop, requires good drainage',
     'seasons': ['Mar', 'Apr', 'May'], 'harvest_months': 3},
    {'name': 'Potatoes', 'category': 'Vegetable', 'icon': '🥔',
     'N': (150, 220), 'P': (8, 13), 'K': (550, 750), 'pH': (5.0, 6.5), 'OC': (1.0, 2.5),
     'description': 'Tuber crop, prefers slightly acidic soil',
     'seasons': ['Feb', 'Mar', 'Apr'], 'harvest_months': 4},
    {'name': 'Onions', 'category': 'Vegetable', 'icon': '🧅',
     'N': (140, 200), 'P': (7, 11), 'K': (400, 600), 'pH': (6.0, 7.0), 'OC': (1.0, 2.0),
     'description': 'Shallow-rooted, requires consistent moisture',
     'seasons': ['Oct', 'Nov', 'Mar'], 'harvest_months': 4},
    {'name': 'Carrots', 'category': 'Vegetable', 'icon': '🥕',
     'N': (120, 180), 'P': (8, 12), 'K': (450, 650), 'pH': (6.0, 7.0), 'OC': (1.0, 2.0),
     'description': 'Root vegetable, needs loose soil',
     'seasons': ['Mar', 'Apr', 'Aug', 'Sep'], 'harvest_months': 3},
    {'name': 'Lettuce', 'category': 'Vegetable', 'icon': '🥬',
     'N': (130, 190), 'P': (7, 11), 'K': (400, 600), 'pH': (6.0, 7.0), 'OC': (1.2, 2.5),
     'description': 'Leafy green, short growing season',
     'seasons': ['Mar', 'Apr', 'Sep', 'Oct'], 'harvest_months': 2},

    # Fruits
    {'name': 'Strawberries', 'category': 'Fruit', 'icon': '🍓',
     'N': (100, 160), 'P': (8, 12), 'K': (450, 650), 'pH': (5.5, 6.5), 'OC': (1.5, 3.0),
     'description': 'Berry crop, prefers acidic soil',
     'seasons': ['Apr', 'May'], 'harvest_months': 2},
    {'name': 'Watermelon', 'category': 'Fruit', 'icon': '🍉',
     'N': (120, 180), 'P': (8, 12), 'K': (500, 700), 'pH': (6.0, 7.0), 'OC': (1.0, 2.0),
     'description': 'Large fruit, needs space and warmth',
     'seasons': ['May', 'Jun'], 'harvest_months': 3},

    # Cash Crops
    {'name': 'Cotton', 'category': 'Cash Crop', 'icon': '🌱',
     'N': (150, 220), 'P': (8, 13), 'K': (450, 650), 'pH': (6.0, 7.5), 'OC': (0.8, 1.5),
     'description': 'Fiber crop, drought-resistant',
     'seasons': ['Apr', 'May'], 'harvest_months': 5},
    {'name': 'Sunflower', 'category': 'Cash Crop', 'icon': '🌻',
     'N': (100, 170), 'P': (7, 11), 'K': (400, 600), 'pH': (6.0, 7.5), 'OC': (0.8, 1.8),
     'description': 'Oilseed crop, adaptable',
     'seasons': ['Apr', 'May', 'Jun'], 'harvest_months': 3},
    {'name': 'Sugarcane', 'category': 'Cash Crop', 'icon': '🎋',
     'N': (200, 300), 'P': (9, 15), 'K': (550, 800), 'pH': (6.0, 7.5), 'OC': (1.2, 2.5),
     'description': 'High nutrient demand, long-season crop',
     'seasons': ['Feb', 'Mar'], 'harvest_months': 12},

    # Herbs
    {'name': 'Mint', 'category': 'Herb', 'icon': '🌿',
     'N': (120, 180), 'P': (7, 10), 'K': (350, 500), 'pH': (6.0, 7.0), 'OC': (1.5, 2.5),
     'description': 'Aromatic herb, spreads easily',
     'seasons': ['Mar', 'Apr', 'May'], 'harvest_months': 2},
    {'name': 'Basil', 'category': 'Herb', 'icon': '🌿',
     'N': (130, 190), 'P': (7, 11), 'K': (400, 550), 'pH': (6.0, 7.5), 'OC': (1.2, 2.0),
     'description': 'Culinary herb, warm-season crop',
     'seasons': ['Apr', 'May', 'Jun'], 'harvest_months': 2},
]


# ============================================================================
# MODEL CLASS — Enhanced with SHAP + metadata
# ============================================================================

class SoilFertilityModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.shap_explainer = None
        self.metadata = {}
        self.feature_names = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']
        self.class_names = {0: 'Low', 1: 'Medium', 2: 'High'}
        self.is_trained = False

    def load(self, model_dir=None):
        model_dir = model_dir or AppConfig.MODEL_DIR
        model_path = os.path.join(model_dir, 'rf_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True

        # Load SHAP explainer (optional)
        shap_path = os.path.join(model_dir, 'shap_explainer.pkl')
        if os.path.exists(shap_path):
            try:
                self.shap_explainer = joblib.load(shap_path)
            except Exception:
                self.shap_explainer = None

        # Load metadata (optional)
        meta_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

        return True

    def predict(self, soil_data):
        if not self.is_trained:
            raise ModelError("Model not loaded. Train first: python train_model.py")

        df = pd.DataFrame([soil_data], columns=self.feature_names)
        X_scaled = self.scaler.transform(df)

        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Feature importance
        feat_imp = {}
        if hasattr(self.model, 'feature_importances_'):
            feat_imp = dict(zip(self.feature_names,
                                [round(float(v), 4) for v in self.model.feature_importances_]))

        # SHAP explanation
        shap_exp = {}
        if self.shap_explainer is not None:
            try:
                sv = self.shap_explainer.shap_values(X_scaled)
                if isinstance(sv, list):
                    vals = sv[int(prediction)][0]
                else:
                    vals = sv[0]
                shap_exp = dict(zip(self.feature_names,
                                    [round(float(v), 4) for v in vals]))
            except Exception:
                pass

        return {
            'class': int(prediction),
            'class_name': self.class_names[prediction],
            'probabilities': {self.class_names[i]: round(float(p), 4)
                              for i, p in enumerate(probabilities)},
            'confidence': round(float(max(probabilities)), 4),
            'feature_importance': feat_imp,
            'shap_explanation': shap_exp,
        }

    def train(self, X, y):
        """Fallback in-memory training."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def save(self, model_dir=None):
        model_dir = model_dir or AppConfig.MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'rf_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))


# ============================================================================
# CROP RECOMMENDATION ENGINE (same logic, cleaner code)
# ============================================================================

def calculate_crop_suitability(soil_data, crop):
    scores, matched, challenges = [], [], []

    for nutrient in ['N', 'P', 'K']:
        val = soil_data[nutrient]
        lo, hi = crop[nutrient]
        if lo <= val <= hi:
            scores.append(100)
            matched.append(f"{nutrient} optimal ({val} mg/kg)")
        elif val < lo:
            scores.append(max(0, 100 - ((lo - val) / lo) * 100))
            challenges.append(f"{nutrient} below optimal ({val} vs {lo}-{hi})")
        else:
            scores.append(max(0, 100 - ((val - hi) / hi) * 50))
            challenges.append(f"{nutrient} above optimal ({val} vs {lo}-{hi})")

    val = soil_data['pH']
    lo, hi = crop['pH']
    if lo <= val <= hi:
        scores.append(100)
        matched.append(f"pH optimal at {val}")
    else:
        dist = min(abs(val - lo), abs(val - hi))
        scores.append(max(0, 100 - dist * 30))
        challenges.append(f"pH {'too acidic' if val < lo else 'too alkaline'} ({val} vs {lo}-{hi})")

    val = soil_data['OC']
    lo, hi = crop['OC']
    if lo <= val <= hi:
        scores.append(100)
        matched.append(f"Organic carbon good ({val}%)")
    elif val < lo:
        scores.append(max(0, 100 - ((lo - val) / lo) * 80))
        challenges.append(f"OC low ({val}% vs {lo}-{hi}%)")
    else:
        scores.append(90)
        matched.append(f"Excellent organic matter ({val}%)")

    overall = np.mean(scores)
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

def calculate_soil_health_score(soil_data):
    weights = {
        'N': 0.18, 'P': 0.15, 'K': 0.15, 'pH': 0.15,
        'OC': 0.12, 'EC': 0.05, 'S': 0.05,
        'Zn': 0.04, 'Fe': 0.03, 'Cu': 0.03, 'Mn': 0.03, 'B': 0.02
    }
    optimal = {
        'N': (150, 300), 'P': (7, 10), 'K': (400, 700), 'pH': (6.5, 7.5),
        'EC': (0.4, 0.8), 'OC': (0.8, 2.0), 'S': (10, 30),
        'Zn': (0.2, 0.5), 'Fe': (0.3, 1.0), 'Cu': (0.5, 2.0),
        'Mn': (2, 10), 'B': (0.5, 3.0)
    }

    total, breakdown = 0, {}
    for nutrient, weight in weights.items():
        val = soil_data.get(nutrient, 0)
        lo, hi = optimal[nutrient]
        if lo <= val <= hi:
            score = 100
        elif val < lo:
            score = max(0, (val / lo) * 100)
        else:
            score = max(0, 100 - ((val - hi) / hi) * 50)
        total += score * weight
        breakdown[nutrient] = {'score': round(score, 1), 'weight': weight, 'value': val}

    grade = 'A' if total >= 85 else 'B' if total >= 70 else 'C' if total >= 55 else 'D'
    return {'overall_score': round(total, 1), 'grade': grade, 'breakdown': breakdown}


# ============================================================================
# FULL ANALYSIS ENGINE
# ============================================================================

def analyze_soil(soil_data, model):
    prediction = model.predict(soil_data)

    # Top influencing factors
    sorted_feats = sorted(prediction['feature_importance'].items(),
                          key=lambda x: x[1], reverse=True)
    key_factors = [f"{f} level ({soil_data[f]}) — high influence"
                   for f, _ in sorted_feats[:4]]

    # Strengths & deficiencies
    strengths, deficiencies, recs = [], [], []
    checks = {
        'N':  (150, 300, 'mg/kg'), 'P':  (7, 10, 'mg/kg'),
        'K':  (400, 700, 'mg/kg'), 'pH': (6.5, 7.5, ''),
        'OC': (0.8, 2.0, '%'),
    }
    for p, (lo, hi, unit) in checks.items():
        v = soil_data[p]
        if lo <= v <= hi:
            strengths.append(f"{p} optimal ({v} {unit})")
        elif v < lo:
            deficiencies.append(f"{p} below optimal ({v} {unit}, target: {lo}-{hi})")

    # Fertilizer recommendations
    if soil_data['N'] < 150:
        recs.append("Apply nitrogen fertilizer (urea or ammonium nitrate)")
    if soil_data['P'] < 7:
        recs.append("Add phosphate fertilizer (DAP or superphosphate)")
    if soil_data['K'] < 400:
        recs.append("Apply potassium fertilizer (MOP or potassium sulfate)")
    if soil_data['pH'] < 6.5:
        recs.append("Apply agricultural lime to raise pH")
    elif soil_data['pH'] > 7.5:
        recs.append("Add sulfur or acidifying fertilizers to lower pH")
    if soil_data['OC'] < 0.8:
        recs.append("Add compost, manure, or cover crops for organic matter")
    if not recs:
        recs = ["Maintain current management", "Monitor annually", "Consider crop rotation"]

    level = prediction['class_name']
    summaries = {
        'High': "Excellent conditions with well-balanced nutrients.",
        'Medium': "Good potential with some areas for improvement.",
        'Low': "Significant amendments needed to improve fertility.",
    }

    return {
        'analysis_id': hashlib.md5(
            f"{json.dumps(soil_data, sort_keys=True)}{time.time()}".encode()
        ).hexdigest()[:12],
        'timestamp': datetime.now().isoformat(),
        'suitability': level,
        'confidence': f"{int(prediction['confidence'] * 100)}%",
        'confidenceScore': int(prediction['confidence'] * 100),
        'probabilities': prediction['probabilities'],
        'keyFactors': key_factors,
        'deficiencies': deficiencies,
        'strengths': strengths,
        'recommendations': recs,
        'summary': f"Soil analysis indicates {level.lower()} suitability. {summaries.get(level, '')}",
        'recommendedCrops': recommend_crops(soil_data),
        'shap_explanation': prediction.get('shap_explanation', {}),
        'soil_health_score': calculate_soil_health_score(soil_data),
    }


# ============================================================================
# INPUT VALIDATION
# ============================================================================

FIELD_RANGES = {
    'N':  (0, 400),  'P':  (0, 15),   'K':  (0, 1000),
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
app.config['SECRET_KEY'] = AppConfig.SECRET_KEY
CORS(app, origins=AppConfig.CORS_ORIGINS.split(','))
setup_logging(app)

rate_limiter = RateLimiter(AppConfig.RATE_LIMIT)
request_counter = {'total': 0, 'predictions': 0}
start_time = time.time()

# Load model
soil_model = SoilFertilityModel()
if soil_model.load():
    app.logger.info("✓ Model loaded successfully")
    if soil_model.metadata:
        app.logger.info(f"  Type: {soil_model.metadata.get('model_type', 'N/A')}")
        app.logger.info(f"  Accuracy: {soil_model.metadata.get('test_accuracy', 'N/A')}")
else:
    app.logger.warning("✗ No model found. Run: python train_model.py")


# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.before_request
def before_request():
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
        'model_type': soil_model.metadata.get('model_type', 'Random Forest'),
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

        # Check cache
        if prediction_cache is not None:
            key = make_cache_key(soil_data)
            if key in prediction_cache:
                cached = prediction_cache[key].copy()
                cached['cached'] = True
                return jsonify(cached), 200

        result = analyze_soil(soil_data, soil_model)
        request_counter['predictions'] += 1

        if prediction_cache is not None:
            prediction_cache[make_cache_key(soil_data)] = result

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
        'model_type': soil_model.metadata.get('model_type', 'Random Forest Classifier'),
        'features': soil_model.feature_names,
        'classes': soil_model.class_names,
        'is_trained': soil_model.is_trained,
        'has_shap': soil_model.shap_explainer is not None,
    }
    for key in ['test_accuracy', 'f1_weighted', 'cv_mean', 'cv_std', 'trained_at', 'best_params']:
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
        print(f"  Accuracy:    {soil_model.metadata.get('test_accuracy', '?')}")
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