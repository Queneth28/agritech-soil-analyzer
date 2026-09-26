import os

# Must be set before importing app: the SQLAlchemy engine is bound at import time
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ.setdefault('RATE_LIMIT_PER_MINUTE', '1000')

import pytest
from app import app, db, prediction_cache, rate_limiter


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.app_context():
        db.drop_all()
        db.create_all()
    if prediction_cache is not None:
        prediction_cache.clear()
    rate_limiter.requests.clear()
    with app.test_client() as client:
        yield client


@pytest.fixture
def good_soil():
    return {
        "N": 200, "P": 8.5, "K": 550, "pH": 6.8, "EC": 0.55, "OC": 1.15,
        "S": 15.5, "Zn": 0.30, "Fe": 0.65, "Cu": 1.25, "Mn": 5.50, "B": 1.85
    }


@pytest.fixture
def poor_soil():
    return {
        "N": 120, "P": 6.0, "K": 350, "pH": 7.8, "EC": 0.70, "OC": 0.60,
        "S": 8.0, "Zn": 0.18, "Fe": 0.40, "Cu": 0.90, "Mn": 3.00, "B": 0.40
    }
