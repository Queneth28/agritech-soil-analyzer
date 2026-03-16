import pytest
from app import app, db


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
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
