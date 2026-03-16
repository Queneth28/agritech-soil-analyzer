"""
Backend test suite — runs with pytest, no live server required.
Uses Flask test client + in-memory SQLite via conftest.py fixtures.
"""
import pytest


# ============================================================================
# INTEGRATION TESTS (HTTP via Flask test client)
# ============================================================================

def test_health_check(client):
    r = client.get('/api/health')
    data = r.get_json()
    assert r.status_code == 200
    assert data['status'] == 'healthy'
    assert 'uptime_seconds' in data


def test_model_info(client):
    r = client.get('/api/model/info')
    data = r.get_json()
    assert r.status_code == 200
    assert 'features' in data
    assert len(data['features']) == 12


def test_prediction_good_soil(client, good_soil):
    r = client.post('/api/predict', json=good_soil)
    data = r.get_json()
    assert r.status_code == 200
    assert data['suitability'] in ('High', 'Medium', 'Low')
    assert 'confidence' in data
    assert 'analysis_id' in data
    assert 'soil_health_score' in data
    assert 'recommendedCrops' in data
    assert len(data['recommendedCrops']) > 0


def test_prediction_saved_to_db(client, good_soil):
    client.post('/api/predict', json=good_soil)
    r = client.get('/api/history')
    assert r.status_code == 200
    history = r.get_json()
    assert len(history) == 1
    assert 'suitability' in history[0]
    assert 'soil_data' in history[0]


def test_shap_values_present(client, good_soil):
    r = client.post('/api/predict', json=good_soil)
    data = r.get_json()
    assert r.status_code == 200
    assert 'shap_explanation' in data


def test_crops_database(client):
    r = client.get('/api/crops')
    data = r.get_json()
    assert r.status_code == 200
    assert len(data) > 0
    assert any('seasons' in crop for crop in data)


def test_validate_missing_fields(client):
    r = client.post('/api/predict', json={"N": 200, "P": 8.5})
    data = r.get_json()
    assert r.status_code == 400
    assert 'details' in data


def test_validate_out_of_range(client, good_soil):
    bad = {**good_soil, "pH": 99, "N": -50}
    r = client.post('/api/predict', json=bad)
    assert r.status_code == 400


def test_compare_endpoint(client, good_soil, poor_soil):
    r = client.post('/api/compare', json={"sample_a": good_soil, "sample_b": poor_soil})
    data = r.get_json()
    assert r.status_code == 200
    assert 'sample_a' in data
    assert 'sample_b' in data
    assert 'differences' in data
    assert 'health_comparison' in data


def test_seasonal_calendar(client):
    r = client.get('/api/seasonal-calendar')
    data = r.get_json()
    assert r.status_code == 200
    assert len(data) > 0
    assert 'planting_months' in data[0]


def test_soil_health_score(client, good_soil):
    r = client.post('/api/soil-health-score', json=good_soil)
    data = r.get_json()
    assert r.status_code == 200
    assert 'overall_score' in data
    assert 'grade' in data
    assert data['grade'] in ('A', 'B', 'C', 'D')


def test_history_delete(client, good_soil):
    client.post('/api/predict', json=good_soil)
    history = client.get('/api/history').get_json()
    record_id = history[0]['id']
    r = client.delete(f'/api/history/{record_id}')
    assert r.status_code == 200
    assert len(client.get('/api/history').get_json()) == 0


def test_404_handling(client):
    r = client.get('/api/nonexistent')
    assert r.status_code == 404


def test_response_time_header(client, good_soil):
    r = client.post('/api/predict', json=good_soil)
    assert r.status_code == 200
    assert 'X-Response-Time' in r.headers


# ============================================================================
# UNIT TESTS (no HTTP, no fixtures needed)
# ============================================================================

def test_validate_soil_input_returns_clean_data():
    from app import validate_soil_input
    clean = validate_soil_input({
        "N": 200, "P": 8.5, "K": 550, "pH": 6.8, "EC": 0.55, "OC": 1.15,
        "S": 15.5, "Zn": 0.30, "Fe": 0.65, "Cu": 1.25, "Mn": 5.50, "B": 1.85
    })
    assert clean['N'] == 200.0
    assert isinstance(clean['pH'], float)
    assert len(clean) == 12


def test_validate_soil_input_raises_on_missing():
    from app import validate_soil_input, ValidationError
    with pytest.raises(ValidationError) as exc_info:
        validate_soil_input({"N": 200})
    assert exc_info.value.status_code == 400


def test_validate_soil_input_raises_on_out_of_range():
    from app import validate_soil_input, ValidationError
    with pytest.raises(ValidationError):
        validate_soil_input({
            "N": 200, "P": 8.5, "K": 550, "pH": 99,
            "EC": 0.55, "OC": 1.15, "S": 15.5,
            "Zn": 0.30, "Fe": 0.65, "Cu": 1.25, "Mn": 5.50, "B": 1.85
        })


def test_calculate_soil_health_score_range():
    from app import calculate_soil_health_score
    result = calculate_soil_health_score({
        "N": 200, "P": 8.5, "K": 550, "pH": 6.8, "EC": 0.55, "OC": 1.15,
        "S": 15.5, "Zn": 0.30, "Fe": 0.65, "Cu": 1.25, "Mn": 5.50, "B": 1.85
    })
    assert 0 <= result['overall_score'] <= 100
    assert result['grade'] in ('A', 'B', 'C', 'D')
    assert 'breakdown' in result


def test_calculate_soil_health_score_poor_soil():
    from app import calculate_soil_health_score
    result = calculate_soil_health_score({
        "N": 10, "P": 1, "K": 50, "pH": 4.0, "EC": 0.1, "OC": 0.1,
        "S": 1, "Zn": 0.01, "Fe": 0.01, "Cu": 0.01, "Mn": 0.1, "B": 0.01
    })
    assert result['grade'] in ('C', 'D')


def test_calculate_crop_suitability():
    from app import calculate_crop_suitability, CROP_DATABASE
    soil = {"N": 200, "P": 8.5, "K": 550, "pH": 6.8, "OC": 1.15}
    wheat = next(c for c in CROP_DATABASE if c['name'] == 'Wheat')
    result = calculate_crop_suitability(soil, wheat)
    assert 0 <= result['suitabilityScore'] <= 100
    assert result['priority'] in ('Excellent', 'Good', 'Fair')
    assert 'matchedParameters' in result
    assert 'potentialChallenges' in result
