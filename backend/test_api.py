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
    from crop_profiles import CROPS
    assert data['suitability'] in {c['name'] for c in CROPS.values()}
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
    soil = {"N": 125, "P": 6.5, "K": 255, "pH": 6.5, "EC": 0.34, "OC": 0.58,
            "S": 12, "Zn": 0.62, "Fe": 2.1, "Cu": 0.72, "Mn": 4.2, "B": 0.32}
    sorgho = next(c for c in CROP_DATABASE if c['name'] == 'Sorgho')
    result = calculate_crop_suitability(soil, sorgho)
    assert 0 <= result['suitabilityScore'] <= 100
    assert result['priority'] in ('Excellent', 'Good', 'Fair')
    assert 'matchedParameters' in result
    assert 'potentialChallenges' in result


# ============================================================================
# AGRONOMIC CONSISTENCY — Burkina Faso context
# ============================================================================

LIXISOL = {"N": 90, "P": 4.5, "K": 175, "pH": 6.3, "EC": 0.28, "OC": 0.35,
           "S": 8, "Zn": 0.38, "Fe": 1.4, "Cu": 0.45, "Mn": 2.8, "B": 0.18}


def test_crop_database_matches_ml_classes():
    """Rule-based crops must be the same set the ML dataset is labelled with."""
    from app import CROP_DATABASE
    from crop_profiles import CROPS
    assert {c['name'] for c in CROP_DATABASE} == {c['name'] for c in CROPS.values()}


def test_rule_score_matches_dataset_scoring():
    from app import calculate_crop_suitability, CROP_DATABASE
    from crop_profiles import CROPS, crop_suitability
    mil_rules = next(c for c in CROP_DATABASE if c['name'] == 'Mil')
    mil_data = next(c for c in CROPS.values() if c['name'] == 'Mil')
    assert calculate_crop_suitability(LIXISOL, mil_rules)['suitabilityScore'] == \
        int(crop_suitability(LIXISOL, mil_data) * 100)


def test_poor_sandy_soil_favours_millet_and_sorghum():
    from app import recommend_crops
    top3 = [c['name'] for c in recommend_crops(LIXISOL)[:3]]
    assert 'Mil' in top3 or 'Sorgho' in top3


def test_no_liming_advice_at_neutral_ph():
    from app import fertilizer_recommendations
    recs = ' '.join(fertilizer_recommendations(LIXISOL)).lower()
    assert 'raise ph' not in recs
    assert 'organic matter' in recs      # OC 0.35 % is low
    assert 'phosph' in recs              # P 4.5 is low


def test_liming_advice_on_acid_soil():
    from app import fertilizer_recommendations
    recs = ' '.join(fertilizer_recommendations({**LIXISOL, 'pH': 5.0})).lower()
    assert 'raise ph' in recs


def test_health_score_penalises_acid_ph_by_distance():
    from app import calculate_soil_health_score
    ok = calculate_soil_health_score({**LIXISOL, 'pH': 6.3})['breakdown']['pH']['score']
    acid = calculate_soil_health_score({**LIXISOL, 'pH': 4.8})['breakdown']['pH']['score']
    assert ok == 100
    assert acid <= 60


def test_cached_prediction_gets_new_history_entry(client, good_soil):
    a = client.post('/api/predict', json=good_soil).get_json()
    b = client.post('/api/predict', json=good_soil).get_json()
    assert a['analysis_id'] != b['analysis_id']
    assert len(client.get('/api/history').get_json()) == 2


def test_health_grade_orders_burkina_soil_types():
    from app import calculate_soil_health_score
    from generate_burkina_dataset import SOIL_TYPES
    grade = {name: calculate_soil_health_score({p: d['mean'] for p, d in st['params'].items()})['grade']
             for name, st in SOIL_TYPES.items()}
    assert grade['Lithosol'] == 'D'
    assert grade['Lixisol'] == 'C'
    assert grade['Luvisol'] == 'B'
    assert grade['Bas-fond'] == 'A'


# ============================================================================
# CROP SUITABILITY MODEL
# ============================================================================

def test_model_is_loaded_crop_suitability(client):
    info = client.get('/api/model/info').get_json()
    assert info['is_trained'] is True
    assert info['task'] == 'crop_suitability'
    assert len(info['crops']) == 9
    assert 'test_mae' in info


def test_prediction_scores_every_crop_and_ranks_them(client):
    data = client.post('/api/predict', json=LIXISOL).get_json()
    scores = data['cropScores']
    assert len(scores) == 9
    assert all(0 <= s <= 1 for s in scores.values())
    assert list(scores.values()) == sorted(scores.values(), reverse=True)
    assert data['recommendedCrop'] == next(iter(scores))
    assert data['scoreSource'] == 'model'
    cards = [c['name'] for c in data['recommendedCrops']]
    assert cards[0] == data['recommendedCrop']
    assert len(cards) == 9


def test_model_agrees_with_agronomic_rules():
    """The model learns crop_profiles scoring; it must not drift far from it."""
    from app import soil_model
    from crop_profiles import all_crop_scores
    predicted = soil_model.predict(LIXISOL)['scores']
    for crop, rule_score in all_crop_scores(LIXISOL).items():
        assert abs(predicted[crop] - rule_score) < 0.08, crop


def test_degraded_soil_ranks_demanding_crops_last(client):
    lithosol = {"N": 62, "P": 2.8, "K": 128, "pH": 5.9, "EC": 0.22, "OC": 0.20,
                "S": 5, "Zn": 0.22, "Fe": 0.9, "Cu": 0.28, "Mn": 1.7, "B": 0.11}
    scores = client.post('/api/predict', json=lithosol).get_json()['cropScores']
    ranking = list(scores)
    assert ranking[0] == 'Mil'
    assert set(ranking[-3:]) & {'Maïs', 'Coton'}


def test_shap_explains_recommended_crop(client):
    data = client.post('/api/predict', json=LIXISOL).get_json()
    assert set(data['shap_explanation']) == set(LIXISOL)
    assert any(data['recommendedCrop'] in f for f in data['keyFactors'])


def test_falls_back_to_rules_without_model(tmp_path):
    from app import CropSuitabilityModel
    m = CropSuitabilityModel()
    assert m.load(str(tmp_path)) is False
    pred = m.predict(LIXISOL)
    assert pred['source'] == 'rules'
    assert len(pred['scores']) == 9


def test_excess_non_toxic_nutrient_is_not_crop_failure():
    from crop_profiles import param_score
    # P far above the acceptable max: reduced advantage, not zero
    assert param_score(40, 10, 15, 7, 15, 'P') == 0.5
    # Boron above the acceptable max can be toxic: zero
    assert param_score(3.0, 0.3, 1.2, 0.2, 1.5, 'B') == 0.0
