"""
Test Suite for AgriTech Soil Analyzer API
Tests all endpoints including the 3 new ones.

HOW TO RUN:
  1. Start the backend:  python app.py
  2. In another terminal: python test_api.py
"""

import requests
import json
import sys
import time

API_URL = "http://localhost:5000"

# Two different soil samples for testing
SAMPLE_GOOD = {
    "N": 200, "P": 8.5, "K": 550, "pH": 6.8,
    "EC": 0.55, "OC": 1.15, "S": 15.5,
    "Zn": 0.30, "Fe": 0.65, "Cu": 1.25, "Mn": 5.50, "B": 1.85
}

SAMPLE_POOR = {
    "N": 120, "P": 6.0, "K": 350, "pH": 7.8,
    "EC": 0.70, "OC": 0.60, "S": 8.0,
    "Zn": 0.18, "Fe": 0.40, "Cu": 0.90, "Mn": 3.00, "B": 0.40
}

passed, failed = 0, 0


def test(name, func):
    """Run a single test and track results."""
    global passed, failed
    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")
    try:
        if func():
            passed += 1
            print(f"  ✓ PASSED")
        else:
            failed += 1
            print(f"  ✗ FAILED")
    except Exception as e:
        failed += 1
        print(f"  ✗ EXCEPTION: {e}")


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_health():
    r = requests.get(f"{API_URL}/api/health")
    d = r.json()
    print(f"    status: {d['status']}")
    print(f"    model:  {d['model_loaded']}")
    print(f"    type:   {d.get('model_type', '?')}")
    print(f"    uptime: {d.get('uptime_seconds', 0)}s")
    return r.status_code == 200 and d['status'] == 'healthy'


def test_model_info():
    r = requests.get(f"{API_URL}/api/model/info")
    d = r.json()
    print(f"    type:     {d.get('model_type', '?')}")
    print(f"    features: {len(d['features'])}")
    print(f"    has_shap: {d.get('has_shap', False)}")
    if 'test_accuracy' in d:
        print(f"    accuracy: {d['test_accuracy']}")
    return r.status_code == 200 and 'features' in d


def test_prediction():
    r = requests.post(f"{API_URL}/api/predict", json=SAMPLE_GOOD)
    d = r.json()
    print(f"    suitability:  {d['suitability']}")
    print(f"    confidence:   {d['confidence']}")
    print(f"    analysis_id:  {d.get('analysis_id', '?')}")
    print(f"    health score: {d.get('soil_health_score', {}).get('overall_score', '?')}")
    print(f"    health grade: {d.get('soil_health_score', {}).get('grade', '?')}")
    print(f"    SHAP values:  {len(d.get('shap_explanation', {})) > 0}")
    print(f"    crops:        {len(d.get('recommendedCrops', []))}")
    if d.get('recommendedCrops'):
        top = d['recommendedCrops'][0]
        print(f"    top crop:     {top['name']} ({top['suitabilityScore']}%)")
    return (r.status_code == 200 and 'suitability' in d
            and 'soil_health_score' in d and 'recommendedCrops' in d)


def test_shap_values():
    r = requests.post(f"{API_URL}/api/predict", json=SAMPLE_GOOD)
    d = r.json()
    shap = d.get('shap_explanation', {})
    if shap:
        print(f"    SHAP features: {len(shap)}")
        top3 = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for feat, val in top3:
            print(f"      {feat}: {val:+.4f} {'↑' if val > 0 else '↓'}")
    else:
        print(f"    No SHAP values (optional — still passes)")
    return r.status_code == 200


def test_crops_database():
    r = requests.get(f"{API_URL}/api/crops")
    d = r.json()
    print(f"    total crops:   {len(d)}")
    has_seasons = sum(1 for c in d if c.get('seasons'))
    print(f"    with seasons:  {has_seasons}")
    return r.status_code == 200 and len(d) > 0


def test_missing_fields():
    r = requests.post(f"{API_URL}/api/predict", json={"N": 200, "P": 8.5})
    d = r.json()
    print(f"    status:  {r.status_code}")
    print(f"    error:   {d.get('error', '?')}")
    if 'details' in d:
        print(f"    missing: {len(d['details'])} fields")
    return r.status_code == 400


def test_out_of_range():
    bad = {**SAMPLE_GOOD, "pH": 99, "N": -50}
    r = requests.post(f"{API_URL}/api/predict", json=bad)
    print(f"    status: {r.status_code}")
    return r.status_code == 400


def test_compare():
    """Test NEW /api/compare endpoint."""
    payload = {"sample_a": SAMPLE_GOOD, "sample_b": SAMPLE_POOR}
    r = requests.post(f"{API_URL}/api/compare", json=payload)
    d = r.json()
    print(f"    status:      {r.status_code}")
    if r.status_code == 200:
        print(f"    sample A:    {d['sample_a']['suitability']}")
        print(f"    sample B:    {d['sample_b']['suitability']}")
        print(f"    health A:    {d['health_comparison']['sample_a_score']}")
        print(f"    health B:    {d['health_comparison']['sample_b_score']}")
        print(f"    improvement: {d['health_comparison']['improvement']}")
        return 'differences' in d
    return False


def test_seasonal_calendar():
    """Test NEW /api/seasonal-calendar endpoint."""
    r = requests.get(f"{API_URL}/api/seasonal-calendar")
    d = r.json()
    print(f"    status: {r.status_code}")
    print(f"    crops:  {len(d)}")
    if d:
        c = d[0]
        print(f"    sample: {c['name']} → plant: {c['planting_months']}")
    return r.status_code == 200 and len(d) > 0


def test_soil_health_score():
    """Test NEW /api/soil-health-score endpoint."""
    r = requests.post(f"{API_URL}/api/soil-health-score", json=SAMPLE_GOOD)
    d = r.json()
    print(f"    status: {r.status_code}")
    print(f"    score:  {d.get('overall_score', '?')}")
    print(f"    grade:  {d.get('grade', '?')}")
    return r.status_code == 200 and 'overall_score' in d and 'grade' in d


def test_404():
    r = requests.get(f"{API_URL}/api/nonexistent")
    print(f"    status: {r.status_code}")
    return r.status_code == 404


def test_response_time():
    t0 = time.time()
    r = requests.post(f"{API_URL}/api/predict", json=SAMPLE_GOOD)
    ms = (time.time() - t0) * 1000
    print(f"    time:   {ms:.0f}ms")
    print(f"    header: {r.headers.get('X-Response-Time', '?')}")
    return r.status_code == 200 and ms < 2000


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def main():
    print("=" * 55)
    print("  AGRITECH SOIL ANALYZER — API TEST SUITE")
    print(f"  Target: {API_URL}")
    print("=" * 55)

    tests = [
        ("1. Health Check",              test_health),
        ("2. Model Info",                test_model_info),
        ("3. Soil Prediction",           test_prediction),
        ("4. SHAP Explanation",          test_shap_values),
        ("5. Crops Database",            test_crops_database),
        ("6. Invalid: Missing Fields",   test_missing_fields),
        ("7. Invalid: Out of Range",     test_out_of_range),
        ("8. Compare Samples (NEW)",     test_compare),
        ("9. Seasonal Calendar (NEW)",   test_seasonal_calendar),
        ("10. Soil Health Score (NEW)",  test_soil_health_score),
        ("11. 404 Handling",             test_404),
        ("12. Response Time (<2s)",      test_response_time),
    ]

    for name, func in tests:
        test(name, func)

    total = passed + failed
    print(f"\n{'=' * 55}")
    print(f"  RESULTS: {passed}/{total} passed ({passed/total*100:.0f}%)")
    if failed == 0:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print(f"  ⚠️  {failed} test(s) failed")
    print("=" * 55)

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()