"""
Model Training — Crop Suitability (multi-output regression)
AgriTech Soil Analyzer

Predicts a suitability score (0-1) for each of the 9 Burkina Faso crops
from the 12 soil test values. The app ranks crops by predicted score.

Pipeline:
  1. Load soil_data.csv (generate with: python generate_burkina_dataset.py)
  2. Train Random Forest (native multi-output) and LightGBM (one model per crop)
     with a randomized hyperparameter search
  3. Evaluate on a held-out test set:
       - MAE in score points (0-100), overall and per crop
       - R²
       - Top-1 agreement: predicted best crop == true best crop
       - Top-3 hit rate: true best crop is in the predicted top 3
       - Spearman rank correlation of the full 9-crop ranking
       - Regret: score points lost by planting the predicted #1 crop
  4. Pick the model with the lowest MAE, cross-validate it, save it with
     metadata and plots

HOW TO RUN:
  python generate_burkina_dataset.py
  python train_model.py
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, spearmanr, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor

from crop_profiles import CROPS, FEATURES
from model_utils import feature_importance

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠ LightGBM not installed — only Random Forest will be trained.")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    TUNING_ITERATIONS = 20
    TUNING_CV = 3

    FEATURES = FEATURES
    CROP_NAMES = [c['name'] for c in CROPS.values()]
    TARGETS = [f'score_{name}' for name in CROP_NAMES]

    MODEL_FILE = 'crop_model.pkl'

    # Kept deliberately small: the model is committed to git and loaded on a
    # small Render instance, so a forest of fully grown trees is not an option.
    RF_PARAM_DIST = {
        'n_estimators': randint(80, 200),
        'max_depth': [12, 16, 20],
        'min_samples_leaf': randint(3, 10),
        'max_features': [0.5, 0.7, 1.0],
    }

    LGB_PARAM_DIST = {
        'estimator__n_estimators': randint(200, 700),
        'estimator__learning_rate': uniform(0.02, 0.08),
        'estimator__num_leaves': randint(15, 63),
        'estimator__min_child_samples': randint(10, 40),
        'estimator__subsample': uniform(0.7, 0.3),
        'estimator__colsample_bytree': uniform(0.7, 0.3),
        'estimator__reg_lambda': uniform(0.0, 1.0),
    }


# ============================================================================
# DATA
# ============================================================================

def load_data(filepath='soil_data.csv'):
    print(f"\n📂 Loading {filepath}...")
    if not os.path.exists(filepath):
        raise SystemExit(f"   ✗ {filepath} not found. Run: python generate_burkina_dataset.py")
    df = pd.read_csv(filepath)
    missing = [c for c in Config.FEATURES + Config.TARGETS if c not in df.columns]
    if missing:
        raise SystemExit(f"   ✗ Missing columns {missing}. Regenerate with generate_burkina_dataset.py")
    print(f"   ✓ {len(df)} samples, {len(Config.FEATURES)} features, {len(Config.TARGETS)} crop scores")
    return df


# ============================================================================
# TRAINING
# ============================================================================

def tune(name, estimator, param_dist, X, y):
    print(f"\n🌲 Training {name} ({Config.TUNING_ITERATIONS} parameter combinations)...")
    search = RandomizedSearchCV(
        estimator, param_dist, n_iter=Config.TUNING_ITERATIONS, cv=Config.TUNING_CV,
        scoring='neg_mean_absolute_error', random_state=Config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    print(f"   ✓ CV MAE: {-search.best_score_ * 100:.2f} points")
    return search.best_estimator_, search.best_params_


def train_random_forest(X, y):
    rf = RandomForestRegressor(random_state=Config.RANDOM_STATE, n_jobs=-1)
    return tune('Random Forest', rf, Config.RF_PARAM_DIST, X, y)


def train_lightgbm(X, y):
    base = lgb.LGBMRegressor(random_state=Config.RANDOM_STATE, verbose=-1, n_jobs=1)
    return tune('LightGBM', MultiOutputRegressor(base), Config.LGB_PARAM_DIST, X, y)


# ============================================================================
# EVALUATION
# ============================================================================

def ranking_metrics(y_true, y_pred):
    true_best = y_true.argmax(axis=1)
    pred_order = np.argsort(-y_pred, axis=1)
    top1 = float(np.mean(pred_order[:, 0] == true_best))
    top3 = float(np.mean([t in row[:3] for t, row in zip(true_best, pred_order)]))
    rho = float(np.nanmean([spearmanr(t, p).correlation for t, p in zip(y_true, y_pred)]))
    # Regret: score points lost by planting the predicted #1 instead of the true #1
    rows = np.arange(len(y_true))
    regret = float(np.mean(y_true[rows, true_best] - y_true[rows, pred_order[:, 0]]) * 100)
    return top1, top3, rho, regret


def evaluate(model, name, X_test, y_test):
    y_pred = np.clip(model.predict(X_test), 0, 1)
    mae = mean_absolute_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)
    per_crop = {c: mean_absolute_error(y_test[:, i], y_pred[:, i]) * 100
                for i, c in enumerate(Config.CROP_NAMES)}
    top1, top3, rho, regret = ranking_metrics(y_test, y_pred)

    print(f"\n{'─' * 50}\n📊 {name}\n{'─' * 50}")
    print(f"   MAE:            {mae:.2f} points (0-100 scale)")
    print(f"   R²:             {r2:.4f}")
    print(f"   Top-1 agreement:{top1 * 100:6.1f}%")
    print(f"   Top-3 hit rate: {top3 * 100:6.1f}%")
    print(f"   Spearman rank:  {rho:.3f}")
    print(f"   Mean regret:    {regret:.2f} points (true #1 vs predicted #1)")
    print("   MAE per crop:   " + ', '.join(f"{c} {v:.1f}" for c, v in per_crop.items()))

    return {'mae': mae, 'r2': r2, 'top1': top1, 'top3': top3, 'spearman': rho, 'regret': regret,
            'mae_per_crop': per_crop, 'y_pred': y_pred}


# ============================================================================
# PLOTS
# ============================================================================

def save_plots(results, winner_name, y_test, save_path='models/plots'):
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results)
    x = np.arange(len(Config.CROP_NAMES))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [results[name]['mae_per_crop'][c] for c in Config.CROP_NAMES]
        axes[0].bar(x + (i - len(names) / 2 + 0.5) * w, vals, w, label=name)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(Config.CROP_NAMES, rotation=30)
    axes[0].set_ylabel('MAE (score points)')
    axes[0].set_title('Prediction error per crop')
    axes[0].legend()

    y_pred = results[winner_name]['y_pred']
    axes[1].scatter(y_test.ravel() * 100, y_pred.ravel() * 100, s=2, alpha=0.3)
    axes[1].plot([0, 100], [0, 100], color='black', linewidth=1)
    axes[1].set_xlabel('True suitability')
    axes[1].set_ylabel('Predicted suitability')
    axes[1].set_title(f'{winner_name} — predicted vs true (all crops)')

    plt.tight_layout()
    plt.savefig(f'{save_path}/model_evaluation.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Plot saved: {save_path}/model_evaluation.png")


# ============================================================================
# SAVE
# ============================================================================

def save_model(model, metadata, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, Config.MODEL_FILE)
    joblib.dump(model, path, compress=3)
    print(f"\n💾 Model → {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    with open(os.path.join(save_dir, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   Meta  → {save_dir}/model_metadata.json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("🌱 AGRITECH — CROP SUITABILITY MODEL TRAINING")
    print("=" * 60)

    df = load_data('soil_data.csv')
    X = df[Config.FEATURES].values
    y = df[Config.TARGETS].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)

    candidates = {}
    rf, rf_params = train_random_forest(X_tr, y_tr)
    candidates['Random Forest'] = (rf, rf_params)
    if HAS_LGB:
        lgbm, lgb_params = train_lightgbm(X_tr, y_tr)
        candidates['LightGBM'] = (lgbm, lgb_params)

    results = {name: evaluate(m, name, X_te, y_te) for name, (m, _) in candidates.items()}
    w_name = min(results, key=lambda n: results[n]['mae'])
    winner, w_params = candidates[w_name]
    w_res = results[w_name]

    print(f"\n{'=' * 60}\n🏆 WINNER: {w_name} (MAE {w_res['mae']:.2f})\n{'=' * 60}")

    print(f"\n🔁 {Config.CV_FOLDS}-fold cross-validation of {w_name} on the training set...")
    cv = KFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    cv_mae = -cross_val_score(winner, X_tr, y_tr, cv=cv, scoring='neg_mean_absolute_error') * 100
    print(f"   CV MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f} points")

    # Refit on all data now that the model is chosen and evaluated
    winner.fit(X, y)

    imp = feature_importance(winner)
    print("\n   Feature importance (mean over crops):")
    for f, v in sorted(zip(Config.FEATURES, imp), key=lambda t: -t[1]):
        print(f"     {f:3s} {v:.3f} {'█' * int(v * 60)}")

    try:
        save_plots(results, w_name, y_te)
    except Exception as e:
        print(f"   ⚠ Plots skipped: {e}")

    save_model(winner, {
        'task': 'crop_suitability',
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'model_type': w_name,
        'model_file': Config.MODEL_FILE,
        'features': Config.FEATURES,
        'crop_names': Config.CROP_NAMES,
        'n_samples': int(len(df)),
        'test_mae': round(w_res['mae'], 3),
        'test_r2': round(w_res['r2'], 4),
        'top1_agreement': round(w_res['top1'], 4),
        'top3_hit_rate': round(w_res['top3'], 4),
        'spearman': round(w_res['spearman'], 4),
        'mean_regret': round(w_res['regret'], 3),
        'cv_mae_mean': round(float(cv_mae.mean()), 3),
        'cv_mae_std': round(float(cv_mae.std()), 3),
        'mae_per_crop': {c: round(v, 3) for c, v in w_res['mae_per_crop'].items()},
        'best_params': {k: (v.item() if hasattr(v, 'item') else v) for k, v in w_params.items()},
    })

    print(f"\n✅ TRAINING COMPLETE — {w_name}, MAE {w_res['mae']:.2f} points, "
          f"top-3 hit rate {w_res['top3'] * 100:.1f}%")


if __name__ == '__main__':
    main()
