"""
Model Training Script - Enhanced for Production
AgriTech Soil Analyzer

WHAT CHANGED vs your original:
  1. Trains BOTH Random Forest AND XGBoost — picks the best one automatically
  2. Hyperparameter tuning via RandomizedSearchCV (50 combos tested)
  3. SHAP explainability — explains WHY the model made each prediction
  4. Stratified K-Fold cross-validation (5 folds) for reliable accuracy
  5. Model versioning — saves metadata (date, accuracy, params) as JSON
  6. Outlier detection and reporting
  7. More evaluation metrics: F1, precision, recall (not just accuracy)

HOW TO RUN:
  python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from scipy.stats import randint, uniform
import joblib
import os
import json
from datetime import datetime

# Try importing XGBoost (installed via requirements.txt)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ XGBoost not installed. Will train Random Forest only.")
    print("  Install with: pip install xgboost")

# Try importing SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠ SHAP not installed. Explainability will be skipped.")
    print("  Install with: pip install shap")

# Matplotlib — non-interactive backend for servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# CONFIGURATION — Edit these values if needed
# ============================================================================

class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    TUNING_ITERATIONS = 50   # How many hyperparameter combos to try

    # Your 12 soil features (must match your CSV columns)
    FEATURES = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']
    TARGET = 'Output'        # Column name for the target variable

    # What each class number means
    CLASS_NAMES = {0: 'Low', 1: 'Medium', 2: 'High'}

    # Random Forest hyperparameter search space
    RF_PARAM_DIST = {
        'n_estimators': randint(80, 300),
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': randint(2, 12),
        'min_samples_leaf': randint(1, 8),
        'max_features': ['sqrt', 'log2', None],
    }

    # XGBoost hyperparameter search space
    XGB_PARAM_DIST = {
        'n_estimators': randint(80, 300),
        'max_depth': randint(3, 12),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 8),
        'gamma': uniform(0, 0.5),
    }


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data(filepath='soil_data.csv'):
    """
    Load soil dataset from CSV file.
    If the file doesn't exist, generates synthetic data for demonstration.
    """
    print(f"\n📂 Loading data from: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"   ✗ File not found: {filepath}")
        print("   → Generating synthetic dataset for demonstration...")
        return generate_synthetic_data()

    print(f"   ✓ Loaded: {df.shape[0]} samples, {df.shape[1]} columns")

    # Validate that required columns exist
    required = set(Config.FEATURES + [Config.TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Report missing values
    missing_vals = df[Config.FEATURES].isnull().sum()
    if missing_vals.sum() > 0:
        print("   ⚠ Missing values found:")
        for col, count in missing_vals[missing_vals > 0].items():
            print(f"     {col}: {count} ({count/len(df)*100:.1f}%)")

    # Report class distribution
    print(f"\n   Class distribution:")
    for cls_val, count in df[Config.TARGET].value_counts().sort_index().items():
        label = Config.CLASS_NAMES.get(cls_val, f"Class {cls_val}")
        print(f"     {label}: {count} ({count/len(df)*100:.1f}%)")

    return df


def generate_synthetic_data(n_samples=1000):
    """Generate synthetic soil data based on realistic ranges from the research paper."""
    np.random.seed(Config.RANDOM_STATE)

    data = {
        'N':  np.random.uniform(138, 270, n_samples),
        'P':  np.random.uniform(6.8, 9.9, n_samples),
        'K':  np.random.uniform(338, 718, n_samples),
        'pH': np.random.uniform(7.46, 7.64, n_samples),
        'EC': np.random.uniform(0.40, 0.75, n_samples),
        'OC': np.random.uniform(0.7, 1.11, n_samples),
        'S':  np.random.uniform(5.9, 26.0, n_samples),
        'Zn': np.random.uniform(0.24, 0.34, n_samples),
        'Fe': np.random.uniform(0.31, 0.86, n_samples),
        'Cu': np.random.uniform(0.77, 1.69, n_samples),
        'Mn': np.random.uniform(2.43, 8.71, n_samples),
        'B':  np.random.uniform(0.11, 2.29, n_samples),
    }
    df = pd.DataFrame(data)

    def classify_soil(row):
        score = 0
        if 150 <= row['N'] <= 250: score += 3
        elif 100 <= row['N'] < 150 or 250 < row['N'] <= 300: score += 2
        else: score += 1
        if 7 <= row['P'] <= 10: score += 3
        elif 5 <= row['P'] < 7 or 10 < row['P'] <= 12: score += 2
        else: score += 1
        if 400 <= row['K'] <= 700: score += 3
        elif 300 <= row['K'] < 400 or 700 < row['K'] <= 800: score += 2
        else: score += 1
        if 6.5 <= row['pH'] <= 7.5: score += 2
        else: score += 1
        if score >= 10: return 2  # High
        elif score >= 7: return 1  # Medium
        else: return 0            # Low

    df[Config.TARGET] = df.apply(classify_soil, axis=1)
    df.to_csv('soil_data_synthetic.csv', index=False)
    print(f"   ✓ Synthetic dataset: {len(df)} samples → soil_data_synthetic.csv")
    return df


# ============================================================================
# STEP 2: PREPROCESS
# ============================================================================

def preprocess_data(df):
    """Clean data, handle missing values, scale features, split into train/test."""
    print("\n🔧 Preprocessing...")

    X = df[Config.FEATURES].copy()
    y = df[Config.TARGET].copy()

    # Handle missing values with median (more robust than mean)
    if X.isnull().sum().sum() > 0:
        print("   → Imputing missing values (median strategy)")
        X = X.fillna(X.median())

    # Detect outliers (Z-score > 3) — report only, don't remove
    print("   → Checking for outliers (Z-score > 3):")
    outlier_found = False
    for col in Config.FEATURES:
        z = np.abs((X[col] - X[col].mean()) / X[col].std())
        n = (z > 3).sum()
        if n > 0:
            print(f"     {col}: {n} outliers")
            outlier_found = True
    if not outlier_found:
        print("     None detected")

    # Stratified split — preserves class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    print(f"   → Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Scale features (Z-score normalization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   → StandardScaler applied")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================================
# STEP 3: TRAIN MODELS
# ============================================================================

def train_random_forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning."""
    print(f"\n🌲 Training Random Forest...")
    print(f"   Searching {Config.TUNING_ITERATIONS} parameter combinations...")

    base = RandomForestClassifier(
        random_state=Config.RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    search = RandomizedSearchCV(
        base, Config.RF_PARAM_DIST,
        n_iter=Config.TUNING_ITERATIONS,
        cv=StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True,
                           random_state=Config.RANDOM_STATE),
        scoring='f1_weighted',
        random_state=Config.RANDOM_STATE,
        n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)

    print(f"   ✓ Best CV F1: {search.best_score_:.4f}")
    print(f"   ✓ Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def train_xgboost(X_train, y_train):
    """Train XGBoost with hyperparameter tuning."""
    if not HAS_XGB:
        return None, {}, 0.0

    print(f"\n⚡ Training XGBoost...")
    print(f"   Searching {Config.TUNING_ITERATIONS} parameter combinations...")

    n_classes = len(np.unique(y_train))
    base = xgb.XGBClassifier(
        objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
        num_class=n_classes if n_classes > 2 else None,
        random_state=Config.RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='mlogloss' if n_classes > 2 else 'logloss',
        verbosity=0
    )
    search = RandomizedSearchCV(
        base, Config.XGB_PARAM_DIST,
        n_iter=Config.TUNING_ITERATIONS,
        cv=StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True,
                           random_state=Config.RANDOM_STATE),
        scoring='f1_weighted',
        random_state=Config.RANDOM_STATE,
        n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)

    print(f"   ✓ Best CV F1: {search.best_score_:.4f}")
    print(f"   ✓ Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


# ============================================================================
# STEP 4: EVALUATE
# ============================================================================

def evaluate_model(model, name, X_train, X_test, y_train, y_test):
    """Run full evaluation: accuracy, F1, precision, recall, CV, confusion matrix."""
    print(f"\n{'─' * 50}")
    print(f"📊 EVALUATION: {name}")
    print(f"{'─' * 50}")

    train_acc = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')

    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True,
                         random_state=Config.RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"   Train Accuracy:  {train_acc:.4f}")
    print(f"   Test Accuracy:   {test_acc:.4f}")
    print(f"   F1 (weighted):   {f1:.4f}")
    print(f"   Precision:       {prec:.4f}")
    print(f"   Recall:          {rec:.4f}")
    print(f"   CV Accuracy:     {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")

    # Classification report
    unique = sorted(set(y_test))
    labels = [Config.CLASS_NAMES[i] for i in unique]
    print(f"\n{classification_report(y_test, y_pred, target_names=labels)}")

    # Feature importance
    cm = confusion_matrix(y_test, y_pred)
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({
            'feature': Config.FEATURES,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("   Feature Importance:")
        for _, row in fi.iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"     {row['feature']:4s} {row['importance']:.4f} {bar}")

    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_weighted': f1,
        'precision': prec,
        'recall': rec,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': cm,
        'y_pred': y_pred,
    }


# ============================================================================
# STEP 5: SHAP EXPLAINABILITY
# ============================================================================

def compute_shap_explainer(model, X_train_scaled, name):
    """Build a SHAP explainer for the winning model."""
    if not HAS_SHAP:
        print("\n   ⚠ SHAP not available — skipping explainability")
        return None

    print(f"\n🔍 Computing SHAP explainer for {name}...")
    try:
        explainer = shap.TreeExplainer(model)
        # Validate on a small sample
        sample = X_train_scaled[:min(100, len(X_train_scaled))]
        _ = explainer.shap_values(sample)
        print(f"   ✓ SHAP explainer created successfully")
        return explainer
    except Exception as e:
        print(f"   ⚠ SHAP failed: {e}")
        return None


# ============================================================================
# STEP 6: SAVE PLOTS
# ============================================================================

def save_plots(rf_results, xgb_results, winner_name, save_path='models/plots'):
    """Generate and save comparison plots."""
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Model comparison
    metrics = ['test_accuracy', 'f1_weighted', 'precision', 'recall']
    rf_vals = [rf_results[m] for m in metrics]

    if xgb_results:
        xgb_vals = [xgb_results[m] for m in metrics]
        x = np.arange(len(metrics))
        w = 0.35
        axes[0].bar(x - w/2, rf_vals, w, label='Random Forest', color='#2196F3', alpha=0.85)
        axes[0].bar(x + w/2, xgb_vals, w, label='XGBoost', color='#FF9800', alpha=0.85)
        axes[0].legend()
    else:
        x = np.arange(len(metrics))
        axes[0].bar(x, rf_vals, color='#2196F3', alpha=0.85)

    axes[0].set_xticks(x if not xgb_results else x)
    axes[0].set_xticklabels(['Accuracy', 'F1', 'Precision', 'Recall'])
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Model Comparison (Winner: {winner_name})')
    axes[0].set_ylim(0, 1.05)

    # 2. Confusion matrix
    winner_results = rf_results if 'Random Forest' in winner_name else (xgb_results or rf_results)
    n_classes = len(winner_results['confusion_matrix'])
    labels = [Config.CLASS_NAMES.get(i, str(i)) for i in range(n_classes)]
    sns.heatmap(winner_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_ylabel('True')
    axes[1].set_xlabel('Predicted')
    axes[1].set_title(f'Confusion Matrix — {winner_name}')

    plt.tight_layout()
    plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Plot saved: {save_path}/model_comparison.png")
    plt.close()


# ============================================================================
# STEP 7: SAVE MODEL + METADATA
# ============================================================================

def save_everything(model, scaler, shap_explainer, metadata, save_dir='models'):
    """Save model, scaler, SHAP explainer, and metadata JSON."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model (always as rf_model.pkl so app.py can find it)
    joblib.dump(model, f'{save_dir}/rf_model.pkl')
    joblib.dump(scaler, f'{save_dir}/scaler.pkl')
    print(f"\n💾 Model  → {save_dir}/rf_model.pkl")
    print(f"   Scaler → {save_dir}/scaler.pkl")

    # Save SHAP explainer
    if shap_explainer is not None:
        joblib.dump(shap_explainer, f'{save_dir}/shap_explainer.pkl')
        print(f"   SHAP   → {save_dir}/shap_explainer.pkl")

    # Save metadata as JSON
    def to_native(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    meta = {
        'trained_at': datetime.now().isoformat(),
        'model_type': metadata.get('model_type', 'Unknown'),
        'test_accuracy': float(metadata.get('test_accuracy', 0)),
        'f1_weighted': float(metadata.get('f1_weighted', 0)),
        'cv_mean': float(metadata.get('cv_mean', 0)),
        'cv_std': float(metadata.get('cv_std', 0)),
        'best_params': {k: str(v) for k, v in metadata.get('best_params', {}).items()},
        'features': Config.FEATURES,
        'n_classes': len(Config.CLASS_NAMES),
        'class_names': {str(k): v for k, v in Config.CLASS_NAMES.items()},
    }
    meta = json.loads(json.dumps(meta, default=to_native))
    with open(f'{save_dir}/model_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"   Meta   → {save_dir}/model_metadata.json")


# ============================================================================
# MAIN PIPELINE — Run this
# ============================================================================

def main():
    print("=" * 60)
    print("🌱 AGRITECH SOIL ANALYZER — MODEL TRAINING")
    print("   Random Forest + XGBoost | Tuning | SHAP")
    print("=" * 60)

    # 1. Load
    df = load_data('soil_data.csv')

    # 2. Preprocess
    X_tr, X_te, y_tr, y_te, scaler = preprocess_data(df)

    # 3. Train both models
    rf_model, rf_params, rf_cv = train_random_forest(X_tr, y_tr)

    xgb_model, xgb_params, xgb_cv = None, {}, 0.0
    if HAS_XGB:
        xgb_model, xgb_params, xgb_cv = train_xgboost(X_tr, y_tr)

    # 4. Evaluate both
    rf_res = evaluate_model(rf_model, "Random Forest", X_tr, X_te, y_tr, y_te)

    xgb_res = None
    if xgb_model is not None:
        xgb_res = evaluate_model(xgb_model, "XGBoost", X_tr, X_te, y_tr, y_te)

    # 5. Pick the winner
    if xgb_res and xgb_res['f1_weighted'] > rf_res['f1_weighted']:
        winner, w_name, w_res, w_params = xgb_model, "XGBoost", xgb_res, xgb_params
    else:
        winner, w_name, w_res, w_params = rf_model, "Random Forest", rf_res, rf_params

    print(f"\n{'=' * 60}")
    print(f"🏆 WINNER: {w_name}")
    print(f"   Accuracy: {w_res['test_accuracy']:.4f}")
    print(f"   F1 Score: {w_res['f1_weighted']:.4f}")
    print(f"{'=' * 60}")

    # 6. SHAP
    shap_exp = compute_shap_explainer(winner, X_tr, w_name)

    # 7. Plots
    try:
        save_plots(rf_res, xgb_res, w_name)
    except Exception as e:
        print(f"   ⚠ Plots skipped: {e}")

    # 8. Save
    save_everything(winner, scaler, shap_exp, {
        'model_type': w_name,
        'test_accuracy': w_res['test_accuracy'],
        'f1_weighted': w_res['f1_weighted'],
        'cv_mean': w_res['cv_mean'],
        'cv_std': w_res['cv_std'],
        'best_params': w_params,
    })

    print(f"\n{'=' * 60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Model:    {w_name}")
    print(f"   Accuracy: {w_res['test_accuracy']*100:.1f}%")
    print(f"   F1:       {w_res['f1_weighted']*100:.1f}%")
    print(f"   Deploy:   models/rf_model.pkl")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()