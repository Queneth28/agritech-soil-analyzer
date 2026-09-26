"""
Burkina Faso / Sahel Region — Crop Recommendation Dataset Generator
AgriTech Soil Analyzer

Generates realistic soil profiles for Burkina Faso and labels each sample
with the most suitable crop based on agronomic thresholds validated against
ICRISAT Sahelian Center, FAO Soils Bulletin, and INERA Burkina Faso research.

Soil types modeled (proportions reflect actual land cover in Burkina Faso):
  Lixisol  (Sandy ferruginous) : 40%  — Plateau Central, northern Sahel
  Luvisol  (Sandy loam)        : 25%  — Centre-West, South-West Burkina
  Bas-fond (Lowland/riverside) : 15%  — River valleys, irrigated bas-fonds
  Lithosol (Degraded laterite) : 15%  — Degraded plateau, laterite outcrops
  Vertisol (Black cotton soil) :  5%  — Lowland depressions, clay hollows

Crops covered:
  0  Sorgho    (Sorghum bicolor)        — primary staple, drought-tolerant
  1  Mil       (Pennisetum glaucum)     — most drought-tolerant Sahelian crop
  2  Niébé     (Vigna unguiculata)      — nitrogen-fixing legume, staple
  3  Arachide  (Arachis hypogaea)       — legume, important cash+food crop
  4  Maïs      (Zea mays)              — high-input, needs good soils
  5  Coton     (Gossypium hirsutum)     — primary cash crop, heavy feeder
  6  Sésame    (Sesamum indicum)        — drought-tolerant export cash crop
  7  Soja      (Glycine max)            — legume, higher input requirements
  8  Riz       (Oryza sativa)           — irrigated bas-fond zones only

Output: soil_data.csv
  Columns: N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B, Output, crop_name

Usage:
  python generate_burkina_dataset.py
  python generate_burkina_dataset.py --samples 5000 --output soil_data.csv
"""

import numpy as np
import pandas as pd
import argparse
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

from crop_profiles import CROPS, FEATURES, crop_suitability  # noqa: E402


# ============================================================================
# SOIL TYPE DISTRIBUTIONS — Burkina Faso / Sahel
# Each type defines mean and std for each parameter.
# Values are clipped to agronomically realistic min/max bounds.
# ============================================================================
#
# Parameters (units):
#   N   = Total Nitrogen     (mg/kg)
#   P   = Available P Bray   (mg/kg)
#   K   = Exchangeable K     (mg/kg)
#   pH  = soil pH (water)
#   EC  = Electrical Conductivity (dS/m)
#   OC  = Organic Carbon     (%)
#   S   = Available Sulfur   (mg/kg)
#   Zn  = Available Zinc     (mg/kg)
#   Fe  = Available Iron     (mg/kg)
#   Cu  = Available Copper   (mg/kg)
#   Mn  = Available Manganese(mg/kg)
#   B   = Available Boron    (mg/kg)

SOIL_TYPES = {
    'Lixisol': {
        'proportion': 0.40,
        'description': 'Sandy ferruginous — Plateau Central, northern Sahel',
        # Dominant soil of the Sahel plateau. Sandy, low OC, low P, prone to
        # surface crusting. Supports millet and sorghum under rain-fed conditions.
        'params': {
            'N':  {'mean': 90,   'std': 25,   'min': 40,   'max': 200 },
            'P':  {'mean': 4.5,  'std': 1.5,  'min': 2.0,  'max': 9.0 },
            'K':  {'mean': 175,  'std': 50,   'min': 80,   'max': 340 },
            'pH': {'mean': 6.3,  'std': 0.35, 'min': 5.6,  'max': 7.2 },
            'EC': {'mean': 0.28, 'std': 0.10, 'min': 0.05, 'max': 0.65},
            'OC': {'mean': 0.35, 'std': 0.12, 'min': 0.10, 'max': 0.72},
            'S':  {'mean': 8,    'std': 3,    'min': 2,    'max': 20  },
            'Zn': {'mean': 0.38, 'std': 0.18, 'min': 0.08, 'max': 1.1 },
            'Fe': {'mean': 1.4,  'std': 0.55, 'min': 0.4,  'max': 3.2 },
            'Cu': {'mean': 0.45, 'std': 0.18, 'min': 0.08, 'max': 1.3 },
            'Mn': {'mean': 2.8,  'std': 1.1,  'min': 0.5,  'max': 7.0 },
            'B':  {'mean': 0.18, 'std': 0.07, 'min': 0.04, 'max': 0.55},
        },
    },
    'Luvisol': {
        'proportion': 0.25,
        'description': 'Sandy loam — Centre-West, Hauts-Bassins, Sud-Ouest',
        # Better textured soils with moderate OC. Support diverse crops.
        # Common in the cotton belt (SOFITEX zones) and groundnut areas.
        'params': {
            'N':  {'mean': 125,  'std': 32,   'min': 55,   'max': 240 },
            'P':  {'mean': 6.5,  'std': 1.8,  'min': 3.0,  'max': 12.0},
            'K':  {'mean': 255,  'std': 72,   'min': 110,  'max': 460 },
            'pH': {'mean': 6.5,  'std': 0.38, 'min': 5.7,  'max': 7.4 },
            'EC': {'mean': 0.34, 'std': 0.12, 'min': 0.08, 'max': 0.72},
            'OC': {'mean': 0.58, 'std': 0.16, 'min': 0.20, 'max': 1.05},
            'S':  {'mean': 12,   'std': 4,    'min': 4,    'max': 26  },
            'Zn': {'mean': 0.62, 'std': 0.22, 'min': 0.18, 'max': 1.5 },
            'Fe': {'mean': 2.1,  'std': 0.65, 'min': 0.7,  'max': 3.8 },
            'Cu': {'mean': 0.72, 'std': 0.22, 'min': 0.18, 'max': 1.6 },
            'Mn': {'mean': 4.2,  'std': 1.3,  'min': 1.2,  'max': 8.5 },
            'B':  {'mean': 0.32, 'std': 0.10, 'min': 0.09, 'max': 0.75},
        },
    },
    'Bas-fond': {
        'proportion': 0.15,
        'description': 'Lowland/riverside — Mouhoun, Nakambé, Nazinon valleys',
        # Rich alluvial/colluvial soils in seasonal river valleys.
        # Highest fertility. Supports rice, maize, soybean, market gardening.
        # OC and N are significantly higher than upland soils.
        'params': {
            'N':  {'mean': 165,  'std': 42,   'min': 75,   'max': 290 },
            'P':  {'mean': 8.2,  'std': 2.0,  'min': 4.0,  'max': 14.5},
            'K':  {'mean': 330,  'std': 85,   'min': 145,  'max': 570 },
            'pH': {'mean': 6.2,  'std': 0.42, 'min': 5.2,  'max': 7.2 },
            'EC': {'mean': 0.50, 'std': 0.16, 'min': 0.15, 'max': 1.05},
            'OC': {'mean': 0.88, 'std': 0.26, 'min': 0.38, 'max': 1.65},
            'S':  {'mean': 16,   'std': 5,    'min': 6,    'max': 32  },
            'Zn': {'mean': 0.82, 'std': 0.26, 'min': 0.28, 'max': 1.85},
            'Fe': {'mean': 2.9,  'std': 0.82, 'min': 1.1,  'max': 4.8 },
            'Cu': {'mean': 0.95, 'std': 0.27, 'min': 0.28, 'max': 1.9 },
            'Mn': {'mean': 5.8,  'std': 1.6,  'min': 1.8,  'max': 10.5},
            'B':  {'mean': 0.42, 'std': 0.13, 'min': 0.14, 'max': 0.95},
        },
    },
    'Lithosol': {
        'proportion': 0.15,
        'description': 'Degraded laterite — laterite outcrops, eroded plateau',
        # Severely degraded soils over laterite cuirasse.
        # Very low OC and N. Almost no available P.
        # Only millet and sesame are viable. Common in the Sahel degraded zones.
        'params': {
            'N':  {'mean': 62,   'std': 20,   'min': 22,   'max': 130 },
            'P':  {'mean': 2.8,  'std': 0.9,  'min': 1.0,  'max': 6.5 },
            'K':  {'mean': 128,  'std': 38,   'min': 55,   'max': 245 },
            'pH': {'mean': 5.9,  'std': 0.48, 'min': 4.8,  'max': 7.0 },
            'EC': {'mean': 0.22, 'std': 0.09, 'min': 0.03, 'max': 0.55},
            'OC': {'mean': 0.20, 'std': 0.07, 'min': 0.05, 'max': 0.42},
            'S':  {'mean': 5,    'std': 2,    'min': 1,    'max': 12  },
            'Zn': {'mean': 0.22, 'std': 0.09, 'min': 0.04, 'max': 0.58},
            'Fe': {'mean': 0.9,  'std': 0.38, 'min': 0.15, 'max': 2.1 },
            'Cu': {'mean': 0.28, 'std': 0.11, 'min': 0.04, 'max': 0.75},
            'Mn': {'mean': 1.7,  'std': 0.65, 'min': 0.28, 'max': 4.0 },
            'B':  {'mean': 0.11, 'std': 0.04, 'min': 0.02, 'max': 0.32},
        },
    },
    'Vertisol': {
        'proportion': 0.05,
        'description': 'Black cotton soil — clay depressions, lowland hollows',
        # Heavy clay soils that shrink/crack when dry.
        # Higher pH and K than other types. Moderate-good fertility.
        # Cotton and sorghum perform well. Difficult to work without mechanization.
        'params': {
            'N':  {'mean': 145,  'std': 36,   'min': 65,   'max': 260 },
            'P':  {'mean': 7.2,  'std': 2.0,  'min': 3.0,  'max': 13.5},
            'K':  {'mean': 390,  'std': 92,   'min': 175,  'max': 620 },
            'pH': {'mean': 7.2,  'std': 0.42, 'min': 6.4,  'max': 8.3 },
            'EC': {'mean': 0.62, 'std': 0.20, 'min': 0.18, 'max': 1.25},
            'OC': {'mean': 0.78, 'std': 0.22, 'min': 0.28, 'max': 1.40},
            'S':  {'mean': 15,   'std': 5,    'min': 4,    'max': 30  },
            'Zn': {'mean': 0.72, 'std': 0.22, 'min': 0.20, 'max': 1.5 },
            'Fe': {'mean': 2.6,  'std': 0.75, 'min': 0.9,  'max': 4.8 },
            'Cu': {'mean': 0.85, 'std': 0.25, 'min': 0.22, 'max': 1.7 },
            'Mn': {'mean': 5.2,  'std': 1.6,  'min': 1.5,  'max': 9.5 },
            'B':  {'mean': 0.38, 'std': 0.11, 'min': 0.10, 'max': 0.78},
        },
    },
}


# ============================================================================
# PROFILE GENERATION
# ============================================================================

def generate_profile(soil_type_def):
    """Sample one soil profile from a soil type's parameter distributions."""
    profile = {}
    for param, dist in soil_type_def['params'].items():
        value = np.random.normal(dist['mean'], dist['std'])
        value = np.clip(value, dist['min'], dist['max'])
        # Round to realistic precision
        if param in ('pH', 'EC', 'OC', 'Zn', 'Fe', 'Cu', 'Mn', 'B'):
            value = round(float(value), 2)
        else:
            value = round(float(value), 1)
        profile[param] = value
    return profile


def assign_crop(soil, noise_std=0.015, min_confidence=0.52, min_margin=0.10):
    """
    Score all crops for a soil profile and return the best-fit crop.

    Noise is kept very low (1.5%) — just enough to prevent a perfectly
    rule-based dataset while still producing learnable, consistent labels.

    Returns None if the winning crop's score is too low (min_confidence)
    or if the top two crops are too close (min_margin). These ambiguous
    samples are discarded — they would add noise to training, not signal.
    """
    scores = {}
    for crop_id, crop_def in CROPS.items():
        base_score = crop_suitability(soil, crop_def)
        noise = np.random.normal(0, noise_std)
        scores[crop_id] = np.clip(base_score + noise, 0.0, 1.0)

    sorted_scores = sorted(scores.values(), reverse=True)
    top_score   = sorted_scores[0]
    second_score = sorted_scores[1]
    margin = top_score - second_score

    # Discard ambiguous samples — no clear winner
    if top_score < min_confidence or margin < min_margin:
        return None, None, scores

    best_crop_id = max(scores, key=scores.get)
    return best_crop_id, top_score, scores


# ============================================================================
# MAIN GENERATOR
# ============================================================================

def generate_dataset(n_samples=6000, output_path='soil_data.csv'):
    """
    Generate a full labeled dataset for Burkina Faso crop recommendation.

    Generates 4x the target samples, filters out ambiguous ones, then
    ensures every crop class has at least min_per_class samples by
    running additional targeted generation for underrepresented crops.
    """
    MIN_PER_CLASS = 400   # minimum samples per crop class
    OVERSAMPLE    = 4     # generate this many times n_samples before filtering

    print("=" * 60)
    print("BURKINA FASO CROP RECOMMENDATION DATASET GENERATOR")
    print(f"Target: {n_samples} samples — Sahel / Burkina Faso")
    print("=" * 60)

    soil_type_names = list(SOIL_TYPES.keys())
    proportions     = [SOIL_TYPES[t]['proportion'] for t in soil_type_names]

    def _generate_batch(total):
        counts = [int(p * total) for p in proportions]
        counts[0] += total - sum(counts)
        rows = []
        discarded = 0
        for soil_type_name, count in zip(soil_type_names, counts):
            soil_type_def = SOIL_TYPES[soil_type_name]
            for _ in range(count):
                profile = generate_profile(soil_type_def)
                crop_id, top_score, _ = assign_crop(profile)
                if crop_id is None:
                    discarded += 1
                    continue
                row = dict(profile)
                row['Output']     = crop_id
                row['crop_name']  = CROPS[crop_id]['name']
                row['soil_type']  = soil_type_name
                row['confidence'] = round(top_score, 3)
                rows.append(row)
        return rows, discarded

    # Phase 1: generate large batch and filter
    print(f"\nPhase 1: generating {n_samples * OVERSAMPLE} candidates...")
    rows, discarded = _generate_batch(n_samples * OVERSAMPLE)
    print(f"  Kept {len(rows)} clear samples, discarded {discarded} ambiguous ones")

    df = pd.DataFrame(rows)

    # Phase 2: top up underrepresented classes
    print("Phase 2: balancing underrepresented crops...")
    for crop_id in sorted(CROPS.keys()):
        current = (df['Output'] == crop_id).sum()
        if current < MIN_PER_CLASS:
            needed = MIN_PER_CLASS - current
            print(f"  {CROPS[crop_id]['name']:<12} has {current} — generating {needed} more")
            extra_rows = []
            attempts = 0
            # Generate from all soil types until we have enough
            while len(extra_rows) < needed and attempts < needed * 50:
                soil_type_name = np.random.choice(soil_type_names, p=proportions)
                profile = generate_profile(SOIL_TYPES[soil_type_name])
                c_id, top_score, _ = assign_crop(profile)
                if c_id == crop_id:
                    row = dict(profile)
                    row['Output']     = crop_id
                    row['crop_name']  = CROPS[crop_id]['name']
                    row['soil_type']  = soil_type_name
                    row['confidence'] = round(top_score, 3)
                    extra_rows.append(row)
                attempts += 1
            if extra_rows:
                df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    # Phase 3: trim to n_samples, keeping class balance
    if len(df) > n_samples:
        df = df.groupby('Output', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(MIN_PER_CLASS, int(n_samples * len(x) / len(df)))),
                                random_state=RANDOM_STATE)
        )
        # If still over, random trim
        if len(df) > n_samples:
            df = df.sample(n_samples, random_state=RANDOM_STATE)

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    total = len(df)

    # ── Class distribution report ─────────────────────────────────
    print(f"\nFinal crop distribution ({total} samples):")
    for crop_id in sorted(CROPS.keys()):
        name  = CROPS[crop_id]['name']
        count = (df['Output'] == crop_id).sum()
        bar   = '█' * int(count / total * 40)
        pct   = count / total * 100
        print(f"  {crop_id}  {name:<12} {count:>5} ({pct:4.1f}%)  {bar}")

    # ── Parameter stats ───────────────────────────────────────────
    print("\nSoil parameter statistics:")
    stats = df[FEATURES].describe().loc[['mean', 'std', 'min', 'max']]
    print(stats.round(2).to_string())

    # ── Save ─────────────────────────────────────────────────────
    full_path = output_path.replace('.csv', '_full.csv')
    df.to_csv(full_path, index=False)
    print(f"\nFull dataset (with soil_type, confidence) → {full_path}")

    train_cols = FEATURES + ['Output']
    df[train_cols].to_csv(output_path, index=False)
    print(f"Training dataset → {output_path}")
    print(f"\nTotal samples : {total}")
    print(f"Classes       : {df['Output'].nunique()} crops")
    print(f"Avg confidence: {df['confidence'].mean():.3f} (higher = cleaner labels)")
    print("=" * 60)

    return df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Burkina Faso crop recommendation dataset'
    )
    parser.add_argument(
        '--samples', type=int, default=5000,
        help='Number of soil profiles to generate (default: 5000)'
    )
    parser.add_argument(
        '--output', type=str, default='soil_data.csv',
        help='Output CSV filename (default: soil_data.csv)'
    )
    args = parser.parse_args()

    df = generate_dataset(n_samples=args.samples, output_path=args.output)
