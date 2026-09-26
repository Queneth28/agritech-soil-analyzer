"""
Burkina Faso / Sahel Region — Crop Suitability Dataset Generator
AgriTech Soil Analyzer

Generates realistic soil profiles for Burkina Faso and labels each profile
with a suitability score (0-1) for EVERY crop, using the agronomic
requirements in crop_profiles.py (FAO, ICRISAT, INERA references).

Why per-crop scores instead of one "best crop" label: most soils suit
several crops (on average ~5 of the 9 score >= 0.75). Forcing a single
label and discarding "ambiguous" soils left a dataset that was 98% millet.
Scores keep all the information; the app ranks crops by predicted score.

Soil types modeled (proportions reflect actual land cover in Burkina Faso):
  Lixisol  (Sandy ferruginous) : 40%  — Plateau Central, northern Sahel
  Luvisol  (Sandy loam)        : 25%  — Centre-West, South-West Burkina
  Bas-fond (Lowland/riverside) : 15%  — River valleys, irrigated bas-fonds
  Lithosol (Degraded laterite) : 15%  — Degraded plateau, laterite outcrops
  Vertisol (Black cotton soil) :  5%  — Lowland depressions, clay hollows
plus a share of "Broad" profiles sampled uniformly over wide input ranges
(fertilized fields, acid or saline soils) so the model does not have to
extrapolate on values users can legitimately enter.

Crops: Sorgho, Mil, Niébé, Arachide, Maïs, Coton, Sésame, Soja, Riz

Output: soil_data.csv
  Columns: N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B,
           score_<crop> for each crop, Output (index of best crop)
  soil_data_full.csv additionally has soil_type and best_crop.

Usage:
  python generate_burkina_dataset.py
  python generate_burkina_dataset.py --samples 8000 --output soil_data.csv
"""

import numpy as np
import pandas as pd
import argparse

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

from crop_profiles import CROPS, FEATURES, all_crop_scores  # noqa: E402


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


# Wide ranges for the "Broad" component (roughly the app's accepted inputs,
# trimmed to values seen in real West African soil tests)
BROAD_RANGES = {
    'N': (20, 350), 'P': (1, 40), 'K': (50, 800), 'pH': (4.5, 8.5),
    'EC': (0.02, 1.5), 'OC': (0.05, 2.5), 'S': (1, 40), 'Zn': (0.03, 2.5),
    'Fe': (0.1, 5.0), 'Cu': (0.03, 2.5), 'Mn': (0.2, 12.0), 'B': (0.02, 1.5),
}
BROAD_SHARE = 0.15
CROP_NAMES = [c['name'] for c in CROPS.values()]
SCORE_COLUMNS = [f'score_{name}' for name in CROP_NAMES]


def generate_broad_profile():
    profile = {}
    for param, (lo, hi) in BROAD_RANGES.items():
        value = np.random.uniform(lo, hi)
        profile[param] = round(float(value), 2 if param not in ('N', 'P', 'K', 'S') else 1)
    return profile


def label_profile(soil, noise_std=0.02):
    """
    Suitability of every crop for this soil. A little Gaussian noise stands
    in for field variability (rainfall, management) not captured by the soil test.
    """
    scores = all_crop_scores(soil)
    return {f'score_{name}': round(float(np.clip(s + np.random.normal(0, noise_std), 0, 1)), 4)
            for name, s in scores.items()}


# ============================================================================
# MAIN GENERATOR
# ============================================================================

def generate_dataset(n_samples=8000, output_path='soil_data.csv'):
    print("=" * 60)
    print("BURKINA FASO CROP SUITABILITY DATASET GENERATOR")
    print(f"Target: {n_samples} samples — Sahel / Burkina Faso")
    print("=" * 60)

    n_broad = int(n_samples * BROAD_SHARE)
    n_typed = n_samples - n_broad
    names = list(SOIL_TYPES.keys())
    counts = [int(SOIL_TYPES[t]['proportion'] * n_typed) for t in names]
    counts[0] += n_typed - sum(counts)

    rows = []
    for soil_type, count in zip(names, counts):
        for _ in range(count):
            profile = generate_profile(SOIL_TYPES[soil_type])
            rows.append({**profile, **label_profile(profile), 'soil_type': soil_type})
    for _ in range(n_broad):
        profile = generate_broad_profile()
        rows.append({**profile, **label_profile(profile), 'soil_type': 'Broad'})

    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df['Output'] = df[SCORE_COLUMNS].values.argmax(axis=1)
    df['best_crop'] = [CROP_NAMES[i] for i in df['Output']]

    # ── Reports ───────────────────────────────────────────────────
    print("\nSoil types: " + ', '.join(f"{t} {int((df['soil_type'] == t).sum())}"
                                        for t in names + ['Broad']))
    print("\nMean suitability per crop, by soil type:")
    print((df.groupby('soil_type')[SCORE_COLUMNS].mean() * 100).round(0)
          .rename(columns=lambda c: c[6:]).to_string())
    print("\nBest crop distribution (for reference only — not the training target):")
    for i, name in enumerate(CROP_NAMES):
        count = int((df['Output'] == i).sum())
        print(f"  {name:<10} {count:>5} ({count / len(df) * 100:4.1f}%)  {'█' * int(count / len(df) * 40)}")
    good = (df[SCORE_COLUMNS] >= 0.75).sum(axis=1)
    print(f"\nCrops with score >= 0.75 per soil: mean {good.mean():.1f}")

    # ── Save ─────────────────────────────────────────────────────
    full_path = output_path.replace('.csv', '_full.csv')
    df.to_csv(full_path, index=False)
    df[FEATURES + SCORE_COLUMNS + ['Output']].to_csv(output_path, index=False)
    print(f"\nFull dataset → {full_path}")
    print(f"Training dataset → {output_path} ({len(df)} samples)")
    print("=" * 60)
    return df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Burkina Faso crop suitability dataset')
    parser.add_argument('--samples', type=int, default=8000,
                        help='Number of soil profiles to generate (default: 8000)')
    parser.add_argument('--output', type=str, default='soil_data.csv',
                        help='Output CSV filename (default: soil_data.csv)')
    args = parser.parse_args()
    generate_dataset(n_samples=args.samples, output_path=args.output)
