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

FEATURES = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']

# ============================================================================
# CROP SUITABILITY DEFINITIONS
# Sources: FAO Soils Bulletin No.10, ICRISAT West Africa Bulletin,
#          INERA Burkina Faso technical sheets, IRD Sahel pedology studies,
#          SoilGrids Africa validation reports
# ============================================================================
#
# optimal    → parameter value that gives maximum crop performance
# acceptable → wider range where crop grows but with reduced performance
# weights    → relative importance of each parameter for this specific crop
#              (must sum to 1.0)

CROPS = {
    0: {
        'name': 'Sorgho',
        # Sorghum is the backbone of Sahelian agriculture.
        # pH and N are the primary limiting factors in Burkina Faso soils.
        # Tolerates moderately poor soils better than maize.
        'weights': {
            'N': 0.18, 'P': 0.15, 'K': 0.13, 'pH': 0.20,
            'EC': 0.10, 'OC': 0.12, 'S': 0.04, 'Zn': 0.04,
            'Fe': 0.01, 'Cu': 0.01, 'Mn': 0.01, 'B': 0.01,
        },
        'optimal': {
            'N':  (80,  200), 'P':  (6,   12),  'K':  (150, 450),
            'pH': (6.0, 7.0), 'EC': (0.1, 0.8), 'OC': (0.4, 1.5),
            'S':  (6,   22),  'Zn': (0.3, 2.0), 'Fe': (0.5, 4.0),
            'Cu': (0.3, 1.5), 'Mn': (1.5, 8.0), 'B':  (0.15, 1.0),
        },
        'acceptable': {
            'N':  (50,  260), 'P':  (4,   15),  'K':  (100, 600),
            'pH': (5.5, 7.5), 'EC': (0.0, 1.2), 'OC': (0.2, 2.0),
            'S':  (3,   35),  'Zn': (0.2, 3.0), 'Fe': (0.3, 5.0),
            'Cu': (0.1, 2.0), 'Mn': (0.8, 10.0),'B':  (0.08, 1.5),
        },
    },
    1: {
        'name': 'Mil',
        # Pearl millet thrives in the poorest, sandiest Sahelian soils.
        # Extremely drought-tolerant. Zn deficiency is a key issue in Sahel.
        # Low P and OC are acceptable — it outcompetes all crops in poor soils.
        'weights': {
            'N': 0.14, 'P': 0.11, 'K': 0.11, 'pH': 0.18,
            'EC': 0.13, 'OC': 0.09, 'S': 0.04, 'Zn': 0.09,
            'Fe': 0.03, 'Cu': 0.02, 'Mn': 0.03, 'B': 0.03,
        },
        'optimal': {
            'N':  (50,  160), 'P':  (3,   9),   'K':  (100, 350),
            'pH': (5.5, 7.0), 'EC': (0.1, 0.6), 'OC': (0.2, 1.0),
            'S':  (3,   18),  'Zn': (0.2, 1.8), 'Fe': (0.5, 4.0),
            'Cu': (0.2, 1.5), 'Mn': (1.0, 7.0), 'B':  (0.1, 0.9),
        },
        'acceptable': {
            'N':  (30,  220), 'P':  (2,   13),  'K':  (70,  500),
            'pH': (5.0, 7.5), 'EC': (0.0, 1.0), 'OC': (0.1, 1.8),
            'S':  (2,   28),  'Zn': (0.1, 2.5), 'Fe': (0.3, 5.0),
            'Cu': (0.1, 2.0), 'Mn': (0.5, 10.0),'B':  (0.05, 1.5),
        },
    },
    2: {
        'name': 'Niébé',
        # Cowpea fixes atmospheric N so soil N is less critical.
        # P is the most limiting nutrient — essential for nodulation.
        # Very important in Burkina food security (protein source).
        'weights': {
            'N': 0.08, 'P': 0.22, 'K': 0.15, 'pH': 0.18,
            'EC': 0.10, 'OC': 0.13, 'S': 0.04, 'Zn': 0.04,
            'Fe': 0.02, 'Cu': 0.01, 'Mn': 0.01, 'B': 0.02,
        },
        'optimal': {
            'N':  (30,  120), 'P':  (7,   13),  'K':  (150, 420),
            'pH': (6.0, 7.0), 'EC': (0.1, 0.6), 'OC': (0.4, 1.3),
            'S':  (6,   20),  'Zn': (0.3, 1.8), 'Fe': (0.5, 3.5),
            'Cu': (0.3, 1.5), 'Mn': (1.5, 7.0), 'B':  (0.15, 0.9),
        },
        'acceptable': {
            'N':  (20,  180), 'P':  (5,   15),  'K':  (100, 580),
            'pH': (5.5, 7.5), 'EC': (0.0, 0.9), 'OC': (0.3, 2.0),
            'S':  (3,   32),  'Zn': (0.2, 2.5), 'Fe': (0.3, 4.5),
            'Cu': (0.1, 2.0), 'Mn': (0.8, 9.0), 'B':  (0.08, 1.4),
        },
    },
    3: {
        'name': 'Arachide',
        # Groundnut is pH-sensitive — does NOT tolerate alkaline soils.
        # Low EC tolerance. Boron important for pod fill.
        # Dominant in Centre-Sud and Hauts-Bassins regions.
        'weights': {
            'N': 0.08, 'P': 0.18, 'K': 0.15, 'pH': 0.22,
            'EC': 0.12, 'OC': 0.12, 'S': 0.03, 'Zn': 0.04,
            'Fe': 0.01, 'Cu': 0.01, 'Mn': 0.01, 'B': 0.03,
        },
        'optimal': {
            'N':  (40,  130), 'P':  (6,   12),  'K':  (150, 450),
            'pH': (5.8, 6.5), 'EC': (0.1, 0.5), 'OC': (0.4, 1.2),
            'S':  (5,   18),  'Zn': (0.3, 1.8), 'Fe': (0.5, 3.5),
            'Cu': (0.3, 1.5), 'Mn': (1.5, 7.0), 'B':  (0.2, 1.0),
        },
        'acceptable': {
            'N':  (20,  190), 'P':  (4,   15),  'K':  (100, 600),
            'pH': (5.5, 7.0), 'EC': (0.0, 0.7), 'OC': (0.3, 1.8),
            'S':  (3,   28),  'Zn': (0.2, 2.5), 'Fe': (0.3, 4.5),
            'Cu': (0.1, 2.0), 'Mn': (0.8, 9.0), 'B':  (0.1, 1.5),
        },
    },
    4: {
        'name': 'Maïs',
        # Highest nutrient demands of all Sahelian crops.
        # Needs good N, P, K AND adequate OC. Zn deficiency very common in Sahel.
        # Only recommended on good quality soils (Luvisol, bas-fond).
        'weights': {
            'N': 0.22, 'P': 0.18, 'K': 0.15, 'pH': 0.18,
            'EC': 0.05, 'OC': 0.12, 'S': 0.02, 'Zn': 0.06,
            'Fe': 0.01, 'Cu': 0.01, 'Mn': 0.01, 'B': 0.00,
        },
        'optimal': {
            'N':  (140, 260), 'P':  (10,  15),  'K':  (200, 560),
            'pH': (6.0, 7.0), 'EC': (0.1, 0.7), 'OC': (0.6, 1.6),
            'S':  (8,   25),  'Zn': (0.5, 2.0), 'Fe': (0.8, 4.0),
            'Cu': (0.3, 1.5), 'Mn': (2.0, 8.0), 'B':  (0.2, 1.0),
        },
        'acceptable': {
            'N':  (100, 320), 'P':  (7,   15),  'K':  (150, 700),
            'pH': (5.8, 7.2), 'EC': (0.0, 0.9), 'OC': (0.4, 2.2),
            'S':  (5,   35),  'Zn': (0.3, 3.0), 'Fe': (0.5, 5.0),
            'Cu': (0.1, 2.0), 'Mn': (1.0, 10.0),'B':  (0.1, 1.5),
        },
    },
    5: {
        'name': 'Coton',
        # Cotton is the #1 cash crop in Burkina Faso (SOFITEX zones).
        # Very heavy feeder — K is critical for fiber quality.
        # B deficiency causes boll shedding — most B-sensitive crop.
        'weights': {
            'N': 0.17, 'P': 0.15, 'K': 0.18, 'pH': 0.15,
            'EC': 0.05, 'OC': 0.12, 'S': 0.07, 'Zn': 0.03,
            'Fe': 0.01, 'Cu': 0.01, 'Mn': 0.01, 'B': 0.05,
        },
        'optimal': {
            'N':  (120, 240), 'P':  (8,   14),  'K':  (200, 560),
            'pH': (6.0, 7.5), 'EC': (0.1, 0.9), 'OC': (0.5, 1.6),
            'S':  (8,   26),  'Zn': (0.3, 2.0), 'Fe': (0.5, 4.0),
            'Cu': (0.3, 1.5), 'Mn': (1.5, 8.0), 'B':  (0.3, 1.2),
        },
        'acceptable': {
            'N':  (90,  290), 'P':  (6,   15),  'K':  (150, 700),
            'pH': (5.8, 7.8), 'EC': (0.0, 1.2), 'OC': (0.4, 2.2),
            'S':  (5,   38),  'Zn': (0.2, 3.0), 'Fe': (0.3, 5.0),
            'Cu': (0.1, 2.0), 'Mn': (0.8, 10.0),'B':  (0.2, 1.5),
        },
    },
    6: {
        'name': 'Sésame',
        # Sesame tolerates wide pH range and poor soils.
        # S is unusually important — sesame is a sulfur-accumulating crop.
        # Does NOT tolerate waterlogging (high EC is a flag).
        # Growing export crop in eastern and central Burkina Faso.
        'weights': {
            'N': 0.15, 'P': 0.13, 'K': 0.13, 'pH': 0.18,
            'EC': 0.11, 'OC': 0.10, 'S': 0.08, 'Zn': 0.06,
            'Fe': 0.02, 'Cu': 0.01, 'Mn': 0.02, 'B': 0.01,
        },
        'optimal': {
            'N':  (80,  190), 'P':  (5,   11),  'K':  (150, 420),
            'pH': (5.5, 7.2), 'EC': (0.1, 0.7), 'OC': (0.3, 1.3),
            'S':  (7,   24),  'Zn': (0.3, 2.0), 'Fe': (0.5, 3.5),
            'Cu': (0.2, 1.5), 'Mn': (1.5, 7.5), 'B':  (0.15, 1.0),
        },
        'acceptable': {
            'N':  (50,  250), 'P':  (3,   14),  'K':  (100, 570),
            'pH': (5.0, 7.8), 'EC': (0.0, 1.0), 'OC': (0.2, 1.9),
            'S':  (4,   35),  'Zn': (0.2, 2.8), 'Fe': (0.3, 5.0),
            'Cu': (0.1, 2.0), 'Mn': (0.8, 10.0),'B':  (0.08, 1.5),
        },
    },
    7: {
        'name': 'Soja',
        # Soybean has the strictest pH requirement of all crops here.
        # pH outside 5.8-7.2 sharply reduces nodulation and yield.
        # EC must be low. Growing in Hauts-Bassins and Cascades regions.
        'weights': {
            'N': 0.07, 'P': 0.18, 'K': 0.15, 'pH': 0.25,
            'EC': 0.12, 'OC': 0.13, 'S': 0.03, 'Zn': 0.03,
            'Fe': 0.01, 'Cu': 0.01, 'Mn': 0.01, 'B': 0.01,
        },
        'optimal': {
            'N':  (50,  160), 'P':  (8,   13),  'K':  (180, 490),
            'pH': (6.0, 7.0), 'EC': (0.1, 0.5), 'OC': (0.5, 1.5),
            'S':  (6,   20),  'Zn': (0.4, 1.8), 'Fe': (0.5, 3.5),
            'Cu': (0.3, 1.5), 'Mn': (1.5, 7.0), 'B':  (0.2, 1.0),
        },
        'acceptable': {
            'N':  (30,  210), 'P':  (6,   15),  'K':  (140, 620),
            'pH': (5.8, 7.2), 'EC': (0.0, 0.7), 'OC': (0.4, 2.0),
            'S':  (4,   28),  'Zn': (0.2, 2.5), 'Fe': (0.3, 4.5),
            'Cu': (0.1, 2.0), 'Mn': (0.8, 9.0), 'B':  (0.1, 1.5),
        },
    },
    8: {
        'name': 'Riz',
        # Rice requires irrigated conditions (bas-fonds or Office du Niger style).
        # High N and OC are essential. Slightly acidic pH preferred.
        # Fe availability is naturally high in flooded soils — good sign.
        # Only viable near rivers: Mouhoun, Nakambé, Nazinon basins.
        'weights': {
            'N': 0.20, 'P': 0.18, 'K': 0.12, 'pH': 0.20,
            'EC': 0.08, 'OC': 0.15, 'S': 0.02, 'Zn': 0.01,
            'Fe': 0.02, 'Cu': 0.01, 'Mn': 0.01, 'B': 0.00,
        },
        'optimal': {
            'N':  (120, 240), 'P':  (8,   14),  'K':  (150, 460),
            'pH': (5.5, 6.8), 'EC': (0.1, 0.8), 'OC': (0.6, 1.6),
            'S':  (7,   22),  'Zn': (0.3, 1.8), 'Fe': (1.0, 4.5),
            'Cu': (0.3, 1.5), 'Mn': (2.0, 9.0), 'B':  (0.15, 1.0),
        },
        'acceptable': {
            'N':  (80,  290), 'P':  (5,   15),  'K':  (100, 620),
            'pH': (5.0, 7.2), 'EC': (0.0, 1.2), 'OC': (0.4, 2.2),
            'S':  (4,   32),  'Zn': (0.2, 2.5), 'Fe': (0.5, 5.0),
            'Cu': (0.1, 2.0), 'Mn': (1.0, 11.0),'B':  (0.08, 1.5),
        },
    },
}

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
# SUITABILITY SCORING
# ============================================================================

def param_score(value, opt_min, opt_max, acc_min, acc_max):
    """
    Score a single parameter value for a given crop requirement.

    Returns:
        1.0  — value in optimal range (maximum performance)
        0.5–1.0 — value in acceptable but not optimal (linear interpolation)
        0.0  — value outside acceptable range (crop failure or poor growth)
    """
    if opt_min <= value <= opt_max:
        return 1.0
    elif acc_min <= value < opt_min:
        # Below optimal but acceptable: linear 0.5 → 1.0
        span = opt_min - acc_min
        return 0.5 + 0.5 * (value - acc_min) / span if span > 0 else 0.5
    elif opt_max < value <= acc_max:
        # Above optimal but acceptable: linear 1.0 → 0.5
        span = acc_max - opt_max
        return 0.5 + 0.5 * (acc_max - value) / span if span > 0 else 0.5
    else:
        return 0.0


def crop_suitability(soil, crop_def):
    """
    Compute weighted suitability score [0, 1] for a crop given a soil profile.
    """
    score = 0.0
    for param in FEATURES:
        val = soil[param]
        opt   = crop_def['optimal'][param]
        acc   = crop_def['acceptable'][param]
        w     = crop_def['weights'][param]
        score += w * param_score(val, opt[0], opt[1], acc[0], acc[1])
    return score


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


def assign_crop(soil, noise_std=0.04):
    """
    Score all crops for a soil profile and return the best-fit crop.
    A small random noise is added to scores to simulate real-world variation
    and prevent the dataset from being perfectly rule-based.
    """
    scores = {}
    for crop_id, crop_def in CROPS.items():
        base_score = crop_suitability(soil, crop_def)
        # Add realistic noise (±4%) — real fields deviate from pure rules
        noise = np.random.normal(0, noise_std)
        scores[crop_id] = np.clip(base_score + noise, 0.0, 1.0)

    best_crop_id = max(scores, key=scores.get)
    return best_crop_id, scores[best_crop_id], scores


# ============================================================================
# MAIN GENERATOR
# ============================================================================

def generate_dataset(n_samples=5000, output_path='soil_data.csv'):
    """
    Generate a full labeled dataset for Burkina Faso crop recommendation.
    """
    print("=" * 60)
    print("BURKINA FASO CROP RECOMMENDATION DATASET GENERATOR")
    print(f"Target: {n_samples} samples — Sahel / Burkina Faso")
    print("=" * 60)

    # Determine how many samples per soil type
    soil_type_names = list(SOIL_TYPES.keys())
    proportions     = [SOIL_TYPES[t]['proportion'] for t in soil_type_names]
    counts          = [int(p * n_samples) for p in proportions]
    # Assign remainder to most common type
    counts[0] += n_samples - sum(counts)

    print("\nSoil type distribution:")
    for name, count in zip(soil_type_names, counts):
        print(f"  {name:<12} {count:>5} samples  — {SOIL_TYPES[name]['description']}")

    rows = []
    for soil_type_name, count in zip(soil_type_names, counts):
        soil_type_def = SOIL_TYPES[soil_type_name]
        for _ in range(count):
            profile = generate_profile(soil_type_def)
            crop_id, top_score, all_scores = assign_crop(profile)
            row = dict(profile)
            row['Output']    = crop_id
            row['crop_name'] = CROPS[crop_id]['name']
            row['soil_type'] = soil_type_name
            row['confidence'] = round(top_score, 3)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # ── Class distribution report ─────────────────────────────────
    print("\nCrop distribution (recommended crop counts):")
    dist = df['crop_name'].value_counts().sort_index()
    for crop_name, count in df.groupby('Output')['crop_name'].first().items():
        n = (df['Output'] == crop_name).sum()
    for crop_id in sorted(CROPS.keys()):
        name  = CROPS[crop_id]['name']
        count = (df['Output'] == crop_id).sum()
        bar   = '█' * int(count / n_samples * 40)
        pct   = count / n_samples * 100
        print(f"  {crop_id}  {name:<12} {count:>5} ({pct:4.1f}%)  {bar}")

    # ── Parameter stats ───────────────────────────────────────────
    print("\nSoil parameter statistics:")
    stats = df[FEATURES].describe().loc[['mean', 'std', 'min', 'max']]
    print(stats.round(2).to_string())

    # ── Save ─────────────────────────────────────────────────────
    # Save full version with metadata
    full_path = output_path.replace('.csv', '_full.csv')
    df.to_csv(full_path, index=False)
    print(f"\nFull dataset (with soil_type, confidence) → {full_path}")

    # Save training version (only features + Output, as train_model.py expects)
    train_cols = FEATURES + ['Output']
    df[train_cols].to_csv(output_path, index=False)
    print(f"Training dataset → {output_path}")
    print(f"\nTotal samples: {len(df)}")
    print(f"Classes: {df['Output'].nunique()} crops")
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
