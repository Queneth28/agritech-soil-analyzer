"""
Crop agronomic profiles — Burkina Faso / Sahel

Single source of truth for crop soil requirements. Shared by:
  - generate_burkina_dataset.py  (labels the training data)
  - app.py                        (rule-based suitability, calendar, crop list)

Keeping both on the same thresholds guarantees that the ML model's
recommendation and the rule-based crop ranking never contradict each other.
"""

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
# CROP CALENDAR & DISPLAY INFO — Burkina Faso rainfed season
# ============================================================================
# The single rainy season runs roughly May/June → September/October, starting
# earlier in the Sud-Ouest (~900–1200 mm) and later in the Sahel (<600 mm).
# Sowing windows follow INERA / Ministère de l'Agriculture recommendations:
# sow after the first useful rain (≥20 mm), late sowing costs yield.
# cycle_months = typical sowing-to-harvest duration for common local varieties.

CROP_INFO = {
    'Sorgho':   {'category': 'Cereal',    'icon': '🌾', 'seasons': ['May', 'Jun', 'Jul'], 'cycle_months': 4,
                 'description': 'Main Sahelian staple, tolerates dry spells and moderately poor soils'},
    'Mil':      {'category': 'Cereal',    'icon': '🌿', 'seasons': ['Jun', 'Jul'],        'cycle_months': 3,
                 'description': 'Most drought-tolerant cereal, suited to sandy and degraded soils'},
    'Niébé':    {'category': 'Legume',    'icon': '🫘', 'seasons': ['Jul', 'Aug'],        'cycle_months': 3,
                 'description': 'Nitrogen-fixing legume, protein source, good rotation partner for cereals'},
    'Arachide': {'category': 'Legume',    'icon': '🥜', 'seasons': ['Jun', 'Jul'],        'cycle_months': 4,
                 'description': 'Food and cash legume, needs slightly acidic, well-drained soil'},
    'Maïs':     {'category': 'Cereal',    'icon': '🌽', 'seasons': ['Jun', 'Jul'],        'cycle_months': 4,
                 'description': 'High-yield cereal, needs fertile soil and good rainfall or bas-fond'},
    'Coton':    {'category': 'Cash Crop', 'icon': '🌱', 'seasons': ['May', 'Jun'],        'cycle_months': 5,
                 'description': 'Main cash crop, heavy K and B feeder, sow before end of June'},
    'Sésame':   {'category': 'Cash Crop', 'icon': '🌻', 'seasons': ['Jul'],               'cycle_months': 3,
                 'description': 'Export oilseed, tolerates poor soils but not waterlogging'},
    'Soja':     {'category': 'Legume',    'icon': '🫘', 'seasons': ['Jun', 'Jul'],        'cycle_months': 4,
                 'description': 'Protein legume, strict pH requirements, best in the south-west'},
    'Riz':      {'category': 'Cereal',    'icon': '🌾', 'seasons': ['Jul', 'Aug', 'Jan', 'Feb'], 'cycle_months': 4,
                 'description': 'Bas-fond or irrigated only; Jan–Feb sowing is dry-season irrigated rice'},
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
