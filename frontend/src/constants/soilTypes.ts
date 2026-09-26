// Typical soil-test values per Burkina Faso soil type (means from
// backend/generate_burkina_dataset.py SOIL_TYPES). Used to fill values the
// lab did not measure; the result then flags them as estimated.
export const SOIL_TYPES = ['unknown', 'Lixisol', 'Luvisol', 'Bas-fond', 'Lithosol', 'Vertisol'] as const;
export type SoilType = typeof SOIL_TYPES[number];

export const TYPICAL_VALUES: Record<SoilType, Record<string, number>> = {
  Lixisol:  { N: 90,  P: 4.5, K: 175, pH: 6.3, EC: 0.28, OC: 0.35, S: 8,  Zn: 0.38, Fe: 1.4, Cu: 0.45, Mn: 2.8, B: 0.18 },
  Luvisol:  { N: 125, P: 6.5, K: 255, pH: 6.5, EC: 0.34, OC: 0.58, S: 12, Zn: 0.62, Fe: 2.1, Cu: 0.72, Mn: 4.2, B: 0.32 },
  'Bas-fond': { N: 165, P: 8.2, K: 330, pH: 6.2, EC: 0.5, OC: 0.88, S: 16, Zn: 0.82, Fe: 2.9, Cu: 0.95, Mn: 5.8, B: 0.42 },
  Lithosol: { N: 62,  P: 2.8, K: 128, pH: 5.9, EC: 0.22, OC: 0.2,  S: 5,  Zn: 0.22, Fe: 0.9, Cu: 0.28, Mn: 1.7, B: 0.11 },
  Vertisol: { N: 145, P: 7.2, K: 390, pH: 7.2, EC: 0.62, OC: 0.78, S: 15, Zn: 0.72, Fe: 2.6, Cu: 0.85, Mn: 5.2, B: 0.38 },
  // Weighted by the share of each soil type in Burkina Faso
  unknown:  { N: 109, P: 5.4, K: 222, pH: 6.3, EC: 0.34, OC: 0.49, S: 10, Zn: 0.5,  Fe: 1.8, Cu: 0.59, Mn: 3.6, B: 0.25 },
};
