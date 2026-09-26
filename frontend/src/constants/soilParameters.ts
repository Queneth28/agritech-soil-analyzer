// Optimal ranges mirror SOIL_OPTIMAL_RANGES in backend/app.py (West African / Sahelian soils)
const SOIL_PARAMETERS = [
  { id: 'N', unit: 'mg/kg', range: { min: 0, max: 400 }, optimal: { min: 100, max: 250 }, placeholder: 'ex: 60-250', isPrimary: true },
  { id: 'P', unit: 'mg/kg', range: { min: 0, max: 60 }, optimal: { min: 8, max: 25 }, placeholder: 'ex: 3-15', isPrimary: true },
  { id: 'K', unit: 'mg/kg', range: { min: 0, max: 1000 }, optimal: { min: 150, max: 500 }, placeholder: 'ex: 120-450', isPrimary: true },
  { id: 'pH', unit: '', range: { min: 0, max: 14 }, optimal: { min: 5.8, max: 7.0 }, placeholder: 'ex: 5.5-7.2', isPrimary: true },
  { id: 'EC', unit: 'dS/m', range: { min: 0, max: 2 }, optimal: { min: 0, max: 0.8 }, placeholder: 'ex: 0.1-0.7' },
  { id: 'OC', unit: '%', range: { min: 0, max: 5 }, optimal: { min: 0.8, max: 2.0 }, placeholder: 'ex: 0.2-1.2' },
  { id: 'S', unit: 'mg/kg', range: { min: 0, max: 50 }, optimal: { min: 8, max: 25 }, placeholder: 'ex: 4-20' },
  { id: 'Zn', unit: 'mg/kg', range: { min: 0, max: 2 }, optimal: { min: 0.5, max: 2.0 }, placeholder: 'ex: 0.2-1.2' },
  { id: 'Fe', unit: 'mg/kg', range: { min: 0, max: 5 }, optimal: { min: 0.8, max: 4.0 }, placeholder: 'ex: 0.8-3.5' },
  { id: 'Cu', unit: 'mg/kg', range: { min: 0, max: 5 }, optimal: { min: 0.3, max: 1.5 }, placeholder: 'ex: 0.3-1.2' },
  { id: 'Mn', unit: 'mg/kg', range: { min: 0, max: 20 }, optimal: { min: 2, max: 8 }, placeholder: 'ex: 1.5-7' },
  { id: 'B', unit: 'mg/kg', range: { min: 0, max: 5 }, optimal: { min: 0.2, max: 1.0 }, placeholder: 'ex: 0.1-0.6' }
];

export default SOIL_PARAMETERS;
