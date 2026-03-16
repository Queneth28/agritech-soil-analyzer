const SOIL_PARAMETERS = [
  { id: 'N', unit: 'mg/kg', range: { min: 0, max: 400 }, optimal: { min: 150, max: 300 }, placeholder: 'ex: 138-270', isPrimary: true },
  { id: 'P', unit: 'mg/kg', range: { min: 0, max: 15 }, optimal: { min: 7, max: 10 }, placeholder: 'ex: 6.8-9.9', isPrimary: true },
  { id: 'K', unit: 'mg/kg', range: { min: 0, max: 1000 }, optimal: { min: 400, max: 700 }, placeholder: 'ex: 338-718', isPrimary: true },
  { id: 'pH', unit: '', range: { min: 0, max: 14 }, optimal: { min: 6.5, max: 7.5 }, placeholder: 'ex: 7.46-7.64', isPrimary: true },
  { id: 'EC', unit: 'dS/m', range: { min: 0, max: 2 }, optimal: { min: 0.4, max: 0.8 }, placeholder: 'ex: 0.40-0.75' },
  { id: 'OC', unit: '%', range: { min: 0, max: 5 }, optimal: { min: 0.8, max: 2.0 }, placeholder: 'ex: 0.7-1.11' },
  { id: 'S', unit: 'mg/kg', range: { min: 0, max: 50 }, optimal: { min: 10, max: 30 }, placeholder: 'ex: 5.9-26.0' },
  { id: 'Zn', unit: 'mg/kg', range: { min: 0, max: 2 }, optimal: { min: 0.2, max: 0.5 }, placeholder: 'ex: 0.24-0.34' },
  { id: 'Fe', unit: 'mg/kg', range: { min: 0, max: 5 }, optimal: { min: 0.3, max: 1.0 }, placeholder: 'ex: 0.31-0.86' },
  { id: 'Cu', unit: 'mg/kg', range: { min: 0, max: 5 }, optimal: { min: 0.5, max: 2.0 }, placeholder: 'ex: 0.77-1.69' },
  { id: 'Mn', unit: 'mg/kg', range: { min: 0, max: 20 }, optimal: { min: 2, max: 10 }, placeholder: 'ex: 2.43-8.71' },
  { id: 'B', unit: 'mg/kg', range: { min: 0, max: 5 }, optimal: { min: 0.5, max: 3.0 }, placeholder: 'ex: 0.11-2.29' }
];

export default SOIL_PARAMETERS;
