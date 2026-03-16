const FERTILIZERS = {
  N: [
    { name: 'Urea', composition: '46-0-0', nContent: 46, cost: 0.5, type: 'chemical' },
    { name: 'Ammonium Sulfate', composition: '21-0-0', nContent: 21, cost: 0.4, type: 'chemical' },
    { name: 'Compost', composition: '2-1-1', nContent: 2, cost: 0.2, type: 'organic' },
  ],
  P: [
    { name: 'DAP', composition: '18-46-0', pContent: 46, cost: 0.6, type: 'chemical' },
    { name: 'SSP', composition: '0-16-0', pContent: 16, cost: 0.3, type: 'chemical' },
    { name: 'Bone Meal', composition: '3-15-0', pContent: 15, cost: 0.4, type: 'organic' },
  ],
  K: [
    { name: 'MOP (Potash)', composition: '0-0-60', kContent: 60, cost: 0.5, type: 'chemical' },
    { name: 'SOP', composition: '0-0-50', kContent: 50, cost: 0.7, type: 'chemical' },
    { name: 'Wood Ash', composition: '0-1-3', kContent: 3, cost: 0.1, type: 'organic' },
  ],
  micronutrients: [
    { name: 'Zinc Sulfate', nutrient: 'Zn', content: 36, cost: 1.5 },
    { name: 'Ferrous Sulfate', nutrient: 'Fe', content: 19, cost: 1.2 },
    { name: 'Copper Sulfate', nutrient: 'Cu', content: 25, cost: 2.0 },
    { name: 'Manganese Sulfate', nutrient: 'Mn', content: 32, cost: 1.8 },
    { name: 'Borax', nutrient: 'B', content: 11, cost: 2.5 },
  ]
};

export default FERTILIZERS;
