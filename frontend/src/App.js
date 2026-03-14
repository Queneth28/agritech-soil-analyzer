import './App.css';
import React, { useState, useCallback, useMemo, memo, useEffect, useRef } from 'react';
import {
  FlaskConical, Loader2, AlertCircle, CheckCircle2,
  Download, Globe, History, Trash2, Calendar, Sun, Moon,
  TrendingUp, TrendingDown, Info, X, BarChart3, Leaf, Sprout
} from 'lucide-react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { format } from 'date-fns';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, Legend, ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  Tooltip, CartesianGrid, Cell
} from 'recharts';

// ============================================================================
// API CONFIGURATION — auto-detects network so it works on any device
// ============================================================================

const API_BASE = process.env.REACT_APP_API_URL || `http://${window.location.hostname}:5000`;

const API_ENDPOINTS = {
  predict: `${API_BASE}/api/predict`,
  crops: `${API_BASE}/api/crops`,
  health: `${API_BASE}/api/health`,
  modelInfo: `${API_BASE}/api/model/info`,
  compare: `${API_BASE}/api/compare`,
  seasonalCalendar: `${API_BASE}/api/seasonal-calendar`,
  soilHealthScore: `${API_BASE}/api/soil-health-score`,
};

// ============================================================================
// FERTILIZER DATABASE
// ============================================================================

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

// ============================================================================
// FULL TRANSLATIONS — English + French (ALL sections translated)
// ============================================================================

const TRANSLATIONS = {
  en: {
    appTitle: 'AgriTech Soil Analyzer',
    appSubtitle: 'AI-Powered Land Suitability Assessment System',
    progressLabel: 'Data Entry Progress',
    panelTitle: 'Soil Nutrients',
    analyzeButton: 'Analyze Soil Fertility',
    analyzing: 'Analyzing...',
    emptyTitle: 'Ready for Analysis',
    emptyDescription: 'Complete all 12 soil nutrient measurements to receive comprehensive AI-powered fertility analysis and personalized crop cultivation suggestions',
    suitabilityLabel: 'Land Suitability Classification',
    confidence: 'Confidence',
    score: 'Score',
    recommendedCrops: 'Recommended Crops',
    cropsDescription: 'Best cultivation options based on your soil analysis',
    high: 'High',
    medium: 'Medium',
    low: 'Low',
    exportPDF: 'Export PDF Report',
    viewHistory: 'Analysis History',
    hideHistory: 'Hide History',
    fertilizerTitle: 'Fertilizer Recommendations',
    fertilizerDescription: 'Specific fertilizers needed to optimize your soil',
    noHistory: 'No analysis history yet',
    deleteAnalysis: 'Delete',
    soilHealth: 'Soil Health Score',
    soilHealthDesc: 'Weighted index based on all nutrient levels',
    nutrientChart: 'Nutrient Status vs Optimal Levels',
    shapTitle: 'Why This Classification?',
    shapDescription: 'SHAP values show how each nutrient influenced the prediction',
    seasonalTitle: 'Planting Calendar',
    seasonalDescription: 'Optimal planting months for recommended crops',
    themeLight: 'Light Mode',
    themeDark: 'Dark Mode',
    primaryBadge: 'Primary',
    apiError: 'Failed to analyze soil. Please ensure the backend server is running.',
    analysisComplete: 'Analysis complete',
    analysisDeleted: 'Analysis deleted',
    loadedFromHistory: 'Loaded from history',
    analysisFailed: 'Analysis failed',
    // Fertilizer section
    deficiency: 'Deficiency',
    composition: 'Composition',
    dosage: 'Dosage',
    type: 'Type',
    estCost: 'Est. Cost',
    applicationTimingLabel: 'Application Timing:',
    applicationTimingText: 'Apply fertilizers 2-3 weeks before planting. Split nitrogen applications for long-season crops.',
    // Crop section
    plantLabel: 'Plant',
    // SHAP section
    positiveInfluence: 'Positive influence',
    negativeInfluence: 'Negative influence',
    // Priority
    priorityHigh: 'High',
    priorityMedium: 'Medium',
    priorityExcellent: 'Excellent',
    priorityGood: 'Good',
    priorityFair: 'Fair',
    // Crop names
    cropNames: {},
    categoryNames: {},
    // Validation
    validationErrors: {
      invalidNumber: 'Must be a valid number',
      minValue: 'Minimum value is',
      maxValue: 'Maximum value is'
    },
    parameters: {
      N: { label: 'Nitrogen (N)', description: 'Essential macronutrient for vegetative growth and chlorophyll production' },
      P: { label: 'Phosphorus (P)', description: 'Critical for energy transfer, root development, and flowering' },
      K: { label: 'Potassium (K)', description: 'Regulates water balance, disease resistance, and fruit quality' },
      pH: { label: 'pH Level', description: 'Measures soil acidity/alkalinity; affects nutrient availability' },
      EC: { label: 'Electrical Conductivity (EC)', description: 'Indicates soil salinity and dissolved salt concentration' },
      OC: { label: 'Organic Carbon (OC)', description: 'Indicates organic matter content and soil health' },
      S: { label: 'Sulfur (S)', description: 'Important for protein synthesis and enzyme activation' },
      Zn: { label: 'Zinc (Zn)', description: 'Essential micronutrient for enzyme function and hormone production' },
      Fe: { label: 'Iron (Fe)', description: 'Critical for chlorophyll synthesis and respiration' },
      Cu: { label: 'Copper (Cu)', description: 'Important for photosynthesis and reproductive growth' },
      Mn: { label: 'Manganese (Mn)', description: 'Activates enzymes and aids in chlorophyll formation' },
      B: { label: 'Boron (B)', description: 'Essential for cell wall formation and reproductive development' }
    },
  },
  fr: {
    appTitle: 'Analyseur de Sol AgriTech',
    appSubtitle: "Système d'Évaluation de l'Aptitude des Terres Alimenté par IA",
    progressLabel: 'Progression de la Saisie',
    panelTitle: 'Nutriments du Sol',
    analyzeButton: 'Analyser la Fertilité du Sol',
    analyzing: 'Analyse en cours...',
    emptyTitle: "Prêt pour l'Analyse",
    emptyDescription: "Complétez les 12 mesures de nutriments du sol pour recevoir une analyse complète de la fertilité alimentée par IA et des suggestions personnalisées de culture",
    suitabilityLabel: "Classification de l'Aptitude des Terres",
    confidence: 'Confiance',
    score: 'Score',
    recommendedCrops: 'Cultures Recommandées',
    cropsDescription: 'Meilleures options de culture basées sur votre analyse de sol',
    high: 'Élevée',
    medium: 'Moyenne',
    low: 'Faible',
    exportPDF: 'Exporter Rapport PDF',
    viewHistory: 'Historique des Analyses',
    hideHistory: "Masquer l'Historique",
    fertilizerTitle: "Recommandations d'Engrais",
    fertilizerDescription: 'Engrais spécifiques nécessaires pour optimiser votre sol',
    noHistory: "Aucun historique d'analyse",
    deleteAnalysis: 'Supprimer',
    soilHealth: 'Score de Santé du Sol',
    soilHealthDesc: 'Indice pondéré basé sur tous les niveaux de nutriments',
    nutrientChart: 'État des Nutriments vs Niveaux Optimaux',
    shapTitle: 'Pourquoi cette Classification ?',
    shapDescription: 'Les valeurs SHAP montrent comment chaque nutriment a influencé la prédiction',
    seasonalTitle: 'Calendrier de Plantation',
    seasonalDescription: 'Mois de plantation optimaux pour les cultures recommandées',
    themeLight: 'Mode Clair',
    themeDark: 'Mode Sombre',
    primaryBadge: 'Principal',
    apiError: "Échec de l'analyse du sol. Veuillez vous assurer que le serveur backend est en cours d'exécution.",
    analysisComplete: 'Analyse terminée',
    analysisDeleted: 'Analyse supprimée',
    loadedFromHistory: "Chargé depuis l'historique",
    analysisFailed: "Échec de l'analyse",
    // Fertilizer section
    deficiency: 'Carence',
    composition: 'Composition',
    dosage: 'Dosage',
    type: 'Type',
    estCost: 'Coût est.',
    applicationTimingLabel: "Période d'application :",
    applicationTimingText: "Appliquer les engrais 2 à 3 semaines avant la plantation. Fractionner les apports d'azote pour les cultures à cycle long.",
    // Crop section
    plantLabel: 'Planter',
    // SHAP section
    positiveInfluence: 'Influence positive',
    negativeInfluence: 'Influence négative',
    // Priority
    priorityHigh: 'Élevée',
    priorityMedium: 'Moyenne',
    priorityExcellent: 'Excellent',
    priorityGood: 'Bon',
    priorityFair: 'Moyen',
    // Crop names
    cropNames: {
      'Wheat': 'Blé', 'Rice': 'Riz', 'Maize (Corn)': 'Maïs', 'Barley': 'Orge',
      'Soybeans': 'Soja', 'Chickpeas': 'Pois chiches', 'Peas': 'Pois',
      'Tomatoes': 'Tomates', 'Potatoes': 'Pommes de terre', 'Onions': 'Oignons',
      'Carrots': 'Carottes', 'Lettuce': 'Laitue',
      'Strawberries': 'Fraises', 'Watermelon': 'Pastèque',
      'Cotton': 'Coton', 'Sunflower': 'Tournesol', 'Sugarcane': 'Canne à sucre',
      'Mint': 'Menthe', 'Basil': 'Basilic',
    },
    categoryNames: {
      'Cereal': 'Céréale', 'Legume': 'Légumineuse', 'Vegetable': 'Légume',
      'Fruit': 'Fruit', 'Cash Crop': 'Culture de rente', 'Herb': 'Herbe aromatique',
    },
    // Validation
    validationErrors: {
      invalidNumber: 'Doit être un nombre valide',
      minValue: 'La valeur minimale est',
      maxValue: 'La valeur maximale est'
    },
    parameters: {
      N: { label: 'Azote (N)', description: 'Macronutriment essentiel pour la croissance végétative et la production de chlorophylle' },
      P: { label: 'Phosphore (P)', description: "Essentiel pour le transfert d'énergie, le développement racinaire et la floraison" },
      K: { label: 'Potassium (K)', description: "Régule l'équilibre hydrique, la résistance aux maladies et la qualité des fruits" },
      pH: { label: 'Niveau de pH', description: "Mesure l'acidité/alcalinité du sol ; affecte la disponibilité des nutriments" },
      EC: { label: 'Conductivité Électrique (CE)', description: 'Indique la salinité du sol et la concentration en sels dissous' },
      OC: { label: 'Carbone Organique (CO)', description: 'Indique la teneur en matière organique et la santé du sol' },
      S: { label: 'Soufre (S)', description: "Important pour la synthèse des protéines et l'activation des enzymes" },
      Zn: { label: 'Zinc (Zn)', description: "Micronutriment essentiel pour la fonction enzymatique et la production d'hormones" },
      Fe: { label: 'Fer (Fe)', description: 'Essentiel pour la synthèse de la chlorophylle et la respiration' },
      Cu: { label: 'Cuivre (Cu)', description: 'Important pour la photosynthèse et la croissance reproductive' },
      Mn: { label: 'Manganèse (Mn)', description: 'Active les enzymes et aide à la formation de la chlorophylle' },
      B: { label: 'Bore (B)', description: 'Essentiel pour la formation de la paroi cellulaire et le développement reproductif' }
    },
  }
};

const BACKEND_TRANSLATIONS = {
  en: {
    summaries: {
      'High': 'Soil analysis indicates excellent conditions for cultivation. Outstanding potential with optimal nutrient levels.',
      'Medium': 'Soil analysis indicates medium suitability. Good potential with some areas for improvement.',
      'Low': 'Soil analysis indicates low suitability. Significant amendments needed to improve fertility.'
    }
  },
  fr: {
    summaries: {
      'High': "L'analyse du sol indique d'excellentes conditions pour la culture. Potentiel exceptionnel avec des niveaux de nutriments optimaux.",
      'Medium': "L'analyse du sol indique une aptitude moyenne pour la culture. Bon potentiel avec quelques domaines à améliorer.",
      'Low': "L'analyse du sol indique une faible aptitude pour la culture. Des amendements importants sont nécessaires pour améliorer la fertilité."
    }
  }
};

// ============================================================================
// CONFIGURATION
// ============================================================================

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

// ============================================================================
// HELPER: Translate crop name, category, priority
// ============================================================================

const translateCrop = (crop, t) => ({
  ...crop,
  name: t.cropNames[crop.name] || crop.name,
  category: t.categoryNames[crop.category] || crop.category,
  priority: t[`priority${crop.priority}`] || crop.priority,
});

// ============================================================================
// HELPER: Calculate fertilizer recommendations
// ============================================================================

const calculateFertilizerRecommendations = (soilData) => {
  const recommendations = [];

  const checkMacro = (id, fertArray, contentKey) => {
    const value = parseFloat(soilData[id]);
    const param = SOIL_PARAMETERS.find(p => p.id === id);
    const optimal = (param.optimal.min + param.optimal.max) / 2;
    if (value < optimal * 0.8) {
      const deficiency = optimal - value;
      const fert = fertArray[0];
      const kgPerHectare = Math.round((deficiency / fert[contentKey]) * 100);
      recommendations.push({
        nutrientId: id,
        fertilizer: fert.name,
        composition: fert.composition,
        dosage: `${kgPerHectare} kg/hectare`,
        cost: `$${(kgPerHectare * fert.cost).toFixed(2)}/hectare`,
        type: fert.type,
        priority: 'High'
      });
    }
  };

  checkMacro('N', FERTILIZERS.N, 'nContent');
  checkMacro('P', FERTILIZERS.P, 'pContent');
  checkMacro('K', FERTILIZERS.K, 'kContent');

  ['Zn', 'Fe', 'Cu', 'Mn', 'B'].forEach(id => {
    const value = parseFloat(soilData[id]);
    const param = SOIL_PARAMETERS.find(p => p.id === id);
    const optimal = (param.optimal.min + param.optimal.max) / 2;
    if (value < optimal * 0.7) {
      const fert = FERTILIZERS.micronutrients.find(f => f.nutrient === id);
      if (fert) {
        recommendations.push({
          nutrientId: id,
          fertilizer: fert.name,
          composition: `${fert.content}% ${id}`,
          dosage: '5-10 kg/hectare',
          cost: `$${(7.5 * fert.cost).toFixed(2)}/hectare`,
          type: 'chemical',
          priority: 'Medium'
        });
      }
    }
  });

  return recommendations;
};

// ============================================================================
// HELPER: Export PDF — FIXED with autoTable()
// ============================================================================

const exportToPDF = (soilData, result, language) => {
  const doc = new jsPDF();
  const t = TRANSLATIONS[language];

  // Header
  doc.setFillColor(0, 229, 255);
  doc.rect(0, 0, 210, 40, 'F');
  doc.setTextColor(10, 25, 41);
  doc.setFontSize(24);
  doc.setFont(undefined, 'bold');
  doc.text('AgriTech Soil Analyzer', 105, 20, { align: 'center' });
  doc.setFontSize(12);
  doc.text(language === 'fr' ? 'Rapport Professionnel de Fertilite du Sol' : 'Professional Soil Fertility Report', 105, 30, { align: 'center' });

  doc.setTextColor(100);
  doc.setFontSize(10);
  doc.text(`${language === 'fr' ? 'Date du rapport' : 'Report Date'}: ${format(new Date(), 'MMMM dd, yyyy')}`, 105, 50, { align: 'center' });

  // Health Score
  if (result.soil_health_score) {
    doc.setTextColor(10, 25, 41);
    doc.setFontSize(14);
    doc.text(`${t.soilHealth}: ${result.soil_health_score.overall_score}/100 (${result.soil_health_score.grade})`, 20, 62);
  }

  doc.setFontSize(16);
  doc.text(t.suitabilityLabel, 20, 75);
  doc.setFontSize(12);
  doc.text(`Classification: ${result.suitability}`, 20, 85);
  doc.text(`${t.confidence}: ${result.confidence}`, 20, 92);

  // Soil parameters table
  autoTable(doc, {
    startY: 102,
    head: [[language === 'fr' ? 'Paramètre' : 'Parameter', language === 'fr' ? 'Valeur' : 'Value', language === 'fr' ? 'Unité' : 'Unit', language === 'fr' ? 'Statut' : 'Status']],
    body: Object.entries(soilData).map(([key, value]) => {
      const param = SOIL_PARAMETERS.find(p => p.id === key);
      const numValue = parseFloat(value);
      const isOptimal = numValue >= param.optimal.min && numValue <= param.optimal.max;
      return [t.parameters[key].label, value, param.unit || '-', isOptimal ? (language === 'fr' ? 'Optimal' : 'Optimal') : (language === 'fr' ? 'À améliorer' : 'Needs Attention')];
    }),
    theme: 'grid',
    headStyles: { fillColor: [0, 229, 255], textColor: [10, 25, 41] }
  });

  // SHAP page
  if (result.shap_explanation && Object.keys(result.shap_explanation).length > 0) {
    doc.addPage();
    doc.setFontSize(16);
    doc.setTextColor(10, 25, 41);
    doc.text(t.shapTitle, 20, 20);
    doc.setFontSize(10);
    doc.text(t.shapDescription, 20, 28);

    autoTable(doc, {
      startY: 36,
      head: [[language === 'fr' ? 'Nutriment' : 'Nutrient', 'SHAP', language === 'fr' ? 'Direction' : 'Direction']],
      body: Object.entries(result.shap_explanation)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .map(([key, val]) => [
          t.parameters[key]?.label || key,
          val.toFixed(4),
          val > 0 ? (language === 'fr' ? '↑ Positif' : '↑ Positive') : val < 0 ? (language === 'fr' ? '↓ Négatif' : '↓ Negative') : '— Neutre'
        ]),
      theme: 'grid',
      headStyles: { fillColor: [0, 229, 255], textColor: [10, 25, 41] }
    });
  }

  // Fertilizers page
  const fertilizers = calculateFertilizerRecommendations(soilData);
  if (fertilizers.length > 0) {
    doc.addPage();
    doc.setFontSize(16);
    doc.setTextColor(10, 25, 41);
    doc.text(t.fertilizerTitle, 20, 20);

    autoTable(doc, {
      startY: 30,
      head: [[language === 'fr' ? 'Nutriment' : 'Nutrient', language === 'fr' ? 'Engrais' : 'Fertilizer', t.dosage, t.estCost, language === 'fr' ? 'Priorité' : 'Priority']],
      body: fertilizers.map(f => [
        t.parameters[f.nutrientId]?.label || f.nutrientId,
        f.fertilizer, f.dosage, f.cost,
        f.priority === 'High' ? t.priorityHigh : t.priorityMedium
      ]),
      theme: 'grid',
      headStyles: { fillColor: [0, 229, 255], textColor: [10, 25, 41] }
    });
  }

  // Crops page
  if (result.recommendedCrops?.length > 0) {
    doc.addPage();
    doc.setFontSize(16);
    doc.setTextColor(10, 25, 41);
    doc.text(t.recommendedCrops, 20, 20);

    autoTable(doc, {
      startY: 30,
      head: [[language === 'fr' ? 'Culture' : 'Crop', language === 'fr' ? 'Catégorie' : 'Category', language === 'fr' ? 'Aptitude' : 'Suitability', language === 'fr' ? 'Priorité' : 'Priority', language === 'fr' ? 'Plantation' : 'Planting']],
      body: result.recommendedCrops.slice(0, 10).map(crop => {
        const tc = translateCrop(crop, t);
        return [tc.name, tc.category, `${tc.suitabilityScore}%`, tc.priority, (crop.plantingSeasons || []).join(', ')];
      }),
      theme: 'grid',
      headStyles: { fillColor: [0, 229, 255], textColor: [10, 25, 41] }
    });
  }

  // Footer on all pages
  const pageCount = doc.internal.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setTextColor(150);
    doc.text(`Page ${i} / ${pageCount}`, 105, 285, { align: 'center' });
    doc.text(language === 'fr' ? 'Généré par AgriTech Soil Analyzer' : 'Generated by AgriTech Soil Analyzer', 105, 290, { align: 'center' });
  }

  doc.save(`Soil-Analysis-Report-${format(new Date(), 'yyyy-MM-dd')}.pdf`);
};

// ============================================================================
// TOAST NOTIFICATION SYSTEM
// ============================================================================

const ToastContainer = ({ toasts, removeToast }) => (
  <div className="toast-container" role="alert" aria-live="polite">
    {toasts.map(toast => (
      <div key={toast.id} className={`toast toast-${toast.type}`}>
        <div className="toast-content">
          {toast.type === 'success' && <CheckCircle2 size={18} />}
          {toast.type === 'error' && <AlertCircle size={18} />}
          {toast.type === 'info' && <Info size={18} />}
          <span>{toast.message}</span>
        </div>
        <button onClick={() => removeToast(toast.id)} className="toast-close" aria-label="Dismiss">
          <X size={14} />
        </button>
      </div>
    ))}
  </div>
);

// ============================================================================
// SOIL HEALTH GAUGE
// ============================================================================

const SoilHealthGauge = ({ healthData, t }) => {
  if (!healthData) return null;
  const { overall_score, grade } = healthData;
  const circumference = 2 * Math.PI * 60;
  const offset = circumference * (1 - overall_score / 100);
  const colors = { A: '#00E5FF', B: '#4DD0E1', C: '#FFB300', D: '#ef4444' };
  const color = colors[grade] || '#80DEEA';

  return (
    <div className="result-card" role="region" aria-label={t.soilHealth}>
      <h3 className="card-title"><Leaf size={22} style={{ color }} />{t.soilHealth}</h3>
      <p className="card-description">{t.soilHealthDesc}</p>
      <div className="health-gauge-container">
        <svg width="160" height="160" viewBox="0 0 140 140" aria-hidden="true">
          <circle cx="70" cy="70" r="60" fill="none" stroke="rgba(0,229,255,0.1)" strokeWidth="10" />
          <circle cx="70" cy="70" r="60" fill="none" stroke={color} strokeWidth="10"
            strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round"
            transform="rotate(-90 70 70)" style={{ transition: 'stroke-dashoffset 1s ease-out' }} />
        </svg>
        <div className="health-gauge-text">
          <span className="health-gauge-score" style={{ color }}>{Math.round(overall_score)}</span>
          <span className="health-gauge-label">/ 100</span>
        </div>
      </div>
      <div className="health-grade" style={{ background: `${color}22`, borderColor: `${color}44` }}>
        <span style={{ color }}>Grade {grade}</span>
      </div>
    </div>
  );
};

// ============================================================================
// SHAP EXPLANATION
// ============================================================================

const ShapExplanation = ({ shapData, t }) => {
  if (!shapData || Object.keys(shapData).length === 0) return null;
  const sorted = Object.entries(shapData).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8);
  const chartData = sorted.map(([key, val]) => ({
    name: t.parameters[key]?.label?.replace(/\s*\([^)]*\)/, '') || key,
    value: parseFloat(val.toFixed(3)),
    fill: val > 0 ? '#00E5FF' : '#ef4444'
  }));

  return (
    <div className="result-card" role="region" aria-label={t.shapTitle}>
      <h3 className="card-title"><BarChart3 size={22} style={{ color: '#00E5FF' }} />{t.shapTitle}</h3>
      <p className="card-description">{t.shapDescription}</p>
      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={chartData} layout="vertical" margin={{ left: 80, right: 20, top: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,229,255,0.1)" />
            <XAxis type="number" stroke="#80DEEA" tick={{ fill: '#80DEEA', fontSize: 12 }} />
            <YAxis dataKey="name" type="category" stroke="#80DEEA" tick={{ fill: '#B3E5FC', fontSize: 12 }} width={75} />
            <Tooltip contentStyle={{ background: 'rgba(26,35,50,0.95)', border: '1px solid rgba(0,229,255,0.3)', borderRadius: 8 }} labelStyle={{ color: '#E1F5FE' }} itemStyle={{ color: '#80DEEA' }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="shap-legend">
        <span className="shap-positive"><TrendingUp size={14} /> {t.positiveInfluence}</span>
        <span className="shap-negative"><TrendingDown size={14} /> {t.negativeInfluence}</span>
      </div>
    </div>
  );
};

// ============================================================================
// SEASONAL CALENDAR
// ============================================================================

const SeasonalCalendar = ({ crops, t }) => {
  if (!crops || crops.length === 0) return null;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const topCrops = crops.filter(c => c.plantingSeasons?.length > 0).slice(0, 6);
  if (topCrops.length === 0) return null;

  return (
    <div className="result-card" role="region" aria-label={t.seasonalTitle}>
      <h3 className="card-title"><Calendar size={22} style={{ color: '#00E5FF' }} />{t.seasonalTitle}</h3>
      <p className="card-description">{t.seasonalDescription}</p>
      <div className="seasonal-grid">
        <div className="seasonal-header">
          <div className="seasonal-crop-label"></div>
          {months.map(m => <div key={m} className="seasonal-month">{m}</div>)}
        </div>
        {topCrops.map((crop, i) => {
          const translatedName = t.cropNames[crop.name] || crop.name;
          return (
            <div key={i} className="seasonal-row">
              <div className="seasonal-crop-label" title={translatedName}>{translatedName}</div>
              {months.map(m => {
                const isPlanting = (crop.plantingSeasons || []).includes(m);
                return (
                  <div key={m} className={`seasonal-cell ${isPlanting ? 'planting' : ''}`}
                    title={isPlanting ? `${t.plantLabel} ${translatedName} — ${m}` : ''} />
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ============================================================================
// LOADING SKELETON
// ============================================================================

const ResultSkeleton = () => (
  <div className="results-panel" aria-busy="true">
    {[1, 2, 3].map(i => (
      <div key={i} className="result-card skeleton-card">
        <div className="skeleton-line skeleton-title" />
        <div className="skeleton-line skeleton-text" />
        <div className="skeleton-line skeleton-text short" />
      </div>
    ))}
  </div>
);

// ============================================================================
// PARAMETER INPUT
// ============================================================================

const ParameterInput = memo(({ param, value, onChange, error, isOptimal, t }) => {
  const handleChange = useCallback((e) => onChange(param.id, e.target.value), [param.id, onChange]);
  return (
    <div className="input-group">
      <label htmlFor={param.id} className="input-label">
        {t.parameters[param.id].label}
        {param.isPrimary && <span className="primary-badge">{t.primaryBadge}</span>}
      </label>
      <div style={{ position: 'relative' }}>
        <input id={param.id} type="number" step="0.01" min={param.range.min} max={param.range.max}
          value={value} onChange={handleChange} placeholder={param.placeholder}
          className={`input-field ${isOptimal ? 'optimal' : ''} ${error ? 'error' : ''}`}
          style={{ paddingRight: param.unit ? '4.5rem' : '1.125rem' }}
          aria-invalid={!!error} aria-describedby={error ? `${param.id}-error` : `${param.id}-desc`} />
        {param.unit && <span className="input-unit">{param.unit}</span>}
      </div>
      {error && <div id={`${param.id}-error`} className="error-message" role="alert"><AlertCircle size={14} /><span>{error}</span></div>}
      <p id={`${param.id}-desc`} className="input-description">{t.parameters[param.id].description}</p>
    </div>
  );
});
ParameterInput.displayName = 'ParameterInput';

// ============================================================================
// NUTRIENT RADAR CHART
// ============================================================================

const NutrientChart = ({ soilData, t }) => {
  const chartData = ['N', 'P', 'K', 'pH', 'OC', 'S'].map(id => {
    const param = SOIL_PARAMETERS.find(p => p.id === id);
    const value = parseFloat(soilData[id] || 0);
    const optimal = (param.optimal.min + param.optimal.max) / 2;
    return { nutrient: t.parameters[id].label.replace(/\s*\([^)]*\)/, ''), current: Math.round(Math.min(120, (value / optimal) * 100)), optimal: 100 };
  });
  return (
    <div className="result-card" role="region" aria-label={t.nutrientChart}>
      <h3 className="card-title"><BarChart3 size={22} style={{ color: '#00E5FF' }} />{t.nutrientChart}</h3>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <RadarChart data={chartData}>
            <PolarGrid stroke="rgba(0, 229, 255, 0.2)" />
            <PolarAngleAxis dataKey="nutrient" stroke="#80DEEA" />
            <PolarRadiusAxis angle={90} domain={[0, 120]} stroke="#80DEEA" />
            <Radar name={language === 'fr' ? 'Actuel' : 'Current'} dataKey="current" stroke="#00E5FF" fill="#00E5FF" fillOpacity={0.3} />
            <Radar name="Optimal" dataKey="optimal" stroke="#4DD0E1" fill="#4DD0E1" fillOpacity={0.1} />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// Need language at module scope for NutrientChart legend
let language = 'en';

// ============================================================================
// MAIN APP
// ============================================================================

function App() {
  const [lang, setLang] = useState('en');
  language = lang; // sync for NutrientChart

  const [theme, setTheme] = useState(() => {
    try { return window.localStorage.getItem('agritech_theme') || 'dark'; } catch { return 'dark'; }
  });
  const [soilData, setSoilData] = useState(() =>
    SOIL_PARAMETERS.reduce((acc, p) => ({ ...acc, [p.id]: '' }), {})
  );
  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [toasts, setToasts] = useState([]);
  const resultsRef = useRef(null);

  const t = TRANSLATIONS[lang];

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try { window.localStorage.setItem('agritech_theme', theme); } catch {}
  }, [theme]);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem('agritech_analysis_history');
      if (saved) setAnalysisHistory(JSON.parse(saved));
    } catch {}
  }, []);

  const addToast = useCallback((message, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000);
  }, []);

  const removeToast = useCallback((id) => setToasts(prev => prev.filter(t => t.id !== id)), []);

  const saveToHistory = useCallback((data, res) => {
    const entry = { id: Date.now(), date: new Date().toISOString(), soilData: data, result: res };
    const updated = [entry, ...analysisHistory].slice(0, 10);
    setAnalysisHistory(updated);
    try { window.localStorage.setItem('agritech_analysis_history', JSON.stringify(updated)); } catch {}
  }, [analysisHistory]);

  const deleteFromHistory = useCallback((id) => {
    const updated = analysisHistory.filter(a => a.id !== id);
    setAnalysisHistory(updated);
    try { window.localStorage.setItem('agritech_analysis_history', JSON.stringify(updated)); } catch {}
    addToast(t.analysisDeleted, 'info');
  }, [analysisHistory, addToast, t]);

  const loadFromHistory = useCallback((analysis) => {
    setSoilData(analysis.soilData);
    setResult(analysis.result);
    setShowHistory(false);
    addToast(t.loadedFromHistory, 'success');
  }, [addToast, t]);

  const validateField = useCallback((paramId, value) => {
    if (!value || value === '') return null;
    const param = SOIL_PARAMETERS.find(p => p.id === paramId);
    const num = parseFloat(value);
    if (isNaN(num)) return t.validationErrors.invalidNumber;
    if (num < param.range.min) return `${t.validationErrors.minValue} ${param.range.min}`;
    if (num > param.range.max) return `${t.validationErrors.maxValue} ${param.range.max}`;
    return null;
  }, [t]);

  const handleInputChange = useCallback((paramId, value) => {
    setSoilData(prev => ({ ...prev, [paramId]: value }));
    setErrors(prev => ({ ...prev, [paramId]: validateField(paramId, value) }));
    setApiError(null);
  }, [validateField]);

  const isOptimalValue = useCallback((paramId, value) => {
    if (!value) return false;
    const param = SOIL_PARAMETERS.find(p => p.id === paramId);
    const num = parseFloat(value);
    return num >= param.optimal.min && num <= param.optimal.max;
  }, []);

  const progress = useMemo(() => {
    const filled = Object.values(soilData).filter(v => v !== '').length;
    return Math.round((filled / SOIL_PARAMETERS.length) * 100);
  }, [soilData]);

  const isFormValid = useMemo(() =>
    Object.values(soilData).every(v => v !== '') && Object.values(errors).every(e => !e),
  [soilData, errors]);

  const translateSummary = useCallback((suitability) =>
    BACKEND_TRANSLATIONS[lang]?.summaries?.[suitability] || result?.summary || '',
  [lang, result]);

  const analyzeSoil = useCallback(async () => {
    if (!isFormValid) return;
    setLoading(true); setApiError(null); setResult(null);
    try {
      const payload = Object.entries(soilData).reduce((acc, [k, v]) => ({ ...acc, [k]: parseFloat(v) }), {});
      const response = await fetch(API_ENDPOINTS.predict, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `API Error: ${response.status}`);
      }
      const data = await response.json();
      setResult(data);
      saveToHistory(soilData, data);
      addToast(`${t.analysisComplete}: ${data.suitability}`, 'success');
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (error) {
      setApiError(error.message || t.apiError);
      addToast(t.analysisFailed, 'error');
    } finally {
      setLoading(false);
    }
  }, [soilData, isFormValid, t, saveToHistory, addToast]);

  const fertilizers = useMemo(() => {
    if (!result || !isFormValid) return [];
    return calculateFertilizerRecommendations(soilData);
  }, [result, soilData, isFormValid]);

  // ========================================================================
  return (
    <div className={`app-container theme-${theme}`}>
      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {/* Header */}
      <header className="header" role="banner">
        <div className="header-content">
          <div className="header-icon" aria-hidden="true">
            <img src="/app-icon.png" alt="" style={{ width: '32px', height: '32px', borderRadius: '6px' }}
              onError={(e) => { e.target.style.display = 'none'; e.target.parentNode.innerHTML = '<span style="font-size:1.5rem">🌱</span>'; }} />
          </div>
          <div className="header-title">
            <h1>{t.appTitle}</h1>
            <p className="header-subtitle">{t.appSubtitle}</p>
          </div>
          <div className="header-actions">
            <button onClick={() => setTheme(p => p === 'dark' ? 'light' : 'dark')} className="header-btn"
              aria-label={theme === 'dark' ? t.themeLight : t.themeDark}>
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <button onClick={() => setShowHistory(!showHistory)} className={`header-btn ${showHistory ? 'active' : ''}`} aria-expanded={showHistory}>
              <History size={18} /><span className="btn-text">{showHistory ? t.hideHistory : t.viewHistory}</span>
            </button>
            <button onClick={() => setLang(p => p === 'en' ? 'fr' : 'en')} className="header-btn accent">
              <Globe size={18} /><span className="btn-text">{lang === 'en' ? 'FR' : 'EN'}</span>
            </button>
          </div>
        </div>
      </header>

      {/* History */}
      {showHistory && (
        <div className="progress-container">
          <div className="result-card">
            <div className="card-title-row"><History size={24} style={{ color: '#00E5FF' }} /><h3 className="card-title">{t.viewHistory}</h3></div>
            {analysisHistory.length === 0 ? (
              <p className="empty-history">{t.noHistory}</p>
            ) : (
              <div className="history-list">
                {analysisHistory.map(a => (
                  <div key={a.id} className="history-item" onClick={() => loadFromHistory(a)} tabIndex={0} role="button"
                    onKeyDown={e => e.key === 'Enter' && loadFromHistory(a)}>
                    <div>
                      <div className="history-date"><Calendar size={16} style={{ color: '#00E5FF' }} /><span>{format(new Date(a.date), 'MMM dd, yyyy — HH:mm')}</span></div>
                      <div className="history-result">
                        {t.suitabilityLabel}: <strong>{a.result.suitability}</strong>
                        {a.result.soil_health_score && <span> | {t.soilHealth}: {Math.round(a.result.soil_health_score.overall_score)}/100</span>}
                      </div>
                    </div>
                    <button onClick={e => { e.stopPropagation(); deleteFromHistory(a.id); }} className="history-delete"><Trash2 size={16} /></button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Progress */}
      <div className="progress-container">
        <div className="progress-card" role="progressbar" aria-valuenow={progress} aria-valuemin="0" aria-valuemax="100">
          <div className="progress-header">
            <span className="progress-label">{t.progressLabel}</span>
            <span className="progress-percentage">{progress}%</span>
          </div>
          <div className="progress-bar-bg"><div className="progress-bar-fill" style={{ width: `${progress}%` }} /></div>
        </div>
      </div>

      {/* Main */}
      <div className="main-content">
        {/* Input Panel */}
        <div className="input-panel" role="form" aria-label={t.panelTitle}>
          <div className="panel-header"><FlaskConical /><h2 className="panel-title">{t.panelTitle}</h2></div>
          {SOIL_PARAMETERS.map(param => (
            <ParameterInput key={param.id} param={param} value={soilData[param.id]}
              onChange={handleInputChange} error={errors[param.id]}
              isOptimal={isOptimalValue(param.id, soilData[param.id])} t={t} />
          ))}
          <button onClick={analyzeSoil} disabled={!isFormValid || loading} className="analyze-button" aria-busy={loading}>
            {loading ? <><Loader2 className="animate-spin" size={20} />{t.analyzing}</> : <><FlaskConical size={20} />{t.analyzeButton}</>}
          </button>
          {apiError && <div className="error-message" style={{ marginTop: '1rem' }} role="alert"><AlertCircle size={16} /><span>{apiError}</span></div>}
        </div>

        {/* Results */}
        <div className="results-panel" ref={resultsRef}>
          {loading ? <ResultSkeleton /> : !result ? (
            <div className="result-card">
              <div className="empty-state">
                <div className="empty-icon" aria-hidden="true">
                  <img src="/app-icon.png" alt="" style={{ width: '56px', height: '56px', borderRadius: '12px' }}
                    onError={(e) => { e.target.style.display = 'none'; e.target.parentNode.innerHTML = '<span style="font-size:3rem">🌱</span>'; }} />
                </div>
                <h3 className="empty-title">{t.emptyTitle}</h3>
                <p className="empty-description">{t.emptyDescription}</p>
              </div>
            </div>
          ) : (
            <>
              {/* Export */}
              <button onClick={() => exportToPDF(soilData, result, lang)} className="export-button">
                <Download size={20} />{t.exportPDF}
              </button>

              {/* Health Gauge */}
              <SoilHealthGauge healthData={result.soil_health_score} t={t} />

              {/* Nutrient Chart */}
              <NutrientChart soilData={soilData} t={t} />

              {/* Suitability */}
              <div className={`result-card suitability-card ${result.suitability.toLowerCase()}`} role="region" aria-label={t.suitabilityLabel}>
                <div className="suitability-header">
                  <div>
                    <p className="suitability-level">{t.suitabilityLabel}</p>
                    <h2 className="suitability-value">{t[result.suitability.toLowerCase()] || result.suitability}</h2>
                  </div>
                  <CheckCircle2 size={48} aria-hidden="true" />
                </div>
                <p className="suitability-summary">{translateSummary(result.suitability)}</p>
                <div className="confidence-badges">
                  <div className="confidence-badge"><p>{t.confidence}</p><div className="value">{result.confidence}</div></div>
                  <div className="confidence-badge"><p>{t.score}</p><div className="value">{result.confidenceScore}/100</div></div>
                </div>
              </div>

              {/* SHAP */}
              <ShapExplanation shapData={result.shap_explanation} t={t} />

              {/* Fertilizers — FULLY TRANSLATED */}
              {fertilizers.length > 0 && (
                <div className="result-card" role="region" aria-label={t.fertilizerTitle}>
                  <h3 className="card-title">{t.fertilizerTitle}</h3>
                  <p className="card-description">{t.fertilizerDescription}</p>
                  <div className="fertilizer-grid">
                    {fertilizers.map((fert, i) => (
                      <div key={i} className={`fertilizer-card ${fert.priority === 'High' ? 'high' : ''}`}>
                        <div className="fertilizer-header">
                          <div>
                            <h4 className="fertilizer-name">{fert.fertilizer}</h4>
                            <p className="fertilizer-nutrient">{t.parameters[fert.nutrientId]?.label || fert.nutrientId} — {t.deficiency}</p>
                          </div>
                          <span className={`priority-badge ${fert.priority.toLowerCase()}`}>
                            {fert.priority === 'High' ? t.priorityHigh : t.priorityMedium}
                          </span>
                        </div>
                        <div className="fertilizer-details">
                          <div><span className="detail-label">{t.composition}:</span> <span className="detail-value">{fert.composition}</span></div>
                          <div><span className="detail-label">{t.dosage}:</span> <span className="detail-value">{fert.dosage}</span></div>
                          <div><span className="detail-label">{t.type}:</span> <span className="detail-value">{fert.type}</span></div>
                          <div><span className="detail-label">{t.estCost}:</span> <span className="detail-value accent">{fert.cost}</span></div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="fertilizer-tip">
                    <strong style={{ color: '#00E5FF' }}>{t.applicationTimingLabel}</strong> {t.applicationTimingText}
                  </div>
                </div>
              )}

              {/* Seasonal Calendar — TRANSLATED */}
              <SeasonalCalendar crops={result.recommendedCrops} t={t} />

              {/* Crops — FULLY TRANSLATED */}
              {result.recommendedCrops?.length > 0 && (
                <div className="result-card" role="region" aria-label={t.recommendedCrops}>
                  <h3 className="card-title"><Sprout size={22} style={{ color: '#00E5FF' }} />{t.recommendedCrops}</h3>
                  <p className="card-description">{t.cropsDescription}</p>
                  <div className="crops-grid">
                    {result.recommendedCrops.slice(0, 6).map((crop, i) => {
                      const tc = translateCrop(crop, t);
                      return (
                        <div key={i} className="crop-card">
                          <div className="crop-info">
                            <h4 className="crop-name">{tc.name}</h4>
                            <p className="crop-category">{tc.category}</p>
                            {crop.plantingSeasons?.length > 0 && (
                              <p className="crop-season">{t.plantLabel}: {crop.plantingSeasons.join(', ')}</p>
                            )}
                          </div>
                          <div className="crop-score">
                            <div className="crop-score-value">{tc.suitabilityScore}%</div>
                            <p className="crop-priority">{tc.priority}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;