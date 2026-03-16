import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { format } from 'date-fns';
import SOIL_PARAMETERS from '../constants/soilParameters';
import FERTILIZERS from '../constants/fertilizers';
import { TRANSLATIONS, translateCrop } from '../constants/translations';

export const calculateFertilizerRecommendations = (soilData) => {
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

export const exportToPDF = (soilData, result, language) => {
  const doc = new jsPDF();
  const t = TRANSLATIONS[language];

  // Header
  doc.setFillColor(0, 229, 255);
  doc.rect(0, 0, 210, 40, 'F');
  doc.setTextColor(10, 25, 41);
  doc.setFontSize(24);
  doc.setFont('helvetica', 'bold');
  doc.text('AgriTech Soil Analyzer', 105, 20, { align: 'center' });
  doc.setFontSize(12);
  doc.text(language === 'fr' ? 'Rapport Professionnel de Fertilite du Sol' : 'Professional Soil Fertility Report', 105, 30, { align: 'center' });

  doc.setTextColor(100);
  doc.setFontSize(10);
  doc.text(`${language === 'fr' ? 'Date du rapport' : 'Report Date'}: ${format(new Date(), 'MMMM dd, yyyy')}`, 105, 50, { align: 'center' });

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

  autoTable(doc, {
    startY: 102,
    head: [[language === 'fr' ? 'Paramètre' : 'Parameter', language === 'fr' ? 'Valeur' : 'Value', language === 'fr' ? 'Unité' : 'Unit', language === 'fr' ? 'Statut' : 'Status']],
    body: Object.entries(soilData).map(([key, value]) => {
      const param = SOIL_PARAMETERS.find(p => p.id === key);
      const numValue = parseFloat(value as string);
      const isOptimal = numValue >= param.optimal.min && numValue <= param.optimal.max;
      return [t.parameters[key].label, value, param.unit || '-', isOptimal ? 'Optimal' : (language === 'fr' ? 'À améliorer' : 'Needs Attention')];
    }),
    theme: 'grid',
    headStyles: { fillColor: [0, 229, 255], textColor: [10, 25, 41] }
  });

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
        .sort((a, b) => Math.abs(b[1] as number) - Math.abs(a[1] as number))
        .map(([key, val]) => [
          t.parameters[key]?.label || key,
          (val as number).toFixed(4),
          (val as number) > 0 ? (language === 'fr' ? '↑ Positif' : '↑ Positive') : (val as number) < 0 ? (language === 'fr' ? '↓ Négatif' : '↓ Negative') : '— Neutre'
        ]),
      theme: 'grid',
      headStyles: { fillColor: [0, 229, 255], textColor: [10, 25, 41] }
    });
  }

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

  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setTextColor(150);
    doc.text(`Page ${i} / ${pageCount}`, 105, 285, { align: 'center' });
    doc.text(language === 'fr' ? 'Généré par AgriTech Soil Analyzer' : 'Generated by AgriTech Soil Analyzer', 105, 290, { align: 'center' });
  }

  doc.save(`Soil-Analysis-Report-${format(new Date(), 'yyyy-MM-dd')}.pdf`);
};
