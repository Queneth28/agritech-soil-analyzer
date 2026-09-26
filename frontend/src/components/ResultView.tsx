import React, { useState } from 'react';
import { ArrowLeft, Download, Plus, ListChecks, Trophy, AlertTriangle, Leaf } from 'lucide-react';
import SoilHealthGauge from './SoilHealthGauge';
import NutrientChart from './NutrientChart';
import ShapExplanation from './ShapExplanation';
import SeasonalCalendar from './SeasonalCalendar';
import CropIcon from './CropIcon';
import { fmt, translateCrop, translateMonths } from '../constants/translations';

const SowCard = ({ result, t }) => {
  const crops = result.recommendedCrops || [];
  const best = crops.find(c => c.name === result.recommendedCrop);
  const name = t.cropNames[result.recommendedCrop] || result.recommendedCrop;
  const others = crops.filter(c => c.name !== result.recommendedCrop).slice(0, 3)
    .map(c => (t.cropNames[c.name] || c.name).toLowerCase()).join(', ');
  const estimated = result.estimatedFields || [];
  return (
    <section className="sow-card" aria-label={t.sowTitle}>
      <div className="sow-eyebrow">{t.sowTitle}</div>
      <div className="sow-main">
        <div className="sow-crop">
          <CropIcon name={result.recommendedCrop} size="lg" />
          <h2 className="sow-name">{name}</h2>
        </div>
        <div className="sow-score"><b>{result.confidenceScore}</b><span>{t.outOf}</span></div>
      </div>
      <p className="sow-detail">
        {best?.plantingSeasons?.length ? fmt(t.sowWindow, { months: translateMonths(best.plantingSeasons, t) }) + ' · ' : ''}
        {others && fmt(t.alsoPossible, { crops: others })}
      </p>
      {estimated.length > 0 && (
        <div className="estimate-flag">
          <AlertTriangle size={18} aria-hidden="true" />
          {fmt(t.estimatedWarning, { n: estimated.length, fields: estimated.join(', ') })}
        </div>
      )}
    </section>
  );
};

const Actions = ({ actions, t }) => {
  const [showAll, setShowAll] = useState(false);
  const shown = showAll ? actions : actions.slice(0, 3);
  return (
    <section className="card" aria-label={t.actionPlan}>
      <h3 className="card-title"><ListChecks size={24} aria-hidden="true" />{fmt(t.actionsBeforeSowing, { n: Math.min(3, actions.length) })}</h3>
      <ol className="action-list">
        {shown.map((a, i) => (
          <li key={a.code} className="action-item">
            <span className="action-num" aria-hidden="true">{i + 1}</span>
            <div><div className="action-title">{a.title}</div><div className="action-detail">{a.detail}</div></div>
          </li>
        ))}
      </ol>
      {actions.length > 3 && (
        <button className="btn btn-ghost btn-block more-toggle" onClick={() => setShowAll(v => !v)} aria-expanded={showAll}>
          {showAll ? t.showLess : fmt(t.moreActions, { n: actions.length })}
        </button>
      )}
    </section>
  );
};

const Ranking = ({ crops, t }) => (
  <section className="card" aria-label={t.cropRanking}>
    <h3 className="card-title"><Trophy size={24} aria-hidden="true" />{t.cropRanking}</h3>
    <div className="rank-list">
      {crops.map(crop => {
        const tc = translateCrop(crop, t);
        return (
          <div key={crop.name} className="rank-row">
            <CropIcon name={crop.name} />
            <div>
              <div className="rank-name">{tc.name}</div>
              <div className="rank-sub">{tc.category}{crop.plantingSeasons?.length ? ` · ${translateMonths(crop.plantingSeasons, t)}` : ''}</div>
            </div>
            <div className="bar" aria-hidden="true"><div style={{ width: `${tc.suitabilityScore}%` }} /></div>
            <div className="rank-score">{tc.suitabilityScore}</div>
          </div>
        );
      })}
    </div>
  </section>
);

// Older saved results carry only the English recommendation strings
const actionsOf = (result) => result.actions
  || (result.recommendations || []).map((r, i) => {
    const [title, ...rest] = r.split(': ');
    return { code: `r${i}`, title: rest.length ? title : r, detail: rest.join(': ') };
  });

const ResultView = ({ result, soilData, lang, t, onEdit, onNew, onExport, resultsRef }) => (
  <div className="result-page" ref={resultsRef} tabIndex={-1} aria-live="polite">
    <div className="result-toolbar">
      <button className="btn btn-outline" onClick={onEdit} aria-label={t.editValues}>
        <ArrowLeft size={20} aria-hidden="true" /><span className="btn-label">{t.editValues}</span>
      </button>
      <span className="result-meta">{result.parcel || ''}</span>
      <button className="btn btn-outline export-button" onClick={onExport} aria-label={t.exportPDF}>
        <Download size={20} aria-hidden="true" /><span className="btn-label">{t.exportPDF}</span>
      </button>
      <button className="btn btn-signal" onClick={onNew}>
        <Plus size={20} aria-hidden="true" />{t.newAnalysisBtn}
      </button>
    </div>

    <div className="result-grid">
      <div className="result-col">
        <SowCard result={result} t={t} />
        <SoilHealthGauge healthData={result.soil_health_score} deficiencyIds={result.deficiencyIds || []} t={t} />
        <Actions actions={actionsOf(result)} t={t} />
      </div>
      <div className="result-col">
        <Ranking crops={result.recommendedCrops || []} t={t} />
        <SeasonalCalendar crops={result.recommendedCrops} t={t} />
      </div>
    </div>

    <h2 className="section-heading"><Leaf size={16} aria-hidden="true" />{t.analysisDetails}</h2>
    <div className="details-grid">
      <NutrientChart soilData={soilData} t={t} lang={lang} />
      <ShapExplanation shapData={result.shap_explanation} t={t} />
    </div>
  </div>
);

export default ResultView;
