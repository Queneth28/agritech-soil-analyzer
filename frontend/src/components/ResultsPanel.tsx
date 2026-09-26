import React, { useState } from 'react';
import { Download, Sprout, ListChecks, Trophy, FlaskConical, Leaf, Sparkles, Package } from 'lucide-react';
import SoilHealthGauge from './SoilHealthGauge';
import NutrientChart from './NutrientChart';
import ShapExplanation from './ShapExplanation';
import SeasonalCalendar from './SeasonalCalendar';
import ResultSkeleton from './ResultSkeleton';
import CropIcon from './CropIcon';
import { translateCrop } from '../constants/translations';

const EmptyState = ({ t }) => (
  <section className="card">
    <div className="empty-state">
      <div className="empty-icon" aria-hidden="true"><Sprout size={36} /></div>
      <h2 className="empty-title">{t.emptyTitle}</h2>
      <p className="empty-description">{t.emptyDescription}</p>
      <div className="empty-steps" aria-hidden="true">
        <span className="empty-step"><FlaskConical size={14} />{t.soilHealth}</span>
        <span className="empty-step"><Trophy size={14} />{t.recommendedCropLabel}</span>
        <span className="empty-step"><ListChecks size={14} />{t.actionPlan}</span>
      </div>
    </div>
  </section>
);

const Hero = ({ result, cropScores, t, exportToPDF }) => {
  const name = t.cropNames[result.recommendedCrop] || result.recommendedCrop;
  const top = Object.entries(cropScores || {}).sort(([, a]: any, [, b]: any) => b - a).slice(0, 4);
  return (
    <section className="card hero-card" aria-label={t.recommendedCropLabel}>
      <div className="hero-top">
        <span className="ml-badge"><Sparkles size={14} />{result.scoreSource === 'rules' ? t.rulesFallback : t.mlCropBadge}</span>
        <button onClick={exportToPDF} className="btn btn-secondary export-button" aria-label={t.exportPDF} title={t.exportPDF}>
          <Download size={16} aria-hidden="true" /><span className="btn-label">{t.exportPDF}</span>
        </button>
      </div>
      <div className="hero-main">
        <CropIcon name={result.recommendedCrop} size="lg" />
        <div>
          <p className="eyebrow">{t.recommendedCropLabel}</p>
          <h2 className="hero-name">{name}</h2>
        </div>
        <div className="hero-score">
          <div className="hero-score-value num">{result.confidenceScore}</div>
          <div className="hero-score-label"><span className="long">{t.cropSuitability} · </span>{t.outOf}</div>
        </div>
      </div>
      <p className="suitability-summary">{result.summary}</p>
      {top.length > 1 && (
        <div className="crop-probabilities">
          <p className="crop-prob-title">{t.topCropProbabilities}</p>
          {top.map(([crop, score]: [string, any]) => (
            <div key={crop} className="crop-prob-row">
              <span className="crop-prob-name"><CropIcon name={crop} size="sm" />{t.cropNames[crop] || crop}</span>
              <div className="crop-prob-bar-wrap" aria-hidden="true">
                <div className="crop-prob-bar" style={{ width: `${Math.round(score * 100)}%` }} />
              </div>
              <span className="crop-prob-pct">{Math.round(score * 100)}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
};

const ActionPlan = ({ result, t }) => (
  <section className="card" aria-label={t.actionPlan}>
    <h3 className="card-title"><ListChecks size={20} />{t.actionPlan}</h3>
    <p className="card-description">{t.actionPlanDesc}</p>
    <ol className="action-list">
      {(result.recommendations || []).map((rec, i) => <li key={i} className="action-item">{rec}</li>)}
    </ol>
    {result.deficiencies?.length > 0 && (<>
      <p className="chip-label">{t.deficienciesLabel}</p>
      <div className="chip-row">
        {result.deficiencies.map((d, i) => <span key={i} className="chip warn">{d.split(' (')[0]}</span>)}
      </div>
    </>)}
    {result.strengths?.length > 0 && (<>
      <p className="chip-label">{t.strengthsLabel}</p>
      <div className="chip-row">
        {result.strengths.map((s, i) => <span key={i} className="chip good">{s.split(' (')[0]}</span>)}
      </div>
    </>)}
  </section>
);

const CropRanking = ({ crops, t }) => {
  const [showAll, setShowAll] = useState(false);
  if (!crops?.length) return null;
  const visible = showAll ? crops : crops.slice(0, 6);
  return (
    <section className="card" aria-label={t.cropRanking}>
      <h3 className="card-title"><Trophy size={20} />{t.cropRanking}</h3>
      <p className="card-description">{t.cropsDescription}</p>
      <div className="crops-list">
        {visible.map((crop) => {
          const tc = translateCrop(crop, t);
          return (
            <div key={crop.name} className="crop-row">
              <CropIcon name={crop.name} />
              <div className="crop-info">
                <div className="crop-name">{tc.name}</div>
                <div className="crop-category">
                  <span>{tc.category}</span>
                  {crop.plantingSeasons?.length > 0 && <span>{t.plantLabel}: {crop.plantingSeasons.join(', ')}</span>}
                </div>
              </div>
              <div className="crop-score-bar" aria-hidden="true"><div style={{ width: `${tc.suitabilityScore}%` }} /></div>
              <div>
                <div className="crop-score-value">{tc.suitabilityScore}</div>
                <div className="crop-priority">{tc.priority}</div>
              </div>
            </div>
          );
        })}
      </div>
      {crops.length > 6 && (
        <button onClick={() => setShowAll(p => !p)} className="btn btn-ghost btn-block" style={{ marginTop: 8 }}>
          {showAll ? t.showLess : `${t.showMore} (${crops.length - 6})`}
        </button>
      )}
    </section>
  );
};

const FertilizerProducts = ({ fertilizers, t }) => (
  <section className="card" aria-label={t.fertilizerTitle}>
    <h3 className="card-title"><Package size={20} />{t.fertilizerTitle}</h3>
    <p className="card-description">{t.fertilizerDescription}</p>
    <div className="fertilizer-grid">
      {fertilizers.map((fert, i) => (
        <div key={i} className={`fertilizer-card ${fert.priority === 'High' ? 'high' : ''}`}>
          <div className="fertilizer-header">
            <div>
              <h4 className="fertilizer-name">{fert.fertilizer}</h4>
              <p className="fertilizer-nutrient">{t.parameters[fert.nutrientId]?.label || fert.nutrientId}</p>
            </div>
            <span className={`priority-badge ${fert.priority.toLowerCase()}`}>
              {fert.priority === 'High' ? t.priorityHigh : t.priorityMedium}
            </span>
          </div>
          <div className="fertilizer-details">
            <div><span className="detail-label">{t.composition}:</span> <span className="detail-value">{fert.composition}</span></div>
            <div><span className="detail-label">{t.dosage}:</span> <span className="detail-value">{fert.dosage}</span></div>
            <div><span className="detail-label">{t.estCost}:</span> <span className="detail-value accent">{fert.cost}</span></div>
          </div>
        </div>
      ))}
    </div>
    <div className="fertilizer-tip">
      <strong>{t.applicationTimingLabel}</strong> {t.applicationTimingText}
    </div>
  </section>
);

const ResultsPanel = ({ result, loading, soilData, lang, t, fertilizers, exportToPDF, resultsRef, translateSummary = null }) => {
  // Suitability 0-1 per crop; older backends sent it as 'probabilities'
  const cropScores = result?.cropScores || result?.probabilities;
  return (
    <div className="results-panel" ref={resultsRef} tabIndex={-1} aria-live="polite">
      {loading ? <ResultSkeleton /> : !result ? <EmptyState t={t} /> : (
        <>
          <Hero result={result} cropScores={cropScores} t={t} exportToPDF={exportToPDF} />

          <div className="results-grid">
            <SoilHealthGauge healthData={result.soil_health_score} t={t} />
            <ActionPlan result={result} t={t} />
          </div>

          <CropRanking crops={result.recommendedCrops} t={t} />
          <SeasonalCalendar crops={result.recommendedCrops} t={t} />
          {fertilizers.length > 0 && <FertilizerProducts fertilizers={fertilizers} t={t} />}

          <h2 className="section-heading"><Leaf size={14} aria-hidden="true" />{t.analysisDetails}</h2>
          <div className="details-grid">
            <NutrientChart soilData={soilData} t={t} lang={lang} />
            <ShapExplanation shapData={result.shap_explanation} t={t} />
          </div>
        </>
      )}
    </div>
  );
};

export default ResultsPanel;
