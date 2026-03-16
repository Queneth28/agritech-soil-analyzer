import React, { useState } from 'react';
import { Download, CheckCircle2, Sprout } from 'lucide-react';
import SoilHealthGauge from './SoilHealthGauge';
import NutrientChart from './NutrientChart';
import ShapExplanation from './ShapExplanation';
import SeasonalCalendar from './SeasonalCalendar';
import ResultSkeleton from './ResultSkeleton';
import { translateCrop } from '../constants/translations';

const ResultsPanel = ({ result, loading, soilData, lang, t, fertilizers, exportToPDF, translateSummary, resultsRef }) => {
  const [showAllCrops, setShowAllCrops] = useState(false);
  return (
  <div className="results-panel" ref={resultsRef} tabIndex={-1}>
    {loading ? <ResultSkeleton /> : !result ? (
      <div className="result-card">
        <div className="empty-state">
          <div className="empty-icon" aria-hidden="true">
            <img src="/app-icon.png" alt="" style={{ width: '56px', height: '56px', borderRadius: '12px' }}
              onError={(e) => {
                const img = e.target as HTMLImageElement;
                img.style.display = 'none';
                const span = document.createElement('span');
                span.style.fontSize = '3rem';
                span.textContent = '🌱';
                img.parentNode?.appendChild(span);
              }} />
          </div>
          <h3 className="empty-title">{t.emptyTitle}</h3>
          <p className="empty-description">{t.emptyDescription}</p>
        </div>
      </div>
    ) : (
      <>
        <button onClick={exportToPDF} className="export-button">
          <Download size={20} />{t.exportPDF}
        </button>

        <SoilHealthGauge healthData={result.soil_health_score} t={t} />

        <NutrientChart soilData={soilData} t={t} lang={lang} />

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

        <ShapExplanation shapData={result.shap_explanation} t={t} />

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

        <SeasonalCalendar crops={result.recommendedCrops} t={t} />

        {result.recommendedCrops?.length > 0 && (
          <div className="result-card" role="region" aria-label={t.recommendedCrops}>
            <h3 className="card-title"><Sprout size={22} style={{ color: '#00E5FF' }} />{t.recommendedCrops}</h3>
            <p className="card-description">{t.cropsDescription}</p>
            <div className="crops-grid">
              {(showAllCrops ? result.recommendedCrops : result.recommendedCrops.slice(0, 6)).map((crop, i) => {
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
            {result.recommendedCrops.length > 6 && (
              <button onClick={() => setShowAllCrops(p => !p)} className="utility-btn" style={{ marginTop: '0.75rem', width: '100%' }}>
                {showAllCrops ? t.showLess : `${t.showMore} (${result.recommendedCrops.length - 6})`}
              </button>
            )}
          </div>
        )}
      </>
    )}
  </div>
  );
};

export default ResultsPanel;
