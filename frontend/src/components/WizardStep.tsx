import React from 'react';
import { ArrowLeft, ArrowRight, Loader2, AlertCircle, Sparkles, Save } from 'lucide-react';
import ParameterInput from './ParameterInput';
import SOIL_PARAMETERS from '../constants/soilParameters';
import { SOIL_TYPES } from '../constants/soilTypes';
import { fmt } from '../constants/translations';

export const STEP_FIELDS = [[], ['N', 'P', 'K'], ['pH', 'OC', 'EC'], ['S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']];
// Steps where a lab often leaves values out, so estimating them is offered
const ESTIMABLE_STEPS = [2, 3];

const WizardStep = ({ step, t, soilData, errors, estimated, onFieldChange, parcel, setParcel, soilType, setSoilType,
  onUseTypical, onBack, onNext, canNext, loading, apiError, notice, onLoadSample }) => {
  const total = STEP_FIELDS.length;
  const ids = STEP_FIELDS[step];
  const isLast = step === total - 1;
  const missing = ids.filter(id => soilData[id] === '');
  const nextLabel = isLast ? t.seeResult : fmt(t.nextTo, { step: t.steps[step + 1].name.toLowerCase() });

  return (
    <main className="step-main">
      <div className="step-progress" aria-hidden="true">
        {STEP_FIELDS.map((_, i) => <span key={i} className={i < step ? 'done' : i === step ? 'current' : ''} />)}
      </div>

      {notice && (
        <div className="notice" role="status">
          <Save size={22} aria-hidden="true" />
          <div><strong>{notice.title}</strong><p>{notice.text}</p></div>
        </div>
      )}

      <div className="step-head">
        <div className="step-count">{fmt(t.stepOf, { n: step + 1, total })}</div>
        <h2 className="step-title">{t.steps[step].name}</h2>
        <p className="step-intro">{t.stepIntro[step]}</p>
      </div>

      {step === 0 ? (
        <div className="field-list">
          <div className="big-field">
            <label htmlFor="parcel" className="big-label">{t.parcelLabel}</label>
            <input id="parcel" className="text-input" value={parcel} maxLength={80}
              placeholder={t.parcelPlaceholder} onChange={e => setParcel(e.target.value)} aria-describedby="parcel-hint" />
            <p id="parcel-hint" className="field-hint">{t.parcelHint}</p>
          </div>
          <div className="big-field">
            <label htmlFor="soilType" className="big-label">{t.soilTypeLabel}</label>
            <select id="soilType" className="select-input" value={soilType} onChange={e => setSoilType(e.target.value)} aria-describedby="soil-hint">
              {SOIL_TYPES.map(st => <option key={st} value={st}>{t.soilTypes[st]}</option>)}
            </select>
            <p id="soil-hint" className="field-hint">{t.soilTypeHint}</p>
          </div>
          <button type="button" className="sample-link" onClick={onLoadSample}>{t.loadSample}</button>
        </div>
      ) : (
        <>
          <div className={`field-list ${ids.length > 3 ? 'two-col' : ''}`}>
            {ids.map(id => (
              <ParameterInput key={id} param={SOIL_PARAMETERS.find(p => p.id === id)} value={soilData[id]}
                onChange={onFieldChange} error={errors[id]} estimated={estimated.includes(id)} t={t} />
            ))}
          </div>
          {ESTIMABLE_STEPS.includes(step) && missing.length > 0 && (
            <button type="button" className="btn btn-outline typical-btn" onClick={() => onUseTypical(missing)}>
              <Sparkles size={18} aria-hidden="true" />{t.useTypical}
            </button>
          )}
        </>
      )}

      <details className="tip-inline">
        <summary>{t.tipTitles[step]}</summary>
        <p>{t.stepTips[step]}</p>
      </details>

      {apiError && (
        <div className="api-error" role="alert">
          <AlertCircle size={20} aria-hidden="true" /><span>{apiError}</span>
          <button className="btn btn-outline" onClick={onNext}>{t.retry}</button>
        </div>
      )}

      <div className="step-footer">
        {step > 0 && (
          <button className="btn btn-outline btn-xl" onClick={onBack} aria-label={t.back}>
            <ArrowLeft size={20} aria-hidden="true" /><span className="btn-label">{t.back}</span>
          </button>
        )}
        <button className="btn btn-signal btn-xl analyze-button" onClick={onNext} disabled={!canNext || loading} aria-busy={loading}>
          {loading ? <><Loader2 className="animate-spin" size={22} />{t.analyzing}</> : <>{nextLabel}<ArrowRight size={22} aria-hidden="true" /></>}
        </button>
      </div>
    </main>
  );
};

export default WizardStep;
