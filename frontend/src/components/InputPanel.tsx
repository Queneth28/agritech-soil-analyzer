import React from 'react';
import { FlaskConical, Loader2, RotateCcw, ClipboardList, AlertCircle } from 'lucide-react';
import ParameterInput from './ParameterInput';
import SOIL_PARAMETERS from '../constants/soilParameters';

const GROUPS = [
  { key: 'groupMacro', ids: ['N', 'P', 'K'] },
  { key: 'groupSoil', ids: ['pH', 'OC', 'EC'] },
  { key: 'groupMicro', ids: ['S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'] },
];

const InputPanel = ({ soilData, errors, handleInputChange, isOptimalValue, isFormValid, loading, onAnalyze, apiError, onLoadSample, onClear, progress, t }) => {
  const filled = Object.values(soilData).filter(v => v !== '').length;
  return (
    <section className="card input-panel" aria-label={t.panelTitle}>
      <div className="panel-header">
        <div className="panel-header-row">
          <h2 className="panel-title"><FlaskConical size={20} />{t.panelTitle}</h2>
          <span className="progress-meta num">{filled}/{SOIL_PARAMETERS.length} {t.fieldsCompleted}</span>
        </div>
        <div className="progress-bar-bg" role="progressbar" aria-label={t.progressLabel}
          aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
          <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <div className="panel-body">
        {GROUPS.map(group => (
          <fieldset key={group.key} className="field-group">
            <legend>{t[group.key]}</legend>
            <div className="field-grid">
              {group.ids.map(id => {
                const param = SOIL_PARAMETERS.find(p => p.id === id);
                return (
                  <ParameterInput key={id} param={param} value={soilData[id]}
                    onChange={handleInputChange} error={errors[id]}
                    isOptimal={isOptimalValue(id, soilData[id])} t={t} />
                );
              })}
            </div>
          </fieldset>
        ))}
      </div>

      <div className="panel-footer">
        {apiError && (
          <div className="api-error" role="alert">
            <AlertCircle size={16} /><span>{apiError}</span>
            <button onClick={onAnalyze} className="btn btn-secondary">{t.retry}</button>
          </div>
        )}
        <div className="utility-buttons">
          <button onClick={onLoadSample} className="btn btn-secondary" type="button">
            <ClipboardList size={16} />{t.loadSample}
          </button>
          <button onClick={onClear} className="btn btn-ghost" type="button">
            <RotateCcw size={16} />{t.clearAll}
          </button>
        </div>
        <button onClick={onAnalyze} disabled={!isFormValid || loading} className="btn btn-primary btn-lg btn-block analyze-button" aria-busy={loading}>
          {loading
            ? <><Loader2 className="animate-spin" size={20} />{t.analyzing}</>
            : <><FlaskConical size={20} />{t.analyzeButton}</>}
        </button>
      </div>
    </section>
  );
};

export default InputPanel;
