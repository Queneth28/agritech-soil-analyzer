import React from 'react';
import { FlaskConical, Loader2, AlertCircle } from 'lucide-react';
import ParameterInput from './ParameterInput';
import SOIL_PARAMETERS from '../constants/soilParameters';

const InputPanel = ({ soilData, errors, handleInputChange, isOptimalValue, isFormValid, loading, onAnalyze, apiError, onLoadSample, onClear, t }) => (
  <div className="input-panel" role="form" aria-label={t.panelTitle}>
    <div className="panel-header"><FlaskConical /><h2 className="panel-title">{t.panelTitle}</h2></div>
    {SOIL_PARAMETERS.map(param => (
      <ParameterInput key={param.id} param={param} value={soilData[param.id]}
        onChange={handleInputChange} error={errors[param.id]}
        isOptimal={isOptimalValue(param.id, soilData[param.id])} t={t} />
    ))}
    <div className="utility-buttons">
      <button onClick={onLoadSample} className="utility-btn" type="button">
        <FlaskConical size={14} />{t.loadSample}
      </button>
      <button onClick={onClear} className="utility-btn" type="button">
        ✕ {t.clearAll}
      </button>
    </div>
    <button onClick={onAnalyze} disabled={!isFormValid || loading} className="analyze-button" aria-busy={loading}>
      {loading
        ? <><Loader2 className="animate-spin" size={20} />{t.analyzing}</>
        : <><FlaskConical size={20} />{t.analyzeButton}</>}
    </button>
    {apiError && (
      <div className="error-message" style={{ marginTop: '1rem', flexWrap: 'wrap', gap: '0.5rem' }} role="alert">
        <AlertCircle size={16} /><span style={{ flex: 1 }}>{apiError}</span>
        <button onClick={onAnalyze} className="utility-btn" style={{ flex: 'none', padding: '0.3rem 0.75rem' }}>{t.retry}</button>
      </div>
    )}
  </div>
);

export default InputPanel;
