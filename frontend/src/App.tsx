import './App.css';
import { useState, useCallback, useMemo, useEffect, useRef } from 'react';

import { TRANSLATIONS, BACKEND_TRANSLATIONS } from './constants/translations';
import SOIL_PARAMETERS from './constants/soilParameters';
import { analyzeSoil as apiAnalyzeSoil, fetchHistory, deleteHistoryItem } from './utils/api';
import { calculateFertilizerRecommendations, exportToPDF } from './utils/pdf';

import ToastContainer from './components/ToastContainer';
import Header from './components/Header';
import HistoryPanel from './components/HistoryPanel';
import InputPanel from './components/InputPanel';
import ResultsPanel from './components/ResultsPanel';
import HelpPanel from './components/HelpPanel';

const SAMPLE_SOIL = { N:'140', P:'35', K:'180', pH:'6.5', EC:'1.2', OC:'2.1', S:'18', Zn:'1.8', Fe:'8', Cu:'0.9', Mn:'5', B:'0.7' };

function App() {
  const [lang, setLang] = useState('en');
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
  const [showHelp, setShowHelp] = useState(false);
  const resultsRef = useRef(null);

  const t = TRANSLATIONS[lang];

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try { window.localStorage.setItem('agritech_theme', theme); } catch {}
  }, [theme]);

  useEffect(() => {
    fetchHistory()
      .then(data => setAnalysisHistory(data))
      .catch(() => {
        try {
          const saved = window.localStorage.getItem('agritech_analysis_history');
          if (saved) setAnalysisHistory(JSON.parse(saved));
        } catch {}
      });
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

  const deleteFromHistory = useCallback(async (id) => {
    const updated = analysisHistory.filter(a => a.id !== id);
    setAnalysisHistory(updated);
    try { window.localStorage.setItem('agritech_analysis_history', JSON.stringify(updated)); } catch {}
    try { await deleteHistoryItem(id); } catch {}
    addToast(t.analysisDeleted, 'info');
  }, [analysisHistory, addToast, t]);

  const loadFromHistory = useCallback((analysis) => {
    setSoilData(analysis.soilData || analysis.soil_data || {});
    setResult(analysis.result || null);
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

  const handleAnalyzeSoil = useCallback(async () => {
    if (!isFormValid) return;
    setLoading(true); setApiError(null); setResult(null);
    try {
      const payload = Object.entries(soilData).reduce((acc, [k, v]) => ({ ...acc, [k]: parseFloat(v as string) }), {});
      const data = await apiAnalyzeSoil(payload);
      setResult(data);
      saveToHistory(soilData, data);
      addToast(`${t.analysisComplete}: ${data.suitability}`, 'success');
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        resultsRef.current?.focus();
      }, 100);
    } catch (error) {
      setApiError((error as Error).message || t.apiError);
      addToast(t.analysisFailed, 'error');
    } finally {
      setLoading(false);
    }
  }, [soilData, isFormValid, t, saveToHistory, addToast]);

  const loadSample = useCallback(() => setSoilData(SAMPLE_SOIL), []);
  const clearAll = useCallback(() => setSoilData(SOIL_PARAMETERS.reduce((acc, p) => ({ ...acc, [p.id]: '' }), {})), []);

  const fertilizers = useMemo(() => {
    if (!result || !isFormValid) return [];
    return calculateFertilizerRecommendations(soilData);
  }, [result, soilData, isFormValid]);

  const handleExportPDF = useCallback(() => {
    if (result) exportToPDF(soilData, result, lang);
  }, [soilData, result, lang]);

  return (
    <div className={`app-container theme-${theme}`}>
      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {showHelp && <HelpPanel onClose={() => setShowHelp(false)} t={t} />}

      <Header lang={lang} setLang={setLang} theme={theme} setTheme={setTheme}
        showHistory={showHistory} setShowHistory={setShowHistory}
        historyCount={analysisHistory.length} onShowHelp={() => setShowHelp(true)} t={t} />

      {showHistory && (
        <HistoryPanel analysisHistory={analysisHistory}
          loadFromHistory={loadFromHistory} deleteFromHistory={deleteFromHistory} t={t} />
      )}

      <div className="progress-container">
        <div className="progress-card" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
          <div className="progress-header">
            <span className="progress-label">{t.progressLabel}</span>
            <span className="progress-percentage">{progress}%</span>
          </div>
          <div className="progress-bar-bg"><div className="progress-bar-fill" style={{ width: `${progress}%` }} /></div>
        </div>
      </div>

      <div className="main-content">
        <InputPanel soilData={soilData} errors={errors} handleInputChange={handleInputChange}
          isOptimalValue={isOptimalValue} isFormValid={isFormValid} loading={loading}
          onAnalyze={handleAnalyzeSoil} apiError={apiError}
          onLoadSample={loadSample} onClear={clearAll} t={t} />

        <ResultsPanel result={result} loading={loading} soilData={soilData} lang={lang} t={t}
          fertilizers={fertilizers} exportToPDF={handleExportPDF}
          translateSummary={translateSummary} resultsRef={resultsRef} />
      </div>
    </div>
  );
}

export default App;
