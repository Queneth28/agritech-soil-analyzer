import './App.css';
import { useState, useCallback, useMemo, useEffect, useRef } from 'react';

import { TRANSLATIONS, fmt } from './constants/translations';
import SOIL_PARAMETERS from './constants/soilParameters';
import { TYPICAL_VALUES, SoilType } from './constants/soilTypes';
import { analyzeSoil, fetchHistory, deleteHistoryItem, NetworkError } from './utils/api';
import { readQueue, enqueue, removeFromQueue } from './utils/offlineQueue';
import { parseNumber } from './utils/numbers';
import { exportToPDF } from './utils/pdf';

import ToastContainer from './components/ToastContainer';
import Header from './components/Header';
import HistoryPanel from './components/HistoryPanel';
import HelpPanel from './components/HelpPanel';
import StepNav from './components/StepNav';
import WizardStep, { STEP_FIELDS } from './components/WizardStep';
import LivePreview from './components/LivePreview';
import ResultView from './components/ResultView';
import BottomNav from './components/BottomNav';

// Typical Luvisol (sandy loam, Centre-Ouest) profile
const SAMPLE_SOIL = { N: '125', P: '6.5', K: '255', pH: '6.5', EC: '0.34', OC: '0.58', S: '12', Zn: '0.62', Fe: '2.1', Cu: '0.72', Mn: '4.2', B: '0.32' };
const EMPTY_SOIL = SOIL_PARAMETERS.reduce((acc, p) => ({ ...acc, [p.id]: '' }), {} as Record<string, string>);

const readPref = (key: string, fallback: string) => {
  try { return window.localStorage.getItem(key) || fallback; } catch { return fallback; }
};
const writePref = (key: string, value: string) => {
  try { window.localStorage.setItem(key, value); } catch {}
};

function App() {
  // French first: the app is built for Burkina Faso
  const [lang, setLang] = useState(() => readPref('agritech_lang', 'fr'));
  const [theme, setTheme] = useState(() => readPref('agritech_theme',
    window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));

  const [step, setStep] = useState(0);
  const [maxReached, setMaxReached] = useState(0);
  const [view, setView] = useState<'wizard' | 'result'>('wizard');
  const [parcel, setParcel] = useState('');
  const [soilType, setSoilType] = useState<SoilType>('unknown');
  const [soilData, setSoilData] = useState<Record<string, string>>(EMPTY_SOIL);
  const [estimated, setEstimated] = useState<string[]>([]);
  const [errors, setErrors] = useState<Record<string, string | null>>({});

  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [notice, setNotice] = useState(null);

  const [online, setOnline] = useState(() => navigator.onLine !== false);
  const [pending, setPending] = useState(() => readQueue());
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [toasts, setToasts] = useState([]);
  const resultsRef = useRef(null);

  const t = TRANSLATIONS[lang];

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    writePref('agritech_theme', theme);
  }, [theme]);
  useEffect(() => {
    document.documentElement.lang = lang;
    writePref('agritech_lang', lang);
  }, [lang]);

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
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(x => x.id !== id)), 4500);
  }, []);
  const removeToast = useCallback((id) => setToasts(prev => prev.filter(x => x.id !== id)), []);

  const saveToHistory = useCallback((data, res) => {
    setAnalysisHistory(prev => {
      const updated = [{ id: Date.now() + Math.random(), date: new Date().toISOString(), soilData: data, result: res }, ...prev].slice(0, 20);
      try { window.localStorage.setItem('agritech_analysis_history', JSON.stringify(updated)); } catch {}
      return updated;
    });
  }, []);

  const deleteFromHistory = useCallback(async (id) => {
    setAnalysisHistory(prev => {
      const updated = prev.filter(a => a.id !== id);
      try { window.localStorage.setItem('agritech_analysis_history', JSON.stringify(updated)); } catch {}
      return updated;
    });
    try { await deleteHistoryItem(id); } catch {}
    addToast(t.analysisDeleted, 'info');
  }, [addToast, t]);

  // ── Form values ─────────────────────────────────────────────────────────
  const validateField = useCallback((paramId, value) => {
    if (value === '' || value == null) return null;
    const param = SOIL_PARAMETERS.find(p => p.id === paramId);
    const num = parseNumber(value);
    if (num === null) return t.validationErrors.invalidNumber;
    if (num < param.range.min) return `${t.validationErrors.minValue} ${param.range.min}`;
    if (num > param.range.max) return `${t.validationErrors.maxValue} ${param.range.max}`;
    return null;
  }, [t]);

  const handleFieldChange = useCallback((paramId, value) => {
    setSoilData(prev => ({ ...prev, [paramId]: value }));
    setErrors(prev => ({ ...prev, [paramId]: validateField(paramId, value) }));
    setEstimated(prev => prev.filter(id => id !== paramId));
    setApiError(null);
  }, [validateField]);

  const applyTypicalValues = useCallback((ids: string[]) => {
    const typical = TYPICAL_VALUES[soilType];
    setSoilData(prev => ids.reduce((acc, id) => ({ ...acc, [id]: String(typical[id]).replace('.', t.decimalSep || '.') }), prev));
    setErrors(prev => ids.reduce((acc, id) => ({ ...acc, [id]: null }), prev));
    setEstimated(prev => Array.from(new Set([...prev, ...ids])));
  }, [soilType, t]);

  const numericSoil = useMemo(() =>
    Object.fromEntries(Object.entries(soilData).map(([k, v]) => [k, parseNumber(v)])) as Record<string, number | null>,
  [soilData]);

  const stepValid = (s: number) => STEP_FIELDS[s].every(id => soilData[id] !== '' && !errors[id]);
  const canNext = stepValid(step);

  // ── Live preview: provisional ranking, missing values typical ───────────
  const requestPreview = useCallback((uptoStep: number) => {
    const filled = STEP_FIELDS.slice(1, uptoStep + 1).flat();
    if (!online || filled.length === 0) return;
    const typical = TYPICAL_VALUES[soilType];
    const soil = Object.fromEntries(SOIL_PARAMETERS.map(p => [p.id, numericSoil[p.id] ?? typical[p.id]]));
    analyzeSoil({ soil, lang }, { preview: true }).then(setPreview).catch(() => {});
  }, [online, soilType, numericSoil, lang]);

  // ── Analysis ────────────────────────────────────────────────────────────
  const resetForm = useCallback(() => {
    setSoilData(EMPTY_SOIL); setErrors({}); setEstimated([]); setParcel(''); setSoilType('unknown');
    setStep(0); setMaxReached(0); setPreview(null); setResult(null); setApiError(null); setView('wizard');
  }, []);

  const queueAnalysis = useCallback((request) => {
    setPending(enqueue(request));
    resetForm();
    setNotice({ title: t.queuedTitle, text: t.queuedText });
  }, [resetForm, t]);

  const runAnalysis = useCallback(async () => {
    const request = {
      soil: numericSoil as Record<string, number>, lang, estimated,
      parcel: parcel.trim() || undefined,
    };
    if (!navigator.onLine) { queueAnalysis(request); return; }
    setLoading(true); setApiError(null);
    try {
      const data = await analyzeSoil(request);
      setResult(data);
      setView('result');
      saveToHistory(soilData, data);
      addToast(`${t.analysisComplete}: ${t.cropNames[data.suitability] || data.suitability}`, 'success');
      window.scrollTo({ top: 0 });
      setTimeout(() => resultsRef.current?.focus(), 50);
    } catch (error) {
      if (error instanceof NetworkError) queueAnalysis(request);
      else { setApiError((error as Error).message || t.apiError); addToast(t.analysisFailed, 'error'); }
    } finally {
      setLoading(false);
    }
  }, [numericSoil, lang, estimated, parcel, soilData, queueAnalysis, saveToHistory, addToast, t]);

  const goToStep = useCallback((s: number) => {
    setStep(s); setMaxReached(m => Math.max(m, s)); setNotice(null);
    window.scrollTo({ top: 0 });
  }, []);

  const handleNext = useCallback(() => {
    if (!canNext) return;
    if (step === STEP_FIELDS.length - 1) { runAnalysis(); return; }
    requestPreview(step);
    goToStep(step + 1);
  }, [canNext, step, runAnalysis, requestPreview, goToStep]);

  // ── Offline queue: send saved analyses when the network comes back ──────
  const flushQueue = useCallback(async () => {
    const queue = readQueue();
    if (!queue.length) return;
    let sent = 0;
    for (const item of queue) {
      try {
        const data = await analyzeSoil(item);
        const asStrings = Object.fromEntries(Object.entries(item.soil).map(([k, v]) => [k, String(v)]));
        saveToHistory(asStrings, data);
        setPending(removeFromQueue(item.id));
        sent += 1;
      } catch (err) {
        if (err instanceof NetworkError) break;
        setPending(removeFromQueue(item.id)); // invalid request: do not retry forever
      }
    }
    if (sent) addToast(fmt(t.queuedSent, { n: sent }), 'success');
  }, [saveToHistory, addToast, t]);

  useEffect(() => {
    const up = () => { setOnline(true); flushQueue(); };
    const down = () => setOnline(false);
    window.addEventListener('online', up);
    window.addEventListener('offline', down);
    if (navigator.onLine !== false) flushQueue();
    return () => { window.removeEventListener('online', up); window.removeEventListener('offline', down); };
  }, [flushQueue]);

  // Result text comes from the backend: fetch it again when the language changes
  useEffect(() => {
    if (!result || result.lang === lang || !online) return;
    const soil = Object.fromEntries(Object.entries(numericSoil).map(([k, v]) => [k, v ?? 0]));
    analyzeSoil({ soil, lang, estimated: result.estimatedFields, parcel: result.parcel }, { preview: true })
      .then(data => setResult(prev => ({ ...data, analysis_id: prev.analysis_id, timestamp: prev.timestamp })))
      .catch(() => {});
  }, [lang]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── History ─────────────────────────────────────────────────────────────
  const loadFromHistory = useCallback((analysis) => {
    const data = analysis.soilData || analysis.soil_data || {};
    setSoilData({ ...EMPTY_SOIL, ...Object.fromEntries(Object.entries(data).map(([k, v]) => [k, String(v)])) });
    setErrors({});
    setEstimated(analysis.result?.estimatedFields || []);
    setParcel(analysis.result?.parcel || '');
    setResult(analysis.result || null);
    setView(analysis.result ? 'result' : 'wizard');
    setShowHistory(false);
    addToast(t.loadedFromHistory, 'success');
  }, [addToast, t]);

  const loadSample = useCallback(() => {
    setSoilData(SAMPLE_SOIL); setErrors({}); setEstimated([]);
    setSoilType('Luvisol');
    setParcel(lang === 'fr' ? 'Exemple : Luvisol de Koudougou' : 'Example: Koudougou Luvisol');
    setMaxReached(STEP_FIELDS.length - 1);
  }, [lang]);

  const handleExportPDF = useCallback(() => {
    if (result) exportToPDF(numericSoil, result, lang);
  }, [numericSoil, result, lang]);

  const activeTab = showHistory ? 'history' : 'analyze';

  return (
    <div className="app-container">
      <ToastContainer toasts={toasts} removeToast={removeToast} />
      {showHelp && <HelpPanel onClose={() => setShowHelp(false)} t={t} />}

      <Header lang={lang} setLang={setLang} theme={theme} setTheme={setTheme}
        showHistory={showHistory} setShowHistory={setShowHistory}
        historyCount={analysisHistory.length} onShowHelp={() => setShowHelp(true)}
        online={online} pendingCount={pending.length} t={t} />

      {showHistory && (
        <HistoryPanel analysisHistory={analysisHistory}
          loadFromHistory={loadFromHistory} deleteFromHistory={deleteFromHistory} t={t} />
      )}

      {view === 'result' && result ? (
        <ResultView result={result} soilData={numericSoil} lang={lang} t={t}
          onEdit={() => { setView('wizard'); goToStep(1); }} onNew={resetForm}
          onExport={handleExportPDF} resultsRef={resultsRef} />
      ) : (
        <div className="wizard">
          <StepNav step={step} maxReached={maxReached} onGo={goToStep} t={t} />
          <WizardStep step={step} t={t} soilData={soilData} errors={errors} estimated={estimated}
            onFieldChange={handleFieldChange} parcel={parcel} setParcel={setParcel}
            soilType={soilType} setSoilType={setSoilType} onUseTypical={applyTypicalValues}
            onBack={() => goToStep(step - 1)} onNext={handleNext} canNext={canNext}
            loading={loading} apiError={apiError} notice={notice} onLoadSample={loadSample} />
          <LivePreview preview={preview} step={step} t={t} />
        </div>
      )}

      <BottomNav active={activeTab} t={t}
        onAnalyze={() => { setShowHistory(false); window.scrollTo({ top: 0 }); }}
        onHistory={() => { setShowHistory(true); window.scrollTo({ top: 0 }); }}
        onGuide={() => setShowHelp(true)} />
    </div>
  );
}

export default App;
