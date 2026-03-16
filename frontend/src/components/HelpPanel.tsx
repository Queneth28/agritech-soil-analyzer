import React, { useState } from 'react';
import { X, FlaskConical, BarChart3, FileDown, History, Sprout, Leaf, ChevronDown, ChevronUp, Lightbulb } from 'lucide-react';

const STEPS = [
  {
    icon: <FlaskConical size={22} />,
    key: 'step1',
  },
  {
    icon: <Leaf size={22} />,
    key: 'step2',
  },
  {
    icon: <BarChart3 size={22} />,
    key: 'step3',
  },
  {
    icon: <Sprout size={22} />,
    key: 'step4',
  },
  {
    icon: <FileDown size={22} />,
    key: 'step5',
  },
  {
    icon: <History size={22} />,
    key: 'step6',
  },
];

const PARAM_GUIDE = [
  { id: 'N',  optimal: '150–300 mg/kg', note: 'helpParams.N'  },
  { id: 'P',  optimal: '7–10 mg/kg',    note: 'helpParams.P'  },
  { id: 'K',  optimal: '400–700 mg/kg', note: 'helpParams.K'  },
  { id: 'pH', optimal: '6.5–7.5',       note: 'helpParams.pH' },
  { id: 'EC', optimal: '0.4–0.8 dS/m',  note: 'helpParams.EC' },
  { id: 'OC', optimal: '0.8–2.0 %',     note: 'helpParams.OC' },
  { id: 'S',  optimal: '10–30 mg/kg',   note: 'helpParams.S'  },
  { id: 'Zn', optimal: '0.2–0.5 mg/kg', note: 'helpParams.Zn' },
  { id: 'Fe', optimal: '0.3–1.0 mg/kg', note: 'helpParams.Fe' },
  { id: 'Cu', optimal: '0.5–2.0 mg/kg', note: 'helpParams.Cu' },
  { id: 'Mn', optimal: '2–10 mg/kg',    note: 'helpParams.Mn' },
  { id: 'B',  optimal: '0.5–3.0 mg/kg', note: 'helpParams.B'  },
];

const HelpPanel = ({ onClose, t }: { onClose: () => void; t: any }) => {
  const [showParams, setShowParams] = useState(false);

  return (
    <div className="help-overlay" role="dialog" aria-modal="true" aria-label={t.helpTitle} onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="help-panel">
        <div className="help-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div className="help-icon"><Lightbulb size={22} /></div>
            <div>
              <h2 className="help-title">{t.helpTitle}</h2>
              <p className="help-subtitle">{t.helpSubtitle}</p>
            </div>
          </div>
          <button onClick={onClose} className="help-close" aria-label={t.close || 'Close'}>
            <X size={20} />
          </button>
        </div>

        <div className="help-body">
          <div className="help-steps">
            {STEPS.map((step, i) => (
              <div key={step.key} className="help-step">
                <div className="help-step-number">{i + 1}</div>
                <div className="help-step-icon">{step.icon}</div>
                <div className="help-step-content">
                  <h3 className="help-step-title">{t[step.key + 'Title']}</h3>
                  <p className="help-step-desc">{t[step.key + 'Desc']}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="help-tips">
            <h3 className="help-tips-title">💡 {t.tipsTitle}</h3>
            <ul className="help-tips-list">
              {(t.tips as string[]).map((tip: string, i: number) => (
                <li key={i}>{tip}</li>
              ))}
            </ul>
          </div>

          <button
            className="help-params-toggle"
            onClick={() => setShowParams(p => !p)}
            aria-expanded={showParams}
          >
            <FlaskConical size={16} />
            {t.paramGuideTitle}
            {showParams ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>

          {showParams && (
            <div className="help-params">
              <p className="help-params-desc">{t.paramGuideDesc}</p>
              <div className="help-params-grid">
                {PARAM_GUIDE.map(p => (
                  <div key={p.id} className="help-param-card">
                    <div className="help-param-id">{p.id}</div>
                    <div className="help-param-label">{t.parameters[p.id]?.label}</div>
                    <div className="help-param-optimal">✓ {p.optimal}</div>
                    <p className="help-param-note">{t[p.note]}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HelpPanel;
