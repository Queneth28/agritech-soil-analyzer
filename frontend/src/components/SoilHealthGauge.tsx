import React from 'react';
import { Leaf } from 'lucide-react';

const R = 60;
const CIRCUMFERENCE = 2 * Math.PI * R;

const SoilHealthGauge = ({ healthData, t }) => {
  if (!healthData) return null;
  const { overall_score, grade } = healthData;
  const offset = CIRCUMFERENCE * (1 - Math.max(0, Math.min(100, overall_score)) / 100);

  return (
    <section className={`card health-card grade-${grade}`} aria-label={t.soilHealth}>
      <h3 className="card-title"><Leaf size={20} />{t.soilHealth}</h3>
      <p className="card-description">{t.soilHealthDesc}</p>
      <div className="health-gauge-container">
        <svg width="168" height="168" viewBox="0 0 140 140" aria-hidden="true">
          <circle className="gauge-track" cx="70" cy="70" r={R} fill="none" strokeWidth="11" />
          <circle className="gauge-value" cx="70" cy="70" r={R} fill="none" strokeWidth="11"
            strokeDasharray={CIRCUMFERENCE} strokeDashoffset={offset} strokeLinecap="round"
            transform="rotate(-90 70 70)" style={{ transition: 'stroke-dashoffset 0.8s ease-out' }} />
        </svg>
        <div className="health-gauge-text">
          <span className="health-gauge-score">{Math.round(overall_score)}</span>
          <span className="health-gauge-label">{t.outOf || '/ 100'}</span>
        </div>
      </div>
      <div className="health-grade"><span>Grade {grade}</span></div>
    </section>
  );
};

export default SoilHealthGauge;
