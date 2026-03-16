import React from 'react';
import { Leaf } from 'lucide-react';

const SoilHealthGauge = ({ healthData, t }) => {
  if (!healthData) return null;
  const { overall_score, grade } = healthData;
  const circumference = 2 * Math.PI * 60;
  const offset = circumference * (1 - overall_score / 100);
  const colors = { A: '#00E5FF', B: '#4DD0E1', C: '#FFB300', D: '#ef4444' };
  const color = colors[grade] || '#80DEEA';

  return (
    <div className="result-card" role="region" aria-label={t.soilHealth}>
      <h3 className="card-title"><Leaf size={22} style={{ color }} />{t.soilHealth}</h3>
      <p className="card-description">{t.soilHealthDesc}</p>
      <div className="health-gauge-container">
        <svg width="160" height="160" viewBox="0 0 140 140" aria-hidden="true">
          <circle cx="70" cy="70" r="60" fill="none" stroke="rgba(0,229,255,0.1)" strokeWidth="10" />
          <circle cx="70" cy="70" r="60" fill="none" stroke={color} strokeWidth="10"
            strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round"
            transform="rotate(-90 70 70)" style={{ transition: 'stroke-dashoffset 1s ease-out' }} />
        </svg>
        <div className="health-gauge-text">
          <span className="health-gauge-score" style={{ color }}>{Math.round(overall_score)}</span>
          <span className="health-gauge-label">/ 100</span>
        </div>
      </div>
      <div className="health-grade" style={{ background: `${color}22`, borderColor: `${color}44` }}>
        <span style={{ color }}>Grade {grade}</span>
      </div>
    </div>
  );
};

export default SoilHealthGauge;
