import React from 'react';
import { fmt } from '../constants/translations';

/** Compact soil health summary: grade tile, score and what to correct. */
const SoilHealthGauge = ({ healthData, t, deficiencyIds = [] as string[] }) => {
  if (!healthData) return null;
  const { overall_score, grade } = healthData;
  const title = grade === 'A' || grade === 'B' ? t.healthGood : grade === 'C' ? t.healthFair : t.healthPoor;
  const toCorrect = deficiencyIds.map(id => (t.parameters[id]?.label || id).replace(/\s*\([^)]*\)/, '').toLowerCase());

  return (
    <section className={`card health-card grade-${grade}`} aria-label={t.soilHealth}>
      <div className="health-grade-tile" aria-hidden="true">{grade}</div>
      <div>
        <h3 className="health-title">{title}</h3>
        <p className="health-sub">
          <span className="sr-only">{t.soilHealth}: </span>
          <span>Grade {grade}</span> · <span className="num">{Math.round(overall_score)}</span>/100
        </p>
        <p className="health-sub">{toCorrect.length ? fmt(t.toCorrect, { list: toCorrect.join(', ') }) : t.nothingToCorrect}</p>
      </div>
    </section>
  );
};

export default SoilHealthGauge;
