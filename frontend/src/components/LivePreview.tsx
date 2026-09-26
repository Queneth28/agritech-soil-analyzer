import React from 'react';

/** Provisional crop ranking shown while the wizard is being filled (desktop). */
const LivePreview = ({ preview, step, t }) => {
  const top = preview ? Object.entries(preview.cropScores || {}).slice(0, 4) : [];
  return (
    <aside className="step-aside" aria-label={t.livePreview}>
      <section aria-live="polite">
        <div className="aside-label">{t.livePreview}</div>
        {top.length === 0 ? (
          <p className="preview-note">{t.livePreviewEmpty}</p>
        ) : (
          <>
            <p style={{ marginTop: 8 }}>{t.livePreviewDesc}</p>
            <div className="preview-list">
              {top.map(([crop, score]: [string, any]) => (
                <div key={crop} className="preview-row">
                  <span>{t.cropNames[crop] || crop}</span>
                  <div className="bar" aria-hidden="true"><div style={{ width: `${Math.round(score * 100)}%` }} /></div>
                  <span className="num" style={{ textAlign: 'right' }}>{Math.round(score * 100)}</span>
                </div>
              ))}
            </div>
            <p className="preview-note">{t.livePreviewNote}</p>
          </>
        )}
      </section>
      <section className="tip-card" aria-label={t.tipTitles[step]}>
        <h3>{t.tipTitles[step]}</h3>
        <p>{t.stepTips[step]}</p>
      </section>
    </aside>
  );
};

export default LivePreview;
