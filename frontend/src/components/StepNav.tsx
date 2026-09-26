import React from 'react';

/** Desktop list of wizard steps; completed steps can be revisited. */
const StepNav = ({ step, maxReached, onGo, t }) => {
  const items = [...t.steps, t.resultStep];
  return (
    <nav className="step-nav" aria-label={t.stepsLabel || 'Steps'}>
      {items.map((s, i) => {
        const state = i < step ? 'done' : i === step ? 'current' : 'todo';
        const reachable = i < items.length - 1 && i <= maxReached && i !== step;
        return (
          <button key={s.name} className={`step-item ${state}`} disabled={!reachable && state !== 'current'}
            aria-current={state === 'current' ? 'step' : undefined} onClick={() => reachable && onGo(i)}>
            <span className="step-dot" aria-hidden="true">{state === 'done' ? '✓' : i + 1}</span>
            <span>
              <span className="step-name">{s.name}</span>
              <span className="step-sub">{s.sub}</span>
            </span>
          </button>
        );
      })}
    </nav>
  );
};

export default StepNav;
