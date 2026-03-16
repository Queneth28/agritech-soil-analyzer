import React from 'react';

const ResultSkeleton = () => (
  <div className="results-panel" aria-busy="true">
    {[1, 2, 3].map(i => (
      <div key={i} className="result-card skeleton-card">
        <div className="skeleton-line skeleton-title" />
        <div className="skeleton-line skeleton-text" />
        <div className="skeleton-line skeleton-text short" />
      </div>
    ))}
  </div>
);

export default ResultSkeleton;
