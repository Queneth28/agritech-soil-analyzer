import React from 'react';
import { History, Calendar, Trash2 } from 'lucide-react';
import { format } from 'date-fns';

const HistoryPanel = ({ analysisHistory, loadFromHistory, deleteFromHistory, t }) => (
  <div className="progress-container">
    <div className="result-card">
      <div className="card-title-row">
        <History size={24} style={{ color: '#00E5FF' }} />
        <h3 className="card-title">{t.viewHistory}</h3>
      </div>
      {analysisHistory.length === 0 ? (
        <p className="empty-history">{t.noHistory}</p>
      ) : (
        <div className="history-list">
          {analysisHistory.map(a => (
            <div key={a.id} className="history-item" onClick={() => loadFromHistory(a)} tabIndex={0} role="button"
              onKeyDown={e => e.key === 'Enter' && loadFromHistory(a)}>
              <div>
                <div className="history-date">
                  <Calendar size={16} style={{ color: '#00E5FF' }} />
                  <span>{format(new Date(a.date || a.timestamp), 'MMM dd, yyyy — HH:mm')}</span>
                </div>
                <div className="history-result">
                  {t.suitabilityLabel}: <strong>{a.result?.suitability || a.suitability}</strong>
                  {(a.result?.soil_health_score || a.health_score) && (
                    <span> | {t.soilHealth}: {Math.round(a.result?.soil_health_score?.overall_score || a.health_score)}/100</span>
                  )}
                </div>
              </div>
              <button onClick={e => { e.stopPropagation(); deleteFromHistory(a.id); }} className="history-delete">
                <Trash2 size={16} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  </div>
);

export default HistoryPanel;
