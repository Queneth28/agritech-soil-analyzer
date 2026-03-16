import React from 'react';
import { Calendar } from 'lucide-react';

const SeasonalCalendar = ({ crops, t }) => {
  if (!crops || crops.length === 0) return null;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const topCrops = crops.filter(c => c.plantingSeasons?.length > 0).slice(0, 6);
  if (topCrops.length === 0) return null;

  return (
    <div className="result-card" role="region" aria-label={t.seasonalTitle}>
      <h3 className="card-title"><Calendar size={22} style={{ color: '#00E5FF' }} />{t.seasonalTitle}</h3>
      <p className="card-description">{t.seasonalDescription}</p>
      <div className="seasonal-grid">
        <div className="seasonal-header">
          <div className="seasonal-crop-label"></div>
          {months.map(m => <div key={m} className="seasonal-month">{m}</div>)}
        </div>
        {topCrops.map((crop, i) => {
          const translatedName = t.cropNames[crop.name] || crop.name;
          return (
            <div key={i} className="seasonal-row">
              <div className="seasonal-crop-label" title={translatedName}>{translatedName}</div>
              {months.map(m => {
                const isPlanting = (crop.plantingSeasons || []).includes(m);
                return (
                  <div key={m} className={`seasonal-cell ${isPlanting ? 'planting' : ''}`}
                    title={isPlanting ? `${t.plantLabel} ${translatedName} — ${m}` : ''} />
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default SeasonalCalendar;
