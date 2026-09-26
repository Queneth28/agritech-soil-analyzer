import React from 'react';
import { CalendarDays } from 'lucide-react';
import { translateMonths } from '../constants/translations';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const SeasonalCalendar = ({ crops, t }) => {
  if (!crops || crops.length === 0) return null;
  const topCrops = crops.filter(c => c.plantingSeasons?.length > 0).slice(0, 6);
  if (topCrops.length === 0) return null;

  return (
    <section className="card" aria-label={t.seasonalTitle}>
      <h3 className="card-title"><CalendarDays size={24} aria-hidden="true" />{t.seasonalTitle}</h3>
      <p className="card-description">{t.seasonalDescription}</p>
      <div className="seasonal-grid">
        <div className="seasonal-header" aria-hidden="true">
          <div />
          {MONTHS.map((m, i) => <div key={m} className="seasonal-month">{t.monthInitials?.[i] || m[0]}</div>)}
        </div>
        {topCrops.map((crop, i) => {
          const name = t.cropNames[crop.name] || crop.name;
          return (
            <div key={i} className="seasonal-row" role="group"
              aria-label={`${name}: ${t.plantLabel} ${translateMonths(crop.plantingSeasons, t)}`}>
              <div className="seasonal-crop-label" title={name}>{name}</div>
              {MONTHS.map(m => {
                const isPlanting = crop.plantingSeasons.includes(m);
                return <div key={m} className={`seasonal-cell ${isPlanting ? 'planting' : ''}`}
                  title={isPlanting ? `${t.plantLabel} ${name} — ${t.months?.[m] || m}` : (t.months?.[m] || m)} />;
              })}
            </div>
          );
        })}
      </div>
    </section>
  );
};

export default SeasonalCalendar;
