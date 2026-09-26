import React from 'react';
import { Radar as RadarIcon } from 'lucide-react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, Legend, ResponsiveContainer
} from 'recharts';
import SOIL_PARAMETERS from '../constants/soilParameters';
import useThemeColors from '../utils/useThemeColors';

const NutrientChart = ({ soilData, t, lang }) => {
  const c = useThemeColors();
  const chartData = ['N', 'P', 'K', 'pH', 'OC', 'S'].map(id => {
    const param = SOIL_PARAMETERS.find(p => p.id === id);
    const value = parseFloat(soilData[id] || 0);
    const optimal = (param.optimal.min + param.optimal.max) / 2;
    return {
      nutrient: id,
      current: Math.round(Math.min(120, (value / optimal) * 100)),
      optimal: 100
    };
  });

  return (
    <section className="card" aria-label={t.nutrientChart}>
      <h3 className="card-title"><RadarIcon size={20} />{t.nutrientChart}</h3>
      <div style={{ width: '100%', height: 280 }}>
        <ResponsiveContainer>
          <RadarChart data={chartData} outerRadius="72%">
            <PolarGrid stroke={c['chart-grid']} />
            <PolarAngleAxis dataKey="nutrient" tick={{ fill: c['chart-text'], fontSize: 13, fontWeight: 700 }} />
            <PolarRadiusAxis angle={90} domain={[0, 120]} tick={false} axisLine={false} />
            <Radar name="Optimal" dataKey="optimal" stroke={c['chart-2']} fill={c['chart-2']} fillOpacity={0.08} strokeDasharray="4 4" />
            <Radar name={lang === 'fr' ? 'Votre sol' : 'Your soil'} dataKey="current" stroke={c['chart-1']} fill={c['chart-1']} fillOpacity={0.25} strokeWidth={2} />
            <Legend wrapperStyle={{ fontSize: 13, color: c['chart-text'] }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};

export default NutrientChart;
