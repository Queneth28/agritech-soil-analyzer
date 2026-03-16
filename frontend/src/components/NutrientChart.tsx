import React from 'react';
import { BarChart3 } from 'lucide-react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, Legend, ResponsiveContainer
} from 'recharts';
import SOIL_PARAMETERS from '../constants/soilParameters';

const NutrientChart = ({ soilData, t, lang }) => {
  const chartData = ['N', 'P', 'K', 'pH', 'OC', 'S'].map(id => {
    const param = SOIL_PARAMETERS.find(p => p.id === id);
    const value = parseFloat(soilData[id] || 0);
    const optimal = (param.optimal.min + param.optimal.max) / 2;
    return {
      nutrient: t.parameters[id].label.replace(/\s*\([^)]*\)/, ''),
      current: Math.round(Math.min(120, (value / optimal) * 100)),
      optimal: 100
    };
  });

  return (
    <div className="result-card" role="region" aria-label={t.nutrientChart}>
      <h3 className="card-title"><BarChart3 size={22} style={{ color: '#00E5FF' }} />{t.nutrientChart}</h3>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <RadarChart data={chartData}>
            <PolarGrid stroke="rgba(0, 229, 255, 0.2)" />
            <PolarAngleAxis dataKey="nutrient" stroke="#80DEEA" />
            <PolarRadiusAxis angle={90} domain={[0, 120]} stroke="#80DEEA" />
            <Radar name={lang === 'fr' ? 'Actuel' : 'Current'} dataKey="current" stroke="#00E5FF" fill="#00E5FF" fillOpacity={0.3} />
            <Radar name="Optimal" dataKey="optimal" stroke="#4DD0E1" fill="#4DD0E1" fillOpacity={0.1} />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default NutrientChart;
