import React from 'react';
import { BarChart3, TrendingUp, TrendingDown } from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  Tooltip, CartesianGrid, Cell
} from 'recharts';

const ShapExplanation = ({ shapData, t }: { shapData: Record<string, number> | null; t: any }) => {
  if (!shapData || Object.keys(shapData).length === 0) return null;
  const sorted = Object.entries(shapData).sort((a, b) => Math.abs(b[1] as number) - Math.abs(a[1] as number)).slice(0, 8);
  const chartData = sorted.map(([key, val]) => ({
    name: t.parameters[key]?.label?.replace(/\s*\([^)]*\)/, '') || key,
    value: parseFloat((val as number).toFixed(3)),
    fill: (val as number) > 0 ? '#00E5FF' : '#ef4444'
  }));

  return (
    <div className="result-card" role="region" aria-label={t.shapTitle}>
      <h3 className="card-title"><BarChart3 size={22} style={{ color: '#00E5FF' }} />{t.shapTitle}</h3>
      <p className="card-description">{t.shapDescription}</p>
      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={chartData} layout="vertical" margin={{ left: 80, right: 20, top: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,229,255,0.1)" />
            <XAxis type="number" stroke="#80DEEA" tick={{ fill: '#80DEEA', fontSize: 12 }} />
            <YAxis dataKey="name" type="category" stroke="#80DEEA" tick={{ fill: '#B3E5FC', fontSize: 12 }} width={75} />
            <Tooltip contentStyle={{ background: 'rgba(26,35,50,0.95)', border: '1px solid rgba(0,229,255,0.3)', borderRadius: 8 }} labelStyle={{ color: '#E1F5FE' }} itemStyle={{ color: '#80DEEA' }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="shap-legend">
        <span className="shap-positive"><TrendingUp size={14} /> {t.positiveInfluence}</span>
        <span className="shap-negative"><TrendingDown size={14} /> {t.negativeInfluence}</span>
      </div>
    </div>
  );
};

export default ShapExplanation;
