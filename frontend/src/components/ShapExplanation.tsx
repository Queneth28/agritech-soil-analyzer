import React from 'react';
import { BarChart3, TrendingUp, TrendingDown } from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  Tooltip, CartesianGrid, Cell, ReferenceLine
} from 'recharts';
import useThemeColors from '../utils/useThemeColors';

const ShapExplanation = ({ shapData, t }: { shapData: Record<string, number> | null; t: any }) => {
  const c = useThemeColors();
  if (!shapData || Object.keys(shapData).length === 0) return null;
  const sorted = Object.entries(shapData).sort((a, b) => Math.abs(b[1] as number) - Math.abs(a[1] as number)).slice(0, 8);
  // SHAP values are on the 0-1 suitability scale: show them as score points
  const chartData = sorted.map(([key, val]) => ({
    name: key,
    label: t.parameters[key]?.label || key,
    value: Math.round((val as number) * 1000) / 10,
  }));

  return (
    <section className="card" aria-label={t.shapTitle}>
      <h3 className="card-title"><BarChart3 size={20} />{t.shapTitle}</h3>
      <p className="card-description">{t.shapDescription}</p>
      <div style={{ width: '100%', height: 250 }}>
        <ResponsiveContainer>
          <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 16, top: 4, bottom: 4 }}>
            <CartesianGrid horizontal={false} stroke={c['chart-grid']} />
            <XAxis type="number" tick={{ fill: c['chart-text'], fontSize: 12 }} stroke={c['chart-grid']} />
            <YAxis dataKey="name" type="category" width={36} tick={{ fill: c['chart-text'], fontSize: 13, fontWeight: 700 }} stroke={c['chart-grid']} />
            <ReferenceLine x={0} stroke={c['chart-text']} />
            <Tooltip
              cursor={{ fill: c['chart-grid'], opacity: 0.4 }}
              formatter={(v: number) => [`${v > 0 ? '+' : ''}${v} pts`, '']}
              labelFormatter={(_, p) => p?.[0]?.payload?.label || ''}
              contentStyle={{ background: c.surface, border: `1px solid ${c.border}`, borderRadius: 10, color: c.text }}
              itemStyle={{ color: c.text }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18}>
              {chartData.map((entry, i) => <Cell key={i} fill={entry.value >= 0 ? c['chart-1'] : c['chart-neg']} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="shap-legend">
        <span className="shap-positive"><TrendingUp size={14} /> {t.positiveInfluence}</span>
        <span className="shap-negative"><TrendingDown size={14} /> {t.negativeInfluence}</span>
      </div>
    </section>
  );
};

export default ShapExplanation;
