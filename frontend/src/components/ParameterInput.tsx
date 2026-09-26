import React, { memo, useCallback } from 'react';
import { AlertCircle } from 'lucide-react';
import { parseNumber } from '../utils/numbers';

interface Props {
  param: { id: string; unit: string; range: { min: number; max: number }; optimal: { min: number; max: number }; placeholder: string };
  value: string;
  onChange: (id: string, value: string) => void;
  error: string | null;
  estimated?: boolean;
  t: any;
}

// Display scale for the gauge: a little beyond the target so both "too low"
// and "too high" have room.
const scaleMax = (p: Props['param']) => Math.min(p.range.max, p.optimal.max * 1.6);

const ParameterInput = memo(({ param, value, onChange, error, estimated = false, t }: Props) => {
  const handleChange = useCallback((e) => onChange(param.id, e.target.value), [param.id, onChange]);
  const num = parseNumber(value);
  const status = error || num === null ? null
    : num < param.optimal.min ? 'low' : num > param.optimal.max ? 'high' : 'good';
  const max = scaleMax(param);
  const pct = (v: number) => Math.max(0, Math.min(100, (v / max) * 100));
  const fmt = (v: number) => String(v).replace('.', t.decimalSep || '.');
  const describedBy = error ? `${param.id}-error` : `${param.id}-meta`;

  return (
    <div className="big-field">
      <label htmlFor={param.id} className="big-label">
        {t.parameters[param.id].label}
        {estimated && <span className="estimated-tag">{t.estimatedBadge}</span>}
      </label>
      <div className="big-input-wrap">
        <input id={param.id} type="text" inputMode="decimal" autoComplete="off"
          value={value} onChange={handleChange} placeholder={param.placeholder.replace(/^ex:\s*/, '')}
          className={`big-input ${status ? `is-${status}` : ''} ${error ? 'is-error' : ''} ${estimated ? 'is-estimated' : ''}`}
          aria-invalid={!!error} aria-describedby={describedBy} />
        {param.unit && <span className="big-unit" aria-hidden="true">{param.unit}</span>}
      </div>
      {error ? (
        <div id={`${param.id}-error`} className="error-message" role="alert"><AlertCircle size={16} /><span>{error}</span></div>
      ) : (
        <>
          <div className="gauge" aria-hidden="true">
            <div className="gauge-zone" style={{ left: `${pct(param.optimal.min)}%`, width: `${pct(param.optimal.max) - pct(param.optimal.min)}%` }} />
            {num !== null && <div className={`gauge-mark ${status === 'good' ? '' : 'off'}`} style={{ left: `calc(${pct(num)}% - 12px)` }} />}
          </div>
          <div id={`${param.id}-meta`} className="field-meta">
            <span className="field-target">
              {t.targetShort} {fmt(param.optimal.min)}–{fmt(param.optimal.max)}{param.unit ? ` ${param.unit}` : ''}
            </span>
            {status && (
              <span className={`field-state ${status === 'good' ? 'good' : 'off'}`}>
                {status === 'good' ? `✓ ${t.goodLevel}` : status === 'low' ? `▼ ${t.tooLow}` : `▲ ${t.tooHigh}`}
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
});
ParameterInput.displayName = 'ParameterInput';

export default ParameterInput;
