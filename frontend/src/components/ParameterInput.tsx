import React, { memo, useCallback } from 'react';
import { AlertCircle } from 'lucide-react';

interface Props {
  param: { id: string; unit: string; range: { min: number; max: number }; optimal: { min: number; max: number }; placeholder: string; isPrimary?: boolean };
  value: string;
  onChange: (id: string, value: string) => void;
  error: string | null;
  isOptimal: boolean;
  t: any;
}

const statusOf = (value: string, optimal: { min: number; max: number }) => {
  if (value === '' || value == null) return null;
  const num = parseFloat(value);
  if (isNaN(num)) return null;
  if (num < optimal.min) return 'low';
  if (num > optimal.max) return 'high';
  return 'optimal';
};

const ParameterInput = memo(({ param, value, onChange, error, isOptimal, t }: Props) => {
  const handleChange = useCallback((e) => onChange(param.id, e.target.value), [param.id, onChange]);
  const status = error ? null : statusOf(value, param.optimal);
  const statusLabel = status && t[`status${status[0].toUpperCase()}${status.slice(1)}`];

  return (
    <div className="input-group">
      <label htmlFor={param.id} className="input-label" title={t.parameters[param.id].description}>
        {t.parameters[param.id].label}
        {param.isPrimary && <span className="primary-badge">{t.primaryBadge}</span>}
      </label>
      <div className="input-wrap">
        <input id={param.id} type="number" inputMode="decimal" step="0.01" min={param.range.min} max={param.range.max}
          value={value} onChange={handleChange} placeholder={param.placeholder.replace(/^ex:\s*/, '')}
          className={`input-field ${isOptimal ? 'optimal' : ''} ${error ? 'error' : ''}`}
          aria-invalid={!!error} aria-describedby={error ? `${param.id}-error` : `${param.id}-desc`} />
        {param.unit && <span className="input-unit" aria-hidden="true">{param.unit}</span>}
      </div>
      {error ? (
        <div id={`${param.id}-error`} className="error-message" role="alert"><AlertCircle size={14} /><span>{error}</span></div>
      ) : (
        <div id={`${param.id}-desc`} className="input-meta">
          <span className="input-target">{t.targetLabel} {param.optimal.min}–{param.optimal.max}</span>
          {status && <span className={`status-chip ${status}`}>{statusLabel}</span>}
          <span className="sr-only">{t.parameters[param.id].description}</span>
        </div>
      )}
    </div>
  );
});
ParameterInput.displayName = 'ParameterInput';

export default ParameterInput;
