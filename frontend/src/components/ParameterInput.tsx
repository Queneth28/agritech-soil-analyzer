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

const ParameterInput = memo(({ param, value, onChange, error, isOptimal, t }: Props) => {
  const handleChange = useCallback((e) => onChange(param.id, e.target.value), [param.id, onChange]);
  return (
    <div className="input-group">
      <label htmlFor={param.id} className="input-label">
        {t.parameters[param.id].label}
        {param.isPrimary && <span className="primary-badge">{t.primaryBadge}</span>}
      </label>
      <div style={{ position: 'relative' }}>
        <input id={param.id} type="number" step="0.01" min={param.range.min} max={param.range.max}
          value={value} onChange={handleChange} placeholder={param.placeholder}
          className={`input-field ${isOptimal ? 'optimal' : ''} ${error ? 'error' : ''}`}
          style={{ paddingRight: param.unit ? '4.5rem' : '1.125rem' }}
          aria-invalid={!!error} aria-describedby={error ? `${param.id}-error` : `${param.id}-desc`} />
        {param.unit && <span className="input-unit">{param.unit}</span>}
      </div>
      {error && <div id={`${param.id}-error`} className="error-message" role="alert"><AlertCircle size={14} /><span>{error}</span></div>}
      <p id={`${param.id}-desc`} className="input-description">{t.parameters[param.id].description}</p>
    </div>
  );
});
ParameterInput.displayName = 'ParameterInput';

export default ParameterInput;
