import { render, screen, fireEvent } from '@testing-library/react';
import ParameterInput from '../ParameterInput';
import { TRANSLATIONS } from '../../constants/translations';

const t = TRANSLATIONS.fr;
const param = {
  id: 'P',
  unit: 'mg/kg',
  range: { min: 0, max: 60 },
  optimal: { min: 8, max: 25 },
  placeholder: 'ex: 3-15',
};

test('renders label from translations', () => {
  render(<ParameterInput param={param} value="" onChange={() => {}} error={null} t={t} />);
  expect(screen.getByLabelText(/Phosphore/i)).toBeInTheDocument();
});

test('calls onChange with correct id and value', () => {
  const onChange = jest.fn();
  render(<ParameterInput param={param} value="" onChange={onChange} error={null} t={t} />);
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '6,5' } });
  expect(onChange).toHaveBeenCalledWith('P', '6,5');
});

test('reads a French decimal comma and flags a low value', () => {
  render(<ParameterInput param={param} value="6,5" onChange={() => {}} error={null} t={t} />);
  expect(screen.getByText(/Trop bas/)).toBeInTheDocument();
  expect(screen.getByText('Cible 8–25 mg/kg')).toBeInTheDocument();
});

test('shows a good level inside the target', () => {
  render(<ParameterInput param={param} value="12" onChange={() => {}} error={null} t={t} />);
  expect(screen.getByText(/Bon niveau/)).toBeInTheDocument();
});

test('marks estimated values', () => {
  render(<ParameterInput param={param} value="5.4" onChange={() => {}} error={null} estimated t={t} />);
  expect(screen.getByText(t.estimatedBadge)).toBeInTheDocument();
});

test('renders error message when error prop is set', () => {
  render(<ParameterInput param={param} value="999" onChange={() => {}} error="La valeur maximale est 60" t={t} />);
  expect(screen.getByRole('alert')).toHaveTextContent('La valeur maximale est 60');
});

test('does not render error when error is null', () => {
  render(<ParameterInput param={param} value="12" onChange={() => {}} error={null} t={t} />);
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});
