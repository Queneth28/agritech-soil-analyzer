import { render, screen, fireEvent } from '@testing-library/react';
import ParameterInput from '../ParameterInput';
import { TRANSLATIONS } from '../../constants/translations';

const t = TRANSLATIONS.en;
const param = {
  id: 'N',
  unit: 'mg/kg',
  range: { min: 0, max: 400 },
  optimal: { min: 150, max: 300 },
  placeholder: 'ex: 138-270',
  isPrimary: true
};

test('renders label from translations', () => {
  render(<ParameterInput param={param} value="" onChange={() => {}} error={null} isOptimal={false} t={t} />);
  expect(screen.getByLabelText(/Nitrogen/i)).toBeInTheDocument();
});

test('renders Primary badge for primary params', () => {
  render(<ParameterInput param={param} value="" onChange={() => {}} error={null} isOptimal={false} t={t} />);
  expect(screen.getByText(t.primaryBadge)).toBeInTheDocument();
});

test('calls onChange with correct id and value', () => {
  const onChange = jest.fn();
  render(<ParameterInput param={param} value="" onChange={onChange} error={null} isOptimal={false} t={t} />);
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '200' } });
  expect(onChange).toHaveBeenCalledWith('N', '200');
});

test('renders error message when error prop is set', () => {
  render(<ParameterInput param={param} value="999" onChange={() => {}} error="Maximum value is 400" isOptimal={false} t={t} />);
  expect(screen.getByRole('alert')).toHaveTextContent('Maximum value is 400');
});

test('does not render error when error is null', () => {
  render(<ParameterInput param={param} value="200" onChange={() => {}} error={null} isOptimal={true} t={t} />);
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});
