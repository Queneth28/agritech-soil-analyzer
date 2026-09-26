import { render, screen } from '@testing-library/react';
import SoilHealthGauge from '../SoilHealthGauge';
import { TRANSLATIONS } from '../../constants/translations';

const t = TRANSLATIONS.fr;

test('renders nothing when healthData is null', () => {
  const { container } = render(<SoilHealthGauge healthData={null} t={t} />);
  expect(container.firstChild).toBeNull();
});

test('renders score and grade when data is provided', () => {
  render(<SoilHealthGauge healthData={{ overall_score: 78.4, grade: 'B' }} t={t} />);
  expect(screen.getByText('78')).toBeInTheDocument();
  expect(screen.getByText(/Grade B/i)).toBeInTheDocument();
  expect(screen.getByText(t.healthGood)).toBeInTheDocument();
});

test('lists what to correct in plain words', () => {
  render(<SoilHealthGauge healthData={{ overall_score: 62, grade: 'C' }} deficiencyIds={['P', 'OC']} t={t} />);
  expect(screen.getByText(t.healthFair)).toBeInTheDocument();
  expect(screen.getByText(/À corriger : phosphore, carbone organique/i)).toBeInTheDocument();
});
