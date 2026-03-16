import { render, screen } from '@testing-library/react';
import SoilHealthGauge from '../SoilHealthGauge';
import { TRANSLATIONS } from '../../constants/translations';

const t = TRANSLATIONS.en;

test('renders nothing when healthData is null', () => {
  const { container } = render(<SoilHealthGauge healthData={null} t={t} />);
  expect(container.firstChild).toBeNull();
});

test('renders score and grade when data is provided', () => {
  render(<SoilHealthGauge healthData={{ overall_score: 78.4, grade: 'B' }} t={t} />);
  expect(screen.getByText('78')).toBeInTheDocument();
  expect(screen.getByText(/Grade B/i)).toBeInTheDocument();
});

test('renders section heading', () => {
  render(<SoilHealthGauge healthData={{ overall_score: 90, grade: 'A' }} t={t} />);
  expect(screen.getByText(t.soilHealth)).toBeInTheDocument();
});
