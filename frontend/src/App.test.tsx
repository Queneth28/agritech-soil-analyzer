import { render, screen, fireEvent } from '@testing-library/react';

jest.mock('./utils/pdf', () => ({
  exportToPDF: jest.fn(),
  calculateFertilizerRecommendations: jest.fn(() => []),
}));

import App from './App';

beforeEach(() => {
  window.localStorage.clear();
  global.fetch = jest.fn(() => Promise.reject(new Error('offline in tests'))) as any;
});

test('opens on the first step in French', () => {
  render(<App />);
  expect(screen.getAllByText('AgriTech Sol').length).toBeGreaterThan(0);
  expect(screen.getByText('Étape 1 sur 4')).toBeInTheDocument();
});

test('walks to the major nutrients step and blocks Next until filled', () => {
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: /Suivant : éléments majeurs/i }));
  expect(screen.getByText('Étape 2 sur 4')).toBeInTheDocument();
  const next = screen.getByRole('button', { name: /Suivant : état du sol/i });
  expect(next).toBeDisabled();
  fireEvent.change(screen.getByLabelText(/Azote/i), { target: { value: '125' } });
  fireEvent.change(screen.getByLabelText(/Phosphore/i), { target: { value: '6,5' } });
  fireEvent.change(screen.getByLabelText(/Potassium/i), { target: { value: '255' } });
  expect(next).not.toBeDisabled();
});

test('fills unmeasured micronutrients with typical values', () => {
  render(<App />);
  fireEvent.click(screen.getByText(/Charger/i));
  for (let i = 0; i < 3; i++) fireEvent.click(screen.getByRole('button', { name: /Suivant/i }));
  expect(screen.getByText('Étape 4 sur 4')).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText(/Zinc/i), { target: { value: '' } });
  fireEvent.change(screen.getByLabelText(/Bore/i), { target: { value: '' } });
  fireEvent.click(screen.getByRole('button', { name: /valeurs typiques/i }));
  expect(screen.getAllByText(TRANSLATIONS_FR_EST).length).toBe(2);
});

const TRANSLATIONS_FR_EST = 'estimée';
