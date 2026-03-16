import { render, screen } from '@testing-library/react';

jest.mock('./utils/pdf', () => ({
  exportToPDF: jest.fn(),
  calculateFertilizerRecommendations: jest.fn(() => []),
}));

import App from './App';

test('renders app title', () => {
  render(<App />);
  expect(screen.getByText(/AgriTech Soil Analyzer/i)).toBeInTheDocument();
});
