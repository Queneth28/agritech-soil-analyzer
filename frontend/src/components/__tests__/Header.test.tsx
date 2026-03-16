import { render, screen, fireEvent } from '@testing-library/react';
import Header from '../Header';
import { TRANSLATIONS } from '../../constants/translations';

const t = TRANSLATIONS.en;
const defaultProps = {
  lang: 'en',
  setLang: jest.fn(),
  theme: 'dark',
  setTheme: jest.fn(),
  showHistory: false,
  setShowHistory: jest.fn(),
  t,
};

test('renders app title', () => {
  render(<Header {...defaultProps} />);
  expect(screen.getByText(t.appTitle)).toBeInTheDocument();
});

test('shows FR button when lang is en', () => {
  render(<Header {...defaultProps} lang="en" />);
  expect(screen.getByText('FR')).toBeInTheDocument();
});

test('shows EN button when lang is fr', () => {
  render(<Header {...defaultProps} lang="fr" t={TRANSLATIONS.fr} />);
  expect(screen.getByText('EN')).toBeInTheDocument();
});

test('calls setLang when language button is clicked', () => {
  const setLang = jest.fn();
  render(<Header {...defaultProps} setLang={setLang} />);
  fireEvent.click(screen.getByText('FR'));
  expect(setLang).toHaveBeenCalled();
});

test('shows sun icon aria-label in dark mode', () => {
  render(<Header {...defaultProps} theme="dark" />);
  expect(screen.getByLabelText(t.themeLight)).toBeInTheDocument();
});

test('shows moon icon aria-label in light mode', () => {
  render(<Header {...defaultProps} theme="light" />);
  expect(screen.getByLabelText(t.themeDark)).toBeInTheDocument();
});
