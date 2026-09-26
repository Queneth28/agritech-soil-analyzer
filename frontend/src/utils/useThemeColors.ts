import { useEffect, useState } from 'react';

const TOKENS = ['chart-1', 'chart-2', 'chart-neg', 'chart-grid', 'chart-text', 'surface', 'border', 'text'] as const;
type Token = typeof TOKENS[number];
export type ThemeColors = Record<Token, string>;

const read = (): ThemeColors => {
  const style = getComputedStyle(document.documentElement);
  return TOKENS.reduce((acc, token) => {
    acc[token] = style.getPropertyValue(`--${token}`).trim() || '#888';
    return acc;
  }, {} as ThemeColors);
};

/**
 * Chart libraries draw SVG with literal colours, so they cannot follow CSS
 * variables directly. Read the design tokens and re-read on theme change.
 */
export default function useThemeColors(): ThemeColors {
  const [colors, setColors] = useState<ThemeColors>(read);
  useEffect(() => {
    const observer = new MutationObserver(() => setColors(read()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);
  return colors;
}
