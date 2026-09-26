/** Parse a user-typed number, accepting a French decimal comma ("6,5"). */
export function parseNumber(value: string | number | null | undefined): number | null {
  if (value === null || value === undefined) return null;
  const text = String(value).trim().replace(',', '.');
  if (text === '' || !/^-?\d*\.?\d+$|^-?\d+\.$/.test(text)) return null;
  const num = parseFloat(text);
  return Number.isFinite(num) ? num : null;
}
