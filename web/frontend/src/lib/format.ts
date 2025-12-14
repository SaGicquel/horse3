const normalizeNumber = (value: unknown): number | null => {
  if (value === null || value === undefined) return null;
  const num = typeof value === 'string' ? Number(value) : (value as number);
  if (!Number.isFinite(num)) return null;
  return num;
};

/**
 * Formate une probabilité/ratio en pourcentage avec 2 décimales.
 * Si la valeur est comprise entre -1 et 1, on considère que c'est un ratio (ex: 0.12 => 12%).
 * Si la valeur est déjà en pourcentage (ex: 12), on ne rescales pas.
 */
export const percent = (value: unknown, fallback = 'N/A'): string => {
  const num = normalizeNumber(value);
  if (num === null) return fallback;

  const scaled = Math.abs(num) <= 1 ? num * 100 : num;
  return `${scaled.toFixed(2)}%`;
};

/**
 * Formate des cotes en conservant 2 décimales.
 */
export const odds = (value: unknown, fallback = 'N/A'): string => {
  const num = normalizeNumber(value);
  if (num === null) return fallback;
  return num.toFixed(2);
};

/**
 * Formate un montant monétaire avec 2 décimales.
 */
export const money = (
  value: unknown,
  fallback = 'N/A',
  { currency = '€', withCurrency = true } = {}
): string => {
  const num = normalizeNumber(value);
  if (num === null) return fallback;
  const formatted = num.toFixed(2);
  return withCurrency ? `${formatted}${currency}` : formatted;
};

export default {
  percent,
  odds,
  money,
};
