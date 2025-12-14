import { describe, expect, it } from 'vitest';
import { money, odds, percent } from './format';

describe('format.percent', () => {
  it('scales ratios under 1 to percentages with 2 decimals', () => {
    expect(percent(0.1234)).toBe('12.34%');
    expect(percent('0.4')).toBe('40.00%');
  });

  it('keeps values >= 1 as percents without rescaling', () => {
    expect(percent(12)).toBe('12.00%');
    expect(percent(250.499)).toBe('250.50%');
  });

  it('returns fallback for invalid input', () => {
    expect(percent(null, '--')).toBe('--');
    expect(percent('not-a-number')).toBe('N/A');
  });
});

describe('format.odds', () => {
  it('rounds to two decimals', () => {
    expect(odds(3.456)).toBe('3.46');
    expect(odds('2')).toBe('2.00');
  });

  it('falls back when value is not numeric', () => {
    expect(odds(undefined, '—')).toBe('—');
  });
});

describe('format.money', () => {
  it('formats with currency by default', () => {
    expect(money(10)).toBe('10.00€');
    expect(money('15.239')).toBe('15.24€');
  });

  it('supports toggling the currency suffix', () => {
    expect(money(42, 'N/A', { withCurrency: false })).toBe('42.00');
    expect(money(5, 'N/A', { currency: '$' })).toBe('5.00$');
  });

  it('returns the provided fallback for invalid values', () => {
    expect(money(NaN, '--')).toBe('--');
  });
});
