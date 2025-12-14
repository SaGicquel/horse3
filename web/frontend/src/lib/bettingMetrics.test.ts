import { describe, expect, it } from 'vitest';
import { resolveBetMetrics } from './bettingMetrics';

describe('resolveBetMetrics', () => {
  it('uses value_place/kelly_place for SIMPLE PLACÉ selections', () => {
    const bet = {
      bet_type: 'SIMPLE PLACÉ',
      value: 12.3,
      kelly: 4.2,
      value_place: -7.2,
      kelly_place: 0,
      cote: 7.71,
      p_win: 0.12,
      p_place: 0.361,
    };
    const metrics = resolveBetMetrics(bet);
    expect(metrics.source).toBe('place');
    expect(metrics.valuePercent).toBe(-7.2);
    expect(metrics.kellyPercent).toBe(0);
    expect(metrics.odds).toBeCloseTo(7.71 / 3, 6);
  });

  it('prefers matching all_bet_types entry when available', () => {
    const bet = {
      bet_type: 'E/P (GAGNANT-PLACÉ)',
      all_bet_types: [{ type: 'E/P (GAGNANT-PLACÉ)', value: 9.5, kelly: 3.1 }],
      value: 20,
      value_place: -2,
    };
    const metrics = resolveBetMetrics(bet);
    expect(metrics.source).toBe('ep');
    expect(metrics.valuePercent).toBe(9.5);
    expect(metrics.kellyPercent).toBe(3.1);
  });
});

