type BetTypeSource = 'win' | 'place' | 'ep' | 'unknown';

export type ResolvedBetMetrics = {
  betType: string;
  source: BetTypeSource;
  p: number;
  odds: number;
  valuePercent: number;
  kellyPercent: number | null;
};

const toNumber = (value: unknown): number | null => {
  if (value === null || value === undefined) return null;
  const num = typeof value === 'string' ? Number(value) : (value as number);
  if (!Number.isFinite(num)) return null;
  return num;
};

const normalizeBetType = (value: unknown): string => String(value ?? '').trim();

const findBetTypeEntry = (bet: any, betType: string): any | null => {
  const all = Array.isArray(bet?.all_bet_types) ? bet.all_bet_types : [];
  if (!betType) return null;
  const wanted = betType.toUpperCase();
  return all.find((entry: any) => String(entry?.type ?? '').toUpperCase() === wanted) ?? null;
};

export const resolveBetMetrics = (bet: any): ResolvedBetMetrics => {
  const betType = normalizeBetType(bet?.bet_type ?? bet?.betType) || 'SIMPLE PLACÉ';
  const entry = findBetTypeEntry(bet, betType);

  const pWin = toNumber(bet?.p_win ?? bet?.proba_win ?? bet?.probabilite) ?? 0;
  const pPlace = toNumber(bet?.p_place) ?? (pWin ? Math.min(pWin * 2.5, 0.9) : 0);

  const winOdds = toNumber(bet?.cote ?? bet?.odds ?? bet?.market) ?? 0;
  const placeOdds = toNumber(bet?.cote_place) ?? (winOdds ? Math.max(1.05, winOdds / 3) : 0);

  const getEntryValue = () => toNumber(entry?.value);
  const getEntryKelly = () => toNumber(entry?.kelly);

  // E/P (Gagnant + Placé)
  if (betType.toUpperCase().includes('E/P')) {
    const valuePercent =
      getEntryValue() ??
      (winOdds > 0 && placeOdds > 0 ? (pWin * winOdds + pPlace * placeOdds - 2) * 50 : 0);
    const kellyPercent =
      getEntryKelly() ??
      Math.min(toNumber(bet?.kelly) ?? 0, toNumber(bet?.kelly_place) ?? 0) ??
      null;
    const odds = winOdds > 0 && placeOdds > 0 ? (winOdds + placeOdds) / 2 : winOdds || placeOdds || 0;
    return { betType, source: 'ep', p: pPlace, odds, valuePercent, kellyPercent };
  }

  // GAGNANT (simple)
  if (betType.toUpperCase().includes('GAGNANT') && !betType.toUpperCase().includes('PLACÉ')) {
    const valuePercent = getEntryValue() ?? (toNumber(bet?.value) ?? toNumber(bet?.value_pct) ?? toNumber(bet?.value_bet) ?? 0);
    const kellyPercent = getEntryKelly() ?? (toNumber(bet?.kelly) ?? toNumber(bet?.kelly_pct) ?? null);
    return { betType, source: 'win', p: pWin, odds: winOdds, valuePercent, kellyPercent };
  }

  // PLACÉ (simple)
  if (betType.toUpperCase().includes('PLACÉ')) {
    const valuePercent = getEntryValue() ?? (toNumber(bet?.value_place) ?? 0);
    const kellyPercent = getEntryKelly() ?? (toNumber(bet?.kelly_place) ?? null);
    return { betType, source: 'place', p: pPlace, odds: placeOdds, valuePercent, kellyPercent };
  }

  // Fallback: utilise la value "win" si rien d'autre n'est dispo.
  const fallbackValue = toNumber(bet?.value) ?? toNumber(bet?.value_pct) ?? toNumber(bet?.value_bet) ?? 0;
  const fallbackKelly = toNumber(bet?.kelly) ?? toNumber(bet?.kelly_pct) ?? null;
  return { betType, source: 'unknown', p: pWin, odds: winOdds, valuePercent: fallbackValue, kellyPercent: fallbackKelly };
};
