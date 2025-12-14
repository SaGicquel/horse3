import { money, odds, percent } from './format';

type ExportOptions = {
  filename?: string;
  meta?: Record<string, unknown>;
};

const normalizePick = (pick: any) => {
  const pWin = pick?.p_win ?? pick?.proba_win ?? pick?.probabilite ?? pick?.pWin ?? 0;
  const value = pick?.value ?? pick?.value_pct ?? pick?.value_bet ?? pick?.valuePct ?? 0;
  const kelly = pick?.kelly ?? pick?.kelly_pct ?? pick?.kellyPct ?? 0;
  const stake = pick?.stake ?? pick?.mise ?? null;
  const oddsValue = pick?.cote ?? pick?.odds ?? pick?.market ?? pick?.market_odds ?? 0;

  return {
    type: pick?.bet_type || pick?.type || 'GAGNANT',
    name: pick?.nom || pick?.name || pick?.cheval || pick?.selection || '',
    hippodrome: pick?.hippodrome || pick?.venue || pick?.track || '',
    race: pick?.race_key || pick?.raceKey || pick?.race || pick?.course || '',
    odds: odds(oddsValue),
    pWin: percent(pWin),
    value: percent(value),
    kelly: percent(kelly),
    stake: stake !== null ? money(stake) : 'N/A',
  };
};

const normalizePortfolio = (portfolio: any) => {
  const totalStake = portfolio?.total_stake ?? portfolio?.totalStake ?? null;
  const totalEV = portfolio?.total_ev ?? portfolio?.totalEV ?? null;
  const expectedROI = portfolio?.expected_roi ?? portfolio?.roi ?? null;
  const positions = portfolio?.positions || portfolio?.bets || [];

  return {
    summary: {
      totalStake: money(totalStake),
      totalEV: money(totalEV),
      expectedROI: percent(expectedROI),
      positions: Array.isArray(positions) ? positions.length : 0,
    },
    positions,
  };
};

const escapeCsvCell = (value: unknown) => {
  if (value === null || value === undefined) return '';
  const str = String(value);
  const needsQuotes = /[",\n]/.test(str);
  const escaped = str.replace(/"/g, '""');
  return needsQuotes ? `"${escaped}"` : escaped;
};

const downloadFile = (filename: string, mime: string, content: string) => {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

const defaultFilename = (prefix: string) =>
  `${prefix}_${new Date().toISOString().split('T')[0]}`;

export const exportPicksCSV = (picks: any[] = [], options: ExportOptions = {}) => {
  const { filename = defaultFilename('picks') } = options;
  const headers = ['Type Pari', 'Cheval', 'Hippodrome', 'Course', 'Cote', 'p(win)', 'Value', 'Kelly', 'Mise'];
  const rows = picks.map(normalizePick).map((pick) => [
    pick.type,
    pick.name,
    pick.hippodrome,
    pick.race,
    pick.odds,
    pick.pWin,
    pick.value,
    pick.kelly,
    pick.stake,
  ]);

  const csv = [headers, ...rows]
    .map((row) => row.map(escapeCsvCell).join(','))
    .join('\n');

  downloadFile(`${filename}.csv`, 'text/csv', csv);
};

export const exportPicksJSON = (picks: any[] = [], options: ExportOptions = {}) => {
  const { filename = defaultFilename('picks'), meta = {} } = options;
  const payload = {
    exported_at: new Date().toISOString(),
    meta,
    picks: picks.map((pick) => ({
      ...normalizePick(pick),
      raw: pick,
    })),
  };

  downloadFile(`${filename}.json`, 'application/json', JSON.stringify(payload, null, 2));
};

export const exportPortfolioCSV = (portfolio: any, options: ExportOptions = {}) => {
  const { filename = defaultFilename('portfolio') } = options;
  const normalized = normalizePortfolio(portfolio);
  const headers = ['Type Pari', 'Cheval', 'Hippodrome', 'Course', 'Cote', 'p(win)', 'Value', 'Kelly', 'Mise'];
  const positions = (normalized.positions || []).map(normalizePick);

  const summaryLines = [
    ['Total Stake', normalized.summary.totalStake],
    ['EV Totale', normalized.summary.totalEV],
    ['ROI Attendu', normalized.summary.expectedROI],
    ['Nb Positions', normalized.summary.positions],
    [],
  ]
    .map((row) => row.map(escapeCsvCell).join(','))
    .join('\n');

  const rows = positions
    .map((pick) => [
      pick.type,
      pick.name,
      pick.hippodrome,
      pick.race,
      pick.odds,
      pick.pWin,
      pick.value,
      pick.kelly,
      pick.stake,
    ])
    .map((row) => row.map(escapeCsvCell).join(','))
    .join('\n');

  const csv = `${summaryLines}${headers.map(escapeCsvCell).join(',')}\n${rows}`;
  downloadFile(`${filename}.csv`, 'text/csv', csv);
};

export const exportPortfolioJSON = (portfolio: any, options: ExportOptions = {}) => {
  const { filename = defaultFilename('portfolio'), meta = {} } = options;
  const normalized = normalizePortfolio(portfolio);

  const payload = {
    exported_at: new Date().toISOString(),
    meta,
    summary: normalized.summary,
    positions: normalized.positions.map((p: any) => ({
      ...normalizePick(p),
      raw: p,
    })),
  };

  downloadFile(`${filename}.json`, 'application/json', JSON.stringify(payload, null, 2));
};
