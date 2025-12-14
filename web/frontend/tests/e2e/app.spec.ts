import { expect, test } from '@playwright/test';

const BANKROLL_CAP_RATIO = 0.8;
const mockPicks = [
  {
    race_key: 'RACE-001',
    discipline: 'PLAT',
    hippodrome: 'ParisLongchamp',
    heure: '12:00',
    nb_partants: 12,
    value: 12,
    kelly: 6,
    p_win: 0.22,
    cote: 3.4,
    nom: 'Bet Alpha',
    bet_type: 'GAGNANT',
    bet_risk: 'Faible',
  },
  {
    race_key: 'RACE-002',
    discipline: 'TROT',
    hippodrome: 'Vincennes',
    heure: '13:00',
    nb_partants: 14,
    value: 9,
    kelly: 4,
    p_win: 0.18,
    cote: 4.1,
    nom: 'Bet Bravo',
    bet_type: 'GAGNANT',
    bet_risk: 'Modéré',
  },
];

const mockRaceAnalysis = {
  race_key: 'RACE-001',
  horses: [
    { nom: 'Alpha', value: 10, kelly: 15, p_win: 0.2, p_place: 0.4, fair: 3.2, market: 3.8, bucket: 'SÛR' },
    { nom: 'Bravo', value: 7, kelly: 20, p_win: 0.18, p_place: 0.32, fair: 4.1, market: 4.5, bucket: 'ÉQUIL.' },
    { nom: 'Charlie', value: 7, kelly: 5, p_win: 0.14, p_place: 0.3, fair: 5.2, market: 6.0, bucket: 'RISQ.' },
    { nom: 'Delta', value: 2, kelly: 10, p_win: 0.1, p_place: 0.2, fair: 8.3, market: 10.0 },
  ],
};

const mockPortfolio = {
  total_stake: 750,
  total_ev: 120,
  expected_roi: 0.12,
  positions: [
    { nom: 'Alpha', value: 10, kelly: 15, p_win: 0.2, cote: 3.2, race_key: 'RACE-001', hippodrome: 'ParisLongchamp' },
  ],
};

test.beforeEach(async ({ page }) => {
  await page.route('**/healthz', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'ok', service: 'test-api' }),
    }),
  );

  await page.route('**/picks/today', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockPicks),
    }),
  );

  await page.route('**/analyze/**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockRaceAnalysis),
    }),
  );

  await page.route('**/portfolio/today', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockPortfolio),
    }),
  );
});

test('backend health badge shows when API is reachable', async ({ page }) => {
  await page.goto('/courses');
  await expect(page.getByText('Backend OK')).toBeVisible();
});

test('courses filter hides low value picks and sorts by value then kelly', async ({ page }) => {
  await page.goto('/courses');

  const viewButtons = page.getByRole('button', { name: 'Voir' });
  await expect(viewButtons.first()).toBeVisible();
  await viewButtons.first().click();
  const rows = page.locator('tbody tr');
  await expect(rows).toHaveCount(4);

  await page.getByTestId('value-threshold-input').fill('5');

  await expect(rows).toHaveCount(3);
  await expect(page.getByText('Delta')).toHaveCount(0);
  await expect(rows.first()).toContainText('Alpha');
  await expect(rows.nth(1)).toContainText('Bravo');
});

test('conseils portfolio keeps stake under bankroll cap and exports', async ({ page }) => {
  await page.goto('/conseils');

  const bankrollInput = page.getByTestId('bankroll-input');
  await bankrollInput.fill('1000');

  await page.getByRole('button', { name: /Portefeuille/i }).click();
  const totalStakeLocator = page.getByTestId('portfolio-total-stake');
  await expect(totalStakeLocator).toBeVisible();

  const totalStakeText = (await totalStakeLocator.textContent()) || '';
  const totalStake = Number(totalStakeText.replace(/[^\d.-]/g, ''));
  expect(totalStake).toBeLessThanOrEqual(1000 * BANKROLL_CAP_RATIO);

  const csvDownload = page.waitForEvent('download');
  await page.getByTestId('portfolio-export-csv').click();
  expect((await csvDownload).suggestedFilename()).toContain('portfolio');

  const jsonDownload = page.waitForEvent('download');
  await page.getByTestId('portfolio-export-json').click();
  expect((await jsonDownload).suggestedFilename()).toMatch(/\.json$/);
});
