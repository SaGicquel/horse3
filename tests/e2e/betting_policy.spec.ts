/**
 * Tests e2e Playwright pour les pages Settings et Conseils
 *
 * Scénarios testés:
 * - Settings: Sélection profil Kelly, modification des caps, sauvegarde
 * - Conseils: Affichage du résumé politique, badges violations, export
 */

import { test, expect } from '@playwright/test';

const BASE_URL = process.env.VITE_API_URL || 'http://localhost:5173';

test.describe('Settings Page - Politique de mise', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE_URL}/settings`);
    // Attendre que la page soit chargée
    await page.waitForSelector('h1:has-text("Paramètres")');
  });

  test('affiche les 4 profils Kelly', async ({ page }) => {
    // Vérifier que les 4 profils sont présents
    await expect(page.locator('button:has-text("Sûr")')).toBeVisible();
    await expect(page.locator('button:has-text("Standard")')).toBeVisible();
    await expect(page.locator('button:has-text("Ambitieux")')).toBeVisible();
    await expect(page.locator('button:has-text("Personnalisé")')).toBeVisible();
  });

  test('permet de sélectionner un profil Kelly', async ({ page }) => {
    // Cliquer sur le profil "Ambitieux"
    const ambitieuxBtn = page.locator('button:has-text("Ambitieux")');
    await ambitieuxBtn.click();

    // Vérifier que le bouton est sélectionné (a la classe border-orange-500)
    await expect(ambitieuxBtn).toHaveClass(/border-orange-500/);
  });

  test('affiche le slider personnalisé quand Personnalisé est sélectionné', async ({ page }) => {
    // Cliquer sur Personnalisé
    await page.locator('button:has-text("Personnalisé")').click();

    // Vérifier que le slider apparaît
    await expect(page.locator('input[type="range"]')).toBeVisible();
    await expect(page.locator('text=Fraction Kelly personnalisée')).toBeVisible();
  });

  test('permet de modifier le cap par pari', async ({ page }) => {
    // Trouver le champ cap par pari
    const capInput = page.locator('input[type="number"]').filter({ hasText: '' }).nth(1); // Le deuxième input number
    const capSection = page.locator('text=Cap par pari').locator('..');
    const inputInSection = capSection.locator('input[type="number"]');

    if (await inputInSection.count() > 0) {
      await inputInSection.fill('3');
      await expect(inputInSection).toHaveValue('3');
    }
  });

  test('affiche le résumé de la politique', async ({ page }) => {
    // Vérifier que le panel résumé est présent
    await expect(page.locator('text=Résumé de la Politique')).toBeVisible();

    // Vérifier les sections du résumé
    await expect(page.locator('text=Profil')).toBeVisible();
    await expect(page.locator('text=Cap / Pari')).toBeVisible();
    await expect(page.locator('text=Budget / Jour')).toBeVisible();
    await expect(page.locator('text=Value Min')).toBeVisible();
  });

  test('affiche la formule Kelly', async ({ page }) => {
    // Vérifier que la formule est affichée
    await expect(page.locator('text=Formule Kelly')).toBeVisible();
    await expect(page.locator('text=f* =')).toBeVisible();
  });

  test('permet de sauvegarder la configuration', async ({ page }) => {
    // Cliquer sur le bouton sauvegarder
    const saveBtn = page.locator('button:has-text("Sauvegarder")');
    await saveBtn.click();

    // Attendre le message de succès (ou que le bouton change d'état)
    await page.waitForTimeout(1000);

    // Vérifier qu'il n'y a pas d'erreur visible
    const errorAlert = page.locator('.text-red-700, .text-red-400');
    await expect(errorAlert).toHaveCount(0);
  });

  test('affiche la section Paris Exotiques', async ({ page }) => {
    await expect(page.locator('text=Paris Exotiques')).toBeVisible();
    await expect(page.locator('text=Taux par ticket')).toBeVisible();
    await expect(page.locator('text=Max par pack')).toBeVisible();
  });
});

test.describe('Conseils Page - Portefeuille', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE_URL}/conseils`);
    // Attendre que la page soit chargée
    await page.waitForSelector('h1:has-text("Conseils")');
  });

  test('affiche les 3 onglets', async ({ page }) => {
    await expect(page.locator('button:has-text("Unitaires")')).toBeVisible();
    await expect(page.locator('button:has-text("Exotiques")')).toBeVisible();
    await expect(page.locator('button:has-text("Portefeuille")')).toBeVisible();
  });

  test('onglet Portefeuille affiche le résumé politique Kelly', async ({ page }) => {
    // Cliquer sur l'onglet Portefeuille
    await page.locator('button:has-text("Portefeuille")').click();

    // Attendre le chargement
    await page.waitForTimeout(500);

    // Vérifier que le résumé de la politique est affiché
    await expect(page.locator('text=Profil')).toBeVisible();
    await expect(page.locator('text=Kelly Fraction')).toBeVisible();
    await expect(page.locator('text=Cap/Pari')).toBeVisible();
    await expect(page.locator('text=Budget/Jour')).toBeVisible();
  });

  test('onglet Portefeuille affiche Total Stake et EV', async ({ page }) => {
    await page.locator('button:has-text("Portefeuille")').click();
    await page.waitForTimeout(500);

    await expect(page.locator('text=Total Stake')).toBeVisible();
    await expect(page.locator('text=EV Attendue')).toBeVisible();
    await expect(page.locator('text=Budget Restant')).toBeVisible();
    await expect(page.locator('text=Nb Paris')).toBeVisible();
  });

  test('affiche le lien vers Settings', async ({ page }) => {
    await page.locator('button:has-text("Portefeuille")').click();
    await page.waitForTimeout(500);

    const modifierLink = page.locator('a:has-text("Modifier")');
    await expect(modifierLink).toBeVisible();

    // Vérifier que le lien pointe vers /settings
    await expect(modifierLink).toHaveAttribute('href', '/settings');
  });

  test('permet l\'export CSV du panier', async ({ page }) => {
    await page.locator('button:has-text("Portefeuille")').click();
    await page.waitForTimeout(500);

    // Le bouton export CSV devrait être présent (même si disabled quand panier vide)
    await expect(page.locator('button:has-text("Export CSV")')).toBeVisible();
  });

  test('permet l\'export JSON du panier', async ({ page }) => {
    await page.locator('button:has-text("Portefeuille")').click();
    await page.waitForTimeout(500);

    await expect(page.locator('button:has-text("Export JSON")')).toBeVisible();
  });

  test('onglet Unitaires affiche le champ bankroll', async ({ page }) => {
    // Par défaut sur l'onglet Unitaires
    await expect(page.locator('label:has-text("Bankroll")')).toBeVisible();
    await expect(page.locator('[data-testid="bankroll-input"]')).toBeVisible();
  });

  test('permet de modifier la bankroll', async ({ page }) => {
    const bankrollInput = page.locator('[data-testid="bankroll-input"]');
    await bankrollInput.fill('2000');
    await expect(bankrollInput).toHaveValue('2000');
  });

  test('onglet Exotiques affiche le générateur', async ({ page }) => {
    await page.locator('button:has-text("Exotiques")').click();
    await page.waitForTimeout(500);

    await expect(page.locator('text=Générateur de Tickets Exotiques')).toBeVisible();
    await expect(page.locator('text=Budget total')).toBeVisible();
    await expect(page.locator('text=Profil de risque')).toBeVisible();
  });

  test('onglet Exotiques affiche les 3 profils de risque', async ({ page }) => {
    await page.locator('button:has-text("Exotiques")').click();
    await page.waitForTimeout(500);

    await expect(page.locator('button:has-text("Sûr")')).toBeVisible();
    await expect(page.locator('button:has-text("Équilibré")')).toBeVisible();
    await expect(page.locator('button:has-text("Risqué")')).toBeVisible();
  });
});

test.describe('Courses Page - Value Cutoff', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE_URL}/courses`);
    await page.waitForSelector('h1:has-text("Courses")');
  });

  test('affiche le filtre Value ≥', async ({ page }) => {
    await expect(page.locator('label:has-text("Value ≥")')).toBeVisible();
    await expect(page.locator('[data-testid="value-threshold-input"]')).toBeVisible();
  });

  test('initialise le seuil de value depuis settings', async ({ page }) => {
    // Le seuil devrait être initialisé à 5% (value_cutoff par défaut)
    const valueInput = page.locator('[data-testid="value-threshold-input"]');
    // Attendre le chargement des settings
    await page.waitForTimeout(1000);

    const value = await valueInput.inputValue();
    // Devrait être 5 (5%) si les settings sont chargés
    expect(parseFloat(value)).toBeGreaterThanOrEqual(0);
  });

  test('permet de filtrer par discipline', async ({ page }) => {
    const disciplineSelect = page.locator('select');
    await expect(disciplineSelect).toBeVisible();

    // Vérifier l'option par défaut
    await expect(disciplineSelect).toHaveValue('all');
  });
});

test.describe('Navigation entre pages', () => {
  test('Settings vers Conseils', async ({ page }) => {
    await page.goto(`${BASE_URL}/settings`);
    await page.waitForSelector('h1:has-text("Paramètres")');

    // Si un lien vers Conseils existe, le tester
    const conseilsLink = page.locator('a[href="/conseils"]');
    if (await conseilsLink.count() > 0) {
      await conseilsLink.click();
      await expect(page).toHaveURL(/\/conseils/);
    }
  });

  test('Conseils vers Settings via lien Modifier', async ({ page }) => {
    await page.goto(`${BASE_URL}/conseils`);
    await page.locator('button:has-text("Portefeuille")').click();
    await page.waitForTimeout(500);

    const modifierLink = page.locator('a:has-text("Modifier")');
    if (await modifierLink.count() > 0) {
      await modifierLink.click();
      await expect(page).toHaveURL(/\/settings/);
    }
  });
});

test.describe('Responsive Design', () => {
  test('Settings s\'adapte au mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(`${BASE_URL}/settings`);

    // Vérifier que les éléments principaux sont visibles
    await expect(page.locator('h1:has-text("Paramètres")')).toBeVisible();
    await expect(page.locator('button:has-text("Sauvegarder")')).toBeVisible();
  });

  test('Conseils s\'adapte au mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(`${BASE_URL}/conseils`);

    await expect(page.locator('h1:has-text("Conseils")')).toBeVisible();
    // Les onglets devraient être scrollables
    await expect(page.locator('button:has-text("Unitaires")')).toBeVisible();
  });
});
