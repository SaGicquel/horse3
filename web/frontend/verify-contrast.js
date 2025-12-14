#!/usr/bin/env node

/**
 * Script de vérification des contrastes WCAG 2.1
 * Vérifie automatiquement tous les ratios de contraste dans les composants
 */

// Fonction pour calculer la luminance relative
function getLuminance(r, g, b) {
  const [rs, gs, bs] = [r, g, b].map(val => {
    val = val / 255;
    return val <= 0.03928 ? val / 12.92 : Math.pow((val + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

// Fonction pour calculer le ratio de contraste
function getContrastRatio(rgb1, rgb2) {
  const lum1 = getLuminance(...rgb1);
  const lum2 = getLuminance(...rgb2);
  const brightest = Math.max(lum1, lum2);
  const darkest = Math.min(lum1, lum2);
  return (brightest + 0.05) / (darkest + 0.05);
}

// Conversion hex vers RGB
function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? [
    parseInt(result[1], 16),
    parseInt(result[2], 16),
    parseInt(result[3], 16)
  ] : null;
}

// Définition des couleurs de la palette
const colors = {
  // Palette principale
  primary: '#9D3656',
  primaryHover: '#812C47',
  accent: '#F5C3CE',
  
  // Couleurs de fond
  white: '#FFFFFF',
  dark: '#0F172A',
  
  // Échelle de gris
  gray900: '#111827',
  gray700: '#374151',
  gray600: '#4B5563',
  gray400: '#9CA3AF',
  gray300: '#D1D5DB',
  
  // Couleurs sémantiques
  red600: '#DC2626',
  red500: '#EF4444',
};

// Définition des tests de contraste
const contrastTests = [
  // Boutons primaires
  { name: 'Bouton primaire (texte blanc sur #9D3656)', fg: 'white', bg: 'primary', minRatio: 4.5, level: 'AA' },
  { name: 'Bouton primaire hover (texte blanc sur #812C47)', fg: 'white', bg: 'primaryHover', minRatio: 4.5, level: 'AA' },
  
  // Boutons secondaires
  { name: 'Bouton secondaire (#9D3656 sur blanc)', fg: 'primary', bg: 'white', minRatio: 4.5, level: 'AA' },
  { name: 'Bouton secondaire hover (#9D3656 sur #F5C3CE)', fg: 'primary', bg: 'accent', minRatio: 4.5, level: 'AA' },
  
  // Focus rings
  { name: 'Focus ring (#F5C3CE sur blanc)', fg: 'accent', bg: 'white', minRatio: 3.0, level: 'AA Large' },
  
  // Texte sur fond blanc (Card)
  { name: 'Texte principal (gray-900 sur blanc)', fg: 'gray900', bg: 'white', minRatio: 7.0, level: 'AAA' },
  { name: 'Texte secondaire (gray-600 sur blanc)', fg: 'gray600', bg: 'white', minRatio: 7.0, level: 'AAA' },
  { name: 'Labels (gray-700 sur blanc)', fg: 'gray700', bg: 'white', minRatio: 7.0, level: 'AAA' },
  { name: 'Placeholders (gray-400 sur blanc)', fg: 'gray400', bg: 'white', minRatio: 4.5, level: 'AA' },
  
  // Borders
  { name: 'Border input (gray-300 sur blanc)', fg: 'gray300', bg: 'white', minRatio: 3.0, level: 'AA Large' },
  
  // StatCard - tendance positive
  { name: 'StatCard positif - icône (#9D3656 sur bg-[#9D3656]/15)', fg: 'primary', bg: 'accent', minRatio: 3.0, level: 'AA Large' },
  { name: 'StatCard positif - texte (#9D3656 sur blanc)', fg: 'primary', bg: 'white', minRatio: 4.5, level: 'AA' },
  
  // StatCard - tendance négative
  { name: 'StatCard négatif - texte (red-600 sur blanc)', fg: 'red600', bg: 'white', minRatio: 4.5, level: 'AA' },
  
  // Navigation (fond sombre)
  { name: 'Navigation - titre (blanc sur #0F172A)', fg: 'white', bg: 'dark', minRatio: 7.0, level: 'AAA' },
];

// Fonction pour obtenir le niveau WCAG
function getWCAGLevel(ratio, minRatio) {
  if (ratio >= 7.0) return 'AAA';
  if (ratio >= 4.5) return 'AA';
  if (ratio >= 3.0) return 'AA Large';
  return 'FAIL';
}

// Exécution des tests
console.log('\n🎨 VÉRIFICATION DES CONTRASTES WCAG 2.1\n');
console.log('='.repeat(80));
console.log('\n');

let passed = 0;
let failed = 0;
const results = [];

contrastTests.forEach(test => {
  const fgRgb = hexToRgb(colors[test.fg]);
  const bgRgb = hexToRgb(colors[test.bg]);
  const ratio = getContrastRatio(fgRgb, bgRgb);
  const level = getWCAGLevel(ratio, test.minRatio);
  const status = ratio >= test.minRatio ? '✅ PASS' : '❌ FAIL';
  
  if (ratio >= test.minRatio) {
    passed++;
  } else {
    failed++;
  }
  
  results.push({
    name: test.name,
    ratio: ratio.toFixed(2),
    level,
    expectedLevel: test.level,
    status,
    fg: colors[test.fg],
    bg: colors[test.bg]
  });
});

// Affichage des résultats
results.forEach(result => {
  console.log(`${result.status} ${result.name}`);
  console.log(`   Ratio: ${result.ratio}:1 (${result.level}) - Requis: ${result.expectedLevel}`);
  console.log(`   Couleurs: ${result.fg} sur ${result.bg}`);
  console.log('');
});

// Résumé
console.log('='.repeat(80));
console.log(`\n📊 RÉSUMÉ\n`);
console.log(`Total des tests: ${contrastTests.length}`);
console.log(`✅ Réussis: ${passed}`);
console.log(`❌ Échoués: ${failed}`);
console.log(`📈 Taux de réussite: ${((passed / contrastTests.length) * 100).toFixed(1)}%\n`);

if (failed === 0) {
  console.log('🎉 PARFAIT ! Tous les contrastes respectent les normes WCAG 2.1\n');
  process.exit(0);
} else {
  console.log('⚠️  ATTENTION : Certains contrastes ne respectent pas les normes WCAG 2.1\n');
  process.exit(1);
}
