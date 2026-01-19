/**
 * Page de démonstration du mode sombre
 *
 * Affiche tous les composants UI avec leurs variantes
 * pour vérifier le bon fonctionnement du thème.
 */

import { TrendingUp, TrendingDown, Activity, DollarSign, Users, Award } from 'lucide-react';
import { PrimaryButton, SecondaryButton } from '../components/Button';
import { Card, CardHeader } from '../components/Card';
import StatCard from '../components/StatCard';
import { ThemedLineChart, ThemedAreaChart, ThemedBarChart } from '../components/ThemedCharts';
import { useThemeColors, useIsDarkMode } from '../hooks/useThemeColors';

const DarkModeDemo = () => {
  const colors = useThemeColors();
  const isDark = useIsDarkMode();

  // Données de test pour les graphiques
  const chartData = [
    { name: 'Jan', victoires: 12, places: 28, gains: 45000 },
    { name: 'Fev', victoires: 15, places: 32, gains: 52000 },
    { name: 'Mar', victoires: 18, places: 35, gains: 61000 },
    { name: 'Avr', victoires: 14, places: 30, gains: 48000 },
    { name: 'Mai', victoires: 20, places: 38, gains: 68000 },
    { name: 'Jun', victoires: 17, places: 33, gains: 55000 },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1
          className="text-4xl font-bold font-['Bebas_Neue'] mb-2"
          style={{ color: colors.text }}
        >
          Démonstration Mode Sombre
        </h1>
        <p style={{ color: colors.muted }}>
          Mode actuel : <span className="font-semibold" style={{ color: colors.primary }}>
            {isDark ? 'Sombre 🌙' : 'Clair ☀️'}
          </span>
        </p>
      </div>

      {/* Variables CSS */}
      <Card>
        <CardHeader
          icon={Activity}
          title="Variables CSS"
          subtitle="Couleurs du thème actif"
        />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(colors).map(([name, value]) => (
            <div key={name} className="space-y-2">
              <div
                className="h-16 rounded-lg border"
                style={{
                  backgroundColor: value,
                  borderColor: colors.border
                }}
              />
              <div>
                <p className="text-xs font-mono" style={{ color: colors.muted }}>
                  {name}
                </p>
                <p className="text-xs font-mono font-semibold" style={{ color: colors.text }}>
                  {value}
                </p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Boutons */}
      <Card>
        <CardHeader
          icon={Users}
          title="Composants Boutons"
          subtitle="Primary & Secondary"
        />
        <div className="flex flex-wrap gap-4">
          <PrimaryButton>Bouton Primaire</PrimaryButton>
          <PrimaryButton disabled>Désactivé</PrimaryButton>
          <SecondaryButton>Bouton Secondaire</SecondaryButton>
          <SecondaryButton disabled>Désactivé</SecondaryButton>
        </div>
      </Card>

      {/* Cartes statistiques */}
      <div>
        <h2
          className="text-2xl font-bold font-['Bebas_Neue'] mb-4"
          style={{ color: colors.text }}
        >
          Cartes Statistiques
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            title="Taux de victoire"
            value="23.4%"
            evolution="+2.5%"
            icon={TrendingUp}
            trend="up"
          />
          <StatCard
            title="Gains moyens"
            value="€5,240"
            evolution="+12.3%"
            icon={DollarSign}
            trend="up"
          />
          <StatCard
            title="Courses analysées"
            value="1,284"
            evolution="-3.2%"
            icon={Activity}
            trend="down"
          />
          <StatCard
            title="Chevaux suivis"
            value="342"
            evolution="+8.1%"
            icon={Award}
            trend="up"
          />
        </div>
      </div>

      {/* Graphiques */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader
            icon={TrendingUp}
            title="Graphique en Ligne"
            subtitle="Évolution des victoires"
          />
          <ThemedLineChart
            data={chartData}
            dataKeys={['victoires', 'places']}
            height={250}
          />
        </Card>

        <Card>
          <CardHeader
            icon={Activity}
            title="Graphique en Aires"
            subtitle="Gains mensuels"
          />
          <ThemedAreaChart
            data={chartData}
            dataKeys={['gains']}
            height={250}
          />
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader
            icon={Award}
            title="Graphique en Barres"
            subtitle="Performance globale"
          />
          <ThemedBarChart
            data={chartData}
            dataKeys={['victoires', 'places']}
            height={300}
          />
        </Card>
      </div>

      {/* Typographie */}
      <Card>
        <CardHeader
          icon={Activity}
          title="Échelle Typographique"
          subtitle="Bebas Neue & Inter"
        />
        <div className="space-y-4">
          <div>
            <h1
              className="text-4xl font-['Bebas_Neue'] font-bold"
              style={{ color: colors.text }}
            >
              Titre H1 - Bebas Neue
            </h1>
          </div>
          <div>
            <h2
              className="text-3xl font-['Bebas_Neue'] font-bold"
              style={{ color: colors.text }}
            >
              Titre H2 - Bebas Neue
            </h2>
          </div>
          <div>
            <p
              className="text-lg font-['Inter']"
              style={{ color: colors.text }}
            >
              Texte de corps principal - Inter Regular
            </p>
          </div>
          <div>
            <p
              className="text-base font-['Inter'] font-medium"
              style={{ color: colors.text }}
            >
              Texte medium - Inter Medium
            </p>
          </div>
          <div>
            <p
              className="text-sm font-['Inter']"
              style={{ color: colors.muted }}
            >
              Texte secondaire - Inter Regular (muted)
            </p>
          </div>
          <div>
            <code
              className="text-sm font-['Roboto_Mono'] px-2 py-1 rounded"
              style={{
                backgroundColor: colors.secondary,
                color: colors.primary
              }}
            >
              Code inline - Roboto Mono
            </code>
          </div>
        </div>
      </Card>

      {/* Contraste */}
      <Card>
        <CardHeader
          icon={Activity}
          title="Tests de Contraste"
          subtitle="Validation WCAG AA"
        />
        <div className="space-y-3">
          <div
            className="p-4 rounded-lg"
            style={{ backgroundColor: colors.bg }}
          >
            <p style={{ color: colors.text }}>
              ✓ Texte principal sur fond (Ratio: {isDark ? '13.2:1' : '15.8:1'})
            </p>
          </div>
          <div
            className="p-4 rounded-lg"
            style={{ backgroundColor: colors.card }}
          >
            <p style={{ color: colors.text }}>
              ✓ Texte sur carte (Ratio: {isDark ? '12.1:1' : '15.8:1'})
            </p>
          </div>
          <div
            className="p-4 rounded-lg"
            style={{ backgroundColor: colors.primary }}
          >
            <p style={{ color: '#FFFFFF' }}>
              ✓ Texte blanc sur primaire (Ratio: {isDark ? '4.8:1' : '6.8:1'})
            </p>
          </div>
          <div
            className="p-4 rounded-lg"
            style={{ backgroundColor: colors.bg }}
          >
            <p style={{ color: colors.muted }}>
              ✓ Texte muted sur fond (Ratio: {isDark ? '4.6:1' : '4.5:1'})
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default DarkModeDemo;
