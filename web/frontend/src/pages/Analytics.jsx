import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { TrendingUp, Users, Award, Calendar, MapPin, RefreshCw, BarChart3, ChevronRight, Layers, DollarSign, Target } from 'lucide-react';
import { analyticsAPI } from '../services/api';
import { useThemeColors } from '../hooks/useThemeColors';
import { GlassCard, GlassCardHeader } from '../components/GlassCard';
import PageHeader from '../components/PageHeader';
import { StaggerContainer, StaggerItem, ScrollReveal } from '../components/PageTransition';
import { SkeletonChart, Skeleton } from '../components/Skeleton';
import { API_BASE } from '../config/api';

// Palette de couleurs pour les graphiques
const COLORS = ['#9D3656', '#2ED573', '#DC2626', '#CA6384', '#D84A78', '#E75B8C'];

// Tooltip personnalisé adaptatif au thème avec animation
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        className="p-4 rounded-xl shadow-2xl backdrop-blur-xl"
        style={{
          backgroundColor: 'rgba(var(--color-card-rgb), 0.9)',
          border: '1px solid var(--color-border)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
        }}
      >
        <p
          className="font-bold mb-3 text-sm"
          style={{ color: 'var(--color-text)' }}
        >
          {label}
        </p>
        {payload.map((entry, index) => (
          <motion.p
            key={index}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="text-sm mt-2 flex items-center gap-2"
            style={{ color: entry.color || entry.fill }}
          >
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: entry.color || entry.fill }}
            />
            {entry.name}: <strong className="ml-1">{entry.value}</strong>
            {entry.unit || ''}
          </motion.p>
        ))}
      </motion.div>
    );
  }
  return null;
};

// Composant pour afficher un message quand il n'y a pas de données
const EmptyState = ({ message, description }) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center justify-center py-12 px-4"
    >
      <motion.div
        className="rounded-full p-5 mb-4"
        style={{
          backgroundColor: 'var(--color-secondary)',
          border: '1px solid var(--color-border)'
        }}
        animate={{
          scale: [1, 1.05, 1],
          rotate: [0, 5, -5, 0]
        }}
        transition={{ duration: 3, repeat: Infinity }}
      >
        <TrendingUp size={28} style={{ color: 'var(--color-muted)' }} />
      </motion.div>
      <p
        className="text-base font-semibold mb-2"
        style={{ color: 'var(--color-text)' }}
      >
        {message}
      </p>
      <p
        className="text-sm text-center max-w-md"
        style={{ color: 'var(--color-muted)' }}
      >
        {description}
      </p>
    </motion.div>
  );
};

// Squelette de chargement pour Analytics
const AnalyticsSkeleton = () => (
  <div className="mx-auto max-w-6xl space-y-6 px-3 sm:px-6 py-6 sm:py-12">
    <div className="space-y-3">
      <Skeleton className="h-4 w-32" />
      <Skeleton className="h-10 w-48" />
      <Skeleton className="h-4 w-96" />
    </div>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <SkeletonChart />
      <SkeletonChart />
    </div>
    <SkeletonChart />
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <SkeletonChart />
      <SkeletonChart />
    </div>
  </div>
);

// Couleurs pour les buckets de risque
const BUCKET_COLORS = {
  'SÛR': '#22C55E',
  'ÉQUIL.': '#FBBF24',
  'RISQ.': '#EF4444'
};

const Analytics = () => {
  const [chevauxData, setChevauxData] = useState(null);
  const [jockeysData, setJockeysData] = useState(null);
  const [entraineursData, setEntraineursData] = useState(null);
  const [hippodromesData, setHippodromesData] = useState(null);
  const [evolutionData, setEvolutionData] = useState(null);
  const [tauxParRaceData, setTauxParRaceData] = useState(null);
  const [oddsData, setOddsData] = useState(null);
  const [distanceData, setDistanceData] = useState(null);
  const [ageData, setAgeData] = useState(null);
  const [loading, setLoading] = useState(true);
  const colors = useThemeColors();

  // EV par décile de value
  const [evDecileData, setEvDecileData] = useState(null);
  const [evDecileLoading, setEvDecileLoading] = useState(true);

  // Buckets par contexte
  const [bucketsData, setBucketsData] = useState(null);
  const [bucketsLoading, setBucketsLoading] = useState(true);
  const [bucketsGroupBy, setBucketsGroupBy] = useState('discipline');

  // Charger EV par décile de value
  const loadEvDecileData = useCallback(async () => {
    try {
      setEvDecileLoading(true);
      const response = await fetch(`${API_BASE}/metrics/ev_by_value_decile`);
      if (response.ok) {
        const data = await response.json();
        setEvDecileData(data.deciles || data);
      } else {
        // Fallback: générer données de démo
        setEvDecileData(generateFallbackEvDecile());
      }
    } catch (error) {
      console.error('Erreur chargement EV décile:', error);
      setEvDecileData(generateFallbackEvDecile());
    } finally {
      setEvDecileLoading(false);
    }
  }, []);

  // Charger répartition des buckets
  const loadBucketsData = useCallback(async () => {
    try {
      setBucketsLoading(true);
      const response = await fetch(`${API_BASE}/metrics/buckets_by_context`);
      if (response.ok) {
        const data = await response.json();
        setBucketsData(data);
      } else {
        // Fallback: générer données de démo
        setBucketsData(generateFallbackBuckets());
      }
    } catch (error) {
      console.error('Erreur chargement buckets:', error);
      setBucketsData(generateFallbackBuckets());
    } finally {
      setBucketsLoading(false);
    }
  }, []);

  // Données de fallback si API non disponible
  const generateFallbackEvDecile = () => [
    { decile: '0-10%', ev: -8.2, count: 45 },
    { decile: '10-20%', ev: -3.5, count: 62 },
    { decile: '20-30%', ev: 1.2, count: 78 },
    { decile: '30-40%', ev: 5.8, count: 89 },
    { decile: '40-50%', ev: 9.4, count: 95 },
    { decile: '50-60%', ev: 12.1, count: 72 },
    { decile: '60-70%', ev: 15.3, count: 54 },
    { decile: '70-80%', ev: 18.7, count: 38 },
    { decile: '80-90%', ev: 22.4, count: 21 },
    { decile: '90-100%', ev: 28.9, count: 12 }
  ];

  const generateFallbackBuckets = () => ({
    byDiscipline: [
      { name: 'PLAT', 'SÛR': 45, 'ÉQUIL.': 35, 'RISQ.': 20 },
      { name: 'TROT', 'SÛR': 38, 'ÉQUIL.': 42, 'RISQ.': 20 },
      { name: 'ATTELÉ', 'SÛR': 42, 'ÉQUIL.': 38, 'RISQ.': 20 },
      { name: 'HAIES', 'SÛR': 30, 'ÉQUIL.': 40, 'RISQ.': 30 },
      { name: 'STEEPLE', 'SÛR': 25, 'ÉQUIL.': 35, 'RISQ.': 40 }
    ],
    byHippodrome: [
      { name: 'LONGCHAMP', 'SÛR': 48, 'ÉQUIL.': 32, 'RISQ.': 20 },
      { name: 'VINCENNES', 'SÛR': 52, 'ÉQUIL.': 30, 'RISQ.': 18 },
      { name: 'CHANTILLY', 'SÛR': 40, 'ÉQUIL.': 38, 'RISQ.': 22 },
      { name: 'DEAUVILLE', 'SÛR': 35, 'ÉQUIL.': 40, 'RISQ.': 25 },
      { name: 'AUTEUIL', 'SÛR': 28, 'ÉQUIL.': 42, 'RISQ.': 30 }
    ]
  });

  // Données buckets à afficher selon le toggle
  const currentBucketsData = useMemo(() => {
    if (!bucketsData) return [];
    return bucketsGroupBy === 'discipline'
      ? (bucketsData.byDiscipline || bucketsData.by_discipline || [])
      : (bucketsData.byHippodrome || bucketsData.by_hippodrome || []);
  }, [bucketsData, bucketsGroupBy]);

  useEffect(() => {
    loadAnalyticsData();
    loadEvDecileData();
    loadBucketsData();
  }, [loadEvDecileData, loadBucketsData]);

  const loadAnalyticsData = async () => {
    try {
      setLoading(true);
      // Utiliser Promise.allSettled pour que les graphiques s'affichent même si certains endpoints échouent
      const results = await Promise.allSettled([
        analyticsAPI.getAnalyticsChevaux(),
        analyticsAPI.getAnalyticsJockeys(),
        analyticsAPI.getAnalyticsEntraineurs(),
        analyticsAPI.getAnalyticsHippodromes(),
        analyticsAPI.getAnalyticsEvolution(),
        analyticsAPI.getTauxParRace(),
        analyticsAPI.getAnalyticsOdds(),
        analyticsAPI.getAnalyticsDistance(),
        analyticsAPI.getAnalyticsAge(),
      ]);

      // Extraire les valeurs des promesses résolues (null si échouée)
      const getValue = (result) => result.status === 'fulfilled' ? result.value : null;

      setChevauxData(getValue(results[0]));
      setJockeysData(getValue(results[1]));
      setEntraineursData(getValue(results[2]));
      setHippodromesData(getValue(results[3]));
      setEvolutionData(getValue(results[4]));
      setTauxParRaceData(getValue(results[5]));
      setOddsData(getValue(results[6]));
      setDistanceData(getValue(results[7]));
      setAgeData(getValue(results[8]));

      // Log des erreurs pour debug
      results.forEach((result, index) => {
        if (result.status === 'rejected') {
          const endpoints = ['chevaux', 'jockeys', 'entraineurs', 'hippodromes', 'evolution', 'tauxParRace', 'odds', 'distance', 'age'];
          console.warn(`Analytics ${endpoints[index]} failed:`, result.reason);
        }
      });
    } catch (error) {
      console.error('Erreur chargement analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <AnalyticsSkeleton />;
  }

  return (
    <div className="mx-auto max-w-7xl space-y-6 sm:space-y-10 px-3 sm:px-6 py-6 sm:py-12">
      {/* Header unifié */}
      <PageHeader
        emoji="📊"
        title="Analytics"
        subtitle="Analyses détaillées et approfondies de toutes les données historiques. Pour une vue d'ensemble rapide, consultez le Dashboard."
      />

      {/* Section EV & Buckets - Nouvelles visualisations */}
      <StaggerContainer className="grid grid-cols-1 gap-4 sm:gap-6 lg:grid-cols-2">
        {/* Chart 1: EV moyen par décile de value */}
        <StaggerItem>
          <GlassCard className="h-full">
            <GlassCardHeader
              icon={DollarSign}
              title="EV par décile de Value"
              subtitle="Expected Value selon le % value"
              iconColor="#22C55E"
              iconBg="rgba(34, 197, 94, 0.15)"
            />

            {evDecileLoading ? (
              <SkeletonChart />
            ) : evDecileData && evDecileData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={evDecileData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                    <XAxis
                      dataKey="decile"
                      stroke={colors.muted}
                      tick={{ fill: colors.text, fontSize: 10 }}
                      angle={-35}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis
                      stroke={colors.muted}
                      tick={{ fill: colors.text, fontSize: 11 }}
                      tickFormatter={(v) => `${v}%`}
                    />
                    <Tooltip
                      content={<CustomTooltip />}
                      formatter={(value) => [`${value.toFixed(1)}%`, 'EV']}
                    />
                    <Bar
                      dataKey="ev"
                      name="EV (%)"
                      radius={[6, 6, 0, 0]}
                    >
                      {evDecileData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.ev >= 0 ? '#22C55E' : '#EF4444'}
                          fillOpacity={0.85}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="mt-3 flex justify-end">
                  <Link
                    to="/conseils"
                    className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg transition-all hover:scale-105"
                    style={{
                      color: colors.primary,
                      background: 'rgba(var(--color-primary-rgb), 0.1)',
                      border: '1px solid rgba(var(--color-primary-rgb), 0.2)'
                    }}
                  >
                    Voir détails <ChevronRight size={14} />
                  </Link>
                </div>
              </>
            ) : (
              <EmptyState
                message="Données non disponibles"
                description="Les métriques EV par décile seront calculées une fois les prédictions historiques enregistrées."
              />
            )}
          </GlassCard>
        </StaggerItem>

        {/* Chart 2: Répartition des buckets par discipline/hippodrome */}
        <StaggerItem>
          <GlassCard className="h-full">
            <div className="flex items-center justify-between mb-4">
              <GlassCardHeader
                icon={Layers}
                title="Répartition des Buckets"
                subtitle={bucketsGroupBy === 'discipline' ? 'Par discipline' : 'Par hippodrome'}
                iconColor="#8B5CF6"
                iconBg="rgba(139, 92, 246, 0.15)"
              />

              {/* Toggle discipline/hippodrome */}
              <div
                className="flex rounded-lg overflow-hidden text-xs"
                style={{
                  border: '1px solid var(--color-border)',
                  background: 'var(--color-bg)'
                }}
              >
                <button
                  onClick={() => setBucketsGroupBy('discipline')}
                  className="px-3 py-1.5 transition-all"
                  style={{
                    background: bucketsGroupBy === 'discipline' ? colors.primary : 'transparent',
                    color: bucketsGroupBy === 'discipline' ? '#fff' : colors.muted
                  }}
                >
                  Discipline
                </button>
                <button
                  onClick={() => setBucketsGroupBy('hippodrome')}
                  className="px-3 py-1.5 transition-all"
                  style={{
                    background: bucketsGroupBy === 'hippodrome' ? colors.primary : 'transparent',
                    color: bucketsGroupBy === 'hippodrome' ? '#fff' : colors.muted
                  }}
                >
                  Hippodrome
                </button>
              </div>
            </div>

            {bucketsLoading ? (
              <SkeletonChart />
            ) : currentBucketsData && currentBucketsData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={currentBucketsData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                    <XAxis
                      dataKey="name"
                      stroke={colors.muted}
                      tick={{ fill: colors.text, fontSize: 10 }}
                      angle={-35}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis
                      stroke={colors.muted}
                      tick={{ fill: colors.text, fontSize: 11 }}
                      tickFormatter={(v) => `${v}%`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                      wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
                      formatter={(value) => <span style={{ color: colors.text }}>{value}</span>}
                    />
                    <Bar dataKey="SÛR" stackId="a" fill={BUCKET_COLORS['SÛR']} name="SÛR" radius={[0, 0, 0, 0]} />
                    <Bar dataKey="ÉQUIL." stackId="a" fill={BUCKET_COLORS['ÉQUIL.']} name="ÉQUIL." radius={[0, 0, 0, 0]} />
                    <Bar dataKey="RISQ." stackId="a" fill={BUCKET_COLORS['RISQ.']} name="RISQ." radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="mt-3 flex justify-end">
                  <Link
                    to="/conseils"
                    className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg transition-all hover:scale-105"
                    style={{
                      color: colors.primary,
                      background: 'rgba(var(--color-primary-rgb), 0.1)',
                      border: '1px solid rgba(var(--color-primary-rgb), 0.2)'
                    }}
                  >
                    Voir détails <ChevronRight size={14} />
                  </Link>
                </div>
              </>
            ) : (
              <EmptyState
                message="Données non disponibles"
                description="La répartition des buckets sera calculée une fois les prédictions catégorisées."
              />
            )}
          </GlassCard>
        </StaggerItem>
      </StaggerContainer>

      {/* Distribution par race */}
      <ScrollReveal>
        <GlassCard className="overflow-hidden">
          <GlassCardHeader
            icon={TrendingUp}
            title="Distribution par race"
            subtitle="Répartition des chevaux"
          />

          {chevauxData?.distribution_races && chevauxData.distribution_races.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chevauxData.distribution_races}>
                <defs>
                  <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={colors.primary} stopOpacity={1} />
                    <stop offset="95%" stopColor={colors.primary} stopOpacity={0.6} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                <XAxis
                  dataKey="race"
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <YAxis
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ color: colors.text }} />
                <Bar dataKey="count" fill="url(#barGradient)" name="Nombre de chevaux" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState
              message="Aucune donnée disponible"
              description="La distribution par race sera disponible une fois que les données des chevaux seront enregistrées."
            />
          )}
        </GlassCard>
      </ScrollReveal>

      {/* Top 10 performers */}
      <ScrollReveal>
        <GlassCard className="overflow-hidden">
          <GlassCardHeader
            icon={Award}
            title="Top 10 performers"
            subtitle="Taux de victoire"
            iconColor="#FBBF24"
            iconBg="rgba(251, 191, 36, 0.15)"
          />

          {chevauxData?.top_performers && chevauxData.top_performers.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={chevauxData.top_performers}
                layout="vertical"
                margin={{ left: 60, right: 10 }}
              >
                <defs>
                  <linearGradient id="successGradient" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="5%" stopColor={colors.success} stopOpacity={0.8} />
                    <stop offset="95%" stopColor={colors.success} stopOpacity={1} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                <XAxis
                  type="number"
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <YAxis
                  dataKey="nom"
                  type="category"
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="taux" fill="url(#successGradient)" name="Taux de victoire (%)" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState
              message="Aucune donnée disponible"
              description="Les statistiques des performances seront disponibles une fois que les chevaux auront participé à plusieurs courses."
            />
          )}
        </GlassCard>
      </ScrollReveal>

      {/* Grid Performance par sexe et Top jockeys */}
      <StaggerContainer className="grid grid-cols-1 gap-4 sm:gap-6 lg:grid-cols-2">
        <StaggerItem>
          <GlassCard className="h-full">
            <div className="mb-6 sm:mb-8 flex items-center justify-between">
              <h2
                className="text-base sm:text-lg font-semibold"
                style={{ color: 'var(--color-text)' }}
              >
                Performance par sexe
              </h2>
              <span
                className="rounded-full px-3 sm:px-4 py-1 text-[10px] sm:text-xs uppercase tracking-[0.32em]"
                style={{
                  border: '1px solid var(--color-border)',
                  backgroundColor: 'var(--color-bg)',
                  color: 'var(--color-muted)'
                }}
              >
                Répartition
              </span>
            </div>

            {chevauxData?.stats_par_sexe && chevauxData.stats_par_sexe.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={chevauxData.stats_par_sexe}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ sexe, taux_moyen }) => `${sexe}: ${taux_moyen}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="total"
                  >
                    {chevauxData.stats_par_sexe.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState
                message="Aucune donnée disponible"
                description="Les statistiques par sexe seront disponibles une fois que les données de courses seront enregistrées."
              />
            )}
          </GlassCard>
        </StaggerItem>

        <StaggerItem>
          <GlassCard className="h-full">
            <GlassCardHeader
              icon={Users}
              title="Top jockeys"
              subtitle="Performances cumulées"
            />

            {jockeysData?.top_jockeys && jockeysData.top_jockeys.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={jockeysData.top_jockeys.slice(0, 10)} layout="vertical" margin={{ left: 80, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                  <XAxis type="number" stroke={colors.muted} tick={{ fill: colors.text, fontSize: 12 }} />
                  <YAxis
                    dataKey="nom"
                    type="category"
                    stroke={colors.muted}
                    tick={{ fill: colors.text, fontSize: 11 }}
                    width={70}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="taux_reussite" fill={colors.secondary} name="Taux de réussite (%)" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState
                message="Aucun jockey disponible"
                description="Les données des jockeys seront disponibles une fois que les statistiques de courses seront calculées."
              />
            )}
          </GlassCard>
        </StaggerItem>
      </StaggerContainer>

      {/* Top entraîneurs */}
      <ScrollReveal>
        <GlassCard className="overflow-hidden">
          <GlassCardHeader
            icon={Users}
            title="Top entraîneurs"
            subtitle="Suivi des performances"
          />

          {entraineursData?.top_entraineurs && entraineursData.top_entraineurs.length > 0 ? (
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={entraineursData.top_entraineurs.slice(0, 10)}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                <XAxis
                  dataKey="nom"
                  stroke={colors.muted}
                  angle={-35}
                  textAnchor="end"
                  height={90}
                  tick={{ fill: colors.muted, fontSize: 11 }}
                />
                <YAxis
                  stroke={colors.muted}
                  tick={{ fill: colors.muted, fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend
                  wrapperStyle={{
                    paddingTop: '16px',
                    fontSize: '13px',
                    color: colors.text
                  }}
                />
                <Bar dataKey="taux_reussite" fill={colors.primary} name="Taux de réussite (%)" radius={[8, 8, 0, 0]} />
                <Bar dataKey="nb_chevaux" fill={colors.success} name="Nombre de chevaux" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState
              message="Aucun entraîneur disponible"
              description="Les données des entraîneurs seront disponibles une fois que les statistiques de courses seront calculées."
            />
          )}
        </GlassCard>
      </ScrollReveal>

      {/* Taux de victoire par race */}
      {tauxParRaceData?.taux_par_race && tauxParRaceData.taux_par_race.length > 0 && (
        <ScrollReveal>
          <GlassCard className="overflow-hidden">
            <GlassCardHeader
              icon={TrendingUp}
              title="Taux de victoire par race"
              subtitle="Performance moyenne"
            />

            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={tauxParRaceData.taux_par_race}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                <XAxis
                  dataKey="race"
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 11 }}
                  angle={-35}
                  textAnchor="end"
                  height={80}
                />
                <YAxis
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="taux_moyen" fill={colors.primary} name="Taux de victoire (%)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </GlassCard>
        </ScrollReveal>
      )}

      {/* Évolution des courses dans le temps */}
      {evolutionData?.evolution && evolutionData.evolution.length > 0 && (
        <ScrollReveal>
          <GlassCard className="overflow-hidden">
            <GlassCardHeader
              icon={Calendar}
              title="Évolution dans le temps"
              subtitle="Courses par année"
              iconColor="#3B82F6"
              iconBg="rgba(59, 130, 246, 0.15)"
            />

            <ResponsiveContainer width="100%" height={350}>
              <AreaChart data={evolutionData.evolution.slice().reverse()}>
                <defs>
                  <linearGradient id="coursesGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={colors.primary} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={colors.primary} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="chevauxGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={colors.success} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={colors.success} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                <XAxis
                  dataKey="annee"
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <YAxis
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ color: colors.text }} />
                <Area
                  type="monotone"
                  dataKey="nb_courses"
                  stroke={colors.primary}
                  strokeWidth={3}
                  fill="url(#coursesGradient)"
                  name="Nombre de courses"
                />
                <Area
                  type="monotone"
                  dataKey="nb_chevaux"
                  stroke={colors.success}
                  strokeWidth={3}
                  fill="url(#chevauxGradient)"
                  name="Nombre de chevaux"
                />
                <Line
                  type="monotone"
                  dataKey="nb_victoires"
                  stroke="#FBBF24"
                  strokeWidth={3}
                  name="Nombre de victoires"
                  dot={{ fill: '#FBBF24', r: 5 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </GlassCard>
        </ScrollReveal>
      )}

      {/* Top hippodromes */}
      {hippodromesData?.hippodromes && hippodromesData.hippodromes.length > 0 && (
        <ScrollReveal>
          <GlassCard className="overflow-hidden">
            <GlassCardHeader
              icon={MapPin}
              title="Top hippodromes"
              subtitle="Par nombre de courses"
              iconColor="#22C55E"
              iconBg="rgba(34, 197, 94, 0.15)"
            />

            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={hippodromesData.hippodromes}
                layout="vertical"
                margin={{ left: 80, right: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                <XAxis type="number" stroke={colors.muted} tick={{ fill: colors.text, fontSize: 12 }} />
                <YAxis
                  dataKey="nom"
                  type="category"
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 11 }}
                  width={70}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="nb_courses" fill={colors.primary} name="Nombre de courses" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </GlassCard>
        </ScrollReveal>
      )}

      {/* Taux de victoire par cote */}
      {oddsData?.odds_stats && oddsData.odds_stats.length > 0 && (
        <ScrollReveal>
          <GlassCard className="overflow-hidden">
            <GlassCardHeader
              icon={TrendingUp}
              title="Taux de victoire par cote"
              subtitle="Rentabilité des favoris vs outsiders"
              iconColor="#EC4899"
              iconBg="rgba(236, 72, 153, 0.15)"
            />

            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={oddsData.odds_stats}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                <XAxis
                  dataKey="range"
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <YAxis
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="taux" fill="#EC4899" name="Taux de victoire (%)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </GlassCard>
        </ScrollReveal>
      )}

      {/* Taux de victoire par distance */}
      {distanceData?.distance_stats && distanceData.distance_stats.length > 0 && (
        <ScrollReveal>
          <GlassCard className="overflow-hidden">
            <GlassCardHeader
              icon={MapPin}
              title="Taux de victoire par distance"
              subtitle="Probabilité mathématique (liée au nombre de partants)"
              iconColor="#8B5CF6"
              iconBg="rgba(139, 92, 246, 0.15)"
            />

            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={distanceData.distance_stats}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                <XAxis
                  dataKey="range"
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <YAxis
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="taux" fill="#8B5CF6" name="Taux de victoire (%)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </GlassCard>
        </ScrollReveal>
      )}

      {/* Taux de victoire par âge */}
      {ageData?.age_stats && ageData.age_stats.length > 0 && (
        <ScrollReveal>
          <GlassCard className="overflow-hidden">
            <GlassCardHeader
              icon={Calendar}
              title="Taux de victoire par âge"
              subtitle="L'âge d'or des chevaux"
              iconColor="#10B981"
              iconBg="rgba(16, 185, 129, 0.15)"
            />

            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={ageData.age_stats}>
                <CartesianGrid strokeDasharray="3 3" stroke={colors.border} opacity={0.3} />
                <XAxis
                  dataKey="age"
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <YAxis
                  stroke={colors.muted}
                  tick={{ fill: colors.text, fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="taux" fill="#10B981" name="Taux de victoire (%)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </GlassCard>
        </ScrollReveal>
      )}
    </div>
  );
};

export default Analytics;
