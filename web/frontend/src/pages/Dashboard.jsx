import { useState, useEffect, memo, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, DollarSign, BarChart2, Zap, Activity, BarChartHorizontal, RefreshCw, Target, AlertTriangle, Percent } from 'lucide-react';
import { dashboardAPI } from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import AnimatedStatCard from '../components/AnimatedStatCard';
import { GlassCard, GlassCardHeader } from '../components/GlassCard';
import { SkeletonDashboard, Skeleton, SkeletonStatCard } from '../components/Skeleton';
import { StaggerContainer, StaggerItem } from '../components/PageTransition';
import BackendStatus from '../components/BackendStatus';
import { API_BASE } from '../config/api';

// Composant Performance Item avec animation et glassmorphism (Mémorisé)
const PerformanceItem = memo(({ name, percentage, result, change, index }) => {
  const resultClasses = useMemo(() => {
    if (result === 'Gagné') return 'bg-success/15 text-success border-success/20';
    if (result === 'Placé') return 'bg-warning/15 text-warning border-warning/20';
    if (result === 'Perdu') return 'bg-error/15 text-error border-error/20';
    return 'bg-info/15 text-info border-info/20';
  }, [result]);

  return (
    <motion.div
      className="p-4 rounded-xl flex items-center gap-4 glass-panel border border-neutral-200/50 dark:border-white/5 hover:border-neutral-300 dark:hover:border-white/20 transition-colors"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      whileHover={{
        scale: 1.02,
        backgroundColor: 'rgba(var(--color-card-rgb), 0.6)'
      }}
    >
      <div className="flex-1">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">{name}</span>
        </div>
        <div className="w-full rounded-full h-2 overflow-hidden bg-neutral-200/30 dark:bg-neutral-700/30 border border-neutral-300/30 dark:border-white/10">
          <motion.div
            className="h-2 rounded-full bg-gradient-to-r from-[#ec4899] to-[#f472b6] shadow-[0_0_10px_rgba(236,72,153,0.4)]"
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 1, delay: index * 0.1, ease: "easeOut" }}
          />
        </div>
        <span className="text-xs font-bold mt-1 inline-block text-neutral-700 dark:text-neutral-400">
          {percentage}%
        </span>
      </div>
      <div className="text-center w-24">
        <motion.p
          className={`font-bold text-xs px-3 py-1.5 rounded-lg border backdrop-blur-sm ${resultClasses}`}
          whileHover={{ scale: 1.05 }}
        >
          {result}
        </motion.p>
        <p
          className={`text-xs mt-1 font-semibold ${change.startsWith('+') ? 'text-success' : 'text-error'}`}
        >
          {change}
        </p>
      </div>
    </motion.div>
  );
});

// Composant Indicateur avec animation et glassmorphism (Mémorisé)
const PredictiveVariable = memo(({ name, percentage, index }) => (
  <motion.div
    className="py-3"
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: index * 0.05 }}
  >
    <div className="flex justify-between items-center mb-2">
      <span className="text-sm text-neutral-900 dark:text-neutral-100">{name}</span>
      <motion.span
        className="text-sm font-bold px-2 py-0.5 rounded-md text-[#db2777] dark:text-[#f472b6] bg-[#ec48991a] border border-[#ec489933]"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: index * 0.05 + 0.2 }}
      >
        {percentage}%
      </motion.span>
    </div>
    <div className="w-full rounded-full h-2 overflow-hidden bg-neutral-200/30 dark:bg-neutral-700/30 border border-neutral-300/30 dark:border-white/10">
      <motion.div
        className="h-2 rounded-full bg-gradient-to-r from-[#db2777] to-[#f472b6]"
        initial={{ width: 0 }}
        animate={{ width: `${percentage}%` }}
        transition={{ duration: 0.8, delay: index * 0.05, ease: "easeOut" }}
      />
    </div>
  </motion.div>
));

// Custom Tooltip pour les graphiques avec glassmorphism (Mémorisé)
const CustomChartTooltip = memo(({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="px-4 py-3 rounded-xl shadow-lg glass-panel border border-neutral-200/50 dark:border-white/20">
        <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="text-sm mt-1" style={{ color: entry.color }}>
            {entry.name}: <strong>{entry.value}€</strong>
          </p>
        ))}
      </div>
    );
  }
  return null;
});


const Dashboard = () => {
  const [token, setToken] = useState(() => localStorage.getItem('hrp_token'));
  const [dashboardData, setDashboardData] = useState(null);
  const [monitoringData, setMonitoringData] = useState(null);
  const [calibrationData, setCalibrationData] = useState(null);
  const [calibrationLoading, setCalibrationLoading] = useState(true);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Charger les données de calibration
  const loadCalibrationData = useCallback(async () => {
    try {
      setCalibrationLoading(true);
      const response = await fetch(`${API_BASE}/calibration/health`);
      if (response.ok) {
        const data = await response.json();
        setCalibrationData(data);
      } else {
        setCalibrationData(null);
      }
    } catch (error) {
      console.error('Erreur chargement calibration:', error);
      setCalibrationData(null);
    } finally {
      setCalibrationLoading(false);
    }
  }, []);

  // Charger les données avec useCallback pour éviter les rerenderings
  const loadDashboardData = useCallback(async (forceRefresh = false) => {
    try {
      setLoading(true);
      setErrorMessage(null);

      // Utiliser forceRefresh pour invalider le cache si nécessaire
      const [data, monitoring] = await Promise.all([
        dashboardAPI.getDashboardData(token, undefined, forceRefresh),
        dashboardAPI.getMonitoringData(token, undefined, forceRefresh)
      ]);
      setDashboardData(data);
      setMonitoringData(monitoring);
    } catch (error) {
      console.error('Erreur chargement dashboard:', error);
      setErrorMessage(error?.message || 'Impossible de charger le dashboard.');
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    loadDashboardData(false);
    loadCalibrationData();
  }, [loadDashboardData, loadCalibrationData]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    // Invalider le cache et forcer le rechargement
    dashboardAPI.refresh();
    await Promise.all([loadDashboardData(true), loadCalibrationData()]);
    setRefreshing(false);
  }, [loadDashboardData, loadCalibrationData]);

  // Synchroniser le token stocké (utilisé pour les données personnelles)
  useEffect(() => {
    const syncToken = () => setToken(localStorage.getItem('hrp_token'));
    syncToken();
    window.addEventListener('focus', syncToken);
    window.addEventListener('storage', syncToken);
    return () => {
      window.removeEventListener('focus', syncToken);
      window.removeEventListener('storage', syncToken);
    };
  }, []);

  // Calcule si la calibration a des alertes
  const calibrationAlert = useMemo(() => {
    if (!calibrationData) return null;

    const ece = calibrationData.ece_7d ?? calibrationData.ece;
    const lastCalibration = calibrationData.last_calibration;

    let alerts = [];

    // Alerte si ECE > 0.02
    if (ece !== undefined && ece > 0.02) {
      alerts.push(`ECE élevé: ${(ece * 100).toFixed(2)}% (seuil: 2%)`);
    }

    // Alerte si dernière calibration > 7 jours
    if (lastCalibration) {
      const lastDate = new Date(lastCalibration);
      const daysSince = Math.floor((Date.now() - lastDate.getTime()) / (1000 * 60 * 60 * 24));
      if (daysSince > 7) {
        alerts.push(`Calibration obsolète: ${daysSince} jours`);
      }
    }

    return alerts.length > 0 ? alerts : null;
  }, [calibrationData]);

  // Formate une date en format humain
  const formatDateHuman = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMinutes = Math.floor(diffMs / (1000 * 60));

    if (diffMinutes < 60) return `Il y a ${diffMinutes} min`;
    if (diffHours < 24) return `Il y a ${diffHours}h`;
    if (diffDays === 1) return 'Hier';
    if (diffDays < 7) return `Il y a ${diffDays} jours`;
    return date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' });
  };

  if (loading) {
    return <SkeletonDashboard />;
  }

  if (!dashboardData) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p className="text-neutral-700 dark:text-neutral-400">{errorMessage || 'Erreur de chargement des données'}</p>
      </div>
    );
  }

  const defaultStats = {
    taux_reussite: 0,
    evolution_reussite: 0,
    roi_moyen: 0,
    evolution_roi: 0,
    courses_analysees: 0,
    evolution_courses: 0,
    modeles_actifs: 0,
  };

  const stats = { ...defaultStats, ...(dashboardData?.stats || {}) };
  const performances_recentes = dashboardData?.performances_recentes || [];
  const variables_predictives = dashboardData?.variables_predictives || [];

  const formatNumber = (num) => {
    return new Intl.NumberFormat('fr-FR').format(num);
  };

  return (
    <div className="max-w-7xl mx-auto px-3 sm:px-6 py-6 sm:py-12">
      {/* Backend Status en en-tête */}
      <BackendStatus showWhenHealthy={false} position="top" />

      {/* Header avec animation */}
      <motion.header
        className="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div>
          <motion.h1
            className="text-3xl sm:text-4xl font-bold text-neutral-900 dark:text-neutral-100"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            Horse3 Champion
          </motion.h1>
          <motion.p
            className="text-sm sm:text-base mt-1 text-neutral-700 dark:text-neutral-400"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            Vue d'ensemble des performances et analyses
          </motion.p>
        </div>
        <motion.button
          onClick={handleRefresh}
          className="glass-button text-[#db2777] dark:text-[#f472b6] hover:bg-white/10"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <motion.div
            animate={refreshing ? { rotate: 360 } : {}}
            transition={{ duration: 1, repeat: refreshing ? Infinity : 0, ease: "linear" }}
          >
            <RefreshCw size={16} />
          </motion.div>
          Actualiser
        </motion.button>
      </motion.header>

      {/* Stats Cards avec animations stagger */}
      <StaggerContainer className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-8">
        <StaggerItem className="h-full">
          <AnimatedStatCard
            icon={TrendingUp}
            title="Taux de Réussite"
            value={stats.taux_reussite}
            suffix="%"
            change={`${stats.evolution_reussite >= 0 ? '+' : ''}${stats.evolution_reussite}%`}
            trend={stats.evolution_reussite >= 0 ? "up" : "down"}
            delay={0}
          />
        </StaggerItem>
        <StaggerItem className="h-full">
          <AnimatedStatCard
            icon={DollarSign}
            title="ROI Moyen"
            value={stats.roi_moyen}
            suffix="%"
            change={`${stats.evolution_roi >= 0 ? '+' : ''}${stats.evolution_roi}%`}
            trend={stats.evolution_roi >= 0 ? "up" : "down"}
            delay={0.1}
          />
        </StaggerItem>
        <StaggerItem className="h-full">
          <AnimatedStatCard
            icon={BarChart2}
            title="Courses Analysées"
            value={stats.courses_analysees}
            change={`+${stats.evolution_courses}`}
            trend="up"
            delay={0.2}
          />
        </StaggerItem>
        <StaggerItem className="h-full">
          <AnimatedStatCard
            icon={Zap}
            title="Modèles Actifs"
            value={stats.modeles_actifs}
            changeLabel="Professionnels"
            trend="up"
            delay={0.3}
          />
        </StaggerItem>
      </StaggerContainer>

      {/* Calibration & KPIs Section */}
      <motion.div
        className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35, duration: 0.5 }}
      >
        {/* Carte Calibration */}
        <GlassCard hover={false}>
          <GlassCardHeader
            icon={Target}
            title="Santé Calibration"
            subtitle="Qualité des probabilités"
          />

          {calibrationLoading ? (
            <div className="space-y-4">
              <Skeleton width="100%" height={60} borderRadius={12} />
              <div className="grid grid-cols-3 gap-3">
                <Skeleton width="100%" height={70} borderRadius={8} />
                <Skeleton width="100%" height={70} borderRadius={8} />
                <Skeleton width="100%" height={70} borderRadius={8} />
              </div>
            </div>
          ) : calibrationData ? (
            <div className="space-y-4">
              {/* Alerte si seuils dépassés */}
              {calibrationAlert && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="p-3 rounded-lg flex items-start gap-3 bg-warning/15 border border-warning/30"
                >
                  <AlertTriangle size={20} className="text-warning flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-warning">Attention requise</p>
                    <ul className="text-xs text-warning/80 mt-1 space-y-0.5">
                      {calibrationAlert.map((alert, i) => (
                        <li key={i}>• {alert}</li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              )}

              {/* Métriques */}
              <div className="grid grid-cols-3 gap-3">
                <div className="p-3 rounded-lg text-center glass-panel border border-neutral-200/50 dark:border-white/10">
                  <p className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">ECE 7j</p>
                  <p className={`text-xl font-bold ${(calibrationData.ece_7d ?? calibrationData.ece) > 0.02
                    ? 'text-warning'
                    : 'text-success'
                    }`}>
                    {((calibrationData.ece_7d ?? calibrationData.ece ?? 0) * 100).toFixed(2)}%
                  </p>
                </div>

                <div className="p-3 rounded-lg text-center glass-panel border border-neutral-200/50 dark:border-white/10">
                  <p className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">Brier 7j</p>
                  <p className="text-xl font-bold text-info">
                    {((calibrationData.brier_7d ?? calibrationData.brier ?? 0) * 100).toFixed(2)}%
                  </p>
                </div>

                <div className="p-3 rounded-lg text-center glass-panel border border-neutral-200/50 dark:border-white/10">
                  <p className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">Dernière sync</p>
                  <p className="text-sm font-bold text-secondary-600 dark:text-secondary-400">
                    {formatDateHuman(calibrationData.last_calibration)}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-6">
              <Target size={32} className="mx-auto text-neutral-600 dark:text-neutral-500 mb-2" />
              <p className="text-sm text-neutral-600 dark:text-neutral-400">Données de calibration non disponibles</p>
            </div>
          )}
        </GlassCard>

        {/* Carte KPIs */}
        <GlassCard hover={false}>
          <GlassCardHeader
            icon={Percent}
            title="KPIs Betting"
            subtitle="Indicateurs clés"
          />

          {calibrationLoading ? (
            <div className="grid grid-cols-3 gap-3">
              <Skeleton width="100%" height={70} borderRadius={8} />
              <Skeleton width="100%" height={70} borderRadius={8} />
              <Skeleton width="100%" height={70} borderRadius={8} />
            </div>
          ) : (calibrationData?.kpis || monitoringData) ? (
            <div className="grid grid-cols-3 gap-3">
              <div className="p-3 rounded-lg text-center glass-panel border border-neutral-200/50 dark:border-white/10">
                <p className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">ROI</p>
                <p className={`text-xl font-bold ${(calibrationData?.kpis?.roi ?? monitoringData?.roi ?? 0) >= 0
                  ? 'text-success'
                  : 'text-error'
                  }`}>
                  {(calibrationData?.kpis?.roi ?? monitoringData?.roi ?? 0).toFixed(1)}%
                </p>
              </div>

              <div className="p-3 rounded-lg text-center glass-panel border border-neutral-200/50 dark:border-white/10">
                <p className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">Hit Rate</p>
                <p className="text-xl font-bold text-info">
                  {(calibrationData?.kpis?.hit_rate ?? monitoringData?.win_rate ?? 0).toFixed(1)}%
                </p>
              </div>

              <div className="p-3 rounded-lg text-center glass-panel border border-neutral-200/50 dark:border-white/10">
                <p className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">Turnover</p>
                <p className="text-xl font-bold text-secondary-600 dark:text-secondary-400">
                  {formatNumber(calibrationData?.kpis?.turnover ?? monitoringData?.total_bets ?? 0)}
                </p>
              </div>
            </div>
          ) : (
            <div className="text-center py-6">
              <Percent size={32} className="mx-auto text-neutral-600 dark:text-neutral-500 mb-2" />
              <p className="text-sm text-neutral-600 dark:text-neutral-400">KPIs non disponibles</p>
            </div>
          )}
        </GlassCard>
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-8">
        {/* Top Performances */}
        <motion.div
          className="lg:col-span-2"
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
        >
          <GlassCard hover={false}>
            <GlassCardHeader
              icon={Activity}
              title="Top Performances"
              subtitle="Dernières analyses"
            />
            <div className="space-y-3">
              {performances_recentes && performances_recentes.length > 0 ? (
                performances_recentes.map((perf, index) => (
                  <PerformanceItem
                    key={index}
                    name={perf.nom_course}
                    percentage={Math.round(perf.probabilite)}
                    result={perf.resultat}
                    change={perf.evolution}
                    index={index}
                  />
                ))
              ) : (
                <p className="text-sm py-8 text-center text-neutral-600 dark:text-neutral-400">
                  Aucune performance disponible
                </p>
              )}
            </div>
          </GlassCard>
        </motion.div>

        {/* Indicateurs Clés */}
        <motion.div
          className="lg:col-span-3"
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
        >
          <GlassCard hover={false}>
            <GlassCardHeader
              icon={BarChartHorizontal}
              title="Indicateurs Clés"
              subtitle="Variables prédictives"
            />
            <div className="space-y-2">
              {variables_predictives && variables_predictives.length > 0 ? (
                variables_predictives.map((var_pred, index) => (
                  <PredictiveVariable
                    key={index}
                    name={var_pred.nom}
                    percentage={var_pred.importance}
                    index={index}
                  />
                ))
              ) : (
                <p className="text-sm py-8 text-center text-neutral-600 dark:text-neutral-400">
                  Aucun indicateur disponible
                </p>
              )}
            </div>
          </GlassCard>
        </motion.div>
      </div>

      {/* Section Monitoring Robot */}
      {monitoringData && (
        <motion.div
          className="space-y-6"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.5 }}
        >
          <div className="flex items-center gap-3">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            >
              <Zap size={24} className="text-warning" />
            </motion.div>
            <h2 className="text-xl font-bold text-neutral-900 dark:text-neutral-100">
              Monitoring Robot (Paper Trading)
            </h2>
          </div>

          <StaggerContainer className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
            <StaggerItem>
              <AnimatedStatCard
                icon={Activity}
                title="Paris Total"
                value={monitoringData.total_bets}
                changeLabel={`${monitoringData.pending_bets} en cours`}
                trend="up"
                delay={0}
              />
            </StaggerItem>
            <StaggerItem>
              <AnimatedStatCard
                icon={TrendingUp}
                title="Win Rate"
                value={monitoringData.win_rate}
                suffix="%"
                changeLabel="Sur paris terminés"
                trend={monitoringData.win_rate > 30 ? "up" : "down"}
                delay={0.1}
              />
            </StaggerItem>
            <StaggerItem>
              <AnimatedStatCard
                icon={DollarSign}
                title="P&L Net"
                value={monitoringData.pnl_net}
                suffix=" €"
                change={`${monitoringData.roi}% ROI`}
                trend={monitoringData.pnl_net >= 0 ? "up" : "down"}
                delay={0.2}
              />
            </StaggerItem>
            <StaggerItem>
              <AnimatedStatCard
                icon={BarChart2}
                title="Paris Terminés"
                value={monitoringData.finished_bets}
                changeLabel="Résultats connus"
                trend="up"
                delay={0.3}
              />
            </StaggerItem>
          </StaggerContainer>

          {/* Graphique P&L avec glassmorphism */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
          >
            <GlassCard hover={false} className="overflow-hidden">
              <GlassCardHeader
                icon={TrendingUp}
                title="Évolution du P&L Cumulé"
                subtitle={monitoringData?.data_scope === 'user' ? 'PNL réel basé sur vos paris' : 'Performance historique (paper trading)'}
              />
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={monitoringData.pnl_history}>
                    <defs>
                      <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--color-primary)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="var(--color-primary)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" opacity={0.3} />
                    <XAxis
                      dataKey="date"
                      stroke="var(--color-text-muted)"
                      tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }}
                      axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    />
                    <YAxis
                      stroke="var(--color-text-muted)"
                      tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }}
                      axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                    />
                    <Tooltip content={<CustomChartTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="pnl_cumul"
                      stroke="var(--color-primary)"
                      strokeWidth={3}
                      fill="url(#pnlGradient)"
                      dot={{ r: 4, fill: 'var(--color-primary)', strokeWidth: 2, stroke: 'var(--color-card)' }}
                      activeDot={{ r: 8, fill: 'var(--color-primary)', strokeWidth: 2, stroke: 'white' }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
};

export default Dashboard;
