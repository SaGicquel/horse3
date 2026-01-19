import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bot, BarChart3, Target, TrendingUp, Sparkles, Play, Loader2,
  AlertTriangle, CheckCircle, Calendar, Percent, DollarSign,
  ArrowUpRight, ArrowDownRight, Activity, Zap, RefreshCw,
  ChevronDown, ChevronUp, Info
} from 'lucide-react';
import Chat from '../components/Chat';
import { GlassCard, GlassCardHeader } from '../components/GlassCard';
import PageHeader from '../components/PageHeader';
import { StaggerContainer, StaggerItem, ScrollReveal } from '../components/PageTransition';
import { Skeleton } from '../components/Skeleton';
import { API_BASE } from '../config/api';

const FeatureCard = ({ icon: Icon, title, description, delay = 0 }) => (
  <motion.div
    className="flex flex-col gap-3 p-5 rounded-2xl relative overflow-hidden group h-full"
    style={{
      background: 'rgba(var(--color-card-rgb, 255, 255, 255), 0.55)',
      backdropFilter: 'blur(24px)',
      WebkitBackdropFilter: 'blur(24px)',
      border: '1px solid rgba(var(--color-border-rgb), 0.1)',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.6)'
    }}
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.5 }}
    whileHover={{
      scale: 1.02,
      y: -5,
      boxShadow: '0 20px 50px rgba(157, 54, 86, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.8)'
    }}
  >
    {/* Shine effect on hover */}
    <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -skew-x-12 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000" />
    </div>

    <motion.div
      className="flex items-center gap-3"
      whileHover={{ x: 5 }}
    >
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center"
        style={{ backgroundColor: 'rgba(var(--color-primary-rgb), 0.15)' }}
      >
        <Icon size={20} style={{ color: 'var(--color-primary)' }} />
      </div>
      <span className="text-lg font-semibold" style={{ color: 'var(--color-text)' }}>
        {title}
      </span>
    </motion.div>
    <p className="text-sm leading-relaxed" style={{ color: 'var(--color-muted)' }}>
      {description}
    </p>
  </motion.div>
);

// Carte métrique avec animation
const MetricCard = ({ label, value, suffix = '', icon: Icon, color = 'primary', trend = null }) => {
  const colorMap = {
    primary: 'var(--color-primary)',
    success: '#22C55E',
    warning: '#FBBF24',
    danger: '#EF4444',
    info: '#3B82F6'
  };

  const bgMap = {
    primary: 'rgba(var(--color-primary-rgb), 0.15)',
    success: 'rgba(34, 197, 94, 0.15)',
    warning: 'rgba(251, 191, 36, 0.15)',
    danger: 'rgba(239, 68, 68, 0.15)',
    info: 'rgba(59, 130, 246, 0.15)'
  };

  return (
    <motion.div
      className="p-4 rounded-xl relative overflow-hidden"
      style={{
        background: 'rgba(var(--color-card-rgb), 0.5)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(12px)'
      }}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      <div className="flex items-start justify-between mb-2">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: bgMap[color] }}
        >
          <Icon size={16} style={{ color: colorMap[color] }} />
        </div>
        {trend !== null && (
          <div className={`flex items-center gap-0.5 text-xs ${trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trend >= 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
            {Math.abs(trend).toFixed(1)}%
          </div>
        )}
      </div>
      <p className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">{label}</p>
      <p className="text-xl font-bold" style={{ color: colorMap[color] }}>
        {typeof value === 'number' ? value.toFixed(2) : value}{suffix}
      </p>
    </motion.div>
  );
};

// Ligne de tableau période
const PeriodRow = ({ period, isTop = true }) => (
  <motion.tr
    className="border-b border-neutral-200 dark:border-white/5 hover:bg-neutral-50 dark:hover:bg-white/5 transition-colors"
    initial={{ opacity: 0, x: isTop ? -10 : 10 }}
    animate={{ opacity: 1, x: 0 }}
  >
    <td className="px-3 py-2 text-sm text-neutral-600 dark:text-neutral-300">{period.period}</td>
    <td className={`px-3 py-2 text-sm font-medium ${period.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
      {period.roi >= 0 ? '+' : ''}{period.roi.toFixed(1)}%
    </td>
    <td className="px-3 py-2 text-sm text-neutral-500 dark:text-neutral-400">{period.bets}</td>
    <td className={`px-3 py-2 text-sm ${period.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
      {period.pnl >= 0 ? '+' : ''}{period.pnl.toFixed(0)}€
    </td>
  </motion.tr>
);

const Backtest = () => {
  // État du formulaire
  const [formData, setFormData] = useState({
    startDate: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    endDate: new Date().toISOString().split('T')[0],
    valueCutoff: 5,
    kellyFraction: 0.25,
    capPercent: 5,
    market: 'PMU',
    budget: 1000
  });

  // État backtest
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showChat, setShowChat] = useState(false);

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const validateForm = useCallback(() => {
    const newWarnings = [];

    if (formData.kellyFraction > 0.5) {
      newWarnings.push('Kelly > 50% est très agressif, risque de ruine élevé');
    }
    if (formData.capPercent > 10) {
      newWarnings.push('Cap > 10% expose à une variance importante');
    }
    if (formData.valueCutoff < 0) {
      newWarnings.push('Value cutoff négatif inclura des paris -EV');
    }

    const startDate = new Date(formData.startDate);
    const endDate = new Date(formData.endDate);
    if (startDate >= endDate) {
      newWarnings.push('La date de début doit être antérieure à la date de fin');
    }

    setWarnings(newWarnings);
    return newWarnings.filter(w => w.includes('doit être')).length === 0;
  }, [formData]);

  const runBacktest = async () => {
    if (!validateForm()) {
      setError('Veuillez corriger les erreurs du formulaire');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${API_BASE}/backtest/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_date: formData.startDate,
          end_date: formData.endDate,
          value_cutoff: formData.valueCutoff,
          kelly_fraction: formData.kellyFraction,
          cap_percent: formData.capPercent,
          market: formData.market,
          budget: formData.budget
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Erreur ${response.status}`);
      }

      const data = await response.json();
      setResults(data);

      // Vérifier contraintes violées
      if (data.constraints_violated && data.constraints_violated.length > 0) {
        setWarnings(prev => [...prev, ...data.constraints_violated]);
      }
    } catch (err) {
      console.error('Erreur backtest:', err);
      setError(err.message || 'Erreur lors du backtest');

      // Générer des résultats de démo en cas d'erreur API
      setResults(generateDemoResults());
    } finally {
      setLoading(false);
    }
  };

  // Résultats de démonstration si l'API n'est pas disponible
  const generateDemoResults = () => ({
    roi: 12.4,
    ev_per_bet: 3.2,
    turnover: 4850,
    max_drawdown: -18.5,
    sharpe: 1.42,
    total_bets: 156,
    wins: 42,
    win_rate: 26.9,
    profit: 601.4,
    avg_odds: 4.8,
    best_periods: [
      { period: 'Nov 2025 S2', roi: 28.4, bets: 18, pnl: 245 },
      { period: 'Oct 2025 S4', roi: 22.1, bets: 14, pnl: 189 },
      { period: 'Sep 2025 S3', roi: 18.7, bets: 21, pnl: 156 }
    ],
    worst_periods: [
      { period: 'Oct 2025 S1', roi: -24.2, bets: 12, pnl: -198 },
      { period: 'Sep 2025 S1', roi: -15.8, bets: 16, pnl: -142 },
      { period: 'Nov 2025 S1', roi: -11.3, bets: 15, pnl: -98 }
    ],
    is_demo: true
  });

  return (
    <div className="mx-auto max-w-7xl space-y-6 sm:space-y-10 px-3 sm:px-6 py-6 sm:py-12">
      {/* Header unifié */}
      <PageHeader
        emoji="📈"
        title="Backtest Paramétrable"
        subtitle="Testez vos stratégies sur les données historiques et analysez les performances."
      />

      {/* Formulaire Backtest */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <GlassCard>
          <GlassCardHeader
            icon={Target}
            title="Paramètres du Backtest"
            subtitle="Configurez votre stratégie"
          />

          <div className="space-y-6">
            {/* Paramètres principaux */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Date début */}
              <div>
                <label className="block text-xs text-neutral-600 dark:text-neutral-400 mb-1.5">
                  <Calendar size={12} className="inline mr-1" />
                  Date début
                </label>
                <input
                  type="date"
                  name="startDate"
                  value={formData.startDate}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 bg-neutral-100 dark:bg-white/5 border border-neutral-200 dark:border-white/10 rounded-lg text-sm text-[var(--color-text)] focus:outline-none focus:border-purple-500/50"
                />
              </div>

              {/* Date fin */}
              <div>
                <label className="block text-xs text-neutral-600 dark:text-neutral-400 mb-1.5">
                  <Calendar size={12} className="inline mr-1" />
                  Date fin
                </label>
                <input
                  type="date"
                  name="endDate"
                  value={formData.endDate}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 bg-neutral-100 dark:bg-white/5 border border-neutral-200 dark:border-white/10 rounded-lg text-sm text-[var(--color-text)] focus:outline-none focus:border-purple-500/50"
                />
              </div>

              {/* Value Cutoff */}
              <div>
                <label className="block text-xs text-neutral-600 dark:text-neutral-400 mb-1.5">
                  <Percent size={12} className="inline mr-1" />
                  Value cutoff (%)
                </label>
                <input
                  type="number"
                  name="valueCutoff"
                  value={formData.valueCutoff}
                  onChange={handleInputChange}
                  min={-10}
                  max={100}
                  step={1}
                  className="w-full px-3 py-2 bg-neutral-100 dark:bg-white/5 border border-neutral-200 dark:border-white/10 rounded-lg text-sm text-[var(--color-text)] focus:outline-none focus:border-purple-500/50"
                />
                <p className="text-[10px] text-neutral-500 dark:text-neutral-400 mt-1">Min value% pour parier</p>
              </div>

              {/* Marché */}
              <div>
                <label className="block text-xs text-neutral-600 dark:text-neutral-400 mb-1.5">
                  <DollarSign size={12} className="inline mr-1" />
                  Marché
                </label>
                <select
                  name="market"
                  value={formData.market}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 bg-neutral-100 dark:bg-white/5 border border-neutral-200 dark:border-white/10 rounded-lg text-sm text-[var(--color-text)] focus:outline-none focus:border-purple-500/50"
                >
                  <option value="PMU">PMU</option>
                  <option value="BOOK">Bookmaker</option>
                  <option value="BETFAIR">Betfair</option>
                </select>
              </div>
            </div>

            {/* Toggle paramètres avancés */}
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-xs text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors"
            >
              {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              Paramètres avancés
            </button>

            {/* Paramètres avancés */}
            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="grid grid-cols-1 sm:grid-cols-3 gap-4 overflow-hidden"
                >
                  {/* Kelly Fraction */}
                  <div>
                    <label className="block text-xs text-neutral-600 dark:text-neutral-400 mb-1.5">
                      Kelly fraction
                      <span className="ml-1 text-[10px] text-neutral-500 dark:text-neutral-400">({(formData.kellyFraction * 100).toFixed(0)}%)</span>
                    </label>
                    <input
                      type="range"
                      name="kellyFraction"
                      value={formData.kellyFraction}
                      onChange={handleInputChange}
                      min={0.05}
                      max={1}
                      step={0.05}
                      className="w-full accent-purple-500"
                    />
                    <div className="flex justify-between text-[10px] text-gray-500">
                      <span>5%</span>
                      <span>100%</span>
                    </div>
                  </div>

                  {/* Cap % */}
                  <div>
                    <label className="block text-xs text-neutral-600 dark:text-neutral-400 mb-1.5">
                      Cap stake max (%)
                      <span className="ml-1 text-[10px] text-neutral-500 dark:text-neutral-400">({formData.capPercent}%)</span>
                    </label>
                    <input
                      type="range"
                      name="capPercent"
                      value={formData.capPercent}
                      onChange={handleInputChange}
                      min={1}
                      max={20}
                      step={0.5}
                      className="w-full accent-purple-500"
                    />
                    <div className="flex justify-between text-[10px] text-gray-500">
                      <span>1%</span>
                      <span>20%</span>
                    </div>
                  </div>

                  {/* Budget initial */}
                  <div>
                    <label className="block text-xs text-neutral-600 dark:text-neutral-400 mb-1.5">
                      Budget initial (€)
                    </label>
                    <input
                      type="number"
                      name="budget"
                      value={formData.budget}
                      onChange={handleInputChange}
                      min={100}
                      max={100000}
                      step={100}
                      className="w-full px-3 py-2 bg-neutral-100 dark:bg-white/5 border border-neutral-200 dark:border-white/10 rounded-lg text-sm text-[var(--color-text)] focus:outline-none focus:border-purple-500/50"
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Alertes */}
            <AnimatePresence>
              {warnings.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="p-3 rounded-lg space-y-1"
                  style={{
                    background: 'rgba(251, 191, 36, 0.1)',
                    border: '1px solid rgba(251, 191, 36, 0.3)'
                  }}
                >
                  {warnings.map((warning, i) => (
                    <div key={i} className="flex items-start gap-2 text-xs text-yellow-400">
                      <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
                      {warning}
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Erreur */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="p-3 rounded-lg flex items-start gap-2"
                  style={{
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.3)'
                  }}
                >
                  <AlertTriangle size={14} className="flex-shrink-0 mt-0.5 text-red-400" />
                  <span className="text-xs text-red-400">{error}</span>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Bouton Run */}
            <div className="flex justify-center">
              <motion.button
                onClick={runBacktest}
                disabled={loading}
                className="flex items-center gap-2 px-8 py-3 rounded-xl font-medium text-white transition-all disabled:opacity-50"
                style={{
                  background: 'linear-gradient(135deg, var(--color-primary), #9D3656)',
                  boxShadow: '0 4px 20px rgba(157, 54, 86, 0.3)'
                }}
              >
                {loading ? (
                  <>
                    <Loader2 size={18} className="animate-spin" />
                    Calcul en cours...
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    Lancer le Backtest
                  </>
                )}
              </motion.button>
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {/* Résultats */}
      <AnimatePresence>
        {results && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className="space-y-6"
          >
            {/* Banner démo si applicable */}
            {results.is_demo && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="p-3 rounded-lg flex items-center gap-2"
                style={{
                  background: 'rgba(59, 130, 246, 0.1)',
                  border: '1px solid rgba(59, 130, 246, 0.3)'
                }}
              >
                <Info size={14} className="text-blue-400" />
                <span className="text-xs text-blue-400">
                  Mode démo - L'API backtest n'est pas disponible, données simulées affichées
                </span>
              </motion.div>
            )}

            {/* Métriques clés */}
            <GlassCard>
              <GlassCardHeader
                icon={Activity}
                title="Résultats du Backtest"
                subtitle={`${formData.startDate} → ${formData.endDate}`}
                iconColor="#22C55E"
                iconBg="rgba(34, 197, 94, 0.15)"
              />

              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                <MetricCard
                  label="ROI"
                  value={results.roi}
                  suffix="%"
                  icon={TrendingUp}
                  color={results.roi >= 0 ? 'success' : 'danger'}
                />
                <MetricCard
                  label="EV / Pari"
                  value={results.ev_per_bet}
                  suffix="%"
                  icon={Target}
                  color={results.ev_per_bet >= 0 ? 'success' : 'danger'}
                />
                <MetricCard
                  label="Turnover"
                  value={results.turnover}
                  suffix="€"
                  icon={DollarSign}
                  color="info"
                />
                <MetricCard
                  label="Max Drawdown"
                  value={results.max_drawdown}
                  suffix="%"
                  icon={ArrowDownRight}
                  color="danger"
                />
                <MetricCard
                  label="Sharpe Ratio"
                  value={results.sharpe}
                  suffix=""
                  icon={Zap}
                  color={results.sharpe >= 1 ? 'success' : results.sharpe >= 0.5 ? 'warning' : 'danger'}
                />
                <MetricCard
                  label="Win Rate"
                  value={results.win_rate}
                  suffix="%"
                  icon={CheckCircle}
                  color="primary"
                />
              </div>

              {/* Stats secondaires */}
              <div className="mt-4 pt-4 border-t border-neutral-200 dark:border-white/10 grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
                <div>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">Total Paris</p>
                  <p className="text-lg font-bold text-[var(--color-text)]">{results.total_bets}</p>
                </div>
                <div>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">Gagnés</p>
                  <p className="text-lg font-bold text-green-400">{results.wins}</p>
                </div>
                <div>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">Profit Net</p>
                  <p className={`text-lg font-bold ${results.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {results.profit >= 0 ? '+' : ''}{results.profit?.toFixed(0)}€
                  </p>
                </div>
                <div>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">Cote Moyenne</p>
                  <p className="text-lg font-bold text-yellow-400">{results.avg_odds?.toFixed(2)}</p>
                </div>
              </div>
            </GlassCard>

            {/* Tableaux Top / Nadir */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Meilleures périodes */}
              <GlassCard>
                <GlassCardHeader
                  icon={ArrowUpRight}
                  title="Top Périodes"
                  subtitle="Meilleures performances"
                  iconColor="#22C55E"
                  iconBg="rgba(34, 197, 94, 0.15)"
                />
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-neutral-200 dark:border-white/10">
                        <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400">Période</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400">ROI</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400">Paris</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400">P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(results.best_periods || []).map((period, i) => (
                        <PeriodRow key={i} period={period} isTop={true} />
                      ))}
                    </tbody>
                  </table>
                </div>
              </GlassCard>

              {/* Pires périodes */}
              <GlassCard>
                <GlassCardHeader
                  icon={ArrowDownRight}
                  title="Nadir Périodes"
                  subtitle="Périodes difficiles"
                  iconColor="#EF4444"
                  iconBg="rgba(239, 68, 68, 0.15)"
                />
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-neutral-200 dark:border-white/10">
                        <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400">Période</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400">ROI</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400">Paris</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400">P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(results.worst_periods || []).map((period, i) => (
                        <PeriodRow key={i} period={period} isTop={false} />
                      ))}
                    </tbody>
                  </table>
                </div>
              </GlassCard>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle Assistant IA */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <button
          onClick={() => setShowChat(!showChat)}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl transition-all"
          style={{
            background: 'rgba(var(--color-card-rgb), 0.5)',
            border: '1px solid rgba(255, 255, 255, 0.1)'
          }}
        >
          <Bot size={18} style={{ color: 'var(--color-primary)' }} />
          <span className="text-sm" style={{ color: 'var(--color-text)' }}>
            {showChat ? 'Masquer' : 'Afficher'} l'Assistant IA
          </span>
          {showChat ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </motion.div>

      {/* Chat Component (optionnel) */}
      <AnimatePresence>
        {showChat && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <GlassCard className="overflow-hidden" hover={false}>
              <Chat />
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info Section */}
      <ScrollReveal>
        <GlassCard className="px-6 sm:px-12 py-8 sm:py-12">
          <div className="relative mx-auto flex flex-col gap-6">
            <motion.div className="text-center">
              <motion.div
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full mb-4"
                style={{
                  backgroundColor: 'rgba(var(--color-primary-rgb), 0.1)',
                  border: '1px solid rgba(var(--color-primary-rgb), 0.2)'
                }}
                animate={{ scale: [1, 1.02, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Sparkles size={16} style={{ color: 'var(--color-primary)' }} />
                <span className="text-sm font-medium" style={{ color: 'var(--color-primary)' }}>
                  Guide
                </span>
              </motion.div>
              <h2
                className="text-xl sm:text-2xl font-semibold"
                style={{ color: 'var(--color-text)' }}
              >
                Comprendre les métriques
              </h2>
            </motion.div>

            <StaggerContainer className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <StaggerItem>
                <FeatureCard
                  icon={TrendingUp}
                  title="ROI (Return on Investment)"
                  description="Profit net divisé par le turnover total. Un ROI de 10% signifie 10€ de profit pour 100€ misés."
                  delay={0}
                />
              </StaggerItem>
              <StaggerItem>
                <FeatureCard
                  icon={Target}
                  title="Sharpe Ratio"
                  description="Mesure le rendement ajusté au risque. > 1 = excellent, 0.5-1 = bon, < 0.5 = à améliorer."
                  delay={0.1}
                />
              </StaggerItem>
              <StaggerItem>
                <FeatureCard
                  icon={ArrowDownRight}
                  title="Max Drawdown"
                  description="Perte maximale depuis un pic. Indicateur clé du risque de ruine de la stratégie."
                  delay={0.2}
                />
              </StaggerItem>
              <StaggerItem>
                <FeatureCard
                  icon={Zap}
                  title="Kelly Criterion"
                  description="Formule optimale de sizing. Une fraction de 25% = 1/4 du Kelly pour réduire la variance."
                  delay={0.3}
                />
              </StaggerItem>
            </StaggerContainer>
          </div>
        </GlassCard>
      </ScrollReveal>
    </div>
  );
};

export default Backtest;
