import React, { useState, useEffect } from 'react';
import { GlassCard } from '../components/GlassCard';
import { TrendingUp, BarChart3, Calendar, Target, AlertTriangle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

const HistoricalStats = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchHistoricalStats();
  }, []);

  const fetchHistoricalStats = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8001/historical-stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Erreur récupération stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatROI = (roi) => {
    const sign = roi >= 0 ? '+' : '';
    const color = roi >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
    return { text: `${sign}${roi.toFixed(1)}%`, color };
  };

  const getDrawdownColor = (dd) => {
    if (dd <= 5) return 'text-green-600 dark:text-green-400';
    if (dd <= 15) return 'text-orange-600 dark:text-orange-400';
    return 'text-red-600 dark:text-red-400';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="text-center py-8">
        <p className="text-neutral-500 dark:text-neutral-400">Impossible de charger les statistiques</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">📊 Historique & Stats</h1>
          <p className="text-neutral-600 dark:text-neutral-400">Performance et analyse du modèle champion</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-neutral-200 dark:border-neutral-700">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Vue d\'ensemble', icon: <BarChart3 className="w-4 h-4" /> },
            { id: 'monthly', label: 'ROI Mensuel', icon: <Calendar className="w-4 h-4" /> },
            { id: 'evolution', label: 'Évolution', icon: <TrendingUp className="w-4 h-4" /> }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 py-2 px-1 border-b-2 font-medium text-sm ${activeTab === tab.id
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300 hover:border-neutral-300 dark:hover:border-neutral-600'
                }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Métriques principales */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">

            <GlassCard className="border-blue-200 dark:border-blue-800 p-4">
              <div className="pb-3">
                <h3 className="text-sm font-medium text-blue-600 dark:text-blue-400 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  ROI Global
                </h3>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  +22.71%
                </div>
                <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">Backtest validé</p>
              </div>
            </GlassCard>

            <GlassCard className="border-orange-200 dark:border-orange-800 p-4">
              <div className="pb-3">
                <h3 className="text-sm font-medium text-orange-600 dark:text-orange-400 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  Drawdown Actuel
                </h3>
              </div>
              <div>
                <div className={`text-2xl font-bold ${getDrawdownColor(stats.drawdown_actuel)}`}>
                  {stats.drawdown_actuel.toFixed(1)}%
                </div>
                <p className="text-xs text-orange-600 dark:text-orange-400 mt-1">
                  Max: {stats.drawdown_max.toFixed(1)}%
                </p>
              </div>
            </GlassCard>

            <GlassCard className="border-green-200 dark:border-green-800 p-4">
              <div className="pb-3">
                <h3 className="text-sm font-medium text-green-600 dark:text-green-400 flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Série Gagnante
                </h3>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-900 dark:text-green-100">
                  {stats.serie_gagnante}
                </div>
                <p className="text-xs text-green-600 dark:text-green-400 mt-1">paris consécutifs</p>
              </div>
            </GlassCard>

            <GlassCard className="border-neutral-200 dark:border-neutral-700 p-4">
              <div className="pb-3">
                <h3 className="text-sm font-medium text-neutral-600 dark:text-neutral-400 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  Total Paris
                </h3>
              </div>
              <div>
                <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
                  {stats.nb_paris_total.toLocaleString('fr-FR')}
                </div>
                <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-1">depuis le lancement</p>
              </div>
            </GlassCard>
          </div>

          {/* Métriques de performance */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Performance Summary */}
            <GlassCard className="p-6">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">📈 Résumé Performance</h3>
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-center py-2 border-b border-neutral-200 dark:border-neutral-700">
                  <span className="text-neutral-600 dark:text-neutral-400">Sharpe Ratio</span>
                  <span className="font-bold text-blue-600 dark:text-blue-400">3.599</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-neutral-200 dark:border-neutral-700">
                  <span className="text-neutral-600 dark:text-neutral-400">Série perdante max</span>
                  <span className="font-bold text-red-600 dark:text-red-400">{stats.serie_perdante}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-neutral-200 dark:border-neutral-700">
                  <span className="text-neutral-600 dark:text-neutral-400">Win Rate</span>
                  <span className="font-bold text-green-600 dark:text-green-400">10.09%</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-neutral-200 dark:border-neutral-700">
                  <span className="text-neutral-600 dark:text-neutral-400">Calibration ECE</span>
                  <span className="font-bold text-blue-600 dark:text-blue-400">0.0112</span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-neutral-600 dark:text-neutral-400">Stratégie</span>
                  <span className="bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 px-2.5 py-0.5 rounded-full text-xs font-medium">Blend + Kelly</span>
                </div>
              </div>
            </GlassCard>

            {/* Risk Metrics */}
            <GlassCard className="p-6">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">⚠️ Métriques de Risque</h3>
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-center py-2 border-b border-neutral-200 dark:border-neutral-700">
                  <span className="text-neutral-600 dark:text-neutral-400">Drawdown Maximum</span>
                  <span className={`font-bold ${getDrawdownColor(stats.drawdown_max)}`}>
                    {stats.drawdown_max.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-neutral-200 dark:border-neutral-700">
                  <span className="text-neutral-600 dark:text-neutral-400">Drawdown Actuel</span>
                  <span className={`font-bold ${getDrawdownColor(stats.drawdown_actuel)}`}>
                    {stats.drawdown_actuel.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-neutral-200 dark:border-neutral-700">
                  <span className="text-neutral-600 dark:text-neutral-400">Temps en Drawdown</span>
                  <span className="font-bold text-orange-600 dark:text-orange-400">91.2%</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-neutral-200 dark:border-neutral-700">
                  <span className="text-neutral-600 dark:text-neutral-400">Kelly Fraction</span>
                  <span className="font-bold text-blue-600 dark:text-blue-400">25%</span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-neutral-600 dark:text-neutral-400">Mise Max / Pari</span>
                  <span className="font-bold text-neutral-900 dark:text-neutral-100">5% BK</span>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>
      )}

      {/* Monthly ROI Tab */}
      {activeTab === 'monthly' && (
        <div className="space-y-6">
          <GlassCard className="p-6">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">📅 ROI par Mois</h3>
            </div>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={Object.entries(stats.roi_mensuel).map(([month, roi]) => ({ month, roi }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="month" stroke="var(--color-text)" fontSize={12} />
                  <YAxis stroke="var(--color-text)" fontSize={12} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'var(--color-card)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
                    formatter={(value) => [`${value.toFixed(1)}%`, 'ROI']}
                    labelFormatter={(label) => `Mois: ${label}`}
                  />
                  <Bar dataKey="roi" fill="#3B82F6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(stats.roi_mensuel).map(([month, roi]) => {
              const roiFormat = formatROI(roi);
              return (
                <GlassCard key={month} className="border-neutral-200 dark:border-neutral-700 p-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-neutral-600 dark:text-neutral-400">{month}</span>
                    <span className={`text-lg font-bold ${roiFormat.color}`}>
                      {roiFormat.text}
                    </span>
                  </div>
                </GlassCard>
              );
            })}
          </div>
        </div>
      )}

      {/* Evolution Tab */}
      {activeTab === 'evolution' && (
        <GlassCard className="p-6">
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">📈 Évolution de la Bankroll (30 derniers jours)</h3>
          </div>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stats.bankroll_evolution}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(value) => new Date(value).toLocaleDateString('fr-FR', { month: 'short', day: 'numeric' })}
                  stroke="var(--color-text)"
                  fontSize={12}
                />
                <YAxis stroke="var(--color-text)" fontSize={12} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--color-card)', borderColor: 'var(--color-border)', color: 'var(--color-text)' }}
                  formatter={(value, name) => [
                    name === 'bankroll' ? `${value.toLocaleString('fr-FR')} €` : `${value.toFixed(1)}%`,
                    name === 'bankroll' ? 'Bankroll' : 'ROI'
                  ]}
                  labelFormatter={(label) => new Date(label).toLocaleDateString('fr-FR')}
                />
                <Line
                  type="monotone"
                  dataKey="bankroll"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>
      )}

      {/* Footer info */}
      <GlassCard className="border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/10 p-4">
        <div className="flex items-start gap-3">
          <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
            <span className="text-white text-xs font-bold">ℹ️</span>
          </div>
          <div className="text-sm text-blue-800 dark:text-blue-200">
            <p className="font-semibold mb-1">📊 Note sur les Statistiques</p>
            <p>
              Ces statistiques sont basées sur le backtest du modèle champion XGBoost avec la stratégie "Blend + Kelly".
              Le système utilise une calibration Platt optimisée et un blend modèle/marché pour maximiser la performance
              tout en contrôlant le risque via le critère de Kelly fractionné.
            </p>
          </div>
        </div>
      </GlassCard>
    </div>
  );
};

export default HistoricalStats;
