import React, { useState, useEffect } from 'react';
import { GlassCard, GlassCardHeader } from '../components/GlassCard';
import {
  TrendingUp,
  Target,
  Wallet,
  BarChart3,
  Crown,
  Zap,
  Shield,
  ArrowRight,
  DollarSign
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const UserDashboard = () => {
  const [todayStats, setTodayStats] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchTodayData();
  }, []);

  const fetchTodayData = async () => {
    try {
      setLoading(true);
      const today = new Date().toISOString().split('T')[0];

      // Fetch aujourd'hui conseils et portefeuille en parallèle
      const [adviceResponse, portfolioResponse] = await Promise.all([
        fetch(`http://localhost:8001/daily-advice?date_str=${today}`),
        fetch(`http://localhost:8001/portfolio?date_str=${today}`)
      ]);

      if (adviceResponse.ok) {
        const adviceData = await adviceResponse.json();
        setTodayStats({
          nb_conseils: adviceData.length,
          value_moyenne: adviceData.reduce((sum, a) => sum + a.value, 0) / adviceData.length || 0,
          mise_totale: adviceData.reduce((sum, a) => sum + a.mise_conseillee, 0),
          ev_moyenne: adviceData.reduce((sum, a) => sum + a.ev_pct, 0) / adviceData.length || 0
        });
      }

      if (portfolioResponse.ok) {
        const portfolioData = await portfolioResponse.json();
        setPortfolio(portfolioData);
      }

    } catch (error) {
      console.error('Erreur récupération données:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risque) => {
    if (risque <= 10) return 'text-green-600 dark:text-green-400';
    if (risque <= 25) return 'text-orange-600 dark:text-orange-400';
    return 'text-red-600 dark:text-red-400';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">

      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-700 rounded-xl p-8 text-white shadow-lg">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <Crown className="w-8 h-8 text-yellow-400" />
              <h1 className="text-3xl font-bold text-red-600">Horse3 Champion</h1>
            </div>
            <p className="text-blue-100 text-lg">
              Modèle XGBoost - ROI: 22.71% - Sharpe: 3.599
            </p>
            <div className="flex items-center gap-4 mt-4">
              <span className="bg-green-500/20 border border-green-400/30 text-white px-3 py-1 rounded-full flex items-center text-sm font-medium">
                <Shield className="w-3 h-3 mr-1" />
                Calibré ECE: 0.0112
              </span>
              <span className="bg-yellow-500/20 border border-yellow-400/30 text-white px-3 py-1 rounded-full flex items-center text-sm font-medium">
                <Zap className="w-3 h-3 mr-1" />
                Blend + Kelly
              </span>
            </div>
          </div>

          <div className="text-right">
            <p className="text-blue-200">Aujourd'hui</p>
            <p className="text-2xl font-bold">
              {new Date().toLocaleDateString('fr-FR', {
                weekday: 'long',
                day: 'numeric',
                month: 'long'
              })}
            </p>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      {todayStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">

          <GlassCard
            className="hover:bg-blue-50 dark:hover:bg-blue-900/10 cursor-pointer border-blue-200 dark:border-blue-800"
            onClick={() => navigate('/daily-advice')}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-600 dark:text-blue-400 text-sm font-medium">Paris Conseillés</p>
                <p className="text-3xl font-bold text-blue-900 dark:text-blue-100">{todayStats.nb_conseils}</p>
              </div>
              <Target className="w-8 h-8 text-blue-600 dark:text-blue-400" />
            </div>
            <div className="flex items-center justify-between mt-4">
              <p className="text-xs text-blue-700 dark:text-blue-400">Voir les conseils</p>
              <ArrowRight className="w-4 h-4 text-blue-700 dark:text-blue-400" />
            </div>
          </GlassCard>

          <GlassCard className="border-green-200 dark:border-green-800">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-600 dark:text-green-400 text-sm font-medium">Value Moyenne</p>
                <p className="text-3xl font-bold text-green-900 dark:text-green-100">
                  +{todayStats.value_moyenne.toFixed(1)}%
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-600 dark:text-green-400" />
            </div>
            <div className="mt-4">
              <p className="text-xs text-green-700 dark:text-green-400">
                EV: +{todayStats.ev_moyenne.toFixed(1)}%
              </p>
            </div>
          </GlassCard>

          <GlassCard
            className="hover:bg-orange-50 dark:hover:bg-orange-900/10 cursor-pointer border-orange-200 dark:border-orange-800"
            onClick={() => navigate('/portfolio')}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-orange-600 dark:text-orange-400 text-sm font-medium">Mise Totale</p>
                <p className="text-3xl font-bold text-orange-900 dark:text-orange-100">
                  {todayStats.mise_totale.toFixed(0)}€
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-orange-600 dark:text-orange-400" />
            </div>
            <div className="flex items-center justify-between mt-4">
              <p className="text-xs text-orange-700 dark:text-orange-400">Voir portefeuille</p>
              <ArrowRight className="w-4 h-4 text-orange-700 dark:text-orange-400" />
            </div>
          </GlassCard>

          <GlassCard
            className="hover:bg-purple-50 dark:hover:bg-purple-900/10 cursor-pointer border-purple-200 dark:border-purple-800"
            onClick={() => navigate('/historical-stats')}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-600 dark:text-purple-400 text-sm font-medium">ROI Global</p>
                <p className="text-3xl font-bold text-purple-900 dark:text-purple-100">+22.7%</p>
              </div>
              <BarChart3 className="w-8 h-8 text-purple-600 dark:text-purple-400" />
            </div>
            <div className="flex items-center justify-between mt-4">
              <p className="text-xs text-purple-700 dark:text-purple-400">Voir stats</p>
              <ArrowRight className="w-4 h-4 text-purple-700 dark:text-purple-400" />
            </div>
          </GlassCard>
        </div>
      )}

      {/* Main Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Conseils du Jour */}
        <GlassCard className="border-blue-200 dark:border-blue-800">
          <div className="flex items-center gap-2 mb-4 text-blue-700 dark:text-blue-400">
            <Target className="w-5 h-5" />
            <h3 className="font-semibold text-lg">Conseils du Jour</h3>
          </div>
          <p className="text-neutral-800 dark:text-neutral-400 mb-6">
            Découvrez les paris recommandés par le modèle champion avec probabilités calibrées,
            value betting et mises optimisées Kelly.
          </p>
          <button
            onClick={() => navigate('/daily-advice')}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-xl flex items-center justify-center gap-2 transition-colors"
          >
            <Target className="w-4 h-4" />
            Voir les Conseils
          </button>
        </GlassCard>

        {/* Portefeuille */}
        <GlassCard className="border-green-200 dark:border-green-800">
          <div className="flex items-center gap-2 mb-4 text-green-700 dark:text-green-400">
            <Wallet className="w-5 h-5" />
            <h3 className="font-semibold text-lg">Mon Portefeuille</h3>
          </div>
          <p className="text-neutral-800 dark:text-neutral-400 mb-6">
            Gérez votre bankroll, suivez vos mises quotidiennes et contrôlez votre exposition
            au risque avec des seuils intelligents.
          </p>
          {portfolio && (
            <div className="mb-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="flex justify-between items-center text-sm">
                <span className="text-green-700 dark:text-green-300">Risque aujourd'hui:</span>
                <span className={`font-bold ${getRiskColor(portfolio.risque_pct)}`}>
                  {portfolio.risque_pct.toFixed(1)}%
                </span>
              </div>
            </div>
          )}
          <button
            onClick={() => navigate('/portfolio')}
            className="w-full bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded-xl flex items-center justify-center gap-2 transition-colors"
          >
            <Wallet className="w-4 h-4" />
            Gérer le Portefeuille
          </button>
        </GlassCard>

        {/* Historique & Stats */}
        <GlassCard className="border-purple-200 dark:border-purple-800">
          <div className="flex items-center gap-2 mb-4 text-purple-700 dark:text-purple-400">
            <BarChart3 className="w-5 h-5" />
            <h3 className="font-semibold text-lg">Performance</h3>
          </div>
          <p className="text-neutral-800 dark:text-neutral-400 mb-6">
            Analysez les performances historiques, le ROI mensuel, les drawdowns et
            l'évolution de votre capital sur la durée.
          </p>
          <div className="mb-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-purple-700 dark:text-purple-300">Sharpe Ratio:</span>
                <span className="font-bold ml-1 text-neutral-900 dark:text-neutral-100">3.599</span>
              </div>
              <div>
                <span className="text-purple-700 dark:text-purple-300">Max DD:</span>
                <span className="font-bold ml-1 text-neutral-900 dark:text-neutral-100">25.6%</span>
              </div>
            </div>
          </div>
          <button
            onClick={() => navigate('/historical-stats')}
            className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded-xl flex items-center justify-center gap-2 transition-colors"
          >
            <BarChart3 className="w-4 h-4" />
            Voir les Stats
          </button>
        </GlassCard>
      </div>

      {/* System Info */}
      <GlassCard className="border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-white/5">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center">
              <Crown className="w-6 h-6 text-yellow-400" />
            </div>
            <div>
              <h3 className="font-semibold text-neutral-900 dark:text-neutral-100">Modèle Champion XGBoost</h3>

              <p className="text-sm text-neutral-800 dark:text-neutral-400">
                Version 1.0 - Calibré Platt - Blend + Kelly - Validé par backtest
              </p>
            </div>
          </div>

          <div className="flex items-center gap-6 text-sm">
            <div className="text-center">
              <p className="text-neutral-700 dark:text-neutral-400">Dernière mise à jour</p>
              <p className="font-semibold text-neutral-900 dark:text-neutral-100">8 déc. 2025</p>
            </div>
            <div className="text-center">
              <p className="text-neutral-700 dark:text-neutral-400">Status</p>
              <span className="bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                Opérationnel
              </span>
            </div>
          </div>
        </div>
      </GlassCard>

      {/* Quick Tips */}
      <GlassCard className="border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/10">
        <div className="flex items-start gap-3">
          <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
            <span className="text-white text-xs font-bold">💡</span>
          </div>
          <div>
            <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">🚀 Conseils pour Optimiser vos Gains</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800 dark:text-blue-200">
              <div>
                <p className="font-medium mb-1">📊 Suivez les métriques :</p>
                <p>• Maintenez un risque entre 5-15% de votre bankroll<br />• Diversifiez vos paris sur plusieurs courses<br />• Respectez les mises conseillées par Kelly</p>
              </div>
              <div>
                <p className="font-medium mb-1">🎯 Maximisez la performance :</p>
                <p>• Privilégiez les paris à forte value (&gt;10%)<br />• Suivez l'évolution de votre ROI mensuel<br />• Ajustez votre bankroll selon vos résultats</p>
              </div>
            </div>
          </div>
        </div>
      </GlassCard>
    </div>
  );
};

export default UserDashboard;