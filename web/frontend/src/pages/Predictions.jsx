import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  ChartBarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  StarIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  AdjustmentsHorizontalIcon
} from '@heroicons/react/24/outline';
import GlassCard from '../components/GlassCard';
import { API_BASE } from '../config/api';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.05 }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { type: "spring", stiffness: 100 }
  }
};

export default function Predictions() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState('score');
  const [filterAction, setFilterAction] = useState('all');

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/predictions?limit=50`);
      const data = await response.json();
      setPredictions(data.predictions || []);
    } catch (error) {
      console.error('Erreur:', error);
    } finally {
      setLoading(false);
    }
  };

  const getScoreGradient = (score) => {
    if (score >= 75) return 'from-green-500 to-emerald-500';
    if (score >= 60) return 'from-yellow-500 to-amber-500';
    if (score >= 40) return 'from-orange-500 to-red-500';
    return 'from-red-500 to-pink-500';
  };

  const getActionColor = (action) => {
    switch (action) {
      case 'PARIER': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'SURVEILLER': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'ÉVITER': return 'bg-red-500/20 text-red-400 border-red-500/30';
      default: return 'bg-neutral-200/50 dark:bg-neutral-500/20 text-neutral-600 dark:text-neutral-400';
    }
  };

  const getNiveauIcon = (niveau) => {
    switch (niveau) {
      case 'FORT': return '🔥';
      case 'MOYEN': return '⚡';
      case 'FAIBLE': return '💡';
      default: return '❌';
    }
  };

  const filteredPredictions = predictions
    .filter(p => filterAction === 'all' || p.recommendation?.action === filterAction)
    .sort((a, b) => {
      if (sortBy === 'score') return b.score - a.score;
      if (sortBy === 'value') return b.value_bet - a.value_bet;
      if (sortBy === 'cote') return a.cote - b.cote;
      if (sortBy === 'proba') return b.probabilite - a.probabilite;
      return 0;
    });

  // Stats globales
  const stats = {
    total: predictions.length,
    parier: predictions.filter(p => p.recommendation?.action === 'PARIER').length,
    surveiller: predictions.filter(p => p.recommendation?.action === 'SURVEILLER').length,
    scoreMax: Math.max(...predictions.map(p => p.score), 0),
    valueMoyenne: predictions.length > 0
      ? (predictions.reduce((sum, p) => sum + (p.value_bet || 0), 0) / predictions.length).toFixed(1)
      : 0
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 px-4 sm:px-0">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col md:flex-row md:items-center md:justify-between gap-4"
      >
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            📜 Historique des prédictions
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">
            Archive des signaux récents basés sur notre algorithme de scoring
          </p>
        </div>
      </motion.div>

      {/* Stats Cards */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-2 md:grid-cols-4 gap-4"
      >
        <motion.div variants={itemVariants}>
          <GlassCard className="text-center">
            <div className="text-3xl font-bold text-blue-400">{stats.total}</div>
            <div className="text-sm text-neutral-600 dark:text-neutral-400">Prédictions</div>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard className="text-center">
            <div className="text-3xl font-bold text-green-400">{stats.parier}</div>
            <div className="text-sm text-neutral-600 dark:text-neutral-400">À Parier</div>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard className="text-center">
            <div className="text-3xl font-bold text-yellow-400">{stats.surveiller}</div>
            <div className="text-sm text-neutral-600 dark:text-neutral-400">À Surveiller</div>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard className="text-center">
            <div className="text-3xl font-bold text-purple-400">{stats.valueMoyenne}%</div>
            <div className="text-sm text-neutral-600 dark:text-neutral-400">Value Moyenne</div>
          </GlassCard>
        </motion.div>
      </motion.div>

      {/* Filtres */}
      <GlassCard>
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <AdjustmentsHorizontalIcon className="h-5 w-5 text-neutral-500 dark:text-neutral-400" />
            <span className="text-neutral-600 dark:text-neutral-400">Filtres:</span>
          </div>

          <div className="flex flex-wrap gap-2">
            {['all', 'PARIER', 'SURVEILLER', 'ÉVITER'].map(action => (
              <button
                key={action}
                onClick={() => setFilterAction(action)}
                className={`px-3 py-1 rounded-full text-sm transition-all ${filterAction === action
                    ? 'bg-blue-500 text-white'
                    : 'bg-neutral-100 dark:bg-white/5 text-neutral-600 dark:text-neutral-400 hover:bg-neutral-200 dark:hover:bg-white/10'
                  }`}
              >
                {action === 'all' ? 'Tous' : action}
              </button>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-2">
            <span className="text-neutral-600 dark:text-neutral-400 text-sm">Trier par:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-3 py-1 bg-neutral-100 dark:bg-white/5 border border-neutral-200 dark:border-white/10 rounded-lg text-[var(--color-text)] text-sm"
            >
              <option value="score">Score</option>
              <option value="value">Value Bet</option>
              <option value="cote">Cote</option>
              <option value="proba">Probabilité</option>
            </select>
          </div>
        </div>
      </GlassCard>

      {/* Liste des prédictions */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-48 bg-neutral-200 dark:bg-white/5 rounded-xl animate-pulse" />
          ))}
        </div>
      ) : predictions.length === 0 ? (
        <GlassCard className="text-center py-12 border border-yellow-500/30">
          <div className="text-6xl mb-4">🎯</div>
          <h3 className="text-xl font-semibold text-[var(--color-text)] mb-2">Aucune prédiction disponible</h3>
          <p className="text-neutral-600 dark:text-neutral-400 mb-4">
            Les prédictions sont générées à partir des courses du jour.
          </p>
          <div className="bg-neutral-100 dark:bg-white/5 rounded-lg p-4 max-w-md mx-auto">
            <p className="text-sm text-neutral-500 dark:text-neutral-400">
              💡 <strong>Comment obtenir des prédictions ?</strong>
            </p>
            <ul className="text-sm text-neutral-500 dark:text-neutral-400 mt-2 space-y-1 text-left">
              <li>• Vérifiez qu'il y a des courses programmées aujourd'hui</li>
              <li>• Les prédictions sont calculées ~2h avant chaque course</li>
              <li>• Consultez la page "Courses du jour" pour voir le programme</li>
            </ul>
          </div>
        </GlassCard>
      ) : (
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-2 gap-4"
        >
          {filteredPredictions.map((p, index) => (
            <motion.div key={`${p.race_key}-${p.nom}`} variants={itemVariants}>
              <GlassCard className="h-full hover:bg-neutral-50 dark:hover:bg-white/10 transition-all">
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-[var(--color-text)]">{p.nom}</h3>
                    <p className="text-sm text-neutral-500 dark:text-neutral-400">{p.hippodrome}</p>
                  </div>

                  {/* Score circulaire */}
                  <div className="relative">
                    <svg className="w-16 h-16 transform -rotate-90">
                      <circle
                        cx="32"
                        cy="32"
                        r="28"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="transparent"
                        className="text-neutral-200 dark:text-neutral-700"
                      />
                      <circle
                        cx="32"
                        cy="32"
                        r="28"
                        stroke="url(#gradient)"
                        strokeWidth="4"
                        fill="transparent"
                        strokeDasharray={`${(p.score / 100) * 176} 176`}
                        strokeLinecap="round"
                      />
                      <defs>
                        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" className={p.score >= 60 ? 'text-green-500' : 'text-red-500'} stopColor="currentColor" />
                          <stop offset="100%" className={p.score >= 60 ? 'text-emerald-500' : 'text-orange-500'} stopColor="currentColor" />
                        </linearGradient>
                      </defs>
                    </svg>
                    <span className={`absolute inset-0 flex items-center justify-center text-sm font-bold ${p.score >= 60 ? 'text-green-400' : 'text-orange-400'
                      }`}>
                      {p.score}
                    </span>
                  </div>
                </div>

                {/* Métriques */}
                <div className="grid grid-cols-3 gap-2 mb-4">
                  <div className="text-center p-2 bg-neutral-100 dark:bg-white/5 rounded-lg">
                    <div className="text-lg font-bold text-green-400">{p.cote}</div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">Cote</div>
                  </div>
                  <div className="text-center p-2 bg-neutral-100 dark:bg-white/5 rounded-lg">
                    <div className="text-lg font-bold text-blue-400">{p.probabilite}%</div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">Proba</div>
                  </div>
                  <div className="text-center p-2 bg-neutral-100 dark:bg-white/5 rounded-lg">
                    <div className={`text-lg font-bold ${p.value_bet > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {p.value_bet > 0 ? '+' : ''}{p.value_bet}%
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">Value</div>
                  </div>
                </div>

                {/* Indicateurs */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {p.tendance && (
                    <span className={`px-2 py-1 rounded text-xs flex items-center gap-1 ${p.tendance === '-' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                      }`}>
                      {p.tendance === '-' ? <ArrowTrendingDownIcon className="h-3 w-3" /> : <ArrowTrendingUpIcon className="h-3 w-3" />}
                      Cote {p.tendance === '-' ? 'en baisse' : 'en hausse'}
                    </span>
                  )}
                  {p.avis_entraineur && (
                    <span className={`px-2 py-1 rounded text-xs ${p.avis_entraineur === 'POSITIF' ? 'bg-green-500/20 text-green-400' :
                        p.avis_entraineur === 'NEGATIF' ? 'bg-red-500/20 text-red-400' :
                          'bg-gray-500/20 text-gray-400'
                      }`}>
                      Avis: {p.avis_entraineur}
                    </span>
                  )}
                </div>

                {/* Recommandation */}
                {p.recommendation && (
                  <div className={`p-3 rounded-lg border ${getActionColor(p.recommendation.action)}`}>
                    <div className="flex items-center justify-between">
                      <span className="font-semibold flex items-center gap-2">
                        {getNiveauIcon(p.recommendation.niveau)}
                        {p.recommendation.action}
                      </span>
                      <span className="text-sm opacity-75">
                        Mise: {p.recommendation.mise_recommandee}
                      </span>
                    </div>
                    <p className="text-xs mt-1 opacity-75">{p.recommendation.raison}</p>
                  </div>
                )}

                {/* Résultat réel */}
                {p.resultat && p.resultat !== 'En attente' && (
                  <div className={`mt-3 pt-3 border-t border-neutral-200 dark:border-white/10 flex items-center gap-2 ${p.resultat === 'Gagné' ? 'text-green-500 dark:text-green-400' :
                      p.resultat === 'Placé' ? 'text-yellow-500 dark:text-yellow-400' : 'text-neutral-500 dark:text-neutral-400'
                    }`}>
                    {p.resultat === 'Gagné' ? <CheckCircleIcon className="h-5 w-5" /> :
                      p.resultat === 'Placé' ? <StarIcon className="h-5 w-5" /> :
                        <XCircleIcon className="h-5 w-5" />}
                    <span className="font-medium">{p.resultat}</span>
                  </div>
                )}
              </GlassCard>
            </motion.div>
          ))}
        </motion.div>
      )}

      {/* Message si pas de résultats après filtrage */}
      {!loading && predictions.length > 0 && filteredPredictions.length === 0 && (
        <GlassCard className="text-center py-12">
          <div className="text-6xl mb-4">🔍</div>
          <p className="text-neutral-500 dark:text-neutral-400">Aucune prédiction ne correspond aux filtres sélectionnés</p>
          <button
            onClick={() => setFilterAction('all')}
            className="mt-4 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition"
          >
            Réinitialiser les filtres
          </button>
        </GlassCard>
      )}
    </div>
  );
}
