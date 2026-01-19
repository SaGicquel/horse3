import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  CurrencyDollarIcon,
  ArrowTrendingUpIcon,
  ChartPieIcon,
  BoltIcon,
  AdjustmentsHorizontalIcon,
  CheckBadgeIcon,
  ExclamationTriangleIcon,
  FireIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';
import GlassCard from '../components/GlassCard';
import { API_BASE } from '../config/api';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08 }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1
  }
};

export default function Betting() {
  const [recommendations, setRecommendations] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    minScore: 60,
    minValue: 0,
    maxCote: 50
  });
  const [selectedBets, setSelectedBets] = useState([]);
  const [bankroll, setBankroll] = useState(1000);

  useEffect(() => {
    fetchRecommendations();
  }, [filters]);

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        min_score: filters.minScore,
        min_value: filters.minValue,
        max_cote: filters.maxCote
      });
      const response = await fetch(`${API_BASE}/api/betting/recommendations?${params}`);
      const data = await response.json();
      setRecommendations(data.recommendations || []);
      setStats(data.stats);
    } catch (error) {
      console.error('Erreur:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleBet = (rec) => {
    setSelectedBets(prev => {
      const exists = prev.find(b => b.nom === rec.nom && b.race_key === rec.race_key);
      if (exists) {
        return prev.filter(b => !(b.nom === rec.nom && b.race_key === rec.race_key));
      }
      return [...prev, rec];
    });
  };

  const calculateMise = (rec) => {
    // Kelly Criterion simplifié
    const mise_pct = rec.recommendation.niveau === 'FORT' ? 0.05 :
                     rec.recommendation.niveau === 'MOYEN' ? 0.02 : 0.01;
    return Math.round(bankroll * mise_pct);
  };

  const totalMise = selectedBets.reduce((sum, bet) => sum + calculateMise(bet), 0);
  const potentialGain = selectedBets.reduce((sum, bet) => sum + (calculateMise(bet) * bet.cote), 0);

  const getNiveauStyle = (niveau) => {
    switch (niveau) {
      case 'FORT':
        return {
          bg: 'bg-gradient-to-r from-green-500/20 to-emerald-500/20',
          border: 'border-green-500/50',
          text: 'text-green-400',
          icon: <FireIcon className="h-6 w-6" />
        };
      case 'MOYEN':
        return {
          bg: 'bg-gradient-to-r from-yellow-500/20 to-amber-500/20',
          border: 'border-yellow-500/50',
          text: 'text-yellow-400',
          icon: <BoltIcon className="h-6 w-6" />
        };
      case 'FAIBLE':
        return {
          bg: 'bg-gradient-to-r from-blue-500/20 to-cyan-500/20',
          border: 'border-blue-500/50',
          text: 'text-blue-400',
          icon: <SparklesIcon className="h-6 w-6" />
        };
      default:
        return {
          bg: 'bg-gray-500/20',
          border: 'border-gray-500/50',
          text: 'text-gray-400',
          icon: <ExclamationTriangleIcon className="h-6 w-6" />
        };
    }
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
          <h1 className="text-3xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
            💰 Recommandations de Paris
          </h1>
          <p className="text-gray-400 mt-1">
            Suggestions optimisées basées sur l'analyse Value Bet
          </p>
        </div>
      </motion.div>

      {/* Stats Summary */}
      {stats && (
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-2 md:grid-cols-5 gap-4"
        >
          <motion.div variants={itemVariants}>
            <GlassCard className="text-center">
              <FireIcon className="h-8 w-8 mx-auto text-green-400 mb-2" />
              <div className="text-2xl font-bold text-green-400">{stats.paris_forts}</div>
              <div className="text-xs text-gray-400">Paris Forts</div>
            </GlassCard>
          </motion.div>

          <motion.div variants={itemVariants}>
            <GlassCard className="text-center">
              <BoltIcon className="h-8 w-8 mx-auto text-yellow-400 mb-2" />
              <div className="text-2xl font-bold text-yellow-400">{stats.paris_moyens}</div>
              <div className="text-xs text-gray-400">Paris Moyens</div>
            </GlassCard>
          </motion.div>

          <motion.div variants={itemVariants}>
            <GlassCard className="text-center">
              <ChartPieIcon className="h-8 w-8 mx-auto text-blue-400 mb-2" />
              <div className="text-2xl font-bold text-blue-400">{stats.total_recommandations}</div>
              <div className="text-xs text-gray-400">Total</div>
            </GlassCard>
          </motion.div>

          <motion.div variants={itemVariants}>
            <GlassCard className="text-center">
              <ArrowTrendingUpIcon className="h-8 w-8 mx-auto text-purple-400 mb-2" />
              <div className="text-2xl font-bold text-purple-400">{stats.value_moyenne}%</div>
              <div className="text-xs text-gray-400">Value Moyenne</div>
            </GlassCard>
          </motion.div>

          <motion.div variants={itemVariants}>
            <GlassCard className="text-center">
              <CheckBadgeIcon className="h-8 w-8 mx-auto text-pink-400 mb-2" />
              <div className="text-2xl font-bold text-pink-400">{stats.score_moyen}</div>
              <div className="text-xs text-gray-400">Score Moyen</div>
            </GlassCard>
          </motion.div>
        </motion.div>
      )}

      {/* Filtres et Bankroll */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <GlassCard className="lg:col-span-2">
          <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4 flex items-center gap-2">
            <AdjustmentsHorizontalIcon className="h-5 w-5 text-blue-400" />
            Filtres
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Score minimum</label>
              <input
                type="range"
                min="0"
                max="100"
                value={filters.minScore}
                onChange={(e) => setFilters(f => ({ ...f, minScore: parseInt(e.target.value) }))}
                className="w-full"
              />
              <div className="text-center text-blue-400 font-bold">{filters.minScore}</div>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">Value minimum (%)</label>
              <input
                type="range"
                min="-20"
                max="50"
                value={filters.minValue}
                onChange={(e) => setFilters(f => ({ ...f, minValue: parseInt(e.target.value) }))}
                className="w-full"
              />
              <div className="text-center text-green-400 font-bold">{filters.minValue}%</div>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">Cote maximum</label>
              <input
                type="range"
                min="2"
                max="100"
                value={filters.maxCote}
                onChange={(e) => setFilters(f => ({ ...f, maxCote: parseInt(e.target.value) }))}
                className="w-full"
              />
              <div className="text-center text-yellow-400 font-bold">{filters.maxCote}</div>
            </div>
          </div>
        </GlassCard>

        <GlassCard>
          <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4 flex items-center gap-2">
            <CurrencyDollarIcon className="h-5 w-5 text-green-400" />
            Bankroll
          </h3>

          <input
            type="number"
            value={bankroll}
            onChange={(e) => setBankroll(parseInt(e.target.value) || 0)}
            className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-[var(--color-text)] text-2xl font-bold text-center"
          />
          <p className="text-center text-gray-400 text-sm mt-2">€</p>
        </GlassCard>
      </div>

      {/* Panier de paris */}
      {selectedBets.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <GlassCard className="border border-green-500/30 bg-green-500/5">
            <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4">🎰 Votre sélection ({selectedBets.length} paris)</h3>

            <div className="space-y-2 mb-4">
              {selectedBets.map((bet, i) => (
                <div key={i} className="flex items-center justify-between p-2 bg-white/5 rounded-lg">
                  <div>
                    <span className="font-medium text-[var(--color-text)]">{bet.nom}</span>
                    <span className="text-gray-400 text-sm ml-2">@ {bet.cote}</span>
                  </div>
                  <div className="text-green-400 font-bold">{calculateMise(bet)}€</div>
                </div>
              ))}
            </div>

            <div className="flex justify-between items-center pt-4 border-t border-white/10">
              <div>
                <span className="text-gray-400">Mise totale:</span>
                <span className="text-[var(--color-text)] font-bold ml-2">{totalMise}€</span>
              </div>
              <div>
                <span className="text-gray-400">Gain potentiel:</span>
                <span className="text-green-400 font-bold ml-2">{potentialGain.toFixed(0)}€</span>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* Liste des recommandations */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(9)].map((_, i) => (
            <div key={i} className="h-64 bg-white/5 rounded-xl animate-pulse" />
          ))}
        </div>
      ) : (
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          {recommendations.map((rec, index) => {
            const style = getNiveauStyle(rec.recommendation?.niveau);
            const isSelected = selectedBets.some(b => b.nom === rec.nom && b.race_key === rec.race_key);

            return (
              <motion.div key={`${rec.race_key}-${rec.nom}`} variants={itemVariants}>
                <GlassCard
                  className={`h-full transition-all cursor-pointer ${style.bg} border ${style.border} ${
                    isSelected ? 'ring-2 ring-green-500' : ''
                  }`}
                  onClick={() => toggleBet(rec)}
                >
                  {/* Badge niveau */}
                  <div className={`absolute top-3 right-3 ${style.text}`}>
                    {style.icon}
                  </div>

                  {/* Header */}
                  <div className="mb-4">
                    <h3 className="text-lg font-bold text-[var(--color-text)] pr-8">{rec.nom}</h3>
                    <p className="text-sm text-gray-400">{rec.hippodrome}</p>
                    {rec.type_course && (
                      <span className="text-xs text-gray-500">{rec.type_course} • {rec.distance}m</span>
                    )}
                  </div>

                  {/* Métriques principales */}
                  <div className="grid grid-cols-3 gap-2 mb-4">
                    <div className="text-center p-2 bg-black/20 rounded-lg">
                      <div className="text-xl font-bold text-[var(--color-text)]">{rec.cote}</div>
                      <div className="text-xs text-gray-500">Cote</div>
                    </div>
                    <div className="text-center p-2 bg-black/20 rounded-lg">
                      <div className="text-xl font-bold text-blue-400">{rec.score}</div>
                      <div className="text-xs text-gray-500">Score</div>
                    </div>
                    <div className="text-center p-2 bg-black/20 rounded-lg">
                      <div className={`text-xl font-bold ${rec.value_bet > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {rec.value_bet > 0 ? '+' : ''}{rec.value_bet}%
                      </div>
                      <div className="text-xs text-gray-500">Value</div>
                    </div>
                  </div>

                  {/* Probabilité */}
                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Probabilité estimée</span>
                      <span className="text-purple-400 font-bold">{rec.probabilite}%</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${rec.probabilite}%` }}
                        transition={{ duration: 1, delay: index * 0.1 }}
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                      />
                    </div>
                  </div>

                  {/* Recommandation */}
                  <div className={`p-3 rounded-lg bg-black/30 ${style.text}`}>
                    <div className="flex justify-between items-center">
                      <span className="font-bold">{rec.recommendation?.action}</span>
                      <span className="text-sm">Mise: {rec.recommendation?.mise_recommandee}</span>
                    </div>
                    <p className="text-xs opacity-75 mt-1">{rec.recommendation?.raison}</p>
                  </div>

                  {/* Mise suggérée */}
                  <div className="mt-4 pt-4 border-t border-white/10 flex justify-between items-center">
                    <span className="text-gray-400">Mise suggérée:</span>
                    <span className="text-xl font-bold text-green-400">{calculateMise(rec)}€</span>
                  </div>

                  {/* Résultat réel si disponible */}
                  {rec.resultat_reel && rec.resultat_reel !== 'En attente' && (
                    <div className={`mt-3 text-center py-2 rounded-lg ${
                      rec.resultat_reel === 'Gagné' ? 'bg-green-500/20 text-green-400' :
                      rec.resultat_reel === 'Placé' ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {rec.resultat_reel === 'Gagné' ? '✅' : rec.resultat_reel === 'Placé' ? '🥈' : '❌'} {rec.resultat_reel}
                    </div>
                  )}

                  {/* Indicateur sélection */}
                  {isSelected && (
                    <div className="absolute top-3 left-3">
                      <CheckBadgeIcon className="h-6 w-6 text-green-400" />
                    </div>
                  )}
                </GlassCard>
              </motion.div>
            );
          })}
        </motion.div>
      )}

      {/* Message si pas de résultats */}
      {!loading && recommendations.length === 0 && (
        <GlassCard className="text-center py-12">
          <div className="text-6xl mb-4">🎯</div>
          <p className="text-gray-400">Aucune recommandation ne correspond aux critères</p>
          <p className="text-gray-500 text-sm mt-2">Essayez d'ajuster les filtres</p>
        </GlassCard>
      )}
    </div>
  );
}
