import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import useAuthenticatedUserSettings from '../hooks/useAuthenticatedUserSettings';
import {
  TrophyIcon,
  FireIcon,
  BoltIcon,
  ChartBarIcon,
  CurrencyEuroIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  InformationCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ScaleIcon
} from '@heroicons/react/24/outline';
import GlassCard from '../components/GlassCard';
import { API_BASE } from '../config/api';

// Configuration des profils utilisateur
const USER_PROFILES = {
  PRUDENT: {
    emoji: '🛡️',
    nom: 'Prudent',
    description: 'Sécurisé - Variance faible',
    color: 'blue'
  },
  STANDARD: {
    emoji: '⚖️',
    nom: 'Standard',
    description: 'Équilibré - Compromis risque/rendement',
    color: 'green'
  },
  AGRESSIF: {
    emoji: '🚀',
    nom: 'Agressif',
    description: 'Variance élevée, potentiel maximum',
    color: 'red'
  }
};

// Composant de configuration utilisateur authentifié
const UserConfigPanel = ({ userSettings, simulation, onRefresh, isAuthenticated }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!isAuthenticated || !userSettings) {
    return (
      <GlassCard className="mb-6">
        <div className="p-6 text-center">
          <ExclamationTriangleIcon className="w-12 h-12 mx-auto mb-3 text-yellow-500" />
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-white mb-2">Authentification requise</h3>
          <p className="text-neutral-600 dark:text-gray-400">
            Connectez-vous pour accéder aux recommandations personnalisées
          </p>
        </div>
      </GlassCard>
    );
  }

  return (
    <GlassCard className="mb-6">
      <div className="p-6">
        {/* En-tête avec toggle */}
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <ScaleIcon className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-white">Configuration Personnelle</h3>
              <p className="text-neutral-600 dark:text-gray-400 text-sm">
                Bankroll: {userSettings.bankroll}€ • Profil: {USER_PROFILES[userSettings.profil_risque]?.emoji} {USER_PROFILES[userSettings.profil_risque]?.nom}
              </p>
            </div>
          </div>
          {isExpanded ? (
            <ChevronUpIcon className="w-5 h-5 text-neutral-500 dark:text-gray-400" />
          ) : (
            <ChevronDownIcon className="w-5 h-5 text-neutral-500 dark:text-gray-400" />
          )}
        </div>

        {/* Panneau de configuration */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="mt-6 border-t border-gray-700 pt-6"
            >
              {/* Slider Bankroll */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-neutral-700 dark:text-gray-300 mb-3">
                  Bankroll: {bankroll}€
                </label>
                <input
                  type="range"
                  min="100"
                  max="2000"
                  step="50"
                  value={bankroll}
                  onChange={(e) => handleBankrollChange(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-neutral-500 dark:text-gray-500 mt-1">
                  <span>100€</span>
                  <span>1000€</span>
                  <span>2000€</span>
                </div>
              </div>

              {/* Sélection du profil */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-neutral-700 dark:text-gray-300 mb-3">
                  Profil de risque
                </label>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {Object.entries(USER_PROFILES).map(([key, profile]) => {
                    const isSelected = profil === key;
                    const config = profiles[key] || {};

                    return (
                      <motion.button
                        key={key}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => handleProfilChange(key)}
                        className={`p-4 rounded-lg border transition-all ${isSelected
                          ? 'border-blue-500 bg-blue-500/10 text-blue-600 dark:text-blue-400'
                          : 'border-neutral-300 dark:border-gray-600 bg-neutral-100 dark:bg-gray-800/50 text-neutral-600 dark:text-gray-300 hover:border-neutral-400 dark:hover:border-gray-500'
                          }`}
                      >
                        <div className="text-2xl mb-2">{profile.emoji}</div>
                        <div className="text-sm font-medium">{profile.nom}</div>
                        <div className="text-xs text-gray-400 mt-1">
                          {config.description || profile.description}
                        </div>
                        {config.max_stake_pct && (
                          <div className="text-xs text-neutral-500 dark:text-gray-500 mt-2">
                            Max: {config.max_stake_pct}% par pari
                          </div>
                        )}
                      </motion.button>
                    );
                  })}
                </div>
              </div>

              {/* Simulation du budget */}
              {simulation && (
                <div className="bg-neutral-100 dark:bg-gray-800/50 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-neutral-900 dark:text-white mb-3">💰 Budget journalier</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-neutral-600 dark:text-gray-400">Budget du jour:</span>
                      <span className="text-green-600 dark:text-green-400 ml-2 font-medium">
                        {simulation.budget_journalier}€
                      </span>
                    </div>
                    <div>
                      <span className="text-neutral-600 dark:text-gray-400">Max par pari:</span>
                      <span className="text-blue-600 dark:text-blue-400 ml-2 font-medium">
                        {simulation.mise_max_par_pari}€
                      </span>
                    </div>
                  </div>
                  <div className="mt-3 text-xs text-neutral-500 dark:text-gray-400">
                    {simulation.exemple}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </GlassCard>
  );
};

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
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

// Helper pour extraire les infos de course
const getRaceInfo = (raceKey) => {
  if (!raceKey) return { label: '??', code: '' };
  const parts = raceKey.split('|');
  // parts: [date, R, C, code]
  if (parts.length >= 4) {
    return {
      label: `${parts[1]} ${parts[2]}`,
      code: parts[3]
    };
  }
  return { label: raceKey, code: '' };
};

// Composant pour une carte de recommandation
const RecommandationCard = ({ rec, type }) => {
  const [expanded, setExpanded] = useState(false);
  const raceInfo = getRaceInfo(rec.race_key);

  const typeConfig = {
    SUR: {
      emoji: '🟢',
      bg: 'bg-gradient-to-br from-green-900/20 to-green-900/5',
      border: 'border-green-500/30',
      text: 'text-green-700 dark:text-green-400',
      accent: 'bg-green-500',
      badgeBg: 'bg-green-500/20',
      label: 'SÛR',
      desc: 'Faible risque'
    },
    EQUILIBRE: {
      emoji: '🟡',
      bg: 'bg-gradient-to-br from-yellow-900/20 to-yellow-900/5',
      border: 'border-yellow-500/30',
      text: 'text-yellow-700 dark:text-yellow-400',
      accent: 'bg-yellow-500',
      badgeBg: 'bg-yellow-500/20',
      label: 'ÉQUILIBRÉ',
      desc: 'Risque moyen'
    },
    RISQUE: {
      emoji: '🔴',
      bg: 'bg-gradient-to-br from-red-900/20 to-red-900/5',
      border: 'border-red-500/30',
      text: 'text-red-700 dark:text-red-400',
      accent: 'bg-red-500',
      badgeBg: 'bg-red-500/20',
      label: 'RISQUÉ',
      desc: 'Haut potentiel'
    }
  };

  const config = typeConfig[type] || typeConfig.EQUILIBRE;

  return (
    <motion.div
      variants={itemVariants}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      className={`relative overflow-hidden ${config.bg} ${config.border} border rounded-xl cursor-pointer group shadow-lg hover:shadow-xl transition-all duration-300`}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Barre latérale colorée */}
      <div className={`absolute top-0 left-0 w-1 h-full ${config.accent} opacity-50 group-hover:opacity-100 transition-opacity`} />

      <div className="p-5 pl-6">
        {/* En-tête : Course et Badge */}
        <div className="flex justify-between items-start mb-4">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 bg-neutral-100 dark:bg-white/10 rounded-md text-xs font-bold text-neutral-900 dark:text-white tracking-wider border border-neutral-200 dark:border-white/5">
              {raceInfo.label}
            </span>
            <span className="text-xs font-medium text-neutral-500 dark:text-gray-400">{raceInfo.code}</span>
          </div>
          <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${config.badgeBg} ${config.text} border ${config.border} flex items-center gap-1`}>
            <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse"></span>
            {config.label}
          </span>
        </div>

        {/* Type de pari recommandé */}
        <div className="mb-4">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-blue-600/30 to-purple-600/30 rounded-lg border border-blue-500/30">
            <span className="text-lg">🎰</span>
            <span className="text-sm font-bold text-neutral-900 dark:text-white">{rec.type_pari || 'GAGNANT'}</span>
          </div>
        </div>

        {/* Contenu Principal */}
        <div className="flex justify-between items-center mb-5">
          <div>
            <h3 className="text-xl font-bold text-neutral-900 dark:text-white group-hover:text-blue-400 transition-colors mb-1">
              {rec.cheval}
            </h3>
            <div className="flex items-center gap-2 text-sm text-neutral-500 dark:text-gray-400">
              <span className="bg-neutral-200 dark:bg-white/5 px-1.5 rounded text-neutral-700 dark:text-gray-300">N°{rec.numero}</span>
              <span>•</span>
              <span className="truncate max-w-[150px]">{rec.hippodrome}</span>
            </div>
          </div>
          <div className="text-right pl-4">
            <div className="text-3xl font-bold text-neutral-900 dark:text-white tracking-tight">{rec.cote?.toFixed(1)}</div>
            <div className="text-[10px] text-neutral-500 dark:text-gray-500 uppercase tracking-wider font-medium">Cote</div>
          </div>
        </div>

        {/* Indicateurs Clés */}
        <div className="grid grid-cols-2 gap-3 mb-2">
          <div className="bg-neutral-100 dark:bg-black/20 rounded-lg p-2 flex items-center justify-between border border-neutral-200 dark:border-white/5">
            <span className="text-xs text-neutral-500 dark:text-gray-400">Confiance</span>
            <span className="text-sm font-bold text-purple-600 dark:text-purple-400">{rec.proba_succes?.toFixed(0)}%</span>
          </div>
          {rec.esperance > 0 && (
            <div className="bg-neutral-100 dark:bg-black/20 rounded-lg p-2 flex items-center justify-between border border-neutral-200 dark:border-white/5">
              <span className="text-xs text-neutral-500 dark:text-gray-400">Value</span>
              <span className="text-sm font-bold text-green-600 dark:text-green-400">+{rec.esperance?.toFixed(1)}%</span>
            </div>
          )}
        </div>

        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="pt-4 mt-4 border-t border-neutral-200 dark:border-white/10 space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-neutral-500 dark:text-gray-500 text-xs uppercase tracking-wider block mb-1">Mise conseillée</span>
                    <span className="text-neutral-900 dark:text-white font-medium text-sm">{rec.mise_recommandee}</span>
                  </div>
                  <div>
                    <span className="text-neutral-500 dark:text-gray-500 text-xs uppercase tracking-wider block mb-1">Gain potentiel</span>
                    <span className="text-green-600 dark:text-green-400 font-medium text-sm">{rec.gain_potentiel}</span>
                  </div>
                </div>

                <div>
                  <span className="text-neutral-500 dark:text-gray-500 text-xs uppercase tracking-wider block mb-1">Analyse IA</span>
                  <p className="text-neutral-700 dark:text-gray-300 text-sm leading-relaxed bg-neutral-50 dark:bg-white/5 p-3 rounded-lg border border-neutral-200 dark:border-white/5">
                    {rec.raison}
                  </p>
                </div>

                {rec.details && rec.details.length > 0 && (
                  <div>
                    <span className="text-neutral-500 dark:text-gray-500 text-xs uppercase tracking-wider block mb-2">Signaux détectés</span>
                    <div className="flex flex-wrap gap-2">
                      {rec.details.map((detail, i) => (
                        <span key={i} className="text-xs text-neutral-700 dark:text-gray-300 bg-neutral-50 dark:bg-white/5 px-2.5 py-1.5 rounded-md border border-neutral-200 dark:border-white/5 flex items-center gap-1.5">
                          <CheckCircleIcon className="w-3 h-3 text-blue-600 dark:text-blue-400" />
                          {detail}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="mt-4 flex justify-center">
          <motion.div
            animate={{ rotate: expanded ? 180 : 0 }}
            transition={{ duration: 0.3 }}
          >
            <ChevronDownIcon className="h-4 w-4 text-neutral-600 dark:text-gray-600 group-hover:text-neutral-400 dark:group-hover:text-gray-400 transition-colors" />
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
};

// Composant principal
export default function Paris() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [selectedType, setSelectedType] = useState('all');
  const [error, setError] = useState(null);
  const [simulation, setSimulation] = useState(null);

  // Utiliser le hook authentifié pour les paramètres utilisateur
  const {
    userSettings,
    loading: settingsLoading,
    error: settingsError,
    isAuthenticated,
    bankroll,
    profilRisque
  } = useAuthenticatedUserSettings();

  // Fonction pour obtenir le token d'auth
  const getAuthToken = () => localStorage.getItem('authToken');

  // Headers avec authentification
  const getAuthHeaders = () => {
    const token = getAuthToken();
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` }),
    };
  };

  useEffect(() => {
    if (isAuthenticated && userSettings) {
      fetchRecommandations();
      fetchSimulation();
    }
  }, [isAuthenticated, userSettings]);

  const fetchSimulation = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/profils/simulation?bankroll=${bankroll}&profil=${profilRisque}`);
      const result = await response.json();
      setSimulation(result);
    } catch (err) {
      console.error('Erreur simulation:', err);
    }
  };

  const fetchRecommandations = async () => {
    if (!isAuthenticated) {
      setError('Authentification requise');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE}/api/paris/recommandations-jour`, {
        headers: getAuthHeaders()
      });

      if (response.status === 401) {
        setError('Session expirée, veuillez vous reconnecter');
        setLoading(false);
        return;
      }

      const result = await response.json();
      setData(result);
    } catch (err) {
      console.error('Erreur chargement recommandations:', err);
      setError('Impossible de charger les recommandations');
    } finally {
      setLoading(false);
    }
  };

  const getAllRecommandations = () => {
    if (!data?.meilleurs_paris) return [];

    const all = [];

    if (selectedType === 'all' || selectedType === 'sur') {
      data.meilleurs_paris.sur?.forEach(r => all.push({ ...r, type: 'SUR' }));
    }
    if (selectedType === 'all' || selectedType === 'equilibre') {
      data.meilleurs_paris.equilibre?.forEach(r => all.push({ ...r, type: 'EQUILIBRE' }));
    }
    if (selectedType === 'all' || selectedType === 'risque') {
      data.meilleurs_paris.risque?.forEach(r => all.push({ ...r, type: 'RISQUE' }));
    }

    return all;
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto space-y-6 px-4 sm:px-0">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-neutral-200 dark:bg-white/10 rounded w-1/3"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-48 bg-neutral-100 dark:bg-white/5 rounded-xl"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-0">
        <GlassCard className="text-center py-12">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-600 dark:text-red-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-[var(--color-text)]">{error}</h3>
          <button
            onClick={fetchRecommandations}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            Réessayer
          </button>
        </GlassCard>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6 px-4 sm:px-0">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col md:flex-row md:items-center md:justify-between gap-4"
      >
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-green-400 via-yellow-400 to-red-400 bg-clip-text text-transparent">
            🎯 Recommandations Paris
          </h1>
          <p className="text-neutral-600 dark:text-gray-400 mt-1">
            {data?.date} • {data?.nb_courses || 0} courses analysées
          </p>
        </div>

        {/* Filtres */}
        <div className="flex flex-wrap gap-2">
          {[
            { key: 'all', label: 'Tous', icon: ChartBarIcon },
            { key: 'sur', label: 'Sûrs 🟢', count: data?.resume?.total_paris_sur },
            { key: 'equilibre', label: 'Équilibrés 🟡', count: data?.resume?.total_paris_equilibre },
            { key: 'risque', label: 'Risqués 🔴', count: data?.resume?.total_paris_risque },
          ].map(filter => (
            <button
              key={filter.key}
              onClick={() => setSelectedType(filter.key)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${selectedType === filter.key
                ? 'bg-blue-500 text-white'
                : 'bg-neutral-100 dark:bg-white/5 text-neutral-600 dark:text-gray-400 hover:bg-neutral-200 dark:hover:bg-white/10'
                }`}
            >
              {filter.label}
              {filter.count !== undefined && (
                <span className="ml-1 opacity-60">({filter.count})</span>
              )}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Configuration utilisateur */}
      {!isAuthenticated ? (
        <GlassCard className="border border-yellow-500/30">
          <div className="p-6 text-center">
            <ExclamationTriangleIcon className="h-12 w-12 text-yellow-600 dark:text-yellow-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-[var(--color-text)] mb-2">
              Authentification requise
            </h3>
            <p className="text-neutral-600 dark:text-gray-400 mb-4">
              Connectez-vous pour accéder aux recommandations de paris personnalisées
            </p>
            <button
              onClick={() => window.location.href = '/login'}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
            >
              Se connecter
            </button>
          </div>
        </GlassCard>
      ) : (
        <UserConfigPanel
          userSettings={userSettings}
          simulation={simulation}
          onRefresh={fetchRecommandations}
          isAuthenticated={isAuthenticated}
        />
      )}

      {/* Résumé du profil utilisateur */}
      {data?.profil_utilisateur && (
        <GlassCard className="border border-blue-500/30">
          <div className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="text-2xl">{USER_PROFILES[data.profil_utilisateur.profil]?.emoji}</div>
                <div>
                  <h3 className="font-medium text-neutral-900 dark:text-white">
                    Profil {data.profil_utilisateur.profil} • Bankroll {data.profil_utilisateur.bankroll}€
                  </h3>
                  <p className="text-sm text-neutral-500 dark:text-gray-400">{data.profil_utilisateur.description}</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-neutral-500 dark:text-gray-400">Mise totale aujourd'hui</div>
                <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                  {data.profil_utilisateur.total_mise_jour}€
                </div>
                <div className="text-xs text-neutral-500 dark:text-gray-500">
                  {data.profil_utilisateur.pourcentage_bankroll_utilise}% de votre bankroll
                </div>
              </div>
            </div>
          </div>
        </GlassCard>
      )}

      {/* Stats rapides */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-gradient-to-br from-green-900/40 to-green-900/10 border border-green-500/20 rounded-xl p-6 relative overflow-hidden group hover:border-green-500/40 transition-colors"
        >
          <div className="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-green-500/20 rounded-full blur-2xl group-hover:bg-green-500/30 transition-colors"></div>
          <div className="relative z-10">
            <div className="flex justify-between items-start mb-4">
              <div className="p-3 bg-green-500/20 rounded-lg">
                <CheckCircleIcon className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <span className="text-3xl font-bold text-neutral-900 dark:text-white">{data?.resume?.total_paris_sur || 0}</span>
            </div>
            <h3 className="text-lg font-semibold text-green-700 dark:text-green-400 mb-1">Paris Sûrs</h3>
            <p className="text-sm text-neutral-600 dark:text-gray-400">Cote &lt; 5 • Faible risque</p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gradient-to-br from-yellow-900/40 to-yellow-900/10 border border-yellow-500/20 rounded-xl p-6 relative overflow-hidden group hover:border-yellow-500/40 transition-colors"
        >
          <div className="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-yellow-500/20 rounded-full blur-2xl group-hover:bg-yellow-500/30 transition-colors"></div>
          <div className="relative z-10">
            <div className="flex justify-between items-start mb-4">
              <div className="p-3 bg-yellow-500/20 rounded-lg">
                <ScaleIcon className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
              </div>
              <span className="text-3xl font-bold text-neutral-900 dark:text-white">{data?.resume?.total_paris_equilibre || 0}</span>
            </div>
            <h3 className="text-lg font-semibold text-yellow-700 dark:text-yellow-400 mb-1">Paris Équilibrés</h3>
            <p className="text-sm text-neutral-600 dark:text-gray-400">Cote 5-15 • Risque moyen</p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-gradient-to-br from-red-900/40 to-red-900/10 border border-red-500/20 rounded-xl p-6 relative overflow-hidden group hover:border-red-500/40 transition-colors"
        >
          <div className="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-red-500/20 rounded-full blur-2xl group-hover:bg-red-500/30 transition-colors"></div>
          <div className="relative z-10">
            <div className="flex justify-between items-start mb-4">
              <div className="p-3 bg-red-500/20 rounded-lg">
                <FireIcon className="w-6 h-6 text-red-600 dark:text-red-400" />
              </div>
              <span className="text-3xl font-bold text-neutral-900 dark:text-white">{data?.resume?.total_paris_risque || 0}</span>
            </div>
            <h3 className="text-lg font-semibold text-red-700 dark:text-red-400 mb-1">Paris Risqués</h3>
            <p className="text-sm text-neutral-600 dark:text-gray-400">Cote &gt; 15 • Haut potentiel</p>
          </div>
        </motion.div>
      </div>

      {/* Recommandations */}
      {getAllRecommandations().length === 0 ? (
        <GlassCard className="text-center py-12">
          <InformationCircleIcon className="h-12 w-12 text-blue-600 dark:text-blue-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-[var(--color-text)] mb-2">
            Aucune recommandation disponible
          </h3>
          <p className="text-neutral-600 dark:text-gray-400">
            Les recommandations sont générées pour les courses du jour.
            <br />Revenez plus tard ou vérifiez qu'il y a des courses programmées.
          </p>
        </GlassCard>
      ) : (
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          {getAllRecommandations().map((rec, index) => (
            <RecommandationCard key={`${rec.race_key}-${rec.numero}-${index}`} rec={rec} type={rec.type} />
          ))}
        </motion.div>
      )}

      {/* Légende */}
      <GlassCard className="mt-6">
        <h3 className="text-lg font-semibold text-[var(--color-text)] mb-4 flex items-center gap-2">
          <InformationCircleIcon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
          Guide des Recommandations
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/20">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">🟢</span>
              <span className="font-medium text-green-700 dark:text-green-400">PARI SÛR</span>
            </div>
            <ul className="text-sm text-neutral-600 dark:text-gray-400 space-y-1">
              <li>• Cote inférieure à 5</li>
              <li>• Score ≥ 60/100</li>
              <li>• Mise: 3-5% bankroll</li>
              <li>• Idéal pour débutants</li>
            </ul>
          </div>

          <div className="p-4 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">🟡</span>
              <span className="font-medium text-yellow-700 dark:text-yellow-400">PARI ÉQUILIBRÉ</span>
            </div>
            <ul className="text-sm text-neutral-600 dark:text-gray-400 space-y-1">
              <li>• Cote entre 5 et 15</li>
              <li>• Value bet positive</li>
              <li>• Mise: 2% bankroll</li>
              <li>• Bon rapport risque/gain</li>
            </ul>
          </div>

          <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/20">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">🔴</span>
              <span className="font-medium text-red-700 dark:text-red-400">PARI RISQUÉ</span>
            </div>
            <ul className="text-sm text-neutral-600 dark:text-gray-400 space-y-1">
              <li>• Cote supérieure à 15</li>
              <li>• Signaux positifs détectés</li>
              <li>• Mise: 1% bankroll max</li>
              <li>• Fort potentiel de gain</li>
            </ul>
          </div>
        </div>

        <div className="mt-4 p-4 bg-neutral-50 dark:bg-white/5 rounded-lg">
          <p className="text-sm text-neutral-500 dark:text-gray-400">
            ⚠️ <strong>Avertissement:</strong> Les paris sportifs comportent des risques.
            Ces recommandations sont basées sur des analyses statistiques et ne garantissent pas de gains.
            Jouez de manière responsable et ne misez que ce que vous pouvez vous permettre de perdre.
          </p>
        </div>
      </GlassCard>
    </div>
  );
}
