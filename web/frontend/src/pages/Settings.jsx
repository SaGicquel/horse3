import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Save, Info, AlertTriangle, Check, RefreshCw, Shield, Zap, Target, Settings as SettingsIcon, User, DollarSign, LogIn, UserPlus } from 'lucide-react';
import { GlassCard, GlassCardHeader } from '../components/GlassCard';
import { motion } from 'framer-motion';
import useAuthenticatedUserSettings from '../hooks/useAuthenticatedUserSettings';
import { useAuth } from '../context/AuthContext';

// Profils Kelly avec descriptions
const KELLY_PROFILES = [
  { id: 'SUR', label: 'Sûr', fraction: 0.25, color: 'success', icon: Shield, description: 'Très conservateur, 25% du Kelly théorique' },
  { id: 'STANDARD', label: 'Standard', fraction: 0.33, color: 'info', icon: Target, description: 'Équilibré, 33% du Kelly (recommandé)' },
  { id: 'AMBITIEUX', label: 'Ambitieux', fraction: 0.50, color: 'warning', icon: Zap, description: 'Agressif, 50% du Kelly théorique' },
  { id: 'PERSONNALISE', label: 'Personnalisé', fraction: null, color: 'primary', icon: SettingsIcon, description: 'Définir votre propre fraction' },
];

const Settings = () => {
  const navigate = useNavigate();
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState(false);

  // Utiliser le contexte d'auth global pour vérifier l'authentification
  const { isAuthenticated: authIsAuthenticated, isLoading: authIsLoading } = useAuth();

  // Hook pour les paramètres utilisateur authentifiés
  const {
    userSettings,
    loading: settingsLoading,
    error,
    updateSettings,

    // Valeurs individuelles
    bankroll,
    profilRisque,
    kellyProfile,
    customKellyFraction,
    valueCutoff,
    capPerBet,
    dailyBudgetRate,
    maxUnitBetsPerRace,
    roundingIncrementEur,
    perTicketRate,
    maxPackRate,
    marketMode,
    takeoutRate,

    // Setters
    setBankroll,
    setProfilRisque,
    setKellyProfile,
    setCustomKellyFraction,
    setValueCutoff,
    setCapPerBet,
    setDailyBudgetRate,
    setMaxUnitBetsPerRace,
    setRoundingIncrementEur,
    setPerTicketRate,
    setMaxPackRate,
    setMarketMode,
    setTakeoutRate,
  } = useAuthenticatedUserSettings();

  // Message de succès temporaire - DOIT être avant tout return conditionnel
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  // Gérer les changements de profil Kelly
  const handleKellyProfileChange = async (profileId) => {
    const profile = KELLY_PROFILES.find(p => p.id === profileId);
    if (profile && profile.fraction !== null) {
      await updateSettings({
        kelly_profile: profileId,
        custom_kelly_fraction: profile.fraction
      });
    } else {
      await setKellyProfile(profileId);
    }
  };

  // Calcul de la fraction Kelly actuelle
  const getCurrentKellyFraction = () => {
    if (kellyProfile === 'PERSONNALISE') {
      return customKellyFraction;
    }
    const profile = KELLY_PROFILES.find(p => p.id === kellyProfile);
    return profile?.fraction || 0.33;
  };

  const handleBulkSave = async () => {
    setSaving(true);
    setSuccess(false);
    try {
      // Toutes les modifications sont déjà sauvegardées automatiquement
      // Cette fonction est juste pour le feedback utilisateur
      setSuccess(true);
    } catch (err) {
      console.error('Erreur:', err);
    } finally {
      setSaving(false);
    }
  };

  // Attendre que l'auth context soit chargé
  if (authIsLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  // Si non authentifié (via AuthContext), afficher un message avec options de connexion
  if (!authIsAuthenticated) {
    return (
      <div className="flex items-center justify-center min-h-screen px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <GlassCard className="p-8 text-center max-w-md mx-auto">
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring" }}
            >
              <SettingsIcon className="w-16 h-16 mx-auto mb-4 text-primary-500" />
            </motion.div>

            <h2 className="text-2xl font-bold mb-3 text-neutral-900 dark:text-neutral-100">
              Accès aux Paramètres
            </h2>

            <p className="text-neutral-600 dark:text-neutral-400 mb-6">
              Connectez-vous à votre compte pour accéder à vos paramètres personnalisés :
            </p>

            <ul className="text-left text-sm text-neutral-600 dark:text-neutral-400 mb-6 space-y-2">
              <li className="flex items-center gap-2">
                <span className="text-primary-500">✓</span> Bankroll et gestion du risque
              </li>
              <li className="flex items-center gap-2">
                <span className="text-primary-500">✓</span> Profil Kelly personnalisé
              </li>
              <li className="flex items-center gap-2">
                <span className="text-primary-500">✓</span> Caps et limites de mise
              </li>
              <li className="flex items-center gap-2">
                <span className="text-primary-500">✓</span> Configuration paris exotiques
              </li>
            </ul>

            <div className="flex flex-col sm:flex-row gap-3">
              <motion.button
                onClick={() => navigate('/login')}
                className="flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-500/25"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <LogIn size={18} />
                Se connecter
              </motion.button>

              <motion.button
                onClick={() => navigate('/register')}
                className="flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold border-2 border-primary-500 text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/20"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <UserPlus size={18} />
                Créer un compte
              </motion.button>
            </div>

            <p className="text-xs text-neutral-500 dark:text-neutral-500 mt-4">
              Vos paramètres sont synchronisés sur tous vos appareils
            </p>
          </GlassCard>
        </motion.div>
      </div>
    );
  }

  // Loading des settings après vérification d'authentification
  if (settingsLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center">
        <div className="text-error mb-4">
          <AlertTriangle className="h-12 w-12 mx-auto" />
        </div>
        <h2 className="text-xl font-bold text-neutral-800 dark:text-neutral-200 mb-2">Erreur de chargement</h2>
        <p className="text-neutral-600 dark:text-neutral-400 mb-4">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="glass-button-primary flex items-center mx-auto gap-2"
        >
          <RefreshCw size={18} />
          Réessayer
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-3 sm:px-6 py-6 sm:py-12">
      <motion.div
        className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">Paramètres de Trading</h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-1">Politique de mise par défaut et gestion du risque</p>
        </div>
      </motion.div>

      {/* Section Paramètres Utilisateur */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-8"
      >
        <GlassCard>
          <GlassCardHeader
            icon={User}
            title="Paramètres Personnel"
            subtitle="Configuration de votre profil de risque et bankroll"
          />
          <div className="p-6 space-y-6">
            {/* Bankroll */}
            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-3">
                <DollarSign className="inline w-4 h-4 mr-2" />
                Bankroll: {bankroll}€
              </label>
              <input
                type="range"
                min="100"
                max="50000"
                step="100"
                value={bankroll}
                onChange={(e) => setBankroll(parseInt(e.target.value))}
                className="w-full h-2 bg-neutral-200 dark:bg-neutral-700 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                <span>100€</span>
                <span>5000€</span>
                <span>50000€</span>
              </div>
            </div>

            {/* Profil de risque */}
            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-3">
                Profil de risque
              </label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {[
                  { key: 'PRUDENT', emoji: '🛡️', nom: 'Prudent', desc: 'Conservateur, faible variance' },
                  { key: 'STANDARD', emoji: '⚖️', nom: 'Standard', desc: 'Équilibré, bon compromis' },
                  { key: 'AGRESSIF', emoji: '🚀', nom: 'Agressif', desc: 'Variance élevée, potentiel max' }
                ].map(profile => (
                  <button
                    key={profile.key}
                    onClick={() => setProfilRisque(profile.key)}
                    className={`p-4 rounded-lg border transition-all text-left ${profilRisque === profile.key
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-neutral-200 dark:border-neutral-700 hover:border-neutral-300 dark:hover:border-neutral-600'
                      }`}
                  >
                    <div className="text-2xl mb-2">{profile.emoji}</div>
                    <div className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                      {profile.nom}
                    </div>
                    <div className="text-xs text-neutral-600 dark:text-neutral-400 mt-1">
                      {profile.desc}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Résumé */}
            <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4">
              <h4 className="text-sm font-medium text-neutral-900 dark:text-neutral-100 mb-2">
                Configuration actuelle
              </h4>
              <div className="text-sm text-neutral-600 dark:text-neutral-400">
                <div>💰 Bankroll: <span className="font-medium text-green-600 dark:text-green-400">{bankroll}€</span></div>
                <div>🎯 Profil: <span className="font-medium text-blue-600 dark:text-blue-400">{profilRisque}</span></div>
                <div className="text-xs mt-2 text-neutral-500 dark:text-neutral-500">
                  Ces paramètres sont automatiquement sauvegardés et s'appliquent à toutes les pages.
                </div>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {error && (
        <motion.div
          className="mb-6 bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 p-4 rounded shadow-sm"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
            <p className="text-red-600 dark:text-red-400">{error}</p>
          </div>
        </motion.div>
      )}

      {success && (
        <motion.div
          className="mb-6 bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-4 rounded shadow-sm"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <div className="flex items-center">
            <Check className="h-5 w-5 text-green-500 mr-2" />
            <p className="text-green-600 dark:text-green-400">Configuration sauvegardée avec succès !</p>
          </div>
        </motion.div>
      )}

      <div className="space-y-6">

        {/* Bankroll Input */}
        <GlassCard
          className="bg-gradient-to-r from-[#db2777] to-[#9d174d] text-white border-none"
          hover={false}
          animate={true}
        >
          <div className="flex items-center justify-between p-6">
            <div>
              <h2 className="text-xl font-bold">Bankroll Actuelle</h2>
              <p className="text-primary-100 text-sm">Base de calcul pour les mises</p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="100"
                min="100"
                value={bankroll}
                onChange={(e) => setBankroll(parseFloat(e.target.value) || 1000)}
                className="w-32 px-4 py-2 rounded-lg bg-white/10 border border-white/20 text-white text-xl font-bold text-right focus:outline-none focus:ring-2 focus:ring-white/30 placeholder-white/50"
              />
              <span className="text-2xl font-bold">€</span>
            </div>
          </div>
        </GlassCard>

        {/* Kelly Profile Selection */}
        <GlassCard hover={false} animate={true} delay={0.1}>
          <div className="mb-6 border-b border-neutral-200 dark:border-neutral-700 pb-4 p-6">
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">Profil Kelly</h2>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">Sélectionnez votre niveau de risque</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 px-6 pb-6">
            {KELLY_PROFILES.map((profile) => {
              const Icon = profile.icon;
              const isSelected = kellyProfile === profile.id;

              let colorClass = '';
              if (isSelected) {
                if (profile.color === 'success') colorClass = 'border-green-500 bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400';
                else if (profile.color === 'info') colorClass = 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400';
                else if (profile.color === 'warning') colorClass = 'border-amber-500 bg-amber-50 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400';
                else colorClass = 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400';
              } else {
                colorClass = 'border-neutral-200 dark:border-neutral-700 hover:border-primary-300 dark:hover:border-primary-700 text-neutral-600 dark:text-neutral-400';
              }

              return (
                <button
                  key={profile.id}
                  type="button"
                  onClick={() => handleKellyProfileChange(profile.id)}
                  className={`p-4 rounded-xl border-2 transition-all text-left flex flex-col h-full ${colorClass}`}
                >
                  <Icon className={`h-8 w-8 mb-3 ${isSelected ? '' : 'text-neutral-400'}`} />
                  <div className="font-semibold mb-1">{profile.label}</div>
                  <div className="text-xs opacity-80 mb-2">
                    {profile.fraction !== null ? `${(profile.fraction * 100).toFixed(0)}% Kelly` : 'Custom'}
                  </div>
                  <div className="text-xs opacity-60 mt-auto">{profile.description}</div>
                </button>
              );
            })}
          </div>

          {/* Custom Kelly Fraction */}
          {(kellyProfile === 'PERSONNALISE') && (
            <motion.div
              className="mx-6 mb-6 p-4 rounded-lg bg-primary-50 dark:bg-primary-900/10 border border-primary-100 dark:border-primary-900/20"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
            >
              <label className="block text-sm font-medium text-primary-700 dark:text-primary-300 mb-2">
                Fraction Kelly personnalisée
              </label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.05"
                  value={customKellyFraction}
                  onChange={(e) => setCustomKellyFraction(parseFloat(e.target.value))}
                  className="flex-1 h-2 bg-neutral-200 dark:bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
                />
                <span className="text-xl font-bold text-primary-600 dark:text-primary-400 w-16 text-right">
                  {(customKellyFraction * 100).toFixed(0)}%
                </span>
              </div>
            </motion.div>
          )}
        </GlassCard>

        {/* Caps & Limits */}
        <GlassCard hover={false} animate={true} delay={0.2}>
          <div className="mb-6 border-b border-neutral-200 dark:border-neutral-700 pb-4 p-6 flex justify-between items-center">
            <div>
              <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">Caps & Limites</h2>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">Contrôle du risque de ruine</p>
            </div>
            <div className="text-neutral-400" title="Plafonds pour éviter les pertes catastrophiques">
              <Info className="h-5 w-5 cursor-help" />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 px-6 pb-6">
            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                Cap par pari
              </label>
              <div className="relative">
                <input
                  type="number"
                  step="0.5"
                  min="0.5"
                  max="10"
                  value={(capPerBet * 100)}
                  onChange={(e) => setCapPerBet(parseFloat(e.target.value) / 100)}
                  className="w-full px-4 py-2 rounded-lg bg-white dark:bg-white/5 border border-neutral-200 dark:border-white/10 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500 pr-12"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500">%</span>
              </div>
              <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                Max: {(bankroll * capPerBet).toFixed(2)}€ par pari
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                Budget journalier
              </label>
              <div className="relative">
                <input
                  type="number"
                  step="1"
                  min="1"
                  max="50"
                  value={(dailyBudgetRate * 100)}
                  onChange={(e) => setDailyBudgetRate(parseFloat(e.target.value) / 100)}
                  className="w-full px-4 py-2 rounded-lg bg-white dark:bg-white/5 border border-neutral-200 dark:border-white/10 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500 pr-12"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500">%</span>
              </div>
              <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                Max: {(bankroll * dailyBudgetRate).toFixed(2)}€ / jour
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                Value cutoff (EV min)
              </label>
              <div className="relative">
                <input
                  type="number"
                  step="1"
                  min="0"
                  max="30"
                  value={(valueCutoff * 100)}
                  onChange={(e) => setValueCutoff(parseFloat(e.target.value) / 100)}
                  className="w-full px-4 py-2 rounded-lg bg-white dark:bg-white/5 border border-neutral-200 dark:border-white/10 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500 pr-12"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500">%</span>
              </div>
              <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                Ne pas parier si value &lt; {(valueCutoff * 100).toFixed(0)}%
              </p>
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-neutral-200 dark:border-neutral-700 px-6 pb-6">
            <div className="max-w-xs">
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                Arrondi des mises
              </label>
              <select
                value={roundingIncrementEur}
                onChange={(e) => setRoundingIncrementEur(parseFloat(e.target.value))}
                className="w-full px-4 py-2 rounded-lg bg-white dark:bg-white/5 border border-neutral-200 dark:border-white/10 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="0.5">0.50€</option>
                <option value="1">1.00€</option>
                <option value="2">2.00€</option>
                <option value="5">5.00€</option>
              </select>
            </div>
          </div>
        </GlassCard>

        {/* Exotics Settings */}
        <GlassCard hover={false} animate={true} delay={0.3}>
          <div className="mb-6 border-b border-neutral-200 dark:border-neutral-700 pb-4 p-6">
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">Paris Exotiques (Trio/Quinté)</h2>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">Limites spécifiques pour les combinaisons</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 px-6 pb-6">
            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                Taux par ticket
              </label>
              <div className="relative">
                <input
                  type="number"
                  step="0.05"
                  min="0.1"
                  max="5"
                  value={(perTicketRate * 100)}
                  onChange={(e) => setPerTicketRate(parseFloat(e.target.value) / 100)}
                  className="w-full px-4 py-2 rounded-lg bg-white dark:bg-white/5 border border-neutral-200 dark:border-white/10 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500 pr-12"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500">%</span>
              </div>
              <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                Max: {(bankroll * perTicketRate).toFixed(2)}€ par combinaison
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                Max par pack
              </label>
              <div className="relative">
                <input
                  type="number"
                  step="0.5"
                  min="1"
                  max="20"
                  value={(maxPackRate * 100)}
                  onChange={(e) => setMaxPackRate(parseFloat(e.target.value) / 100)}
                  className="w-full px-4 py-2 rounded-lg bg-white dark:bg-white/5 border border-neutral-200 dark:border-white/10 text-neutral-900 dark:text-neutral-100 focus:outline-none focus:ring-2 focus:ring-primary-500 pr-12"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500">%</span>
              </div>
              <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                Max: {(bankroll * maxPackRate).toFixed(2)}€ total exotiques
              </p>
            </div>
          </div>
        </GlassCard>

        {/* Market Settings */}
        <GlassCard hover={false} animate={true} delay={0.4}>
          <div className="mb-6 border-b border-neutral-200 dark:border-neutral-700 pb-4 p-6 flex justify-between items-center">
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">Marché & Environnement</h2>
            <div className="text-neutral-400" title="Configuration de l'environnement de pari">
              <Info className="h-5 w-5 cursor-help" />
            </div>
          </div>
          <div className="px-6 pb-6">
            <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">Mode de Marché</label>
            <div className="flex flex-col sm:flex-row items-center gap-4">
              <button
                type="button"
                onClick={() => setMarketMode('parimutuel')}
                className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all w-full ${marketMode === 'parimutuel'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 font-medium'
                  : 'border-neutral-200 dark:border-neutral-700 hover:border-neutral-300 dark:hover:border-neutral-600 text-neutral-600 dark:text-neutral-400'
                  }`}
              >
                PMU (Parimutuel) - {(takeoutRate * 100).toFixed(0)}% prélèvement
              </button>
              <button
                type="button"
                onClick={() => setMarketMode('exchange')}
                className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all w-full ${marketMode === 'exchange'
                  ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 font-medium'
                  : 'border-neutral-200 dark:border-neutral-700 hover:border-neutral-300 dark:hover:border-neutral-600 text-neutral-600 dark:text-neutral-400'
                  }`}
              >
                Book (Exchange)
              </button>
            </div>
          </div>
        </GlassCard>

        {/* Summary Panel */}
        <GlassCard
          className="bg-neutral-900 text-white border-neutral-800"
          hover={false}
          animate={true}
          delay={0.5}
        >
          <div className="p-6">
            <h2 className="text-lg font-bold mb-4">📊 Résumé de la Politique</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-neutral-400">Profil</div>
                <div className="text-xl font-bold">{kellyProfile}</div>
                <div className="text-xs text-neutral-400">{(getCurrentKellyFraction() * 100).toFixed(0)}% Kelly</div>
              </div>
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-neutral-400">Cap / Pari</div>
                <div className="text-xl font-bold">{(bankroll * capPerBet).toFixed(0)}€</div>
                <div className="text-xs text-neutral-400">{(capPerBet * 100).toFixed(1)}% bankroll</div>
              </div>
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-neutral-400">Budget / Jour</div>
                <div className="text-xl font-bold">{(bankroll * dailyBudgetRate).toFixed(0)}€</div>
                <div className="text-xs text-neutral-400">{(dailyBudgetRate * 100).toFixed(0)}% bankroll</div>
              </div>
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-neutral-400">Value Min</div>
                <div className="text-xl font-bold">{(valueCutoff * 100).toFixed(0)}%</div>
                <div className="text-xs text-neutral-400">EV cutoff</div>
              </div>
            </div>
            <div className="mt-4 text-xs text-neutral-500">
              Formule Kelly: f* = (p×(o-1) - (1-p)) / (o-1) • Mise = bankroll × min(kelly_fraction × f*, cap_per_bet)
            </div>
          </div>
        </GlassCard>


        {/* Notification de succès automatique */}
        {success && (
          <motion.div
            className="bg-green-100 dark:bg-green-900/30 border border-green-200 dark:border-green-700 
                     rounded-lg p-4 flex items-center mb-6"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Check className="h-5 w-5 text-green-600 dark:text-green-400 mr-2" />
            <p className="text-green-700 dark:text-green-300">
              Tous vos paramètres sont sauvegardés automatiquement !
            </p>
          </motion.div>
        )}

        {/* Note informative */}
        <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 
                      rounded-lg p-4 flex items-start">
          <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-3 mt-0.5" />
          <div className="text-blue-700 dark:text-blue-300">
            <p className="font-medium mb-1">Configuration automatique</p>
            <p className="text-sm">
              Vos paramètres sont automatiquement sauvegardés sur votre compte utilisateur
              et synchronisés entre toutes les pages. Plus besoin de cliquer sur "Sauvegarder" !
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
