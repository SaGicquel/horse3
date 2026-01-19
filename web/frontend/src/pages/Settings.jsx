import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Save, Info, AlertTriangle, Check, RefreshCw, Shield, Zap, Target, Settings as SettingsIcon, User, DollarSign, LogIn, UserPlus } from 'lucide-react';
import { GlassCard, GlassCardHeader } from '../components/GlassCard';
import PageHeader from '../components/PageHeader';
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
  const [syncLoading, setSyncLoading] = useState(false);
  const [syncResult, setSyncResult] = useState(null);

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
              >
                <LogIn size={18} />
                Se connecter
              </motion.button>

              <motion.button
                onClick={() => navigate('/register')}
                className="flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold border-2 border-primary-500 text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/20"
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
    <div className="max-w-7xl mx-auto px-3 sm:px-6 py-6 sm:py-12">
      {/* Header unifié */}
      <PageHeader
        emoji="⚙️"
        title="Paramètres de Trading"
        subtitle="Politique de mise par défaut et gestion du risque"
      />

      {/* ============================================ */}
      {/* SECTION STRATÉGIE RECOMMANDÉE */}
      {/* ============================================ */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="mb-8"
      >
        <GlassCard className="border-2 border-emerald-500/30 bg-gradient-to-br from-emerald-500/5 to-green-500/5">
          <div className="p-6">
            {/* Header avec badge */}
            <div className="flex flex-wrap items-start justify-between gap-4 mb-4">
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-xl bg-emerald-500/20">
                  <Shield className="w-8 h-8 text-emerald-500" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-xl font-bold text-neutral-900 dark:text-white">
                      Stratégie Ultra-Conservateur
                    </h2>
                    <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
                      RECOMMANDÉE
                    </span>
                  </div>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                    Validée par backtest sur 5 ans de données (661k courses)
                  </p>
                </div>
              </div>
            </div>

            {/* Stats du backtest */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              <div className="p-3 rounded-lg bg-white/50 dark:bg-white/5 border border-emerald-500/20">
                <div className="text-xs text-neutral-500 dark:text-neutral-400">ROI sur 1 an</div>
                <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">+31%</div>
              </div>
              <div className="p-3 rounded-lg bg-white/50 dark:bg-white/5 border border-emerald-500/20">
                <div className="text-xs text-neutral-500 dark:text-neutral-400">Sharpe Ratio</div>
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">3.67</div>
              </div>
              <div className="p-3 rounded-lg bg-white/50 dark:bg-white/5 border border-emerald-500/20">
                <div className="text-xs text-neutral-500 dark:text-neutral-400">Semaines +</div>
                <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">78%</div>
              </div>
              <div className="p-3 rounded-lg bg-white/50 dark:bg-white/5 border border-emerald-500/20">
                <div className="text-xs text-neutral-500 dark:text-neutral-400">Max Drawdown</div>
                <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">27%</div>
              </div>
            </div>

            {/* Paramètres clés */}
            <div className="bg-neutral-100 dark:bg-neutral-800/50 rounded-lg p-4 mb-4">
              <h4 className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-3">
                📋 Paramètres optimaux utilisés
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                <div className="flex items-center gap-2">
                  <span className="text-emerald-500">✓</span>
                  <span className="text-neutral-600 dark:text-neutral-400">Kelly: <strong className="text-neutral-900 dark:text-white">12%</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-emerald-500">✓</span>
                  <span className="text-neutral-600 dark:text-neutral-400">Value min: <strong className="text-neutral-900 dark:text-white">15%</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-emerald-500">✓</span>
                  <span className="text-neutral-600 dark:text-neutral-400">Cotes max: <strong className="text-neutral-900 dark:text-white">6</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-emerald-500">✓</span>
                  <span className="text-neutral-600 dark:text-neutral-400">Paris/jour: <strong className="text-neutral-900 dark:text-white">3 max</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-emerald-500">✓</span>
                  <span className="text-neutral-600 dark:text-neutral-400">Type: <strong className="text-neutral-900 dark:text-white">PLACÉ uniquement</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-emerald-500">✓</span>
                  <span className="text-neutral-600 dark:text-neutral-400">Proba min: <strong className="text-neutral-900 dark:text-white">18%</strong></span>
                </div>
              </div>
            </div>

            {/* Note explicative */}
            <div className="flex items-start gap-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
              <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-700 dark:text-blue-300">
                <strong>Pourquoi cette stratégie ?</strong> Elle offre le meilleur équilibre entre régularité (78% de semaines positives)
                et rendement (+31% ROI). Vous réduisez les séries de pertes tout en maintenant une croissance stable.
                <span className="block mt-1 text-blue-600 dark:text-blue-400">
                  Vous pouvez personnaliser ces paramètres ci-dessous si vous le souhaitez.
                </span>
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {/* ============================================ */}
      {/* SECTION GESTION AUTOMATIQUE DU BANKROLL */}
      {/* ============================================ */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.08 }}
        className="mb-8"
      >
        <GlassCard className="border-2 border-purple-500/30 bg-gradient-to-br from-purple-500/5 to-pink-500/5">
          <div className="p-6">
            {/* Header */}
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 rounded-xl bg-purple-500/20">
                <DollarSign className="w-8 h-8 text-purple-500" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-neutral-900 dark:text-white">
                  💰 Gestion Automatique du Bankroll
                </h2>
                <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-1">
                  Comment vos gains/pertes sont réintégrés au capital
                </p>
              </div>
            </div>

            {/* Explication des zones */}
            <div className="space-y-4 mb-6">
              {/* Zone Micro/Small */}
              <div className="p-4 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800">
                <div className="flex items-start gap-3">
                  <span className="text-2xl">🚀</span>
                  <div>
                    <h4 className="font-semibold text-emerald-700 dark:text-emerald-300 mb-1">
                      Zones Micro (&lt;50€) et Small (50-500€)
                    </h4>
                    <p className="text-sm text-emerald-600 dark:text-emerald-400 mb-2">
                      <strong>Compound automatique</strong> - Vos gains sont ajoutés immédiatement à votre bankroll.
                    </p>
                    <ul className="text-xs text-emerald-600 dark:text-emerald-400 space-y-1">
                      <li>✓ Mise fixe (2€ ou 5€), donc risque contrôlé</li>
                      <li>✓ Permet de passer plus vite en zone supérieure</li>
                      <li>✓ Les pertes sont également soustraites immédiatement</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Zone Full */}
              <div className="p-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
                <div className="flex items-start gap-3">
                  <span className="text-2xl">⚖️</span>
                  <div>
                    <h4 className="font-semibold text-amber-700 dark:text-amber-300 mb-1">
                      Zone Full (&gt;500€)
                    </h4>
                    <p className="text-sm text-amber-600 dark:text-amber-400 mb-2">
                      <strong>Mise à jour hebdomadaire avec réserve</strong> - Plus prudent car mises proportionnelles.
                    </p>
                    <ul className="text-xs text-amber-600 dark:text-amber-400 space-y-1">
                      <li>✓ <strong>Profit &gt; 0</strong>: BK = BK + 50% du profit (50% en réserve)</li>
                      <li>✓ <strong>Perte &lt; 10%</strong>: BK inchangé (absorber le choc)</li>
                      <li>✓ <strong>Perte &gt; 10%</strong>: BK = BK réel (descendre de zone si &lt;500€)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Votre zone actuelle */}
            <div className="bg-neutral-100 dark:bg-neutral-800/50 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-3">
                📊 Votre situation actuelle
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-3 rounded-lg bg-white/50 dark:bg-white/5">
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">Bankroll</div>
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">{bankroll}€</div>
                </div>
                <div className="text-center p-3 rounded-lg bg-white/50 dark:bg-white/5">
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">Zone</div>
                  <div className={`text-2xl font-bold ${bankroll < 50 ? 'text-red-500' :
                    bankroll < 500 ? 'text-amber-500' : 'text-emerald-500'
                    }`}>
                    {bankroll < 50 ? 'MICRO' : bankroll < 500 ? 'SMALL' : 'FULL'}
                  </div>
                </div>
                <div className="text-center p-3 rounded-lg bg-white/50 dark:bg-white/5">
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">Mode</div>
                  <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                    {bankroll < 500 ? 'Compound Auto' : 'Hebdo +50%'}
                  </div>
                </div>
              </div>

              {/* Conseil personnalisé */}
              <div className="mt-4 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                <div className="text-sm text-blue-700 dark:text-blue-300">
                  {bankroll < 50 ? (
                    <>
                      <strong>💡 Conseil Micro:</strong> Misez 2€ fixe par pari. Votre objectif: atteindre 50€ pour passer en zone Small.
                      {' '}À chaque gain, votre BK augmente automatiquement.
                    </>
                  ) : bankroll < 500 ? (
                    <>
                      <strong>💡 Conseil Small:</strong> Misez 5€ fixe par pari. Votre objectif: atteindre 500€ pour passer en zone Full.
                      {' '}Les gains sont ajoutés automatiquement à votre capital.
                    </>
                  ) : (
                    <>
                      <strong>💡 Conseil Full:</strong> Vos mises sont calculées avec Kelly 12%.
                      {' '}Mettez à jour votre bankroll chaque dimanche en ajoutant 50% de vos gains de la semaine.
                      {' '}Gardez 50% en réserve de sécurité.
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Bouton de synchronisation */}
            <div className="mt-6 pt-4 border-t border-purple-200 dark:border-purple-800">
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                <button
                  onClick={async () => {
                    setSyncLoading(true);
                    setSyncResult(null);
                    try {
                      const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
                      const token = localStorage.getItem('authToken');
                      const response = await fetch(`${API_BASE}/api/bets/summary`, {
                        headers: {
                          'Authorization': `Bearer ${token}`,
                          'Content-Type': 'application/json'
                        }
                      });
                      if (!response.ok) throw new Error('Erreur lors de la récupération des paris');
                      const data = await response.json();
                      const pnl = data.pnl_net || 0;
                      const totalBets = data.total_bets || 0;
                      const finishedBets = data.finished_bets || 0;

                      // Règles par zone
                      const currentBK = bankroll;
                      let newBK = currentBK;
                      let rule = '';

                      if (currentBK < 500) {
                        // MICRO/SMALL: compound automatique
                        newBK = Math.max(10, currentBK + pnl);
                        rule = 'Compound auto: BK + PnL';
                      } else {
                        // FULL: règles hebdomadaires
                        if (pnl > 0) {
                          newBK = currentBK + (pnl * 0.5);
                          rule = 'Profit: +50% des gains (50% en réserve)';
                        } else if (pnl < 0 && Math.abs(pnl) < currentBK * 0.1) {
                          newBK = currentBK; // Absorber les petites pertes
                          rule = 'Perte <10%: BK inchangé (absorption)';
                        } else if (pnl < 0) {
                          newBK = Math.max(10, currentBK + pnl);
                          rule = 'Perte >10%: BK réel appliqué';
                        }
                      }

                      setSyncResult({
                        success: true,
                        pnl: pnl,
                        oldBK: currentBK,
                        newBK: Math.round(newBK * 100) / 100,
                        rule: rule,
                        totalBets: totalBets,
                        finishedBets: finishedBets
                      });
                    } catch (err) {
                      setSyncResult({ success: false, error: err.message });
                    } finally {
                      setSyncLoading(false);
                    }
                  }}
                  disabled={syncLoading}
                  className="flex items-center gap-2 px-6 py-3 rounded-lg bg-purple-600 hover:bg-purple-700 text-white font-medium transition-all disabled:opacity-50"
                >
                  <RefreshCw className={`w-5 h-5 ${syncLoading ? 'animate-spin' : ''}`} />
                  {syncLoading ? 'Calcul en cours...' : '🔄 Synchroniser mon BK'}
                </button>
                <p className="text-xs text-neutral-500 dark:text-neutral-400">
                  Calcule automatiquement votre nouveau bankroll basé sur vos résultats de paris
                </p>
              </div>

              {/* Résultat de la synchronisation */}
              {syncResult && (
                <div className={`mt-4 p-4 rounded-lg ${syncResult.success
                  ? 'bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800'
                  : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
                  }`}>
                  {syncResult.success ? (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 text-emerald-700 dark:text-emerald-300">
                        <Check className="w-5 h-5" />
                        <span className="font-semibold">Calcul terminé !</span>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                        <div>
                          <div className="text-neutral-500 dark:text-neutral-400">Paris terminés</div>
                          <div className="font-bold text-neutral-900 dark:text-white">{syncResult.finishedBets}</div>
                        </div>
                        <div>
                          <div className="text-neutral-500 dark:text-neutral-400">P&L Net</div>
                          <div className={`font-bold ${syncResult.pnl >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                            {syncResult.pnl >= 0 ? '+' : ''}{syncResult.pnl.toFixed(2)}€
                          </div>
                        </div>
                        <div>
                          <div className="text-neutral-500 dark:text-neutral-400">BK Actuel</div>
                          <div className="font-bold text-neutral-900 dark:text-white">{syncResult.oldBK}€</div>
                        </div>
                        <div>
                          <div className="text-neutral-500 dark:text-neutral-400">Nouveau BK</div>
                          <div className="font-bold text-purple-600 dark:text-purple-400">{syncResult.newBK}€</div>
                        </div>
                      </div>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        <strong>Règle appliquée:</strong> {syncResult.rule}
                      </div>
                      {syncResult.newBK !== syncResult.oldBK && (
                        <button
                          onClick={() => {
                            setBankroll(syncResult.newBK, true);
                            setSyncResult(null);
                          }}
                          className="w-full mt-2 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white font-medium transition-all"
                        >
                          ✅ Appliquer le nouveau BK ({syncResult.newBK}€)
                        </button>
                      )}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-red-700 dark:text-red-300">
                      <AlertTriangle className="w-5 h-5" />
                      <span>Erreur: {syncResult.error}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </GlassCard>
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
                min="10"
                max="50000"
                step="10"
                value={bankroll}
                onChange={(e) => setBankroll(parseInt(e.target.value), false)}
                onMouseUp={(e) => setBankroll(parseInt(e.target.value), true)}
                onTouchEnd={(e) => setBankroll(parseInt(e.target.value), true)}
                className="w-full h-2 bg-neutral-200 dark:bg-neutral-700 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                <span>10€</span>
                <span>500€</span>
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
                step="10"
                min="10"
                value={bankroll}
                onChange={(e) => setBankroll(parseFloat(e.target.value) || 100)}
                onBlur={(e) => setBankroll(parseFloat(e.target.value) || 100, true)}
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


        {/* Summary Panel - Adapté à la zone */}
        <GlassCard
          className="bg-neutral-900 text-white border-neutral-800"
          hover={false}
          animate={true}
          delay={0.4}
        >
          <div className="p-6">
            <h2 className="text-lg font-bold mb-4">📊 Résumé de la Politique</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-neutral-400">Zone</div>
                <div className={`text-xl font-bold ${bankroll < 50 ? 'text-red-400' : bankroll < 500 ? 'text-amber-400' : 'text-emerald-400'}`}>
                  {bankroll < 50 ? 'MICRO' : bankroll < 500 ? 'SMALL' : 'FULL'}
                </div>
                <div className="text-xs text-neutral-400">
                  {bankroll < 50 ? 'Flat 2€' : bankroll < 500 ? 'Flat 5€' : 'Kelly 12%'}
                </div>
              </div>
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-neutral-400">Mise / Pari</div>
                <div className="text-xl font-bold">
                  {bankroll < 50 ? '2€' : bankroll < 500 ? '5€' : `${Math.min(bankroll * 0.03, 30).toFixed(0)}€`}
                </div>
                <div className="text-xs text-neutral-400">
                  {bankroll < 500 ? 'Mise fixe' : 'Max 3% BK'}
                </div>
              </div>
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-neutral-400">Paris / Jour</div>
                <div className="text-xl font-bold">
                  {bankroll < 50 ? '2 max' : bankroll < 500 ? '3 max' : '3 max'}
                </div>
                <div className="text-xs text-neutral-400">Limite quotidienne</div>
              </div>
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-neutral-400">Value Min</div>
                <div className="text-xl font-bold">
                  {bankroll < 50 ? '15%' : bankroll < 500 ? '12%' : '15%'}
                </div>
                <div className="text-xs text-neutral-400">EV cutoff</div>
              </div>
            </div>
            <div className="mt-4 text-xs text-neutral-500">
              {bankroll < 500
                ? 'Mode Flat: mise fixe pour contrôler la variance. Compound automatique des gains.'
                : 'Mode Kelly: mise = bankroll × kelly_fraction × f*. Mise à jour hebdo avec réserve 50%.'
              }
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
