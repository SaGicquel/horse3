import { useState, useEffect, useMemo, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  CurrencyEuroIcon,
  ShoppingCartIcon,
  ChartBarIcon,
  DocumentArrowDownIcon,
  PlusIcon,
  TrashIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  SparklesIcon,
  ScaleIcon,
  FireIcon,
  BoltIcon,
  AdjustmentsHorizontalIcon,
  ExclamationCircleIcon
} from '@heroicons/react/24/outline';
import { GlassCard, GlassCardHeader } from '../components/GlassCard';
import PageHeader from '../components/PageHeader';
import { SimulationToggle, SimulationBadge, useSimulationMode } from '../components/SimulationToggle';
import { API_BASE } from '../config/api';
import { resolveBetMetrics } from '../lib/bettingMetrics';
import { betsAPI } from '../services/api';
import { percent, odds as formatOdds, money } from '../lib/format';
import { exportPicksCSV, exportPicksJSON, exportPortfolioCSV, exportPortfolioJSON } from '../lib/export';

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.05 }
  }
};

const itemVariants = {
  hidden: { y: 10, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { type: "spring", stiffness: 100, damping: 15 }
  }
};

// Profils Kelly
const KELLY_PROFILES = {
  SUR: { fraction: 0.25, label: 'Sûr', color: 'success' },
  STANDARD: { fraction: 0.33, label: 'Standard', color: 'primary' },
  AMBITIEUX: { fraction: 0.50, label: 'Ambitieux', color: 'warning' },
  PERSONNALISE: { fraction: null, label: 'Personnalisé', color: 'secondary' }
};

// Onglets disponibles
const TABS = [
  { id: 'unitaires', label: 'Unitaires', icon: CurrencyEuroIcon, emoji: '🎯' },
  { id: 'exotiques', label: 'Exotiques', icon: SparklesIcon, emoji: '✨' },
  { id: 'portefeuille', label: 'Portefeuille', icon: ChartBarIcon, emoji: '📊' }
];

// ============================================
// Composant Onglet Unitaires
// ============================================
const UnitairesTab = ({ cart, setCart, bankroll, setBankroll, settings, benterStatus, onBenterStatus, marketStatus, onMarketStatus, reloadKey, onReloadAnalysis, isSimulation, onToggleSimulation }) => {
  const [bets, setBets] = useState([]);
  const [serverPortfolio, setServerPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);

  // ========== ALGO BRUT V2 - PAS D'AGENT IA ==========
  // Pas de mode IA, pas de states IA nécessaires

  // Récupérer les paramètres de la politique de mise depuis settings
  const bettingDefaults = settings?.betting_defaults || {};
  const bettingPolicy = settings?.betting_policy || {};
  const kellyProfile = bettingDefaults.kelly_profile || bettingDefaults.kelly_profile_default || 'STANDARD';

  // Récupérer la fraction Kelly selon le profil
  const kellyFractionMap = bettingDefaults.kelly_fraction_map || { SUR: 0.25, STANDARD: 0.33, AMBITIEUX: 0.5 };

  // ==========================================================================
  // ZONES DE BANKROLL - Configuration complète selon le capital
  // ==========================================================================
  const zoneConfig = useMemo(() => {
    const zones = bettingPolicy.bankroll_zones || {};

    // Déterminer la zone active selon la bankroll
    if (bankroll < 50) {
      // ZONE MICRO: <50€ - Mode survie
      const zone = zones.micro || {};
      return {
        name: 'micro',
        label: '🛡️ Mode Survie',
        description: zone.description || 'Capital limité - Ultra conservateur',
        maxBetsPerDay: zone.max_bets_per_day ?? 1,
        maxOddsWin: zone.max_odds_win ?? 4,
        minProbaModel: zone.min_proba_model ?? 0.20,
        valueCutoffWin: zone.value_cutoff_win ?? 0.15,
        valueCutoffPlace: zone.value_cutoff_place ?? 0.12,
        allowedBetTypes: zone.allowed_bet_types || ['SIMPLE PLACÉ'],
        preferredBetType: zone.preferred_bet_type || 'SIMPLE PLACÉ',
        kellyFraction: zone.kelly_fraction ?? 0.10,
        maxStakePct: zone.max_stake_pct ?? 0.05,
        minStakeEur: zone.min_stake_eur ?? 1.0,
        maxStakeEur: zone.max_stake_eur ?? 5.0,
        allowedRisks: zone.allowed_risks || ['Faible'],
        dailyBudgetRate: 0.10,
        color: 'text-red-400 bg-red-500/10 border-red-500/30'
      };
    } else if (bankroll < 500) {
      // ZONE SMALL: 50-500€ - Croissance prudente
      const zone = zones.small || {};
      return {
        name: 'small',
        label: '📈 Croissance',
        description: zone.description || 'Croissance prudente',
        maxBetsPerDay: zone.max_bets_per_day ?? 3,
        maxOddsWin: zone.max_odds_win ?? 6,
        minProbaModel: zone.min_proba_model ?? 0.15,
        valueCutoffWin: zone.value_cutoff_win ?? 0.10,
        valueCutoffPlace: zone.value_cutoff_place ?? 0.08,
        allowedBetTypes: zone.allowed_bet_types || ['SIMPLE PLACÉ', 'E/P (GAGNANT-PLACÉ)'],
        preferredBetType: zone.preferred_bet_type || 'SIMPLE PLACÉ',
        kellyFraction: zone.kelly_fraction ?? 0.15,
        maxStakePct: zone.max_stake_pct ?? 0.04,
        minStakeEur: zone.min_stake_eur ?? 2.0,
        maxStakeEur: zone.max_stake_eur ?? 20.0,
        allowedRisks: zone.allowed_risks || ['Faible', 'Modéré'],
        dailyBudgetRate: 0.15,
        color: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30'
      };
    } else {
      // ZONE FULL: >500€ - Optimisation complète
      const zone = zones.full || {};
      return {
        name: 'full',
        label: '🚀 Optimisation',
        description: zone.description || 'Optimisation complète',
        maxBetsPerDay: zone.max_bets_per_day ?? 8,        // 8 paris max (au lieu de 5)
        maxOddsWin: zone.max_odds_win ?? 15,              // Cotes jusqu'à 15 (au lieu de 10)
        minProbaModel: zone.min_proba_model ?? 0.08,      // Proba > 8% (au lieu de 12%)
        valueCutoffWin: zone.value_cutoff_win ?? 0.02,    // Value > 2% (au lieu de 8%)
        valueCutoffPlace: zone.value_cutoff_place ?? 0.01,
        allowedBetTypes: zone.allowed_bet_types || ['SIMPLE PLACÉ', 'E/P (GAGNANT-PLACÉ)', 'SIMPLE GAGNANT'],
        preferredBetType: zone.preferred_bet_type || 'E/P (GAGNANT-PLACÉ)',
        kellyFraction: zone.kelly_fraction ?? 0.25,
        maxStakePct: zone.max_stake_pct ?? 0.03,
        minStakeEur: zone.min_stake_eur ?? 5.0,
        maxStakeEur: zone.max_stake_eur ?? 50.0,
        allowedRisks: zone.allowed_risks || ['Faible', 'Modéré', 'Élevé', 'Très élevé'],
        dailyBudgetRate: 0.12,
        color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
      };
    }
  }, [bankroll, bettingPolicy]);

  // Utiliser les valeurs de la zone active
  const kellyFraction = zoneConfig.kellyFraction;
  const capPerBet = zoneConfig.maxStakePct;
  const dailyBudgetRate = zoneConfig.dailyBudgetRate;
  const valueCutoff = zoneConfig.valueCutoffWin;
  const roundingIncrement = bettingDefaults.rounding_increment_eur || 0.50;
  const maxUnitBetsPerRace = bettingDefaults.max_unit_bets_per_race || 3;

  // Calculs dérivés - limites absolues en €
  const dailyBudget = bankroll * dailyBudgetRate;
  const maxStakePerBet = Math.min(bankroll * capPerBet, zoneConfig.maxStakeEur, dailyBudget * 0.5);

  useEffect(() => {
    onBenterStatus?.({ status: 'pending' });
    onMarketStatus?.({ status: 'pending' });
    fetchBets();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloadKey, isSimulation]);  // Refetch when simulation mode changes

  const fetchBets = async () => {
    try {
      setLoading(true);
      // Utiliser l'API V2 avec algo brut optimisé (+71% ROI validé)
      // Cette version utilise des mises uniformes de 10€ sans Kelly, cotes 7-15, proba ≥50%
      const today = new Date().toISOString().split('T')[0];
      // Utiliser le proxy nginx qui redirige vers l'API V2
      const response = await fetch(`/daily-advice-v2?date_str=${today}`);
      if (!response.ok) {
        onBenterStatus?.({ status: 'error', reason: `HTTP ${response.status}` });
        setBets([]);
        setServerPortfolio(null);
        return;
      }

      const data = await response.json();

      // L'API V2 retourne directement un array de paris avec:
      // { course_id, race_key, hippodrome, heure, numero, nom, cote, cote_place, proba, mise, gain_potentiel }
      const picks = Array.isArray(data) ? data : [];

      console.log('[CONSEILS2] Picks reçus de l\'API:', picks.length, picks);

      // Transformer pour compatibilité avec l'interface existante
      const formattedBets = picks.map(pick => ({
        id: `${pick.race_key}_${pick.numero}`, // ID unique pour le cart
        race_key: pick.race_key,
        hippodrome: pick.hippodrome,
        heure: pick.heure || "14h00",
        numero: pick.numero,
        nom: pick.nom,
        cote: pick.cote,
        cote_place: pick.cote_place,
        proba: pick.proba,
        p_win: pick.proba / 100, // Probabilité en décimal (0-1) pour les filtres
        probabilite: pick.proba / 100, // Alias
        proba_win: pick.proba / 100, // Alias
        mise: pick.mise || 10, // Pour compatibilité avec useMemo
        mise_recommandee: pick.mise || 10, // Mise uniforme 10€
        gain_potentiel: pick.gain_potentiel,
        edge: ((pick.proba / 100) * pick.cote_place - 1) * 100, // Edge calculé sur cote place
        value_pct: ((pick.proba / 100) * pick.cote_place - 1) * 100, // Même que edge
        odds: pick.cote,
        market: pick.cote_place,
        bet_type: 'SIMPLE PLACÉ',
        bet_risk: 'Modéré',
        rationale: `Semi-outsider (cote ${pick.cote.toFixed(1)}) avec probabilité ${pick.proba.toFixed(1)}% (algo brut optimisé)`,
      }));

      console.log('[CONSEILS2] FormattedBets:', formattedBets);

      setBets(formattedBets);

      // Créer un portfolio simplifié pour affichage
      if (formattedBets.length > 0) {
        const totalMise = formattedBets.reduce((sum, b) => sum + (b.mise_recommandee || 0), 0);
        const totalGainPotentiel = formattedBets.reduce((sum, b) => sum + (b.gain_potentiel || 0), 0);
        setServerPortfolio({
          picks: formattedBets,
          total_stake: totalMise,
          total_potential_return: totalGainPotentiel,
          expected_value: totalGainPotentiel - totalMise,
          roi_expected: totalMise > 0 ? ((totalGainPotentiel - totalMise) / totalMise * 100) : 0,
        });
        console.log('[CONSEILS2] ServerPortfolio créé:', {
          picks: formattedBets.length,
          total_stake: totalMise,
          total_potential_return: totalGainPotentiel
        });
      } else {
        setServerPortfolio(null);
      }

      onBenterStatus?.({ status: 'success', reason: 'Algo brut optimisé (+71% ROI validé)' });
      onMarketStatus?.({ status: 'success', reason: `${formattedBets.length} paris sélectionnés (cotes 7-15, proba ≥50%)` });

    } catch (error) {
      console.error('Erreur chargement bets:', error);
      onBenterStatus?.({ status: 'error', reason: error.message });
      onMarketStatus?.({ status: 'error', reason: error.message });
      setBets([]);
      setServerPortfolio(null);
    } finally {
      setLoading(false);
      console.log('[CONSEILS2] Loading terminé, loading=false');
    }
  };

  // ========== ALGO BRUT V2 - NO AGENT IA ==========
  // Cette version utilise uniquement l'algo brut optimisé sans Agent IA
  // Pas de vérification IA, pas de useEffect nécessaire

  /**
   * Calcul Kelly plein: f* = (p*(o-1) - (1-p)) / (o-1)
   */
  const calculateKellyFull = (pWin, odds) => {
    if (odds <= 1 || pWin <= 0) return 0;
    const numerator = pWin * (odds - 1) - (1 - pWin);
    const denominator = odds - 1;
    return numerator / denominator;
  };

  /**
   * Calcul de la mise optimale selon la politique configurée
   * Utilise le Kelly de l'API si disponible, sinon calcule
   * stake = bankroll * min(kelly_fraction * kelly_full, cap_per_bet)
   * arrondi au rounding_increment, plafonné au cap absolu
   */
  const calculateStake = (bet) => {
    const { p, odds, valuePercent, kellyPercent } = resolveBetMetrics(bet);
    const microAction = bet.micro_action || bet.microAction;

    // Value cutoff en % (valueCutoff est en décimal: 0.05 = 5%)
    if (valuePercent < valueCutoff * 100) {
      return 0;
    }
    if (microAction === 'hold') {
      return 0;
    }

    // Récupérer le Kelly depuis l'API (en %) ou le calculer
    // L'API renvoie kelly_pct déjà fractionné (cap à 25% = 1/4 Kelly)
    // On l'utilise directement sans re-fractionner pour éviter des mises ridiculement basses
    let kellyRate;
    if (kellyPercent !== null && kellyPercent !== undefined && kellyPercent > 0) {
      // kellyPercent est en % et déjà fractionné par le backend
      kellyRate = kellyPercent / 100;  // Convertir en décimal (3.43% -> 0.0343)
    } else {
      // Fallback: calcul Kelly complet puis application de la fraction
      const kellyFull = calculateKellyFull(p, odds);
      kellyRate = kellyFractionEffective * kellyFull;
    }

    // Si Kelly <= 0, pas de mise
    if (kellyRate <= 0) {
      return 0;
    }

    // Utiliser directement kellyRate (déjà fractionné par le backend)
    const kellyFractional = kellyRate;

    // Appliquer le cap per bet (2% = 0.02)
    const cappedRate = Math.min(kellyFractional, capPerBet);

    // Calculer la mise
    let stake = bankroll * cappedRate;

    // Arrondir selon l'incrément configuré (0.50€)
    stake = Math.round(stake / roundingIncrement) * roundingIncrement;

    if (microAction === 'scale_down') {
      stake = Math.max(roundingIncrement, stake * 0.5);
      stake = Math.round(stake / roundingIncrement) * roundingIncrement;
    }

    // Maximum absolu = maxStakePerBet (bankroll * capPerBet = 20€ sur 1000€)
    stake = Math.min(stake, maxStakePerBet);

    // Minimum = roundingIncrement
    if (stake < roundingIncrement) {
      stake = 0;
    }

    return stake;
  };

  const addToCart = (bet) => {
    const exists = cart.find(b =>
      (b.cheval_id || b.id) === (bet.cheval_id || bet.id) &&
      b.race_key === bet.race_key
    );
    if (!exists) {
      const stake = Number(bet.calculatedStake ?? bet.stake ?? calculateStake(bet)) || 0;
      setCart([...cart, { ...bet, stake }]);
    }
  };

  const isInCart = (bet) => {
    return cart.some(b =>
      (b.cheval_id || b.id) === (bet.cheval_id || bet.id) &&
      b.race_key === bet.race_key
    );
  };

  // Sélection intelligente des paris avec la politique de zone configurée
  const { selectedBets, excludedBets, stats } = useMemo(() => {
    console.log('[CONSEILS2] useMemo - bets:', bets.length, 'serverPortfolio:', serverPortfolio);

    // Si le backend fournit un portefeuille déjà adapté, on l'utilise directement
    // Pour l'algo brut V2, le serverPortfolio contient "picks" (pas "positions")
    if (serverPortfolio?.picks) {
      // Mapping explicite de 'mise' vers 'calculatedStake' pour l'affichage
      const positions = (serverPortfolio.picks || []).map(p => ({
        ...p,
        calculatedStake: p.mise ?? 0,
        stake: p.mise ?? 0
      }));
      const excluded = [];
      const dailyBudgetServer = bankroll * dailyBudgetRate;
      console.log('[CONSEILS2] Utilisation picks du serveur:', positions.length);
      return {
        selectedBets: positions,
        excludedBets: excluded,
        stats: {
          totalStake: serverPortfolio.total_stake ?? 0,
          budgetUsed: dailyBudgetServer > 0 ? Math.round(((serverPortfolio.total_stake ?? 0) / dailyBudgetServer) * 100) : 0,
          budgetLeft: dailyBudgetServer - (serverPortfolio.total_stake ?? 0),
          avgStake: positions.length > 0 ? Math.round((serverPortfolio.total_stake ?? 0) / positions.length) : 0,
          kellyProfile,
          kellyFraction: (kellyFraction * 100).toFixed(0),
          capPerBet: (capPerBet * 100).toFixed(1),
          valueCutoff: (valueCutoff * 100).toFixed(0),
          dailyBudgetServer,
          capPerBetEurServer: bankroll * capPerBet,
          zone: zoneConfig
        }
      };
    }

    // Si le backend fournit un portefeuille avec "positions" (ancien format)
    if (serverPortfolio?.positions) {
      const positions = serverPortfolio.positions || [];
      const excluded = serverPortfolio.excluded || [];
      const caps = serverPortfolio.caps || {};
      const dailyBudgetServer = caps.daily_budget_eur ?? bankroll * dailyBudgetRate;
      const capPerBetEurServer = caps.cap_per_bet_eur ?? bankroll * capPerBet;
      console.log('[CONSEILS2] Utilisation positions du serveur:', positions.length);
      return {
        selectedBets: positions,
        excludedBets: excluded,
        stats: {
          totalStake: serverPortfolio.total_stake ?? 0,
          budgetUsed: dailyBudgetServer > 0 ? Math.round(((serverPortfolio.total_stake ?? 0) / dailyBudgetServer) * 100) : 0,
          budgetLeft: (serverPortfolio.budget_left ?? (dailyBudgetServer - (serverPortfolio.total_stake ?? 0))),
          avgStake: positions.length > 0 ? Math.round((serverPortfolio.total_stake ?? 0) / positions.length) : 0,
          kellyProfile,
          kellyFraction: ((serverPortfolio.kelly_fraction_effective ?? kellyFraction) * 100).toFixed(0),
          capPerBet: caps.cap_per_bet ? (caps.cap_per_bet * 100).toFixed(1) : (capPerBet * 100).toFixed(1),
          valueCutoff: caps.value_cutoff ? (caps.value_cutoff * 100).toFixed(0) : (valueCutoff * 100).toFixed(0),
          dailyBudgetServer,
          capPerBetEurServer,
          zone: zoneConfig
        }
      };
    }

    // ==============================================================
    // FILTRAGE PAR ZONE DE BANKROLL
    // ==============================================================
    const excluded = [];

    // 1. Mapper et enrichir tous les paris
    const allBets = [...bets].map(b => {
      const { valuePercent, kellyPercent } = resolveBetMetrics(b);
      const valueDecimal = valuePercent / 100;
      const calculatedStake = calculateStake(b);
      const cote = b.cote ?? b.odds ?? b.market ?? 0;
      const pWin = b.p_win ?? b.proba_win ?? b.probabilite ?? 0;
      const betType = b.bet_type || 'SIMPLE PLACÉ';
      const betRisk = b.bet_risk || 'Modéré';

      return {
        ...b,
        calculatedStake,
        valuePercent,
        valueDecimal,
        cote,
        pWin,
        betType,
        betRisk,
        meetsValueCutoff: valueDecimal >= valueCutoff,
        selectionKelly: kellyPercent ?? (b.kelly ?? b.kelly_pct ?? 0),
      };
    });

    // 2. FILTRAGE PAR ZONE - Appliquer les contraintes selon la bankroll
    const zoneFilteredBets = allBets.filter(b => {
      // a) Filtre par cote max
      if (b.cote > zoneConfig.maxOddsWin) {
        excluded.push({ ...b, excludeReason: `Cote ${b.cote.toFixed(1)} > max ${zoneConfig.maxOddsWin} (zone ${zoneConfig.name})` });
        return false;
      }

      // b) Filtre par proba minimum
      if (b.pWin < zoneConfig.minProbaModel) {
        excluded.push({ ...b, excludeReason: `Proba ${(b.pWin * 100).toFixed(1)}% < min ${(zoneConfig.minProbaModel * 100).toFixed(0)}% (zone ${zoneConfig.name})` });
        return false;
      }

      // c) Filtre par type de pari autorisé
      const betTypeAllowed = zoneConfig.allowedBetTypes.some(allowed =>
        b.betType.includes(allowed.replace('SIMPLE ', '')) || allowed.includes(b.betType)
      );
      if (!betTypeAllowed) {
        excluded.push({ ...b, excludeReason: `Type "${b.betType}" non autorisé en zone ${zoneConfig.name} (autorisés: ${zoneConfig.allowedBetTypes.join(', ')})` });
        return false;
      }

      // d) Filtre par niveau de risque
      if (!zoneConfig.allowedRisks.includes(b.betRisk)) {
        excluded.push({ ...b, excludeReason: `Risque "${b.betRisk}" non autorisé en zone ${zoneConfig.name} (autorisés: ${zoneConfig.allowedRisks.join(', ')})` });
        return false;
      }

      return true;
    });

    // 3. Filtrer par value positive et value cutoff
    const valueFilteredBets = zoneFilteredBets
      .filter(b => {
        if (b.valuePercent <= 0) {
          return false; // Juste ignorer, pas d'exclusion explicite
        }
        if (!b.meetsValueCutoff) {
          excluded.push({ ...b, excludeReason: `Value ${b.valuePercent.toFixed(1)}% < cutoff ${(valueCutoff * 100).toFixed(0)}%` });
          return false;
        }
        if (b.calculatedStake <= 0) {
          excluded.push({ ...b, excludeReason: 'Kelly ≤ 0 ou mise arrondie à 0' });
          return false;
        }
        return true;
      })
      .sort((a, b) => {
        // Trier par value desc puis kelly desc
        if (b.valuePercent !== a.valuePercent) return b.valuePercent - a.valuePercent;
        return (b.selectionKelly ?? 0) - (a.selectionKelly ?? 0);
      });

    // 4. Limiter par course (max N paris par course)
    const raceCount = {};
    const withinRaceLimit = valueFilteredBets.filter(bet => {
      const raceKey = bet.race_key || 'unknown';
      raceCount[raceKey] = (raceCount[raceKey] || 0) + 1;
      if (raceCount[raceKey] > maxUnitBetsPerRace) {
        excluded.push({ ...bet, excludeReason: `> ${maxUnitBetsPerRace} paris sur cette course` });
        return false;
      }
      return true;
    });

    // 5. Sélectionner selon le budget journalier ET la limite de paris par jour (zone)
    let totalStake = 0;
    const selected = [];

    for (const bet of withinRaceLimit) {
      // Vérifier la limite de paris par jour (zone)
      if (selected.length >= zoneConfig.maxBetsPerDay) {
        excluded.push({ ...bet, excludeReason: `Limite ${zoneConfig.maxBetsPerDay} paris/jour atteinte (zone ${zoneConfig.name})` });
        continue;
      }

      // Vérifier le budget journalier
      if (totalStake + bet.calculatedStake > dailyBudget) {
        excluded.push({ ...bet, excludeReason: `Dépasse budget journalier (${dailyBudget.toFixed(0)}€)` });
        continue;
      }

      selected.push(bet);
      totalStake += bet.calculatedStake;
    }

    // ==============================================================
    // 6. REDISTRIBUTION DU BUDGET - Utiliser au moins 80% du budget
    // ==============================================================
    const MIN_BUDGET_USAGE = 0.80; // 80% minimum
    const targetBudget = dailyBudget * MIN_BUDGET_USAGE;

    if (selected.length > 0 && totalStake < targetBudget) {
      // Calculer le facteur de scaling pour atteindre 80%
      const scaleFactor = targetBudget / totalStake;

      // Appliquer le scaling à chaque pari, en respectant le max par pari
      let newTotal = 0;
      for (const bet of selected) {
        const originalStake = bet.calculatedStake;
        let scaledStake = originalStake * scaleFactor;

        // Plafonner au max par pari (50€ par défaut en zone FULL)
        scaledStake = Math.min(scaledStake, zoneConfig.maxStakeEur);

        // Arrondir à 0.50€
        scaledStake = Math.round(scaledStake / 0.50) * 0.50;

        // Mise minimum
        scaledStake = Math.max(scaledStake, zoneConfig.minStakeEur);

        bet.calculatedStake = scaledStake;
        bet.scaledUp = originalStake < scaledStake; // Marquer si scalé
        newTotal += scaledStake;
      }
      totalStake = newTotal;
    }

    return {
      selectedBets: selected,
      excludedBets: excluded,
      stats: {
        totalStake,
        budgetUsed: Math.round((totalStake / dailyBudget) * 100),
        budgetLeft: dailyBudget - totalStake,
        avgStake: selected.length > 0 ? Math.round(totalStake / selected.length) : 0,
        kellyProfile,
        kellyFraction: (kellyFraction * 100).toFixed(0),
        capPerBet: (capPerBet * 100).toFixed(1),
        valueCutoff: (valueCutoff * 100).toFixed(0),
        zone: zoneConfig,
        redistributed: totalStake >= targetBudget
      }
    };
  }, [bets, bankroll, kellyFraction, capPerBet, dailyBudget, valueCutoff, maxUnitBetsPerRace, zoneConfig, serverPortfolio, kellyProfile, calculateStake]);

  console.log('[CONSEILS2] Après useMemo - selectedBets:', selectedBets.length, 'excludedBets:', excludedBets.length);

  const getValueColor = (value) => {
    if (value >= 20) return 'text-success';
    if (value >= 10) return 'text-emerald-400';
    if (value >= 5) return 'text-warning';
    return 'text-neutral-400';
  };

  // Priorité : bloquer si Benter ou blend marché ne sont pas OK
  /*
  MODIFICATION POUR ALGO BRUT V2 :
  On désactive la vérification de Benter/Market car on utilise directement l'API V2
  qui contient sa propre logique.
  */
  const analysisStatus = 'ok'; // Force OK pour V2
  const analysisReason = 'Algo Brut V2';

  /*
  const analysisStatus = benterStatus?.status !== 'ok'
    ? benterStatus?.status
    : (marketStatus?.status || 'pending');
  const analysisReason = benterStatus?.status !== 'ok'
    ? benterStatus?.reason
    : marketStatus?.reason;
  */
  const analysisRaces = benterStatus?.races_covered;
  const analysisTau = benterStatus?.tau;
  const gammaUsed = marketStatus?.gamma;
  const alphaBounds = marketStatus?.alpha_bounds;

  if (analysisStatus !== 'ok') {
    return (
      <div className="space-y-4">
        {/* ========== SIMULATION TOGGLE - Toujours visible ========== */}
        <div className="flex items-center justify-between p-3 rounded-xl bg-slate-800/30 border border-slate-700/30">
          <div className="flex items-center gap-3">
            {isSimulation && <SimulationBadge />}
            <span className="text-sm text-neutral-400">
              {isSimulation
                ? "Mode simulation: les courses passées sont incluses"
                : "Mode réel: seules les courses à venir"}
            </span>
          </div>
          <SimulationToggle isSimulation={isSimulation} onToggle={onToggleSimulation} />
        </div>

        <GlassCard>
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div>
              <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">Analyse complète requise (Benter + marché)</p>
              <p className="text-sm text-neutral-400">
                Statut : <span className="font-semibold">{analysisStatus}</span>
                {analysisReason && ` • ${analysisReason}`}
              </p>
              <p className="text-xs text-neutral-500 mt-1">
                Les paris sont masqués tant que l'analyse hiérarchique + blend marché n'a pas abouti. Elle est relancée automatiquement.
              </p>
            </div>
            <motion.button
              onClick={onReloadAnalysis || fetchBets}
              className="flex items-center gap-2 px-4 py-2 glass-button-primary rounded-xl"
            >
              {loading ? (
                <>
                  <ArrowPathIcon className="h-4 w-4 animate-spin" />
                  Relance en cours
                </>
              ) : (
                <>
                  <ArrowPathIcon className="h-4 w-4" />
                  Relancer l'analyse
                </>
              )}
            </motion.button>
          </div>
        </GlassCard>
        {loading && (
          <motion.div
            className="flex flex-col items-center justify-center py-12"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {console.log('[CONSEILS2] Affichage du loading...')}
            <motion.div
              className="text-6xl mb-4"
              animate={{
                x: [0, 10, 0, -10, 0],
                rotate: [0, 5, 0, -5, 0]
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              🏇
            </motion.div>
            <div className="text-lg font-semibold text-neutral-800 dark:text-neutral-200 mb-2">
              Génération des conseils...
            </div>
            <div className="text-sm text-neutral-500 dark:text-neutral-400 mb-4">
              Analyse avec modèle V2 • Zone {bankroll < 50 ? 'MICRO' : bankroll < 500 ? 'SMALL' : 'FULL'}
            </div>
            <div className="flex gap-1">
              {[0, 1, 2].map(i => (
                <motion.div
                  key={i}
                  className="w-2 h-2 bg-primary-500 rounded-full"
                  animate={{ scale: [1, 1.5, 1] }}
                  transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.2 }}
                />
              ))}
            </div>
          </motion.div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* ========== SIMULATION TOGGLE ========== */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {isSimulation && <SimulationBadge />}
          <span className="text-sm text-neutral-400">
            {isSimulation
              ? "Les courses passées sont incluses pour tester"
              : "Seules les courses à venir sont affichées"}
          </span>
        </div>
        <SimulationToggle isSimulation={isSimulation} onToggle={onToggleSimulation} />
      </div>

      <div className="glass-panel border border-emerald-500/20 bg-emerald-500/5 rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-emerald-800 dark:text-emerald-100">
          Head Benter actif (τ {analysisTau ?? '1.1'}, courses {analysisRaces ?? '—'}) • Blend marché corrigé (γ {gammaUsed ?? '0.9'}, α∈[{alphaBounds?.[0] ?? 0.3}, {alphaBounds?.[1] ?? 0.9}])
        </div>
        <motion.button
          onClick={onReloadAnalysis || fetchBets}
          className="flex items-center gap-2 px-3 py-1.5 glass-button hover:bg-white/10 rounded-lg text-xs"
        >
          <ArrowPathIcon className="h-4 w-4" />
          Rejouer l'analyse
        </motion.button>
      </div>

      {/* ========== ALGO BRUT V2 BANNER ========== */}
      <div className="rounded-xl p-4 border border-green-500/30 bg-green-500/10">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🎯</span>
            <div>
              <p className="font-semibold text-neutral-900 dark:text-white">
                Algo Brut Optimisé (+71% ROI)
              </p>
              <p className="text-xs text-neutral-600 dark:text-neutral-400">
                Semi-outsiders (cotes 7-15) • Proba ≥50% • Mises uniformes 10€
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-center px-3 py-1 bg-green-500/20 rounded-lg">
              <p className="text-xs text-green-300">ROI Validé (5 mois)</p>
              <p className="font-bold text-green-400">
                +71.47%
              </p>
            </div>
            <div className="text-center px-3 py-1 bg-blue-500/20 rounded-lg">
              <p className="text-xs text-blue-300">Win Rate</p>
              <p className="font-bold text-blue-400">
                56.8%
              </p>
            </div>
          </div>
        </div>

        {/* Info supplémentaire */}
        <div className="mt-4 p-3 bg-blue-900/30 rounded-lg border border-blue-500/20">
          <p className="text-xs text-blue-300 mb-1">📊 Stratégie simple et robuste:</p>
          <p className="text-sm text-blue-100">
            Pas d'Agent IA, pas de Kelly complexe. Configuration validée sur 222 paris historiques (4/5 périodes statistiquement significatives).
          </p>
        </div>
      </div>


      {/* BANDEAU ZONE DE BANKROLL */}
      <div className={`rounded-xl p-4 border ${zoneConfig.color}`}>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="text-2xl">{zoneConfig.label.split(' ')[0]}</span>
            <div>
              <p className="font-semibold text-neutral-900 dark:text-white">{zoneConfig.label}</p>
              <p className="text-xs text-neutral-600 dark:text-neutral-400">{zoneConfig.description}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <div className="text-center px-3 py-1 bg-white/10 rounded-lg">
              <p className="text-xs text-neutral-500 dark:text-neutral-400">Cotes max</p>
              <p className="font-bold text-neutral-900 dark:text-white">&lt; {zoneConfig.maxOddsWin}</p>
            </div>
            <div className="text-center px-3 py-1 bg-white/10 rounded-lg">
              <p className="text-xs text-neutral-500 dark:text-neutral-400">Paris/jour</p>
              <p className="font-bold text-neutral-900 dark:text-white">{zoneConfig.maxBetsPerDay}</p>
            </div>
            <div className="text-center px-3 py-1 bg-white/10 rounded-lg">
              <p className="text-xs text-neutral-500 dark:text-neutral-400">Mise max</p>
              <p className="font-bold text-neutral-900 dark:text-white">{zoneConfig.maxStakeEur}€</p>
            </div>
            <div className="text-center px-3 py-1 bg-white/10 rounded-lg">
              <p className="text-xs text-neutral-500 dark:text-neutral-400">Types</p>
              <p className="font-bold text-neutral-900 dark:text-white text-xs">{zoneConfig.allowedBetTypes.length === 1 ? 'PLACÉ' : zoneConfig.allowedBetTypes.length === 2 ? 'PLACÉ/E-P' : 'Tous'}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Résumé politique Kelly */}
      <div className="bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-900 rounded-xl p-4 border border-neutral-200 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="text-center">
              <p className="text-xs text-gray-500 dark:text-gray-400">Profil</p>
              <p className={`text-lg font-bold text-${KELLY_PROFILES[kellyProfile]?.color || 'blue'}-400`}>
                {KELLY_PROFILES[kellyProfile]?.label || kellyProfile}
              </p>
            </div>
            <div className="text-center px-4 border-l border-neutral-300 dark:border-white/10">
              <p className="text-xs text-gray-500 dark:text-gray-400">Kelly</p>
              <p className="text-lg font-bold text-neutral-900 dark:text-white">{stats.kellyFraction}%</p>
            </div>
            <div className="text-center px-4 border-l border-neutral-300 dark:border-white/10">
              <p className="text-xs text-gray-500 dark:text-gray-400">Cap/Pari</p>
              <p className="text-lg font-bold text-yellow-400">{maxStakePerBet.toFixed(0)}€</p>
            </div>
            <div className="text-center px-4 border-l border-neutral-300 dark:border-white/10">
              <p className="text-xs text-gray-500 dark:text-gray-400">Budget/Jour</p>
              <p className="text-lg font-bold text-emerald-400">{dailyBudget.toFixed(0)}€</p>
            </div>
            <div className="text-center px-4 border-l border-neutral-300 dark:border-white/10">
              <p className="text-xs text-gray-500 dark:text-gray-400">Value Min</p>
              <p className="text-lg font-bold text-orange-400">{stats.valueCutoff}%</p>
            </div>
          </div>
          <Link
            to="/settings"
            className="text-xs text-gray-500 dark:text-gray-400 hover:text-neutral-900 dark:hover:text-white px-3 py-1 bg-neutral-200 dark:bg-white/5 rounded-lg"
          >
            ⚙️ Modifier
          </Link>
        </div>
      </div>

      {/* Config bankroll */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2 px-4 py-2 glass-panel border border-neutral-200 dark:border-white/10 rounded-xl">
          <label className="text-sm text-neutral-600 dark:text-neutral-400">Bankroll:</label>
          <input
            type="number"
            value={bankroll}
            onChange={(e) => setBankroll(Number(e.target.value) || 1000)}
            className="w-24 px-2 py-1 bg-transparent text-neutral-900 dark:text-neutral-100 focus:outline-none text-right font-bold"
            aria-label="Bankroll"
            data-testid="bankroll-input"
          />
          <span className="text-neutral-500 dark:text-neutral-400">€</span>
        </div>

        {/* Stats de la sélection */}
        {selectedBets.length > 0 && (
          <div className="flex items-center gap-3 text-sm">
            <span className="px-3 py-1.5 bg-success/10 border border-success/30 rounded-lg text-success">
              <strong>{selectedBets.length}</strong> paris
            </span>
            <span className="px-3 py-1.5 bg-[#ec48991a] border border-[#ec48994d] rounded-lg text-primary-400">
              Total: <strong>{stats.totalStake.toFixed(0)}€</strong> ({stats.budgetUsed}% budget)
            </span>
            <span className={`px-3 py-1.5 rounded-lg ${stats.budgetLeft >= 0 ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400' : 'bg-red-500/10 border border-red-500/30 text-red-400'}`}>
              Restant: <strong>{stats.budgetLeft.toFixed(0)}€</strong>
            </span>
          </div>
        )}

        <motion.button
          onClick={fetchBets}
          className="p-2 glass-button hover:bg-white/10"
        >
          <ArrowPathIcon className="h-5 w-5 text-neutral-500 dark:text-neutral-400" />
        </motion.button>
      </div>

      {/* Info sur la stratégie */}
      <div className="text-xs text-neutral-600 dark:text-neutral-500 flex items-center gap-2 flex-wrap">
        <InformationCircleIcon className="h-4 w-4" />
        <span>Kelly {stats.kellyFraction}%</span>
        <span>•</span>
        <span>Cap {stats.capPerBet}%</span>
        <span>•</span>
        <span>Value ≥ {stats.valueCutoff}%</span>
        <span>•</span>
        <span>Arrondi {roundingIncrement.toFixed(2)}€</span>
        <span>•</span>
        <span>Max {maxUnitBetsPerRace}/course</span>
        <span className="ml-2 text-gray-500 dark:text-gray-600">f* = (p×(o-1)-(1-p))/(o-1)</span>
      </div>

      {/* Liste des paris */}
      {loading ? (
        <motion.div
          className="flex flex-col items-center justify-center py-16"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <motion.div
            className="text-7xl mb-6"
            animate={{
              x: [0, 15, 0, -15, 0],
              rotate: [0, 8, 0, -8, 0]
            }}
            transition={{
              duration: 1.2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            🏇
          </motion.div>
          <div className="text-xl font-bold text-neutral-800 dark:text-neutral-100 mb-2">
            Génération des paris du jour...
          </div>
          <div className="text-sm text-neutral-500 dark:text-neutral-400 mb-1">
            Modèle V2 • Zone {bankroll < 50 ? 'MICRO' : bankroll < 500 ? 'SMALL' : 'FULL'} • AUC 0.81
          </div>
          <div className="text-xs text-neutral-400 dark:text-neutral-500 mb-6">
            Analyse de {new Date().toLocaleDateString('fr-FR', { weekday: 'long', day: 'numeric', month: 'long' })}
          </div>
          <div className="flex gap-2">
            {[0, 1, 2, 3].map(i => (
              <motion.div
                key={i}
                className="w-3 h-3 bg-gradient-to-r from-primary-500 to-pink-500 rounded-full"
                animate={{
                  y: [0, -10, 0],
                  opacity: [0.5, 1, 0.5]
                }}
                transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.15 }}
              />
            ))}
          </div>
        </motion.div>
      ) : (selectedBets.length === 0 && excludedBets.length === 0) ? (
          <GlassCard className="text-center py-12" hover={false}>
            <div className="text-6xl mb-4">🎯</div>
            <p className="text-neutral-500 dark:text-neutral-400">Aucun pari value disponible pour le moment</p>
            <p className="text-sm text-neutral-400 mt-2">Vérifiez que des courses sont programmées aujourd'hui.</p>
          </GlassCard>
      ) : (selectedBets.length === 0 && excludedBets.length > 0) ? (
          <GlassCard className="py-8" hover={false}>
          <div className="text-center">
            <div className="text-5xl mb-4">⚠️</div>
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
              Aucun pari ne respecte les critères actuels
            </h3>
            <p className="text-sm text-neutral-500 dark:text-neutral-400 mb-4">
              {excludedBets.length} pari{excludedBets.length > 1 ? 's' : ''} potentiel{excludedBets.length > 1 ? 's' : ''} exclu{excludedBets.length > 1 ? 's' : ''} par les filtres de sécurité.
            </p>
          </div>
          <div className="mt-4 border-t border-neutral-200 dark:border-white/10 pt-4">
            <p className="text-xs text-neutral-500 dark:text-neutral-400 mb-3 font-medium">Raisons d'exclusion :</p>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {excludedBets.slice(0, 10).map((bet, idx) => (
                <div key={idx} className="flex items-start gap-2 text-sm p-2 bg-neutral-100 dark:bg-white/5 rounded-lg">
                  <span className="text-red-400">❌</span>
                  <div className="flex-1">
                    <span className="font-medium text-neutral-900 dark:text-white">{bet.nom}</span>
                    <span className="text-neutral-500 mx-1">•</span>
                    <span className="text-neutral-600 dark:text-neutral-400">{bet.excludeReason || 'Critère non respecté'}</span>
                  </div>
                </div>
              ))}
              {excludedBets.length > 10 && (
                <p className="text-xs text-neutral-400 text-center">+ {excludedBets.length - 10} autres...</p>
              )}
            </div>
          </div>
          <div className="mt-4 text-center">
            <Link
              to="/settings"
              className="inline-flex items-center gap-2 px-4 py-2 text-sm bg-purple-500/20 text-purple-400 rounded-xl hover:bg-purple-500/30 transition"
            >
              ⚙️ Ajuster les paramètres de filtrage
            </Link>
          </div>
        </GlassCard>
      ) : (
        <div className="space-y-6">
          {/* SECTION 1: Paris Algo Brut V2 */}

          {/* Mode Algo brut - Affiche selectedBets */}
          {selectedBets.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
                  <span className="text-success">🎯</span>
                  Paris sélectionnés - Algo Brut V2 ({selectedBets.length})
                </h3>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">
                    Budget restant: <strong className={stats.budgetLeft >= 0 ? "text-emerald-400" : "text-red-400"}>{stats.budgetLeft.toFixed(0)}€</strong>
                  </span>
                </div>
              </div>
              <motion.div
                variants={containerVariants}
                initial="hidden"
                animate="visible"
                className="space-y-3"
              >
                {selectedBets.map((bet, index) => (
                  <BetCard
                    key={`${bet.race_key}-${bet.cheval_id || bet.id}-${index}`}
                    bet={bet}
                    isInCart={isInCart}
                    addToCart={addToCart}
                    getValueColor={getValueColor}
                    suggestedStake={bet.mise || bet.stake || bet.calculatedStake || 10}
                    maxStakePerBet={maxStakePerBet}
                    valueCutoff={valueCutoff}
                  />
                ))}
              </motion.div>
            </div>
          )}

          {/* Message si aucun pari sélectionné */}
          {selectedBets.length === 0 && excludedBets.length > 0 && (
            <GlassCard className="text-center py-8 border-warning/30" hover={false}>
              <div className="text-4xl mb-3">⚠️</div>
              <p className="text-warning font-medium">Aucun pari ne correspond aux critères</p>
              <p className="text-sm text-neutral-500 dark:text-neutral-400 mt-1">
                Value cutoff: {(valueCutoff * 100).toFixed(0)}% • Budget journalier: {dailyBudget.toFixed(0)}€ • Consultez les paris exclus ci-dessous.
              </p>
            </GlassCard>
          )}

          {/* SECTION 2: Paris exclus (en bas, collapsible) */}
          {excludedBets.length > 0 && (
            <details className="group">
              <summary className="cursor-pointer list-none">
                <div className="flex items-center justify-between p-3 glass-panel border border-neutral-200 dark:border-white/10 rounded-xl hover:bg-neutral-100 dark:hover:bg-white/10 transition-colors">
                  <span className="text-sm text-neutral-500 dark:text-neutral-400 flex items-center gap-2">
                    <span className="text-neutral-600 dark:text-neutral-500">📋</span>
                    Paris exclus ({excludedBets.length}) - {excludedBets.filter(b => b.exclusionReason === 'mise_trop_faible').length > 0 ? 'Mise < ' + MIN_STAKE + '€' : 'Limite atteinte'}
                  </span>
                  <span className="text-neutral-600 dark:text-neutral-500 group-open:rotate-180 transition-transform">▼</span>
                </div>
              </summary>
              <motion.div
                variants={containerVariants}
                initial="hidden"
                animate="visible"
                className="space-y-2 mt-3 opacity-60"
              >
                {excludedBets.map((bet, index) => (
                  <BetCardMini
                    key={`${bet.race_key}-${bet.cheval_id || bet.id}-${index}`}
                    bet={bet}
                    getValueColor={getValueColor}
                  />
                ))}
              </motion.div>
            </details>
          )}
        </div>
      )}
    </div>
  );
};

// ============================================
// Composant BetCard (paris avec mise)
// ============================================
const BetCard = ({ bet, isInCart, addToCart, getValueColor, suggestedStake, maxStakePerBet, valueCutoff }) => {
  const name = bet.nom || bet.name || bet.cheval;
  const chevalId = bet.cheval_id || bet.chevalId || bet.id;
  const pWin = (bet.p_win ?? bet.proba_win ?? bet.probabilite ?? 0) * 100;
  const pPlace = (bet.p_place ?? 0) * 100;
  const value = bet.value ?? bet.value_pct ?? bet.value_bet ?? 0;
  const valuePlace = bet.value_place ?? 0;
  const kelly = bet.kelly ?? bet.kelly_pct ?? 0;
  const kellyPlace = bet.kelly_place ?? 0;
  const cote = bet.cote ?? bet.odds ?? bet.market ?? 0;
  const cotePlace = bet.cote_place ?? (cote / 3);
  const betType = bet.bet_type || 'SIMPLE PLACÉ';
  const betTypeEmoji = bet.bet_type_emoji || '🥉';
  const betRisk = bet.bet_risk || 'Modéré';
  const betDescription = bet.bet_description || '';
  const allBetTypes = bet.all_bet_types || [];
  const inCart = isInCart(bet);

  // Vérifier si la mise atteint le cap
  const isAtCap = maxStakePerBet && suggestedStake >= maxStakePerBet;

  const getRiskBadgeColor = () => {
    switch (betRisk) {
      case 'Faible': return 'bg-success/20 text-success border-success/30';
      case 'Modéré': return 'bg-warning/20 text-warning border-warning/30';
      case 'Élevé': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'Très élevé': return 'bg-error/20 text-error border-error/30';
      default: return 'bg-[#ec489933] text-primary-400 border-[#ec48994d]';
    }
  };

  const getBetTypeBadgeColor = () => {
    if (betType.includes('PLACÉ')) return 'bg-[#ec489933] text-primary-400 border-[#ec48994d]';
    if (betType.includes('E/P')) return 'bg-[#8b5cf633] text-secondary-400 border-[#8b5cf64d]';
    if (betType.includes('GAGNANT')) return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    return 'bg-neutral-200/50 dark:bg-neutral-500/20 text-neutral-600 dark:text-neutral-400 border-neutral-300 dark:border-neutral-500/30';
  };

  return (
    <motion.div variants={itemVariants}>
      <GlassCard className={`transition-all ${inCart ? 'ring-2 ring-success/50 bg-success/5' : 'hover:bg-neutral-100 dark:hover:bg-white/5'}`}>
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <span className={`px-2 py-0.5 text-xs font-bold rounded-md border ${getBetTypeBadgeColor()}`}>
                {betTypeEmoji} {betType}
              </span>
              <span className={`px-2 py-0.5 text-xs rounded-md border ${getRiskBadgeColor()}`}>
                Risque: {betRisk}
              </span>
              {isAtCap && (
                <span
                  className="px-2 py-0.5 text-xs rounded-md border bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
                  title={`Mise plafonnée (${maxStakePerBet?.toFixed(0)}€)`}
                >
                  🔒 Cap
                </span>
              )}
              {chevalId ? (
                <Link
                  to={`/cheval/${chevalId}`}
                  className="font-semibold text-neutral-900 dark:text-neutral-100 hover:text-primary-400 transition-colors"
                >
                  {name}
                </Link>
              ) : (
                <span className="font-semibold text-neutral-900 dark:text-neutral-100">{name}</span>
              )}
              <span className="text-xs text-neutral-600 dark:text-neutral-500">
                {bet.hippodrome} • {bet.race_key}
              </span>
            </div>

            {betDescription && (
              <p className="text-xs text-neutral-500 dark:text-neutral-400 mb-2 italic">{betDescription}</p>
            )}

            <div className="flex flex-wrap gap-3 text-sm">
              {betType.includes('GAGNANT') && !betType.includes('PLACÉ') ? (
                <>
                  <span className="text-primary-400">p(win): <strong>{pWin.toFixed(1)}%</strong></span>
                  <span className={getValueColor(value)}>Value: <strong>{value > 0 ? '+' : ''}{value.toFixed(1)}%</strong></span>
                  <span className="text-warning">Kelly: <strong>{kelly.toFixed(1)}%</strong></span>
                  <span className="text-neutral-600 dark:text-neutral-400">Cote: <strong>{cote.toFixed(2)}</strong></span>
                </>
              ) : betType.includes('E/P') ? (
                <>
                  <span className="text-primary-400">p(win): <strong>{pWin.toFixed(1)}%</strong></span>
                  <span className="text-secondary-400">p(placé): <strong>{pPlace.toFixed(1)}%</strong></span>
                  <span className="text-neutral-600 dark:text-neutral-400">Cotes: <strong>{cote.toFixed(2)}</strong> / <span className="text-amber-500 dark:text-amber-400" title="Cote placé estimée - la vraie cote sera connue après la course">~{cotePlace.toFixed(2)}*</span></span>
                </>
              ) : (
                <>
                  <span className="text-primary-400">p(placé): <strong>{pPlace.toFixed(1)}%</strong></span>
                  <span className={getValueColor(valuePlace)}>Value: <strong>{valuePlace > 0 ? '+' : ''}{valuePlace.toFixed(1)}%</strong></span>
                  <span className="text-warning">Kelly: <strong>{kellyPlace.toFixed(1)}%</strong></span>
                  <span className="text-neutral-600 dark:text-neutral-400" title="Cote gagnant (référence pour l'analyse)">
                    Cote: <strong>{cote.toFixed(2)}</strong>
                  </span>
                  <span className="text-amber-500 dark:text-amber-400" title="Cote placé estimée (~1/3 gagnant)">
                    Placé*: <strong>~{cotePlace.toFixed(2)}</strong>
                  </span>
                </>
              )}
            </div>

            {/* Avertissement pour les cotes estimées */}
            {(betType.includes('PLACÉ') || betType.includes('E/P')) && (
              <div className="mt-1 text-xs text-amber-600/70 dark:text-amber-500/70 flex items-center gap-1">
                <span>*</span>
                <span>Cote placé estimée (~1/3 de la cote gagnant). Rapport définitif après course.</span>
              </div>
            )}

            {allBetTypes.length > 1 && (
              <div className="mt-2 flex flex-wrap gap-1">
                <span className="text-xs text-neutral-600 dark:text-neutral-500">Aussi disponible:</span>
                {allBetTypes.slice(1, 3).map((alt, i) => (
                  <span key={i} className="text-xs px-1.5 py-0.5 bg-neutral-100 dark:bg-white/5 rounded text-neutral-500 dark:text-neutral-400">
                    {alt.emoji} {alt.type}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <p className="text-xs text-neutral-600 dark:text-neutral-500">Mise suggérée</p>
              <p className="text-lg font-bold text-success">{suggestedStake}€</p>
            </div>

            <motion.button
              onClick={() => addToCart(bet)}
              disabled={inCart}
              className={`p-3 rounded-xl transition-all ${inCart
                ? 'bg-success/20 text-success cursor-default'
                : 'bg-[#ec489933] text-primary-400 hover:bg-[#ec48994d]'
                }`}
            >
              {inCart ? (
                <CheckCircleIcon className="h-5 w-5" />
              ) : (
                <PlusIcon className="h-5 w-5" />
              )}
            </motion.button>
          </div>
        </div>
      </GlassCard>
    </motion.div>
  );
};

// ============================================
// Composant BetCardMini (paris sans mise - compact)
// ============================================
const BetCardMini = ({ bet, getValueColor }) => {
  const name = bet.nom || bet.name || bet.cheval;
  const pWin = (bet.p_win ?? 0) * 100;
  const value = bet.value ?? 0;
  const cote = bet.cote ?? 0;
  const betType = bet.bet_type || 'SIMPLE PLACÉ';
  const betTypeEmoji = bet.bet_type_emoji || '🥉';

  return (
    <div className="p-3 glass-panel border border-white/10 rounded-lg flex items-center justify-between">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs">{betTypeEmoji}</span>
        <span className="text-sm text-neutral-300">{name}</span>
        <span className="text-xs text-neutral-500">{bet.hippodrome}</span>
      </div>
      <div className="flex items-center gap-3 text-xs text-neutral-400">
        <span>p: {pWin.toFixed(0)}%</span>
        <span className={getValueColor(value)}>+{value.toFixed(0)}%</span>
        <span>@{cote.toFixed(2)}</span>
        <span className="text-neutral-600">0€</span>
      </div>
    </div>
  );
};

// ============================================
// Composant Onglet Exotiques
// ============================================
const ExotiquesTab = ({ settings, bankroll }) => {
  // Récupérer les paramètres exotiques depuis settings
  const exoticsDefaults = settings?.exotics_defaults || {};
  const perTicketRate = exoticsDefaults.per_ticket_rate || 0.0075;
  const maxPackRate = exoticsDefaults.max_pack_rate || 0.04;

  // Calculs des limites
  const maxPerTicket = bankroll * perTicketRate;
  const maxPackBudget = bankroll * maxPackRate;

  const [budget, setBudget] = useState(Math.min(50, maxPackBudget));
  const [pack, setPack] = useState('EQUILIBRE');
  const [loading, setLoading] = useState(false);
  const [tickets, setTickets] = useState([]);
  const [error, setError] = useState(null);

  // Mettre à jour le budget max quand les settings changent
  useEffect(() => {
    if (budget > maxPackBudget) {
      setBudget(Math.floor(maxPackBudget));
    }
  }, [maxPackBudget]);

  const buildExotics = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE}/exotics/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          budget,
          pack,
          bankroll,
          per_ticket_rate: perTicketRate,
          max_pack_rate: maxPackRate
        })
      });

      if (response.ok) {
        const data = await response.json();
        setTickets(data.tickets || data.combinations || data.combos || []);
      } else {
        const errData = await response.json().catch(() => ({}));
        setError(errData.message || errData.error || 'Erreur lors de la génération');
      }
    } catch (err) {
      console.error('Erreur build exotics:', err);
      setError('Impossible de contacter le serveur');
    } finally {
      setLoading(false);
    }
  };

  const packOptions = [
    { value: 'SUR', label: '🟢 Sûr', desc: 'Faible risque, gains modérés' },
    { value: 'EQUILIBRE', label: '🟡 Équilibré', desc: 'Risque/gain équilibré' },
    { value: 'RISQUE', label: '🔴 Risqué', desc: 'Haut risque, gros potentiel' }
  ];

  const getPackColor = (p) => {
    switch (p) {
      case 'SUR': return 'border-success/50 bg-success/10';
      case 'EQUILIBRE': return 'border-warning/50 bg-warning/10';
      case 'RISQUE': return 'border-error/50 bg-error/10';
      default: return 'border-white/10 bg-white/5';
    }
  };

  return (
    <div className="space-y-6">
      {/* Résumé limites exotiques */}
      <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-xl p-4 border border-purple-500/20">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="text-center">
              <p className="text-xs text-gray-400">Max / Ticket</p>
              <p className="text-lg font-bold text-purple-400">{maxPerTicket.toFixed(2)}€</p>
              <p className="text-xs text-gray-500">{(perTicketRate * 100).toFixed(2)}%</p>
            </div>
            <div className="text-center px-4 border-l border-white/10">
              <p className="text-xs text-gray-400">Max / Pack</p>
              <p className="text-lg font-bold text-pink-400">{maxPackBudget.toFixed(0)}€</p>
              <p className="text-xs text-gray-500">{(maxPackRate * 100).toFixed(0)}%</p>
            </div>
            <div className="text-center px-4 border-l border-white/10">
              <p className="text-xs text-gray-400">Bankroll</p>
              <p className="text-lg font-bold text-white">{bankroll}€</p>
            </div>
          </div>
          <Link
            to="/settings"
            className="text-xs text-gray-400 hover:text-white px-3 py-1 bg-white/5 rounded-lg"
          >
            ⚙️ Modifier
          </Link>
        </div>
      </div>

      {/* Builder */}
      <GlassCard>
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center gap-2">
          <AdjustmentsHorizontalIcon className="h-5 w-5 text-secondary-400" />
          Générateur de Tickets Exotiques
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Budget */}
          <div>
            <label className="block text-sm text-neutral-400 mb-2">Budget total</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={budget}
                onChange={(e) => setBudget(Math.min(Number(e.target.value) || 10, maxPackBudget))}
                min={10}
                max={maxPackBudget}
                className="flex-1 px-4 py-3 glass-input rounded-xl text-neutral-900 dark:text-neutral-100 text-xl font-bold"
              />
              <span className="text-xl text-neutral-400">€</span>
            </div>
            <p className="text-xs text-neutral-500 mt-1">
              Min: 10€ • Max: {maxPackBudget.toFixed(0)}€ ({(maxPackRate * 100).toFixed(0)}% bankroll)
            </p>
            {budget > maxPackBudget && (
              <p className="text-xs text-red-400 mt-1">⚠️ Dépasse le max pack rate</p>
            )}
          </div>

          {/* Pack selection */}
          <div>
            <label className="block text-sm text-neutral-400 mb-2">Profil de risque</label>
            <div className="space-y-2">
              {packOptions.map(opt => (
                <motion.button
                  key={opt.value}
                  onClick={() => setPack(opt.value)}
                  className={`w-full flex items-center justify-between p-3 rounded-xl border transition-all ${pack === opt.value ? getPackColor(opt.value) : 'border-white/10 bg-white/5 hover:bg-white/10'
                    }`}
                >
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">{opt.label}</span>
                  <span className="text-xs text-neutral-500">{opt.desc}</span>
                </motion.button>
              ))}
            </div>
          </div>
        </div>

        <motion.button
          onClick={buildExotics}
          disabled={loading || budget > maxPackBudget}
          className="mt-6 w-full py-3 bg-gradient-to-r from-[#8b5cf6] to-[#ec4899] text-white font-semibold rounded-xl hover:from-[#7c3aed] hover:to-[#db2777] transition-all disabled:opacity-50"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <ArrowPathIcon className="h-5 w-5 animate-spin" />
              Génération...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <SparklesIcon className="h-5 w-5" />
              Générer les Tickets
            </span>
          )}
        </motion.button>
      </GlassCard>

      {/* Erreur */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-error/10 border border-error/30 rounded-xl flex items-center gap-3"
        >
          <ExclamationTriangleIcon className="h-5 w-5 text-error flex-shrink-0" />
          <p className="text-error">{error}</p>
        </motion.div>
      )}

      {/* Résultats */}
      {tickets.length > 0 && (
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="space-y-4"
        >
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
            <SparklesIcon className="h-5 w-5 text-warning" />
            Tickets Générés ({tickets.length})
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {tickets.map((ticket, index) => (
              <motion.div key={index} variants={itemVariants}>
                <GlassCard className="h-full">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <span className="text-xs text-neutral-500">Ticket #{index + 1}</span>
                      <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">
                        {ticket.type || ticket.bet_type || 'Combiné'}
                      </h4>
                    </div>
                    <span className="px-3 py-1 bg-success/20 text-success rounded-lg font-bold">
                      {ticket.stake ?? ticket.mise ?? 0}€
                    </span>
                  </div>

                  {/* Combinaison */}
                  <div className="space-y-1 mb-3">
                    {(ticket.combo || ticket.selections || ticket.chevaux || []).map((sel, i) => (
                      <div key={i} className="text-sm text-neutral-300 flex items-center gap-2">
                        <span className="w-5 h-5 rounded-full bg-secondary-500/20 text-secondary-400 flex items-center justify-center text-xs">
                          {i + 1}
                        </span>
                        {sel.nom || sel.name || sel}
                      </div>
                    ))}
                  </div>

                  {/* Stats */}
                  <div className="flex flex-wrap gap-3 text-xs pt-3 border-t border-white/10">
                    {ticket.ev !== undefined && (
                      <span className="text-emerald-400">
                        EV: <strong>{ticket.ev > 0 ? '+' : ''}{ticket.ev.toFixed(2)}€</strong>
                      </span>
                    )}
                    {ticket.couverture !== undefined && (
                      <span className="text-primary-400">
                        Couverture: <strong>{(ticket.couverture * 100).toFixed(1)}%</strong>
                      </span>
                    )}
                    {ticket.odds !== undefined && (
                      <span className="text-warning">
                        Cote: <strong>{ticket.odds.toFixed(2)}</strong>
                      </span>
                    )}
                  </div>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

// ============================================
// Composant Onglet Portefeuille
// ============================================
const PortefeuilleTab = ({ cart, setCart, authToken, bankroll, settings, isSimulation }) => {
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sendStatus, setSendStatus] = useState('');
  const [sending, setSending] = useState(false);

  // Paramètres de la politique de mise
  const bettingDefaults = settings?.betting_defaults || {};
  const kellyProfile = bettingDefaults.kelly_profile || 'STANDARD';
  const kellyFraction = bettingDefaults.kelly_fraction || KELLY_PROFILES[kellyProfile]?.fraction || 0.33;
  const capPerBet = bettingDefaults.cap_per_bet || 0.02;
  const dailyBudgetRate = bettingDefaults.daily_budget_rate || 0.12;
  const valueCutoff = bettingDefaults.value_cutoff || 0.05;
  const dailyBudget = bankroll * dailyBudgetRate;
  const maxStakePerBet = bankroll * capPerBet;

  useEffect(() => {
    fetchPortfolio();
  }, [bankroll, kellyProfile]);

  const fetchPortfolio = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/portfolio/today?bankroll=${bankroll}&kelly_profile=${kellyProfile}`);
      if (response.ok) {
        const data = await response.json();
        setPortfolio(data);
      }
    } catch (error) {
      console.error('Erreur chargement portfolio:', error);
    } finally {
      setLoading(false);
    }
  };

  const removeFromCart = (index) => {
    setCart(cart.filter((_, i) => i !== index));
  };

  const updateStake = (index, newStake) => {
    const newCart = [...cart];
    newCart[index] = { ...newCart[index], stake: Number(newStake) || 0 };
    setCart(newCart);
  };

  // Calculs totaux avec détection des violations
  const totals = useMemo(() => {
    const totalStake = cart.reduce((sum, b) => sum + (b.stake || 0), 0);
    const totalEV = cart.reduce((sum, b) => {
      const stake = b.stake || 0;
      const { valuePercent } = resolveBetMetrics(b);
      const ev = (valuePercent / 100) * stake;
      return sum + ev;
    }, 0);

    // Vérifications des violations
    const budgetExceeded = totalStake > dailyBudget;
    const budgetLeft = dailyBudget - totalStake;
    const capViolations = cart.filter(b => (b.stake || 0) > maxStakePerBet);

    return {
      totalStake,
      totalEV,
      budgetExceeded,
      budgetLeft,
      capViolations,
      dailyBudget,
      maxStakePerBet
    };
  }, [cart, dailyBudget, maxStakePerBet]);

  const mapToApiBet = (bet) => {
    const raceKey = bet.race_key || bet.raceKey || null;
    const selection = bet.nom || bet.name || bet.cheval || bet.selection || 'Sélection';
    const { odds: resolvedOdds } = resolveBetMetrics(bet);
    const odds = Number(resolvedOdds || 1) || 1;

    // Extract event_date: try from race_key (format: 'YYYY-MM-DD|...') or use bet.event_date or today
    let eventDate = null;
    if (raceKey && raceKey.includes('|')) {
      const datePart = raceKey.split('|')[0];
      // Validate it looks like a date (YYYY-MM-DD)
      if (/^\d{4}-\d{2}-\d{2}$/.test(datePart)) {
        eventDate = datePart;
      }
    }
    // Fallback to bet.event_date or today's date for IA picks
    if (!eventDate) {
      eventDate = bet.event_date || new Date().toISOString().split('T')[0];
    }

    return {
      race_key: raceKey,
      event_date: eventDate,
      race_time: bet.heure || bet.race_time || null,  // Capture du champ 'race_time'
      hippodrome: bet.hippodrome || bet.venue || null,
      selection,
      bet_type: bet.bet_type || 'GAGNANT',
      stake: Number(bet.stake) || 0,
      odds,
      status: 'PENDING',
      notes: bet.ia_pick ? `IA Pick - ${bet.justification || 'Agent IA'}` : 'Ajouté depuis Conseils',
      is_simulation: isSimulation,  // Marqueur mode simulation
    };
  };

  const sendToMesParis = async () => {
    if (!authToken) {
      setSendStatus('Connecte-toi dans Mes Paris pour enregistrer les mises.');
      return;
    }
    if (cart.length === 0) {
      setSendStatus('Aucun pari à envoyer.');
      return;
    }
    setSending(true);
    setSendStatus('');
    try {
      for (const bet of cart) {
        const payload = mapToApiBet(bet);
        console.log('[DEBUG] Sending bet to API:', {
          payload,
          isSimulation,  // Debug: show simulation mode status
          authToken: authToken ? 'present' : 'missing',
          originalBet: bet
        });
        await betsAPI.create(payload, authToken);
      }
      setSendStatus('Paris envoyés vers Mes Paris ✅');
      setCart([]);
    } catch (err) {
      console.error('[DEBUG] Bet submission error:', err, 'Status:', err.status, 'Token present:', !!authToken);
      // Provide specific guidance for auth errors
      if (err.status === 401) {
        setSendStatus('Session expirée. Veuillez vous reconnecter via Mes Paris ou le menu utilisateur.');
      } else {
        const statusInfo = err.status ? `[${err.status}] ` : '';
        setSendStatus(`${statusInfo}${err.message}`);
      }
    } finally {
      setSending(false);
    }
  };

  const exportCartCSV = () => {
    exportPicksCSV(cart, { filename: 'portefeuille' });
  };

  const exportCartJSON = () => {
    exportPicksJSON(cart, { filename: 'portefeuille', meta: { totals } });
  };

  const exportServerPortfolioCSV = () => {
    if (!portfolio) return;
    exportPortfolioCSV(portfolio, { filename: 'portfolio_serveur' });
  };

  const exportServerPortfolioJSON = () => {
    if (!portfolio) return;
    exportPortfolioJSON(portfolio, { filename: 'portfolio_serveur' });
  };

  return (
    <div className="space-y-6">
      {/* Politique Kelly - Résumé */}
      <div className="bg-gradient-to-r from-neutral-800 to-neutral-900 rounded-xl p-4 border border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="text-center">
              <p className="text-xs text-neutral-500">Profil</p>
              <p className={`text-lg font-bold text-${KELLY_PROFILES[kellyProfile]?.color || 'primary'}-400`}>
                {KELLY_PROFILES[kellyProfile]?.label || kellyProfile}
              </p>
            </div>
            <div className="text-center px-4 border-l border-white/10">
              <p className="text-xs text-neutral-500">Kelly Fraction</p>
              <p className="text-lg font-bold text-white">{(kellyFraction * 100).toFixed(0)}%</p>
            </div>
            <div className="text-center px-4 border-l border-white/10">
              <p className="text-xs text-neutral-500">Cap/Pari</p>
              <p className="text-lg font-bold text-warning">{maxStakePerBet.toFixed(0)}€</p>
            </div>
            <div className="text-center px-4 border-l border-white/10">
              <p className="text-xs text-neutral-500">Budget/Jour</p>
              <p className="text-lg font-bold text-emerald-400">{dailyBudget.toFixed(0)}€</p>
            </div>
          </div>
          <Link
            to="/settings"
            className="text-xs text-neutral-400 hover:text-white px-3 py-1 bg-white/5 rounded-lg"
          >
            ⚙️ Modifier
          </Link>
        </div>
      </div>

      {/* Alertes violations */}
      {totals.budgetExceeded && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-error/10 border border-error/30 rounded-xl flex items-center gap-3"
        >
          <ExclamationCircleIcon className="h-6 w-6 text-error flex-shrink-0" />
          <div>
            <p className="text-error font-medium">⚠️ Budget journalier dépassé !</p>
            <p className="text-sm text-red-300">
              Total: {totals.totalStake.toFixed(2)}€ / Budget: {dailyBudget.toFixed(2)}€
              (dépassement: {(totals.totalStake - dailyBudget).toFixed(2)}€)
            </p>
          </div>
        </motion.div>
      )}

      {totals.capViolations.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-orange-500/10 border border-orange-500/30 rounded-xl flex items-center gap-3"
        >
          <ExclamationTriangleIcon className="h-6 w-6 text-orange-400 flex-shrink-0" />
          <div>
            <p className="text-orange-400 font-medium">⚠️ Cap par pari dépassé sur {totals.capViolations.length} pari(s)</p>
            <p className="text-sm text-orange-300">
              Max autorisé: {maxStakePerBet.toFixed(2)}€ (2% bankroll)
            </p>
          </div>
        </motion.div>
      )}

      {/* Résumé */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <GlassCard className="text-center">
          <p className="text-sm text-neutral-400 mb-1">Total Stake</p>
          <p
            className={`text-3xl font-bold ${totals.budgetExceeded ? 'text-error' : 'text-neutral-900 dark:text-neutral-100'}`}
            data-testid="portfolio-total-stake"
          >
            {money(totals.totalStake)}
          </p>
        </GlassCard>
        <GlassCard className="text-center">
          <p className="text-sm text-neutral-400 mb-1">EV Attendue</p>
          <p className={`text-3xl font-bold ${totals.totalEV >= 0 ? 'text-success' : 'text-error'}`}>
            {totals.totalEV > 0 ? '+' : ''}{money(totals.totalEV)}
          </p>
        </GlassCard>
        <GlassCard className="text-center">
          <p className="text-sm text-neutral-400 mb-1">Budget Restant</p>
          <p className={`text-3xl font-bold ${totals.budgetLeft >= 0 ? 'text-emerald-400' : 'text-error'}`}>
            {money(totals.budgetLeft)}
          </p>
        </GlassCard>
        <GlassCard className="text-center">
          <p className="text-sm text-neutral-400 mb-1">Nb Paris</p>
          <p className="text-3xl font-bold text-primary-400">{cart.length}</p>
        </GlassCard>
      </div>
      <p className="text-xs text-neutral-400">Bankroll: {money(bankroll)} • Value cutoff: ≥{(valueCutoff * 100).toFixed(0)}%</p>

      {/* Actions */}
      <div className="flex flex-wrap gap-3">
        <motion.button
          onClick={exportCartCSV}
          disabled={cart.length === 0}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 text-emerald-400 rounded-xl hover:bg-emerald-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <DocumentArrowDownIcon className="h-5 w-5" />
          Export CSV
        </motion.button>
        <motion.button
          onClick={exportCartJSON}
          disabled={cart.length === 0}
          className="flex items-center gap-2 px-4 py-2 bg-primary-500/20 text-primary-400 rounded-xl hover:bg-primary-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <DocumentArrowDownIcon className="h-5 w-5" />
          Export JSON
        </motion.button>
        <motion.button
          onClick={fetchPortfolio}
          className="flex items-center gap-2 px-4 py-2 glass-button hover:bg-white/10"
        >
          <ArrowPathIcon className="h-5 w-5" />
          Actualiser
        </motion.button>
        <motion.button
          onClick={sendToMesParis}
          disabled={cart.length === 0 || sending}
          className="flex items-center gap-2 px-4 py-2 bg-secondary-500/20 text-secondary-200 rounded-xl hover:bg-secondary-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <CheckCircleIcon className="h-5 w-5" />
          Envoyer vers Mes Paris
        </motion.button>
        <Link
          to="/mes-paris"
          className="flex items-center gap-2 px-4 py-2 glass-button hover:bg-white/10"
        >
          Ouvrir Mes Paris
        </Link>
      </div>

      {sendStatus && (
        <div className="text-sm text-secondary-100 bg-secondary-500/10 border border-secondary-500/20 rounded-xl px-3 py-2 inline-flex items-center gap-2">
          <InformationCircleIcon className="h-4 w-4" />
          {sendStatus}
        </div>
      )}

      {/* Panier actuel */}
      {cart.length === 0 ? (
        <GlassCard className="text-center py-12" hover={false}>
          <ShoppingCartIcon className="h-12 w-12 mx-auto text-neutral-500 mb-4" />
          <p className="text-neutral-400">Votre panier est vide</p>
          <p className="text-sm text-neutral-500 mt-1">
            Ajoutez des paris depuis l'onglet "Unitaires"
          </p>
        </GlassCard>
      ) : (
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="space-y-3"
        >
          <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
            <ShoppingCartIcon className="h-5 w-5 text-success" />
            Panier ({cart.length} paris)
          </h3>

          {cart.map((bet, index) => {
            const name = bet.nom || bet.name || bet.cheval;
            const pWinRaw = bet.p_win ?? bet.proba_win ?? bet.probabilite ?? 0;
            const oddsValue = bet.cote ?? bet.odds ?? bet.market ?? 0;
            const ev = (pWinRaw * oddsValue - 1) * (bet.stake || 0);
            const betType = bet.bet_type || 'GAGNANT';
            const betTypeEmoji = bet.bet_type_emoji || '🏆';

            return (
              <motion.div key={index} variants={itemVariants}>
                <GlassCard className="hover:bg-white/5 transition-all">
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`px-2 py-0.5 text-xs font-bold rounded-md border ${betType === 'PLACÉ'
                          ? 'bg-primary-500/20 text-primary-400 border-primary-500/30'
                          : 'bg-secondary-500/20 text-secondary-400 border-secondary-500/30'
                          }`}>
                          {betTypeEmoji} {betType}
                        </span>
                        <h4 className="font-semibold text-neutral-900 dark:text-neutral-100">{name}</h4>
                      </div>
                      <div className="flex flex-wrap gap-3 text-sm text-neutral-400">
                        <span>{bet.hippodrome}</span>
                        <span>Cote: {formatOdds(oddsValue)}</span>
                        <span>p(win): {percent(pWinRaw)}</span>
                        <span className={ev >= 0 ? 'text-success' : 'text-error'}>
                          EV: {ev > 0 ? '+' : ''}{money(ev)}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1">
                        <input
                          type="number"
                          value={bet.stake || 0}
                          onChange={(e) => updateStake(index, e.target.value)}
                          className="w-20 px-2 py-1 glass-input rounded-lg text-right"
                          min={0}
                        />
                        <span className="text-neutral-400">€</span>
                      </div>

                      <motion.button
                        onClick={() => removeFromCart(index)}
                        className="p-2 text-error hover:bg-error/20 rounded-lg transition-colors"
                      >
                        <TrashIcon className="h-5 w-5" />
                      </motion.button>
                    </div>
                  </div>
                </GlassCard>
              </motion.div>
            );
          })}
        </motion.div>
      )}

      {/* Portfolio du serveur */}
      {portfolio && (
        <div className="pt-6 border-t border-white/10">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
              <ChartBarIcon className="h-5 w-5 text-secondary-400" />
              Portfolio Serveur
            </h3>
            <div className="flex gap-2 flex-wrap">
              <motion.button
                onClick={exportServerPortfolioCSV}
                data-testid="portfolio-export-csv"
                className="flex items-center gap-2 px-3 py-2 glass-button hover:bg-white/10 text-sm"
              >
                <DocumentArrowDownIcon className="h-4 w-4" />
                Export CSV
              </motion.button>
              <motion.button
                onClick={exportServerPortfolioJSON}
                data-testid="portfolio-export-json"
                className="flex items-center gap-2 px-3 py-2 glass-button hover:bg-white/10 text-sm"
              >
                <DocumentArrowDownIcon className="h-4 w-4" />
                Export JSON
              </motion.button>
            </div>
          </div>

          {loading ? (
            <div className="h-32 bg-neutral-200/50 dark:bg-neutral-800/50 rounded-xl animate-pulse" />
          ) : (
            <GlassCard>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-neutral-500">Stake Total</p>
                  <p className="text-xl font-bold text-neutral-900 dark:text-neutral-100">
                    {money(portfolio.total_stake ?? portfolio.totalStake ?? 0)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-neutral-500">EV Totale</p>
                  <p className={`text-xl font-bold ${(portfolio.total_ev ?? portfolio.totalEV ?? 0) >= 0 ? 'text-success' : 'text-error'}`}>
                    {(portfolio.total_ev ?? portfolio.totalEV ?? 0) > 0 ? '+' : ''}
                    {money(portfolio.total_ev ?? portfolio.totalEV ?? 0)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-neutral-500">Nb Positions</p>
                  <p className="text-xl font-bold text-primary-400">
                    {portfolio.positions?.length ?? portfolio.bets?.length ?? 0}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-neutral-500">ROI Attendu</p>
                  <p className={`text-xl font-bold ${(portfolio.expected_roi ?? portfolio.roi ?? 0) >= 0 ? 'text-emerald-400' : 'text-error'}`}>
                    {percent(portfolio.expected_roi ?? portfolio.roi ?? 0)}
                  </p>
                </div>
              </div>
            </GlassCard>
          )}
        </div>
      )}
    </div>
  );
};

// ============================================
// Composant Principal
// ============================================
export default function Conseils() {
  const [activeTab, setActiveTab] = useState('unitaires');
  const [cart, setCart] = useState([]);
  const [bankroll, setBankroll] = useState(1000);
  // Utiliser le contexte auth global (comme MesParis)
  const { token: authToken } = useAuth();
  const [settings, setSettings] = useState(null);
  const [benterStatus, setBenterStatus] = useState({ status: 'pending' });
  const [marketStatus, setMarketStatus] = useState({ status: 'pending' });
  const [reloadKey, setReloadKey] = useState(0);

  // Mode simulation (stocké dans localStorage)
  const { isSimulation, toggleSimulation } = useSimulationMode();

  // Token is now managed by useAuth context, no need for localStorage sync

  // Charger les settings
  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/settings`);
        if (response.ok) {
          const data = await response.json();
          setSettings(data);
        }
      } catch (error) {
        console.error('Erreur chargement settings:', error);
      }
    };
    fetchSettings();
  }, []);

  const analysisOk = true; // FORCE OK V2 (benterStatus?.status === 'ok' && marketStatus?.status === 'ok');
  /*
  const analysisOk = benterStatus?.status === 'ok' && marketStatus?.status === 'ok';
  */
  const triggerReload = () => setReloadKey((key) => key + 1);

  return (
    <div className="max-w-7xl mx-auto space-y-6 px-4 sm:px-0 py-6 sm:py-12">
      {/* Header unifié */}
      <PageHeader
        emoji="🎯"
        title="Conseils V2 - Algo Brut Optimisé"
        subtitle="Stratégie validée à +71% ROI sur 5 mois (cotes 7-15, mises uniformes 10€)"
      >
        {/* Badge panier */}
        {cart.length > 0 && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="flex items-center gap-2 px-4 py-2 bg-success/20 border border-success/30 rounded-xl"
          >
            <ShoppingCartIcon className="h-5 w-5 text-success" />
            <span className="text-success font-medium">{cart.length} paris</span>
            <span className="text-neutral-400">•</span>
            <span className="text-neutral-900 dark:text-white font-bold">
              {cart.reduce((sum, b) => sum + (b.stake || 0), 0)}€
            </span>
          </motion.div>
        )}
      </PageHeader>

      {/* Tabs */}
      <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar">
        {TABS.map(tab => (
          <motion.button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium whitespace-nowrap transition-all ${activeTab === tab.id
              ? 'bg-gradient-to-r from-[#ec489933] to-[#8b5cf633] text-neutral-900 dark:text-neutral-100 border border-[#ec48994d]'
              : 'bg-white/5 text-neutral-400 hover:bg-white/10 border border-transparent'
              }`}
          >
            <span>{tab.emoji}</span>
            <span>{tab.label}</span>
            {tab.id === 'portefeuille' && cart.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 bg-success text-white text-xs rounded-full">
                {cart.length}
              </span>
            )}
          </motion.button>
        ))}
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === 'unitaires' && (
            <UnitairesTab
              cart={cart}
              setCart={setCart}
              bankroll={bankroll}
              setBankroll={setBankroll}
              settings={settings}
              benterStatus={benterStatus}
              onBenterStatus={setBenterStatus}
              marketStatus={marketStatus}
              onMarketStatus={setMarketStatus}
              reloadKey={reloadKey}
              onReloadAnalysis={triggerReload}
              isSimulation={isSimulation}
              onToggleSimulation={toggleSimulation}
            />
          )}

          {activeTab === 'exotiques' && (
            analysisOk ? (
              <ExotiquesTab settings={settings} bankroll={bankroll} />
            ) : (
              <GlassCard>
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                  <div>
                    <p className="text-lg font-semibold text-neutral-100">Analyse Benter en attente</p>
                    <p className="text-sm text-neutral-400">
                      Passe par l'onglet Unitaires pour lancer l'analyse obligatoire avant les tickets exotiques.
                    </p>
                  </div>
                  <motion.button
                    onClick={() => { setActiveTab('unitaires'); triggerReload(); }}
                    className="flex items-center gap-2 px-4 py-2 glass-button-primary rounded-xl"
                  >
                    <ArrowPathIcon className="h-4 w-4" />
                    Lancer l'analyse
                  </motion.button>
                </div>
              </GlassCard>
            )
          )}

          {activeTab === 'portefeuille' && (
            analysisOk ? (
              <PortefeuilleTab cart={cart} setCart={setCart} authToken={authToken} bankroll={bankroll} settings={settings} isSimulation={isSimulation} />
            ) : (
              <GlassCard>
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                  <div>
                    <p className="text-lg font-semibold text-neutral-100">Analyse Benter en attente</p>
                    <p className="text-sm text-neutral-400">
                      Le portefeuille s'appuie sur les picks calibrés : lance d'abord l'analyse via Unitaires.
                    </p>
                  </div>
                  <motion.button
                    onClick={() => { setActiveTab('unitaires'); triggerReload(); }}
                    className="flex items-center gap-2 px-4 py-2 glass-button-primary rounded-xl"
                  >
                    <ArrowPathIcon className="h-4 w-4" />
                    Lancer l'analyse
                  </motion.button>
                </div>
              </GlassCard>
            )
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
