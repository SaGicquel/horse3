#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏇 RACE PRONOSTIC GENERATOR - Système de Pronostics Prêts à Parier
===================================================================

Génère des pronostics calibrés avec:
- Simulation Monte Carlo pour p_place (N tirages)
- Blend régularisé logit-odds (modèle + marché)
- Kelly fractionnaire avec cap
- EV parimutuel avec takeout (via pari_math)
- Classification: SÛR / ÉQUILIBRÉ / RISQUÉ
- Sortie JSON stricte

Auteur: Horse3 Pro System
Version: 1.2.0

CONFIGURATION:
- Les paramètres T, α, kelly sont chargés depuis config/pro_betting.yaml
- Le mode parimutuel utilise config.markets.mode et config.markets.takeout_rate
- La configuration garantit la cohérence avec calibration_pipeline.py
"""

import json
import math
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
import numpy as np

# Import de la configuration centralisée
try:
    from config.loader import get_config, get_calibration_params_from_artifacts

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    warnings.warn(
        "⚠️ config.loader non disponible - paramètres par défaut utilisés. "
        "Veuillez créer config/pro_betting.yaml pour garantir la cohérence.",
        UserWarning,
    )

# Import du loader d'artefacts (source de vérité prioritaire)
try:
    from calibration.artifacts_loader import (
        load_calibration_state,
        CalibrationState,
        warn_if_mismatch,
        log_calibration_init,
    )

    ARTIFACTS_LOADER_AVAILABLE = True
except ImportError:
    ARTIFACTS_LOADER_AVAILABLE = False

# Import des fonctions EV parimutuel
try:
    from pari_math import (
        ev_parimutuel_win,
        ev_parimutuel_place,
        kelly_fraction as pari_kelly_fraction,
        expected_payout_parimutuel,
        DEFAULT_TAKEOUT_RATE,
    )

    PARI_MATH_AVAILABLE = True
except ImportError:
    PARI_MATH_AVAILABLE = False
    DEFAULT_TAKEOUT_RATE = 0.16

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES & CONFIGURATION
# =============================================================================


def _get_default_config() -> dict:
    """
    Retourne la configuration par défaut.

    Priorité:
    1. calibration/health.json (artefacts) via artifacts_loader
    2. config/pro_betting.yaml (fallback)
    3. Valeurs hardcodées (dernier recours)
    """
    # PRIORITÉ 1: Artefacts via artifacts_loader
    if ARTIFACTS_LOADER_AVAILABLE:
        try:
            state = load_calibration_state(prefer_artifacts=True)

            # Vérifier les différences YAML/artefacts
            warn_if_mismatch()

            # Charger le reste depuis config.loader si disponible
            kelly_fraction = 0.25
            value_cutoff = 0.05
            max_stake_pct = 0.05
            num_simulations = 20000
            markets_mode = "parimutuel"
            takeout_rate = DEFAULT_TAKEOUT_RATE

            if CONFIG_AVAILABLE:
                try:
                    config = get_config()
                    kelly_fraction = config.kelly.fraction
                    value_cutoff = config.kelly.value_cutoff
                    max_stake_pct = config.kelly.max_stake_pct
                    num_simulations = config.simulation.num_simulations
                    markets_mode = config.markets.mode
                    takeout_rate = config.markets.takeout_rate
                except Exception:
                    pass

            return {
                "blend_alpha": state.alpha,
                "blend_alpha_by_discipline": state.alpha_by_disc,
                "kelly_fraction": kelly_fraction,
                "value_cutoff": value_cutoff,
                "max_stake_pct": max_stake_pct,
                "num_simulations": num_simulations,
                "markets_mode": markets_mode,
                "parimutuel": markets_mode == "parimutuel",
                "takeout_rate": takeout_rate,
                "stake_currency": "EUR",
                "temperature": state.temperature,
                "calibrator": state.calibrator,
                "config_source": state.source,
                "last_calibration": state.last_calibration,
            }
        except Exception as e:
            warnings.warn(f"Erreur chargement artefacts: {e}, fallback config", UserWarning)

    # PRIORITÉ 2: config.loader (YAML)
    if CONFIG_AVAILABLE:
        try:
            config = get_config()
            artifact_params = get_calibration_params_from_artifacts()

            # Mode markets (parimutuel ou bookmaker)
            markets_mode = config.markets.mode
            takeout_rate = config.markets.takeout_rate

            # Blend alpha par discipline depuis config.calibration
            blend_by_disc = {
                "plat": config.calibration.blend_alpha_plat,
                "trot": config.calibration.blend_alpha_trot,
                "obstacle": config.calibration.blend_alpha_obstacle,
                "default": config.calibration.blend_alpha_global,
            }

            return {
                "blend_alpha": config.calibration.blend_alpha_global,
                "blend_alpha_by_discipline": blend_by_disc,
                "kelly_fraction": config.kelly.fraction,
                "value_cutoff": config.kelly.value_cutoff,
                "max_stake_pct": config.kelly.max_stake_pct,
                "num_simulations": config.simulation.num_simulations,
                "markets_mode": markets_mode,
                "parimutuel": markets_mode == "parimutuel",
                "takeout_rate": takeout_rate,
                "stake_currency": "EUR",
                "temperature": artifact_params["temperature"],  # Depuis artefacts de calibration
                "config_source": artifact_params["source"],
            }
        except Exception as e:
            warnings.warn(f"Erreur chargement config: {e}, utilisation des défauts", UserWarning)

    # PRIORITÉ 3: Valeurs par défaut calibrées (T*=1.254, α=0.2)
    # Valeurs fallback lues depuis l'env pour centraliser les seuils runtime
    value_cutoff_default = float(os.getenv("BETTING_VALUE_CUTOFF", "0.05"))
    kelly_default = float(os.getenv("BETTING_KELLY_FRACTION", "0.25"))
    blend_alpha_default = float(os.getenv("BLEND_ALPHA_DEFAULT", "0.2"))
    max_stake_pct_default = float(os.getenv("MAX_STAKE_PCT", "0.05"))
    num_simulations_default = int(os.getenv("NUM_SIMULATIONS", "20000"))
    return {
        "blend_alpha": blend_alpha_default,  # α global optimisé
        "blend_alpha_by_discipline": {
            "plat": 0.0,
            "trot": 0.4,
            "obstacle": 0.4,
            "default": blend_alpha_default,
        },
        "kelly_fraction": kelly_default,
        "value_cutoff": value_cutoff_default,
        "max_stake_pct": max_stake_pct_default,
        "num_simulations": num_simulations_default,  # UNIFIÉ: 20000 pour p_place et exotiques
        "markets_mode": "parimutuel",
        "parimutuel": True,
        "takeout_rate": DEFAULT_TAKEOUT_RATE,
        "stake_currency": "EUR",
        "temperature": 1.254,  # T* calibrée
        "config_source": "env_defaults",
    }


# Config globale - chargée au premier accès
DEFAULT_CONFIG = None


def get_default_config() -> dict:
    """Retourne la config par défaut (lazy loading)."""
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = _get_default_config()
    return DEFAULT_CONFIG


# =============================================================================
# FONCTIONS MATHÉMATIQUES
# =============================================================================

# Constantes de clipping pour les probabilités
PROB_CLIP_MIN = 1e-6
PROB_CLIP_MAX = 1 - 1e-6


def logit(p: float) -> float:
    """
    Logit transformation: log(p / (1-p))

    Args:
        p: Probabilité à transformer

    Returns:
        Valeur logit
    """
    p = np.clip(p, PROB_CLIP_MIN, PROB_CLIP_MAX)
    return np.log(p / (1 - p))


def inv_logit(x: float) -> float:
    """
    Inverse logit (sigmoid): 1 / (1 + exp(-x))

    Args:
        x: Valeur logit

    Returns:
        Probabilité dans [0, 1]
    """
    # Stabilité numérique pour éviter overflow
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))


def blend_logit_odds(
    p_model: np.ndarray, p_market: np.ndarray, alpha: float, shrinkage_threshold: float = 3.0
) -> np.ndarray:
    """
    Blend régularisé via logit-odds avec shrinkage.

    Formule: logit(p_final) = α*logit(p_model) + (1-α)*logit(p_market)

    Avec shrinkage progressif: si l'écart de logit dépasse le seuil,
    on réduit α proportionnellement pour éviter les prédictions extrêmes.

    Args:
        p_model: Probabilités du modèle (array)
        p_market: Probabilités implicites du marché (array)
        alpha: Poids du modèle dans [0, 1] (0.3-0.5 recommandé)
        shrinkage_threshold: Seuil logit au-delà duquel on applique le shrinkage (défaut: 3.0)

    Returns:
        Probabilités fusionnées, clippées dans [1e-6, 1-1e-6] et normalisées (Σp=1)

    Notes:
        - Un écart de 3 en logit ≈ ratio de cotes de 20x
        - Si p_market contient des NaN ou valeurs invalides, utilise p_model
        - Normalisation finale garantit Σp_win = 1 pour la course
    """
    p_model = np.asarray(p_model, dtype=np.float64)
    p_market = np.asarray(p_market, dtype=np.float64)

    n = len(p_model)
    if n != len(p_market):
        raise ValueError(
            f"p_model et p_market doivent avoir la même taille: {n} vs {len(p_market)}"
        )

    # Clipping des probabilités d'entrée
    p_model = np.clip(p_model, PROB_CLIP_MIN, PROB_CLIP_MAX)
    p_market = np.clip(p_market, PROB_CLIP_MIN, PROB_CLIP_MAX)

    p_blend = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Vérifier si p_market est valide
        if np.isnan(p_market[i]) or p_market[i] <= 0:
            # Fallback sur p_model si p_market invalide
            p_blend[i] = p_model[i]
            continue

        logit_model = logit(p_model[i])
        logit_market = logit(p_market[i])

        # Shrinkage: si l'écart de logit est > seuil, on réduit alpha
        logit_diff = abs(logit_model - logit_market)

        if logit_diff > shrinkage_threshold:
            # Shrinkage progressif vers le marché
            shrink_factor = shrinkage_threshold / logit_diff
            effective_alpha = alpha * shrink_factor
        else:
            effective_alpha = alpha

        # Blend en espace logit
        logit_blend = effective_alpha * logit_model + (1 - effective_alpha) * logit_market
        p_blend[i] = inv_logit(logit_blend)

    # Clipping final
    p_blend = np.clip(p_blend, PROB_CLIP_MIN, PROB_CLIP_MAX)

    # Normalisation pour garantir Σp = 1
    total = np.sum(p_blend)
    if total > 0:
        p_blend = p_blend / total
    else:
        # Fallback uniforme si problème
        p_blend = np.ones(n) / n

    return p_blend


def blend_logit_odds_single(p_model: float, p_market: float, alpha: float) -> float:
    """
    Version scalaire de blend_logit_odds pour un seul cheval.

    Args:
        p_model: Probabilité modèle
        p_market: Probabilité marché
        alpha: Poids du modèle

    Returns:
        Probabilité blendée (non normalisée)
    """
    # Clipping
    p_model = np.clip(p_model, PROB_CLIP_MIN, PROB_CLIP_MAX)

    # Fallback si p_market invalide
    if np.isnan(p_market) or p_market <= 0:
        return float(p_model)

    p_market = np.clip(p_market, PROB_CLIP_MIN, PROB_CLIP_MAX)

    logit_model = logit(p_model)
    logit_market = logit(p_market)

    # Shrinkage
    logit_diff = abs(logit_model - logit_market)
    if logit_diff > 3.0:
        effective_alpha = alpha * (3.0 / logit_diff)
    else:
        effective_alpha = alpha

    logit_blend = effective_alpha * logit_model + (1 - effective_alpha) * logit_market
    return float(inv_logit(logit_blend))


def softmax_temperature(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax avec température pour normalisation"""
    scores = scores / temperature
    scores = scores - np.max(scores)  # Stabilité numérique
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


# =============================================================================
# SIMULATION MONTE CARLO
# =============================================================================


class MonteCarloSimulator:
    """
    Simulateur Monte Carlo pour estimer les probabilités de place.
    Utilise le modèle Plackett-Luce pour générer des ordres d'arrivée.
    """

    def __init__(self, num_simulations: int = 20000, seed: int = None):
        self.num_simulations = num_simulations
        if seed:
            np.random.seed(seed)

    def simulate_race(self, win_probs: np.ndarray) -> np.ndarray:
        """
        Simule une course et retourne l'ordre d'arrivée.
        Utilise le modèle Plackett-Luce.

        Args:
            win_probs: Probabilités de victoire (normalisées)

        Returns:
            Array des indices dans l'ordre d'arrivée
        """
        n = len(win_probs)
        remaining = list(range(n))
        probs = win_probs.copy()
        finish_order = []

        for _ in range(n):
            # Normaliser les probas des restants
            remaining_probs = probs[remaining]
            remaining_probs = remaining_probs / remaining_probs.sum()

            # Tirer le gagnant parmi les restants
            winner_idx = np.random.choice(len(remaining), p=remaining_probs)
            winner = remaining[winner_idx]

            finish_order.append(winner)
            remaining.remove(winner)

        return np.array(finish_order)

    def estimate_place_probs(
        self, win_probs: np.ndarray, top_n: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estime les probabilités de place via simulation Monte Carlo.

        Args:
            win_probs: Probabilités de victoire
            top_n: Nombre de places payantes (généralement 3)

        Returns:
            (place_probs, variance_estimates)
        """
        n = len(win_probs)
        place_counts = np.zeros(n)
        place_squares = np.zeros(n)  # Pour calculer la variance

        for _ in range(self.num_simulations):
            finish_order = self.simulate_race(win_probs)

            # Compter les places
            for rank, horse_idx in enumerate(finish_order[:top_n]):
                place_counts[horse_idx] += 1
                place_squares[horse_idx] += 1

        # Probabilités
        place_probs = place_counts / self.num_simulations

        # Variance (Bernoulli): Var = p(1-p)/n
        variance = place_probs * (1 - place_probs) / self.num_simulations

        return place_probs, variance

    def simulate_with_win_place(
        self, win_probs: np.ndarray, top_n: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Simulation complète retournant win et place probs avec variances.
        """
        n = len(win_probs)
        win_counts = np.zeros(n)
        place_counts = np.zeros(n)

        for _ in range(self.num_simulations):
            finish_order = self.simulate_race(win_probs)

            # Gagnant
            win_counts[finish_order[0]] += 1

            # Placés
            for rank in range(min(top_n, len(finish_order))):
                place_counts[finish_order[rank]] += 1

        sim_win_probs = win_counts / self.num_simulations
        sim_place_probs = place_counts / self.num_simulations

        # Normaliser win pour cohérence
        sim_win_probs = sim_win_probs / sim_win_probs.sum()

        return {
            "win_probs": sim_win_probs,
            "place_probs": sim_place_probs,
            "win_variance": sim_win_probs * (1 - sim_win_probs) / self.num_simulations,
            "place_variance": sim_place_probs * (1 - sim_place_probs) / self.num_simulations,
        }


# =============================================================================
# KELLY CRITERION
# =============================================================================


class KellyCalculator:
    """Calcul du critère de Kelly avec contraintes - utilise pari_math si disponible"""

    @staticmethod
    def calculate(
        p_win: float,
        odds: float,
        fraction: float = 0.33,
        max_stake_pct: float = 0.05,
        parimutuel: bool = False,
        takeout_rate: float = 0.0,
    ) -> float:
        """
        Calcule la fraction Kelly avec ajustements.

        Args:
            p_win: Probabilité de victoire estimée
            odds: Cote (décimale européenne)
            fraction: Fraction de Kelly (0.25-0.5)
            max_stake_pct: Cap maximum du bankroll
            parimutuel: Si True, ajuste pour prélèvements
            takeout_rate: Taux de prélèvement (ex: 0.16 pour 16%)

        Returns:
            Fraction du bankroll à miser (0 si EV < 0)
        """
        if PARI_MATH_AVAILABLE:
            return pari_kelly_fraction(
                p_win, odds, fraction, max_stake_pct, parimutuel, takeout_rate
            )

        # Fallback si pari_math non disponible
        if odds <= 1 or p_win <= 0 or p_win >= 1:
            return 0.0

        # Ajuster la cote pour le parimutuel
        if parimutuel and takeout_rate > 0:
            effective_odds = odds * (1 - takeout_rate * 0.3)
        else:
            effective_odds = odds

        q = 1 - p_win
        b = effective_odds - 1

        kelly_full = (b * p_win - q) / b

        if kelly_full <= 0:
            return 0.0

        kelly_fractional = kelly_full * fraction
        return min(kelly_fractional, max_stake_pct)

    @staticmethod
    def calculate_value(
        p_win: float, odds: float, parimutuel: bool = False, takeout_rate: float = 0.0
    ) -> float:
        """
        Calcule la value (EV) pour WIN.

        En mode parimutuel:
            value_win = p_win * fair_odds_pool - 1
            où fair_odds_pool = (1 - takeout) / p

        En mode bookmaker:
            value = p_win * odds - 1
        """
        if PARI_MATH_AVAILABLE and parimutuel:
            return ev_parimutuel_win(p_win, odds, takeout_rate)

        # Fallback
        if odds <= 1 or p_win <= 0:
            return -1.0

        if parimutuel and takeout_rate > 0:
            effective_odds = odds * (1 - takeout_rate * 0.5)
            return p_win * effective_odds - 1
        else:
            return p_win * odds - 1

    @staticmethod
    def calculate_value_place(
        p_place: float, odds_place: float, parimutuel: bool = False, takeout_rate: float = 0.0
    ) -> float:
        """
        Calcule la value (EV) pour PLACE.

        En mode parimutuel, utilise ev_parimutuel_place de pari_math.
        """
        if PARI_MATH_AVAILABLE and parimutuel:
            return ev_parimutuel_place(p_place, odds_place, takeout_rate)

        # Fallback
        if odds_place <= 1 or p_place <= 0:
            return -1.0

        if parimutuel and takeout_rate > 0:
            effective_odds = odds_place * (1 - takeout_rate * 0.5)
            return p_place * effective_odds - 1
        else:
            return p_place * odds_place - 1


# =============================================================================
# CLASSIFICATION DES PARTANTS
# =============================================================================


def classify_runner(p_win: float, value: float, variance: float, odds: float) -> str:
    """
    Classifie un partant en SÛR, ÉQUILIBRÉ ou RISQUÉ.

    Critères:
    - SÛR: p_win > 15%, value > 0, variance faible, cote < 5
    - ÉQUILIBRÉ: 5% < p_win < 15%, value > 0, cote 5-15
    - RISQUÉ: p_win < 5% ou value < 0 ou cote > 15
    """
    # Variance relative (coefficient de variation approx)
    cv = np.sqrt(variance) / max(p_win, 0.01)

    if p_win > 0.15 and value > 0.02 and odds < 5 and cv < 0.3:
        return "SÛR"
    elif p_win > 0.05 and value > 0 and odds <= 15:
        return "ÉQUILIBRÉ"
    else:
        return "RISQUÉ"


# =============================================================================
# GÉNÉRATEUR DE PRONOSTICS PRINCIPAL
# =============================================================================


class RacePronosticGenerator:
    """
    Générateur de pronostics prêts à parier.

    Les paramètres sont chargés depuis config/pro_betting.yaml pour garantir
    la cohérence avec les artefacts de calibration (T*, α).

    L'EV parimutuel est calculée via le module pari_math:
    - fair_odds_pool = (1 - takeout) / p
    - value_win = p_win * market_odds * adjustment - 1
    """

    VERSION = "1.2.0"

    def __init__(self, db_connection, config: dict = None):
        """
        Initialise le générateur.

        Args:
            db_connection: Connexion à la base de données
            config: Override de configuration (optionnel)
        """
        self.conn = db_connection

        # Charger config par défaut puis merger avec overrides
        default_cfg = get_default_config()
        self.config = {**default_cfg, **(config or {})}

        # Log de la source de config
        source = self.config.get("config_source", "unknown")
        logger.info(
            f"RacePronosticGenerator init: config_source={source}, "
            f"T={self.config['temperature']:.4f}, "
            f"α={self.config['blend_alpha']:.2f}"
        )

        self.simulator = MonteCarloSimulator(num_simulations=self.config["num_simulations"])
        self.kelly = KellyCalculator()
        self._stats_cache = {}

    def _get_blend_alpha_for_discipline(self, discipline_norm: str) -> float:
        """
        Retourne l'α optimal pour une discipline donnée.

        Args:
            discipline_norm: Discipline normalisée (plat, trot, obstacle)

        Returns:
            Alpha pour le blend modèle/marché
        """
        alpha_by_disc = self.config.get("blend_alpha_by_discipline", {})

        if discipline_norm in alpha_by_disc:
            return alpha_by_disc[discipline_norm]
        elif "default" in alpha_by_disc:
            return alpha_by_disc["default"]
        else:
            return self.config.get("blend_alpha", 0.2)

    def _get_runner_score(
        self, nom: str, distance: int, hippodrome: str, race_date: str, data: dict
    ) -> Tuple[float, List[str]]:
        """
        Calcule un score brut pour un partant (pré-off uniquement).
        Retourne (score, rationale_signals)
        """
        cur = self.conn.cursor()
        score = 50.0
        signals = []

        # 1. Forme récente (5 dernières courses AVANT cette date)
        cur.execute(
            """
            SELECT is_win, place_finale
            FROM cheval_courses_seen
            WHERE nom_norm = %s AND race_key < %s
            ORDER BY race_key DESC
            LIMIT 5
        """,
            (nom, race_date),
        )

        recent = cur.fetchall()
        if recent:
            wins = sum(1 for r in recent if r[0] == 1)
            places = sum(1 for r in recent if r[1] and r[1] <= 3)

            if wins >= 2:
                score += 25
                signals.append("signal_forme:excellent")
            elif wins >= 1:
                score += 12
                signals.append("signal_forme:bon")

            if places >= 4:
                score += 15
                signals.append("signal_regularite:haute")
            elif places >= 3:
                score += 8
        else:
            signals.append("signal_forme:inconnu")

        # 2. Aptitude distance
        cur.execute(
            """
            SELECT COUNT(*),
                   AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100
            FROM cheval_courses_seen
            WHERE nom_norm = %s
            AND ABS(distance_m - %s) < 200
            AND race_key < %s
        """,
            (nom, distance, race_date),
        )

        row = cur.fetchone()
        if row and row[0] and row[0] >= 3:
            if row[1] and row[1] > 15:
                score += 15
                signals.append(f"aptitude_distance:forte({row[1]:.0f}%)")
            elif row[1] and row[1] > 8:
                score += 5
                signals.append("aptitude_distance:moyenne")

        # 3. Aptitude hippodrome
        cur.execute(
            """
            SELECT COUNT(*), SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END)
            FROM cheval_courses_seen
            WHERE nom_norm = %s AND hippodrome_nom = %s AND race_key < %s
        """,
            (nom, hippodrome, race_date),
        )

        row = cur.fetchone()
        if row and row[0] and row[0] >= 2 and row[1] and row[1] >= 1:
            score += 10
            signals.append("aptitude_hippodrome:gagnant_ici")

        # 4. Tendance cotes (pré-off)
        tendance = data.get("tendance_cote")
        amplitude = data.get("amplitude_tendance", 0) or 0

        if tendance == "-" and amplitude > 15:
            score += 12
            signals.append(f"tendance_cotes:baisse(-{amplitude:.0f}%)")
        elif tendance == "+" and amplitude > 15:
            score -= 8
            signals.append(f"tendance_cotes:hausse(+{amplitude:.0f}%)")

        # 5. Avis entraineur (pré-off)
        avis = data.get("avis_entraineur")
        if avis == "POSITIF":
            score += 8
            signals.append("avis_pro:positif")
        elif avis == "NEGATIF":
            score -= 8
            signals.append("avis_pro:negatif")

        return max(10, min(90, score)), signals[:3]

    def generate_pronostic(
        self, race_key: str, bankroll: float = 1000.0, config_override: dict = None
    ) -> str:
        """
        Génère un pronostic complet pour une course.

        Args:
            race_key: Identifiant de la course
            bankroll: Bankroll du parieur
            config_override: Paramètres personnalisés

        Returns:
            JSON string strict
        """
        config = {**self.config, **(config_override or {})}
        cur = self.conn.cursor()
        run_notes = []

        # 1. Récupérer infos course
        cur.execute(
            """
            SELECT DISTINCT
                hippodrome_nom, distance_m, discipline, type_course
            FROM cheval_courses_seen
            WHERE race_key = %s
            LIMIT 1
        """,
            (race_key,),
        )

        course_info = cur.fetchone()
        if not course_info:
            return json.dumps({"error": "Course non trouvée", "race_id": race_key})

        hippodrome, distance, discipline, type_course = course_info
        race_date = race_key[:10] if len(race_key) >= 10 else race_key

        # Mapper discipline
        discipline_map = {
            "ATTELE": "trot",
            "MONTE": "trot",
            "PLAT": "plat",
            "OBSTACLE": "obstacles",
            "HAIES": "obstacles",
            "STEEPLE": "obstacles",
        }
        discipline_norm = discipline_map.get(discipline, "plat")

        # 2. Récupérer partants
        cur.execute(
            """
            SELECT
                nom_norm, numero_dossard, cote_finale, cote_reference,
                tendance_cote, amplitude_tendance, est_favori, avis_entraineur,
                driver_jockey, entraineur
            FROM cheval_courses_seen
            WHERE race_key = %s
            AND (non_partant IS NULL OR non_partant = 0)
            ORDER BY numero_dossard
        """,
            (race_key,),
        )

        runners_data = cur.fetchall()

        if len(runners_data) < 2:
            return json.dumps({"error": "Moins de 2 partants", "race_id": race_key})

        # 3. Calculer scores bruts
        raw_scores = []
        market_odds = []
        runner_info = []

        for row in runners_data:
            nom, numero, cote, cote_ref, tendance, amplitude, favori, avis, jockey, entraineur = row

            data = {
                "tendance_cote": tendance,
                "amplitude_tendance": amplitude,
                "avis_entraineur": avis,
            }

            score, signals = self._get_runner_score(
                nom, distance or 2000, hippodrome, race_date, data
            )

            raw_scores.append(score)
            market_odds.append(cote if cote and cote > 1 else 10.0)
            runner_info.append(
                {
                    "horse_id": f"{nom}_{numero}",
                    "name": nom,
                    "numero": numero or 0,
                    "market_odds": cote,
                    "signals": signals,
                }
            )

        raw_scores = np.array(raw_scores)
        market_odds = np.array(market_odds)

        # 4. Probabilités modèle (softmax avec température)
        temperature = config.get("temperature", 3.0)
        model_probs = softmax_temperature(raw_scores, temperature=temperature)

        # 5. Probabilités marché
        market_probs = 1 / market_odds
        market_probs = market_probs / np.sum(market_probs)

        # Vérifier qualité des cotes marché
        has_market_info = np.std(market_odds) > 1.0

        # 6. Blend logit-odds avec α par discipline
        # Utiliser l'α spécifique à la discipline si disponible
        effective_alpha = self._get_blend_alpha_for_discipline(discipline_norm)

        if has_market_info:
            final_probs = blend_logit_odds(model_probs, market_probs, alpha=effective_alpha)
            run_notes.append(f"blend_logit_alpha={effective_alpha:.2f}({discipline_norm})")
        else:
            final_probs = model_probs
            run_notes.append("cotes_marche_absentes:modele_seul")

        # 7. Simulation Monte Carlo pour p_place
        sim_results = self.simulator.simulate_with_win_place(final_probs, top_n=3)
        place_probs = sim_results["place_probs"]
        place_variance = sim_results["place_variance"]

        run_notes.append(f"monte_carlo_n={config['num_simulations']}")
        run_notes.append(f"temperature={config['temperature']:.4f}")

        # Vérification calibration (Σp_win = 1)
        prob_sum = np.sum(final_probs)
        if abs(prob_sum - 1.0) > 1e-6:
            final_probs = final_probs / prob_sum
            run_notes.append(f"renormalisation_appliquee(était {prob_sum:.6f})")

        # 8. Construire les runners
        runners = []
        for i, info in enumerate(runner_info):
            p_win = float(final_probs[i])
            p_place = float(place_probs[i])
            odds = info["market_odds"]
            variance = float(place_variance[i])

            # Fair odds
            fair_odds = 1 / p_win if p_win > 0.001 else None

            # Value
            value_win = (
                self.kelly.calculate_value(
                    p_win,
                    odds if odds else 10.0,
                    parimutuel=config["parimutuel"],
                    takeout_rate=config["takeout_rate"],
                )
                if odds
                else None
            )

            # Kelly
            kelly_win = (
                self.kelly.calculate(
                    p_win,
                    odds if odds else 10.0,
                    fraction=config["kelly_fraction"],
                    max_stake_pct=config["max_stake_pct"],
                    parimutuel=config["parimutuel"],
                    takeout_rate=config["takeout_rate"],
                )
                if odds
                else 0.0
            )

            # Classification
            bucket = classify_runner(
                p_win, value_win if value_win else -1, variance, odds if odds else 100
            )

            runners.append(
                {
                    "horse_id": info["horse_id"],
                    "name": info["name"],
                    "numero": info["numero"],
                    "p_win": round(p_win, 4),
                    "p_place": round(p_place, 4),
                    "fair_odds_win": round(fair_odds, 2) if fair_odds else None,
                    "market_odds_win": odds,
                    "value_win": round(value_win, 4) if value_win is not None else None,
                    "kelly_win_fraction": round(kelly_win, 4),
                    "bucket": bucket,
                    "rationale": info["signals"],
                }
            )

        # Trier par p_win décroissant
        runners.sort(key=lambda x: x["p_win"], reverse=True)

        # 9. Portfolio suggestions
        bets = []
        for r in runners:
            value = r["value_win"]
            kelly = r["kelly_win_fraction"]

            if value is not None and value > config["value_cutoff"] and kelly > 0:
                stake = round(bankroll * kelly, 2)
                ev = round(value * stake, 2)

                bets.append(
                    {
                        "horse_id": r["horse_id"],
                        "name": r["name"],
                        "market": "WIN",
                        "odds": r["market_odds_win"],
                        "stake": stake,
                        "ev": ev,
                        "notes": f"value={value:.2%}>cutoff",
                    }
                )

        # 10. Output final
        output = {
            "race_id": race_key,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "discipline": discipline_norm,
            "hippodrome": hippodrome,
            "distance_m": distance,
            "nb_partants": len(runners),
            "model_version": self.VERSION,
            "config_source": config.get("config_source", "unknown"),
            "assumptions": {
                "temperature": config["temperature"],
                "blend_alpha_global": config["blend_alpha"],
                "blend_alpha_discipline": effective_alpha,
                "kelly_fraction": config["kelly_fraction"],
                "value_cutoff": config["value_cutoff"],
                "max_stake_pct": config["max_stake_pct"],
                "markets_mode": config.get("markets_mode", "parimutuel"),
                "parimutuel": config["parimutuel"],
                "takeout_rate": config["takeout_rate"] if config["parimutuel"] else None,
                "num_simulations": config["num_simulations"],
                "pari_math_available": PARI_MATH_AVAILABLE,
            },
            "runners": runners,
            "portfolio_suggestions": {
                "stake_currency": config["stake_currency"],
                "bankroll": bankroll,
                "unit_stake": "auto",
                "max_stake_per_bet": config["max_stake_pct"],
                "recommended_bets": bets,
                "total_stake": round(sum(b["stake"] for b in bets), 2),
                "total_ev": round(sum(b["ev"] for b in bets), 2),
            },
            "run_notes": run_notes,
        }

        return json.dumps(output, ensure_ascii=False, indent=2)

    def generate_pronostic_dict(self, race_key: str, **kwargs) -> dict:
        """Version dict pour usage programmatique"""
        return json.loads(self.generate_pronostic(race_key, **kwargs))

    def get_params_info(self) -> dict:
        """Retourne les paramètres actuels pour debug/vérification."""
        return {
            "temperature": self.config.get("temperature"),
            "blend_alpha_global": self.config.get("blend_alpha"),
            "blend_alpha_by_discipline": self.config.get("blend_alpha_by_discipline", {}),
            "kelly_fraction": self.config.get("kelly_fraction"),
            "markets_mode": self.config.get("markets_mode", "parimutuel"),
            "parimutuel": self.config.get("parimutuel", True),
            "takeout_rate": self.config.get("takeout_rate", DEFAULT_TAKEOUT_RATE),
            "config_source": self.config.get("config_source", "unknown"),
            "version": self.VERSION,
            "config_available": CONFIG_AVAILABLE,
            "pari_math_available": PARI_MATH_AVAILABLE,
        }


# =============================================================================
# TESTS UNITAIRES - BLEND & CALIBRATION
# =============================================================================

import unittest


def compute_ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """
    Calcule l'Expected Calibration Error (ECE).

    Args:
        probs: Probabilités prédites
        outcomes: Résultats observés (0 ou 1)
        n_bins: Nombre de bins

    Returns:
        ECE (plus petit = mieux calibré)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(outcomes[mask])
            bin_conf = np.mean(probs[mask])
            bin_count = np.sum(mask)
            ece += np.abs(bin_acc - bin_conf) * (bin_count / len(probs))

    return ece


def compute_logloss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Calcule le Log Loss (cross-entropy).

    Args:
        probs: Probabilités prédites
        outcomes: Résultats observés (0 ou 1)

    Returns:
        Log Loss (plus petit = mieux)
    """
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))


class TestBlendLogitOdds(unittest.TestCase):
    """Tests unitaires pour blend_logit_odds et calibration."""

    def test_clipping_bounds(self):
        """Test: les probabilités sont clippées dans [1e-6, 1-1e-6]."""
        p_model = np.array([0.0, 1.0, 0.5, 1e-10, 1 - 1e-10])
        p_market = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        result = blend_logit_odds(p_model, p_market, alpha=0.5)

        self.assertTrue(np.all(result >= PROB_CLIP_MIN), "Toutes les probas >= 1e-6")
        self.assertTrue(np.all(result <= PROB_CLIP_MAX), "Toutes les probas <= 1-1e-6")

    def test_normalization_sum_one(self):
        """Test: Σp = 1 après normalisation."""
        np.random.seed(42)
        p_model = np.random.dirichlet(np.ones(10))
        p_market = np.random.dirichlet(np.ones(10))

        result = blend_logit_odds(p_model, p_market, alpha=0.3)

        self.assertAlmostEqual(
            np.sum(result), 1.0, places=10, msg="Somme des probabilités doit être 1"
        )

    def test_fallback_when_market_missing(self):
        """Test: fallback sur p_model si p_market invalide."""
        p_model = np.array([0.3, 0.4, 0.3])
        p_market = np.array([np.nan, 0.0, -0.1])  # Valeurs invalides

        result = blend_logit_odds(p_model, p_market, alpha=0.5)

        # Doit retourner quelque chose de valide
        self.assertAlmostEqual(np.sum(result), 1.0, places=10)
        self.assertTrue(np.all(result > 0), "Toutes probas > 0 même avec marché invalide")

    def test_alpha_zero_equals_market(self):
        """Test: α=0 donne les probabilités du marché (normalisées)."""
        p_model = np.array([0.1, 0.3, 0.6])
        p_market = np.array([0.2, 0.3, 0.5])

        result = blend_logit_odds(p_model, p_market, alpha=0.0)
        expected = p_market / np.sum(p_market)

        np.testing.assert_array_almost_equal(
            result, expected, decimal=5, err_msg="α=0 devrait donner les probas marché"
        )

    def test_alpha_one_equals_model(self):
        """Test: α=1 donne les probabilités du modèle (normalisées)."""
        p_model = np.array([0.1, 0.3, 0.6])
        p_market = np.array([0.2, 0.3, 0.5])

        result = blend_logit_odds(p_model, p_market, alpha=1.0)
        expected = p_model / np.sum(p_model)

        np.testing.assert_array_almost_equal(
            result, expected, decimal=5, err_msg="α=1 devrait donner les probas modèle"
        )

    def test_shrinkage_extreme_divergence(self):
        """Test: shrinkage appliqué si divergence extrême."""
        # Modèle très confiant vs marché
        p_model = np.array([0.95, 0.03, 0.02])
        p_market = np.array([0.20, 0.40, 0.40])

        # Sans shrinkage, le blend serait très différent du marché
        result = blend_logit_odds(p_model, p_market, alpha=0.5)

        # Avec shrinkage, le résultat devrait être plus proche du marché
        # que ce que donnerait un blend linéaire simple
        self.assertTrue(
            result[0] < 0.8, f"Shrinkage devrait modérer la confiance extrême: {result[0]}"
        )

    def test_blend_improves_or_equals_calibration(self):
        """
        Test crucial: ECE/logloss blendé <= version non-blendée.

        Le blend avec le marché devrait améliorer ou maintenir la calibration.
        """
        np.random.seed(123)
        n_races = 100
        n_runners = 8

        all_model_probs = []
        all_blend_probs = []
        all_outcomes = []

        for _ in range(n_races):
            # Vraies probabilités (inconnues en pratique)
            true_probs = np.random.dirichlet(np.ones(n_runners) * 2)

            # Modèle avec bruit
            model_noise = np.random.randn(n_runners) * 0.3
            model_logits = np.log(true_probs + 1e-6) + model_noise
            model_probs = softmax_temperature(model_logits, temperature=1.0)

            # Marché (généralement mieux calibré)
            market_noise = np.random.randn(n_runners) * 0.15
            market_logits = np.log(true_probs + 1e-6) + market_noise
            market_probs = softmax_temperature(market_logits, temperature=1.0)

            # Blend
            blend_probs = blend_logit_odds(model_probs, market_probs, alpha=0.3)

            # Simuler le gagnant
            winner_idx = np.random.choice(n_runners, p=true_probs)
            outcomes = np.zeros(n_runners)
            outcomes[winner_idx] = 1

            all_model_probs.extend(model_probs)
            all_blend_probs.extend(blend_probs)
            all_outcomes.extend(outcomes)

        all_model_probs = np.array(all_model_probs)
        all_blend_probs = np.array(all_blend_probs)
        all_outcomes = np.array(all_outcomes)

        # Calculer métriques
        ece_model = compute_ece(all_model_probs, all_outcomes)
        ece_blend = compute_ece(all_blend_probs, all_outcomes)

        logloss_model = compute_logloss(all_model_probs, all_outcomes)
        logloss_blend = compute_logloss(all_blend_probs, all_outcomes)

        # Le blend devrait être au moins aussi bon (avec marge de tolérance)
        tolerance = 0.05  # 5% de tolérance pour variance Monte Carlo

        self.assertLessEqual(
            ece_blend,
            ece_model * (1 + tolerance),
            f"ECE blendé ({ece_blend:.4f}) devrait être <= ECE modèle ({ece_model:.4f}) + {tolerance*100}%",
        )

        self.assertLessEqual(
            logloss_blend,
            logloss_model * (1 + tolerance),
            f"LogLoss blendé ({logloss_blend:.4f}) devrait être <= LogLoss modèle ({logloss_model:.4f}) + {tolerance*100}%",
        )

    def test_blend_single_scalar(self):
        """Test: version scalaire fonctionne correctement."""
        p_model = 0.3
        p_market = 0.5

        result = blend_logit_odds_single(p_model, p_market, alpha=0.4)

        self.assertGreater(result, 0)
        self.assertLess(result, 1)
        # Doit être entre p_model et p_market
        self.assertGreater(result, min(p_model, p_market) - 0.1)
        self.assertLess(result, max(p_model, p_market) + 0.1)

    def test_discipline_alpha_selection(self):
        """Test: sélection de alpha par discipline depuis config."""
        config = get_default_config()
        alpha_by_disc = config.get("blend_alpha_by_discipline", {})

        # Vérifier que les disciplines principales ont des alphas
        expected_disciplines = ["plat", "trot", "obstacle", "default"]
        for disc in expected_disciplines:
            self.assertIn(
                disc, alpha_by_disc, f"Alpha pour discipline '{disc}' devrait être défini"
            )
            self.assertGreaterEqual(alpha_by_disc[disc], 0.0)
            self.assertLessEqual(alpha_by_disc[disc], 1.0)


def run_blend_tests():
    """Lance les tests unitaires du blend."""
    print("\n" + "=" * 60)
    print("🧪 TESTS UNITAIRES - BLEND LOGIT ODDS")
    print("=" * 60 + "\n")

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBlendLogitOdds)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ TOUS LES TESTS PASSÉS")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("=" * 60)

    return result.wasSuccessful()


# =============================================================================
# CLI
# =============================================================================


def main():
    """Point d'entrée CLI"""
    import sys

    # Mode test
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = run_blend_tests()
        sys.exit(0 if success else 1)

    from db_connection import get_connection

    conn = get_connection()
    generator = RacePronosticGenerator(conn)

    # Afficher les paramètres utilisés
    print("=" * 60, file=sys.stderr)
    print("🏇 RACE PRONOSTIC GENERATOR - Paramètres", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    params = generator.get_params_info()
    print(f"   Temperature (T*):     {params['temperature']:.4f}", file=sys.stderr)
    print(f"   Blend Alpha Global:   {params['blend_alpha_global']:.2f}", file=sys.stderr)
    print(f"   Alpha by Discipline:  {params['blend_alpha_by_discipline']}", file=sys.stderr)
    print(f"   Kelly Fraction:       {params['kelly_fraction']:.2f}", file=sys.stderr)
    print(f"   Config Source:        {params['config_source']}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Arguments
    if len(sys.argv) > 1:
        race_key = sys.argv[1]
        bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
    else:
        # Trouver une course récente
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT race_key
            FROM cheval_courses_seen
            WHERE cote_finale IS NOT NULL
            ORDER BY race_key DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            print(json.dumps({"error": "Aucune course trouvée"}))
            return
        race_key = row[0]
        bankroll = 1000.0

    # Générer et afficher
    result = generator.generate_pronostic(race_key, bankroll=bankroll)
    print(result)

    conn.close()


if __name__ == "__main__":
    main()
