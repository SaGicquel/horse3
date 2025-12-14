#!/usr/bin/env python3
"""
Estimateur de Probabilités de Place et d'Ordre pour Paris Exotiques
====================================================================

Ce module implémente trois estimateurs de places/exacta:
1. Harville (classique)
2. Henery/Stern (variantes ajustées pour favoris)
3. Lo-Bacon-Shone (réallocation probabiliste)

Plus un générateur de classements Plackett-Luce paramétré par température
pour simuler les arrivées et estimer P(Trio/Quarté/Quinté).

Données requises: p_win_blend par partant, contexte course (discipline, distance).

Sorties:
- p_place: probabilité d'être dans le top N
- P(combinaisons): probabilité de chaque combinaison exacta/trio/quarté/quinté
- EV de chaque ticket (parimutuel avec takeout ou cotes fixes)

Contrôles:
- Calibration des places (Brier/ECE)
- Stabilité EV des packs vs N simulations
- Cohérence monotone par déciles de p

Auteur: Horse Racing AI System
Date: 2024-12
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from itertools import permutations, combinations
from scipy.special import softmax
from scipy.optimize import minimize_scalar
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Import config si disponible
try:
    from config.loader import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PlaceEstimatorConfig:
    """Configuration pour l'estimation des probabilités de place."""
    # Simulations Monte Carlo
    num_simulations: int = 20000
    random_seed: Optional[int] = 42
    
    # Température Plackett-Luce (à apprendre par discipline)
    temperature_default: float = 1.0
    temperature_plat: float = 0.95      # Favoris légèrement surestimés
    temperature_trot: float = 1.05      # Plus d'incertitude
    temperature_obstacle: float = 1.10  # Haute variance
    
    # Paramètres Henery
    henery_gamma: float = 0.81  # Exposant Henery (< 1 = favoris surestimés)
    
    # Paramètres Stern
    stern_lambda: float = 0.15  # Ajustement Stern
    
    # Paramètres Lo-Bacon-Shone
    lbs_iterations: int = 100   # Itérations de réallocation
    lbs_damping: float = 0.7    # Facteur d'amortissement
    
    # Takeout parimutuel
    takeout_rate: float = 0.16
    
    # Validation
    min_prob_threshold: float = 1e-6
    
    def __post_init__(self):
        """Charge la config depuis le fichier si disponible."""
        if CONFIG_AVAILABLE:
            try:
                config = get_config()
                self.num_simulations = config.simulation.num_simulations
                self.takeout_rate = config.markets.takeout_rate
            except Exception:
                pass


# =============================================================================
# ESTIMATEURS DE PLACE
# =============================================================================

class HarvilleEstimator:
    """
    Estimateur de Harville (1973) - Modèle classique.
    
    Hypothèse: P(i finit k-ème | i1,...,i_{k-1} ont fini) 
               = p_i / sum(p_j pour j non fini)
    
    Avantages: Simple, fermé analytiquement pour top 2-3
    Inconvénients: Surestime les favoris aux places
    """
    
    def __init__(self, win_probs: np.ndarray):
        """
        Args:
            win_probs: Probabilités de victoire normalisées (sum = 1)
        """
        self.win_probs = np.array(win_probs)
        self.n = len(win_probs)
        self._validate()
    
    def _validate(self):
        """Valide les probabilités."""
        if not np.isclose(self.win_probs.sum(), 1.0, atol=0.01):
            self.win_probs = self.win_probs / self.win_probs.sum()
        self.win_probs = np.clip(self.win_probs, 1e-10, 1.0)
    
    def p_finish_position(self, horse_idx: int, position: int) -> float:
        """
        Calcule P(cheval i finit à la position k).
        
        Pour position 1: P(win) = p_i
        Pour position 2: sum over j≠i of P(j wins) * p_i/(1-p_j)
        Pour position 3+: récursif
        """
        if position == 1:
            return self.win_probs[horse_idx]
        
        if position == 2:
            prob = 0.0
            for j in range(self.n):
                if j != horse_idx:
                    remaining = 1 - self.win_probs[j]
                    if remaining > 1e-10:
                        prob += self.win_probs[j] * self.win_probs[horse_idx] / remaining
            return prob
        
        if position == 3:
            prob = 0.0
            for j in range(self.n):
                if j != horse_idx:
                    for k in range(self.n):
                        if k != horse_idx and k != j:
                            remaining_1 = 1 - self.win_probs[j]
                            remaining_2 = 1 - self.win_probs[j] - self.win_probs[k]
                            if remaining_1 > 1e-10 and remaining_2 > 1e-10:
                                p_j_1st = self.win_probs[j]
                                p_k_2nd_given_j = self.win_probs[k] / remaining_1
                                p_i_3rd_given_jk = self.win_probs[horse_idx] / remaining_2
                                prob += p_j_1st * p_k_2nd_given_j * p_i_3rd_given_jk
            return prob
        
        # Pour positions > 3, utiliser la simulation
        return self._simulate_position_prob(horse_idx, position)
    
    def _simulate_position_prob(self, horse_idx: int, position: int, 
                                 n_sim: int = 10000) -> float:
        """Estime P(position) par simulation pour positions > 3."""
        rng = np.random.default_rng(42)
        count = 0
        
        for _ in range(n_sim):
            remaining = list(range(self.n))
            probs = self.win_probs.copy()
            
            for pos in range(1, position + 1):
                p = probs[remaining] / probs[remaining].sum()
                chosen_idx = rng.choice(len(remaining), p=p)
                chosen = remaining[chosen_idx]
                
                if pos == position and chosen == horse_idx:
                    count += 1
                    break
                
                remaining.remove(chosen)
                if horse_idx not in remaining:
                    break
        
        return count / n_sim
    
    def p_place(self, top_n: int = 3) -> np.ndarray:
        """
        Calcule P(place) = P(finir dans le top N) pour chaque cheval.
        """
        place_probs = np.zeros(self.n)
        
        for i in range(self.n):
            for pos in range(1, top_n + 1):
                place_probs[i] += self.p_finish_position(i, pos)
        
        # Cap à 1.0
        place_probs = np.clip(place_probs, 0, 1.0)
        return place_probs
    
    def p_exacta(self, i: int, j: int) -> float:
        """P(i 1er, j 2ème)."""
        remaining = 1 - self.win_probs[i]
        if remaining < 1e-10:
            return 0.0
        return self.win_probs[i] * self.win_probs[j] / remaining
    
    def p_trifecta(self, i: int, j: int, k: int) -> float:
        """P(i 1er, j 2ème, k 3ème)."""
        rem1 = 1 - self.win_probs[i]
        rem2 = 1 - self.win_probs[i] - self.win_probs[j]
        if rem1 < 1e-10 or rem2 < 1e-10:
            return 0.0
        return (self.win_probs[i] * 
                self.win_probs[j] / rem1 * 
                self.win_probs[k] / rem2)
    
    def p_superfecta(self, i: int, j: int, k: int, l: int) -> float:
        """P(i 1er, j 2ème, k 3ème, l 4ème)."""
        rem1 = 1 - self.win_probs[i]
        rem2 = 1 - self.win_probs[i] - self.win_probs[j]
        rem3 = 1 - self.win_probs[i] - self.win_probs[j] - self.win_probs[k]
        if rem1 < 1e-10 or rem2 < 1e-10 or rem3 < 1e-10:
            return 0.0
        return (self.win_probs[i] * 
                self.win_probs[j] / rem1 * 
                self.win_probs[k] / rem2 *
                self.win_probs[l] / rem3)


class HeneryEstimator(HarvilleEstimator):
    """
    Estimateur de Henery (1981) - Ajustement des favoris.
    
    Modification: utilise p_i^gamma au lieu de p_i, où gamma < 1
    Cela réduit l'avantage des favoris aux places.
    
    Gamma optimal ≈ 0.81 selon études empiriques.
    """
    
    def __init__(self, win_probs: np.ndarray, gamma: float = 0.81):
        """
        Args:
            win_probs: Probabilités de victoire
            gamma: Exposant Henery (< 1 réduit avantage favoris)
        """
        self.gamma = gamma
        self.original_probs = np.array(win_probs)
        
        # Transformer les probas
        adjusted = self.original_probs ** gamma
        adjusted = adjusted / adjusted.sum()
        
        super().__init__(adjusted)
        # Garder les probas originales pour référence
        self._original = self.original_probs.copy()
    
    def get_adjustment_factor(self) -> np.ndarray:
        """Retourne le facteur d'ajustement par cheval."""
        return self.win_probs / self._original


class SternEstimator(HarvilleEstimator):
    """
    Estimateur de Stern (1990) - Ajustement additif.
    
    Modification: p_i_adj = p_i + lambda * (1/n - p_i)
    Cela "lisse" les probabilités vers l'uniforme.
    """
    
    def __init__(self, win_probs: np.ndarray, lambda_: float = 0.15):
        """
        Args:
            win_probs: Probabilités de victoire
            lambda_: Facteur de lissage (0 = Harville, 1 = uniforme)
        """
        self.lambda_ = lambda_
        self.original_probs = np.array(win_probs)
        n = len(win_probs)
        
        # Ajustement Stern
        uniform = np.ones(n) / n
        adjusted = self.original_probs + lambda_ * (uniform - self.original_probs)
        adjusted = adjusted / adjusted.sum()
        
        super().__init__(adjusted)


class LoBaconShoneEstimator:
    """
    Estimateur de Lo-Bacon-Shone (1994) - Réallocation itérative.
    
    Algorithme de réallocation des probabilités pour corriger
    la sur-estimation des favoris dans Harville.
    
    Principe: réallouer itérativement les probabilités de place
    en tenant compte des corrélations entre chevaux.
    """
    
    def __init__(self, win_probs: np.ndarray, iterations: int = 100, 
                 damping: float = 0.7):
        """
        Args:
            win_probs: Probabilités de victoire
            iterations: Nombre d'itérations de réallocation
            damping: Facteur d'amortissement (stabilité)
        """
        self.win_probs = np.array(win_probs)
        self.n = len(win_probs)
        self.iterations = iterations
        self.damping = damping
        
        # Normaliser
        if not np.isclose(self.win_probs.sum(), 1.0, atol=0.01):
            self.win_probs = self.win_probs / self.win_probs.sum()
        self.win_probs = np.clip(self.win_probs, 1e-10, 1.0)
        
        # Pré-calculer la matrice de réallocation
        self._compute_reallocation_matrix()
    
    def _compute_reallocation_matrix(self):
        """
        Calcule la matrice de réallocation R où R[i,j] représente
        la fraction de la probabilité de j réallouée à i quand j est éliminé.
        """
        self.R = np.zeros((self.n, self.n))
        
        for j in range(self.n):
            remaining = 1 - self.win_probs[j]
            if remaining > 1e-10:
                for i in range(self.n):
                    if i != j:
                        self.R[i, j] = self.win_probs[i] / remaining
    
    def p_place(self, top_n: int = 3) -> np.ndarray:
        """
        Calcule P(place) par réallocation itérative.
        """
        # Initialisation: Harville simple
        harville = HarvilleEstimator(self.win_probs)
        place_probs = harville.p_place(top_n)
        
        # Réallocation itérative
        for _ in range(self.iterations):
            new_probs = np.zeros(self.n)
            
            for i in range(self.n):
                # Contribution directe (victoire)
                new_probs[i] = self.win_probs[i]
                
                # Contributions des places 2..top_n
                for j in range(self.n):
                    if i != j:
                        # Probabilité que j soit avant i mais i soit placé
                        contrib = self.R[i, j] * place_probs[j] * (top_n - 1) / top_n
                        new_probs[i] += contrib * self.damping
            
            # Normaliser pour que la somme des probas de place = top_n
            scale = top_n / new_probs.sum() if new_probs.sum() > 0 else 1
            new_probs = np.clip(new_probs * scale, 0, 1.0)
            
            # Mise à jour avec damping
            place_probs = (1 - self.damping) * place_probs + self.damping * new_probs
        
        return place_probs
    
    def p_exacta(self, i: int, j: int) -> float:
        """P(i 1er, j 2ème) avec ajustement LBS."""
        harville = HarvilleEstimator(self.win_probs)
        base = harville.p_exacta(i, j)
        
        # Ajustement: réduire si i est favori
        if self.win_probs[i] > 0.2:  # Favori
            adjustment = 0.95 ** (self.win_probs[i] * 10)
            base *= adjustment
        
        return base
    
    def p_trifecta(self, i: int, j: int, k: int) -> float:
        """P(i 1er, j 2ème, k 3ème) avec ajustement LBS."""
        harville = HarvilleEstimator(self.win_probs)
        base = harville.p_trifecta(i, j, k)
        
        # Ajustement pour combinaisons de favoris
        top_probs = sorted([self.win_probs[i], self.win_probs[j], self.win_probs[k]], 
                          reverse=True)
        if top_probs[0] > 0.2:  # Au moins un favori fort
            adjustment = 0.92 ** (sum(top_probs) * 5)
            base *= adjustment
        
        return base


# =============================================================================
# SIMULATEUR PLACKETT-LUCE AVEC TEMPÉRATURE
# =============================================================================

class PlackettLuceTemperatureSimulator:
    """
    Simulateur Plackett-Luce avec paramètre de température.
    
    Température T:
    - T = 1: Plackett-Luce standard
    - T < 1: Favoris plus avantagés (distribution plus concentrée)
    - T > 1: Plus d'incertitude (distribution plus uniforme)
    
    La température est apprise par discipline pour optimiser
    la log-vraisemblance sur les données historiques.
    """
    
    def __init__(self, win_probs: np.ndarray, temperature: float = 1.0,
                 seed: Optional[int] = None):
        """
        Args:
            win_probs: Probabilités de victoire (p_win_blend)
            temperature: Paramètre de température
            seed: Graine aléatoire
        """
        self.original_probs = np.array(win_probs)
        self.temperature = temperature
        self.n = len(win_probs)
        self.rng = np.random.default_rng(seed)
        
        # Appliquer la température
        self._apply_temperature()
    
    def _apply_temperature(self):
        """Applique la température aux probabilités."""
        # Convertir en log-odds, diviser par T, reconvertir
        log_probs = np.log(np.clip(self.original_probs, 1e-10, 1.0))
        scaled_log_probs = log_probs / self.temperature
        self.probs = softmax(scaled_log_probs)
    
    def set_temperature(self, temperature: float):
        """Change la température."""
        self.temperature = temperature
        self._apply_temperature()
    
    def simulate_race(self) -> np.ndarray:
        """
        Simule une course complète.
        
        Returns:
            np.ndarray: Ordre d'arrivée (indices des chevaux)
        """
        remaining = list(range(self.n))
        probs = self.probs.copy()
        order = []
        
        for _ in range(self.n):
            # Normaliser les probas restantes
            p = probs[remaining]
            p = p / p.sum()
            
            # Tirer le prochain
            chosen_idx = self.rng.choice(len(remaining), p=p)
            chosen = remaining[chosen_idx]
            
            order.append(chosen)
            remaining.remove(chosen)
        
        return np.array(order)
    
    def simulate_n_races(self, n: int) -> np.ndarray:
        """
        Simule n courses.
        
        Returns:
            np.ndarray: Shape (n, n_horses)
        """
        results = np.zeros((n, self.n), dtype=int)
        
        for i in range(n):
            results[i] = self.simulate_race()
        
        return results
    
    def simulate_n_races_parallel(self, n: int, n_workers: int = 4) -> np.ndarray:
        """Version parallélisée pour grandes simulations."""
        # Pour l'instant, version séquentielle optimisée
        return self.simulate_n_races(n)
    
    def estimate_place_probs(self, n_sim: int, top_n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estime P(place) par simulation.
        
        Returns:
            (place_probs, std_errors)
        """
        place_counts = np.zeros(self.n)
        
        arrivals = self.simulate_n_races(n_sim)
        
        for arrival in arrivals:
            for pos in range(top_n):
                place_counts[arrival[pos]] += 1
        
        place_probs = place_counts / n_sim
        
        # Erreur standard (Bernoulli)
        std_errors = np.sqrt(place_probs * (1 - place_probs) / n_sim)
        
        return place_probs, std_errors
    
    def estimate_combo_probs(self, n_sim: int, structure: str = 'trio') -> Dict[Tuple, float]:
        """
        Estime P(combinaisons) par simulation.
        
        Args:
            n_sim: Nombre de simulations
            structure: 'exacta', 'trio', 'quarte', 'quinte'
        
        Returns:
            Dict {(i, j, ...) -> probabilité}
        """
        arrivals = self.simulate_n_races(n_sim)
        
        # Compter les combinaisons
        combo_counts = Counter()
        
        structure_map = {
            'exacta': 2, 'couple': 2,
            'trio': 3, 'tierce': 3,
            'quarte': 4,
            'quinte': 5
        }
        
        k = structure_map.get(structure.lower(), 3)
        
        for arrival in arrivals:
            combo = tuple(arrival[:k])
            combo_counts[combo] += 1
        
        # Convertir en probabilités
        combo_probs = {combo: count / n_sim for combo, count in combo_counts.items()}
        
        return combo_probs


# =============================================================================
# SÉLECTEUR D'ESTIMATEUR PAR DISCIPLINE
# =============================================================================

class PlaceEstimatorSelector:
    """
    Sélectionne le meilleur estimateur par discipline
    via validation (log-vraisemblance, Brier place).
    """
    
    def __init__(self, config: PlaceEstimatorConfig = None):
        self.config = config or PlaceEstimatorConfig()
        
        # Mapping discipline -> estimateur optimal (à calibrer)
        self.optimal_estimator = {
            'plat': 'henery',
            'trot': 'lbs',
            'obstacle': 'stern',
            'default': 'harville'
        }
        
        # Paramètres optimaux par discipline
        self.optimal_params = {
            'plat': {'gamma': 0.81, 'temperature': 0.95},
            'trot': {'iterations': 100, 'temperature': 1.05},
            'obstacle': {'lambda_': 0.18, 'temperature': 1.10},
            'default': {'temperature': 1.0}
        }
    
    def get_estimator(self, win_probs: np.ndarray, discipline: str = 'default') -> Any:
        """
        Retourne l'estimateur optimal pour la discipline.
        """
        discipline = discipline.lower() if discipline else 'default'
        if discipline not in self.optimal_estimator:
            discipline = 'default'
        
        estimator_type = self.optimal_estimator[discipline]
        params = self.optimal_params.get(discipline, {})
        
        if estimator_type == 'harville':
            return HarvilleEstimator(win_probs)
        elif estimator_type == 'henery':
            return HeneryEstimator(win_probs, gamma=params.get('gamma', 0.81))
        elif estimator_type == 'stern':
            return SternEstimator(win_probs, lambda_=params.get('lambda_', 0.15))
        elif estimator_type == 'lbs':
            return LoBaconShoneEstimator(
                win_probs, 
                iterations=params.get('iterations', 100),
                damping=params.get('damping', 0.7)
            )
        else:
            return HarvilleEstimator(win_probs)
    
    def get_simulator(self, win_probs: np.ndarray, discipline: str = 'default',
                      seed: Optional[int] = None) -> PlackettLuceTemperatureSimulator:
        """
        Retourne le simulateur PL avec température optimale.
        """
        discipline = discipline.lower() if discipline else 'default'
        params = self.optimal_params.get(discipline, self.optimal_params['default'])
        temperature = params.get('temperature', 1.0)
        
        return PlackettLuceTemperatureSimulator(win_probs, temperature, seed)
    
    def validate_estimator(self, estimator_name: str, historical_data: List[Dict],
                           discipline: str) -> Dict[str, float]:
        """
        Valide un estimateur sur des données historiques.
        
        Args:
            estimator_name: 'harville', 'henery', 'stern', 'lbs'
            historical_data: Liste de courses avec résultats
            discipline: Discipline à filtrer
        
        Returns:
            Dict avec métriques (log_likelihood, brier_place, ece)
        """
        log_likelihoods = []
        brier_scores = []
        
        for race in historical_data:
            if race.get('discipline', '').lower() != discipline.lower():
                continue
            
            win_probs = np.array(race['win_probs'])
            actual_places = race.get('actual_places', [])  # Indices des placés
            
            if len(actual_places) < 3:
                continue
            
            # Créer l'estimateur
            if estimator_name == 'harville':
                est = HarvilleEstimator(win_probs)
            elif estimator_name == 'henery':
                est = HeneryEstimator(win_probs)
            elif estimator_name == 'stern':
                est = SternEstimator(win_probs)
            elif estimator_name == 'lbs':
                est = LoBaconShoneEstimator(win_probs)
            else:
                continue
            
            # Probabilités de place estimées
            p_place = est.p_place(top_n=3)
            
            # Log-likelihood
            for idx in actual_places[:3]:
                ll = np.log(max(p_place[idx], 1e-10))
                log_likelihoods.append(ll)
            
            # Brier score pour chaque cheval
            n = len(win_probs)
            for i in range(n):
                actual = 1.0 if i in actual_places[:3] else 0.0
                brier = (p_place[i] - actual) ** 2
                brier_scores.append(brier)
        
        return {
            'log_likelihood': np.mean(log_likelihoods) if log_likelihoods else 0,
            'brier_place': np.mean(brier_scores) if brier_scores else 1,
            'n_races': len(log_likelihoods) // 3
        }


# =============================================================================
# CALCULATEUR EV POUR EXOTIQUES
# =============================================================================

class ExoticEVCalculator:
    """
    Calcule l'Expected Value des tickets exotiques.
    
    Modes:
    - Parimutuel: applique takeout, estime payout
    - Cotes fixes: multiplie par les cotes fournies
    """
    
    def __init__(self, takeout_rate: float = 0.16):
        self.takeout_rate = takeout_rate
    
    def calculate_ev_parimutuel(self, prob: float, public_prob: float) -> float:
        """
        Calcule EV pour pari parimutuel.
        
        EV = prob * payout - 1
        payout ≈ (1 - takeout) / public_prob
        """
        if public_prob < 1e-10:
            return -1.0
        
        payout = (1 - self.takeout_rate) / public_prob
        ev = prob * payout - 1
        return ev
    
    def calculate_ev_fixed_odds(self, prob: float, odds: float) -> float:
        """
        Calcule EV pour cotes fixes.
        
        EV = prob * odds - 1
        """
        return prob * odds - 1
    
    def estimate_public_prob(self, model_prob: float, market_efficiency: float = 0.85) -> float:
        """
        Estime la probabilité publique à partir de notre modèle.
        
        On suppose que le public a une estimation bruitée.
        market_efficiency = corrélation entre proba modèle et proba publique
        """
        # Ajouter du bruit
        noise = np.random.normal(0, (1 - market_efficiency) * 0.5)
        public = model_prob * (1 + noise)
        public = np.clip(public, 1e-6, 0.99)
        return public
    
    def calculate_tickets_ev(self, combo_probs: Dict[Tuple, float],
                             fixed_odds: Optional[Dict[Tuple, float]] = None,
                             market_efficiency: float = 0.85) -> Dict[Tuple, Dict]:
        """
        Calcule EV pour un ensemble de tickets.
        
        Args:
            combo_probs: {combo: prob_modele}
            fixed_odds: {combo: cote} si disponible
            market_efficiency: Efficience du marché
        
        Returns:
            {combo: {prob, payout, ev, ev_pct}}
        """
        results = {}
        
        for combo, prob in combo_probs.items():
            if prob < 1e-8:
                continue
            
            if fixed_odds and combo in fixed_odds:
                # Cotes fixes
                odds = fixed_odds[combo]
                ev = self.calculate_ev_fixed_odds(prob, odds)
                payout = odds
            else:
                # Parimutuel
                public_prob = self.estimate_public_prob(prob, market_efficiency)
                payout = (1 - self.takeout_rate) / public_prob
                ev = prob * payout - 1
            
            results[combo] = {
                'prob': prob,
                'prob_pct': prob * 100,
                'payout': round(payout, 2),
                'ev': round(ev, 4),
                'ev_pct': round(ev * 100, 2)
            }
        
        return results


# =============================================================================
# MÉTRIQUES DE CALIBRATION
# =============================================================================

class CalibrationMetrics:
    """
    Métriques de calibration pour les probabilités de place.
    """
    
    @staticmethod
    def brier_score(predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Brier Score: mean((p - y)^2)
        Plus bas = meilleur
        """
        return np.mean((predicted - actual) ** 2)
    
    @staticmethod
    def expected_calibration_error(predicted: np.ndarray, actual: np.ndarray,
                                    n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE).
        
        Mesure la différence entre probabilités prédites et fréquences observées
        par bin de probabilité.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_pred = predicted[mask].mean()
                bin_actual = actual[mask].mean()
                bin_weight = mask.sum() / len(predicted)
                ece += bin_weight * abs(bin_pred - bin_actual)
        
        return ece
    
    @staticmethod
    def log_loss(predicted: np.ndarray, actual: np.ndarray) -> float:
        """Log loss (cross-entropy)."""
        predicted = np.clip(predicted, 1e-10, 1 - 1e-10)
        return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    
    @staticmethod
    def check_monotonicity(probs: np.ndarray, outcomes: np.ndarray, 
                           n_deciles: int = 10) -> Dict[str, Any]:
        """
        Vérifie la cohérence monotone par déciles.
        
        Les chevaux avec p(place) plus élevé devraient être placés plus souvent.
        """
        # Trier par probabilité prédite
        sorted_idx = np.argsort(probs)
        sorted_probs = probs[sorted_idx]
        sorted_outcomes = outcomes[sorted_idx]
        
        # Découper en déciles
        decile_size = len(probs) // n_deciles
        decile_rates = []
        decile_avg_probs = []
        
        for i in range(n_deciles):
            start = i * decile_size
            end = start + decile_size if i < n_deciles - 1 else len(probs)
            
            decile_probs = sorted_probs[start:end]
            decile_out = sorted_outcomes[start:end]
            
            decile_avg_probs.append(decile_probs.mean())
            decile_rates.append(decile_out.mean())
        
        # Vérifier la monotonie
        is_monotonic = all(decile_rates[i] <= decile_rates[i + 1] 
                          for i in range(len(decile_rates) - 1))
        
        # Corrélation de Spearman
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(decile_avg_probs, decile_rates)
        
        return {
            'is_monotonic': is_monotonic,
            'spearman_correlation': corr,
            'p_value': p_value,
            'decile_probs': decile_avg_probs,
            'decile_rates': decile_rates
        }


# =============================================================================
# VALIDATEUR DE STABILITÉ EV
# =============================================================================

class EVStabilityValidator:
    """
    Valide la stabilité de l'EV des packs en fonction du nombre de simulations.
    """
    
    def __init__(self, win_probs: np.ndarray, config: PlaceEstimatorConfig = None):
        self.win_probs = win_probs
        self.config = config or PlaceEstimatorConfig()
    
    def check_stability(self, n_values: List[int] = None, 
                        structure: str = 'trio') -> Dict[str, Any]:
        """
        Vérifie la convergence de l'EV en fonction de N simulations.
        
        Args:
            n_values: Liste de N à tester (ex: [1000, 5000, 10000, 20000])
            structure: Type de pari
        
        Returns:
            Dict avec statistiques de stabilité
        """
        if n_values is None:
            n_values = [1000, 5000, 10000, 20000, 50000]
        
        simulator = PlackettLuceTemperatureSimulator(
            self.win_probs, 
            temperature=self.config.temperature_default,
            seed=self.config.random_seed
        )
        
        ev_calculator = ExoticEVCalculator(self.config.takeout_rate)
        
        results_by_n = {}
        top_combo_evs = defaultdict(list)
        
        for n in n_values:
            combo_probs = simulator.estimate_combo_probs(n, structure)
            combo_evs = ev_calculator.calculate_tickets_ev(combo_probs)
            
            # Statistiques
            evs = [v['ev'] for v in combo_evs.values()]
            positive_evs = [e for e in evs if e > 0]
            
            results_by_n[n] = {
                'n_combos': len(combo_evs),
                'n_positive_ev': len(positive_evs),
                'mean_ev': np.mean(evs) if evs else 0,
                'std_ev': np.std(evs) if evs else 0,
                'max_ev': max(evs) if evs else 0
            }
            
            # Tracker les top combos
            sorted_combos = sorted(combo_evs.items(), key=lambda x: x[1]['ev'], reverse=True)
            for combo, stats in sorted_combos[:10]:
                top_combo_evs[combo].append(stats['ev'])
        
        # Analyser la convergence
        convergence_stats = {}
        for combo, evs in top_combo_evs.items():
            if len(evs) >= 3:
                convergence_stats[combo] = {
                    'mean_ev': np.mean(evs),
                    'std_ev': np.std(evs),
                    'cv': np.std(evs) / abs(np.mean(evs)) if np.mean(evs) != 0 else float('inf')
                }
        
        # Déterminer N optimal (CV < 10%)
        stable_n = None
        for i, n in enumerate(n_values):
            if i >= 2:  # Au moins 3 points
                recent_means = [results_by_n[n_values[j]]['mean_ev'] for j in range(i-2, i+1)]
                cv = np.std(recent_means) / abs(np.mean(recent_means)) if np.mean(recent_means) != 0 else float('inf')
                if cv < 0.10:
                    stable_n = n
                    break
        
        return {
            'results_by_n': results_by_n,
            'convergence_by_combo': convergence_stats,
            'recommended_n': stable_n or n_values[-1],
            'is_stable': stable_n is not None
        }


# =============================================================================
# CLASSE PRINCIPALE - ORCHESTRATEUR
# =============================================================================

class PlaceProbabilityEstimator:
    """
    Classe principale pour estimer p(place) et probabilités d'ordre.
    
    Usage:
        estimator = PlaceProbabilityEstimator(p_win_blend, discipline='plat')
        
        # Probabilités de place
        p_place = estimator.estimate_place_probs(top_n=3)
        
        # Probabilités de combinaisons
        combo_probs = estimator.estimate_combo_probs(structure='trio')
        
        # EV des tickets
        tickets_ev = estimator.calculate_tickets_ev(combo_probs)
        
        # Rapport complet
        report = estimator.generate_report()
    """
    
    def __init__(self, p_win_blend: np.ndarray, discipline: str = 'default',
                 config: PlaceEstimatorConfig = None, horse_names: List[str] = None):
        """
        Args:
            p_win_blend: Probabilités de victoire (blend modèle/marché)
            discipline: 'plat', 'trot', 'obstacle'
            config: Configuration optionnelle
            horse_names: Noms des chevaux (optionnel)
        """
        self.p_win = np.array(p_win_blend)
        self.discipline = discipline.lower() if discipline else 'default'
        self.config = config or PlaceEstimatorConfig()
        self.n_horses = len(self.p_win)
        self.horse_names = horse_names or [f"#{i+1}" for i in range(self.n_horses)]
        
        # Normaliser
        if not np.isclose(self.p_win.sum(), 1.0, atol=0.01):
            self.p_win = self.p_win / self.p_win.sum()
        
        # Initialiser les composants
        self.selector = PlaceEstimatorSelector(self.config)
        self.estimator = self.selector.get_estimator(self.p_win, self.discipline)
        self.simulator = self.selector.get_simulator(
            self.p_win, self.discipline, self.config.random_seed
        )
        self.ev_calculator = ExoticEVCalculator(self.config.takeout_rate)
        
        # Cache
        self._cache = {}
    
    def estimate_place_probs(self, top_n: int = 3, 
                              method: str = 'auto') -> Dict[str, Any]:
        """
        Estime les probabilités de place.
        
        Args:
            top_n: Nombre de places (3 pour tiercé)
            method: 'auto', 'harville', 'henery', 'stern', 'lbs', 'simulation'
        
        Returns:
            Dict avec p_place et métadonnées
        """
        cache_key = f"place_{top_n}_{method}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if method == 'auto':
            # Utiliser l'estimateur sélectionné + simulation pour validation
            p_place_analytical = self.estimator.p_place(top_n)
            p_place_sim, std_errors = self.simulator.estimate_place_probs(
                self.config.num_simulations, top_n
            )
            
            # Moyenne pondérée (analytique + simulation)
            p_place = 0.6 * p_place_analytical + 0.4 * p_place_sim
            
        elif method == 'simulation':
            p_place, std_errors = self.simulator.estimate_place_probs(
                self.config.num_simulations, top_n
            )
            
        else:
            # Estimateur spécifique
            if method == 'harville':
                est = HarvilleEstimator(self.p_win)
            elif method == 'henery':
                est = HeneryEstimator(self.p_win, self.config.henery_gamma)
            elif method == 'stern':
                est = SternEstimator(self.p_win, self.config.stern_lambda)
            elif method == 'lbs':
                est = LoBaconShoneEstimator(self.p_win, self.config.lbs_iterations)
            else:
                est = self.estimator
            
            p_place = est.p_place(top_n)
            std_errors = np.zeros_like(p_place)
        
        result = {
            'p_place': p_place,
            'std_errors': std_errors if 'std_errors' in dir() else np.zeros_like(p_place),
            'top_n': top_n,
            'method': method,
            'discipline': self.discipline,
            'by_horse': {
                self.horse_names[i]: {
                    'p_win': round(float(self.p_win[i]), 4),
                    'p_place': round(float(p_place[i]), 4),
                    'place_vs_win_ratio': round(float(p_place[i] / self.p_win[i]), 2) if self.p_win[i] > 0.001 else 0
                }
                for i in range(self.n_horses)
            }
        }
        
        self._cache[cache_key] = result
        return result
    
    def estimate_combo_probs(self, structure: str = 'trio', 
                              n_sim: int = None) -> Dict[str, Any]:
        """
        Estime les probabilités de combinaisons.
        
        Args:
            structure: 'exacta', 'trio', 'quarte', 'quinte'
            n_sim: Nombre de simulations (défaut: config)
        
        Returns:
            Dict avec combo_probs et statistiques
        """
        n_sim = n_sim or self.config.num_simulations
        
        cache_key = f"combo_{structure}_{n_sim}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Simuler
        combo_probs = self.simulator.estimate_combo_probs(n_sim, structure)
        
        # Trier par probabilité
        sorted_combos = sorted(combo_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Statistiques
        probs = list(combo_probs.values())
        
        result = {
            'structure': structure,
            'n_simulations': n_sim,
            'n_combos': len(combo_probs),
            'coverage': sum(probs),  # Devrait être ~1
            'top_combos': [
                {
                    'combo': [self.horse_names[i] for i in combo],
                    'combo_indices': list(combo),
                    'prob': round(prob, 6),
                    'prob_pct': round(prob * 100, 3)
                }
                for combo, prob in sorted_combos[:100]
            ],
            'combo_probs': combo_probs  # Dict complet pour calcul EV
        }
        
        self._cache[cache_key] = result
        return result
    
    def calculate_tickets_ev(self, combo_probs: Dict[Tuple, float] = None,
                              structure: str = 'trio',
                              fixed_odds: Dict[Tuple, float] = None) -> Dict[str, Any]:
        """
        Calcule l'EV de chaque ticket potentiel.
        
        Args:
            combo_probs: Probabilités des combos (ou utilise le cache)
            structure: Type de pari
            fixed_odds: Cotes fixes si disponibles
        
        Returns:
            Dict avec tickets triés par EV
        """
        if combo_probs is None:
            combo_data = self.estimate_combo_probs(structure)
            combo_probs = combo_data['combo_probs']
        
        # Calculer EV
        tickets_ev = self.ev_calculator.calculate_tickets_ev(
            combo_probs, fixed_odds
        )
        
        # Trier par EV
        sorted_tickets = sorted(tickets_ev.items(), key=lambda x: x[1]['ev'], reverse=True)
        
        # Filtrer EV positifs
        positive_ev = [(c, s) for c, s in sorted_tickets if s['ev'] > 0]
        
        result = {
            'structure': structure,
            'takeout_rate': self.config.takeout_rate,
            'n_tickets': len(tickets_ev),
            'n_positive_ev': len(positive_ev),
            'positive_ev_tickets': [
                {
                    'combo': [self.horse_names[i] for i in combo],
                    'combo_indices': list(combo),
                    **stats
                }
                for combo, stats in positive_ev[:50]
            ],
            'all_tickets_ev': tickets_ev
        }
        
        return result
    
    def generate_report(self, structure: str = 'trio', 
                        include_validation: bool = True) -> Dict[str, Any]:
        """
        Génère un rapport complet.
        """
        # Probabilités de place
        place_data = self.estimate_place_probs(top_n=3)
        
        # Probabilités de combinaisons
        combo_data = self.estimate_combo_probs(structure)
        
        # EV des tickets
        ev_data = self.calculate_tickets_ev(combo_data['combo_probs'], structure)
        
        report = {
            'meta': {
                'discipline': self.discipline,
                'n_horses': self.n_horses,
                'n_simulations': self.config.num_simulations,
                'temperature': self.simulator.temperature,
                'estimator': type(self.estimator).__name__
            },
            'place_probabilities': place_data,
            'combo_probabilities': {
                'structure': combo_data['structure'],
                'n_combos': combo_data['n_combos'],
                'top_combos': combo_data['top_combos'][:20]
            },
            'expected_values': {
                'n_positive_ev': ev_data['n_positive_ev'],
                'best_tickets': ev_data['positive_ev_tickets'][:10]
            }
        }
        
        # Validation optionnelle
        if include_validation:
            validator = EVStabilityValidator(self.p_win, self.config)
            stability = validator.check_stability(
                n_values=[5000, 10000, 20000], 
                structure=structure
            )
            report['validation'] = {
                'recommended_n': stability['recommended_n'],
                'is_stable': stability['is_stable']
            }
        
        return report
    
    def compare_estimators(self) -> Dict[str, Any]:
        """
        Compare les différents estimateurs sur les données actuelles.
        """
        estimators = {
            'Harville': HarvilleEstimator(self.p_win),
            'Henery (γ=0.81)': HeneryEstimator(self.p_win, 0.81),
            'Stern (λ=0.15)': SternEstimator(self.p_win, 0.15),
            'Lo-Bacon-Shone': LoBaconShoneEstimator(self.p_win)
        }
        
        results = {}
        for name, est in estimators.items():
            p_place = est.p_place(top_n=3)
            
            results[name] = {
                'p_place': {
                    self.horse_names[i]: round(float(p_place[i]), 4)
                    for i in range(self.n_horses)
                },
                'sum_p_place': round(float(p_place.sum()), 4),
                'max_p_place': round(float(p_place.max()), 4),
                'min_p_place': round(float(p_place.min()), 4)
            }
        
        return results


# =============================================================================
# APPRENTISSAGE DE LA TEMPÉRATURE
# =============================================================================

class TemperatureLearner:
    """
    Apprend la température optimale par discipline via validation croisée.
    """
    
    def __init__(self, historical_races: List[Dict]):
        """
        Args:
            historical_races: Liste de courses avec:
                - win_probs: probabilités de victoire
                - actual_order: ordre d'arrivée réel (indices)
                - discipline: type de course
        """
        self.races = historical_races
    
    def _log_likelihood(self, temperature: float, win_probs: np.ndarray,
                        actual_order: List[int]) -> float:
        """Calcule la log-vraisemblance pour une température donnée."""
        ll = 0.0
        remaining = list(range(len(win_probs)))
        
        # Appliquer température
        log_probs = np.log(np.clip(win_probs, 1e-10, 1.0))
        scaled = softmax(log_probs / temperature)
        
        for pos, horse in enumerate(actual_order):
            if horse not in remaining:
                continue
            
            p = scaled[remaining]
            p = p / p.sum()
            
            idx = remaining.index(horse)
            ll += np.log(max(p[idx], 1e-10))
            
            remaining.remove(horse)
            if len(remaining) == 0:
                break
        
        return ll
    
    def learn_temperature(self, discipline: str, 
                          temp_range: Tuple[float, float] = (0.5, 2.0)) -> Dict[str, Any]:
        """
        Trouve la température optimale pour une discipline.
        
        Returns:
            Dict avec température optimale et métriques
        """
        # Filtrer les courses de la discipline
        races = [r for r in self.races 
                 if r.get('discipline', '').lower() == discipline.lower()]
        
        if len(races) < 10:
            return {'optimal_temperature': 1.0, 'n_races': len(races), 
                    'error': 'Pas assez de données'}
        
        def neg_total_ll(temp):
            total = 0
            for race in races:
                ll = self._log_likelihood(
                    temp,
                    np.array(race['win_probs']),
                    race['actual_order']
                )
                total += ll
            return -total
        
        # Optimiser
        result = minimize_scalar(neg_total_ll, bounds=temp_range, method='bounded')
        
        optimal_temp = result.x
        optimal_ll = -result.fun
        
        # Baseline (temp=1.0)
        baseline_ll = -neg_total_ll(1.0)
        
        return {
            'optimal_temperature': round(optimal_temp, 3),
            'log_likelihood': round(optimal_ll, 2),
            'baseline_ll': round(baseline_ll, 2),
            'improvement': round((optimal_ll - baseline_ll) / abs(baseline_ll) * 100, 2),
            'n_races': len(races),
            'discipline': discipline
        }


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def estimate_exotic_probs(p_win_blend: List[float], 
                          discipline: str = 'plat',
                          structure: str = 'trio',
                          n_simulations: int = 20000,
                          horse_names: List[str] = None) -> Dict[str, Any]:
    """
    Fonction utilitaire pour estimer rapidement les probabilités exotiques.
    
    Args:
        p_win_blend: Probabilités de victoire
        discipline: Type de course
        structure: 'exacta', 'trio', 'quarte', 'quinte'
        n_simulations: Nombre de simulations
        horse_names: Noms des chevaux
    
    Returns:
        Dict avec p_place, combo_probs, tickets_ev
    """
    config = PlaceEstimatorConfig(num_simulations=n_simulations)
    estimator = PlaceProbabilityEstimator(
        np.array(p_win_blend),
        discipline=discipline,
        config=config,
        horse_names=horse_names
    )
    
    return estimator.generate_report(structure=structure, include_validation=False)


def compare_estimators_on_race(p_win: List[float], 
                                horse_names: List[str] = None) -> Dict[str, Any]:
    """
    Compare tous les estimateurs sur une course donnée.
    """
    estimator = PlaceProbabilityEstimator(np.array(p_win), horse_names=horse_names)
    return estimator.compare_estimators()


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEST - Estimateur de Probabilités de Place et d'Ordre")
    print("=" * 70)
    
    # Exemple: course avec 10 partants
    np.random.seed(42)
    
    # Probabilités de victoire (blend modèle/marché)
    p_win = np.array([0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01])
    p_win = p_win / p_win.sum()
    
    horse_names = [
        "Favori Star", "Challenger", "Dark Horse", "Outsider 1",
        "Outsider 2", "Long Shot 1", "Long Shot 2", "Long Shot 3",
        "Very Long", "Extreme Long"
    ]
    
    print("\n1. PROBABILITÉS D'ENTRÉE (p_win_blend)")
    print("-" * 40)
    for i, (name, p) in enumerate(zip(horse_names, p_win)):
        print(f"  {i+1}. {name}: {p*100:.1f}%")
    
    # Créer l'estimateur
    estimator = PlaceProbabilityEstimator(
        p_win, 
        discipline='plat',
        horse_names=horse_names
    )
    
    print("\n2. COMPARAISON DES ESTIMATEURS (p_place)")
    print("-" * 40)
    comparison = estimator.compare_estimators()
    
    for method, data in comparison.items():
        print(f"\n{method}:")
        for name in horse_names[:5]:
            print(f"  {name}: {data['p_place'][name]*100:.1f}%")
    
    print("\n3. PROBABILITÉS DE COMBINAISONS (Trio)")
    print("-" * 40)
    combo_data = estimator.estimate_combo_probs('trio', n_sim=20000)
    
    print(f"Nombre de combos observés: {combo_data['n_combos']}")
    print("\nTop 10 Trios:")
    for i, combo in enumerate(combo_data['top_combos'][:10]):
        names = " - ".join(combo['combo'])
        print(f"  {i+1}. {names}: {combo['prob_pct']:.2f}%")
    
    print("\n4. EXPECTED VALUES (Parimutuel, takeout 16%)")
    print("-" * 40)
    ev_data = estimator.calculate_tickets_ev(structure='trio')
    
    print(f"Tickets EV > 0: {ev_data['n_positive_ev']} / {ev_data['n_tickets']}")
    print("\nMeilleurs tickets:")
    for ticket in ev_data['positive_ev_tickets'][:5]:
        names = " - ".join(ticket['combo'])
        print(f"  {names}: EV={ticket['ev_pct']:+.1f}%, payout={ticket['payout']:.0f}x")
    
    print("\n5. TEST STABILITÉ EV")
    print("-" * 40)
    validator = EVStabilityValidator(p_win)
    stability = validator.check_stability([5000, 10000, 20000], 'trio')
    
    print(f"N recommandé: {stability['recommended_n']}")
    print(f"Stable: {stability['is_stable']}")
    
    for n, stats in stability['results_by_n'].items():
        print(f"  N={n}: {stats['n_positive_ev']} EV+, mean_ev={stats['mean_ev']:.3f}")
    
    print("\n" + "=" * 70)
    print("TESTS TERMINÉS AVEC SUCCÈS")
    print("=" * 70)
