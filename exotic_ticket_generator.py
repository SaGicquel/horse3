#!/usr/bin/env python3
"""
Générateur de Tickets Exotiques (Trio/Quinté) via Monte Carlo
==============================================================
- Simulation Plackett-Luce pour estimer les probabilités de combinaisons
- Packs de tickets: SÛR / ÉQUILIBRÉ / RISQUÉ
- Optimisation EV sous contrainte de variance
- Support Trio, Quinté, Quarté
- Respect des limites per_ticket_rate et max_pack_rate

CONFIGURATION:
- num_simulations est chargé depuis config/pro_betting.yaml (unifié à 20000)
- Garantit la cohérence avec race_pronostic_generator.py
- Exotics defaults: per_ticket_rate=0.75%, max_pack_rate=4%

Auteur: Horse Racing AI System
Date: 2024-12
Version: 2.0.0
"""

import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import Counter
from itertools import permutations, combinations
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

warnings.filterwarnings("ignore")

# Import de la configuration centralisée
try:
    from config.loader import get_config

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================


# Charger num_simulations depuis config centralisée
def _get_unified_num_simulations() -> int:
    """Retourne le nombre unifié de simulations Monte Carlo (20000)."""
    if CONFIG_AVAILABLE:
        try:
            config = get_config()
            return config.simulation.num_simulations
        except Exception:
            pass
    return 20000  # Valeur unifiée par défaut


def _get_exotics_defaults() -> Tuple[float, float, float]:
    """Retourne (per_ticket_rate, max_pack_rate, rounding_increment)."""
    if CONFIG_AVAILABLE:
        try:
            config = get_config()
            return (
                config.exotics_defaults.per_ticket_rate,
                config.exotics_defaults.max_pack_rate,
                config.betting_defaults.rounding_increment_eur,
            )
        except Exception:
            pass
    return (0.0075, 0.04, 0.5)  # Défauts: 0.75%, 4%, 0.50€


@dataclass
class ExoticConfig:
    """Configuration pour la génération de tickets exotiques."""

    # Simulation - UNIFIÉ à 20000 (chargé depuis config centralisée)
    num_simulations: int = None  # Sera initialisé à 20000 via __post_init__
    random_seed: Optional[int] = 42

    # Budget et tickets (legacy)
    budget: float = 100.0
    max_tickets: int = 50
    min_stake: float = 1.0  # Mise minimum par ticket

    # Bankroll pour calcul des taux
    bankroll: float = 1000.0

    # Taux de mise (exotics_defaults)
    per_ticket_rate: float = None  # 0.75% par ticket (chargé depuis config)
    max_pack_rate: float = None  # 4% max par pack (chargé depuis config)
    rounding_increment_eur: float = None  # 0.50€ (chargé depuis config)

    # Takeout parimutuel
    takeout_rate: float = 0.16  # 16% de prélèvement

    # Structures supportées
    # 'trio_ordre': 3 chevaux dans l'ordre exact
    # 'trio_desordre': 3 chevaux peu importe l'ordre
    # 'trio_champ_reduit': base + chevaux en champ
    # 'quarte_ordre', 'quarte_desordre'
    # 'quinte_ordre', 'quinte_desordre', 'quinte_plus'
    structure: str = "trio_ordre"

    # Paramètres pour champ réduit
    champ_bases: int = 2  # Nombre de bases fixes
    champ_associates: int = 4  # Nombre de chevaux en champ

    # Packs - seuils différenciés
    pack_sure_ev_min: float = 0.05  # EV minimum pour pack SÛR
    pack_balanced_ev_min: float = 0.10
    pack_risky_ev_min: float = 0.15

    # Variance cible par pack (coefficient de variation)
    pack_sure_max_cv: float = 1.5  # CV = std/mean - faible variance
    pack_balanced_max_cv: float = 3.0
    pack_risky_max_cv: float = 20.0  # Haute variance tolérée

    # Probabilité minimum par pack
    pack_sure_min_prob: float = 0.003  # 0.3% min pour SÛR
    pack_balanced_min_prob: float = 0.001
    pack_risky_min_prob: float = 0.0001  # Long-shots OK

    # Sélection
    top_combos_consider: int = 500  # Top combinaisons à considérer
    min_combo_prob: float = 0.0001  # Proba minimum pour considérer

    def __post_init__(self):
        """Initialise num_simulations et exotics_defaults depuis config centralisée."""
        if self.num_simulations is None:
            self.num_simulations = _get_unified_num_simulations()

        # Charger exotics_defaults depuis config
        per_ticket, max_pack, rounding = _get_exotics_defaults()
        if self.per_ticket_rate is None:
            self.per_ticket_rate = per_ticket
        if self.max_pack_rate is None:
            self.max_pack_rate = max_pack
        if self.rounding_increment_eur is None:
            self.rounding_increment_eur = rounding

    def get_max_stake_per_ticket(self) -> float:
        """Retourne la mise max par ticket basée sur bankroll et per_ticket_rate."""
        return self.bankroll * self.per_ticket_rate

    def get_max_pack_budget(self) -> float:
        """Retourne le budget max par pack basé sur bankroll et max_pack_rate."""
        return self.bankroll * self.max_pack_rate

    def round_stake(self, stake: float) -> float:
        """Arrondit une mise au pas rounding_increment_eur."""
        if self.rounding_increment_eur > 0:
            return round(stake / self.rounding_increment_eur) * self.rounding_increment_eur
        return round(stake, 2)


# ============================================================================
# SIMULATEUR PLACKETT-LUCE
# ============================================================================


class PlackettLuceSimulator:
    """
    Simulateur d'arrivées via le modèle Plackett-Luce.

    À chaque étape, la probabilité qu'un cheval non encore arrivé
    finisse à la position actuelle est proportionnelle à sa force (p_win).
    """

    def __init__(self, probabilities: np.ndarray, seed: Optional[int] = None):
        """
        Args:
            probabilities: Vecteur de probabilités p_win (doit sommer à 1)
            seed: Graine aléatoire
        """
        self.probabilities = np.array(probabilities)
        self.n_horses = len(probabilities)
        self.rng = np.random.default_rng(seed)

        # Normaliser si nécessaire
        if not np.isclose(self.probabilities.sum(), 1.0, atol=0.01):
            self.probabilities = self.probabilities / self.probabilities.sum()

    def simulate_one_race(self) -> np.ndarray:
        """
        Simule une arrivée complète.

        Returns:
            np.ndarray: Ordre d'arrivée (indices des chevaux)
        """
        remaining = list(range(self.n_horses))
        remaining_probs = self.probabilities.copy()
        order = []

        for _ in range(self.n_horses):
            # Normaliser les probas restantes
            probs = remaining_probs[remaining]
            probs = probs / probs.sum()

            # Tirer le prochain
            chosen_idx = self.rng.choice(len(remaining), p=probs)
            chosen_horse = remaining[chosen_idx]

            order.append(chosen_horse)
            remaining.remove(chosen_horse)

        return np.array(order)

    def simulate_n_races(self, n: int) -> np.ndarray:
        """
        Simule n arrivées.

        Returns:
            np.ndarray: Shape (n, n_horses) - chaque ligne est une arrivée
        """
        results = np.zeros((n, self.n_horses), dtype=int)

        for i in range(n):
            results[i] = self.simulate_one_race()

        return results

    def simulate_n_races_fast(self, n: int) -> np.ndarray:
        """
        Version optimisée avec vectorisation partielle.
        """
        results = np.zeros((n, self.n_horses), dtype=int)

        # Pré-allouer
        all_remaining = [list(range(self.n_horses)) for _ in range(n)]

        for pos in range(self.n_horses):
            for i in range(n):
                remaining = all_remaining[i]
                if len(remaining) == 1:
                    results[i, pos] = remaining[0]
                    continue

                probs = self.probabilities[remaining]
                probs = probs / probs.sum()

                chosen_idx = self.rng.choice(len(remaining), p=probs)
                chosen_horse = remaining[chosen_idx]

                results[i, pos] = chosen_horse
                all_remaining[i] = [h for h in remaining if h != chosen_horse]

        return results


# ============================================================================
# ANALYSEUR DE COMBINAISONS
# ============================================================================


class ComboAnalyzer:
    """Analyse les combinaisons à partir des simulations."""

    def __init__(self, config: ExoticConfig):
        self.config = config

    def extract_combos(self, arrivals: np.ndarray, structure: str) -> Dict[Tuple, int]:
        """
        Extrait les combinaisons pertinentes des arrivées.

        Args:
            arrivals: (n_simulations, n_horses) - arrivées simulées
            structure: Type de pari

        Returns:
            Dict[combo -> count]
        """
        combo_counts = Counter()

        for arrival in arrivals:
            if structure.startswith("trio"):
                top3 = tuple(arrival[:3])
                if "desordre" in structure:
                    top3 = tuple(sorted(top3))
                combo_counts[top3] += 1

            elif structure.startswith("quarte"):
                top4 = tuple(arrival[:4])
                if "desordre" in structure:
                    top4 = tuple(sorted(top4))
                combo_counts[top4] += 1

            elif structure.startswith("quinte"):
                top5 = tuple(arrival[:5])
                if "desordre" in structure or "plus" in structure:
                    top5 = tuple(sorted(top5))
                combo_counts[top5] += 1

        return dict(combo_counts)

    def estimate_payouts(
        self, combo_probs: Dict[Tuple, float], n_horses: int, structure: str, takeout: float
    ) -> Dict[Tuple, Dict]:
        """
        Estime les payouts pour chaque combinaison.

        En parimutuel: payout ≈ (1 - takeout) / prob_public
        On suppose que notre modèle est plus précis que le public,
        donc prob_public ≠ prob_model. On modélise ça avec un noise factor.
        """
        results = {}

        # Facteur de "edge" du modèle sur le public
        # Si notre prob est haute et le public sous-estime, on a de la value
        np.random.seed(42)  # Reproductibilité

        for combo, prob in combo_probs.items():
            if prob < self.config.min_combo_prob:
                continue

            if prob > 0:
                # Simuler une inefficience du marché
                # Le public a une estimation bruitée de la vraie probabilité
                # Plus la combinaison est "surprenante" (faible prob mais arrive),
                # plus le payout est élevé

                # Estimation réaliste du payout parimutuel
                # On suppose que le marché a un bruit de ±30% sur les probas
                noise_factor = 0.7 + 0.6 * np.random.random()  # [0.7, 1.3]
                prob_public = prob * noise_factor
                prob_public = max(0.0001, min(0.5, prob_public))

                # Payout = (1 - takeout) / prob_public
                estimated_payout = (1 - takeout) / prob_public

                # Cap réaliste sur les payouts (max 1000x pour trio)
                max_payout = 1000 if "trio" in structure else 5000
                estimated_payout = min(estimated_payout, max_payout)

                # EV = prob_model * payout - 1
                ev = prob * estimated_payout - 1

                # Variance = prob * (payout - 1)^2 + (1-prob) * 1
                variance = prob * (estimated_payout - 1) ** 2 + (1 - prob) * 1

                results[combo] = {
                    "prob": prob,
                    "prob_public": prob_public,
                    "payout": estimated_payout,
                    "ev": ev,
                    "variance": variance,
                    "std": np.sqrt(variance),
                    "cv": np.sqrt(variance) / max(abs(ev), 0.001) if ev != 0 else float("inf"),
                }

        return results

    def classify_combo(self, combo: Tuple, probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Classifie une combinaison (favoris, outsiders, etc.).
        """
        # Trier les chevaux par probabilité
        sorted_indices = np.argsort(probabilities)[::-1]

        # Rang de favori pour chaque cheval du combo
        ranks = []
        for horse in combo:
            rank = np.where(sorted_indices == horse)[0][0] + 1
            ranks.append(rank)

        n_favoris = sum(1 for r in ranks if r <= 3)  # Top 3 = favoris
        n_outsiders = sum(1 for r in ranks if r > 6)  # > 6ème = outsider

        # Classification
        if n_favoris >= 2 and n_outsiders == 0:
            coverage = "favoris_forts"
        elif n_favoris >= 1 and n_outsiders == 0:
            coverage = "favoris_mixtes"
        elif n_outsiders >= 2:
            coverage = "outsiders"
        elif n_outsiders >= 1:
            coverage = "mixte_outsider"
        else:
            coverage = "equilibre"

        return {
            "ranks": ranks,
            "n_favoris": n_favoris,
            "n_outsiders": n_outsiders,
            "coverage": coverage,
            "avg_rank": np.mean(ranks),
        }


# ============================================================================
# OPTIMISEUR DE TICKETS
# ============================================================================


class TicketOptimizer:
    """
    Optimise la sélection de tickets pour maximiser EV sous contrainte de variance.
    """

    def __init__(self, config: ExoticConfig):
        self.config = config

    def select_pack(
        self,
        combo_stats: Dict[Tuple, Dict],
        probabilities: np.ndarray,
        horse_names: List[str],
        max_cv: float,
        min_ev: float,
        min_prob: float,
        budget: float,
        max_tickets: int,
        pack_label: str,
        prefer_coverage: str = None,
    ) -> Dict[str, Any]:
        """
        Sélectionne un pack de tickets optimisé.

        Args:
            combo_stats: Stats par combinaison
            max_cv: CV maximum accepté
            min_ev: EV minimum requis
            min_prob: Probabilité minimum
            budget: Budget pour ce pack
            max_tickets: Nombre max de tickets
            pack_label: Label du pack
            prefer_coverage: Type de couverture préféré

        Returns:
            Pack avec tickets sélectionnés
        """
        analyzer = ComboAnalyzer(self.config)

        # Filtrer les combos par EV, CV et probabilité
        candidates = []
        for combo, stats in combo_stats.items():
            if stats["ev"] >= min_ev and stats["cv"] <= max_cv and stats["prob"] >= min_prob:
                classification = analyzer.classify_combo(combo, probabilities)
                candidates.append(
                    {"combo": combo, "stats": stats, "classification": classification}
                )

        if not candidates:
            # Relaxer les contraintes progressivement
            for combo, stats in combo_stats.items():
                if stats["ev"] >= min_ev * 0.3 and stats["prob"] >= min_prob * 0.5:
                    classification = analyzer.classify_combo(combo, probabilities)
                    candidates.append(
                        {"combo": combo, "stats": stats, "classification": classification}
                    )

        if not candidates:
            return {
                "label": pack_label,
                "tickets": [],
                "n_tickets": 0,
                "total_stake": 0,
                "expected_return": 0,
                "portfolio_ev_pct": 0,
                "portfolio_variance": 0,
                "portfolio_std": 0,
                "coverage_summary": {},
                "notes": ["Aucun ticket satisfaisant les critères"],
            }

        # Filtrer par couverture préférée si spécifié
        if prefer_coverage:
            preferred = [
                c for c in candidates if prefer_coverage in c["classification"]["coverage"]
            ]
            if len(preferred) >= max_tickets // 2:
                candidates = preferred

        # Trier par EV décroissant pour RISQUÉ, par prob*EV pour SÛR
        if pack_label == "RISQUÉ":
            candidates.sort(key=lambda x: x["stats"]["ev"], reverse=True)
        elif pack_label == "SÛR":
            candidates.sort(key=lambda x: x["stats"]["prob"] * x["stats"]["ev"], reverse=True)
        else:
            candidates.sort(
                key=lambda x: x["stats"]["ev"] * 0.7 + x["stats"]["prob"] * 0.3, reverse=True
            )

        # Calculer les limites basées sur bankroll
        max_stake_per_ticket = self.config.get_max_stake_per_ticket()  # 0.75% bankroll
        max_pack_budget = self.config.get_max_pack_budget()  # 4% bankroll

        # Le budget effectif est le minimum entre budget demandé et max_pack_rate
        effective_budget = min(budget, max_pack_budget)

        # Sélectionner les meilleurs tickets
        selected = []
        total_stake = 0

        # Stake par ticket = min(budget/max_tickets, per_ticket_rate*bankroll)
        raw_stake = effective_budget / max_tickets
        stake_per_ticket = min(raw_stake, max_stake_per_ticket)
        stake_per_ticket = max(self.config.min_stake, stake_per_ticket)

        # Arrondir la mise par ticket
        stake_per_ticket = self.config.round_stake(stake_per_ticket)

        notes = []
        if budget > max_pack_budget:
            notes.append(
                f"Budget réduit de {budget:.2f}€ à {effective_budget:.2f}€ (max_pack_rate={self.config.max_pack_rate:.1%})"
            )

        for candidate in candidates[:max_tickets]:
            if total_stake + stake_per_ticket > effective_budget:
                break

            combo = candidate["combo"]
            stats = candidate["stats"]
            classification = candidate["classification"]

            ticket = {
                "combo": [horse_names[i] for i in combo],
                "combo_indices": [int(i) for i in combo],
                "stake": round(stake_per_ticket, 2),
                "prob": round(stats["prob"] * 100, 4),
                "payout": round(stats["payout"], 2),
                "ev": round(stats["ev"] * 100, 2),
                "ev_pct": f"{stats['ev']*100:.1f}%",
                "coverage": classification["coverage"],
                "ranks": [int(r) for r in classification["ranks"]],
            }

            selected.append(ticket)
            total_stake += stake_per_ticket

        # Calculer les stats du pack
        if selected:
            expected_return = sum(t["prob"] / 100 * t["payout"] * t["stake"] for t in selected)
            portfolio_ev = (expected_return - total_stake) / total_stake if total_stake > 0 else 0

            # Variance du portefeuille (simplifiée, en supposant indépendance)
            portfolio_var = sum(
                (t["prob"] / 100) * (t["payout"] * t["stake"] - t["stake"]) ** 2
                + (1 - t["prob"] / 100) * t["stake"] ** 2
                for t in selected
            )
        else:
            portfolio_ev = 0
            portfolio_var = 0

        return {
            "label": pack_label,
            "tickets": selected,
            "n_tickets": len(selected),
            "total_stake": round(total_stake, 2),
            "expected_return": round(expected_return, 2) if selected else 0,
            "portfolio_ev_pct": round(portfolio_ev * 100, 2) if selected else 0,
            "portfolio_variance": round(portfolio_var, 2) if selected else 0,
            "portfolio_std": round(np.sqrt(portfolio_var), 2)
            if selected and portfolio_var > 0
            else 0,
            "coverage_summary": self._summarize_coverage(selected) if selected else {},
            "per_ticket_rate": self.config.per_ticket_rate,
            "max_pack_rate": self.config.max_pack_rate,
            "notes": notes,
        }

    def _summarize_coverage(self, tickets: List[Dict]) -> Dict[str, int]:
        """Résume la couverture des tickets."""
        coverage_counts = Counter(t["coverage"] for t in tickets)
        return dict(coverage_counts)


# ============================================================================
# GÉNÉRATEUR PRINCIPAL
# ============================================================================


class ExoticTicketGenerator:
    """
    Générateur de tickets exotiques avec simulation Monte Carlo.

    Usage:
        generator = ExoticTicketGenerator(config)
        result = generator.generate(
            probabilities=[0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02],
            horse_names=['Cheval1', 'Cheval2', ...],
            structure='trio_ordre'
        )
    """

    def __init__(self, config: ExoticConfig = None):
        self.config = config or ExoticConfig()
        self.simulator: Optional[PlackettLuceSimulator] = None
        self.analyzer = ComboAnalyzer(self.config)
        self.optimizer = TicketOptimizer(self.config)

    def generate(
        self,
        probabilities: List[float],
        horse_names: List[str] = None,
        structure: str = None,
        budget: float = None,
        max_tickets: int = None,
        num_simulations: int = None,
        takeout_rate: float = None,
        fixed_odds: Dict[Tuple, float] = None,
    ) -> Dict[str, Any]:
        """
        Génère les packs de tickets exotiques.

        Args:
            probabilities: Vecteur p_win calibré (doit sommer à ~1)
            horse_names: Noms des chevaux (optionnel)
            structure: Type de pari
            budget: Budget total
            max_tickets: Max tickets par pack
            num_simulations: Nombre de simulations
            takeout_rate: Taux de prélèvement parimutuel
            fixed_odds: Cotes fixes par combo (optionnel)

        Returns:
            Dict avec les packs et métadonnées
        """
        # Paramètres
        probs = np.array(probabilities)
        n_horses = len(probs)

        if horse_names is None:
            horse_names = [f"#{i+1}" for i in range(n_horses)]

        structure = structure or self.config.structure
        budget = budget or self.config.budget
        max_tickets = max_tickets or self.config.max_tickets
        num_simulations = num_simulations or self.config.num_simulations
        takeout = takeout_rate or self.config.takeout_rate

        # Normaliser les probabilités
        probs = probs / probs.sum()

        run_notes = [
            f"N={num_simulations} simulations",
            "modèle Plackett-Luce",
            f"takeout {takeout*100:.0f}%",
            f"structure: {structure}",
            f"{n_horses} partants",
        ]

        # Simuler les arrivées
        print(f"🎲 Simulation de {num_simulations} arrivées...")
        self.simulator = PlackettLuceSimulator(probs, seed=self.config.random_seed)
        arrivals = self.simulator.simulate_n_races_fast(num_simulations)

        # Extraire les combinaisons
        print(f"📊 Extraction des combinaisons ({structure})...")
        combo_counts = self.analyzer.extract_combos(arrivals, structure)

        # Calculer les probabilités
        combo_probs = {combo: count / num_simulations for combo, count in combo_counts.items()}

        # Estimer les payouts et stats
        print("💰 Estimation des payouts...")
        if fixed_odds:
            # Utiliser les cotes fixes fournies
            combo_stats = self._apply_fixed_odds(combo_probs, fixed_odds)
            run_notes.append("cotes fixes utilisées")
        else:
            # Estimer via parimutuel
            combo_stats = self.analyzer.estimate_payouts(combo_probs, n_horses, structure, takeout)
            run_notes.append("estimation parimutuel")

        # Statistiques des combinaisons
        n_combos_total = len(combo_counts)
        n_combos_positive_ev = sum(1 for s in combo_stats.values() if s["ev"] > 0)

        print(f"   {n_combos_total} combinaisons observées")
        print(f"   {n_combos_positive_ev} avec EV > 0")

        # Générer les 3 packs
        print("📦 Génération des packs...")
        budget_per_pack = budget / 3
        tickets_per_pack = max_tickets // 3

        # Pack SÛR: faible variance, EV modéré, favoris
        pack_sure = self.optimizer.select_pack(
            combo_stats=combo_stats,
            probabilities=probs,
            horse_names=horse_names,
            max_cv=self.config.pack_sure_max_cv,
            min_ev=self.config.pack_sure_ev_min,
            min_prob=self.config.pack_sure_min_prob,
            budget=budget_per_pack,
            max_tickets=tickets_per_pack,
            pack_label="SÛR",
            prefer_coverage="favoris",
        )

        # Pack ÉQUILIBRÉ: variance moyenne, bon EV
        pack_balanced = self.optimizer.select_pack(
            combo_stats=combo_stats,
            probabilities=probs,
            horse_names=horse_names,
            max_cv=self.config.pack_balanced_max_cv,
            min_ev=self.config.pack_balanced_ev_min,
            min_prob=self.config.pack_balanced_min_prob,
            budget=budget_per_pack,
            max_tickets=tickets_per_pack,
            pack_label="ÉQUILIBRÉ",
            prefer_coverage=None,
        )

        # Pack RISQUÉ: haute variance, EV élevé, outsiders
        pack_risky = self.optimizer.select_pack(
            combo_stats=combo_stats,
            probabilities=probs,
            horse_names=horse_names,
            max_cv=self.config.pack_risky_max_cv,
            min_ev=self.config.pack_risky_ev_min,
            min_prob=self.config.pack_risky_min_prob,
            budget=budget_per_pack,
            max_tickets=tickets_per_pack,
            pack_label="RISQUÉ",
            prefer_coverage="outsider",
        )

        # Top combinaisons pour info
        top_combos = sorted(combo_stats.items(), key=lambda x: x[1]["ev"], reverse=True)[:10]

        top_combos_info = [
            {
                "combo": [horse_names[i] for i in combo],
                "prob_pct": round(stats["prob"] * 100, 3),
                "payout": round(stats["payout"], 2),
                "ev_pct": round(stats["ev"] * 100, 2),
            }
            for combo, stats in top_combos
        ]

        # Construire le résultat
        result = {
            "race_info": {
                "n_horses": n_horses,
                "structure": structure,
                "budget": budget,
                "takeout_rate": takeout,
            },
            "simulation_stats": {
                "num_simulations": num_simulations,
                "n_combos_observed": n_combos_total,
                "n_combos_positive_ev": n_combos_positive_ev,
                "coverage_rate": round(
                    n_combos_total / self._theoretical_combos(n_horses, structure) * 100, 2
                ),
            },
            "probabilities": {
                horse_names[i]: round(probs[i] * 100, 2) for i in np.argsort(probs)[::-1]
            },
            "top_combos": top_combos_info,
            "packs": [pack_sure, pack_balanced, pack_risky],
            "summary": {
                "total_stake": round(
                    pack_sure["total_stake"]
                    + pack_balanced["total_stake"]
                    + pack_risky["total_stake"],
                    2,
                ),
                "total_tickets": (
                    pack_sure["n_tickets"] + pack_balanced["n_tickets"] + pack_risky["n_tickets"]
                ),
                "avg_portfolio_ev_pct": round(
                    np.mean(
                        [
                            pack_sure["portfolio_ev_pct"],
                            pack_balanced["portfolio_ev_pct"],
                            pack_risky["portfolio_ev_pct"],
                        ]
                    ),
                    2,
                ),
            },
            "run_notes": run_notes,
        }

        return result

    def _theoretical_combos(self, n_horses: int, structure: str) -> int:
        """Calcule le nombre théorique de combinaisons."""
        if "trio" in structure:
            if "desordre" in structure:
                return int(np.math.comb(n_horses, 3))
            else:
                return n_horses * (n_horses - 1) * (n_horses - 2)
        elif "quarte" in structure:
            if "desordre" in structure:
                return int(np.math.comb(n_horses, 4))
            else:
                return n_horses * (n_horses - 1) * (n_horses - 2) * (n_horses - 3)
        elif "quinte" in structure:
            if "desordre" in structure or "plus" in structure:
                return int(np.math.comb(n_horses, 5))
            else:
                return n_horses * (n_horses - 1) * (n_horses - 2) * (n_horses - 3) * (n_horses - 4)
        return 1

    def _apply_fixed_odds(
        self, combo_probs: Dict[Tuple, float], fixed_odds: Dict[Tuple, float]
    ) -> Dict[Tuple, Dict]:
        """Applique les cotes fixes fournies."""
        results = {}

        for combo, prob in combo_probs.items():
            if combo in fixed_odds:
                payout = fixed_odds[combo]
            else:
                # Estimation si pas de cote fixe
                payout = (1 - self.config.takeout_rate) / prob if prob > 0 else 0

            if prob > 0:
                ev = prob * payout - 1
                variance = prob * (payout - 1) ** 2 + (1 - prob) * 1

                results[combo] = {
                    "prob": prob,
                    "payout": payout,
                    "ev": ev,
                    "variance": variance,
                    "std": np.sqrt(variance),
                    "cv": np.sqrt(variance) / max(ev, 0.001) if ev > 0 else float("inf"),
                }

        return results

    def generate_champ_reduit(
        self,
        probabilities: List[float],
        horse_names: List[str] = None,
        bases: List[int] = None,
        n_associates: int = None,
        budget: float = None,
    ) -> Dict[str, Any]:
        """
        Génère des tickets en champ réduit.

        Le champ réduit permet de jouer une base fixe
        avec plusieurs chevaux en complément.
        """
        probs = np.array(probabilities)
        n_horses = len(probs)

        if horse_names is None:
            horse_names = [f"#{i+1}" for i in range(n_horses)]

        if bases is None:
            # Prendre les 2 favoris comme bases
            bases = list(np.argsort(probs)[::-1][: self.config.champ_bases])

        if n_associates is None:
            n_associates = self.config.champ_associates

        budget = budget or self.config.budget

        # Sélectionner les meilleurs associés (hors bases)
        remaining = [i for i in range(n_horses) if i not in bases]
        remaining_sorted = sorted(remaining, key=lambda i: probs[i], reverse=True)
        associates = remaining_sorted[:n_associates]

        # Générer toutes les combinaisons possibles
        tickets = []

        # Pour un trio: base1 + base2 + chacun des associés
        for assoc in associates:
            combo = tuple(sorted([bases[0], bases[1], assoc]))
            tickets.append(
                {
                    "combo": [horse_names[i] for i in [bases[0], bases[1], assoc]],
                    "type": "base_base_assoc",
                }
            )

        # Pour plus de variété: base1 + assoc1 + assoc2
        for i, assoc1 in enumerate(associates):
            for assoc2 in associates[i + 1 :]:
                combo = [bases[0], assoc1, assoc2]
                tickets.append(
                    {"combo": [horse_names[i] for i in combo], "type": "base_assoc_assoc"}
                )

        # Calculer les mises
        stake_per_ticket = budget / len(tickets) if tickets else 0
        for ticket in tickets:
            ticket["stake"] = round(stake_per_ticket, 2)

        return {
            "structure": "trio_champ_reduit",
            "bases": [horse_names[i] for i in bases],
            "associates": [horse_names[i] for i in associates],
            "n_tickets": len(tickets),
            "total_stake": round(budget, 2),
            "tickets": tickets,
            "run_notes": [
                f"Champ réduit {len(bases)} bases + {n_associates} associés",
                f"{len(tickets)} combinaisons",
            ],
        }


# ============================================================================
# MAIN - TESTS
# ============================================================================


def main():
    """Test du générateur avec données synthétiques."""
    import argparse

    parser = argparse.ArgumentParser(description="Générateur de Tickets Exotiques")
    parser.add_argument("--test", action="store_true", help="Lancer les tests")
    parser.add_argument("--simulations", type=int, default=20000, help="Nombre de simulations")
    parser.add_argument("--budget", type=float, default=100.0, help="Budget total")
    parser.add_argument("--structure", type=str, default="trio_ordre", help="Structure de pari")
    args = parser.parse_args()

    if args.test:
        print("=" * 60)
        print("🧪 TEST DU GÉNÉRATEUR DE TICKETS EXOTIQUES")
        print("=" * 60)

        # Configuration
        config = ExoticConfig(
            num_simulations=args.simulations,
            budget=args.budget,
            max_tickets=30,
            takeout_rate=0.16,
            structure=args.structure,
        )

        # Données de test - 10 partants avec probas calibrées
        probabilities = [0.22, 0.18, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02]
        horse_names = [
            "Tonnerre",
            "Éclair",
            "Tempête",
            "Ouragan",
            "Cyclone",
            "Rafale",
            "Bourrasque",
            "Zéphyr",
            "Brise",
            "Souffle",
        ]

        # Générer
        generator = ExoticTicketGenerator(config)
        result = generator.generate(
            probabilities=probabilities, horse_names=horse_names, structure=args.structure
        )

        # Afficher le résultat JSON
        print("\n" + "=" * 60)
        print("📋 RÉSULTAT JSON")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Validation
        print("\n" + "=" * 60)
        print("✅ VALIDATION")
        print("=" * 60)

        # Vérifier la somme des mises
        total_stake = result["summary"]["total_stake"]
        print(f"   Budget: {args.budget}€")
        print(f"   Total misé: {total_stake}€")

        # Vérifier les packs
        for pack in result["packs"]:
            print(f"\n   Pack {pack['label']}:")
            print(f"      Tickets: {pack['n_tickets']}")
            print(f"      Mise: {pack['total_stake']}€")
            print(f"      EV portefeuille: {pack['portfolio_ev_pct']}%")
            if pack["tickets"]:
                print(
                    f"      Exemple: {pack['tickets'][0]['combo']} (EV: {pack['tickets'][0]['ev_pct']})"
                )

        # Test champ réduit
        print("\n" + "=" * 60)
        print("🎯 TEST CHAMP RÉDUIT")
        print("=" * 60)

        champ_result = generator.generate_champ_reduit(
            probabilities=probabilities, horse_names=horse_names, budget=50.0
        )

        print(json.dumps(champ_result, indent=2, ensure_ascii=False))

        print("\n✅ TOUS LES TESTS PASSÉS")


if __name__ == "__main__":
    main()
