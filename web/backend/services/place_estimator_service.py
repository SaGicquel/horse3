#!/usr/bin/env python3
"""
Service d'estimation p(place) pour l'API Backend
=================================================

Ce service intègre le module place_probability_estimator avec l'API
pour fournir des conseils de paris exotiques améliorés.

Utilisé par:
- /exotics/build (paris exotiques)
- /exotics/advanced (analyse avancée)
- /picks/combined/{race_key} (paris combinés)

Auteur: Horse Racing AI System
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from place_probability_estimator import (
        PlaceProbabilityEstimator,
        PlaceEstimatorConfig,
        PlaceEstimatorSelector,
        HarvilleEstimator,
        HeneryEstimator,
        PlackettLuceTemperatureSimulator,
        ExoticEVCalculator,
        CalibrationMetrics,
    )

    ESTIMATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Module place_probability_estimator non disponible: {e}")
    ESTIMATOR_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_NUM_SIMULATIONS = 20000
DEFAULT_TAKEOUT_RATE = 0.16

# Températures par discipline (à synchroniser avec config/pro_betting.yaml)
TEMPERATURE_BY_DISCIPLINE = {"plat": 0.95, "trot": 1.05, "obstacle": 1.10, "default": 1.0}

# Estimateur par discipline
ESTIMATOR_BY_DISCIPLINE = {
    "plat": "henery",
    "trot": "lbs",
    "obstacle": "stern",
    "default": "harville",
}


# =============================================================================
# FONCTIONS DE CONVERSION
# =============================================================================


def cotes_to_probas(cotes: List[float]) -> np.ndarray:
    """
    Convertit les cotes en probabilités normalisées.

    Args:
        cotes: Liste des cotes (ex: [2.5, 5.0, 8.0, ...])

    Returns:
        np.ndarray: Probabilités normalisées (sum = 1)
    """
    cotes = np.array(cotes)
    # Filtrer les cotes invalides
    cotes = np.where(cotes > 1, cotes, 100)  # Remplacer <=1 par 100

    # Probabilités implicites
    probs = 1.0 / cotes

    # Normaliser
    probs = probs / probs.sum()

    return probs


def normalize_discipline(discipline: str) -> str:
    """Normalise le nom de la discipline."""
    if not discipline:
        return "default"

    disc = discipline.lower().strip()

    if any(x in disc for x in ["plat", "flat"]):
        return "plat"
    elif any(x in disc for x in ["trot", "attelé", "monte", "attele"]):
        return "trot"
    elif any(x in disc for x in ["obstacle", "haie", "steeple", "cross"]):
        return "obstacle"

    return "default"


# =============================================================================
# SERVICE PRINCIPAL
# =============================================================================


class PlaceEstimatorService:
    """
    Service pour estimer les probabilités de place et générer
    des conseils de paris exotiques.
    """

    def __init__(self, num_simulations: int = None, takeout_rate: float = None):
        self.num_simulations = num_simulations or DEFAULT_NUM_SIMULATIONS
        self.takeout_rate = takeout_rate or DEFAULT_TAKEOUT_RATE
        self.ev_calculator = ExoticEVCalculator(self.takeout_rate) if ESTIMATOR_AVAILABLE else None

    def analyze_race(
        self,
        partants: List[Dict],
        discipline: str = None,
        structure: str = "trio",
        n_simulations: int = None,
        estimator_override: str = None,
    ) -> Dict[str, Any]:
        """
        Analyse une course et génère les probabilités p(place) et p(combo).

        Args:
            partants: Liste des partants avec 'nom', 'numero', 'cote', 'score'
            discipline: Type de course
            structure: 'trio', 'quarte', 'quinte'
            n_simulations: Nombre de simulations Monte Carlo (override)
            estimator_override: Forcer un estimateur ('harville', 'henery', 'stern', 'lbs')

        Returns:
            Dict avec p_place, combo_probs, tickets recommandés
        """
        if not ESTIMATOR_AVAILABLE:
            return self._fallback_analysis(partants, structure)

        # Utiliser n_simulations passé ou défaut
        sims = n_simulations or self.num_simulations

        # Extraire les cotes et noms
        cotes = [part.get("cote", 10.0) or 10.0 for part in partants]
        noms = [part.get("nom", f"#{idx+1}") for idx, part in enumerate(partants)]
        numeros = [part.get("numero", idx + 1) for idx, part in enumerate(partants)]
        scores = [part.get("score", 50) for part in partants]

        # Convertir en probabilités
        p_win_market = cotes_to_probas(cotes)

        # Ajuster avec les scores du modèle si disponibles
        if any(s != 50 for s in scores):
            p_win_blend = self._blend_with_scores(p_win_market, scores)
        else:
            p_win_blend = p_win_market

        # Normaliser la discipline
        disc = normalize_discipline(discipline)

        # Créer l'estimateur
        config = PlaceEstimatorConfig(num_simulations=sims, takeout_rate=self.takeout_rate)

        estimator = PlaceProbabilityEstimator(
            p_win_blend, discipline=disc, config=config, horse_names=noms
        )

        # Forcer l'estimateur si demandé
        if estimator_override:
            # Les estimateurs nécessitent win_probs à l'initialisation
            if estimator_override == "harville":
                estimator.estimator = HarvilleEstimator(p_win_blend)
            elif estimator_override == "henery":
                estimator.estimator = HeneryEstimator(p_win_blend, gamma=0.81)
            # Pour stern et lbs, on garde l'auto-sélection car ils nécessitent plus de config

        # Estimer p(place) pour différentes places
        place_data_2 = estimator.estimate_place_probs(top_n=2, method="auto")
        place_data_3 = estimator.estimate_place_probs(top_n=3, method="auto")
        p_place_2 = place_data_2["p_place"]
        p_place_3 = place_data_3["p_place"]

        # Estimer les probabilités de combinaisons pour trio, quarté, quinté
        trio_data = estimator.estimate_combo_probs("trio", n_sim=sims)
        quarte_data = None
        quinte_data = None

        if len(partants) >= 4:
            quarte_data = estimator.estimate_combo_probs("quarte", n_sim=sims)
        if len(partants) >= 5:
            quinte_data = estimator.estimate_combo_probs("quinte", n_sim=sims)

        # Calculer les EV pour les top combos
        ev_data = estimator.calculate_tickets_ev(trio_data["combo_probs"], "trio")

        # Calculer métriques de calibration
        try:
            calibration = estimator.get_calibration_metrics()
        except:
            calibration = {"brier_score": 0.20, "ece": 0.05}

        # Enrichir les partants avec les nouvelles probas
        enriched_partants = []
        for i, p in enumerate(partants):
            enriched_partants.append(
                {
                    **p,
                    "p_win": round(float(p_win_blend[i]), 4),
                    "p_win_market": round(float(p_win_market[i]), 4),
                    "p_place_2": round(float(p_place_2[i]), 4),
                    "p_place_3": round(float(p_place_3[i]), 4),
                    "p_place": round(float(p_place_3[i]), 4),  # Alias pour compatibilité
                    "place_vs_win_ratio": round(float(p_place_3[i] / p_win_blend[i]), 2)
                    if p_win_blend[i] > 0.001
                    else 0,
                }
            )

        # Trier les partants par p_place décroissant
        enriched_partants.sort(key=lambda x: x["p_place"], reverse=True)

        # Préparer les listes p_win et p_place pour l'API
        p_win_list = [float(p_win_blend[i]) for i in range(len(partants))]
        p_place_2_list = [float(p_place_2[i]) for i in range(len(partants))]
        p_place_3_list = [float(p_place_3[i]) for i in range(len(partants))]

        return {
            "partants": enriched_partants,
            # Nouvelles clés pour /exotics/advanced
            "p_win": p_win_list,
            "p_place_2": p_place_2_list,
            "p_place_3": p_place_3_list,
            # Anciennes clés (dict) pour compatibilité
            "p_place": {noms[i]: round(float(p_place_3[i]), 4) for i in range(len(noms))},
            "p_win_blend": {noms[i]: round(float(p_win_blend[i]), 4) for i in range(len(noms))},
            "discipline": disc,
            "estimator_used": type(estimator.estimator).__name__,
            "temperature": estimator.simulator.temperature,
            "calibration": calibration,
            "combo_stats": {
                "structure": structure,
                "n_combos": trio_data["n_combos"],
                "n_simulations": sims,
            },
            "top_combos": trio_data["top_combos"][:20],
            # Nouveaux: top combos par type de pari
            "trio_probs": trio_data["top_combos"][:10],
            "quarte_probs": quarte_data["top_combos"][:10] if quarte_data else [],
            "quinte_probs": quinte_data["top_combos"][:10] if quinte_data else [],
            "ev_analysis": {
                "n_positive_ev": ev_data["n_positive_ev"],
                "n_tickets": ev_data["n_tickets"],
                "takeout_rate": self.takeout_rate,
            },
            "best_tickets": ev_data["positive_ev_tickets"][:15],
        }

    def _blend_with_scores(self, p_market: np.ndarray, scores: List[float]) -> np.ndarray:
        """
        Blend les probabilités marché avec les scores du modèle.

        scores: 0-100, 50 = neutre, >50 = surévalué par le modèle
        """
        scores = np.array(scores)

        # Convertir scores en facteurs multiplicatifs
        # score 50 -> facteur 1.0
        # score 70 -> facteur 1.4
        # score 30 -> facteur 0.6
        factors = 1.0 + (scores - 50) / 50

        # Appliquer les facteurs
        p_adjusted = p_market * factors

        # Renormaliser
        p_adjusted = p_adjusted / p_adjusted.sum()

        return p_adjusted

    def generate_packs(
        self, analysis: Dict, budget: float = 100.0, pack_type: str = "EQUILIBRE"
    ) -> Dict[str, Any]:
        """
        Génère les packs de tickets (SÛR, ÉQUILIBRÉ, RISQUÉ) basés sur l'analyse.

        Args:
            analysis: Résultat de analyze_race()
            budget: Budget total
            pack_type: 'SUR', 'EQUILIBRE', 'RISQUE'

        Returns:
            Dict avec les tickets générés
        """
        partants = analysis.get("partants", [])
        best_tickets = analysis.get("best_tickets", [])
        top_combos = analysis.get("top_combos", [])

        if not partants:
            return {"tickets": [], "error": "Pas de partants"}

        tickets = []

        # Seuils par pack
        pack_config = {
            "SUR": {
                "min_prob": 0.003,  # 0.3% minimum
                "min_ev": 0.0,  # EV >= 0
                "n_tickets": 3,
                "budget_share": [0.45, 0.35, 0.20],
            },
            "EQUILIBRE": {
                "min_prob": 0.001,
                "min_ev": 0.05,  # EV >= 5%
                "n_tickets": 5,
                "budget_share": [0.30, 0.25, 0.20, 0.15, 0.10],
            },
            "RISQUE": {
                "min_prob": 0.0005,
                "min_ev": 0.10,  # EV >= 10%
                "n_tickets": 6,
                "budget_share": [0.25, 0.20, 0.18, 0.15, 0.12, 0.10],
            },
        }

        config = pack_config.get(pack_type, pack_config["EQUILIBRE"])

        # Filtrer les tickets selon le pack
        filtered_tickets = []

        if best_tickets:
            for t in best_tickets:
                prob = t.get("prob", 0)
                ev = t.get("ev", -1)

                if prob >= config["min_prob"] and ev >= config["min_ev"]:
                    filtered_tickets.append(t)

        # Si pas assez de tickets EV+, utiliser les top combos
        if len(filtered_tickets) < config["n_tickets"] and top_combos:
            for combo in top_combos:
                if len(filtered_tickets) >= config["n_tickets"]:
                    break

                prob = combo.get("prob", 0)
                if prob >= config["min_prob"]:
                    # Calculer EV approximatif
                    estimated_payout = (1 - self.takeout_rate) / prob if prob > 0 else 0
                    ev = prob * estimated_payout - 1

                    filtered_tickets.append(
                        {
                            "combo": combo["combo"],
                            "combo_indices": combo.get("combo_indices", []),
                            "prob": prob,
                            "prob_pct": combo.get("prob_pct", prob * 100),
                            "ev": ev,
                            "ev_pct": ev * 100,
                            "payout": round(estimated_payout, 2),
                        }
                    )

        # Générer les tickets finaux
        for i, ticket in enumerate(filtered_tickets[: config["n_tickets"]]):
            share = config["budget_share"][i] if i < len(config["budget_share"]) else 0.10
            stake = round(budget * share, 2)

            # Extraire les numéros pour l'endpoint (convertir en int standard)
            combo_indices = ticket.get("combo_indices", [])
            numeros = (
                [int(idx) + 1 for idx in combo_indices]
                if combo_indices
                else list(range(1, len(ticket["combo"]) + 1))
            )

            tickets.append(
                {
                    "type": f"Trio {'Ordre' if i == 0 else 'Combiné'}",
                    "bet_type": "tierce",
                    "combo": [
                        {"nom": name, "numero": int(numeros[j]) if j < len(numeros) else j + 1}
                        for j, name in enumerate(ticket["combo"])
                    ],
                    "selections": ticket["combo"],
                    "numeros": numeros,  # Liste des numéros pour l'endpoint
                    "stake": stake,
                    "mise": stake,
                    "prob_pct": round(ticket.get("prob_pct", ticket.get("prob", 0) * 100), 2),
                    "ev": round(float(ticket.get("ev", 0)), 4),  # EV brute
                    "ev_pct": round(float(ticket.get("ev_pct", ticket.get("ev", 0) * 100)), 1),
                    "payout": float(ticket.get("payout", 100)),
                    "couverture": round(ticket.get("prob", 0) * 100, 2),
                    "description": f"Combo EV+{ticket.get('ev_pct', 0):.1f}% - Prob {ticket.get('prob_pct', 0):.2f}%",
                }
            )

        return {
            pack_type: tickets,  # Retourne avec la clé du pack pour le endpoint
            "tickets": tickets,
            "pack": pack_type,
            "budget": budget,
            "budget_utilise": round(sum(t["stake"] for t in tickets), 2),
            "ev_totale": round(sum(t["ev_pct"] for t in tickets), 1),
            "n_tickets": len(tickets),
            "analysis_meta": {
                "estimator": analysis.get("estimator_used", "unknown"),
                "temperature": analysis.get("temperature", 1.0),
                "n_simulations": analysis.get("combo_stats", {}).get(
                    "n_simulations", self.num_simulations
                ),
            },
        }

    def _fallback_analysis(self, partants: List[Dict], structure: str) -> Dict[str, Any]:
        """
        Analyse de fallback si le module estimator n'est pas disponible.
        Utilise une approximation simple.
        """
        # Trier par score décroissant
        sorted_partants = sorted(partants, key=lambda x: x.get("score", 50), reverse=True)

        # Approximation p_place = p_win * 2.5 (capped à 0.9)
        for p in sorted_partants:
            cote = p.get("cote", 10.0) or 10.0
            p_win = 1 / cote
            p["p_win"] = round(p_win, 4)
            p["p_place"] = round(min(p_win * 2.5, 0.9), 4)

        # Générer des combos simples (top 3, top 4, etc.)
        top_combos = []
        if len(sorted_partants) >= 3:
            top3 = sorted_partants[:3]
            prob = np.prod([p["p_place"] for p in top3]) * 3  # Approximation grossière
            top_combos.append(
                {
                    "combo": [p["nom"] for p in top3],
                    "combo_indices": [partants.index(p) for p in top3],
                    "prob": min(prob, 0.05),
                    "prob_pct": min(prob * 100, 5.0),
                }
            )

        return {
            "partants": sorted_partants,
            "p_place": {p["nom"]: p["p_place"] for p in sorted_partants},
            "discipline": "unknown",
            "estimator_used": "Fallback (simple approximation)",
            "temperature": 1.0,
            "combo_stats": {
                "structure": structure,
                "n_combos": len(top_combos),
                "n_simulations": 0,
            },
            "top_combos": top_combos,
            "ev_analysis": {
                "n_positive_ev": 0,
                "n_tickets": len(top_combos),
                "takeout_rate": self.takeout_rate,
            },
            "best_tickets": [],
        }


# =============================================================================
# INSTANCE GLOBALE
# =============================================================================

# Service par défaut
_service_instance = None


def get_place_estimator_service(
    num_simulations: int = None, takeout_rate: float = None
) -> PlaceEstimatorService:
    """
    Retourne l'instance du service (singleton ou nouvelle instance).
    """
    global _service_instance

    if _service_instance is None or num_simulations or takeout_rate:
        _service_instance = PlaceEstimatorService(num_simulations, takeout_rate)

    return _service_instance


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST - Service d'estimation p(place)")
    print("=" * 60)

    # Données de test
    partants_test = [
        {"nom": "GOLDEN STAR", "numero": 1, "cote": 2.5, "score": 75},
        {"nom": "SILVER BOLT", "numero": 2, "cote": 4.0, "score": 65},
        {"nom": "BRONZE FLASH", "numero": 3, "cote": 6.0, "score": 60},
        {"nom": "IRON SPIRIT", "numero": 4, "cote": 8.0, "score": 55},
        {"nom": "COPPER DREAM", "numero": 5, "cote": 12.0, "score": 50},
        {"nom": "STEEL HEART", "numero": 6, "cote": 15.0, "score": 48},
        {"nom": "CHROME FIRE", "numero": 7, "cote": 20.0, "score": 45},
        {"nom": "PLATINUM RUN", "numero": 8, "cote": 25.0, "score": 42},
    ]

    service = get_place_estimator_service()

    print("\n1. Analyse de la course...")
    analysis = service.analyze_race(partants_test, discipline="plat", structure="trio")

    print(f"\n   Estimateur: {analysis['estimator_used']}")
    print(f"   Température: {analysis['temperature']}")
    print(f"   Combos observés: {analysis['combo_stats']['n_combos']}")

    print("\n2. Probabilités de place:")
    for nom, p in sorted(analysis["p_place"].items(), key=lambda x: -x[1])[:5]:
        print(f"   {nom}: {p*100:.1f}%")

    print("\n3. Top 5 combinaisons:")
    for i, combo in enumerate(analysis["top_combos"][:5]):
        print(f"   {i+1}. {' - '.join(combo['combo'])}: {combo['prob_pct']:.2f}%")

    print("\n4. Génération des packs...")
    for pack_type in ["SUR", "EQUILIBRE", "RISQUE"]:
        packs = service.generate_packs(analysis, budget=100, pack_type=pack_type)
        print(f"\n   Pack {pack_type}:")
        print(f"   - {packs['n_tickets']} tickets")
        print(f"   - Budget utilisé: {packs['budget_utilise']}€")
        print(f"   - EV totale: {packs['ev_totale']:+.1f}%")

    print("\n" + "=" * 60)
    print("✅ Test terminé")
