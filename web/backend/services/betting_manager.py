"""
Betting Manager - Gestion de Bankroll & Calcul de Mises (Kelly)
===============================================================
Orchestre le calcul des mises optimales en fonction des prédictions,
des cotes et de la stratégie de risque (Kelly Fractionné).
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# Import des calculs mathématiques existants
try:
    from pari_math import kelly_fraction_raw
except ImportError:
    # Fallback si pari_math n'est pas dispo (ex: test isolé)
    def kelly_fraction_raw(p: float, odds: float) -> float:
        """Kelly criterion: f* = (bp - q) / b"""
        if odds <= 1:
            return 0.0
        b = odds - 1.0
        q = 1.0 - p
        return max(0.0, (b * p - q) / b)


logger = logging.getLogger(__name__)


@dataclass
class BetSuggestion:
    cheval_id: str
    numero: int
    nom: str
    type_pari: str  # WIN, PLACE
    mise_conseillee: float
    pourcentage_bankroll: float
    kelly_fraction: float
    value_edge: float
    confidence: float
    strategy: str


class BettingManager:
    def __init__(self, bankroll: float = 1000.0, strategy: str = "balanced"):
        """
        Args:
            bankroll: Montant total disponible (défaut: 1000.0)
            strategy: 'conservative' (Kelly/8), 'balanced' (Kelly/4), 'aggressive' (Kelly/2)
        """
        self.bankroll = float(os.getenv("BETTING_BANKROLL", bankroll))
        self.strategy = os.getenv("BETTING_STRATEGY", strategy).lower()
        self.min_stake = float(os.getenv("BETTING_MIN_STAKE", 1.0))
        self.max_stake_pct = float(
            os.getenv("BETTING_MAX_STAKE_PCT", 0.05)
        )  # Max 5% bankroll per bet

        # Kelly dividers per strategy
        self.strategies = {
            "safe": 8.0,
            "conservative": 8.0,
            "balanced": 4.0,
            "aggressive": 2.0,
            "full": 1.0,
        }

        logger.info(f"💰 BettingManager init: Bankroll={self.bankroll}€, Strategy={self.strategy}")

    def calculate_stakes(
        self, predictions: List[Dict[str, Any]], confidence_score: float = 1.0
    ) -> List[BetSuggestion]:
        """
        Calcule les mises optimales pour une liste de prédictions.

        Args:
            predictions: Liste de dicts {'cheval_id', 'numero', 'nom', 'prob_model', 'cote_sp'}
            confidence_score: Score de confiance global du superviseur (0-1) pour moduler la mise.
        """
        suggestions = []

        divider = self.strategies.get(self.strategy, 4.0)

        for pred in predictions:
            prob = pred.get("prob_model", 0.0)
            odds = pred.get("cote_sp", 0.0)

            if prob <= 0 or odds <= 1.0:
                continue

            # Calcul Kelly pur
            # kelly_fraction_raw attend les cotes brutes (decimal odds)
            kelly = kelly_fraction_raw(prob, odds)
            # print(f"DEBUG_INTERNAL: p={prob}, odds={odds}, kelly={kelly}")

            if float(kelly) <= 0:
                continue

            # Application Fractionnement + Modulation Confiance
            # f_star = (Kelly / divider) * confidence
            adjusted_fraction = (kelly / divider) * confidence_score

            # Limites de sécurité (Max Stake %)
            capped_fraction = min(adjusted_fraction, self.max_stake_pct)

            # Calcul montant
            stake = self.bankroll * capped_fraction

            # Arrondi (pas de centimes bizarres, pas de mise < min)
            if stake < self.min_stake:
                # Si en dessous du min, on ne parie pas (ou on force le min si value très forte? Non, on skip)
                continue

            stake = round(stake, 1)  # Arrondi à 10 centimes

            # Calcul Value Edge pour info
            implied_prob = 1.0 / odds
            edge = prob - implied_prob

            suggestions.append(
                BetSuggestion(
                    cheval_id=pred.get("cheval_id", "unknown"),
                    numero=pred.get("numero", 0),
                    nom=pred.get("nom", "Unknown"),
                    type_pari="WIN",  # Pour l'instant focus WIN
                    mise_conseillee=stake,
                    pourcentage_bankroll=round(capped_fraction * 100, 2),
                    kelly_fraction=round(kelly, 4),
                    value_edge=round(edge, 4),
                    confidence=confidence_score,
                    strategy=self.strategy,
                )
            )

        return suggestions

    def get_bankroll_stats(self) -> Dict[str, float]:
        return {
            "total": self.bankroll,
            "max_stake": self.bankroll * self.max_stake_pct,
            "min_stake": self.min_stake,
        }
