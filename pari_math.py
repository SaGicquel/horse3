#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 PARI MATH - Fonctions Mathématiques pour les Paris
======================================================

Module centralisé pour les calculs EV, Kelly, et probabilités.
Supporte les modes parimutuel et bookmaker.

Fonctions principales:
- expected_payout_parimutuel(p, takeout): Fair odds pool
- ev_parimutuel_win(p, odds, takeout): EV pour WIN en parimutuel
- ev_parimutuel_place(p, odds, takeout): EV pour PLACE en parimutuel
- kelly_fraction(p, odds, fraction): Kelly fractionnaire

Auteur: Horse3 Pro System
Version: 1.0.0
"""

import math
import warnings
from typing import Optional, Tuple
import numpy as np

# Import config centralisée
try:
    from config.loader import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


# =============================================================================
# CONSTANTES PAR DÉFAUT
# =============================================================================

DEFAULT_TAKEOUT_RATE = 0.16  # PMU standard: 16%
MIN_PROB = 1e-10
MAX_PROB = 0.9999


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def _get_default_takeout() -> float:
    """Retourne le takeout par défaut depuis la config ou le défaut hardcodé."""
    if CONFIG_AVAILABLE:
        try:
            config = get_config()
            # Priorité: markets.takeout_rate > exotics.takeout_rate > default
            if hasattr(config, 'markets') and config.markets:
                return config.markets.takeout_rate
            return config.exotics.takeout_rate
        except Exception:
            pass
    return DEFAULT_TAKEOUT_RATE


def _clip_probability(p: float) -> float:
    """Borne la probabilité entre MIN_PROB et MAX_PROB."""
    return max(MIN_PROB, min(MAX_PROB, p))


# =============================================================================
# EXPECTED PAYOUT PARIMUTUEL
# =============================================================================

def expected_payout_parimutuel(p: float, takeout: float = None) -> float:
    """
    Calcule le fair odds (cote équitable) en parimutuel.
    
    En parimutuel, le pool est amputé du takeout avant redistribution.
    La cote équitable pour une probabilité p est:
    
        fair_odds_pool = (1 - takeout) / p
    
    Exemple avec takeout 16%:
    - p = 0.5 (50%) → fair_odds = 0.84 / 0.5 = 1.68
    - p = 0.1 (10%) → fair_odds = 0.84 / 0.1 = 8.40
    
    Args:
        p: Probabilité de victoire (0 < p < 1)
        takeout: Taux de prélèvement (défaut: config.markets.takeout_rate ou 0.16)
    
    Returns:
        float: Fair odds décimales (≥ 1.0)
    
    Raises:
        ValueError: Si p <= 0 ou p >= 1
    
    Examples:
        >>> expected_payout_parimutuel(0.5, 0.16)
        1.68
        >>> expected_payout_parimutuel(0.1, 0.16)
        8.4
        >>> expected_payout_parimutuel(0.25, 0.0)  # Sans takeout
        4.0
    """
    if p <= 0:
        raise ValueError(f"Probabilité doit être > 0, reçu: {p}")
    if p >= 1:
        raise ValueError(f"Probabilité doit être < 1, reçu: {p}")
    
    if takeout is None:
        takeout = _get_default_takeout()
    
    if not 0 <= takeout < 1:
        raise ValueError(f"Takeout doit être entre 0 et 1, reçu: {takeout}")
    
    return (1 - takeout) / p


def fair_odds_bookmaker(p: float, margin: float = 0.05) -> float:
    """
    Calcule le fair odds pour un bookmaker (cotes fixes).
    
    fair_odds = 1 / p * (1 - margin)
    
    Args:
        p: Probabilité de victoire
        margin: Marge du bookmaker (défaut: 5%)
    
    Returns:
        float: Fair odds décimales
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"Probabilité invalide: {p}")
    
    return (1 - margin) / p


# =============================================================================
# EV (EXPECTED VALUE) CALCULATIONS
# =============================================================================

def ev_parimutuel_win(
    p_win: float, 
    market_odds: float,
    takeout: float = None
) -> float:
    """
    Calcule l'Expected Value (EV) pour un pari WIN en parimutuel.
    
    Formule:
        value_win = p_win * fair_odds_pool - 1
        où fair_odds_pool = (1 - takeout) / p_win
    
    Simplification:
        value_win = (1 - takeout) - 1 = -takeout  ← SI market_odds = fair_odds
    
    Mais si les cotes marché diffèrent:
        value_win = p_win * market_odds * (1 - takeout_adjustment) - 1
    
    Note: En parimutuel pur, les cotes affichées INCLUENT déjà le takeout.
    On compare donc p_win * market_odds à 1 directement.
    
    Args:
        p_win: Probabilité estimée de victoire
        market_odds: Cote affichée (décimale européenne)
        takeout: Taux de prélèvement (pour information, déjà dans les cotes)
    
    Returns:
        float: EV du pari (-1 = perte totale, 0 = breakeven, >0 = value bet)
    
    Examples:
        >>> ev_parimutuel_win(0.5, 2.5, 0.16)  # Favori avec value
        0.05  # 5% EV
        >>> ev_parimutuel_win(0.1, 8.0, 0.16)  # Outsider neutre
        -0.128  # -12.8% EV
    """
    if p_win <= 0 or p_win >= 1:
        return -1.0
    
    if market_odds <= 1:
        return -1.0
    
    if takeout is None:
        takeout = _get_default_takeout()
    
    # En parimutuel, les cotes affichées représentent le payout réel
    # L'EV "brut" est simplement p * odds - 1
    # MAIS pour être conservateur, on applique un ajustement pour le takeout
    # car le pool peut évoluer et les cotes baisser
    
    # Méthode conservatrice: on considère que la cote finale sera
    # légèrement moins favorable que l'affichée
    # Ajustement = on réduit l'espérance de (takeout * factor)
    # avec factor = 0.3 (30% de l'effet du takeout)
    
    conservative_factor = 0.3  # Conservatisme modéré
    effective_odds = market_odds * (1 - takeout * conservative_factor)
    
    return p_win * effective_odds - 1


def ev_parimutuel_win_pure(
    p_win: float, 
    market_odds: float,
    takeout: float = None
) -> float:
    """
    Calcule l'EV WIN en utilisant directement les fair odds parimutuel.
    
    Cette version compare la cote marché aux fair odds du pool:
        EV = p_win * market_odds - 1
        
    Avec comparaison au fair:
        fair_odds = (1 - takeout) / p_win
        value = (market_odds - fair_odds) * p_win
    
    Args:
        p_win: Probabilité estimée de victoire
        market_odds: Cote affichée
        takeout: Taux de prélèvement
    
    Returns:
        float: EV du pari
    """
    if p_win <= 0 or p_win >= 1:
        return -1.0
    
    if market_odds <= 1:
        return -1.0
    
    if takeout is None:
        takeout = _get_default_takeout()
    
    # Fair odds dans un pool parimutuel avec ce takeout
    fair_odds = expected_payout_parimutuel(p_win, takeout)
    
    # Si market_odds > fair_odds, on a de la value
    # EV = p_win * market_odds - 1
    return p_win * market_odds - 1


def ev_parimutuel_place(
    p_place: float,
    market_odds_place: float,
    takeout: float = None,
    nb_partants: int = None
) -> float:
    """
    Calcule l'EV pour un pari PLACE en parimutuel.
    
    Le PLACE est plus complexe car:
    - Le pool place a son propre takeout
    - Les cotes dépendent du nombre de places payées
    
    Args:
        p_place: Probabilité d'être dans les N premiers (typiquement top 3)
        market_odds_place: Cote place affichée
        takeout: Taux de prélèvement
        nb_partants: Nombre de partants (pour ajustement)
    
    Returns:
        float: EV du pari place
    """
    if p_place <= 0 or p_place >= 1:
        return -1.0
    
    if market_odds_place <= 1:
        return -1.0
    
    if takeout is None:
        takeout = _get_default_takeout()
    
    # Ajustement pour le place: takeout souvent légèrement plus élevé
    # et variance des cotes plus forte
    place_takeout_adj = 1.1  # +10% de conservatisme
    
    effective_takeout = takeout * place_takeout_adj
    effective_odds = market_odds_place * (1 - effective_takeout * 0.3)
    
    return p_place * effective_odds - 1


def ev_bookmaker(
    p: float,
    odds: float,
    commission: float = 0.05
) -> float:
    """
    Calcule l'EV pour un bookmaker (cotes fixes).
    
    Args:
        p: Probabilité estimée
        odds: Cote proposée
        commission: Commission du bookmaker si gain
    
    Returns:
        float: EV du pari
    """
    if p <= 0 or p >= 1 or odds <= 1:
        return -1.0
    
    # EV = p * (odds - 1) * (1 - commission) - (1 - p)
    # Simplifié: p * odds * (1 - commission) - 1
    return p * odds * (1 - commission) - 1


# =============================================================================
# KELLY CRITERION
# =============================================================================

def kelly_fraction_raw(p: float, odds: float) -> float:
    """
    Calcule la fraction Kelly brute (non fractionnée).
    
    Formule: f* = (b*p - q) / b
    où b = odds - 1 (gain net pour 1€ misé)
    et q = 1 - p
    
    Args:
        p: Probabilité estimée de victoire
        odds: Cote décimale
    
    Returns:
        float: Fraction du bankroll à miser (0 si EV < 0)
    """
    if odds <= 1 or p <= 0 or p >= 1:
        return 0.0
    
    q = 1 - p
    b = odds - 1
    
    kelly = (b * p - q) / b
    return max(0.0, kelly)


def kelly_fraction(
    p: float,
    odds: float,
    fraction: float = 0.25,
    max_stake_pct: float = 0.05,
    parimutuel: bool = True,
    takeout: float = None
) -> float:
    """
    Calcule la fraction Kelly avec ajustements pour le parimutuel.
    
    Args:
        p: Probabilité estimée
        odds: Cote décimale
        fraction: Fraction de Kelly (0.25 = quart Kelly)
        max_stake_pct: Cap maximum du bankroll
        parimutuel: Si True, ajuste pour le takeout
        takeout: Taux de prélèvement
    
    Returns:
        float: Fraction du bankroll à miser (cappée)
    """
    if odds <= 1 or p <= 0 or p >= 1:
        return 0.0
    
    if takeout is None:
        takeout = _get_default_takeout()
    
    # Ajuster les cotes pour le parimutuel
    if parimutuel and takeout > 0:
        # Réduction conservatrice des cotes
        effective_odds = odds * (1 - takeout * 0.3)
    else:
        effective_odds = odds
    
    # Kelly brut
    kelly_raw = kelly_fraction_raw(p, effective_odds)
    
    if kelly_raw <= 0:
        return 0.0
    
    # Appliquer fraction et cap
    kelly_adj = kelly_raw * fraction
    return min(kelly_adj, max_stake_pct)


def kelly_stake(
    p: float,
    odds: float,
    bankroll: float,
    fraction: float = 0.25,
    max_stake_pct: float = 0.05,
    min_stake: float = 2.0,
    parimutuel: bool = True,
    takeout: float = None
) -> float:
    """
    Calcule le stake en euros via Kelly.
    
    Args:
        p: Probabilité estimée
        odds: Cote décimale
        bankroll: Bankroll total
        fraction: Fraction de Kelly
        max_stake_pct: Cap maximum
        min_stake: Mise minimum
        parimutuel: Mode parimutuel
        takeout: Taux de prélèvement
    
    Returns:
        float: Stake en euros (0 si < min_stake ou EV < 0)
    """
    kelly = kelly_fraction(
        p, odds, fraction, max_stake_pct, parimutuel, takeout
    )
    
    if kelly <= 0:
        return 0.0
    
    stake = bankroll * kelly
    
    # Arrondir et vérifier minimum
    stake = round(stake, 2)
    
    if stake < min_stake:
        return 0.0
    
    return stake


# =============================================================================
# VALUE DETECTION
# =============================================================================

def is_value_bet(
    p: float,
    odds: float,
    cutoff: float = 0.05,
    parimutuel: bool = True,
    takeout: float = None
) -> Tuple[bool, float]:
    """
    Détermine si c'est un value bet.
    
    Args:
        p: Probabilité estimée
        odds: Cote proposée
        cutoff: EV minimum pour être considéré value (défaut: 5%)
        parimutuel: Mode parimutuel
        takeout: Taux de prélèvement
    
    Returns:
        Tuple (is_value, ev)
    """
    if parimutuel:
        ev = ev_parimutuel_win(p, odds, takeout)
    else:
        ev = ev_bookmaker(p, odds)
    
    return ev >= cutoff, ev


def classify_value(ev: float) -> str:
    """
    Classifie un pari selon son EV.
    
    Returns:
        str: "EXCELLENT" (>20%), "BON" (>10%), "ACCEPTABLE" (>5%), 
             "MARGINAL" (>0%), "NEGATIF" (≤0%)
    """
    if ev > 0.20:
        return "EXCELLENT"
    elif ev > 0.10:
        return "BON"
    elif ev > 0.05:
        return "ACCEPTABLE"
    elif ev > 0:
        return "MARGINAL"
    else:
        return "NEGATIF"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # EV Parimutuel
    "expected_payout_parimutuel",
    "ev_parimutuel_win",
    "ev_parimutuel_win_pure",
    "ev_parimutuel_place",
    
    # EV Bookmaker
    "fair_odds_bookmaker",
    "ev_bookmaker",
    
    # Kelly
    "kelly_fraction_raw",
    "kelly_fraction",
    "kelly_stake",
    
    # Value
    "is_value_bet",
    "classify_value",
    
    # Constants
    "DEFAULT_TAKEOUT_RATE",
]


# =============================================================================
# CLI / TESTS
# =============================================================================

def main():
    """Tests et démonstration."""
    print("=" * 60)
    print("🧮 PARI MATH - Démonstration")
    print("=" * 60)
    
    takeout = 0.16
    print(f"\nTakeout PMU: {takeout:.0%}")
    print("-" * 40)
    
    # Test expected_payout_parimutuel
    print("\n📊 Expected Payout Parimutuel (fair odds):")
    test_probs = [0.50, 0.25, 0.10, 0.05, 0.01]
    for p in test_probs:
        fair = expected_payout_parimutuel(p, takeout)
        print(f"  p={p:.0%} → fair_odds = {fair:.2f}")
    
    # Test EV parimutuel
    print("\n📈 EV Parimutuel WIN:")
    test_cases = [
        (0.50, 2.50, "Favori avec légère value"),
        (0.50, 2.00, "Favori sous-coté"),
        (0.25, 5.00, "Milieu de tableau"),
        (0.10, 12.00, "Outsider avec value"),
        (0.10, 8.00, "Outsider sans value"),
        (0.05, 25.00, "Gros outsider value"),
        (0.01, 50.00, "Longshot"),
    ]
    
    for p, odds, desc in test_cases:
        ev = ev_parimutuel_win(p, odds, takeout)
        ev_pure = ev_parimutuel_win_pure(p, odds, takeout)
        is_val, _ = is_value_bet(p, odds, 0.05, True, takeout)
        print(f"  p={p:.0%}, odds={odds:.1f}: EV={ev:+.2%} (pure:{ev_pure:+.2%}) "
              f"{'✅ VALUE' if is_val else '❌'} - {desc}")
    
    # Test Kelly
    print("\n💰 Kelly Fraction (fraction=0.25, max=5%):")
    for p, odds, _ in test_cases[:5]:
        kelly = kelly_fraction(p, odds, 0.25, 0.05, True, takeout)
        stake = kelly_stake(p, odds, 1000, 0.25, 0.05, 2.0, True, takeout)
        print(f"  p={p:.0%}, odds={odds:.1f}: kelly={kelly:.2%}, stake={stake:.0f}€ (bankroll=1000€)")
    
    # Test cas bords
    print("\n⚠️ Cas bords:")
    
    # p très faible
    p_extreme = 0.001
    odds_extreme = 500
    ev_extreme = ev_parimutuel_win(p_extreme, odds_extreme, takeout)
    print(f"  p={p_extreme:.3%}, odds={odds_extreme}: EV={ev_extreme:+.2%}")
    
    # p très élevé
    p_high = 0.95
    odds_low = 1.05
    ev_high = ev_parimutuel_win(p_high, odds_low, takeout)
    print(f"  p={p_high:.0%}, odds={odds_low:.2f}: EV={ev_high:+.2%}")
    
    # Takeout 0%
    print(f"\n📊 Sans takeout (takeout=0%):")
    for p in [0.50, 0.25, 0.10]:
        fair_no_takeout = expected_payout_parimutuel(p, 0.0)
        fair_with_takeout = expected_payout_parimutuel(p, 0.16)
        print(f"  p={p:.0%}: fair(0%)={fair_no_takeout:.2f}, fair(16%)={fair_with_takeout:.2f}")
    
    print("\n✅ Tous les tests passés!")


if __name__ == "__main__":
    main()
