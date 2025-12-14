"""
Tests unitaires pour la politique de mise par défaut (Kelly fractionnaire).

Tests couverts:
- value <= 0 → stake == 0
- Calcul Kelly correct selon formule
- Cap per bet (2% bankroll)
- Arrondi 0.50€
- PMU takeout (16%)
"""

import pytest
import math
from dataclasses import dataclass
from typing import Optional

# Simulation des profils Kelly
KELLY_PROFILES = {
    "SUR": 0.25,
    "STANDARD": 0.33,
    "AMBITIEUX": 0.50,
    "PERSONNALISE": None
}

# Paramètres par défaut
DEFAULT_CAP_PER_BET = 0.02  # 2%
DEFAULT_DAILY_BUDGET_RATE = 0.12  # 12%
DEFAULT_VALUE_CUTOFF = 0.05  # 5%
DEFAULT_ROUNDING_INCREMENT = 0.50  # 0.50€
PMU_TAKEOUT = 0.16  # 16%


@dataclass
class BettingConfig:
    """Configuration de mise."""
    kelly_profile: str = "STANDARD"
    kelly_fraction: float = 0.33
    cap_per_bet: float = 0.02
    daily_budget_rate: float = 0.12
    value_cutoff: float = 0.05
    rounding_increment_eur: float = 0.50


def calculate_kelly_full(p_win: float, odds: float) -> float:
    """
    Calcule le Kelly plein selon la formule:
    f* = (p*(o-1) - (1-p)) / (o-1)
    
    Args:
        p_win: Probabilité de gagner (0-1)
        odds: Cote décimale
    
    Returns:
        Fraction Kelly optimale (peut être négative si -EV)
    """
    if odds <= 1:
        return 0.0
    numerator = p_win * (odds - 1) - (1 - p_win)
    denominator = odds - 1
    return numerator / denominator


def calculate_value(p_win: float, odds: float) -> float:
    """
    Calcule la value en pourcentage.
    value = (p * odds) - 1
    """
    return (p_win * odds) - 1


def calculate_stake(
    bankroll: float,
    p_win: float,
    odds: float,
    config: BettingConfig
) -> float:
    """
    Calcule la mise optimale selon Kelly fractionnaire avec caps et arrondi.
    
    stake = bankroll * min(kelly_fraction * f*, cap_per_bet)
    arrondi au rounding_increment
    """
    # Calcul de la value
    value = calculate_value(p_win, odds)
    
    # Si value <= cutoff, pas de mise
    if value <= config.value_cutoff:
        return 0.0
    
    # Calcul Kelly plein
    kelly_full = calculate_kelly_full(p_win, odds)
    
    # Si Kelly <= 0, pas de mise
    if kelly_full <= 0:
        return 0.0
    
    # Récupérer la fraction Kelly du profil
    kelly_fraction = KELLY_PROFILES.get(config.kelly_profile) or config.kelly_fraction
    
    # Kelly fractionnaire
    kelly_fractional = kelly_fraction * kelly_full
    
    # Appliquer le cap
    capped = min(kelly_fractional, config.cap_per_bet)
    
    # Calculer la mise
    stake = bankroll * capped
    
    # Arrondir
    rounding = config.rounding_increment_eur
    stake = round(stake / rounding) * rounding
    
    return stake


def calculate_fair_odds_pmu(p_win: float) -> float:
    """
    Calcule la cote juste pour le PMU avec takeout.
    fair_odds_pool = (1 - takeout) / p
    """
    if p_win <= 0:
        return float('inf')
    return (1 - PMU_TAKEOUT) / p_win


# ==============================================================================
# TESTS
# ==============================================================================

class TestValueCalculation:
    """Tests pour le calcul de la value."""
    
    def test_positive_value(self):
        """Value positive quand p*odds > 1."""
        p_win = 0.30
        odds = 4.0
        value = calculate_value(p_win, odds)
        # 0.30 * 4.0 - 1 = 1.2 - 1 = 0.2 = 20%
        assert value == pytest.approx(0.2, rel=1e-6)
    
    def test_negative_value(self):
        """Value négative quand p*odds < 1."""
        p_win = 0.20
        odds = 4.0
        value = calculate_value(p_win, odds)
        # 0.20 * 4.0 - 1 = 0.8 - 1 = -0.2 = -20%
        assert value == pytest.approx(-0.2, rel=1e-6)
    
    def test_zero_value(self):
        """Value = 0 quand p*odds = 1 (fair)."""
        p_win = 0.25
        odds = 4.0
        value = calculate_value(p_win, odds)
        # 0.25 * 4.0 - 1 = 1.0 - 1 = 0
        assert value == pytest.approx(0.0, rel=1e-6)


class TestKellyCalculation:
    """Tests pour le calcul du Kelly plein."""
    
    def test_positive_kelly(self):
        """Kelly positif pour un pari +EV."""
        p_win = 0.30
        odds = 4.0
        kelly = calculate_kelly_full(p_win, odds)
        # f* = (0.30 * 3 - 0.70) / 3 = (0.9 - 0.7) / 3 = 0.2 / 3 ≈ 0.0667
        expected = (0.30 * 3 - 0.70) / 3
        assert kelly == pytest.approx(expected, rel=1e-6)
    
    def test_negative_kelly(self):
        """Kelly négatif pour un pari -EV."""
        p_win = 0.20
        odds = 4.0
        kelly = calculate_kelly_full(p_win, odds)
        # f* = (0.20 * 3 - 0.80) / 3 = (0.6 - 0.8) / 3 = -0.2 / 3 ≈ -0.0667
        assert kelly < 0
    
    def test_zero_kelly_fair_bet(self):
        """Kelly = 0 pour un pari fair."""
        p_win = 0.25
        odds = 4.0
        kelly = calculate_kelly_full(p_win, odds)
        # f* = (0.25 * 3 - 0.75) / 3 = 0 / 3 = 0
        assert kelly == pytest.approx(0.0, rel=1e-6)
    
    def test_kelly_odds_one(self):
        """Kelly = 0 si odds <= 1."""
        kelly = calculate_kelly_full(0.5, 1.0)
        assert kelly == 0.0
        
        kelly = calculate_kelly_full(0.5, 0.5)
        assert kelly == 0.0


class TestStakeCalculation:
    """Tests pour le calcul de la mise avec Kelly fractionnaire."""
    
    def setup_method(self):
        """Configuration par défaut."""
        self.bankroll = 1000.0
        self.config = BettingConfig()
    
    def test_zero_stake_negative_value(self):
        """Mise = 0 si value <= 0."""
        # p=0.20, odds=4 → value = 0.8-1 = -0.2 = -20% → stake = 0
        stake = calculate_stake(self.bankroll, 0.20, 4.0, self.config)
        assert stake == 0.0
    
    def test_zero_stake_below_cutoff(self):
        """Mise = 0 si value < cutoff (5%)."""
        # p=0.26, odds=4 → value = 1.04-1 = 0.04 = 4% < 5% → stake = 0
        stake = calculate_stake(self.bankroll, 0.26, 4.0, self.config)
        assert stake == 0.0
    
    def test_positive_stake_above_cutoff(self):
        """Mise > 0 si value >= cutoff."""
        # p=0.30, odds=4 → value = 1.2-1 = 0.2 = 20% > 5%
        stake = calculate_stake(self.bankroll, 0.30, 4.0, self.config)
        assert stake > 0
    
    def test_stake_uses_kelly_fraction(self):
        """La mise utilise la fraction Kelly du profil."""
        # Profil STANDARD = 33%
        config_standard = BettingConfig(kelly_profile="STANDARD")
        stake_standard = calculate_stake(1000, 0.35, 4.0, config_standard)
        
        # Profil SUR = 25%
        config_sur = BettingConfig(kelly_profile="SUR")
        stake_sur = calculate_stake(1000, 0.35, 4.0, config_sur)
        
        # SUR devrait donner une mise plus faible
        assert stake_sur < stake_standard
    
    def test_stake_capped_at_2_percent(self):
        """La mise est plafonnée à 2% de la bankroll."""
        # Avec un Kelly très élevé, on devrait être cap à 2%
        # p=0.90, odds=1.5 → Kelly élevé
        stake = calculate_stake(self.bankroll, 0.90, 1.5, self.config)
        max_stake = self.bankroll * 0.02  # 20€
        assert stake <= max_stake + self.config.rounding_increment_eur  # Marge pour arrondi
    
    def test_stake_rounded_to_half_euro(self):
        """La mise est arrondie à 0.50€."""
        stake = calculate_stake(self.bankroll, 0.35, 4.0, self.config)
        # Vérifier que stake est un multiple de 0.50
        remainder = stake % 0.50
        assert remainder == pytest.approx(0.0, abs=1e-9) or remainder == pytest.approx(0.50, abs=1e-9)


class TestKellyProfiles:
    """Tests pour les profils Kelly."""
    
    def test_sur_profile_25_percent(self):
        """Profil SUR = 25% Kelly."""
        assert KELLY_PROFILES["SUR"] == 0.25
    
    def test_standard_profile_33_percent(self):
        """Profil STANDARD = 33% Kelly."""
        assert KELLY_PROFILES["STANDARD"] == 0.33
    
    def test_ambitieux_profile_50_percent(self):
        """Profil AMBITIEUX = 50% Kelly."""
        assert KELLY_PROFILES["AMBITIEUX"] == 0.50
    
    def test_personnalise_profile_custom(self):
        """Profil PERSONNALISE permet valeur custom."""
        assert KELLY_PROFILES["PERSONNALISE"] is None


class TestPMUTakeout:
    """Tests pour le calcul des cotes justes PMU."""
    
    def test_fair_odds_with_takeout(self):
        """Cote juste avec prélèvement PMU 16%."""
        p_win = 0.25
        fair_odds = calculate_fair_odds_pmu(p_win)
        # fair = (1 - 0.16) / 0.25 = 0.84 / 0.25 = 3.36
        assert fair_odds == pytest.approx(3.36, rel=1e-6)
    
    def test_fair_odds_high_proba(self):
        """Cote juste pour forte probabilité."""
        p_win = 0.50
        fair_odds = calculate_fair_odds_pmu(p_win)
        # fair = 0.84 / 0.50 = 1.68
        assert fair_odds == pytest.approx(1.68, rel=1e-6)
    
    def test_fair_odds_zero_proba(self):
        """Cote infinie si p = 0."""
        fair_odds = calculate_fair_odds_pmu(0.0)
        assert fair_odds == float('inf')


class TestRoundingBehavior:
    """Tests pour l'arrondi des mises."""
    
    def test_round_down(self):
        """Arrondi vers le bas quand proche du seuil inférieur."""
        # 7.24€ → 7.00€
        stake = 7.24
        rounding = 0.50
        rounded = round(stake / rounding) * rounding
        assert rounded == 7.0
    
    def test_round_up(self):
        """Arrondi vers le haut quand proche du seuil supérieur."""
        # 7.76€ → 8.00€
        stake = 7.76
        rounding = 0.50
        rounded = round(stake / rounding) * rounding
        assert rounded == 8.0
    
    def test_round_to_half(self):
        """Arrondi à 0.50€."""
        # 7.30€ → 7.50€
        stake = 7.30
        rounding = 0.50
        rounded = round(stake / rounding) * rounding
        assert rounded == 7.5


class TestDailyBudget:
    """Tests pour le budget journalier."""
    
    def test_daily_budget_12_percent(self):
        """Budget journalier = 12% de la bankroll."""
        bankroll = 1000.0
        daily_budget = bankroll * DEFAULT_DAILY_BUDGET_RATE
        assert daily_budget == 120.0
    
    def test_max_6_bets_within_budget(self):
        """6 paris au cap (2%) restent dans le budget (12%)."""
        bankroll = 1000.0
        cap = bankroll * DEFAULT_CAP_PER_BET  # 20€
        daily_budget = bankroll * DEFAULT_DAILY_BUDGET_RATE  # 120€
        max_bets_at_cap = daily_budget / cap  # 6 paris
        assert max_bets_at_cap == 6.0


class TestEdgeCases:
    """Tests pour les cas limites."""
    
    def test_very_small_bankroll(self):
        """Bankroll très faible."""
        config = BettingConfig()
        stake = calculate_stake(10.0, 0.35, 4.0, config)
        # Cap = 0.20€, arrondi = 0.50€ → probablement 0 ou 0.50
        assert stake >= 0
    
    def test_very_high_odds(self):
        """Cotes très élevées."""
        config = BettingConfig()
        stake = calculate_stake(1000.0, 0.05, 50.0, config)
        # Devrait être cappé à 2%
        assert stake <= 20.5  # 2% + marge arrondi
    
    def test_probability_one(self):
        """Probabilité = 1 (cas théorique)."""
        kelly = calculate_kelly_full(1.0, 2.0)
        # f* = (1*1 - 0) / 1 = 1 → mise tout
        assert kelly == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
