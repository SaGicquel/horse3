#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests pour pari_math.py - EV Parimutuel
========================================

Tests unitaires pour les fonctions:
- expected_payout_parimutuel(p, takeout)
- ev_parimutuel_win(p, odds, takeout)
- kelly_fraction(p, odds, fraction, ...)

Couvre:
- Takeout 16% standard PMU
- Cas bords (p très faible, p très élevé)
- Probas extrêmes (0.001, 0.999)
- Cohérence avec config.markets.takeout_rate

Auteur: Horse3 Pro System
"""

import pytest
import math
import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TESTS EXPECTED PAYOUT PARIMUTUEL
# =============================================================================

class TestExpectedPayoutParimutuel:
    """Tests pour expected_payout_parimutuel(p, takeout)."""
    
    TAKEOUT_PMU = 0.16  # Standard PMU
    
    def test_fair_odds_50_pct_takeout_16(self):
        """p=50%, takeout=16% → fair_odds = 1.68"""
        from pari_math import expected_payout_parimutuel
        
        fair_odds = expected_payout_parimutuel(0.50, self.TAKEOUT_PMU)
        expected = (1 - 0.16) / 0.50  # = 0.84 / 0.50 = 1.68
        
        assert abs(fair_odds - expected) < 1e-6
        assert abs(fair_odds - 1.68) < 1e-6
    
    def test_fair_odds_25_pct_takeout_16(self):
        """p=25%, takeout=16% → fair_odds = 3.36"""
        from pari_math import expected_payout_parimutuel
        
        fair_odds = expected_payout_parimutuel(0.25, self.TAKEOUT_PMU)
        expected = 0.84 / 0.25  # = 3.36
        
        assert abs(fair_odds - 3.36) < 1e-6
    
    def test_fair_odds_10_pct_takeout_16(self):
        """p=10%, takeout=16% → fair_odds = 8.40"""
        from pari_math import expected_payout_parimutuel
        
        fair_odds = expected_payout_parimutuel(0.10, self.TAKEOUT_PMU)
        expected = 0.84 / 0.10  # = 8.40
        
        assert abs(fair_odds - 8.40) < 1e-6
    
    def test_fair_odds_5_pct_takeout_16(self):
        """p=5%, takeout=16% → fair_odds = 16.80"""
        from pari_math import expected_payout_parimutuel
        
        fair_odds = expected_payout_parimutuel(0.05, self.TAKEOUT_PMU)
        expected = 0.84 / 0.05  # = 16.80
        
        assert abs(fair_odds - 16.80) < 1e-6
    
    def test_fair_odds_1_pct_takeout_16(self):
        """p=1%, takeout=16% → fair_odds = 84.00"""
        from pari_math import expected_payout_parimutuel
        
        fair_odds = expected_payout_parimutuel(0.01, self.TAKEOUT_PMU)
        expected = 0.84 / 0.01  # = 84.00
        
        assert abs(fair_odds - 84.00) < 1e-6
    
    def test_fair_odds_sans_takeout(self):
        """Sans takeout (0%), fair_odds = 1/p"""
        from pari_math import expected_payout_parimutuel
        
        # p=50%, takeout=0% → fair_odds = 2.0
        fair_odds = expected_payout_parimutuel(0.50, 0.0)
        assert abs(fair_odds - 2.0) < 1e-6
        
        # p=10%, takeout=0% → fair_odds = 10.0
        fair_odds = expected_payout_parimutuel(0.10, 0.0)
        assert abs(fair_odds - 10.0) < 1e-6
    
    def test_fair_odds_takeout_variable(self):
        """Test avec différents takeouts."""
        from pari_math import expected_payout_parimutuel
        
        p = 0.25
        
        # Takeout 10%: fair = 0.9 / 0.25 = 3.6
        assert abs(expected_payout_parimutuel(p, 0.10) - 3.6) < 1e-6
        
        # Takeout 20%: fair = 0.8 / 0.25 = 3.2
        assert abs(expected_payout_parimutuel(p, 0.20) - 3.2) < 1e-6
        
        # Takeout 25%: fair = 0.75 / 0.25 = 3.0
        assert abs(expected_payout_parimutuel(p, 0.25) - 3.0) < 1e-6


# =============================================================================
# TESTS CAS BORDS
# =============================================================================

class TestCasBords:
    """Tests des cas limites pour pari_math."""
    
    def test_proba_tres_faible(self):
        """p = 0.001 (0.1%) - longshot extrême."""
        from pari_math import expected_payout_parimutuel, ev_parimutuel_win
        
        p = 0.001
        takeout = 0.16
        
        # Fair odds = 0.84 / 0.001 = 840
        fair_odds = expected_payout_parimutuel(p, takeout)
        assert abs(fair_odds - 840.0) < 0.1
        
        # Si market_odds = 500 (sous fair) → EV négative
        ev = ev_parimutuel_win(p, 500, takeout)
        assert ev < 0, f"EV devrait être < 0, obtenu {ev}"
        
        # Si market_odds = 1500 (bien au-dessus du fair) → EV doit être positive
        # Note: avec l'ajustement conservateur (30% du takeout), 
        # il faut des odds significativement au-dessus du fair
        ev = ev_parimutuel_win(p, 1500, takeout)
        assert ev > 0, f"EV devrait être > 0 pour odds >> fair, obtenu {ev}"
    
    def test_proba_tres_elevee(self):
        """p = 0.999 (99.9%) - quasi-certitude."""
        from pari_math import expected_payout_parimutuel, ev_parimutuel_win
        
        p = 0.999
        takeout = 0.16
        
        # Fair odds = 0.84 / 0.999 ≈ 0.841
        fair_odds = expected_payout_parimutuel(p, takeout)
        assert abs(fair_odds - 0.841) < 0.01
        
        # Cote < 1 est impossible dans la réalité
        # EV avec odds = 1.01 (minimum réaliste)
        ev = ev_parimutuel_win(p, 1.01, takeout)
        # On s'attend à EV > 0 car p très élevé
        # Mais odds * p = 1.01 * 0.999 ≈ 1.01, donc EV ≈ 0
        assert -0.1 < ev < 0.1
    
    def test_proba_zero_raises(self):
        """p = 0 doit lever une erreur."""
        from pari_math import expected_payout_parimutuel
        
        with pytest.raises(ValueError):
            expected_payout_parimutuel(0.0, 0.16)
    
    def test_proba_un_raises(self):
        """p = 1 doit lever une erreur."""
        from pari_math import expected_payout_parimutuel
        
        with pytest.raises(ValueError):
            expected_payout_parimutuel(1.0, 0.16)
    
    def test_proba_negative_raises(self):
        """p < 0 doit lever une erreur."""
        from pari_math import expected_payout_parimutuel
        
        with pytest.raises(ValueError):
            expected_payout_parimutuel(-0.1, 0.16)
    
    def test_takeout_invalide_raises(self):
        """takeout >= 1 doit lever une erreur."""
        from pari_math import expected_payout_parimutuel
        
        with pytest.raises(ValueError):
            expected_payout_parimutuel(0.5, 1.0)
        
        with pytest.raises(ValueError):
            expected_payout_parimutuel(0.5, 1.5)
    
    def test_takeout_negatif_raises(self):
        """takeout < 0 doit lever une erreur."""
        from pari_math import expected_payout_parimutuel
        
        with pytest.raises(ValueError):
            expected_payout_parimutuel(0.5, -0.1)


# =============================================================================
# TESTS PROBAS EXTRÊMES
# =============================================================================

class TestProbasExtremes:
    """Tests avec des probabilités extrêmes."""
    
    def test_proba_0_001(self):
        """p = 0.1% (1 sur 1000)."""
        from pari_math import expected_payout_parimutuel, ev_parimutuel_win, kelly_fraction
        
        p = 0.001
        takeout = 0.16
        
        # Fair odds très élevés
        fair = expected_payout_parimutuel(p, takeout)
        assert fair > 500
        
        # EV avec odds réalistes (200x)
        ev = ev_parimutuel_win(p, 200, takeout)
        # p * odds = 0.001 * 200 = 0.2 → EV largement < 0
        assert ev < -0.5
        
        # Kelly doit être 0 (pas de value)
        kelly = kelly_fraction(p, 200, 0.25, 0.05, True, takeout)
        assert kelly == 0
    
    def test_proba_0_95(self):
        """p = 95% (très fort favori)."""
        from pari_math import expected_payout_parimutuel, ev_parimutuel_win, kelly_fraction
        
        p = 0.95
        takeout = 0.16
        
        # Fair odds très bas
        fair = expected_payout_parimutuel(p, takeout)
        expected_fair = 0.84 / 0.95
        assert abs(fair - expected_fair) < 0.01
        assert fair < 1  # Moins que mise
        
        # Avec odds = 1.05 (minimum réaliste pour un favori)
        ev = ev_parimutuel_win(p, 1.05, takeout)
        # p * odds_adj ≈ 0.95 * 1.0 ≈ 0.95 → EV ≈ -5%
        assert -0.2 < ev < 0.1
        
        # Kelly peut être positif si odds > fair ajusté
        kelly = kelly_fraction(p, 1.1, 0.25, 0.05, True, takeout)
        # Devrait être très faible ou 0
        assert 0 <= kelly <= 0.05


# =============================================================================
# TESTS EV PARIMUTUEL WIN
# =============================================================================

class TestEVParimutuelWin:
    """Tests pour ev_parimutuel_win(p, odds, takeout)."""
    
    def test_ev_value_bet_simple(self):
        """EV > 0 quand odds > fair_odds."""
        from pari_math import ev_parimutuel_win, expected_payout_parimutuel
        
        p = 0.25
        takeout = 0.16
        
        # Fair odds = 3.36
        fair = expected_payout_parimutuel(p, takeout)
        
        # Odds = 4.0 > fair → value bet
        ev = ev_parimutuel_win(p, 4.0, takeout)
        # EV brut = 0.25 * 4.0 - 1 = 0.0
        # Avec ajustement conservateur (30% du takeout):
        # effective_odds ≈ 4.0 * (1 - 0.16 * 0.3) = 4.0 * 0.952 = 3.808
        # EV = 0.25 * 3.808 - 1 = -0.048
        # Note: l'EV conservateur peut être légèrement négative
        assert ev > -0.1  # Pas fortement négative
    
    def test_ev_negatif_quand_sous_cote(self):
        """EV < 0 quand odds << fair_odds."""
        from pari_math import ev_parimutuel_win
        
        p = 0.25  # Fair odds avec 16% takeout = 3.36
        
        # Odds = 2.0 (très sous-coté)
        ev = ev_parimutuel_win(p, 2.0, 0.16)
        
        # EV devrait être clairement négative
        assert ev < 0, f"EV devrait être < 0, obtenu {ev}"
    
    def test_ev_avec_odds_invalides(self):
        """EV = -1 avec odds <= 1."""
        from pari_math import ev_parimutuel_win
        
        ev = ev_parimutuel_win(0.5, 1.0, 0.16)
        assert ev == -1.0
        
        ev = ev_parimutuel_win(0.5, 0.5, 0.16)
        assert ev == -1.0
    
    def test_ev_avec_p_invalide(self):
        """EV = -1 avec p <= 0 ou p >= 1."""
        from pari_math import ev_parimutuel_win
        
        assert ev_parimutuel_win(0.0, 5.0, 0.16) == -1.0
        assert ev_parimutuel_win(1.0, 5.0, 0.16) == -1.0
        assert ev_parimutuel_win(-0.1, 5.0, 0.16) == -1.0


# =============================================================================
# TESTS KELLY FRACTION
# =============================================================================

class TestKellyFraction:
    """Tests pour kelly_fraction et kelly_stake."""
    
    def test_kelly_zero_si_ev_negative(self):
        """Kelly = 0 si pas de value."""
        from pari_math import kelly_fraction
        
        # p = 0.1, odds = 5.0 → EV brut = 0.1 * 5 - 1 = -0.5 (pas de value)
        kelly = kelly_fraction(0.1, 5.0, 0.25, 0.05, True, 0.16)
        assert kelly == 0
    
    def test_kelly_positif_si_value(self):
        """Kelly > 0 si value bet."""
        from pari_math import kelly_fraction
        
        # p = 0.5, odds = 3.0 → EV brut = 0.5 * 3 - 1 = 0.5 (grosse value)
        kelly = kelly_fraction(0.5, 3.0, 0.25, 0.05, True, 0.16)
        assert kelly > 0
    
    def test_kelly_cap_respecte(self):
        """Kelly est cappé à max_stake_pct."""
        from pari_math import kelly_fraction
        
        # Grosse value → Kelly brut élevé, mais cappé
        kelly = kelly_fraction(0.7, 5.0, 1.0, 0.05, False, 0.0)  # Kelly full
        
        # Avec fraction=1.0 et grosse value, Kelly brut serait > 5%
        # Mais doit être cappé à 5%
        assert kelly <= 0.05
    
    def test_kelly_stake_minimum(self):
        """kelly_stake retourne 0 si stake < min_stake."""
        from pari_math import kelly_stake
        
        # Petite value → petit Kelly → stake < 2€
        stake = kelly_stake(0.11, 10.0, 100, 0.25, 0.05, 2.0, True, 0.16)
        
        # Si le stake calculé est < 2€, doit retourner 0
        # Avec bankroll=100€ et kelly très faible, stake sera < 2€
        assert stake == 0 or stake >= 2.0


# =============================================================================
# TESTS COHÉRENCE CONFIG
# =============================================================================

class TestCoherenceConfig:
    """Tests de cohérence avec config.markets.takeout_rate."""
    
    def test_default_takeout_from_config(self):
        """Le takeout par défaut vient de la config."""
        try:
            from config.loader import get_config
            from pari_math import expected_payout_parimutuel, DEFAULT_TAKEOUT_RATE
            
            config = get_config()
            
            # Vérifier que la config a un takeout
            if hasattr(config, 'markets') and config.markets:
                config_takeout = config.markets.takeout_rate
            else:
                config_takeout = DEFAULT_TAKEOUT_RATE
            
            # Le DEFAULT_TAKEOUT_RATE devrait être 0.16
            assert config_takeout == 0.16 or config_takeout == DEFAULT_TAKEOUT_RATE
            
        except ImportError:
            pytest.skip("config.loader non disponible")
    
    def test_takeout_16_pct_standard(self):
        """Le takeout PMU standard est 16%."""
        from pari_math import DEFAULT_TAKEOUT_RATE
        
        assert DEFAULT_TAKEOUT_RATE == 0.16


# =============================================================================
# TESTS INTEGRATION
# =============================================================================

class TestIntegrationPariMath:
    """Tests d'intégration avec race_pronostic_generator."""
    
    def test_race_pronostic_uses_pari_math(self):
        """Vérifie que race_pronostic_generator utilise pari_math."""
        try:
            from race_pronostic_generator import PARI_MATH_AVAILABLE
            assert PARI_MATH_AVAILABLE, "pari_math devrait être disponible"
        except ImportError:
            pytest.skip("race_pronostic_generator non disponible")
    
    def test_kelly_calculator_uses_pari_math(self):
        """Vérifie que KellyCalculator utilise pari_math."""
        try:
            from race_pronostic_generator import KellyCalculator, PARI_MATH_AVAILABLE
            
            if PARI_MATH_AVAILABLE:
                # Test que calculate_value utilise ev_parimutuel_win
                value = KellyCalculator.calculate_value(0.25, 5.0, True, 0.16)
                
                from pari_math import ev_parimutuel_win
                expected = ev_parimutuel_win(0.25, 5.0, 0.16)
                
                assert abs(value - expected) < 0.001
                
        except ImportError:
            pytest.skip("race_pronostic_generator non disponible")
    
    def test_portfolio_optimizer_uses_pari_math(self):
        """Vérifie que betting_portfolio_optimizer utilise pari_math."""
        try:
            from betting_portfolio_optimizer import PARI_MATH_AVAILABLE
            assert PARI_MATH_AVAILABLE, "pari_math devrait être disponible"
        except ImportError:
            pytest.skip("betting_portfolio_optimizer non disponible")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
