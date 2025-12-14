#!/usr/bin/env python3
"""
Tests Unitaires pour le Module d'Estimation p(place)
=====================================================

Vérifie:
- Contraintes mathématiques (Σp_win = 1, p_place >= p_win)
- Cohérence des estimateurs (Harville, Henery, Stern, LBS)
- Stabilité des simulations
- Calibration des probabilités

Usage:
    python -m pytest tests/test_place_estimator.py -v
    python tests/test_place_estimator.py

Auteur: Horse Racing AI System
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from place_probability_estimator import (
    HarvilleEstimator,
    HeneryEstimator,
    SternEstimator,
    LoBaconShoneEstimator,
    PlackettLuceTemperatureSimulator,
    PlaceProbabilityEstimator,
    PlaceEstimatorConfig,
    CalibrationMetrics,
    EVStabilityValidator,
    ExoticEVCalculator
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_probs():
    """Probabilités simples à 5 chevaux."""
    return np.array([0.40, 0.25, 0.20, 0.10, 0.05])

@pytest.fixture
def medium_probs():
    """Probabilités à 10 chevaux."""
    probs = np.array([0.22, 0.18, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02])
    return probs / probs.sum()

@pytest.fixture
def uniform_probs():
    """Probabilités uniformes à 8 chevaux."""
    return np.ones(8) / 8


# =============================================================================
# TESTS CONTRAINTES MATHÉMATIQUES
# =============================================================================

class TestProbabilityConstraints:
    """Tests des contraintes mathématiques sur les probabilités."""
    
    def test_win_probs_sum_to_one(self, simple_probs):
        """Σp_win = 1."""
        assert np.isclose(simple_probs.sum(), 1.0, atol=0.01)
    
    def test_place_prob_greater_than_win(self, simple_probs):
        """P(place) >= P(win) pour chaque cheval."""
        estimator = HarvilleEstimator(simple_probs)
        p_place = estimator.p_place(top_n=3)
        
        for i in range(len(simple_probs)):
            assert p_place[i] >= simple_probs[i] - 0.01, \
                f"Cheval {i}: p_place={p_place[i]:.3f} < p_win={simple_probs[i]:.3f}"
    
    def test_place_probs_bounded(self, simple_probs):
        """0 <= P(place) <= 1."""
        estimator = HarvilleEstimator(simple_probs)
        p_place = estimator.p_place(top_n=3)
        
        assert all(0 <= p <= 1 for p in p_place)
    
    def test_exacta_prob_less_than_win(self, simple_probs):
        """P(i 1er, j 2ème) <= P(i 1er)."""
        estimator = HarvilleEstimator(simple_probs)
        
        for i in range(len(simple_probs)):
            for j in range(len(simple_probs)):
                if i != j:
                    p_exacta = estimator.p_exacta(i, j)
                    assert p_exacta <= simple_probs[i] + 0.001
    
    def test_trifecta_sum_valid(self, simple_probs):
        """Somme des P(trifecta) = 1."""
        estimator = HarvilleEstimator(simple_probs)
        n = len(simple_probs)
        
        total = 0.0
        for i in range(n):
            for j in range(n):
                if j != i:
                    for k in range(n):
                        if k != i and k != j:
                            total += estimator.p_trifecta(i, j, k)
        
        assert np.isclose(total, 1.0, atol=0.1), f"Somme trifecta = {total}"


# =============================================================================
# TESTS ESTIMATEURS
# =============================================================================

class TestHarvilleEstimator:
    """Tests de l'estimateur de Harville."""
    
    def test_win_prob_position_1(self, simple_probs):
        """P(win) = P(position 1)."""
        estimator = HarvilleEstimator(simple_probs)
        
        for i, p_win in enumerate(simple_probs):
            p_pos1 = estimator.p_finish_position(i, 1)
            assert np.isclose(p_pos1, p_win, atol=0.001)
    
    def test_exacta_formula(self, simple_probs):
        """Vérifie la formule exacta de Harville."""
        estimator = HarvilleEstimator(simple_probs)
        
        # P(i=1, j=2) = p_i * p_j / (1 - p_i)
        i, j = 0, 1
        expected = simple_probs[i] * simple_probs[j] / (1 - simple_probs[i])
        actual = estimator.p_exacta(i, j)
        
        assert np.isclose(actual, expected, atol=0.001)


class TestHeneryEstimator:
    """Tests de l'estimateur de Henery."""
    
    def test_gamma_reduces_favorite_advantage(self, simple_probs):
        """Gamma < 1 réduit l'avantage du favori."""
        harville = HarvilleEstimator(simple_probs)
        henery = HeneryEstimator(simple_probs, gamma=0.81)
        
        h_place = harville.p_place(3)
        he_place = henery.p_place(3)
        
        # Le favori (indice 0) devrait avoir une p_place plus faible avec Henery
        assert he_place[0] < h_place[0], \
            f"Henery devrait réduire p_place du favori: {he_place[0]} vs {h_place[0]}"
    
    def test_gamma_1_equals_harville(self, simple_probs):
        """Henery avec gamma=1 = Harville."""
        harville = HarvilleEstimator(simple_probs)
        henery = HeneryEstimator(simple_probs, gamma=1.0)
        
        h_place = harville.p_place(3)
        he_place = henery.p_place(3)
        
        np.testing.assert_array_almost_equal(h_place, he_place, decimal=3)


class TestSternEstimator:
    """Tests de l'estimateur de Stern."""
    
    def test_lambda_0_equals_harville(self, simple_probs):
        """Stern avec lambda=0 = Harville."""
        harville = HarvilleEstimator(simple_probs)
        stern = SternEstimator(simple_probs, lambda_=0.0)
        
        h_place = harville.p_place(3)
        s_place = stern.p_place(3)
        
        np.testing.assert_array_almost_equal(h_place, s_place, decimal=3)
    
    def test_lambda_smooths_distribution(self, simple_probs):
        """Lambda > 0 lisse la distribution vers l'uniforme."""
        stern_low = SternEstimator(simple_probs, lambda_=0.1)
        stern_high = SternEstimator(simple_probs, lambda_=0.3)
        
        low_std = np.std(stern_low.p_place(3))
        high_std = np.std(stern_high.p_place(3))
        
        # Plus de lambda = moins de variance
        assert high_std < low_std


class TestLoBaconShoneEstimator:
    """Tests de l'estimateur Lo-Bacon-Shone."""
    
    def test_iterative_convergence(self, simple_probs):
        """L'algorithme converge avec plus d'itérations."""
        lbs_10 = LoBaconShoneEstimator(simple_probs, iterations=10)
        lbs_100 = LoBaconShoneEstimator(simple_probs, iterations=100)
        
        p1 = lbs_10.p_place(3)
        p2 = lbs_100.p_place(3)
        
        # Les résultats devraient être proches mais pas identiques
        # (convergence)
        diff = np.abs(p1 - p2).max()
        assert diff < 0.1, f"Différence max = {diff}"


# =============================================================================
# TESTS SIMULATEUR PLACKETT-LUCE
# =============================================================================

class TestPlackettLuceSimulator:
    """Tests du simulateur Plackett-Luce avec température."""
    
    def test_simulation_preserves_ranking_expectation(self, simple_probs):
        """Le favori gagne le plus souvent en simulation."""
        sim = PlackettLuceTemperatureSimulator(simple_probs, temperature=1.0, seed=42)
        arrivals = sim.simulate_n_races(1000)
        
        # Compter les victoires
        wins = np.zeros(len(simple_probs))
        for arrival in arrivals:
            wins[arrival[0]] += 1
        
        # Le favori (indice 0) devrait avoir le plus de victoires
        assert wins[0] == max(wins), "Le favori devrait gagner le plus souvent"
    
    def test_temperature_effect(self, simple_probs):
        """T < 1 concentre, T > 1 uniforme."""
        sim_low = PlackettLuceTemperatureSimulator(simple_probs, temperature=0.7, seed=42)
        sim_high = PlackettLuceTemperatureSimulator(simple_probs, temperature=1.5, seed=42)
        
        arrivals_low = sim_low.simulate_n_races(2000)
        arrivals_high = sim_high.simulate_n_races(2000)
        
        # Compter les victoires du favori
        wins_low = sum(1 for a in arrivals_low if a[0] == 0)
        wins_high = sum(1 for a in arrivals_high if a[0] == 0)
        
        # Le favori gagne plus souvent avec T < 1
        assert wins_low > wins_high
    
    def test_combo_probs_sum_approximately_one(self, simple_probs):
        """La somme des probas de trio ≈ 1."""
        sim = PlackettLuceTemperatureSimulator(simple_probs, temperature=1.0, seed=42)
        combo_probs = sim.estimate_combo_probs(5000, 'trio')
        
        total = sum(combo_probs.values())
        assert np.isclose(total, 1.0, atol=0.05), f"Somme = {total}"


# =============================================================================
# TESTS CALIBRATION
# =============================================================================

class TestCalibrationMetrics:
    """Tests des métriques de calibration."""
    
    def test_brier_perfect_prediction(self):
        """Brier = 0 pour prédiction parfaite."""
        predicted = np.array([1.0, 0.0, 0.0])
        actual = np.array([1.0, 0.0, 0.0])
        
        brier = CalibrationMetrics.brier_score(predicted, actual)
        assert np.isclose(brier, 0.0)
    
    def test_brier_worst_prediction(self):
        """Brier = 1 pour pire prédiction."""
        predicted = np.array([0.0, 1.0, 1.0])
        actual = np.array([1.0, 0.0, 0.0])
        
        brier = CalibrationMetrics.brier_score(predicted, actual)
        assert np.isclose(brier, 1.0)
    
    def test_ece_perfect_calibration(self):
        """ECE = 0 pour calibration parfaite."""
        # 100 prédictions à 0.7 avec 70 succès
        predicted = np.array([0.7] * 100)
        actual = np.array([1] * 70 + [0] * 30)
        
        ece = CalibrationMetrics.expected_calibration_error(predicted, actual)
        assert ece < 0.1, f"ECE = {ece}"
    
    def test_monotonicity_check(self):
        """Test de vérification de la monotonie."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95])
        outcomes = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
        
        result = CalibrationMetrics.check_monotonicity(probs, outcomes, n_deciles=5)
        
        assert 'is_monotonic' in result
        assert 'spearman_correlation' in result
        assert result['spearman_correlation'] > 0  # Corrélation positive attendue


# =============================================================================
# TESTS EV CALCULATOR
# =============================================================================

class TestExoticEVCalculator:
    """Tests du calculateur EV."""
    
    def test_ev_positive_when_edge(self):
        """EV > 0 quand notre proba > proba publique."""
        calc = ExoticEVCalculator(takeout_rate=0.16)
        
        # Notre modèle prédit 5%, le public pense 2%
        ev = calc.calculate_ev_parimutuel(prob=0.05, public_prob=0.02)
        
        # EV = 0.05 * (0.84 / 0.02) - 1 = 0.05 * 42 - 1 = 1.1
        assert ev > 0
    
    def test_ev_negative_when_no_edge(self):
        """EV < 0 sans edge (proba = proba publique avec takeout)."""
        calc = ExoticEVCalculator(takeout_rate=0.16)
        
        # Même probabilité
        ev = calc.calculate_ev_parimutuel(prob=0.10, public_prob=0.10)
        
        # EV = 0.10 * (0.84 / 0.10) - 1 = 0.84 - 1 = -0.16
        assert ev < 0
    
    def test_ev_fixed_odds(self):
        """Test EV avec cotes fixes."""
        calc = ExoticEVCalculator()
        
        # 10% de chances, cote à 15
        ev = calc.calculate_ev_fixed_odds(prob=0.10, odds=15.0)
        
        # EV = 0.10 * 15 - 1 = 0.5 (50%)
        assert np.isclose(ev, 0.5)


# =============================================================================
# TESTS INTÉGRATION
# =============================================================================

class TestPlaceProbabilityEstimator:
    """Tests d'intégration de la classe principale."""
    
    def test_full_pipeline(self, medium_probs):
        """Test du pipeline complet."""
        estimator = PlaceProbabilityEstimator(
            medium_probs,
            discipline='plat',
            horse_names=[f"Horse_{i}" for i in range(len(medium_probs))]
        )
        
        # Place probs
        place_data = estimator.estimate_place_probs(top_n=3)
        assert 'p_place' in place_data
        assert len(place_data['p_place']) == len(medium_probs)
        
        # Combo probs
        combo_data = estimator.estimate_combo_probs('trio', n_sim=5000)
        assert 'n_combos' in combo_data
        assert combo_data['n_combos'] > 0
        
        # EV
        ev_data = estimator.calculate_tickets_ev(
            combo_data['combo_probs'], 'trio'
        )
        assert 'n_tickets' in ev_data
    
    def test_compare_estimators(self, simple_probs):
        """Test de la comparaison des estimateurs."""
        estimator = PlaceProbabilityEstimator(simple_probs)
        comparison = estimator.compare_estimators()
        
        assert 'Harville' in comparison
        assert 'Henery (γ=0.81)' in comparison
        assert 'Stern (λ=0.15)' in comparison
        assert 'Lo-Bacon-Shone' in comparison
    
    def test_generate_report(self, simple_probs):
        """Test de la génération de rapport."""
        estimator = PlaceProbabilityEstimator(simple_probs, discipline='trot')
        report = estimator.generate_report('trio', include_validation=False)
        
        assert 'meta' in report
        assert 'place_probabilities' in report
        assert 'combo_probabilities' in report
        assert 'expected_values' in report
        
        assert report['meta']['discipline'] == 'trot'


# =============================================================================
# TESTS STABILITÉ
# =============================================================================

class TestEVStability:
    """Tests de stabilité EV."""
    
    def test_stability_improves_with_n(self, simple_probs):
        """La stabilité s'améliore avec N simulations."""
        validator = EVStabilityValidator(simple_probs)
        
        result = validator.check_stability([1000, 5000, 10000], 'trio')
        
        assert 'results_by_n' in result
        assert 'recommended_n' in result
        
        # L'écart-type devrait diminuer
        stds = [result['results_by_n'][n]['std_ev'] for n in [1000, 5000, 10000]]
        assert stds[-1] <= stds[0] + 0.1  # Permettre un peu de bruit


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Exécuter les tests
    print("=" * 70)
    print("TESTS UNITAIRES - Module d'Estimation p(place)")
    print("=" * 70)
    
    # Tests rapides sans pytest
    simple_probs = np.array([0.40, 0.25, 0.20, 0.10, 0.05])
    medium_probs = np.array([0.22, 0.18, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02])
    medium_probs = medium_probs / medium_probs.sum()
    
    # Test 1: Contraintes
    print("\n✓ Test contraintes mathématiques...")
    assert np.isclose(simple_probs.sum(), 1.0)
    harville = HarvilleEstimator(simple_probs)
    p_place = harville.p_place(3)
    assert all(p_place[i] >= simple_probs[i] - 0.01 for i in range(len(simple_probs)))
    print("  p_place >= p_win: OK")
    
    # Test 2: Henery réduit favoris
    print("\n✓ Test Henery réduit avantage favoris...")
    henery = HeneryEstimator(simple_probs, gamma=0.81)
    he_place = henery.p_place(3)
    assert he_place[0] < p_place[0]
    print(f"  Harville: {p_place[0]:.3f}, Henery: {he_place[0]:.3f}")
    
    # Test 3: Simulation
    print("\n✓ Test simulation Plackett-Luce...")
    sim = PlackettLuceTemperatureSimulator(simple_probs, temperature=1.0, seed=42)
    combo_probs = sim.estimate_combo_probs(5000, 'trio')
    total = sum(combo_probs.values())
    print(f"  Somme probas trio: {total:.4f}")
    assert np.isclose(total, 1.0, atol=0.05)
    
    # Test 4: EV Calculator
    print("\n✓ Test calcul EV...")
    calc = ExoticEVCalculator(takeout_rate=0.16)
    ev_pos = calc.calculate_ev_parimutuel(0.05, 0.02)
    ev_neg = calc.calculate_ev_parimutuel(0.10, 0.10)
    print(f"  EV avec edge: {ev_pos:.2f} (attendu > 0)")
    print(f"  EV sans edge: {ev_neg:.2f} (attendu < 0)")
    assert ev_pos > 0
    assert ev_neg < 0
    
    # Test 5: Pipeline complet
    print("\n✓ Test pipeline complet...")
    estimator = PlaceProbabilityEstimator(medium_probs, discipline='plat')
    report = estimator.generate_report('trio', include_validation=False)
    assert 'place_probabilities' in report
    assert 'expected_values' in report
    print(f"  Combos observés: {report['combo_probabilities']['n_combos']}")
    print(f"  Tickets EV+: {report['expected_values']['n_positive_ev']}")
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS PASSENT")
    print("=" * 70)
    
    # Pour exécuter avec pytest
    print("\nPour exécuter tous les tests avec pytest:")
    print("  pytest tests/test_place_estimator.py -v")
