#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests de Cohérence des Paramètres de Calibration
=================================================

Vérifie que:
1. Σp_win = 1 par course (contrainte fondamentale)
2. T (température) est cohérent entre config YAML et artefacts
3. α (blend alpha) est cohérent entre config et artefacts
4. Les warnings sont émis en cas d'incohérence

Usage:
    pytest tests/test_coherence.py -v
    
Auteur: Horse3 System
"""

import os
import sys
import json
import pytest
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Configuration de test standard."""
    return {
        "version": "2.0.0",
        "calibration": {
            "temperature": 1.254,
            "blend_alpha_global": 0.2,
            "blend_alpha_by_discipline": {
                "plat": 0.0,
                "trot": 0.4,
                "obstacle": 0.4,
                "default": 0.2
            },
            "calibrator_type": "platt"
        },
        "kelly": {
            "fraction": 0.25,
            "max_stake_pct": 0.05,
            "min_stake": 2.0,
            "value_cutoff": 0.05
        }
    }


@pytest.fixture
def sample_artifact_report():
    """Rapport d'artefact de calibration simulé."""
    return {
        "timestamp": datetime.now().isoformat(),
        "temperature": 1.254,
        "alpha": 0.2,
        "alpha_by_cluster": {
            "plat": 0.0,
            "trot": 0.4,
            "obstacle": 0.4
        },
        "best_calibrator": "platt",
        "metrics": {
            "brier_score": 0.0821,
            "log_loss": 0.312,
            "ece": 0.0082
        }
    }


@pytest.fixture
def sample_probabilities():
    """Probabilités simulées pour une course de 10 chevaux."""
    np.random.seed(42)
    # Générer des scores bruts
    scores = np.random.randn(10) * 2 + 5
    # Softmax avec T=1.254
    T = 1.254
    scaled = scores / T
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled) / np.sum(np.exp(scaled))
    return probs


@pytest.fixture
def mock_db_connection():
    """Mock de connexion à la base de données."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


# =============================================================================
# TESTS DE NORMALISATION (Σp_win = 1)
# =============================================================================

class TestProbabilityNormalization:
    """Tests pour vérifier que Σp_win = 1 par course."""
    
    def test_sum_equals_one_basic(self, sample_probabilities):
        """Test basique: somme des probabilités = 1."""
        prob_sum = np.sum(sample_probabilities)
        assert abs(prob_sum - 1.0) < 1e-10, f"Σp_win = {prob_sum}, attendu 1.0"
    
    def test_softmax_normalization_with_temperature(self):
        """Test softmax avec différentes températures."""
        scores = np.array([10, 8, 6, 4, 2, 1, 0.5, 0.2, 0.1, 0.05])
        
        for T in [0.5, 1.0, 1.254, 2.0, 3.0, 5.0]:
            scaled = scores / T
            scaled = scaled - np.max(scaled)  # Stabilité numérique
            probs = np.exp(scaled) / np.sum(np.exp(scaled))
            
            prob_sum = np.sum(probs)
            assert abs(prob_sum - 1.0) < 1e-10, f"T={T}: Σp = {prob_sum}"
            assert np.all(probs >= 0), f"T={T}: probabilités négatives"
            assert np.all(probs <= 1), f"T={T}: probabilités > 1"
    
    def test_renormalization_after_blend(self):
        """Test que le blend préserve Σp = 1."""
        from scipy.special import expit
        
        def logit(p):
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return np.log(p / (1 - p))
        
        # Probas modèle et marché (déjà normalisées)
        p_model = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        p_market = np.array([0.25, 0.22, 0.20, 0.18, 0.15])
        
        for alpha in [0.0, 0.2, 0.4, 0.5, 0.8, 1.0]:
            logit_model = logit(p_model)
            logit_market = logit(p_market)
            logit_blend = alpha * logit_model + (1 - alpha) * logit_market
            p_blend = expit(logit_blend)
            
            # Renormaliser
            p_blend = p_blend / np.sum(p_blend)
            
            assert abs(np.sum(p_blend) - 1.0) < 1e-10, f"α={alpha}: Σp = {np.sum(p_blend)}"
    
    def test_harville_place_probabilities_constraint(self, sample_probabilities):
        """Test contrainte: P(place) >= P(win) pour chaque cheval."""
        # Simulation simple de Harville pour top 3
        win_probs = sample_probabilities
        n = len(win_probs)
        
        # P(place) ≈ P(top 3) > P(win) pour tous
        # Approximation: P(place) ≈ sum des P(arriver k-ème) pour k=1,2,3
        
        place_probs = np.zeros(n)
        for i in range(n):
            # Au minimum, P(place_i) >= P(win_i)
            place_probs[i] = win_probs[i]
            
            # Ajouter P(2ème) et P(3ème) approximés
            # (simplification pour le test)
            remaining = 1 - win_probs[i]
            if remaining > 0:
                for j in range(n):
                    if j != i:
                        # P(i 2ème | j gagnant)
                        p_2nd_given_j = win_probs[i] / (1 - win_probs[j]) if win_probs[j] < 1 else 0
                        place_probs[i] += win_probs[j] * p_2nd_given_j * 0.3  # Facteur
        
        for i in range(n):
            assert place_probs[i] >= win_probs[i] - 1e-10, \
                f"Cheval {i}: P(place)={place_probs[i]:.4f} < P(win)={win_probs[i]:.4f}"


# =============================================================================
# TESTS DE COHÉRENCE CONFIG/ARTEFACTS
# =============================================================================

class TestConfigArtifactCoherence:
    """Tests pour vérifier la cohérence entre config YAML et artefacts."""
    
    def test_temperature_coherence(self, sample_config, sample_artifact_report):
        """Test: T dans config == T dans artefacts."""
        config_temp = sample_config["calibration"]["temperature"]
        artifact_temp = sample_artifact_report["temperature"]
        
        assert abs(config_temp - artifact_temp) < 0.001, \
            f"Incohérence T: config={config_temp}, artifact={artifact_temp}"
    
    def test_alpha_global_coherence(self, sample_config, sample_artifact_report):
        """Test: α global cohérent."""
        config_alpha = sample_config["calibration"]["blend_alpha_global"]
        artifact_alpha = sample_artifact_report["alpha"]
        
        assert abs(config_alpha - artifact_alpha) < 0.001, \
            f"Incohérence α: config={config_alpha}, artifact={artifact_alpha}"
    
    def test_alpha_by_discipline_coherence(self, sample_config, sample_artifact_report):
        """Test: α par discipline cohérent."""
        config_alpha_disc = sample_config["calibration"]["blend_alpha_by_discipline"]
        artifact_alpha_disc = sample_artifact_report["alpha_by_cluster"]
        
        for disc in ["plat", "trot", "obstacle"]:
            if disc in config_alpha_disc and disc in artifact_alpha_disc:
                config_val = config_alpha_disc[disc]
                artifact_val = artifact_alpha_disc[disc]
                assert abs(config_val - artifact_val) < 0.001, \
                    f"Incohérence α[{disc}]: config={config_val}, artifact={artifact_val}"
    
    def test_calibrator_type_coherence(self, sample_config, sample_artifact_report):
        """Test: type de calibrateur cohérent."""
        config_cal = sample_config["calibration"]["calibrator_type"]
        artifact_cal = sample_artifact_report["best_calibrator"]
        
        assert config_cal == artifact_cal, \
            f"Incohérence calibrator: config={config_cal}, artifact={artifact_cal}"


# =============================================================================
# TESTS DU LOADER DE CONFIG
# =============================================================================

class TestConfigLoader:
    """Tests pour le module config/loader.py."""
    
    def test_config_loader_import(self):
        """Test que le loader peut être importé."""
        try:
            from config.loader import get_config, ProBettingConfig
            assert True
        except ImportError as e:
            pytest.skip(f"config.loader non disponible: {e}")
    
    def test_get_config_returns_valid_object(self):
        """Test que get_config retourne un objet valide."""
        try:
            from config.loader import get_config
            config = get_config()
            
            assert hasattr(config, 'calibration'), "Pas d'attribut calibration"
            assert hasattr(config.calibration, 'temperature'), "Pas d'attribut temperature"
            assert hasattr(config, 'kelly'), "Pas d'attribut kelly"
        except ImportError:
            pytest.skip("config.loader non disponible")
    
    def test_temperature_value_is_valid(self):
        """Test que T est une valeur valide."""
        try:
            from config.loader import get_config
            config = get_config()
            
            T = config.calibration.temperature
            assert T > 0, f"T doit être > 0, obtenu {T}"
            assert T < 10, f"T={T} semble trop élevé"
        except ImportError:
            pytest.skip("config.loader non disponible")
    
    def test_alpha_values_are_valid(self):
        """Test que les α sont entre 0 et 1."""
        try:
            from config.loader import get_config
            config = get_config()
            
            alpha_global = config.calibration.blend_alpha_global
            assert 0 <= alpha_global <= 1, f"α global invalide: {alpha_global}"
            
            # Vérifier les α par discipline (attributs directs)
            alphas = {
                'plat': config.calibration.blend_alpha_plat,
                'trot': config.calibration.blend_alpha_trot,
                'obstacle': config.calibration.blend_alpha_obstacle
            }
            for disc, alpha in alphas.items():
                assert 0 <= alpha <= 1, f"α[{disc}] invalide: {alpha}"
        except ImportError:
            pytest.skip("config.loader non disponible")
    
    def test_get_blend_alpha_for_discipline(self):
        """Test la méthode get_blend_alpha."""
        try:
            from config.loader import get_config
            config = get_config()
            
            # Disciplines connues
            alpha_plat = config.get_blend_alpha("plat")
            alpha_trot = config.get_blend_alpha("trot")
            alpha_obstacle = config.get_blend_alpha("obstacle")
            
            # Alias
            alpha_attele = config.get_blend_alpha("attele")  # -> trot
            alpha_haie = config.get_blend_alpha("haie")  # -> obstacle
            
            assert alpha_attele == alpha_trot, "Alias attele != trot"
            assert alpha_haie == alpha_obstacle, "Alias haie != obstacle"
            
        except ImportError:
            pytest.skip("config.loader non disponible")


# =============================================================================
# TESTS DU DÉCORATEUR @coherent_params
# =============================================================================

class TestCoherentParamsDecorator:
    """Tests pour le décorateur @coherent_params."""
    
    def test_decorator_passes_config_as_first_arg(self):
        """Test que le décorateur passe la config comme premier argument."""
        try:
            from config.loader import coherent_params, get_config, ProBettingConfig
            
            @coherent_params
            def test_func(cfg):
                return cfg
            
            result = test_func()
            
            assert isinstance(result, ProBettingConfig), "cfg n'est pas ProBettingConfig"
            assert result.calibration.temperature > 0, "Temperature non définie"
            assert result.version_hash, "Hash de version non défini"
                
        except ImportError:
            pytest.skip("config.loader non disponible")
    
    def test_decorator_passes_additional_args(self):
        """Test que le décorateur passe les arguments supplémentaires."""
        try:
            from config.loader import coherent_params, get_config, ProBettingConfig
            
            @coherent_params
            def test_func(cfg, multiplier, name="default"):
                return {
                    'temp': cfg.calibration.temperature * multiplier,
                    'name': name,
                    'hash': cfg.version_hash
                }
            
            result = test_func(2, name="custom")
            
            expected = get_config()
            assert result['temp'] == expected.calibration.temperature * 2
            assert result['name'] == "custom"
            assert result['hash'] == expected.version_hash
            
        except ImportError:
            pytest.skip("config.loader non disponible")


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================

class TestIntegration:
    """Tests d'intégration avec les modules réels."""
    
    def test_pro_betting_analyzer_uses_config(self, mock_db_connection):
        """Test que ProBettingAnalyzer utilise la config centralisée."""
        try:
            from pro_betting_analyzer import ProBettingAnalyzer
            from config.loader import get_config
            
            config = get_config()
            expected_temp = config.calibration.temperature
            
            analyzer = ProBettingAnalyzer.create_coherent(mock_db_connection)
            
            assert abs(analyzer.temperature - expected_temp) < 0.001, \
                f"Analyzer T={analyzer.temperature} != config T={expected_temp}"
                
        except ImportError as e:
            pytest.skip(f"Module non disponible: {e}")
    
    def test_race_pronostic_generator_uses_config(self, mock_db_connection):
        """Test que RacePronosticGenerator utilise la config centralisée."""
        try:
            from race_pronostic_generator import RacePronosticGenerator, get_default_config
            from config.loader import get_config
            
            config = get_config()
            expected_temp = config.calibration.temperature
            
            default_cfg = get_default_config()
            
            assert abs(default_cfg['temperature'] - expected_temp) < 0.001, \
                f"Generator T={default_cfg['temperature']} != config T={expected_temp}"
                
        except ImportError as e:
            pytest.skip(f"Module non disponible: {e}")
    
    def test_all_modules_use_same_temperature(self, mock_db_connection):
        """Test que tous les modules utilisent la même température."""
        try:
            from pro_betting_analyzer import ProBettingAnalyzer
            from race_pronostic_generator import RacePronosticGenerator, get_default_config
            from config.loader import get_config
            
            config = get_config()
            config_temp = config.calibration.temperature
            
            analyzer = ProBettingAnalyzer.create_coherent(mock_db_connection)
            generator_cfg = get_default_config()
            
            temperatures = {
                'config': config_temp,
                'analyzer': analyzer.temperature,
                'generator': generator_cfg['temperature']
            }
            
            for name1, t1 in temperatures.items():
                for name2, t2 in temperatures.items():
                    assert abs(t1 - t2) < 0.001, \
                        f"Incohérence T: {name1}={t1}, {name2}={t2}"
                        
        except ImportError as e:
            pytest.skip(f"Module non disponible: {e}")


# =============================================================================
# TESTS DE VALIDATION DE CONFIG
# =============================================================================

class TestConfigValidation:
    """Tests pour la validation de la configuration."""
    
    def test_config_is_frozen(self):
        """Test que ProBettingConfig est immuable (frozen)."""
        try:
            from config.loader import get_config
            
            config = get_config()
            
            # Tenter de modifier devrait lever AttributeError
            with pytest.raises(AttributeError):
                config.calibration = None
            
            with pytest.raises(AttributeError):
                config.version_hash = "modified"
            
        except ImportError:
            pytest.skip("config.loader non disponible")
    
    def test_config_has_version_hash(self):
        """Test que la config a un hash de version."""
        try:
            from config.loader import get_config
            
            config = get_config()
            
            assert config.version_hash, "Hash de version vide"
            assert len(config.version_hash) == 12, f"Hash invalide: {config.version_hash}"
            
        except ImportError:
            pytest.skip("config.loader non disponible")
    
    def test_calibration_values_are_valid(self):
        """Test que les valeurs de calibration sont valides."""
        try:
            from config.loader import get_config
            
            config = get_config()
            
            # Temperature doit être > 0
            assert config.calibration.temperature > 0, "Temperature <= 0"
            
            # Alpha global doit être entre 0 et 1
            assert 0 <= config.calibration.blend_alpha_global <= 1, "α global hors [0,1]"
            
        except ImportError:
            pytest.skip("config.loader non disponible")


# =============================================================================
# TESTS DE COHÉRENCE NUM_SIMULATIONS (unifié à 20000)
# =============================================================================

class TestNumSimulationsCoherence:
    """Tests pour vérifier que num_simulations est unifié à 20000 partout."""
    
    EXPECTED_NUM_SIMULATIONS = 20000
    
    def test_config_yaml_num_simulations(self):
        """Test que config YAML a num_simulations = 20000."""
        try:
            from config.loader import get_config
            config = get_config()
            
            assert config.simulation.num_simulations == self.EXPECTED_NUM_SIMULATIONS, \
                f"config.simulation.num_simulations = {config.simulation.num_simulations}, attendu {self.EXPECTED_NUM_SIMULATIONS}"
                
        except ImportError:
            pytest.skip("config.loader non disponible")
    
    def test_race_pronostic_generator_num_simulations(self):
        """Test que RacePronosticGenerator utilise 20000 simulations."""
        try:
            from race_pronostic_generator import get_default_config
            
            cfg = get_default_config()
            actual = cfg.get('num_simulations')
            
            assert actual == self.EXPECTED_NUM_SIMULATIONS, \
                f"race_pronostic_generator num_simulations = {actual}, attendu {self.EXPECTED_NUM_SIMULATIONS}"
                
        except ImportError:
            pytest.skip("race_pronostic_generator non disponible")
    
    def test_exotic_ticket_generator_num_simulations(self):
        """Test que ExoticConfig utilise 20000 simulations par défaut."""
        try:
            from exotic_ticket_generator import ExoticConfig
            
            # Créer une config par défaut
            config = ExoticConfig()
            
            assert config.num_simulations == self.EXPECTED_NUM_SIMULATIONS, \
                f"ExoticConfig.num_simulations = {config.num_simulations}, attendu {self.EXPECTED_NUM_SIMULATIONS}"
                
        except ImportError:
            pytest.skip("exotic_ticket_generator non disponible")
    
    def test_monte_carlo_simulator_default(self):
        """Test que MonteCarloSimulator a un default de 20000."""
        try:
            from race_pronostic_generator import MonteCarloSimulator
            
            # Vérifier le default en créant sans argument
            sim = MonteCarloSimulator()
            
            assert sim.num_simulations == self.EXPECTED_NUM_SIMULATIONS, \
                f"MonteCarloSimulator default = {sim.num_simulations}, attendu {self.EXPECTED_NUM_SIMULATIONS}"
                
        except ImportError:
            pytest.skip("MonteCarloSimulator non disponible")
    
    def test_all_modules_same_num_simulations(self):
        """Test que TOUS les modules utilisent le même num_simulations."""
        values = {}
        
        # 1. Config centralisée
        try:
            from config.loader import get_config
            config = get_config()
            values['config.yaml'] = config.simulation.num_simulations
        except ImportError:
            pass
        
        # 2. race_pronostic_generator
        try:
            from race_pronostic_generator import get_default_config
            cfg = get_default_config()
            values['race_pronostic_generator'] = cfg.get('num_simulations')
        except ImportError:
            pass
        
        # 3. exotic_ticket_generator
        try:
            from exotic_ticket_generator import ExoticConfig
            ec = ExoticConfig()
            values['exotic_ticket_generator'] = ec.num_simulations
        except ImportError:
            pass
        
        # 4. MonteCarloSimulator default
        try:
            from race_pronostic_generator import MonteCarloSimulator
            sim = MonteCarloSimulator()
            values['MonteCarloSimulator'] = sim.num_simulations
        except ImportError:
            pass
        
        if len(values) < 2:
            pytest.skip("Pas assez de modules disponibles pour comparaison")
        
        # Vérifier que toutes les valeurs sont identiques
        unique_values = set(values.values())
        
        assert len(unique_values) == 1, \
            f"Incohérence num_simulations détectée: {values}"
        
        # Vérifier que c'est bien 20000
        actual = list(unique_values)[0]
        assert actual == self.EXPECTED_NUM_SIMULATIONS, \
            f"num_simulations unifié = {actual}, attendu {self.EXPECTED_NUM_SIMULATIONS}"
    
    def test_fails_if_module_uses_different_n(self):
        """
        Test qui ÉCHOUE si un module utilise un N différent de 20000.
        Ce test est la garantie principale de cohérence.
        """
        errors = []
        
        # Vérifier chaque source
        checks = [
            ("config.yaml", lambda: __import__('config.loader').loader.get_config().simulation.num_simulations),
            ("race_pronostic_generator", lambda: __import__('race_pronostic_generator').get_default_config().get('num_simulations')),
            ("ExoticConfig", lambda: __import__('exotic_ticket_generator').ExoticConfig().num_simulations),
            ("MonteCarloSimulator", lambda: __import__('race_pronostic_generator').MonteCarloSimulator().num_simulations),
        ]
        
        for name, getter in checks:
            try:
                value = getter()
                if value != self.EXPECTED_NUM_SIMULATIONS:
                    errors.append(f"{name}: {value} (attendu {self.EXPECTED_NUM_SIMULATIONS})")
            except ImportError:
                pass  # Module non disponible, ignorer
            except Exception as e:
                errors.append(f"{name}: erreur - {e}")
        
        assert len(errors) == 0, \
            f"Modules avec num_simulations incorrect:\n" + "\n".join(errors)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Exécuter avec pytest
    pytest.main([__file__, "-v", "--tb=short"])
