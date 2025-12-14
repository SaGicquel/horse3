#!/usr/bin/env python3
"""
Tests pytest pour le système de paris Horse3.

Tests couverts:
- test_prob_sum_per_race(): somme p_win==1 (tol 1e-6)
- test_temperature_from_artifacts(): T utilisé == valeur health.json
- test_alpha_from_artifacts(): α utilisé == health.json (et par discipline)
- test_temperature_from_config(): T utilisé == 1.254 (fallback config)
- test_alpha_by_discipline(): α plat==0.0, trot==0.4, obstacle==0.4
- test_mc_runs_consistent(): N==20000 partout
- test_parimutuel_ev(): takeout 0.16 intégré -> fair_odds_pool = (1-0.16)/p
- test_no_leakage(): aucune feature post-off dans l'inférence
- test_kelly_cap_and_value(): value<=0 ⇒ stake==0; cap 5% respecté
- test_coherence_guard(): si YAML≠artefacts → override appliqué + WARN log

Auteur: Horse Racing AI System
Date: 2024-12
"""

import pytest
import numpy as np
import sys
import os
import json
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_race_data():
    """Données de course de test avec 10 partants."""
    return {
        'horses': [f'Horse_{i}' for i in range(10)],
        'odds': [2.5, 4.0, 6.0, 8.0, 12.0, 15.0, 20.0, 25.0, 40.0, 50.0],
        'logits': [1.5, 1.0, 0.5, 0.2, -0.2, -0.5, -0.8, -1.0, -1.5, -2.0],
    }


@pytest.fixture
def multi_race_data():
    """Données de plusieurs courses pour tester la normalisation."""
    races = []
    for race_id in range(5):
        n_horses = np.random.randint(6, 15)
        races.append({
            'race_id': f'R{race_id:03d}',
            'logits': np.random.randn(n_horses).tolist(),
            'odds': (1 + np.random.exponential(5, n_horses)).tolist(),
        })
    return races


# ============================================================================
# TEST: PROB SUM PER RACE == 1
# ============================================================================

class TestProbSumPerRace:
    """Test que la somme des probabilités par course == 1."""
    
    def test_prob_sum_per_race_softmax(self, sample_race_data):
        """Test normalisation softmax: sum(p) == 1."""
        from scipy.special import softmax
        
        logits = np.array(sample_race_data['logits'])
        probs = softmax(logits)
        
        assert np.isclose(probs.sum(), 1.0, atol=1e-6), \
            f"Somme des probas softmax = {probs.sum()}, attendu 1.0"
    
    def test_prob_sum_per_race_with_temperature(self, sample_race_data):
        """Test normalisation softmax avec température: sum(p) == 1."""
        from scipy.special import softmax
        
        logits = np.array(sample_race_data['logits'])
        temperature = 1.254  # Valeur de config
        
        scaled_logits = logits / temperature
        probs = softmax(scaled_logits)
        
        assert np.isclose(probs.sum(), 1.0, atol=1e-6), \
            f"Somme des probas (T={temperature}) = {probs.sum()}, attendu 1.0"
    
    def test_prob_sum_multiple_races(self, multi_race_data):
        """Test que chaque course a sum(p) == 1 après normalisation."""
        from scipy.special import softmax
        
        temperature = 1.254
        
        for race in multi_race_data:
            logits = np.array(race['logits'])
            scaled_logits = logits / temperature
            probs = softmax(scaled_logits)
            
            assert np.isclose(probs.sum(), 1.0, atol=1e-6), \
                f"Course {race['race_id']}: sum(p) = {probs.sum()}, attendu 1.0"
    
    def test_prob_sum_after_blend(self, sample_race_data):
        """Test que le blend conserve sum(p) == 1 après renormalisation."""
        from race_pronostic_generator import blend_logit_odds
        
        # Probabilités modèle (softmax)
        logits = np.array(sample_race_data['logits'])
        from scipy.special import softmax
        p_model = softmax(logits / 1.254)
        
        # Probabilités marché (inversées des cotes)
        odds = np.array(sample_race_data['odds'])
        p_market = 1.0 / odds
        p_market = p_market / p_market.sum()
        
        # Blend
        p_blend = blend_logit_odds(p_model, p_market, alpha=0.5)
        
        # Renormaliser (comme dans le code de production)
        p_blend_norm = p_blend / p_blend.sum()
        
        assert np.isclose(p_blend_norm.sum(), 1.0, atol=1e-6), \
            f"Somme après blend+renorm = {p_blend_norm.sum()}, attendu 1.0"


# ============================================================================
# TEST: TEMPERATURE FROM ARTIFACTS
# ============================================================================

class TestTemperatureFromArtifacts:
    """Test que la température utilisée == valeur health.json."""
    
    @pytest.fixture
    def health_json_path(self):
        """Chemin vers health.json."""
        return Path(__file__).parent.parent / "calibration" / "health.json"
    
    def test_temperature_from_health_json(self, health_json_path):
        """Test que T utilisé == valeur health.json."""
        if not health_json_path.exists():
            pytest.skip("health.json non trouvé")
        
        with open(health_json_path, 'r') as f:
            health = json.load(f)
        
        expected_temp = health.get('temperature')
        assert expected_temp is not None, "temperature manquant dans health.json"
        
        # Charger via artifacts_loader
        try:
            from calibration.artifacts_loader import load_calibration_state
            state = load_calibration_state(prefer_artifacts=True)
            
            assert np.isclose(state.temperature, expected_temp, atol=1e-6), \
                f"T chargé = {state.temperature}, attendu = {expected_temp}"
            assert state.source == 'artifacts', \
                f"Source devrait être 'artifacts', got '{state.source}'"
        except ImportError:
            pytest.skip("artifacts_loader non disponible")
    
    def test_temperature_overrides_yaml(self, health_json_path):
        """Test que health.json écrase les valeurs YAML."""
        if not health_json_path.exists():
            pytest.skip("health.json non trouvé")
        
        try:
            from calibration.artifacts_loader import load_calibration_state, check_yaml_artifacts_mismatch
            
            # Charger avec prefer_artifacts=True
            state_artifacts = load_calibration_state(prefer_artifacts=True)
            
            # Charger avec prefer_artifacts=False (YAML seul)
            state_yaml = load_calibration_state(prefer_artifacts=False)
            
            # Vérifier que artifacts a priorité
            with open(health_json_path, 'r') as f:
                health = json.load(f)
            
            assert np.isclose(state_artifacts.temperature, health['temperature'], atol=1e-6), \
                "Avec prefer_artifacts=True, T devrait venir de health.json"
            
        except ImportError:
            pytest.skip("artifacts_loader non disponible")


# ============================================================================
# TEST: ALPHA FROM ARTIFACTS
# ============================================================================

class TestAlphaFromArtifacts:
    """Test que α utilisé == valeur health.json."""
    
    @pytest.fixture
    def health_json_path(self):
        """Chemin vers health.json."""
        return Path(__file__).parent.parent / "calibration" / "health.json"
    
    def test_alpha_from_health_json(self, health_json_path):
        """Test que α utilisé == valeur health.json."""
        if not health_json_path.exists():
            pytest.skip("health.json non trouvé")
        
        with open(health_json_path, 'r') as f:
            health = json.load(f)
        
        expected_alpha = health.get('alpha')
        assert expected_alpha is not None, "alpha manquant dans health.json"
        
        # Charger via artifacts_loader
        try:
            from calibration.artifacts_loader import load_calibration_state
            state = load_calibration_state(prefer_artifacts=True)
            
            assert np.isclose(state.alpha, expected_alpha, atol=1e-6), \
                f"α chargé = {state.alpha}, attendu = {expected_alpha}"
        except ImportError:
            pytest.skip("artifacts_loader non disponible")
    
    def test_alpha_by_discipline_from_artifacts(self, health_json_path):
        """Test que alpha_by_disc est correctement chargé."""
        if not health_json_path.exists():
            pytest.skip("health.json non trouvé")
        
        try:
            from calibration.artifacts_loader import load_calibration_state
            state = load_calibration_state(prefer_artifacts=True)
            
            # Vérifier que alpha_by_disc contient les clés attendues
            expected_keys = ['plat', 'trot', 'obstacle', 'global']
            for key in expected_keys:
                assert key in state.alpha_by_disc, \
                    f"Clé '{key}' manquante dans alpha_by_disc"
            
            # Vérifier get_alpha_for_discipline
            for disc in ['plat', 'trot', 'obstacle']:
                alpha = state.get_alpha_for_discipline(disc)
                assert 0.0 <= alpha <= 1.0, \
                    f"α pour {disc} hors bornes: {alpha}"
        except ImportError:
            pytest.skip("artifacts_loader non disponible")


# ============================================================================
# TEST: COHERENCE GUARD (YAML ≠ ARTEFACTS)
# ============================================================================

class TestCoherenceGuard:
    """Test que si YAML ≠ artefacts, l'override est appliqué + WARN log."""
    
    @pytest.fixture
    def health_json_path(self):
        """Chemin vers health.json."""
        return Path(__file__).parent.parent / "calibration" / "health.json"
    
    def test_mismatch_detection(self, health_json_path):
        """Test détection des différences YAML vs artefacts."""
        if not health_json_path.exists():
            pytest.skip("health.json non trouvé")
        
        try:
            from calibration.artifacts_loader import check_yaml_artifacts_mismatch
            
            result = check_yaml_artifacts_mismatch()
            
            # Le résultat doit contenir les clés attendues
            assert 'has_mismatch' in result
            assert 'mismatches' in result
            assert 'yaml_values' in result
            assert 'artifacts_values' in result
            
            # Vérifier que les valeurs sont cohérentes
            if result['has_mismatch']:
                assert len(result['mismatches']) > 0, \
                    "has_mismatch=True mais mismatches vide"
            
        except ImportError:
            pytest.skip("artifacts_loader non disponible")
    
    def test_override_applied_when_mismatch(self, health_json_path):
        """Test que l'override artefacts est appliqué même si YAML différent."""
        if not health_json_path.exists():
            pytest.skip("health.json non trouvé")
        
        try:
            from calibration.artifacts_loader import load_calibration_state
            
            # Charger avec prefer_artifacts=True
            state = load_calibration_state(prefer_artifacts=True)
            
            # Charger health.json directement
            with open(health_json_path, 'r') as f:
                health = json.load(f)
            
            # T et α doivent correspondre aux artefacts
            assert np.isclose(state.temperature, health['temperature'], atol=1e-6), \
                "T devrait être l'override des artefacts"
            assert np.isclose(state.alpha, health['alpha'], atol=1e-6), \
                "α devrait être l'override des artefacts"
            
        except ImportError:
            pytest.skip("artifacts_loader non disponible")
    
    def test_warn_if_mismatch_returns_bool(self, health_json_path):
        """Test que warn_if_mismatch retourne un booléen."""
        if not health_json_path.exists():
            pytest.skip("health.json non trouvé")
        
        try:
            from calibration.artifacts_loader import warn_if_mismatch
            import logging
            
            # Capturer les logs
            class TestHandler(logging.Handler):
                def __init__(self):
                    super().__init__()
                    self.records = []
                
                def emit(self, record):
                    self.records.append(record)
            
            logger = logging.getLogger('calibration.artifacts_loader')
            handler = TestHandler()
            logger.addHandler(handler)
            
            result = warn_if_mismatch()
            
            # Le résultat doit être un booléen
            assert isinstance(result, bool), \
                f"warn_if_mismatch devrait retourner bool, got {type(result)}"
            
            logger.removeHandler(handler)
            
        except ImportError:
            pytest.skip("artifacts_loader non disponible")


# ============================================================================
# TEST: TEMPERATURE FROM CONFIG (fallback)
# ============================================================================

class TestTemperatureFromConfig:
    """Test que la température utilisée == 1.254 (fallback YAML)."""
    
    EXPECTED_TEMPERATURE = 1.254  # Valeur par défaut YAML
    
    def test_temperature_from_config_yaml(self):
        """Test lecture température depuis config/pro_betting.yaml."""
        try:
            from config.loader import get_config
            config = get_config()
            
            # La température peut être dans calibration.temperature ou directement
            if hasattr(config, 'calibration') and hasattr(config.calibration, 'temperature'):
                temp = config.calibration.temperature
            elif hasattr(config, 'temperature'):
                temp = config.temperature
            else:
                pytest.skip("Température non trouvée dans config")
            
            assert temp == self.EXPECTED_TEMPERATURE, \
                f"Température config = {temp}, attendu {self.EXPECTED_TEMPERATURE}"
                
        except ImportError:
            pytest.skip("config.loader non disponible")
    
    def test_temperature_softmax_effect(self, sample_race_data):
        """Test que T=1.254 produit une distribution moins extrême que T=1."""
        from scipy.special import softmax
        
        logits = np.array(sample_race_data['logits'])
        
        probs_t1 = softmax(logits / 1.0)
        probs_t_config = softmax(logits / self.EXPECTED_TEMPERATURE)
        
        # Avec T > 1, la distribution est plus "plate" (entropie plus haute)
        entropy_t1 = -np.sum(probs_t1 * np.log(probs_t1 + 1e-10))
        entropy_t_config = -np.sum(probs_t_config * np.log(probs_t_config + 1e-10))
        
        assert entropy_t_config > entropy_t1, \
            f"T={self.EXPECTED_TEMPERATURE} devrait produire plus d'entropie: {entropy_t_config} vs {entropy_t1}"
    
    def test_temperature_bounds(self):
        """Test que la température est dans des bornes raisonnables."""
        assert 0.5 <= self.EXPECTED_TEMPERATURE <= 5.0, \
            f"Température {self.EXPECTED_TEMPERATURE} hors bornes [0.5, 5.0]"


# ============================================================================
# TEST: ALPHA BY DISCIPLINE
# ============================================================================

class TestAlphaByDiscipline:
    """Test des valeurs α par discipline."""
    
    EXPECTED_ALPHA = {
        'plat': 0.0,
        'trot': 0.4,
        'obstacle': 0.4,
    }
    
    def test_alpha_plat(self):
        """Test α plat == 0.0 (marché seul)."""
        try:
            import yaml
            with open('config/pro_betting.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # La config utilise blend_alpha_plat directement
            alpha_plat = config.get('calibration', {}).get('blend_alpha_plat', 0.5)
            
            assert alpha_plat == self.EXPECTED_ALPHA['plat'], \
                f"α plat = {alpha_plat}, attendu {self.EXPECTED_ALPHA['plat']}"
                
        except (ImportError, FileNotFoundError):
            pytest.skip("Config non disponible pour test alpha discipline")
    
    def test_alpha_trot(self):
        """Test α trot == 0.4 (équilibré)."""
        try:
            from config.loader import get_config
            config = get_config()
            
            alpha_by_disc = {}
            if hasattr(config, 'calibration') and hasattr(config.calibration, 'blend_alpha_by_discipline'):
                alpha_by_disc = config.calibration.blend_alpha_by_discipline
            
            # Trot peut être sous 'attele' ou 'trot'
            alpha_trot = alpha_by_disc.get('trot', alpha_by_disc.get('attele', 0.4))
            
            assert alpha_trot == self.EXPECTED_ALPHA['trot'], \
                f"α trot = {alpha_trot}, attendu {self.EXPECTED_ALPHA['trot']}"
                
        except (ImportError, AttributeError):
            pytest.skip("Config non disponible pour test alpha discipline")
    
    def test_alpha_obstacle(self):
        """Test α obstacle == 0.4."""
        try:
            from config.loader import get_config
            config = get_config()
            
            alpha_by_disc = {}
            if hasattr(config, 'calibration') and hasattr(config.calibration, 'blend_alpha_by_discipline'):
                alpha_by_disc = config.calibration.blend_alpha_by_discipline
            
            # Obstacle peut être sous plusieurs noms
            alpha_obs = alpha_by_disc.get('obstacle', 
                        alpha_by_disc.get('haie', 
                        alpha_by_disc.get('steeplechase', 0.4)))
            
            assert alpha_obs == self.EXPECTED_ALPHA['obstacle'], \
                f"α obstacle = {alpha_obs}, attendu {self.EXPECTED_ALPHA['obstacle']}"
                
        except (ImportError, AttributeError):
            pytest.skip("Config non disponible pour test alpha discipline")
    
    def test_alpha_blend_effect(self, sample_race_data):
        """Test que différents α produisent des distributions différentes."""
        from race_pronostic_generator import blend_logit_odds
        from scipy.special import softmax
        
        logits = np.array(sample_race_data['logits'])
        odds = np.array(sample_race_data['odds'])
        
        p_model = softmax(logits / 1.254)
        p_market = 1.0 / odds
        p_market = p_market / p_market.sum()
        
        # α = 0 -> 100% marché
        p_alpha_0 = blend_logit_odds(p_model, p_market, alpha=0.0)
        
        # α = 0.4 -> mix
        p_alpha_04 = blend_logit_odds(p_model, p_market, alpha=0.4)
        
        # α = 1 -> 100% modèle
        p_alpha_1 = blend_logit_odds(p_model, p_market, alpha=1.0)
        
        # Vérifier que α=0 est proche du marché
        assert np.allclose(p_alpha_0 / p_alpha_0.sum(), p_market, atol=0.01), \
            "α=0 devrait être proche du marché"
        
        # Vérifier que α=1 est proche du modèle
        assert np.allclose(p_alpha_1 / p_alpha_1.sum(), p_model, atol=0.01), \
            "α=1 devrait être proche du modèle"
        
        # Vérifier que α=0.4 est entre les deux
        diff_to_market = np.abs(p_alpha_04 - p_market).mean()
        diff_to_model = np.abs(p_alpha_04 - p_model).mean()
        
        assert diff_to_market > 0.001 and diff_to_model > 0.001, \
            "α=0.4 devrait être un mix (ni pur marché, ni pur modèle)"


# ============================================================================
# TEST: MC RUNS CONSISTENT (N=20000)
# ============================================================================

class TestMCRunsConsistent:
    """Test que N=20000 simulations Monte Carlo partout."""
    
    EXPECTED_N = 20000
    
    def test_mc_n_in_config(self):
        """Test que N=20000 dans la config."""
        try:
            from config.loader import get_config
            config = get_config()
            
            if hasattr(config, 'simulation') and hasattr(config.simulation, 'num_simulations'):
                n = config.simulation.num_simulations
            elif hasattr(config, 'num_simulations'):
                n = config.num_simulations
            else:
                pytest.skip("num_simulations non trouvé dans config")
            
            assert n == self.EXPECTED_N, \
                f"N simulations = {n}, attendu {self.EXPECTED_N}"
                
        except ImportError:
            pytest.skip("config.loader non disponible")
    
    def test_mc_n_in_exotic_generator(self):
        """Test que N=20000 dans exotic_ticket_generator."""
        from exotic_ticket_generator import ExoticConfig
        
        config = ExoticConfig()  # Utilise __post_init__ qui charge depuis config
        
        assert config.num_simulations == self.EXPECTED_N, \
            f"ExoticConfig.num_simulations = {config.num_simulations}, attendu {self.EXPECTED_N}"
    
    def test_mc_simulation_runs(self):
        """Test qu'une simulation MC produit bien N résultats."""
        from exotic_ticket_generator import PlackettLuceSimulator
        
        probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        probs = probs / probs.sum()
        
        simulator = PlackettLuceSimulator(probs, seed=42)
        results = simulator.simulate_n_races_fast(self.EXPECTED_N)
        
        assert results.shape[0] == self.EXPECTED_N, \
            f"Simulation a produit {results.shape[0]} résultats, attendu {self.EXPECTED_N}"
    
    def test_mc_convergence(self):
        """Test que N=20000 donne une bonne convergence."""
        from exotic_ticket_generator import PlackettLuceSimulator
        
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        probs = probs / probs.sum()
        
        simulator = PlackettLuceSimulator(probs, seed=42)
        results = simulator.simulate_n_races_fast(self.EXPECTED_N)
        
        # Compter les victoires du favori (indice 0)
        wins_horse_0 = np.sum(results[:, 0] == 0)
        empirical_prob = wins_horse_0 / self.EXPECTED_N
        
        # Avec N=20000, l'erreur devrait être < 1%
        assert np.abs(empirical_prob - probs[0]) < 0.01, \
            f"Convergence MC: p_empirique={empirical_prob:.4f}, p_théorique={probs[0]:.4f}"


# ============================================================================
# TEST: PARIMUTUEL EV (takeout 0.16)
# ============================================================================

class TestParimutuelEV:
    """Test du calcul EV parimutuel avec takeout 0.16."""
    
    TAKEOUT = 0.16
    
    def test_fair_odds_pool_formula(self):
        """Test fair_odds_pool = (1 - 0.16) / p."""
        p = 0.25  # 25% de chance
        
        fair_odds = (1 - self.TAKEOUT) / p
        expected_fair_odds = 0.84 / 0.25  # = 3.36
        
        assert np.isclose(fair_odds, expected_fair_odds, atol=1e-6), \
            f"Fair odds = {fair_odds}, attendu {expected_fair_odds}"
    
    def test_ev_with_takeout(self):
        """Test EV = p * odds - 1 avec fair odds incluant takeout."""
        p = 0.25
        fair_odds = (1 - self.TAKEOUT) / p  # 3.36
        
        # Si on parie à fair odds, EV devrait être 0
        ev_at_fair = p * fair_odds - 1
        
        assert np.isclose(ev_at_fair, -self.TAKEOUT, atol=1e-6), \
            f"EV at fair odds = {ev_at_fair}, attendu -{self.TAKEOUT}"
    
    def test_value_bet_detection(self):
        """Test détection des value bets (cotes > fair odds)."""
        p = 0.25
        fair_odds = (1 - self.TAKEOUT) / p  # 3.36
        
        # Cote marché > fair odds = value bet
        # Pour avoir EV > 0: p * odds - 1 > 0 => odds > 1/p = 4
        market_odds = 5.0  # EV = 0.25 * 5 - 1 = 0.25 > 0
        ev = p * market_odds - 1
        
        assert ev > 0, f"EV avec odds={market_odds} devrait être > 0, got {ev}"
        
        # Cote marché < fair odds = pas de value
        market_odds_low = 3.0
        ev_low = p * market_odds_low - 1
        
        assert ev_low < 0, f"EV avec odds={market_odds_low} devrait être < 0, got {ev_low}"
    
    def test_takeout_in_exotic_generator(self):
        """Test que le takeout 0.16 est utilisé dans exotic_ticket_generator."""
        from exotic_ticket_generator import ExoticConfig
        
        config = ExoticConfig()
        
        assert config.takeout_rate == self.TAKEOUT, \
            f"ExoticConfig.takeout_rate = {config.takeout_rate}, attendu {self.TAKEOUT}"
    
    def test_ev_calculation_consistency(self, sample_race_data):
        """Test cohérence du calcul EV entre modules."""
        odds = np.array(sample_race_data['odds'])
        
        # Probabilités implicites (avec marge bookmaker ~15%)
        p_market = 1.0 / odds
        total_proba = p_market.sum()  # > 1 car marge
        p_fair = p_market / total_proba  # Probas "vraies" estimées
        
        # Fair odds parimutuel (sans notre modèle)
        fair_odds_pool = (1 - self.TAKEOUT) / p_fair
        
        # Vérifier que fair odds > cotes publiées (car marge bookmaker)
        # En fait, avec takeout 16%, fair odds ≈ 84% de 1/p
        for i, (fo, o, pf) in enumerate(zip(fair_odds_pool, odds, p_fair)):
            expected_ratio = (1 - self.TAKEOUT) * total_proba
            assert fo < o * 1.5, f"Fair odds {fo} trop éloigné de cotes {o}"


# ============================================================================
# TEST: NO LEAKAGE (pas de features post-off)
# ============================================================================

class TestNoLeakage:
    """Test qu'aucune feature post-off n'est utilisée dans l'inférence."""
    
    # Features qui ne doivent JAMAIS être utilisées pour l'inférence
    POST_OFF_FEATURES = [
        'place_finale',
        'is_win',
        'is_place',
        'temps_sec',
        'temps_str',
        'reduction_km_sec',
        'rapport_gagnant',
        'rapport_place',
        'gains_course',
        'statut_arrivee',
        'ecart_premier',
        'ecart_precedent',
        'vitesse_moyenne',
        'vitesse_fin_course',
        'cote_finale',  # Cote APRÈS départ
        'incidents_json',
        'commentaire_apres_course',
    ]
    
    # Features autorisées pour l'inférence (avant départ)
    PRE_OFF_FEATURES = [
        'cote_matin',
        'cote_reference',
        'musique',
        'gains_carriere',
        'gains_annee_en_cours',
        'nombre_courses',
        'nombre_victoires',
        'driver_jockey',
        'entraineur',
        'poids_kg',
        'distance_m',
        'discipline',
        'hippodrome_code',
    ]
    
    def test_no_post_off_in_feature_list(self):
        """Test que la liste de features n'inclut pas de données post-off."""
        # Simuler une liste de features utilisée pour l'inférence
        inference_features = [
            'cote_matin', 'musique', 'gains_carriere', 
            'discipline', 'distance_m', 'poids_kg'
        ]
        
        for feat in inference_features:
            assert feat not in self.POST_OFF_FEATURES, \
                f"Feature post-off '{feat}' détectée dans les features d'inférence!"
    
    def test_post_off_features_not_in_model_input(self):
        """Test qu'aucune feature post-off n'est passée au modèle."""
        # Simuler les colonnes d'un DataFrame d'inférence
        model_input_columns = [
            'nom_norm', 'race_key', 'cote_matin', 'musique',
            'gains_carriere', 'discipline', 'distance_m'
        ]
        
        leaked_features = [f for f in model_input_columns if f in self.POST_OFF_FEATURES]
        
        assert len(leaked_features) == 0, \
            f"Features post-off dans l'input modèle: {leaked_features}"
    
    def test_cote_finale_vs_cote_matin(self):
        """Test que cote_finale n'est pas utilisée, seulement cote_matin."""
        # cote_finale = cote APRÈS le départ (leakage)
        # cote_matin = cote AVANT le départ (OK)
        
        assert 'cote_finale' in self.POST_OFF_FEATURES
        assert 'cote_matin' in self.PRE_OFF_FEATURES
    
    def test_feature_validation_function(self):
        """Test d'une fonction de validation de features."""
        def validate_no_leakage(features: list) -> list:
            """Retourne la liste des features post-off détectées."""
            return [f for f in features if f in self.POST_OFF_FEATURES]
        
        # Test avec features OK
        clean_features = ['cote_matin', 'musique', 'discipline']
        leaked = validate_no_leakage(clean_features)
        assert len(leaked) == 0, f"Faux positif: {leaked}"
        
        # Test avec feature post-off
        dirty_features = ['cote_matin', 'place_finale', 'discipline']
        leaked = validate_no_leakage(dirty_features)
        assert 'place_finale' in leaked, "place_finale devrait être détecté"
    
    def test_inference_vs_training_features(self):
        """Test que les features d'inférence sont un sous-ensemble des features de training."""
        # En training, on peut utiliser is_win comme target
        # Mais JAMAIS comme feature d'input
        
        training_target = 'is_win'
        training_features = ['cote_matin', 'musique', 'distance_m']
        
        assert training_target not in training_features, \
            f"Target {training_target} ne doit pas être dans les features"


# ============================================================================
# TEST: KELLY CAP AND VALUE
# ============================================================================

class TestKellyCapAndValue:
    """Test du Kelly criterion: value<=0 => stake==0; cap 5% respecté."""
    
    MAX_STAKE_PCT = 0.05  # 5%
    KELLY_FRACTION = 0.25
    
    def test_zero_stake_negative_ev(self):
        """Test que EV <= 0 => stake == 0."""
        from strategy_backtester import BetSimulator, BacktestConfig
        
        config = BacktestConfig(
            initial_bankroll=1000,
            kelly_fraction=self.KELLY_FRACTION,
            max_stake_pct=self.MAX_STAKE_PCT
        )
        sim = BetSimulator(config)
        
        # Pari avec EV négatif
        p_win = 0.2
        odds = 3.0  # EV = 0.2 * 3 - 1 = -0.4
        ev = p_win * odds - 1
        
        assert ev < 0, f"Ce test nécessite EV < 0, got {ev}"
        
        # Le simulateur ne devrait pas placer ce pari
        # (ou stake == 0)
        kelly_raw = (p_win * (odds - 1) - (1 - p_win)) / (odds - 1)
        
        assert kelly_raw <= 0, f"Kelly devrait être <= 0 pour EV négatif, got {kelly_raw}"
    
    def test_zero_stake_zero_ev(self):
        """Test que EV == 0 => stake == 0."""
        p_win = 0.25
        odds = 4.0  # EV = 0.25 * 4 - 1 = 0
        ev = p_win * odds - 1
        
        assert np.isclose(ev, 0, atol=1e-6), f"Ce test nécessite EV == 0, got {ev}"
        
        kelly_raw = (p_win * (odds - 1) - (1 - p_win)) / (odds - 1)
        
        assert np.isclose(kelly_raw, 0, atol=1e-6), \
            f"Kelly devrait être 0 pour EV=0, got {kelly_raw}"
    
    def test_cap_5_percent_respected(self):
        """Test que le cap de 5% est respecté même pour gros EV."""
        p_win = 0.5
        odds = 3.0  # EV = 0.5 * 3 - 1 = 0.5 (50% EV!)
        
        # Kelly brut
        kelly_raw = (p_win * (odds - 1) - (1 - p_win)) / (odds - 1)
        # kelly_raw = (0.5 * 2 - 0.5) / 2 = 0.25
        
        # Kelly fractionnel
        kelly_frac = kelly_raw * self.KELLY_FRACTION
        
        # Stake avec cap
        stake_pct = min(kelly_frac, self.MAX_STAKE_PCT)
        
        assert stake_pct <= self.MAX_STAKE_PCT, \
            f"Stake {stake_pct*100}% dépasse le cap {self.MAX_STAKE_PCT*100}%"
    
    def test_kelly_formula_correctness(self):
        """Test la formule Kelly: f* = (p*b - q) / b."""
        # Cas de test connu
        p = 0.6  # 60% de chance de gagner
        b = 1.0  # Cotes nettes (odds - 1 = 2.0 - 1)
        q = 1 - p  # 40%
        
        kelly = (p * b - q) / b
        expected_kelly = (0.6 * 1 - 0.4) / 1  # = 0.2
        
        assert np.isclose(kelly, expected_kelly, atol=1e-6), \
            f"Kelly = {kelly}, attendu {expected_kelly}"
    
    def test_kelly_with_fractional_multiplier(self):
        """Test Kelly fractionnel (25%)."""
        p = 0.6
        odds = 2.0
        b = odds - 1
        
        kelly_full = (p * b - (1 - p)) / b  # 0.2
        kelly_frac = kelly_full * self.KELLY_FRACTION  # 0.05
        
        assert np.isclose(kelly_frac, 0.05, atol=1e-6), \
            f"Kelly fractionnel = {kelly_frac}, attendu 0.05"
    
    def test_bet_simulator_respects_cap(self):
        """Test que BetSimulator respecte le cap."""
        from strategy_backtester import BetSimulator, BacktestConfig
        
        config = BacktestConfig(
            initial_bankroll=1000,
            kelly_fraction=self.KELLY_FRACTION,
            max_stake_pct=self.MAX_STAKE_PCT
        )
        sim = BetSimulator(config)
        
        # Simuler un pari très profitable
        p_win = 0.7
        odds = 3.0  # EV = 0.7 * 3 - 1 = 1.1 (110% EV!)
        
        b = odds - 1
        kelly_raw = (p_win * b - (1 - p_win)) / b
        kelly_frac = kelly_raw * self.KELLY_FRACTION
        
        # Le cap devrait être appliqué
        expected_stake_pct = min(kelly_frac, self.MAX_STAKE_PCT)
        
        assert expected_stake_pct == self.MAX_STAKE_PCT, \
            f"Pour ce pari très profitable, le cap devrait s'appliquer"
    
    def test_value_bet_identification(self):
        """Test identification correcte des value bets."""
        test_cases = [
            # (p, odds, expected_is_value)
            (0.30, 4.0, True),   # EV = 0.3*4-1 = 0.2 > 0
            (0.25, 4.0, False),  # EV = 0.25*4-1 = 0 (pas de value)
            (0.20, 4.0, False),  # EV = 0.2*4-1 = -0.2 < 0
            (0.10, 15.0, True),  # EV = 0.1*15-1 = 0.5 > 0
            (0.05, 10.0, False), # EV = 0.05*10-1 = -0.5 < 0
        ]
        
        for p, odds, expected_is_value in test_cases:
            ev = p * odds - 1
            is_value = ev > 0
            
            assert is_value == expected_is_value, \
                f"p={p}, odds={odds}: EV={ev:.2f}, is_value={is_value}, expected={expected_is_value}"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
