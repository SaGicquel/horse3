"""
Configuration Loader - Frozen Dataclasses avec Version Hash
===========================================================
Charge pro_betting.yaml → dataclasses immuables + hash de version.

Usage:
    from config.loader import load_config, get_config
    
    cfg = load_config()  # charge config/pro_betting.yaml
    print(cfg.calibration.temperature)  # 1.254
    print(cfg.version_hash)  # hash SHA256 de la config
"""

from __future__ import annotations
import hashlib
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any
from functools import lru_cache

__version__ = "2.0.0"

# ============================================================================
# Frozen Dataclasses
# ============================================================================

@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration de calibration des probabilités."""
    temperature: float = 1.254
    blend_alpha_global: float = 0.2
    blend_alpha_plat: float = 0.0
    blend_alpha_trot: float = 0.4
    blend_alpha_obstacle: float = 0.4
    calibrator: str = "platt"


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration Monte Carlo."""
    num_simulations: int = 20000


@dataclass(frozen=True)
class MarketsConfig:
    """Configuration des marchés de paris."""
    mode: str = "parimutuel"
    takeout_rate: float = 0.16


@dataclass(frozen=True)
class KellyConfig:
    """Configuration Kelly criterion."""
    fraction: float = 0.25
    value_cutoff: float = 0.05
    max_stake_pct: float = 0.05


@dataclass(frozen=True)
class PortfolioConfig:
    """Configuration du portefeuille."""
    correlation_penalty: float = 0.30
    max_same_race: int = 2
    max_same_jockey: int = 3
    max_same_trainer: int = 3
    confidence_level: float = 0.95


@dataclass(frozen=True)
class ExoticsConfig:
    """Configuration des paris exotiques."""
    p_place_simulations: int = 20000
    top_k_couples: int = 50
    top_k_trios: int = 100
    max_coverage: float = 0.30


@dataclass(frozen=True)
class PlaceEstimatorsConfig:
    """Configuration des estimateurs de place/ordre pour exotiques."""
    # Températures Plackett-Luce par discipline
    temperature_default: float = 1.0
    temperature_plat: float = 0.95
    temperature_trot: float = 1.05
    temperature_obstacle: float = 1.10
    
    # Paramètres Henery
    henery_gamma: float = 0.81
    
    # Paramètres Stern
    stern_lambda: float = 0.15
    
    # Paramètres Lo-Bacon-Shone
    lbs_iterations: int = 100
    lbs_damping: float = 0.7
    
    # Estimateur par discipline
    estimator_plat: str = "henery"
    estimator_trot: str = "lbs"
    estimator_obstacle: str = "stern"
    estimator_default: str = "harville"
    
    # Validation
    min_simulations_stable: int = 20000
    cv_threshold: float = 0.10
    
    def get_temperature(self, discipline: str) -> float:
        """Retourne la température pour une discipline."""
        disc = discipline.lower() if discipline else 'default'
        if disc == 'plat':
            return self.temperature_plat
        elif disc == 'trot':
            return self.temperature_trot
        elif disc == 'obstacle':
            return self.temperature_obstacle
        return self.temperature_default
    
    def get_estimator(self, discipline: str) -> str:
        """Retourne le nom de l'estimateur pour une discipline."""
        disc = discipline.lower() if discipline else 'default'
        if disc == 'plat':
            return self.estimator_plat
        elif disc == 'trot':
            return self.estimator_trot
        elif disc == 'obstacle':
            return self.estimator_obstacle
        return self.estimator_default


@dataclass(frozen=True)
class BettingDefaultsConfig:
    """Configuration de la politique de mise par défaut (Kelly fractionnaire)."""
    kelly_profile_default: str = "STANDARD"  # SUR, STANDARD, AMBITIEUX, PERSONNALISE
    kelly_fraction_sur: float = 0.25
    kelly_fraction_standard: float = 0.33
    kelly_fraction_ambitieux: float = 0.50
    custom_kelly_fraction: float = 0.33
    value_cutoff: float = 0.05          # Seuil value minimum >= 5%
    cap_per_bet: float = 0.02           # Cap 2% du bankroll par pari
    daily_budget_rate: float = 0.12     # Budget jour = 12% bankroll
    max_unit_bets_per_race: int = 2     # Max 2 paris unitaires par course
    rounding_increment_eur: float = 0.5 # Arrondir à 0.50€
    
    def get_kelly_fraction(self, profile: str = None) -> float:
        """Retourne la fraction Kelly selon le profil."""
        profile = profile or self.kelly_profile_default
        profile = profile.upper()
        if profile == "SUR":
            return self.kelly_fraction_sur
        elif profile == "AMBITIEUX":
            return self.kelly_fraction_ambitieux
        elif profile == "PERSONNALISE":
            return self.custom_kelly_fraction
        else:  # STANDARD par défaut
            return self.kelly_fraction_standard


@dataclass(frozen=True)
class ExoticsDefaultsConfig:
    """Configuration des paris exotiques (taux/limites)."""
    per_ticket_rate: float = 0.0075   # 0.75% du bankroll par ticket
    max_pack_rate: float = 0.04       # Max 4% du bankroll par pack


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration du backtesting."""
    initial_bankroll: float = 1000.0
    min_races: int = 50


@dataclass(frozen=True)
class ArtifactsConfig:
    """Chemins des artefacts."""
    model_path: str = "models/xgb_proba_v9.joblib"
    predictions_dir: str = "predictions"
    reports_dir: str = "reports"


@dataclass(frozen=True)
class ProBettingConfig:
    """Configuration principale - IMMUABLE."""
    calibration: CalibrationConfig
    simulation: SimulationConfig
    markets: MarketsConfig
    kelly: KellyConfig
    portfolio: PortfolioConfig
    exotics: ExoticsConfig
    betting_defaults: BettingDefaultsConfig
    exotics_defaults: ExoticsDefaultsConfig
    place_estimators: PlaceEstimatorsConfig
    backtest: BacktestConfig
    artifacts: ArtifactsConfig
    version_hash: str = ""
    
    def get_blend_alpha(self, discipline: str) -> float:
        """
        Retourne le blend_alpha pour une discipline donnée.
        
        Args:
            discipline: 'plat', 'trot', 'obstacle', 'attele', 'haie', etc.
            
        Returns:
            Alpha correspondant ou alpha_global par défaut
        """
        # Mapping d'alias
        disc_lower = discipline.lower().strip()
        alias_map = {
            'attele': 'trot',
            'monte': 'trot',
            'haie': 'obstacle',
            'steeple': 'obstacle',
            'cross': 'obstacle',
        }
        disc = alias_map.get(disc_lower, disc_lower)
        
        # Retourner l'alpha correspondant
        if disc == 'plat':
            return self.calibration.blend_alpha_plat
        elif disc == 'trot':
            return self.calibration.blend_alpha_trot
        elif disc == 'obstacle':
            return self.calibration.blend_alpha_obstacle
        else:
            return self.calibration.blend_alpha_global


# ============================================================================
# Hash de version
# ============================================================================

def _compute_hash(data: Dict[str, Any]) -> str:
    """Calcule un hash SHA256 de la configuration."""
    # Sérialisation déterministe
    content = yaml.dump(data, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]


# ============================================================================
# Loader principal
# ============================================================================

def load_config(path: Optional[str] = None) -> ProBettingConfig:
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        path: Chemin vers le fichier YAML (défaut: config/pro_betting.yaml)
        
    Returns:
        ProBettingConfig: Configuration immuable avec hash de version
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        yaml.YAMLError: Si le YAML est invalide
    """
    if path is None:
        # Cherche config/pro_betting.yaml depuis la racine du projet
        root = Path(__file__).parent.parent
        path = root / "config" / "pro_betting.yaml"
    else:
        path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration non trouvée: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Calcul du hash de version
    version_hash = _compute_hash(data)
    
    # Extraction des sections avec valeurs par défaut
    cal_data = data.get('calibration', {})
    sim_data = data.get('simulation', data.get('monte_carlo', {}))
    mkt_data = data.get('markets', {})
    kel_data = data.get('kelly', {})
    prt_data = data.get('portfolio', {})
    exo_data = data.get('exotics', {})
    bkt_data = data.get('backtest', {})
    art_data = data.get('artifacts', {})
    
    # Gestion de la rétrocompatibilité blend_alpha
    blend_data = data.get('blend', {})
    
    calibration = CalibrationConfig(
        temperature=cal_data.get('temperature', 1.254),
        blend_alpha_global=blend_data.get('alpha_global', cal_data.get('blend_alpha_global', 0.2)),
        blend_alpha_plat=blend_data.get('alpha_plat', cal_data.get('blend_alpha_plat', 0.0)),
        blend_alpha_trot=blend_data.get('alpha_trot', cal_data.get('blend_alpha_trot', 0.4)),
        blend_alpha_obstacle=blend_data.get('alpha_obstacle', cal_data.get('blend_alpha_obstacle', 0.4)),
        calibrator=cal_data.get('calibrator', 'platt'),
    )
    
    simulation = SimulationConfig(
        num_simulations=sim_data.get('num_simulations', 20000),
    )
    
    markets = MarketsConfig(
        mode=mkt_data.get('mode', 'parimutuel'),
        takeout_rate=mkt_data.get('takeout_rate', 0.16),
    )
    
    kelly = KellyConfig(
        fraction=kel_data.get('fraction', 0.25),
        value_cutoff=kel_data.get('value_cutoff', 0.05),
        max_stake_pct=kel_data.get('max_stake_pct', 0.05),
    )
    
    portfolio = PortfolioConfig(
        correlation_penalty=prt_data.get('correlation_penalty', 0.30),
        max_same_race=prt_data.get('max_same_race', 2),
        max_same_jockey=prt_data.get('max_same_jockey', 3),
        max_same_trainer=prt_data.get('max_same_trainer', 3),
        confidence_level=prt_data.get('confidence_level', 0.95),
    )
    
    exotics = ExoticsConfig(
        p_place_simulations=exo_data.get('p_place_simulations', sim_data.get('num_simulations', 20000)),
        top_k_couples=exo_data.get('top_k_couples', 50),
        top_k_trios=exo_data.get('top_k_trios', 100),
        max_coverage=exo_data.get('max_coverage', 0.30),
    )
    
    # Nouvelles sections betting_defaults et exotics_defaults
    bet_def_data = data.get('betting_defaults', {})
    kelly_map = bet_def_data.get('kelly_fraction_map', {})
    
    betting_defaults = BettingDefaultsConfig(
        kelly_profile_default=bet_def_data.get('kelly_profile_default', 'STANDARD'),
        kelly_fraction_sur=kelly_map.get('SUR', 0.25),
        kelly_fraction_standard=kelly_map.get('STANDARD', 0.33),
        kelly_fraction_ambitieux=kelly_map.get('AMBITIEUX', 0.50),
        custom_kelly_fraction=bet_def_data.get('custom_kelly_fraction', 0.33),
        value_cutoff=bet_def_data.get('value_cutoff', kel_data.get('value_cutoff', 0.05)),
        cap_per_bet=bet_def_data.get('cap_per_bet', 0.02),
        daily_budget_rate=bet_def_data.get('daily_budget_rate', 0.12),
        max_unit_bets_per_race=bet_def_data.get('max_unit_bets_per_race', 2),
        rounding_increment_eur=bet_def_data.get('rounding_increment_eur', 0.5),
    )
    
    exo_def_data = data.get('exotics_defaults', {})
    exotics_defaults = ExoticsDefaultsConfig(
        per_ticket_rate=exo_def_data.get('per_ticket_rate', 0.0075),
        max_pack_rate=exo_def_data.get('max_pack_rate', 0.04),
    )
    
    backtest = BacktestConfig(
        initial_bankroll=bkt_data.get('initial_bankroll', 1000.0),
        min_races=bkt_data.get('min_races', 50),
    )
    
    artifacts = ArtifactsConfig(
        model_path=art_data.get('model_path', 'models/xgb_proba_v9.joblib'),
        predictions_dir=art_data.get('predictions_dir', 'predictions'),
        reports_dir=art_data.get('reports_dir', 'reports'),
    )
    
    # Nouvelle section place_estimators
    pe_data = data.get('place_estimators', {})
    place_estimators = PlaceEstimatorsConfig(
        temperature_default=pe_data.get('temperature_default', 1.0),
        temperature_plat=pe_data.get('temperature_plat', 0.95),
        temperature_trot=pe_data.get('temperature_trot', 1.05),
        temperature_obstacle=pe_data.get('temperature_obstacle', 1.10),
        henery_gamma=pe_data.get('henery_gamma', 0.81),
        stern_lambda=pe_data.get('stern_lambda', 0.15),
        lbs_iterations=pe_data.get('lbs_iterations', 100),
        lbs_damping=pe_data.get('lbs_damping', 0.7),
        estimator_plat=pe_data.get('estimator_plat', 'henery'),
        estimator_trot=pe_data.get('estimator_trot', 'lbs'),
        estimator_obstacle=pe_data.get('estimator_obstacle', 'stern'),
        estimator_default=pe_data.get('estimator_default', 'harville'),
        min_simulations_stable=pe_data.get('min_simulations_stable', 20000),
        cv_threshold=pe_data.get('cv_threshold', 0.10),
    )
    
    # Création de l'objet config avec hash via object.__setattr__ pour frozen
    config = object.__new__(ProBettingConfig)
    object.__setattr__(config, 'calibration', calibration)
    object.__setattr__(config, 'simulation', simulation)
    object.__setattr__(config, 'markets', markets)
    object.__setattr__(config, 'kelly', kelly)
    object.__setattr__(config, 'portfolio', portfolio)
    object.__setattr__(config, 'exotics', exotics)
    object.__setattr__(config, 'betting_defaults', betting_defaults)
    object.__setattr__(config, 'exotics_defaults', exotics_defaults)
    object.__setattr__(config, 'place_estimators', place_estimators)
    object.__setattr__(config, 'backtest', backtest)
    object.__setattr__(config, 'artifacts', artifacts)
    object.__setattr__(config, 'version_hash', version_hash)
    
    return config


@lru_cache(maxsize=1)
def get_config() -> ProBettingConfig:
    """
    Récupère la configuration (singleton caché).
    
    Returns:
        ProBettingConfig: Configuration immuable
    """
    return load_config()


def clear_config_cache():
    """Vide le cache de configuration (utile pour les tests)."""
    get_config.cache_clear()


# ============================================================================
# Décorateur de cohérence
# ============================================================================

def coherent_params(func):
    """
    Décorateur qui injecte la configuration comme premier argument.
    
    Usage:
        @coherent_params
        def my_function(cfg: ProBettingConfig, other_arg):
            print(cfg.calibration.temperature)
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        cfg = get_config()
        return func(cfg, *args, **kwargs)
    
    return wrapper


# ============================================================================
# Utilitaires
# ============================================================================

def config_summary(cfg: Optional[ProBettingConfig] = None) -> str:
    """Génère un résumé textuel de la configuration."""
    if cfg is None:
        cfg = get_config()
    
    return f"""
ProBettingConfig v{__version__} [hash: {cfg.version_hash}]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calibration:
  - temperature: {cfg.calibration.temperature}
  - blend_alpha_global: {cfg.calibration.blend_alpha_global}
  - calibrator: {cfg.calibration.calibrator}

Simulation:
  - num_simulations: {cfg.simulation.num_simulations}

Markets:
  - mode: {cfg.markets.mode}
  - takeout_rate: {cfg.markets.takeout_rate}

Kelly:
  - fraction: {cfg.kelly.fraction}
  - value_cutoff: {cfg.kelly.value_cutoff}
  - max_stake_pct: {cfg.kelly.max_stake_pct}

Portfolio:
  - correlation_penalty: {cfg.portfolio.correlation_penalty}
  - max_same_race: {cfg.portfolio.max_same_race}
  - confidence_level: {cfg.portfolio.confidence_level}

Exotics:
  - p_place_simulations: {cfg.exotics.p_place_simulations}
  - top_k_couples: {cfg.exotics.top_k_couples}
  - max_coverage: {cfg.exotics.max_coverage}
"""


# ============================================================================
# Rétrocompatibilité avec l'ancien loader
# ============================================================================

# Chemins globaux
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "pro_betting.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "calibration" / "champion"


def reload_config() -> ProBettingConfig:
    """Recharge la configuration (vide le cache)."""
    clear_config_cache()
    return get_config()


def get_latest_calibration_report() -> Optional[Dict[str, Any]]:
    """
    Charge le rapport de calibration depuis calibration/champion/.
    
    Returns:
        Dict ou None si non trouvé
    """
    import glob
    import json as json_mod
    
    # Essayer d'abord le fichier standard du champion
    champion_report = ARTIFACTS_DIR / "calibration_report.json"
    if champion_report.exists():
        try:
            with open(champion_report, 'r') as f:
                return json_mod.load(f)
        except Exception:
            pass
    
    # Fallback sur l'ancien pattern
    pattern = str(ARTIFACTS_DIR / "calibration_report_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    
    if not files:
        return None
    
    try:
        with open(files[0], 'r') as f:
            return json_mod.load(f)
    except Exception:
        return None


def get_calibration_params_from_artifacts() -> Dict[str, Any]:
    """
    Extrait les paramètres de calibration depuis les artefacts ou config.
    
    Returns:
        Dict avec 'temperature', 'blend_alpha', 'source'
    """
    cfg = get_config()
    
    # Essayer de charger depuis artifacts
    report = get_latest_calibration_report()
    
    if report:
        return {
            'temperature': report.get('optimal_temperature', cfg.calibration.temperature),
            'blend_alpha': report.get('optimal_alpha', cfg.calibration.blend_alpha_global),
            'calibrator': report.get('calibrator', cfg.calibration.calibrator),
            'source': 'artifacts'
        }
    
    # Fallback sur config
    return {
        'temperature': cfg.calibration.temperature,
        'blend_alpha': cfg.calibration.blend_alpha_global,
        'calibrator': cfg.calibration.calibrator,
        'source': 'config'
    }


def update_config_from_calibration(
    temperature: float,
    blend_alpha: float,
    calibrator: str = "platt"
) -> bool:
    """
    Met à jour pro_betting.yaml avec les nouveaux paramètres de calibration.
    
    Args:
        temperature: Nouvelle température optimale
        blend_alpha: Nouveau blend alpha global
        calibrator: Méthode de calibration
        
    Returns:
        True si succès
    """
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remplacer les valeurs
        import re
        content = re.sub(r'(temperature:\s*)[\d.]+', f'\\g<1>{temperature}', content)
        content = re.sub(r'(blend_alpha_global:\s*)[\d.]+', f'\\g<1>{blend_alpha}', content)
        content = re.sub(r'(calibrator:\s*)\w+', f'\\g<1>{calibrator}', content)
        
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Invalider le cache
        clear_config_cache()
        
        return True
    except Exception as e:
        print(f"Erreur mise à jour config: {e}")
        return False


# Alias de rétrocompatibilité
NormalizationConfig = SimulationConfig  # Alias


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    try:
        cfg = load_config()
        print(config_summary(cfg))
        print(f"\n✅ Configuration chargée avec succès")
        print(f"   Hash de version: {cfg.version_hash}")
        
        # Test immutabilité
        try:
            cfg.calibration.temperature = 2.0  # type: ignore
            print("❌ ERREUR: Configuration mutable!")
            sys.exit(1)
        except AttributeError:
            print("   ✓ Configuration immuable (frozen)")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)
