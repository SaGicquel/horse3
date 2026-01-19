"""Config module - Centralized configuration for Pro Betting system v2.0."""

from .loader import (
    # Core
    get_config,
    load_config,
    clear_config_cache,
    reload_config,
    coherent_params,
    config_summary,
    # Artifacts
    get_latest_calibration_report,
    get_calibration_params_from_artifacts,
    update_config_from_calibration,
    # Config classes
    ProBettingConfig,
    CalibrationConfig,
    SimulationConfig,
    MarketsConfig,
    KellyConfig,
    PortfolioConfig,
    ExoticsConfig,
    BacktestConfig,
    ArtifactsConfig,
    NormalizationConfig,  # Alias pour SimulationConfig
    # Paths
    PROJECT_ROOT,
    CONFIG_PATH,
    ARTIFACTS_DIR,
    # Version
    __version__,
)

__all__ = [
    # Core
    "get_config",
    "load_config",
    "clear_config_cache",
    "reload_config",
    "coherent_params",
    "config_summary",
    # Artifacts
    "get_latest_calibration_report",
    "get_calibration_params_from_artifacts",
    "update_config_from_calibration",
    # Config classes
    "ProBettingConfig",
    "CalibrationConfig",
    "SimulationConfig",
    "MarketsConfig",
    "KellyConfig",
    "PortfolioConfig",
    "ExoticsConfig",
    "BacktestConfig",
    "ArtifactsConfig",
    "NormalizationConfig",
    # Paths
    "PROJECT_ROOT",
    "CONFIG_PATH",
    "ARTIFACTS_DIR",
    # Version
    "__version__",
]
