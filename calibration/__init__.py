"""
Module Calibration - Gestion des artefacts de calibration
=========================================================

Ce module fournit les outils pour charger et gérer les paramètres
de calibration (T*, α, calibrateur) depuis les artefacts.

Usage:
    from calibration.artifacts_loader import load_calibration_state, CalibrationState
    
    state = load_calibration_state(prefer_artifacts=True)
    print(f"T={state.temperature}, α={state.alpha}, source={state.source}")
"""

from calibration.artifacts_loader import (
    CalibrationState,
    load_calibration_state,
    check_yaml_artifacts_mismatch,
    sync_yaml_from_artifacts,
    log_calibration_init,
    warn_if_mismatch,
)

__all__ = [
    'CalibrationState',
    'load_calibration_state',
    'check_yaml_artifacts_mismatch',
    'sync_yaml_from_artifacts',
    'log_calibration_init',
    'warn_if_mismatch',
]
