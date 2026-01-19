#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artifacts Loader - Source de Vérité pour la Calibration
========================================================

Ce module charge les paramètres de calibration depuis les artefacts (health.json)
comme source de vérité, avec fallback sur le YAML config/pro_betting.yaml.

Usage:
    from calibration.artifacts_loader import load_calibration_state, CalibrationState

    state = load_calibration_state(prefer_artifacts=True)
    print(f"T={state.temperature}, α={state.alpha}, source={state.source}")

La hiérarchie de priorité (prefer_artifacts=True):
    1. calibration/health.json (artefacts)
    2. config/pro_betting.yaml (fallback)
"""

from __future__ import annotations
import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Union

import yaml

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
HEALTH_JSON_PATH = CALIBRATION_DIR / "health.json"
YAML_CONFIG_PATH = PROJECT_ROOT / "config" / "pro_betting.yaml"


# ============================================================================
# DATACLASSES
# ============================================================================


@dataclass
class CalibrationState:
    """
    État de calibration chargé depuis les artefacts ou la config.

    Attributes:
        temperature: T* optimisé (softmax temperature)
        alpha: α global (blend modèle/marché, 0=marché, 1=modèle)
        alpha_by_disc: α par discipline {'plat': 0.0, 'trot': 0.4, ...}
        calibrator: Méthode de calibration ('platt', 'isotonic', 'temperature')
        last_calibration: Timestamp ISO de la dernière calibration
        source: 'artifacts' ou 'yaml'
        artifacts_path: Chemin vers health.json si chargé
        pickles_loaded: Liste des pickles chargés
        metrics: Métriques de calibration (ECE, Brier, etc.)
    """

    temperature: float = 1.254
    alpha: float = 0.2
    alpha_by_disc: Dict[str, float] = field(
        default_factory=lambda: {"plat": 0.0, "trot": 0.4, "obstacle": 0.4, "global": 0.2}
    )
    calibrator: str = "platt"
    last_calibration: Optional[str] = None
    source: str = "yaml"
    artifacts_path: Optional[str] = None
    pickles_loaded: list = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def get_effective_params(self) -> Dict[str, Any]:
        """Retourne les paramètres effectifs pour l'exécution."""
        return {
            "temperature": self.temperature,
            "alpha": self.alpha,
            "alpha_by_disc": self.alpha_by_disc,
            "calibrator": self.calibrator,
            "source": self.source,
            "last_calibration": self.last_calibration,
        }

    def get_alpha_for_discipline(self, discipline: str) -> float:
        """
        Retourne l'α optimal pour une discipline.

        Args:
            discipline: 'plat', 'trot', 'obstacle', 'attele', 'haie', etc.
        """
        disc_lower = discipline.lower().strip()

        # Mapping d'alias
        alias_map = {
            "attele": "trot",
            "monte": "trot",
            "haie": "obstacle",
            "steeple": "obstacle",
            "steeplechase": "obstacle",
            "cross": "obstacle",
        }
        disc = alias_map.get(disc_lower, disc_lower)

        return self.alpha_by_disc.get(disc, self.alpha)

    def summary(self) -> str:
        """Retourne un résumé textuel de l'état."""
        return (
            f"CalibrationState(T={self.temperature:.4f}, α={self.alpha:.2f}, "
            f"calibrator={self.calibrator}, source={self.source})"
        )


# ============================================================================
# LOADERS
# ============================================================================


def _load_health_json() -> Optional[Dict[str, Any]]:
    """
    Charge calibration/health.json s'il existe.

    Returns:
        Dict des données health ou None si non trouvé
    """
    if not HEALTH_JSON_PATH.exists():
        return None

    try:
        with open(HEALTH_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"health.json chargé: {HEALTH_JSON_PATH}")
        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Erreur lecture health.json: {e}")
        return None


def _load_yaml_config() -> Dict[str, Any]:
    """
    Charge config/pro_betting.yaml.

    Returns:
        Dict de configuration
    """
    if not YAML_CONFIG_PATH.exists():
        logger.warning(f"YAML config non trouvé: {YAML_CONFIG_PATH}")
        return {}

    try:
        with open(YAML_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.debug(f"YAML config chargé: {YAML_CONFIG_PATH}")
        return data or {}
    except (yaml.YAMLError, IOError) as e:
        logger.warning(f"Erreur lecture YAML config: {e}")
        return {}


def _load_pickles() -> Dict[str, Any]:
    """
    Charge les fichiers pickle de calibration s'ils existent.

    Returns:
        Dict avec les objets chargés
    """
    pickles = {}

    # Patterns de fichiers pickle à chercher
    patterns = [
        "scaler_temperature_*.pkl",
        "calibrator_platt_*.pkl",
        "calibrator_isotonic_*.pkl",
        "blender_*.pkl",
    ]

    for pattern in patterns:
        files = sorted(CALIBRATION_DIR.glob(pattern), reverse=True)
        if files:
            latest = files[0]
            try:
                with open(latest, "rb") as f:
                    obj = pickle.load(f)
                key = pattern.split("_")[0]  # 'scaler', 'calibrator', 'blender'
                if "platt" in pattern:
                    key = "calibrator_platt"
                elif "isotonic" in pattern:
                    key = "calibrator_isotonic"
                pickles[key] = {
                    "object": obj,
                    "path": str(latest),
                }
                logger.debug(f"Pickle chargé: {latest}")
            except Exception as e:
                logger.warning(f"Erreur chargement pickle {latest}: {e}")

    return pickles


def _extract_alpha_by_disc_from_yaml(yaml_data: Dict) -> Dict[str, float]:
    """Extrait alpha_by_disc depuis le YAML."""
    cal = yaml_data.get("calibration", {})
    return {
        "plat": cal.get("blend_alpha_plat", 0.0),
        "trot": cal.get("blend_alpha_trot", 0.4),
        "obstacle": cal.get("blend_alpha_obstacle", 0.4),
        "global": cal.get("blend_alpha_global", 0.2),
    }


# ============================================================================
# MAIN LOADER
# ============================================================================


def load_calibration_state(
    prefer_artifacts: bool = True,
    yaml_path: Union[str, Path, None] = None,
    health_path: Union[str, Path, None] = None,
    load_pickles: bool = False,
) -> CalibrationState:
    """
    Charge l'état de calibration avec priorité aux artefacts.

    Args:
        prefer_artifacts: Si True, health.json écrase les valeurs YAML
        yaml_path: Chemin vers pro_betting.yaml (optionnel)
        health_path: Chemin vers health.json (optionnel)
        load_pickles: Si True, charge aussi les pickles (scaler, calibrators)

    Returns:
        CalibrationState avec les paramètres effectifs

    Example:
        state = load_calibration_state(prefer_artifacts=True)
        print(f"Calibration: T={state.temperature}, α={state.alpha}, source={state.source}")
    """
    # Chemins
    yaml_file = Path(yaml_path) if yaml_path else YAML_CONFIG_PATH
    health_file = Path(health_path) if health_path else HEALTH_JSON_PATH

    # 1. Charger le YAML comme base
    yaml_data = {}
    if yaml_file.exists():
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Erreur lecture YAML: {e}")

    cal_yaml = yaml_data.get("calibration", {})

    # Valeurs par défaut depuis YAML
    temperature = cal_yaml.get("temperature", 1.254)
    alpha = cal_yaml.get("blend_alpha_global", 0.2)
    alpha_by_disc = _extract_alpha_by_disc_from_yaml(yaml_data)
    calibrator = cal_yaml.get("calibrator", "platt")
    last_calibration = None
    source = "yaml"
    artifacts_path = None
    metrics = {}
    pickles_loaded = []

    # 2. Si prefer_artifacts, charger health.json et écraser
    if prefer_artifacts and health_file.exists():
        try:
            with open(health_file, "r", encoding="utf-8") as f:
                health_data = json.load(f)

            # Écraser avec les valeurs des artefacts
            if "temperature" in health_data:
                temperature = float(health_data["temperature"])

            if "alpha" in health_data:
                alpha = float(health_data["alpha"])
                # Mettre à jour alpha_by_disc.global
                alpha_by_disc["global"] = alpha

            # Alpha par discipline (si présent dans health.json)
            if "alpha_by_disc" in health_data:
                alpha_by_disc.update(health_data["alpha_by_disc"])

            if "calibrator_type" in health_data:
                calibrator = health_data["calibrator_type"]

            if "last_calibration" in health_data:
                last_calibration = health_data["last_calibration"]

            if "metrics" in health_data:
                metrics = health_data["metrics"]

            source = "artifacts"
            artifacts_path = str(health_file)

            logger.info(f"Artefacts chargés: T={temperature:.4f}, α={alpha:.2f}")

        except Exception as e:
            logger.warning(f"Erreur lecture health.json, fallback YAML: {e}")
            source = "yaml"

    # 3. Charger les pickles si demandé
    if load_pickles:
        pickles = _load_pickles()
        pickles_loaded = list(pickles.keys())

    return CalibrationState(
        temperature=temperature,
        alpha=alpha,
        alpha_by_disc=alpha_by_disc,
        calibrator=calibrator,
        last_calibration=last_calibration,
        source=source,
        artifacts_path=artifacts_path,
        pickles_loaded=pickles_loaded,
        metrics=metrics,
    )


def check_yaml_artifacts_mismatch(
    yaml_path: Union[str, Path, None] = None,
    health_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """
    Vérifie s'il y a des différences entre YAML et artefacts.

    Returns:
        Dict avec 'has_mismatch', 'mismatches' (liste des différences),
        'yaml_values', 'artifacts_values'
    """
    yaml_file = Path(yaml_path) if yaml_path else YAML_CONFIG_PATH
    health_file = Path(health_path) if health_path else HEALTH_JSON_PATH

    result = {
        "has_mismatch": False,
        "mismatches": [],
        "yaml_values": {},
        "artifacts_values": {},
        "artifacts_exist": health_file.exists(),
    }

    if not health_file.exists():
        return result

    # Charger YAML
    try:
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
    except:
        return result

    # Charger health.json
    try:
        with open(health_file, "r") as f:
            health_data = json.load(f)
    except:
        return result

    cal_yaml = yaml_data.get("calibration", {})

    # Valeurs YAML
    yaml_temp = cal_yaml.get("temperature", 1.254)
    yaml_alpha = cal_yaml.get("blend_alpha_global", 0.2)
    yaml_calibrator = cal_yaml.get("calibrator", "platt")

    # Valeurs artefacts
    art_temp = health_data.get("temperature", yaml_temp)
    art_alpha = health_data.get("alpha", yaml_alpha)
    art_calibrator = health_data.get("calibrator_type", yaml_calibrator)

    result["yaml_values"] = {
        "temperature": yaml_temp,
        "alpha": yaml_alpha,
        "calibrator": yaml_calibrator,
    }
    result["artifacts_values"] = {
        "temperature": art_temp,
        "alpha": art_alpha,
        "calibrator": art_calibrator,
    }

    # Comparer
    tolerance = 1e-4

    if abs(yaml_temp - art_temp) > tolerance:
        result["mismatches"].append(
            f"temperature: YAML={yaml_temp:.4f} vs artifacts={art_temp:.4f}"
        )

    if abs(yaml_alpha - art_alpha) > tolerance:
        result["mismatches"].append(f"alpha: YAML={yaml_alpha:.2f} vs artifacts={art_alpha:.2f}")

    if yaml_calibrator != art_calibrator:
        result["mismatches"].append(
            f"calibrator: YAML={yaml_calibrator} vs artifacts={art_calibrator}"
        )

    result["has_mismatch"] = len(result["mismatches"]) > 0

    return result


def sync_yaml_from_artifacts(
    yaml_path: Union[str, Path, None] = None,
    health_path: Union[str, Path, None] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Synchronise le YAML depuis les artefacts (health.json).

    Args:
        yaml_path: Chemin vers pro_betting.yaml
        health_path: Chemin vers health.json
        dry_run: Si True, ne modifie pas le fichier

    Returns:
        Dict avec 'success', 'changes', 'message'
    """
    import re

    yaml_file = Path(yaml_path) if yaml_path else YAML_CONFIG_PATH
    health_file = Path(health_path) if health_path else HEALTH_JSON_PATH

    result = {
        "success": False,
        "changes": [],
        "message": "",
    }

    if not health_file.exists():
        result["message"] = f"Artefacts non trouvés: {health_file}"
        return result

    if not yaml_file.exists():
        result["message"] = f"YAML config non trouvé: {yaml_file}"
        return result

    # Charger health.json
    try:
        with open(health_file, "r") as f:
            health_data = json.load(f)
    except Exception as e:
        result["message"] = f"Erreur lecture health.json: {e}"
        return result

    # Lire le YAML brut
    try:
        with open(yaml_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        result["message"] = f"Erreur lecture YAML: {e}"
        return result

    original_content = content

    # Mise à jour température
    if "temperature" in health_data:
        new_temp = health_data["temperature"]
        # Pattern: temperature: <valeur>
        pattern = r"(temperature:\s*)[\d.]+"
        if re.search(pattern, content):
            content = re.sub(pattern, f"\\g<1>{new_temp:.6f}", content)
            result["changes"].append(f"temperature: → {new_temp:.6f}")

    # Mise à jour alpha global
    if "alpha" in health_data:
        new_alpha = health_data["alpha"]
        pattern = r"(blend_alpha_global:\s*)[\d.]+"
        if re.search(pattern, content):
            content = re.sub(pattern, f"\\g<1>{new_alpha:.1f}", content)
            result["changes"].append(f"blend_alpha_global: → {new_alpha:.1f}")

    # Mise à jour calibrator
    if "calibrator_type" in health_data:
        new_cal = health_data["calibrator_type"]
        pattern = r"(calibrator:\s*)\w+"
        if re.search(pattern, content):
            content = re.sub(pattern, f"\\g<1>{new_cal}", content)
            result["changes"].append(f"calibrator: → {new_cal}")

    if content == original_content:
        result["success"] = True
        result["message"] = "Aucune modification nécessaire (déjà synchronisé)"
        return result

    # Écrire si pas dry_run
    if not dry_run:
        try:
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write(content)
            result["success"] = True
            result["message"] = f"YAML mis à jour: {len(result['changes'])} changements"
        except Exception as e:
            result["message"] = f"Erreur écriture YAML: {e}"
    else:
        result["success"] = True
        result["message"] = f"[DRY RUN] {len(result['changes'])} changements à appliquer"

    return result


# ============================================================================
# LOGGING HELPERS
# ============================================================================


def log_calibration_init(state: CalibrationState, logger_instance=None):
    """
    Log l'initialisation de la calibration avec un format standard.

    Format: "Calibration: T=5.0, α=0.0, source=artefacts"
    """
    log = logger_instance or logger
    msg = f"Calibration: T={state.temperature:.4f}, α={state.alpha:.2f}, source={state.source}"
    log.info(msg)
    return msg


def warn_if_mismatch(yaml_path=None, health_path=None, logger_instance=None):
    """
    Log un WARNING si YAML ≠ artefacts.

    Returns:
        True si mismatch détecté
    """
    log = logger_instance or logger

    mismatch = check_yaml_artifacts_mismatch(yaml_path, health_path)

    if mismatch["has_mismatch"]:
        log.warning(
            f"YAML ≠ artefacts détecté! Artefacts utilisés comme source de vérité. "
            f"Différences: {'; '.join(mismatch['mismatches'])}. "
            f"Utilisez 'cli.py calibrate --sync-config' pour synchroniser."
        )
        return True

    return False


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("=" * 60)
    print("🔧 ARTIFACTS LOADER - Test")
    print("=" * 60)

    # Charger l'état
    state = load_calibration_state(prefer_artifacts=True)

    print("\n📊 État de calibration:")
    print(f"   Temperature:      {state.temperature:.6f}")
    print(f"   Alpha (global):   {state.alpha:.2f}")
    print(f"   Alpha by disc:    {state.alpha_by_disc}")
    print(f"   Calibrator:       {state.calibrator}")
    print(f"   Last calibration: {state.last_calibration}")
    print(f"   Source:           {state.source}")
    print(f"   Artifacts path:   {state.artifacts_path}")

    # Vérifier les différences
    print("\n🔍 Vérification YAML vs Artefacts:")
    mismatch = check_yaml_artifacts_mismatch()

    if mismatch["has_mismatch"]:
        print("   ⚠️  Différences détectées:")
        for diff in mismatch["mismatches"]:
            print(f"      - {diff}")
    else:
        print("   ✅ YAML et artefacts synchronisés")

    print("\n" + "=" * 60)
