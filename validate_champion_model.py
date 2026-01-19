#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 CHAMPION MODEL VALIDATOR
============================

Valide que le modèle champion est correctement configuré et accessible
depuis tous les composants du système.
"""

import os
import sys
import json
from pathlib import Path
import pickle
import numpy as np
from datetime import datetime


def test_champion_model_files():
    """Teste l'existence et la validité des fichiers du modèle champion."""
    print("🏆 VALIDATION DU MODÈLE CHAMPION")
    print("=" * 50)

    champion_dir = Path("data/models/champion")
    calibration_dir = Path("calibration/champion")

    # Vérifier les fichiers du modèle
    required_files = {
        "Model XGBoost": champion_dir / "xgboost_model.pkl",
        "Feature Scaler": champion_dir / "feature_scaler.pkl",
        "Feature Imputer": champion_dir / "feature_imputer.pkl",
        "Metadata": champion_dir / "metadata.json",
    }

    calibration_files = {
        "Temperature Scaler": calibration_dir / "scaler_temperature.pkl",
        "Platt Calibrator": calibration_dir / "calibrator_platt.pkl",
        "Calibration Report": calibration_dir / "calibration_report.json",
        "Dynamic Blender": calibration_dir / "dynamic_blender",
    }

    print("\n📁 FICHIERS DU MODÈLE CHAMPION:")
    all_good = True

    for name, path in required_files.items():
        if path.exists():
            if path.suffix == ".json":
                try:
                    with open(path) as f:
                        data = json.load(f)
                    size = f"({len(data)} keys)"
                except:
                    size = "(invalid JSON)"
            else:
                size = f"({path.stat().st_size // 1024}KB)"
            print(f"   ✅ {name}: {path.name} {size}")
        else:
            print(f"   ❌ {name}: {path} (MANQUANT)")
            all_good = False

    print("\n🔬 FICHIERS DE CALIBRATION:")
    for name, path in calibration_files.items():
        if path.exists():
            if path.is_dir():
                files = list(path.glob("*"))
                size = f"({len(files)} files)"
            elif path.suffix == ".json":
                try:
                    with open(path) as f:
                        data = json.load(f)
                    size = f"({len(data)} keys)"
                except:
                    size = "(invalid JSON)"
            else:
                size = f"({path.stat().st_size // 1024}KB)"
            print(f"   ✅ {name}: {path.name} {size}")
        else:
            print(f"   ❌ {name}: {path} (MANQUANT)")
            all_good = False

    return all_good


def test_model_loading():
    """Teste le chargement effectif du modèle champion."""
    print("\n🧠 TEST DE CHARGEMENT DU MODÈLE:")

    try:
        model_path = Path("data/models/champion/xgboost_model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print(f"   ✅ Modèle chargé: {type(model).__name__}")

        if hasattr(model, "feature_names_in_"):
            print(f"   ✅ Features: {len(model.feature_names_in_)} colonnes")

        # Test prédiction dummy
        if hasattr(model, "predict_proba"):
            dummy_data = np.random.rand(
                1, len(model.feature_names_in_) if hasattr(model, "feature_names_in_") else 62
            )
            pred = model.predict_proba(dummy_data)
            print(f"   ✅ Test prédiction: shape {pred.shape}")

        return True

    except Exception as e:
        print(f"   ❌ Erreur chargement: {e}")
        return False


def test_calibration_artifacts():
    """Teste le chargement des artefacts de calibration."""
    print("\n🎯 TEST DES ARTEFACTS DE CALIBRATION:")

    try:
        # Test rapport de calibration
        report_path = Path("calibration/champion/calibration_report.json")
        with open(report_path) as f:
            report = json.load(f)

        print("   ✅ Rapport calibration chargé")

        # Extraire métriques importantes
        if "temperature" in report:
            print(f"   ✅ Temperature: {report['temperature']}")
        if "brier_score" in report:
            print(f"   ✅ Brier Score: {report['brier_score']:.4f}")
        if "ece" in report:
            print(f"   ✅ ECE: {report['ece']:.4f}")

        # Test temperature scaler
        temp_scaler_path = Path("calibration/champion/scaler_temperature.pkl")
        with open(temp_scaler_path, "rb") as f:
            temp_scaler = pickle.load(f)
        print(f"   ✅ Temperature Scaler: {type(temp_scaler).__name__}")

        # Test Platt calibrator
        platt_path = Path("calibration/champion/calibrator_platt.pkl")
        with open(platt_path, "rb") as f:
            platt = pickle.load(f)
        print(f"   ✅ Platt Calibrator: {type(platt).__name__}")

        return True

    except Exception as e:
        print(f"   ❌ Erreur calibration: {e}")
        return False


def test_api_configuration():
    """Teste que l'API est configurée pour utiliser le modèle champion."""
    print("\n🔌 TEST CONFIGURATION API:")

    try:
        # Simuler l'import de l'API pour voir le chemin par défaut
        current_model_path = os.getenv("MODEL_PATH", "data/models/champion/xgboost_model.pkl")
        expected_path = "data/models/champion/xgboost_model.pkl"

        if current_model_path == expected_path:
            print(f"   ✅ Chemin modèle API: {current_model_path}")
        else:
            print(f"   ⚠️ Chemin modèle API: {current_model_path} (attendu: {expected_path})")

        # Test config loader
        try:
            from config.loader import get_calibration_params_from_artifacts

            params = get_calibration_params_from_artifacts()
            print(f"   ✅ Config loader: source={params.get('source', 'unknown')}")
            print(f"      - Temperature: {params.get('temperature', 'N/A')}")
            print(f"      - Blend Alpha: {params.get('blend_alpha', 'N/A')}")
        except Exception as e:
            print(f"   ⚠️ Config loader: {e}")

        return True

    except Exception as e:
        print(f"   ❌ Erreur configuration: {e}")
        return False


def test_cli_picks_integration():
    """Teste que la CLI est bien configurée pour utiliser p_final."""
    print("\n🎯 TEST INTÉGRATION CLI:")

    try:
        # Vérifier que le fichier CLI importe bien les bons modules
        cli_path = Path("cli.py")
        with open(cli_path) as f:
            cli_content = f.read()

        if "race_pronostic_generator" in cli_content:
            print("   ✅ CLI importe race_pronostic_generator")
        else:
            print("   ⚠️ CLI n'importe pas race_pronostic_generator")

        if "get_calibration_params_from_artifacts" in cli_content:
            print("   ✅ CLI utilise les artefacts de calibration")
        else:
            print("   ⚠️ CLI n'utilise pas les artefacts de calibration")

        return True

    except Exception as e:
        print(f"   ❌ Erreur test CLI: {e}")
        return False


def main():
    """Fonction principale de validation."""
    print(f"Validation du modèle champion - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        test_champion_model_files,
        test_model_loading,
        test_calibration_artifacts,
        test_api_configuration,
        test_cli_picks_integration,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Erreur test: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DE LA VALIDATION:")

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"   🎉 TOUS LES TESTS PASSÉS ({passed}/{total})")
        print("   🏆 LE MODÈLE CHAMPION EST OPÉRATIONNEL!")
    else:
        print(f"   ⚠️ {total - passed} TESTS ÉCHOUÉS ({passed}/{total})")
        print("   🔧 Vérifiez les erreurs ci-dessus")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
