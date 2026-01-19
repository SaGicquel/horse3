#!/usr/bin/env python3
"""
SIMPLE CALIBRATION D6 - BASÉE SUR LE RAPPORT
============================================

Utilise les paramètres du rapport de calibration pour appliquer
la calibration sans dépendre des objets pickle.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_calibration_params():
    """Charge les paramètres de calibration depuis le rapport"""

    logger.info("📊 CHARGEMENT PARAMÈTRES DE CALIBRATION")

    # Charger le rapport le plus récent
    report_path = "calibration/calibration_report_20251208_163949.json"

    with open(report_path, "r") as f:
        report = json.load(f)

    logger.info(f"✅ Rapport chargé: {report_path}")

    # Extraire les paramètres
    params = {}

    # Température
    if "stats" in report["metrics"] and "temperature" in report["metrics"]["stats"]:
        params["temperature"] = report["metrics"]["stats"]["temperature"]
        logger.info(f"🌡️  Température: {params['temperature']:.4f}")
    else:
        params["temperature"] = 1.0
        logger.warning("⚠️  Température non trouvée, utilisation de 1.0")

    # Gamma de correction
    if "debias" in report["metrics"] and "gamma" in report["metrics"]["debias"]:
        params["gamma"] = report["metrics"]["debias"]["gamma"]["global"]
        logger.info(f"🔧 Gamma: {params['gamma']:.4f}")
    else:
        params["gamma"] = 1.0
        logger.warning("⚠️  Gamma non trouvé, utilisation de 1.0")

    # Alpha de blend
    if "debias" in report["metrics"] and "alpha_stats" in report["metrics"]["debias"]:
        params["alpha"] = report["metrics"]["debias"]["alpha_stats"]["mean"]
        logger.info(f"⚖️  Alpha: {params['alpha']:.4f}")
    else:
        params["alpha"] = 1.0  # Modèle uniquement
        logger.warning("⚠️  Alpha non trouvé, utilisation modèle uniquement")

    # Type de calibrateur
    if "stats" in report["metrics"] and "calibrator_type" in report["metrics"]["stats"]:
        params["calibrator_type"] = report["metrics"]["stats"]["calibrator_type"]
        logger.info(f"📊 Calibrateur: {params['calibrator_type']}")
    else:
        params["calibrator_type"] = "isotonic"
        logger.warning("⚠️  Calibrateur non spécifié, utilisation isotonic")

    return params


def simple_isotonic_calibration(probabilities, true_labels, test_probabilities):
    """Calibration isotonique simple réentraînée"""

    logger.info("🎯 Calibration isotonique simple...")

    # Créer et entraîner le calibrateur
    calibrator = IsotonicRegression(out_of_bounds="clip")

    # Entraînement sur un échantillon
    sample_size = min(50000, len(probabilities))  # Limiter pour la mémoire
    indices = np.random.choice(len(probabilities), sample_size, replace=False)

    calibrator.fit(probabilities[indices], true_labels[indices])

    # Application sur toutes les données de test
    calibrated = calibrator.predict(test_probabilities)

    logger.info("✅ Calibration isotonique appliquée")

    return calibrated


def apply_simple_calibration(df, params):
    """Applique une calibration simple basée sur les paramètres"""

    logger.info("🔄 APPLICATION CALIBRATION SIMPLE")

    result_df = df.copy()

    # 1. NORMALISATION SOFTMAX PAR COURSE
    logger.info("📐 Étape 1: Normalisation softmax par course")

    temperature = params["temperature"]

    def normalize_race_softmax(group):
        logits = group["logits_model"].values
        scaled_logits = logits / temperature
        probas = softmax(scaled_logits)
        return probas

    normalized_probs = []
    for race_id, group in df.groupby("race_id"):
        race_probs = normalize_race_softmax(group)
        normalized_probs.extend(race_probs)

    result_df["p_model_norm"] = normalized_probs

    logger.info("✅ Normalisation softmax terminée")

    # 2. CALIBRATION APPROXIMATIVE
    logger.info("🎯 Étape 2: Calibration approximative")

    # Pour la calibration, on va utiliser une approche simple
    # basée sur les données d'entraînement disponibles

    # Séparer train/val pour la calibration
    train_data = result_df[result_df["split"] == "train"].copy()

    if len(train_data) > 10000:  # Assez de données pour calibrer
        # Calibration isotonique simple
        calibrated_probs = simple_isotonic_calibration(
            train_data["p_model_norm"].values,
            train_data["label_win"].values,
            result_df["p_model_norm"].values,
        )

        result_df["p_calibrated"] = calibrated_probs
        logger.info("✅ Calibration isotonique appliquée")
    else:
        # Pas assez de données, garder les probabilités normalisées
        result_df["p_calibrated"] = result_df["p_model_norm"]
        logger.warning("⚠️  Pas assez de données train, calibration ignorée")

    # 3. CORRECTION GAMMA DU MARCHÉ
    logger.info("🔀 Étape 3: Correction gamma et blend")

    gamma = params["gamma"]
    alpha = params["alpha"]

    # Probabilités marché brutes
    p_market_raw = 1.0 / result_df["odds_market_preoff"]

    # Normalisation par course
    p_market_norm = []
    for race_id, group in result_df.groupby("race_id"):
        race_market_probs = 1.0 / group["odds_market_preoff"].values
        race_market_probs = race_market_probs / race_market_probs.sum()
        p_market_norm.extend(race_market_probs)

    result_df["p_market_norm"] = p_market_norm

    # Correction gamma
    p_market_corrected_raw = np.power(result_df["p_market_norm"], gamma)

    # Re-normalisation par course
    p_market_corrected = []
    for race_id, group in result_df.groupby("race_id"):
        group_indices = group.index
        group_corrected = p_market_corrected_raw[group_indices]
        group_normalized = group_corrected / group_corrected.sum()
        p_market_corrected.extend(group_normalized)

    result_df["p_market_corrected"] = p_market_corrected

    # Blend modèle/marché
    result_df["p_blend"] = (
        alpha * result_df["p_calibrated"] + (1 - alpha) * result_df["p_market_corrected"]
    )

    # 4. RENORMALISATION FINALE
    logger.info("🔄 Étape 4: Renormalisation finale")

    p_final = []
    for race_id, group in result_df.groupby("race_id"):
        group_indices = group.index
        group_probs = result_df.loc[group_indices, "p_blend"].values
        group_normalized = group_probs / group_probs.sum()
        p_final.extend(group_normalized)

    result_df["p_final"] = p_final

    logger.info("✅ Calibration complète terminée")

    return result_df


def main():
    """Pipeline principal"""

    logger.info("🚀 SIMPLE CALIBRATION D6")
    logger.info("=" * 70)

    try:
        # 1. Chargement des paramètres
        params = load_calibration_params()

        # 2. Chargement des données
        logger.info("📂 Chargement des données adaptées...")
        df = pd.read_csv("data/backtest_predictions_adapted.csv")
        logger.info(f"✅ Données chargées: {len(df):,} lignes")

        # 3. Application de la calibration
        result_df = apply_simple_calibration(df, params)

        # 4. Préparation du fichier final
        logger.info("📋 Préparation du fichier final...")

        output_df = pd.DataFrame()
        output_df["race_key"] = result_df["race_id"]
        output_df["id_cheval"] = result_df["id_cheval"]
        output_df["date_course"] = result_df["date_course"]
        output_df["p_model_win"] = result_df["p_model_win"]
        output_df["p_model_norm"] = result_df["p_model_norm"]
        output_df["p_calibrated"] = result_df["p_calibrated"]
        output_df["p_final"] = result_df["p_final"]
        output_df["is_win"] = result_df["label_win"]
        output_df["place"] = result_df["place"]
        output_df["position_arrivee"] = result_df["position_arrivee"]
        output_df["cote_sp"] = result_df["cote_sp"]
        output_df["split"] = result_df["split"]

        # 5. Sauvegarde
        output_path = "data/backtest_predictions_calibrated.csv"
        logger.info("💾 Sauvegarde du fichier final...")

        output_df.to_csv(output_path, index=False)

        # 6. Statistiques finales
        logger.info("\n📊 STATISTIQUES FINALES:")
        logger.info("=" * 50)
        logger.info(f"📂 Fichier: {output_path}")
        logger.info(f"📊 Lignes: {len(output_df):,}")

        # Vérification normalisation
        race_sums = output_df.groupby("race_key")["p_final"].sum()
        perfect_sums = (np.abs(race_sums - 1.0) < 1e-3).sum()
        logger.info(
            f"🔍 Courses normalisées: {perfect_sums:,}/{len(race_sums):,} ({perfect_sums/len(race_sums)*100:.1f}%)"
        )

        # Distribution des probabilités
        for col in ["p_model_win", "p_final"]:
            stats = output_df[col].describe()
            logger.info(
                f"📈 {col}: min={stats['min']:.4f}, médiane={stats['50%']:.4f}, max={stats['max']:.4f}"
            )

        # Corrélation
        corr = output_df["p_model_win"].corr(output_df["p_final"])
        logger.info(f"🔗 Corrélation original/calibré: {corr:.4f}")

        logger.info("\n🎉 CALIBRATION D6 TERMINÉE!")

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
