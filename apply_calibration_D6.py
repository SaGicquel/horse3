#!/usr/bin/env python3
"""
APPLICATION DE LA CALIBRATION D6 - GÉNÉRATION FINALE
====================================================

Utilise le pipeline de calibration pour transformer toutes les données
et générer backtest_predictions_calibrated.csv complet.
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from datetime import datetime

# Ajouter le chemin pour importer le pipeline
sys.path.append(".")

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Pipeline principal"""

    logger.info("🚀 GÉNÉRATION BACKTEST_PREDICTIONS_CALIBRATED.CSV")
    logger.info("=" * 70)

    try:
        # Importer le pipeline de calibration
        logger.info("📦 Import du pipeline de calibration...")
        from calibration_pipeline import CalibrationPipeline

        # 1. Chargement du pipeline calibré
        logger.info("🔧 Chargement du pipeline calibré...")
        pipeline = CalibrationPipeline.load("calibration")
        logger.info("✅ Pipeline chargé depuis calibration/")

        # 2. Chargement des données adaptées
        logger.info("📂 Chargement des données adaptées...")
        df = pd.read_csv("data/backtest_predictions_adapted.csv")
        logger.info(f"✅ Données chargées: {len(df):,} lignes")

        # 3. Application de la transformation
        logger.info("🔄 Application de la calibration complète...")

        result_df = pipeline.transform(
            df, logits_col="logits_model", race_col="race_id", cluster_col="discipline"
        )

        logger.info(f"✅ Transformation terminée: {len(result_df):,} lignes")

        # 4. Préparation du fichier final
        logger.info("📋 Préparation du fichier final...")

        # Mappage vers le format final attendu
        output_df = pd.DataFrame()
        output_df["race_key"] = result_df["race_id"]
        output_df["id_cheval"] = result_df["id_cheval"]
        output_df["date_course"] = result_df["date_course"]

        # Probabilités (toutes les étapes)
        output_df["p_model_win"] = result_df["p_model_win"]  # Original

        # Nouvelles colonnes de calibration
        if "p_model_norm" in result_df.columns:
            output_df["p_model_norm"] = result_df["p_model_norm"]
        if "p_calibrated" in result_df.columns:
            output_df["p_calibrated"] = result_df["p_calibrated"]
        if "p_final" in result_df.columns:
            output_df["p_final"] = result_df["p_final"]

        # Targets et métadonnées
        output_df["is_win"] = result_df["label_win"]
        output_df["place"] = result_df["place"]
        output_df["position_arrivee"] = result_df["position_arrivee"]
        output_df["cote_sp"] = result_df["cote_sp"]
        output_df["split"] = result_df["split"]

        # Si p_final n'existe pas, utiliser p_calibrated ou p_model_norm
        if "p_final" not in output_df.columns:
            if "p_calibrated" in output_df.columns:
                output_df["p_final"] = output_df["p_calibrated"]
                logger.info("📊 p_final = p_calibrated (pas de blend)")
            elif "p_model_norm" in output_df.columns:
                output_df["p_final"] = output_df["p_model_norm"]
                logger.info("📊 p_final = p_model_norm (pas de calibration)")
            else:
                output_df["p_final"] = output_df["p_model_win"]
                logger.info("📊 p_final = p_model_win (pas de transformation)")

        # 5. Sauvegarde
        output_path = "data/backtest_predictions_calibrated.csv"
        logger.info("💾 Sauvegarde du fichier final...")

        output_df.to_csv(output_path, index=False)

        # 6. Statistiques finales
        logger.info("\n📊 STATISTIQUES FINALES:")
        logger.info("=" * 50)
        logger.info(f"📂 Fichier de sortie: {output_path}")
        logger.info(f"📊 Nombre de lignes: {len(output_df):,}")
        logger.info(f"📊 Colonnes: {list(output_df.columns)}")

        # Distribution des probabilités
        prob_cols = [col for col in output_df.columns if col.startswith("p_")]
        for col in prob_cols:
            if col in output_df.columns:
                stats = output_df[col].describe()
                logger.info(
                    f"📈 {col}: min={stats['min']:.4f}, médiane={stats['50%']:.4f}, max={stats['max']:.4f}"
                )

        # Vérification calibration par course (si applicable)
        if "p_final" in output_df.columns:
            race_sums = output_df.groupby("race_key")["p_final"].sum()
            perfect_sums = (np.abs(race_sums - 1.0) < 1e-3).sum()
            logger.info(
                f"🔍 Courses avec somme ~1.0: {perfect_sums:,}/{len(race_sums):,} ({perfect_sums/len(race_sums)*100:.1f}%)"
            )

        # Comparaison avant/après
        if "p_model_win" in output_df.columns and "p_final" in output_df.columns:
            corr = output_df["p_model_win"].corr(output_df["p_final"])
            logger.info(f"🔗 Corrélation modèle original/final: {corr:.4f}")

        logger.info("\n🎉 CALIBRATION D6 TERMINÉE AVEC SUCCÈS!")
        logger.info(f"📂 Fichier final: {output_path}")

    except Exception as e:
        logger.error(f"❌ Erreur lors de la calibration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

if __name__ == "__main__":
    exit(main())
