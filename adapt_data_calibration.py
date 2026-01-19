#!/usr/bin/env python3
"""
ADAPTATION DES DONNÉES POUR CALIBRATION D6
==========================================

Adapte le fichier backtest_predictions.csv pour être compatible
avec le pipeline de calibration.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def adapt_data_for_calibration():
    """Adapte les données pour la calibration"""

    logger.info("🔧 ADAPTATION DONNÉES POUR CALIBRATION D6")
    logger.info("=" * 70)

    # Chargement des données
    logger.info("📂 Chargement backtest_predictions.csv...")
    df = pd.read_csv("data/backtest_predictions.csv")

    logger.info(f"✅ Données chargées: {len(df):,} lignes")
    logger.info(f"📊 Colonnes disponibles: {list(df.columns)}")

    # Adaptation des colonnes
    logger.info("🔄 Adaptation des colonnes...")

    # 1. race_id depuis race_key
    df["race_id"] = df["race_key"]

    # 2. logits_model depuis p_model_win (conversion probabilité -> logits)
    # Logits = log(p / (1-p))
    # On évite les valeurs extrêmes avec un clipping
    p_clipped = np.clip(df["p_model_win"], 1e-6, 1 - 1e-6)
    df["logits_model"] = np.log(p_clipped / (1 - p_clipped))

    # 3. label_win depuis is_win
    df["label_win"] = df["is_win"]

    # 4. odds_market_preoff depuis cote_sp
    df["odds_market_preoff"] = df["cote_sp"]

    # 5. date_course au bon format
    df["date"] = pd.to_datetime(df["date_course"])

    # 6. discipline depuis les données (si pas disponible, on met 'trot' par défaut)
    if "discipline" not in df.columns:
        # Heuristique simple basée sur les cotes
        # Trot: généralement cotes plus serrées
        # Galop: cotes plus étalées
        median_odds = df.groupby("race_key")["cote_sp"].median()
        df["discipline"] = df["race_key"].map(
            lambda x: "galop" if median_odds.get(x, 10) > 8 else "trot"
        )

    # Statistiques de contrôle
    logger.info("📊 STATISTIQUES DE CONTRÔLE:")

    # Distribution des logits
    logits_stats = df["logits_model"].describe()
    logger.info(f"   📈 Logits - Min: {logits_stats['min']:.3f}, Max: {logits_stats['max']:.3f}")
    logger.info(
        f"   📈 Logits - Médiane: {logits_stats['50%']:.3f}, Moyenne: {logits_stats['mean']:.3f}"
    )

    # Distribution des probabilités originales
    prob_stats = df["p_model_win"].describe()
    logger.info(f"   🎯 Probabilités - Min: {prob_stats['min']:.6f}, Max: {prob_stats['max']:.6f}")
    logger.info(
        f"   🎯 Probabilités - Médiane: {prob_stats['50%']:.4f}, Moyenne: {prob_stats['mean']:.4f}"
    )

    # Couverture cotes
    cotes_coverage = (~df["odds_market_preoff"].isnull()).sum()
    coverage_pct = (cotes_coverage / len(df)) * 100
    logger.info(f"   💰 Couverture cotes: {cotes_coverage:,} ({coverage_pct:.1f}%)")

    # Distribution disciplines
    disc_dist = df["discipline"].value_counts()
    logger.info("   🏇 Distribution disciplines:")
    for disc, count in disc_dist.items():
        pct = (count / len(df)) * 100
        logger.info(f"      {disc}: {count:,} ({pct:.1f}%)")

    # Nombre de courses et chevaux par course
    n_races = df["race_id"].nunique()
    horses_per_race = df.groupby("race_id").size()
    logger.info(f"   🏁 Nombre de courses: {n_races:,}")
    logger.info(
        f"   🐎 Chevaux par course: {horses_per_race.mean():.1f} (min: {horses_per_race.min()}, max: {horses_per_race.max()})"
    )

    # Période temporelle
    date_min = df["date"].min()
    date_max = df["date"].max()
    logger.info(f"   📅 Période: {date_min.strftime('%Y-%m-%d')} à {date_max.strftime('%Y-%m-%d')}")

    # Validation des données
    logger.info("✅ VALIDATION DES DONNÉES ADAPTÉES:")

    # Vérifier cohérence logits/probabilités
    # Reconvertir logits -> proba pour vérifier
    p_reconstructed = 1 / (1 + np.exp(-df["logits_model"]))
    diff_max = np.abs(p_reconstructed - p_clipped).max()
    logger.info(f"   📊 Cohérence logits/probas: erreur max = {diff_max:.8f}")

    # Vérifier valeurs nulles critiques
    null_cols = ["race_id", "logits_model", "label_win", "date"]
    for col in null_cols:
        n_nulls = df[col].isnull().sum()
        if n_nulls > 0:
            logger.warning(f"   ⚠️  Valeurs nulles dans {col}: {n_nulls}")
        else:
            logger.info(f"   ✅ {col}: aucune valeur nulle")

    # Garder les colonnes nécessaires + quelques autres utiles
    output_cols = [
        "race_id",
        "id_cheval",
        "date",
        "date_course",
        "logits_model",
        "p_model_win",  # Modèle
        "label_win",
        "place",
        "position_arrivee",  # Targets
        "odds_market_preoff",
        "cote_sp",  # Marché
        "split",
        "discipline",  # Méta
    ]

    # S'assurer que toutes les colonnes existent
    available_cols = [col for col in output_cols if col in df.columns]
    df_output = df[available_cols].copy()

    # Sauvegarde
    output_path = "data/backtest_predictions_adapted.csv"
    logger.info("💾 Sauvegarde des données adaptées...")
    df_output.to_csv(output_path, index=False)

    logger.info(f"✅ Données adaptées sauvegardées: {output_path}")
    logger.info(
        f"📊 Dimensions finales: {len(df_output):,} lignes × {len(df_output.columns)} colonnes"
    )
    logger.info(f"📋 Colonnes finales: {list(df_output.columns)}")

    return output_path


if __name__ == "__main__":
    adapt_data_for_calibration()
