#!/usr/bin/env python3
"""
GÉNÉRATION PRÉDICTIONS PHASE D5 - XGBoost SAFE
==============================================

Applique le modèle XGBoost SAFE sur tout l'historique
pour générer les prédictions complètes.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_and_preprocessors():
    """Charge le modèle XGBoost SAFE et les preprocesseurs"""

    logger.info("🤖 Chargement du modèle XGBoost SAFE...")

    # Chargement du modèle
    model_path = "models/xgboost_SAFE_victoire.pkl"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"✅ Modèle chargé: {model_path}")

    # Chargement des preprocesseurs
    scaler_path = "models/scaler_SAFE.pkl"
    encoders_path = "models/label_encoders_SAFE.pkl"

    scaler = None
    if Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
        logger.info(f"✅ Scaler chargé: {scaler_path}")

    label_encoders = None
    if Path(encoders_path).exists():
        label_encoders = joblib.load(encoders_path)
        logger.info(f"✅ Label encoders chargés: {encoders_path}")

    return model, scaler, label_encoders


def load_historical_data():
    """Charge les données historiques complètes"""

    logger.info("📊 Chargement des données historiques SAFE...")

    # Chargement du dataset SAFE complet
    data_path = "data/ml_features_SAFE.csv"
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset non trouvé: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"✅ Dataset chargé: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    return df


def preprocess_data(df, scaler, label_encoders):
    """Prétraite les données pour les prédictions"""

    logger.info("🔄 Prétraitement des données...")

    # Identification des colonnes (exactement comme à l'entraînement)
    target_cols = ["position_arrivee", "victoire", "place"]
    id_cols = ["id_performance", "id_course", "nom_norm"]

    # Features exactement comme dans le dataset d'entraînement
    all_cols = set(df.columns)
    feature_cols = [
        col
        for col in df.columns
        if col not in target_cols + id_cols and col != "date_extracted" and col != "annee"
    ]

    logger.info(f"📊 Features à traiter: {len(feature_cols)}")

    # Extraction des features (sans les colonnes temporaires)
    X = df[feature_cols].copy()

    # Séparation numérique/catégoriel
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    logger.info(f"📊 Colonnes numériques: {len(numeric_cols)}")
    logger.info(f"🏷️  Colonnes catégorielles: {len(categorical_cols)}")

    # Encodage des colonnes catégorielles
    if label_encoders and categorical_cols:
        logger.info("🔤 Encodage des variables catégorielles...")

        for col in categorical_cols:
            if col in label_encoders:
                try:
                    X[col] = label_encoders[col].transform(X[col].astype(str))
                except ValueError as e:
                    # Gérer les nouvelles valeurs non vues à l'entraînement
                    logger.warning(f"⚠️  Valeurs non vues en entraînement pour {col}: {e}")
                    # Assigner une valeur par défaut (première classe)
                    unknown_mask = ~X[col].astype(str).isin(label_encoders[col].classes_)
                    X.loc[unknown_mask, col] = label_encoders[col].classes_[0]
                    X[col] = label_encoders[col].transform(X[col].astype(str))

    # Normalisation des colonnes numériques
    if scaler and numeric_cols:
        logger.info("📊 Normalisation des variables numériques...")
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X


def generate_predictions(model, X, batch_size=10000):
    """Génère les prédictions par batch"""

    logger.info(f"🎯 Génération des prédictions (batch size: {batch_size:,})...")

    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size

    predictions = []

    for i in tqdm(range(n_batches), desc="Prédictions"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        X_batch = X.iloc[start_idx:end_idx]

        # Prédiction avec XGBoost
        import xgboost as xgb

        dtest = xgb.DMatrix(X_batch)
        pred_batch = model.predict(dtest)

        predictions.extend(pred_batch)

    return np.array(predictions)


def create_output_dataframe(df, predictions):
    """Crée le DataFrame de sortie avec les prédictions"""

    logger.info("📋 Création du DataFrame de sortie...")

    # Extraction de la date depuis id_course (format: YYYY-MM-DD|R1|C1|VIN)
    date_extracted = df["id_course"].str.split("|").str[0]
    date_course_clean = pd.to_datetime(date_extracted)
    annee = date_course_clean.dt.year

    # Création du race_key (identifiant unique de course)
    race_key = df["id_course"]

    # Détermination du split train/val/test basé sur l'année
    def determine_split(annee):
        if annee <= 2022:
            return "train"
        elif annee == 2023:
            return "val"
        else:
            return "test"

    split = annee.apply(determine_split)

    # Construction du DataFrame final
    output_df = pd.DataFrame(
        {
            "race_key": race_key,
            "id_cheval": df["nom_norm"],  # Nom du cheval
            "date_course": date_course_clean.dt.strftime("%Y-%m-%d"),
            "p_model_win": predictions,  # Probabilité de victoire
            "is_win": df["victoire"],
            "place": df["place"],
            "cote_sp": df.get("cote_matin", np.nan),  # Cote du matin si disponible
            "cote_pm": df.get("cote_pm", np.nan),  # Cote PMU si disponible
            "split": split,
            "position_arrivee": df["position_arrivee"],
        }
    )

    return output_df


def main():
    """Pipeline principal"""

    logger.info("🚀 DÉMARRAGE GÉNÉRATION PRÉDICTIONS D5 - XGBoost SAFE")
    logger.info("=" * 70)

    parser = argparse.ArgumentParser(
        description="Génère les prédictions XGBoost SAFE sur l'historique"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10000, help="Taille des batches pour les prédictions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/backtest_predictions.csv",
        help="Chemin de sortie pour les prédictions",
    )

    args = parser.parse_args()

    try:
        # 1. Chargement du modèle et preprocesseurs
        model, scaler, label_encoders = load_model_and_preprocessors()

        # 2. Chargement des données historiques
        df = load_historical_data()

        # 3. Prétraitement
        X = preprocess_data(df, scaler, label_encoders)

        # 4. Génération des prédictions
        predictions = generate_predictions(model, X, batch_size=args.batch_size)

        # 5. Création du DataFrame final
        output_df = create_output_dataframe(df, predictions)

        # 6. Sauvegarde
        logger.info("💾 Sauvegarde des prédictions...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_df.to_csv(output_path, index=False)

        # 7. Statistiques finales
        logger.info("\n📊 STATISTIQUES FINALES:")
        logger.info("=" * 50)
        logger.info(f"📂 Fichier de sortie: {output_path}")
        logger.info(f"📊 Nombre de prédictions: {len(output_df):,}")
        logger.info(
            f"📅 Période couverte: {output_df['date_course'].min()} à {output_df['date_course'].max()}"
        )

        # Distribution des splits
        split_counts = output_df["split"].value_counts()
        for split_name, count in split_counts.items():
            pct = (count / len(output_df)) * 100
            logger.info(f"   📊 {split_name.upper()}: {count:,} ({pct:.1f}%)")

        # Statistiques des prédictions
        pred_stats = output_df["p_model_win"].describe()
        logger.info("\n🎯 STATISTIQUES PRÉDICTIONS:")
        logger.info(f"   📈 Min: {pred_stats['min']:.4f}")
        logger.info(f"   📈 Moyenne: {pred_stats['mean']:.4f}")
        logger.info(f"   📈 Médiane: {pred_stats['50%']:.4f}")
        logger.info(f"   📈 Max: {pred_stats['max']:.4f}")

        # Validation basique
        victoires_reelles = output_df["is_win"].sum()
        taux_victoires = (victoires_reelles / len(output_df)) * 100
        logger.info("\n✅ VALIDATION:")
        logger.info(f"   🏆 Victoires réelles: {victoires_reelles:,} ({taux_victoires:.1f}%)")
        logger.info(f"   📊 Lignes avec prédictions: {(~output_df['p_model_win'].isna()).sum():,}")

        logger.info("\n🎉 GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
        logger.info(f"📂 Fichier final: {output_path}")

    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
