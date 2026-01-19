import pandas as pd
import pickle
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Chemins par défaut
DEFAULT_MODEL_PATH = "data/models/xgboost_model.pkl"
DATA_PATH = "data/normalized"
OUTPUT_PATH = "data/backtest_predictions.csv"


def load_model(model_path):
    """Charge le modèle XGBoost."""
    logger.info(f"🤖 Chargement du modèle : {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("✅ Modèle chargé avec succès")
    return model


def load_data():
    """Charge les données normalisées (matrices NumPy)."""
    logger.info(f"📊 Chargement des données : {DATA_PATH}")

    datasets = {}

    for split in ["train", "val", "test"]:
        X_file = Path(DATA_PATH) / f"X_{split}.npy"
        y_file = Path(DATA_PATH) / f"y_{split}_victoire.npy"
        csv_file = Path(DATA_PATH) / f"{split}.csv"

        if X_file.exists() and y_file.exists() and csv_file.exists():
            logger.info(f"   Chargement {split}...")

            # Charger matrices NumPy
            X = np.load(X_file, allow_pickle=True)
            y = np.load(y_file, allow_pickle=True)

            # Charger CSV pour les métadonnées
            df_meta = pd.read_csv(csv_file)

            datasets[split] = {"X": X, "y": y, "metadata": df_meta}

            logger.info(f"   ✅ {split}: {X.shape[0]} lignes, {X.shape[1]} features")
        else:
            logger.warning(f"   ⚠️  Fichiers {split} non trouvés")

    return datasets


def extract_features_and_metadata(df):
    """Extrait les features pour le modèle et les métadonnées pour le backtest."""

    # Colonnes de métadonnées à conserver
    meta_cols = [
        "race_key",
        "id_cheval",
        "date_course",
        "is_win",
        "place",
        "cote_sp",
        "cote_pm",
        "nom_cheval",
        "hippodrome",
        "numero_corde",
    ]

    # Colonnes à exclure des features (métadonnées + colonnes texte)
    exclude_cols = [
        "race_key",
        "id_cheval",
        "date_course",
        "is_win",
        "place",
        "nom_cheval",
        "hippodrome",
        "jockey",
        "entraineur",
        "proprietaire",
        "musique",
        "commentaire",
        "temps_secteur",
        "temps_final",
        "id_course",
        "sexe_cheval",
        "discipline",
        "etat_piste",
        "nom_hippodrome",
        "type_piste",
        "hippodrome_ville",
        "hippodrome_top20",
        "ecart",
    ]

    # Features disponibles (seulement numériques)
    available_cols = df.columns.tolist()
    feature_cols = []

    for col in available_cols:
        if col not in exclude_cols:
            # Vérifier si la colonne est numérique
            if df[col].dtype in ["int64", "float64", "bool", "int32", "float32"]:
                feature_cols.append(col)
            elif col.startswith(("discipline_", "sexe_", "etat_", "hippodrome_", "type_")):
                # Colonnes one-hot encodées
                feature_cols.append(col)

    # Métadonnées disponibles
    available_meta_cols = [col for col in meta_cols if col in available_cols]

    logger.info(f"   📊 {len(feature_cols)} features numériques sélectionnées")
    logger.info(f"   📋 {len(available_meta_cols)} métadonnées conservées")

    X = df[feature_cols].fillna(0)
    metadata = df[available_meta_cols].copy()

    return X, metadata


def generate_predictions(model, datasets):
    """Génère les prédictions pour tous les datasets."""

    all_predictions = []

    for split_name, data_dict in datasets.items():
        X = data_dict["X"]
        y = data_dict["y"]
        df_meta = data_dict["metadata"]

        logger.info(
            f"🔮 Prédictions sur {split_name} ({X.shape[0]} lignes, {X.shape[1]} features)..."
        )

        # Prédire les probabilités
        try:
            # XGBoost retourne les probas de classe 1 (victoire)
            pred_probs = model.predict_proba(X)[:, 1]
            logger.info(
                f"   ✅ Prédictions générées (min: {pred_probs.min():.3f}, max: {pred_probs.max():.3f})"
            )
        except Exception as e:
            logger.error(f"   ❌ Erreur prédiction : {e}")
            continue

        # Extraire métadonnées disponibles
        meta_cols = [
            "race_key",
            "id_cheval",
            "date_course",
            "is_win",
            "place",
            "cote_sp",
            "cote_pm",
            "nom_cheval",
            "hippodrome",
            "numero_corde",
        ]

        available_meta_cols = [col for col in meta_cols if col in df_meta.columns]

        # Créer le DataFrame de prédictions
        predictions = df_meta[available_meta_cols].copy()
        predictions["p_model_win"] = pred_probs
        predictions["split"] = split_name
        predictions["y_actual"] = y

        all_predictions.append(predictions)

    # Combiner tous les datasets
    if all_predictions:
        final_df = pd.concat(all_predictions, ignore_index=True)
        logger.info(f"📊 Total : {len(final_df)} prédictions générées")
        return final_df
    else:
        logger.error("❌ Aucune prédiction générée")
        return None


def save_predictions(predictions_df, output_path):
    """Sauvegarde les prédictions."""
    logger.info(f"💾 Sauvegarde : {output_path}")

    # Trier par date puis probabilité décroissante
    if "date_course" in predictions_df.columns:
        predictions_df = predictions_df.sort_values(
            ["date_course", "p_model_win"], ascending=[True, False]
        )

    predictions_df.to_csv(output_path, index=False)
    logger.info(f"✅ {len(predictions_df)} prédictions sauvegardées")

    # Statistiques rapides
    if "p_model_win" in predictions_df.columns:
        logger.info("📊 Statistiques prédictions :")
        logger.info(f"   Probabilité moyenne : {predictions_df['p_model_win'].mean():.3f}")
        logger.info(f"   Écart-type : {predictions_df['p_model_win'].std():.3f}")

        if "y_actual" in predictions_df.columns:
            # Analyse de la performance
            actual_wins = predictions_df["y_actual"].mean()
            logger.info(f"   Taux de victoire réel : {actual_wins:.3f}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Génération de prédictions XGBoost")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Chemin vers le modèle")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Fichier de sortie")
    args = parser.parse_args()

    logger.info("🚀 GÉNÉRATION PRÉDICTIONS XGBOOST")
    logger.info("=" * 50)

    try:
        # Charger le modèle
        model = load_model(args.model)

        # Charger les données
        datasets = load_data()
        if not datasets:
            logger.error("❌ Aucun dataset chargé")
            return

        # Générer les prédictions
        predictions_df = generate_predictions(model, datasets)
        if predictions_df is None:
            return

        # Sauvegarder
        save_predictions(predictions_df, args.output)

        logger.info("🎉 Génération terminée avec succès !")

    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        raise


if __name__ == "__main__":
    main()
