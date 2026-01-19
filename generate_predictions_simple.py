import pandas as pd
import pickle
import logging
import numpy as np
from pathlib import Path

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_features_like_training(df):
    """Prépare les features exactement comme pendant l'entraînement XGBoost."""

    exclude_patterns = [
        "date_course",
        "hippodrome",
        "numero_course",
        "nom_cheval",
        "nom_jockey",
        "nom_entraineur",
        "proprietaire",
        "victoire",
        "place",
        "ecart",
        "id_",
        "sexe_cheval",
        "position_arrivee",
    ]

    # Même logique que dans train_xgboost.py
    feature_cols = [
        col
        for col in df.columns
        if not any(pattern in col.lower() for pattern in exclude_patterns)
        and df[col].dtype in ["int64", "float64", "bool"]
    ]

    X = df[feature_cols].fillna(0).values
    return X, feature_cols


def main():
    logger.info("🚀 GÉNÉRATION PRÉDICTIONS XGBOOST (SIMPLE)")

    # Charger le modèle XGBoost
    model_path = "data/models/xgboost_model.pkl"
    logger.info(f"🤖 Chargement modèle: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Charger les données
    data_path = "data/normalized"
    all_predictions = []
    train_features = None  # Pour garder les features du train

    for split in ["train", "val", "test"]:
        csv_path = Path(data_path) / f"{split}.csv"
        if not csv_path.exists():
            continue

        logger.info(f"📊 Traitement {split}...")
        df = pd.read_csv(csv_path, low_memory=False)

        # Pour le train, découvrir les features
        if split == "train":
            X, feature_cols = prepare_features_like_training(df)
            train_features = feature_cols
            logger.info(f"   Features train: {len(train_features)}")
        else:
            # Pour val/test, utiliser exactement les mêmes features que train
            if train_features is None:
                logger.error("❌ Train doit être traité en premier")
                continue

            # Vérifier que toutes les features train sont présentes
            missing_features = set(train_features) - set(df.columns)
            if missing_features:
                logger.warning(f"   ⚠️  Features manquantes: {missing_features}")
                # Ajouter colonnes manquantes avec des zéros
                for feat in missing_features:
                    df[feat] = 0

            X = df[train_features].fillna(0).values
            feature_cols = train_features

        logger.info(f"   Features: {X.shape[1]}")

        # Prédictions
        try:
            pred_probs = model.predict_proba(X)[:, 1]
            logger.info(
                f"   ✅ Prédictions: min={pred_probs.min():.3f}, max={pred_probs.max():.3f}"
            )
        except Exception as e:
            logger.error(f"   ❌ Erreur: {e}")
            continue

        # Créer DataFrame résultat
        predictions = pd.DataFrame(
            {
                "race_key": df.get("race_key", ""),
                "id_cheval": df.get("id_cheval", ""),
                "date_course": df.get("date_course", ""),
                "hippodrome": df.get("hippodrome", ""),
                "nom_cheval": df.get("nom_cheval", ""),
                "numero_corde": df.get("numero_corde", 0),
                "p_model_win": pred_probs,
                "is_win": df.get("victoire", 0),  # Utiliser 'victoire' au lieu de 'is_win'
                "place": df.get("place", 0),
                "cote_sp": df.get("cote_sp", 0),
                "cote_pm": df.get("cote_pm", 0),
                "split": split,
            }
        )

        all_predictions.append(predictions)

    # Combiner et sauvegarder
    if all_predictions:
        final_df = pd.concat(all_predictions, ignore_index=True)

        # Statistiques
        logger.info(f"📊 Total: {len(final_df)} prédictions")
        logger.info(f"   Prob moyenne: {final_df['p_model_win'].mean():.3f}")
        if "is_win" in final_df.columns:
            logger.info(f"   Victoires réelles: {final_df['is_win'].mean():.3f}")

        # Sauvegarder
        output_path = "data/backtest_predictions.csv"
        final_df.to_csv(output_path, index=False)
        logger.info(f"💾 Sauvegardé: {output_path}")

        logger.info("🎉 Terminé!")
    else:
        logger.error("❌ Aucune prédiction générée")


if __name__ == "__main__":
    main()
