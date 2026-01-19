"""
🔄 Pipeline de Retraining Automatique - Phase 8 Online Learning
================================================================

Script pour retrainer automatiquement le modèle avec les nouveaux feedbacks.

Workflow:
1. Charger données originales d'entraînement
2. Récupérer feedbacks des N derniers jours
3. Merger feedbacks avec données originales
4. Retrainer Stacking Ensemble
5. Valider performance (ROC-AUC > seuil)
6. Sauvegarder nouveau modèle si validation OK
7. Archiver ancien modèle

Usage:
    python train_online.py --days 30 --min-roc-auc 0.70
    python train_online.py --dry-run  # Test sans sauvegarder

Scheduling:
    # Cron (chaque lundi à 3h du matin)
    0 3 * * 1 cd /path/to/horse3 && python train_online.py --days 7 >> logs/retraining.log 2>&1

Auteur: Phase 8 - Online Learning
Date: 2025-11-14
"""

import os
import sys
import json
import pickle
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/retraining.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class OnlineTrainer:
    """Gestionnaire du retraining automatique."""

    def __init__(
        self,
        days: int = 7,
        min_roc_auc: float = 0.70,
        min_new_samples: int = 100,
        dry_run: bool = False,
    ):
        """
        Initialise le trainer.

        Args:
            days: Nombre de jours de feedback à inclure
            min_roc_auc: ROC-AUC minimum pour valider nouveau modèle
            min_new_samples: Nombre minimum de nouveaux échantillons requis
            dry_run: Si True, ne sauvegarde pas le modèle
        """
        self.days = days
        self.min_roc_auc = min_roc_auc
        self.min_new_samples = min_new_samples
        self.dry_run = dry_run

        # Chemins
        self.data_dir = Path("data")
        self.models_dir = Path("data/models")
        self.champion_dir = self.models_dir / "champion"
        self.challenger_dir = self.models_dir / "challenger"
        self.archive_dir = self.models_dir / "archive"

        # Features (62 features comme modèle original)
        self.feature_columns = [
            # Forme récente (7)
            "forme_5c",
            "forme_10c",
            "nb_courses_12m",
            "nb_victoires_12m",
            "nb_places_12m",
            "derniere_place",
            "derniere_victoire",
            # Aptitude (3)
            "aptitude_distance",
            "aptitude_piste",
            "aptitude_hippodrome",
            # Jockey/Entraineur (6)
            "taux_victoires_jockey",
            "taux_places_jockey",
            "taux_victoires_entraineur",
            "taux_places_entraineur",
            "synergie_jockey_cheval",
            "synergie_entraineur_cheval",
            # Course (3)
            "distance_norm",
            "niveau_moyen_concurrent",
            "nb_partants",
            # Marché (5)
            "cote_turfbzh",
            "rang_cote_turfbzh",
            "cote_sp",
            "rang_cote_sp",
            "prediction_ia_gagnant",
            "elo_cheval",
            "ecart_cote_ia",
        ]

        self.metadata = {}

    def load_original_training_data(self) -> pd.DataFrame:
        """Charge les données d'entraînement originales."""
        logger.info("=" * 80)
        logger.info("📦 CHARGEMENT DONNÉES ORIGINALES")
        logger.info("=" * 80)

        # Chercher fichier features
        feature_files = [
            self.data_dir / "ml_features_complete.csv",
            self.data_dir / "ml_features.csv",
            self.data_dir / "normalized" / "X_train.parquet",
        ]

        for filepath in feature_files:
            if filepath.exists():
                logger.info(f"📂 Fichier trouvé: {filepath}")

                if filepath.suffix == ".parquet":
                    df = pd.read_parquet(filepath)
                else:
                    df = pd.read_csv(filepath)

                logger.info(f"   ✅ {len(df):,} lignes chargées")
                return df

        raise FileNotFoundError("❌ Aucun fichier de features trouvé")

    def load_feedback_data(self) -> pd.DataFrame:
        """
        Charge les feedbacks des N derniers jours.

        En production: requête PostgreSQL sur table feedback_results.
        En développement: stub avec données simulées.
        """
        logger.info("=" * 80)
        logger.info(f"📥 CHARGEMENT FEEDBACK ({self.days} derniers jours)")
        logger.info("=" * 80)

        # TODO: Remplacer par vraie requête PostgreSQL
        # SELECT * FROM feedback_results
        # WHERE timestamp_feedback >= NOW() - INTERVAL '{self.days} days'

        # Stub: données simulées pour développement
        logger.warning("⚠️  Mode STUB: utilisation données simulées (pas de PostgreSQL)")

        # Créer DataFrame vide avec bonnes colonnes
        feedback_df = pd.DataFrame(columns=["course_id", "cheval_id", "position_arrivee"])

        logger.info(f"   ℹ️  {len(feedback_df):,} feedbacks trouvés")

        if len(feedback_df) < self.min_new_samples:
            logger.warning(
                f"⚠️  Seulement {len(feedback_df)} feedbacks (min: {self.min_new_samples})"
            )
            logger.warning("   Retraining annulé: pas assez de nouvelles données")
            return None

        return feedback_df

    def merge_data(self, original_df: pd.DataFrame, feedback_df: pd.DataFrame) -> pd.DataFrame:
        """Merge données originales et feedbacks."""
        logger.info("=" * 80)
        logger.info("🔗 MERGE DONNÉES")
        logger.info("=" * 80)

        if feedback_df is None or len(feedback_df) == 0:
            logger.info("   ℹ️  Pas de feedback, utilisation données originales uniquement")
            return original_df

        # TODO: Implémenter vraie logique de merge
        # 1. Convertir feedback en format features
        # 2. Append à original_df
        # 3. Dédupliquer si nécessaire

        logger.info(f"   ✅ {len(original_df):,} lignes originales")
        logger.info(f"   ✅ {len(feedback_df):,} nouveaux feedbacks")

        merged_df = original_df.copy()  # Stub

        logger.info(f"   📊 Total: {len(merged_df):,} lignes")

        return merged_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prépare X et y pour entraînement."""
        logger.info("=" * 80)
        logger.info("🔧 PRÉPARATION FEATURES")
        logger.info("=" * 80)

        # Filtrer colonnes features disponibles
        available_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = set(self.feature_columns) - set(available_features)

        if missing_features:
            logger.warning(f"⚠️  {len(missing_features)} features manquantes: {missing_features}")

        logger.info(f"   ✅ {len(available_features)} features disponibles")

        # Extraire X et y
        X = df[available_features].copy()
        y = df["victoire"].copy()

        # Remplacer NaN par 0
        X = X.fillna(0)

        logger.info(f"   📊 X: {X.shape}")
        logger.info(f"   🎯 y: {y.shape} ({y.sum()} victoires, {100*y.mean():.1f}%)")

        return X, y

    def build_stacking_model(self) -> StackingClassifier:
        """Construit le modèle Stacking Ensemble (même architecture que Phase 6)."""
        logger.info("=" * 80)
        logger.info("🏗️  CONSTRUCTION MODÈLE STACKING")
        logger.info("=" * 80)

        # Base learners (hyperparams optimisés Phase 6)
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )

        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42
        )

        # Meta-learner
        meta_learner = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

        # Stacking
        stacking_model = StackingClassifier(
            estimators=[("rf", rf), ("xgb", xgb_model), ("lgb", lgb_model), ("gb", gb)],
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1,
        )

        logger.info("   ✅ Modèle Stacking configuré")
        logger.info("      - 4 base learners: RF, XGBoost, LightGBM, GB")
        logger.info("      - Meta-learner: LogisticRegression")
        logger.info("      - CV: 5 folds")

        return stacking_model

    def train_and_validate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[StackingClassifier, Dict[str, float]]:
        """Entraîne et valide le modèle."""
        logger.info("=" * 80)
        logger.info("🎓 ENTRAÎNEMENT & VALIDATION")
        logger.info("=" * 80)

        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"   📊 Train: {len(X_train):,} lignes")
        logger.info(f"   📊 Val: {len(X_val):,} lignes")

        # Construire modèle
        model = self.build_stacking_model()

        # Entraîner
        logger.info("   🔄 Entraînement en cours...")
        model.fit(X_train, y_train)
        logger.info("   ✅ Entraînement terminé")

        # Prédictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        # Métriques
        metrics = {
            "roc_auc": roc_auc_score(y_val, y_pred_proba),
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
        }

        logger.info("=" * 80)
        logger.info("📊 MÉTRIQUES VALIDATION")
        logger.info("=" * 80)
        logger.info(f"   🎯 ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"   ✅ Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   🎲 Precision: {metrics['precision']:.4f}")
        logger.info(f"   📈 Recall:    {metrics['recall']:.4f}")

        # Validation seuil
        if metrics["roc_auc"] < self.min_roc_auc:
            logger.error(f"❌ ROC-AUC {metrics['roc_auc']:.4f} < seuil {self.min_roc_auc}")
            logger.error("   Nouveau modèle REJETÉ")
            return None, metrics

        logger.info(f"✅ ROC-AUC {metrics['roc_auc']:.4f} >= seuil {self.min_roc_auc}")
        logger.info("   Nouveau modèle VALIDÉ")

        return model, metrics

    def save_model(self, model: StackingClassifier, metrics: Dict[str, float]) -> Path:
        """Sauvegarde le nouveau modèle comme challenger."""
        logger.info("=" * 80)
        logger.info("💾 SAUVEGARDE MODÈLE")
        logger.info("=" * 80)

        if self.dry_run:
            logger.info("   ℹ️  Mode DRY-RUN: pas de sauvegarde")
            return None

        # Créer dossier challenger
        self.challenger_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sauvegarder modèle
        model_path = self.challenger_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"   ✅ Modèle sauvegardé: {model_path}")

        # Metadata
        metadata = {
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "model_type": "stacking_ensemble",
            "version": "v1.1.0",  # Version challenger
            "metrics": metrics,
            "training": {
                "days_feedback": self.days,
                "min_roc_auc_threshold": self.min_roc_auc,
                "features_count": len(self.feature_columns),
            },
            "git_commit": self._get_git_commit(),
            "created_by": "train_online.py",
        }

        metadata_path = self.challenger_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"   ✅ Metadata sauvegardée: {metadata_path}")

        self.metadata = metadata

        return model_path

    def _get_git_commit(self) -> Optional[str]:
        """Récupère le hash du commit Git actuel."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except:
            return None

    def archive_old_champion(self):
        """Archive l'ancien modèle champion avant promotion."""
        logger.info("=" * 80)
        logger.info("📦 ARCHIVAGE ANCIEN CHAMPION")
        logger.info("=" * 80)

        if not (self.champion_dir / "model.pkl").exists():
            logger.info("   ℹ️  Pas de champion existant à archiver")
            return

        # Créer dossier archive avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_subdir = self.archive_dir / timestamp
        archive_subdir.mkdir(parents=True, exist_ok=True)

        # Copier champion → archive
        shutil.copy(self.champion_dir / "model.pkl", archive_subdir / "model.pkl")

        if (self.champion_dir / "metadata.json").exists():
            shutil.copy(self.champion_dir / "metadata.json", archive_subdir / "metadata.json")

        logger.info(f"   ✅ Champion archivé dans: {archive_subdir}")

    def run(self) -> bool:
        """Exécute le pipeline complet de retraining."""
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 20 + "🔄 PIPELINE RETRAINING AUTOMATIQUE" + " " * 24 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info("")

        try:
            # 1. Charger données originales
            original_df = self.load_original_training_data()

            # 2. Charger feedbacks
            feedback_df = self.load_feedback_data()

            if feedback_df is None:
                logger.warning("⚠️  Retraining annulé: pas assez de feedbacks")
                return False

            # 3. Merger données
            merged_df = self.merge_data(original_df, feedback_df)

            # 4. Préparer features
            X, y = self.prepare_features(merged_df)

            # 5. Entraîner et valider
            model, metrics = self.train_and_validate(X, y)

            if model is None:
                logger.error("❌ Validation échouée: modèle rejeté")
                return False

            # 6. Sauvegarder
            model_path = self.save_model(model, metrics)

            if model_path:
                logger.info("=" * 80)
                logger.info("🎉 RETRAINING RÉUSSI!")
                logger.info("=" * 80)
                logger.info(f"   📍 Nouveau modèle: {model_path}")
                logger.info(f"   🎯 ROC-AUC: {metrics['roc_auc']:.4f}")
                logger.info("   ℹ️  Modèle sauvegardé comme CHALLENGER")
                logger.info("   📝 Prochaine étape: A/B Testing puis promotion si performant")
                logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"❌ ERREUR FATALE: {e}", exc_info=True)
            return False


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Pipeline de retraining automatique - Phase 8 Online Learning"
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Nombre de jours de feedback à inclure (défaut: 7)"
    )
    parser.add_argument(
        "--min-roc-auc",
        type=float,
        default=0.70,
        help="ROC-AUC minimum pour valider nouveau modèle (défaut: 0.70)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Nombre minimum de nouveaux échantillons requis (défaut: 100)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Test sans sauvegarder le modèle")

    args = parser.parse_args()

    # Créer dossier logs
    Path("logs").mkdir(exist_ok=True)

    # Lancer retraining
    trainer = OnlineTrainer(
        days=args.days,
        min_roc_auc=args.min_roc_auc,
        min_new_samples=args.min_samples,
        dry_run=args.dry_run,
    )

    success = trainer.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
