#!/usr/bin/env python3
"""
ENTRAÎNEMENT MODÈLE CONSERVATEUR "SAFE"
========================================

Entraîne un modèle XGBoost avec des hyperparamètres conservateurs
pour une stratégie de betting plus défensive.

Différences vs modèle champion:
- Profondeur réduite (5 vs 8) → moins de surajustement
- Learning rate faible (0.05 vs 0.1) → convergence plus stable
- Régularisation forte (L1=0.5, L2=2.0) → généralisation
- min_child_weight élevé (10 vs 3) → évite patterns rares

Usage:
    python train_model_conservative.py [--dry-run]
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# HYPERPARAMÈTRES CONSERVATEURS
# ============================================================================

CONSERVATIVE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "logloss"],
    # Structure de l'arbre - CONSERVATEUR
    "max_depth": 5,  # Réduit (vs 8) → moins complexe
    "min_child_weight": 10,  # Augmenté (vs 3) → patterns stables
    "gamma": 0.3,  # Coût minimum pour split → arbres simples
    # Apprentissage - LENT ET STABLE
    "learning_rate": 0.05,  # Réduit (vs 0.1) → convergence douce
    "n_estimators": 2000,  # Plus d'itérations (early stop)
    # Régularisation - FORTE
    "reg_alpha": 0.5,  # L1 (vs 0.1) → sélection features
    "reg_lambda": 2.0,  # L2 (vs 1.0) → poids plus petits
    # Subsampling - CONSERVATEUR
    "subsample": 0.7,  # Réduit (vs 0.8)
    "colsample_bytree": 0.6,  # Réduit (vs 0.8)
    "colsample_bylevel": 0.8,
    # Gestion déséquilibre classes
    "scale_pos_weight": 0.8,  # Sous-pondère victoires rares
    # Autres
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",  # Plus rapide
}

# Critères de filtrage conservateurs
FILTER_CRITERIA = {
    "min_partants": 8,  # Exclure courses < 8 partants
    "max_partants": 18,  # Exclure très gros pelotons
    "min_horse_history": 5,  # Min 5 courses historiques
    "exclude_disciplines": [],  # Toutes disciplines OK
    "prefer_disciplines": ["PLAT", "ATTELE"],  # Plus prévisibles
}


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================


def load_data(data_dir: Path = Path("data")):
    """Charge les données SAFE (sans data leakage)"""

    logger.info("📂 CHARGEMENT DES DATASETS SAFE")

    train_path = data_dir / "train_SAFE.csv"
    val_path = data_dir / "val_SAFE.csv"
    test_path = data_dir / "test_SAFE.csv"

    # Vérifier existence
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"❌ Fichier manquant: {path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"✅ Train: {train_df.shape[0]:,} × {train_df.shape[1]}")
    logger.info(f"✅ Val: {val_df.shape[0]:,} × {val_df.shape[1]}")
    logger.info(f"✅ Test: {test_df.shape[0]:,} × {test_df.shape[1]}")

    return train_df, val_df, test_df


def apply_conservative_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les filtres conservateurs aux données"""

    initial_count = len(df)

    # Filtrer par nombre de partants si colonne existe
    if "nb_partants" in df.columns:
        df = df[
            (df["nb_partants"] >= FILTER_CRITERIA["min_partants"])
            & (df["nb_partants"] <= FILTER_CRITERIA["max_partants"])
        ]

    # Filtrer par historique du cheval si colonne existe
    if "nb_courses_cheval" in df.columns:
        df = df[df["nb_courses_cheval"] >= FILTER_CRITERIA["min_horse_history"]]

    final_count = len(df)
    filtered = initial_count - final_count

    if filtered > 0:
        logger.info(f"🔍 Filtré {filtered:,} lignes ({filtered/initial_count*100:.1f}%)")

    return df


def prepare_features(df: pd.DataFrame, feature_cols: list = None):
    """Prépare X et y pour l'entraînement"""

    # Colonnes à exclure
    target_cols = ["position_arrivee", "victoire", "place"]
    id_cols = [
        "id_performance",
        "id_course",
        "nom_norm",
        "date",
        "hippodrome",
        "course_id",
        "race_id",
        "cheval_id",
    ]

    exclude = target_cols + id_cols

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in exclude]

    # Sélectionner seulement les colonnes qui existent
    available_cols = [col for col in feature_cols if col in df.columns]

    X = df[available_cols].copy()

    # Gérer les colonnes catégorielles
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        from sklearn.preprocessing import LabelEncoder

        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Remplacer NaN par 0
    X = X.fillna(0)

    # Target
    if "victoire" in df.columns:
        y = df["victoire"].values
    else:
        y = None

    return X, y, available_cols


# ============================================================================
# ENTRAÎNEMENT
# ============================================================================


def train_conservative_model(X_train, y_train, X_val, y_val, params: dict = None):
    """Entraîne le modèle XGBoost conservateur"""

    logger.info("🚀 ENTRAÎNEMENT MODÈLE CONSERVATEUR")
    logger.info("=" * 60)

    if params is None:
        params = CONSERVATIVE_PARAMS.copy()

    # Afficher les hyperparamètres clés
    logger.info("📊 Hyperparamètres conservateurs:")
    logger.info(f"   • max_depth: {params['max_depth']} (réduit)")
    logger.info(f"   • learning_rate: {params['learning_rate']} (lent)")
    logger.info(f"   • min_child_weight: {params['min_child_weight']} (élevé)")
    logger.info(f"   • reg_alpha: {params['reg_alpha']} (L1 fort)")
    logger.info(f"   • reg_lambda: {params['reg_lambda']} (L2 fort)")

    # Créer DMatrix XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Early stopping agressif
    evals = [(dtrain, "train"), (dval, "val")]

    start_time = time.time()

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=params.get("n_estimators", 2000),
        evals=evals,
        early_stopping_rounds=100,  # Plus patient
        verbose_eval=False,
    )

    train_time = time.time() - start_time

    logger.info(f"⏱️  Temps d'entraînement: {train_time:.1f}s")
    logger.info(f"🌳 Nombre d'arbres: {model.best_iteration}")

    return model, train_time


def evaluate_model(model, X, y, dataset_name: str = "Test"):
    """Évalue le modèle et retourne les métriques"""

    dmatrix = xgb.DMatrix(X)
    y_prob = model.predict(dmatrix)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        "dataset": dataset_name,
        "auc": roc_auc_score(y, y_prob),
        "brier_score": brier_score_loss(y, y_prob),
        "log_loss": log_loss(y, y_prob),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "n_samples": len(y),
        "n_positives": int(sum(y)),
        "positive_rate": float(sum(y) / len(y)),
    }

    logger.info(f"📊 Métriques {dataset_name}:")
    logger.info(f"   🎯 AUC: {metrics['auc']:.4f}")
    logger.info(f"   📉 Brier Score: {metrics['brier_score']:.4f}")
    logger.info(f"   📈 Precision: {metrics['precision']:.4f}")
    logger.info(f"   📈 Recall: {metrics['recall']:.4f}")
    logger.info(f"   📈 F1: {metrics['f1']:.4f}")

    return metrics


# ============================================================================
# SAUVEGARDE
# ============================================================================


def save_model(
    model, scaler, feature_names: list, metrics: dict, output_dir: Path = Path("data/models/safe")
):
    """Sauvegarde le modèle et les artefacts"""

    logger.info(f"💾 SAUVEGARDE DES ARTEFACTS: {output_dir}")

    # Créer le répertoire
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Modèle XGBoost
    model_path = output_dir / "xgboost_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"   ✅ Modèle: {model_path}")

    # 2. Scaler
    if scaler is not None:
        scaler_path = output_dir / "feature_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"   ✅ Scaler: {scaler_path}")

    # 3. Feature names
    features_path = output_dir / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"   ✅ Features: {features_path}")

    # 4. Metadata
    metadata = {
        "model_type": "xgboost_conservative",
        "version": "safe_v1.0",
        "created_at": datetime.now().isoformat(),
        "hyperparameters": {k: v for k, v in CONSERVATIVE_PARAMS.items() if not k.startswith("n_")},
        "filter_criteria": FILTER_CRITERIA,
        "metrics": metrics,
        "n_features": len(feature_names),
        "description": "Modèle conservateur pour stratégie de betting défensive",
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"   ✅ Metadata: {metadata_path}")

    # 5. Imputer (copier depuis champion si existe)
    champion_imputer = Path("data/models/champion/feature_imputer.pkl")
    if champion_imputer.exists():
        import shutil

        shutil.copy(champion_imputer, output_dir / "feature_imputer.pkl")
        logger.info("   ✅ Imputer: copié depuis champion")

    return output_dir


# ============================================================================
# MAIN
# ============================================================================


def main(dry_run: bool = False):
    """Pipeline principal d'entraînement conservateur"""

    logger.info("🛡️ ENTRAÎNEMENT MODÈLE CONSERVATEUR 'SAFE'")
    logger.info("=" * 70)
    logger.info(f"🕐 Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if dry_run:
        logger.info("⚠️  MODE DRY-RUN: Pas de sauvegarde")

    try:
        # 1. Charger les données
        train_df, val_df, test_df = load_data()

        # 2. Appliquer filtres conservateurs
        logger.info("\n🔍 APPLICATION DES FILTRES CONSERVATEURS")
        train_df = apply_conservative_filters(train_df)
        val_df = apply_conservative_filters(val_df)
        test_df = apply_conservative_filters(test_df)

        # 3. Préparer les features
        logger.info("\n📊 PRÉPARATION DES FEATURES")
        X_train, y_train, feature_cols = prepare_features(train_df)
        X_val, y_val, _ = prepare_features(val_df, feature_cols)
        X_test, y_test, _ = prepare_features(test_df, feature_cols)

        logger.info(f"   • Features: {len(feature_cols)}")
        logger.info(f"   • Train: {len(X_train):,} samples")
        logger.info(f"   • Val: {len(X_val):,} samples")
        logger.info(f"   • Test: {len(X_test):,} samples")

        # 4. Normalisation
        logger.info("\n🔄 NORMALISATION")
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

        # 5. Entraînement
        model, train_time = train_conservative_model(X_train_scaled, y_train, X_val_scaled, y_val)

        # 6. Évaluation
        logger.info("\n📈 ÉVALUATION")
        metrics_train = evaluate_model(model, X_train_scaled, y_train, "Train")
        metrics_val = evaluate_model(model, X_val_scaled, y_val, "Validation")
        metrics_test = evaluate_model(model, X_test_scaled, y_test, "Test")

        all_metrics = {
            "train": metrics_train,
            "validation": metrics_val,
            "test": metrics_test,
            "train_time_seconds": train_time,
            "n_features": len(feature_cols),
            "best_iteration": model.best_iteration,
        }

        # 7. Sauvegarde
        if not dry_run:
            output_dir = save_model(
                model=model, scaler=scaler, feature_names=feature_cols, metrics=all_metrics
            )

            # Créer aussi le répertoire de calibration
            calib_dir = Path("calibration/safe")
            calib_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Répertoire calibration créé: {calib_dir}")

        # 8. Résumé
        logger.info("\n" + "=" * 70)
        logger.info("📋 RÉSUMÉ - MODÈLE CONSERVATEUR SAFE")
        logger.info("=" * 70)
        logger.info(f"🎯 AUC Test: {metrics_test['auc']:.4f}")
        logger.info(f"📉 Brier Score Test: {metrics_test['brier_score']:.4f}")
        logger.info(f"📈 Precision Test: {metrics_test['precision']:.4f}")
        logger.info(f"📈 F1 Score Test: {metrics_test['f1']:.4f}")
        logger.info(f"🌳 Nombre d'arbres: {model.best_iteration}")
        logger.info(f"⏱️  Temps total: {train_time:.1f}s")

        if not dry_run:
            logger.info(f"\n✅ Modèle sauvegardé: {output_dir}")
            logger.info("🔜 Prochaine étape: Calibrer avec calibration_pipeline.py")

        return model, all_metrics

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne un modèle XGBoost conservateur")
    parser.add_argument("--dry-run", action="store_true", help="Mode test sans sauvegarde")
    args = parser.parse_args()

    main(dry_run=args.dry_run)
