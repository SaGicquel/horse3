#!/usr/bin/env python3
"""
train_lightgbm.py - Entraînement LightGBM pour prédiction victoires PMU

Phase 5 : Modèles Avancés
LightGBM est ultra-rapide et performant sur données tabulaires avec gradient boosting.

Usage:
    python train_lightgbm.py --input-dir data/normalized --output-dir data/models

Auteur: Phase 5 ML Pipeline
Date: 2025-11-13
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================


def load_splits(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les datasets train/val/test normalisés.

    Args:
        input_dir: Répertoire contenant train.csv, val.csv, test.csv

    Returns:
        Tuple (df_train, df_val, df_test)
    """
    logger.info("📂 Chargement des données normalisées...")

    train_path = input_dir / "train.csv"
    val_path = input_dir / "val.csv"
    test_path = input_dir / "test.csv"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"❌ Fichiers manquants dans {input_dir}. " "Exécutez normalize_features.py d'abord."
        )

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    logger.info(f"   ✅ Train: {len(df_train):,} lignes")
    logger.info(f"   ✅ Val:   {len(df_val):,} lignes")
    logger.info(f"   ✅ Test:  {len(df_test):,} lignes")

    return df_train, df_val, df_test


def prepare_features(
    df: pd.DataFrame, exclude_patterns: list = None, feature_names: list = None
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prépare X et y en excluant colonnes métadonnées et target.

    Args:
        df: DataFrame avec features et target 'victoire'
        exclude_patterns: Liste de patterns à exclure des features
        feature_names: Si fourni, utilise exactement ces colonnes (pour val/test)

    Returns:
        Tuple (X, y, feature_names)
    """
    if exclude_patterns is None:
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
            "position_arrivee",  # DATA LEAKAGE - c'est la target !
        ]

    # Si feature_names fourni, utiliser exactement ces colonnes
    if feature_names is not None:
        feature_cols = feature_names
    else:
        # Filtrer colonnes features (pour train)
        feature_cols = [
            col
            for col in df.columns
            if not any(pattern in col.lower() for pattern in exclude_patterns)
            and df[col].dtype in ["int64", "float64", "bool"]
        ]

    X = df[feature_cols].values
    y = df["victoire"].values

    logger.info(f"   📊 Features sélectionnées: {len(feature_cols)}")
    logger.info(f"   🎯 Distribution target: {100*y.mean():.2f}% victoires")

    return X, y, feature_cols


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict = None,
) -> lgb.LGBMClassifier:
    """
    Entraîne un modèle LightGBM avec early stopping.

    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        params: Hyperparamètres optionnels

    Returns:
        Modèle LightGBM entraîné
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 ENTRAÎNEMENT LIGHTGBM")
    logger.info("=" * 80)

    # Paramètres par défaut optimisés pour LightGBM
    default_params = {
        "n_estimators": 500,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "max_depth": -1,  # -1 = pas de limite
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum(),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    if params:
        default_params.update(params)

    logger.info("📋 Hyperparamètres:")
    for k, v in default_params.items():
        logger.info(f"   {k}: {v}")

    # Création du modèle
    model = lgb.LGBMClassifier(**default_params)

    # Entraînement avec early stopping
    logger.info("")
    logger.info("⏳ Entraînement en cours...")
    start_time = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    elapsed = time.time() - start_time
    logger.info(f"✅ Entraînement terminé en {elapsed:.1f}s")
    logger.info(f"   🌲 Nombre d'arbres: {model.best_iteration_}")

    return model


def evaluate_model(
    model: lgb.LGBMClassifier, X: np.ndarray, y: np.ndarray, dataset_name: str = "Validation"
) -> Dict:
    """
    Évalue le modèle et retourne métriques détaillées.

    Args:
        model: Modèle LightGBM entraîné
        X, y: Données de test
        dataset_name: Nom du dataset (pour affichage)

    Returns:
        Dictionnaire de métriques
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"📊 ÉVALUATION - {dataset_name.upper()}")
    logger.info("=" * 80)

    # Prédictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Métriques
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_pred_proba),
    }

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info("")
    logger.info("🎯 MÉTRIQUES PRINCIPALES:")
    logger.info(f"   Accuracy  : {metrics['accuracy']:.4f} ({100*metrics['accuracy']:.2f}%)")
    logger.info(f"   Precision : {metrics['precision']:.4f}")
    logger.info(f"   Recall    : {metrics['recall']:.4f}")
    logger.info(f"   F1-Score  : {metrics['f1']:.4f}")
    logger.info(f"   ROC-AUC   : {metrics['roc_auc']:.4f} ⭐")

    logger.info("")
    logger.info("📋 CONFUSION MATRIX:")
    logger.info(f"   TN={tn:5,}  FP={fp:4,}")
    logger.info(f"   FN={fn:4,}  TP={tp:4,}")

    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def plot_feature_importance(
    model: lgb.LGBMClassifier, feature_names: list, output_dir: Path, top_n: int = 20
):
    """
    Génère graphique des top N features importantes.

    Args:
        model: Modèle LightGBM entraîné
        feature_names: Noms des features
        output_dir: Répertoire de sauvegarde
        top_n: Nombre de features à afficher
    """
    logger.info("")
    logger.info(f"📊 Top {top_n} features les plus importantes:")
    logger.info("-" * 80)

    # Récupérer importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    # Afficher dans console
    for i, idx in enumerate(indices, 1):
        logger.info(f"   {i:2d}. {feature_names[idx]:<35} : {100*importances[idx]:5.2f}%")

    # Créer graphique
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[indices], color="#2E86AB")
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Importance", fontsize=12)
    plt.title(f"Top {top_n} Features - LightGBM", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = output_dir / "lightgbm_feature_importance.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"   💾 Graphique sauvegardé: {output_path}")


def plot_roc_curve(
    model: lgb.LGBMClassifier, X_val: np.ndarray, y_val: np.ndarray, output_dir: Path
):
    """
    Génère courbe ROC.

    Args:
        model: Modèle LightGBM entraîné
        X_val, y_val: Données de validation
        output_dir: Répertoire de sauvegarde
    """
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    auc = roc_auc_score(y_val, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"LightGBM (AUC = {auc:.4f})", color="#2E86AB")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aléatoire")
    plt.xlabel("Taux de Faux Positifs", fontsize=11)
    plt.ylabel("Taux de Vrais Positifs", fontsize=11)
    plt.title("Courbe ROC - LightGBM", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "lightgbm_roc_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"   💾 Courbe ROC sauvegardée: {output_path}")


def save_model_and_metrics(
    model: lgb.LGBMClassifier,
    metrics_train: Dict,
    metrics_val: Dict,
    metrics_test: Dict,
    feature_names: list,
    output_dir: Path,
):
    """
    Sauvegarde modèle et métriques.

    Args:
        model: Modèle LightGBM entraîné
        metrics_train, metrics_val, metrics_test: Métriques par dataset
        feature_names: Noms des features
        output_dir: Répertoire de sauvegarde
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("💾 SAUVEGARDE MODÈLE ET MÉTRIQUES")
    logger.info("=" * 80)

    # Sauvegarder modèle
    model_path = output_dir / "lightgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"   ✅ Modèle: {model_path}")

    # Sauvegarder métriques JSON
    metrics_all = {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
        "feature_names": feature_names,
        "model_params": model.get_params(),
        "best_iteration": int(model.best_iteration_),  # LightGBM uses underscore
    }

    metrics_path = output_dir / "lightgbm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    logger.info(f"   ✅ Métriques: {metrics_path}")

    logger.info("")
    logger.info("🎉 Entraînement LightGBM terminé avec succès !")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Entraînement LightGBM pour prédiction victoires PMU"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/normalized",
        help="Répertoire contenant données normalisées (default: data/normalized)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Répertoire de sauvegarde modèle (default: data/models)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 LIGHTGBM - PHASE 5 MODÈLES AVANCÉS")
    logger.info("=" * 80)
    logger.info(f"   📂 Input:  {input_dir}")
    logger.info(f"   📂 Output: {output_dir}")
    logger.info("")

    try:
        # 1. Charger données
        df_train, df_val, df_test = load_splits(input_dir)

        # 2. Préparer features (définir colonnes sur train, réutiliser pour val/test)
        X_train, y_train, feature_names = prepare_features(df_train)
        X_val, y_val, _ = prepare_features(df_val, feature_names=feature_names)
        X_test, y_test, _ = prepare_features(df_test, feature_names=feature_names)

        # 3. Entraîner LightGBM
        model = train_lightgbm(X_train, y_train, X_val, y_val)

        # 4. Évaluer sur tous les splits
        metrics_train = evaluate_model(model, X_train, y_train, "Train")
        metrics_val = evaluate_model(model, X_val, y_val, "Validation")
        metrics_test = evaluate_model(model, X_test, y_test, "Test")

        # 5. Visualisations
        plot_feature_importance(model, feature_names, output_dir)
        plot_roc_curve(model, X_val, y_val, output_dir)

        # 6. Sauvegarder
        save_model_and_metrics(
            model, metrics_train, metrics_val, metrics_test, feature_names, output_dir
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ RÉSUMÉ FINAL - LIGHTGBM")
        logger.info("=" * 80)
        logger.info(f"   🎯 ROC-AUC Validation: {metrics_val['roc_auc']:.4f}")
        logger.info(f"   🎯 ROC-AUC Test:       {metrics_test['roc_auc']:.4f}")
        logger.info(f"   📊 Accuracy Test:      {100*metrics_test['accuracy']:.2f}%")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
