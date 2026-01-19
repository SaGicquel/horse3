#!/usr/bin/env python3
"""
optimize_hyperparameters.py - Optimisation Bayésienne avec Optuna

Phase 6 : Optimisation & Ensemble
Recherche optimale d'hyperparamètres pour XGBoost et LightGBM avec Time Series CV.

Usage:
    python optimize_hyperparameters.py --model xgboost --trials 100 --cv-folds 5
    python optimize_hyperparameters.py --model lightgbm --trials 100 --cv-folds 5

Auteur: Phase 6 ML Pipeline
Date: 2025-11-13
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# ============================================================================
# CONFIGURATION LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Désactiver logs verbeux d'Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# CHARGEMENT DONNÉES
# ============================================================================


def load_data(input_dir: Path):
    """Charge les datasets train+val pour optimisation."""
    logger.info("📂 Chargement des données...")

    df_train = pd.read_csv(input_dir / "train.csv")
    df_val = pd.read_csv(input_dir / "val.csv")

    # Combiner train+val pour CV
    df_full = pd.concat([df_train, df_val], ignore_index=True)

    logger.info(f"   ✅ Train+Val: {len(df_full):,} lignes")

    return df_full


def prepare_features(df: pd.DataFrame):
    """Prépare X et y en excluant metadata."""
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
        "position_arrivee",  # DATA LEAKAGE
    ]

    feature_cols = [
        col
        for col in df.columns
        if not any(pattern in col.lower() for pattern in exclude_patterns)
        and df[col].dtype in ["int64", "float64", "bool"]
    ]

    X = df[feature_cols].values
    y = df["victoire"].values

    logger.info(f"   📊 Features: {len(feature_cols)}")
    logger.info(f"   🎯 Victoires: {100*y.mean():.2f}%")

    return X, y, feature_cols


# ============================================================================
# OBJECTIFS OPTUNA
# ============================================================================


def objective_xgboost(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, n_folds: int) -> float:
    """
    Fonction objectif pour optimisation XGBoost.

    Args:
        trial: Essai Optuna
        X, y: Données d'entraînement
        n_folds: Nombre de folds pour TimeSeriesSplit

    Returns:
        Score moyen ROC-AUC sur les folds
    """
    # Search space pour XGBoost
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "gamma": trial.suggest_float("gamma", 0.1, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "scale_pos_weight": (len(y) - y.sum()) / y.sum(),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
    }

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=n_folds)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Ajouter early_stopping dans params pour XGBoost 3.x
        params_with_es = params.copy()
        params_with_es["early_stopping_rounds"] = 20

        model = xgb.XGBClassifier(**params_with_es)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)

        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, y_pred_proba)
        scores.append(score)

        # Pruning intermédiaire
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


def objective_lightgbm(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, n_folds: int) -> float:
    """
    Fonction objectif pour optimisation LightGBM.

    Args:
        trial: Essai Optuna
        X, y: Données d'entraînement
        n_folds: Nombre de folds pour TimeSeriesSplit

    Returns:
        Score moyen ROC-AUC sur les folds
    """
    # Search space pour LightGBM
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "num_leaves": trial.suggest_int("num_leaves", 15, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1.0),
        "scale_pos_weight": (len(y) - y.sum()) / y.sum(),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=n_folds)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, y_pred_proba)
        scores.append(score)

        # Pruning intermédiaire
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


# ============================================================================
# OPTIMISATION
# ============================================================================


def optimize_model(
    model_name: str, X: np.ndarray, y: np.ndarray, n_trials: int, n_folds: int, output_dir: Path
) -> Dict:
    """
    Lance l'optimisation Optuna.

    Args:
        model_name: 'xgboost' ou 'lightgbm'
        X, y: Données d'entraînement
        n_trials: Nombre d'essais Optuna
        n_folds: Nombre de folds pour TimeSeriesCV
        output_dir: Répertoire de sauvegarde

    Returns:
        Dictionnaire des meilleurs paramètres
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"🔍 OPTIMISATION {model_name.upper()} - OPTUNA")
    logger.info("=" * 80)
    logger.info(f"   🎯 Trials: {n_trials}")
    logger.info(f"   📊 CV Folds: {n_folds}")
    logger.info("")

    # Créer étude Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    # Choisir fonction objectif
    if model_name == "xgboost":
        objective_fn = lambda trial: objective_xgboost(trial, X, y, n_folds)
    elif model_name == "lightgbm":
        objective_fn = lambda trial: objective_lightgbm(trial, X, y, n_folds)
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")

    # Lancer optimisation
    logger.info("⏳ Optimisation en cours (cela peut prendre plusieurs minutes)...")
    start_time = time.time()

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[
            lambda study, trial: logger.info(
                f"   Trial {trial.number+1}/{n_trials}: ROC-AUC = {trial.value:.4f}"
            )
            if (trial.number + 1) % 10 == 0
            else None
        ],
    )

    elapsed = time.time() - start_time

    # Résultats
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ OPTIMISATION TERMINÉE")
    logger.info("=" * 80)
    logger.info(f"   ⏱️  Temps total: {elapsed/60:.1f} minutes")
    logger.info(f"   🏆 Meilleur ROC-AUC: {study.best_value:.4f}")
    logger.info(f"   📊 Trials complétés: {len(study.trials)}")
    logger.info(
        f"   ✂️  Trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    logger.info("")

    # Meilleurs paramètres
    logger.info("🎯 MEILLEURS HYPERPARAMÈTRES:")
    logger.info("-" * 80)
    best_params = study.best_params.copy()
    for param, value in sorted(best_params.items()):
        logger.info(f"   {param:<20} : {value}")

    # Sauvegarder
    results = {
        "model": model_name,
        "best_score": study.best_value,
        "best_params": best_params,
        "n_trials": n_trials,
        "n_folds": n_folds,
        "optimization_time_minutes": elapsed / 60,
        "completed_trials": len(study.trials),
        "pruned_trials": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
    }

    output_path = output_dir / f"best_params_{model_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info(f"💾 Résultats sauvegardés: {output_path}")

    # Sauvegarder l'étude Optuna complète
    study_path = output_dir / f"optuna_study_{model_name}.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    logger.info(f"💾 Étude Optuna: {study_path}")

    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Optimisation Bayésienne hyperparamètres avec Optuna"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["xgboost", "lightgbm", "both"],
        help="Modèle à optimiser: xgboost, lightgbm ou both",
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
        default="data/optimization",
        help="Répertoire de sauvegarde résultats (default: data/optimization)",
    )
    parser.add_argument(
        "--trials", type=int, default=100, help="Nombre de trials Optuna (default: 100)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Nombre de folds pour TimeSeriesSplit (default: 5)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("🎯 OPTUNA - OPTIMISATION BAYÉSIENNE HYPERPARAMÈTRES")
    logger.info("=" * 80)
    logger.info(f"   📂 Input:  {input_dir}")
    logger.info(f"   📂 Output: {output_dir}")
    logger.info(f"   🎲 Model:  {args.model}")
    logger.info(f"   🔢 Trials: {args.trials}")
    logger.info(f"   📊 CV Folds: {args.cv_folds}")
    logger.info("")

    try:
        # Charger données
        df_full = load_data(input_dir)
        X, y, feature_names = prepare_features(df_full)

        # Optimiser modèle(s)
        if args.model in ["xgboost", "both"]:
            optimize_model("xgboost", X, y, args.trials, args.cv_folds, output_dir)

        if args.model in ["lightgbm", "both"]:
            optimize_model("lightgbm", X, y, args.trials, args.cv_folds, output_dir)

        logger.info("")
        logger.info("=" * 80)
        logger.info("🎉 OPTIMISATION COMPLÈTE !")
        logger.info("=" * 80)
        logger.info(f"   📁 Résultats dans: {output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
