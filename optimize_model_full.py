#!/usr/bin/env python3
"""
OPTIMISATION COMPLÈTE DU MODÈLE SAFE SUR 5 ANS DE DONNÉES
==========================================================

Pipeline d'optimisation avec:
- Optuna pour la recherche d'hyperparamètres
- Validation croisée temporelle (TimeSeriesSplit)
- Métrique personnalisée: ROI (pas seulement AUC)
- Équilibrage des classes
- Calibration des probabilités

Usage:
    python optimize_model_full.py [--n-trials 50] [--timeout 7200]
"""

import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Réduire les logs Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "data_path": "data/ml_features.csv",  # Toutes les données 5 ans
    "output_dir": "data/models/optimized",
    "n_trials": 100,  # Nombre d'essais Optuna
    "timeout": 7200,  # 2h max
    "n_splits": 5,  # 5-fold temporel
    "early_stopping": 100,
    "random_state": 42,
    # Critères de filtrage safe
    "min_partants": 6,
    "max_partants": 20,
    # Pondération positive (favorise les victoires)
    "boosted_scale_pos_weight": True,
}


# ============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================


def load_all_data(path: str = None) -> pd.DataFrame:
    """Charge toutes les données historiques"""

    if path is None:
        path = CONFIG["data_path"]

    logger.info(f"📂 Chargement de {path}...")

    # Essayer plusieurs fichiers
    paths_to_try = [
        path,
        "data/ml_features_COMPLETE_2020_2025.csv",
        "data/ml_features_SAFE.csv",
        "data/ml_features_fast.csv",
    ]

    df = None
    for p in paths_to_try:
        try:
            df = pd.read_csv(p)
            logger.info(f"✅ Chargé: {p} ({len(df):,} lignes, {len(df.columns)} colonnes)")
            break
        except Exception:
            continue

    if df is None:
        # Fallback: combiner les données SAFE
        logger.info("   Fallback: combinaison des datasets SAFE...")
        train_df = pd.read_csv("data/train_SAFE.csv")
        val_df = pd.read_csv("data/val_SAFE.csv")
        test_df = pd.read_csv("data/test_SAFE.csv")
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        logger.info(f"✅ Combiné: {len(df):,} lignes")

    return df


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], RobustScaler]:
    """Prépare les features et la target"""

    logger.info("🔄 Préparation des données...")

    # Colonnes à exclure
    target_cols = ["position_arrivee", "victoire", "place", "is_winner", "won"]
    id_cols = [
        "id_performance",
        "id_course",
        "nom_norm",
        "date",
        "hippodrome",
        "course_id",
        "race_id",
        "cheval_id",
        "runner_id",
        "horse_name",
    ]

    # Identifier la colonne target
    target_col = None
    for col in ["victoire", "is_winner", "won"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        # Créer target depuis position
        if "position_arrivee" in df.columns:
            df["victoire"] = (df["position_arrivee"] == 1).astype(int)
            target_col = "victoire"
        else:
            raise ValueError("Pas de colonne target trouvée!")

    logger.info(f"   Target: {target_col}")

    # Sélectionner les features
    exclude = target_cols + id_cols
    feature_cols = [col for col in df.columns if col not in exclude]

    X = df[feature_cols].copy()
    y = df[target_col].values

    # Encoder les colonnes catégorielles
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Remplacer NaN par 0
    X = X.fillna(0)

    # Normalisation
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Échantillons: {len(X):,}")
    logger.info(f"   Victoires: {sum(y):,} ({sum(y)/len(y)*100:.2f}%)")

    return X_scaled, y, feature_cols, scaler


def split_temporal(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """Crée des splits temporels pour la validation croisée"""

    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X))


# ============================================================================
# MÉTRIQUE PERSONNALISÉE: ROI
# ============================================================================


def calculate_roi(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.15, simulated_odds: float = 5.0
) -> float:
    """
    Calcule un ROI simulé basé sur les prédictions.

    Args:
        y_true: Labels réels (0/1)
        y_prob: Probabilités prédites
        threshold: Seuil de pari (proba > threshold = pari)
        simulated_odds: Cote moyenne simulée

    Returns:
        ROI en pourcentage
    """
    # Sélectionner les paris (proba > threshold)
    bet_mask = y_prob >= threshold

    if bet_mask.sum() == 0:
        return -100.0  # Pas de paris = très mauvais

    # Gains
    wins = y_true[bet_mask].sum()
    total_bets = bet_mask.sum()

    # ROI = (Gains - Mises) / Mises
    total_stake = total_bets * 1.0  # 1€ par pari
    total_returns = wins * simulated_odds
    roi = (total_returns - total_stake) / total_stake * 100

    return roi


def combined_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Score combiné: AUC + ROI + calibration (Brier)

    Optimise pour:
    - Bon classement (AUC)
    - Profitabilité (ROI)
    - Calibration (Brier bas = bon)
    """
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    roi = calculate_roi(y_true, y_prob, threshold=0.15)

    # Normaliser le ROI entre 0 et 1
    roi_normalized = max(0, min(1, (roi + 50) / 100))  # -50% à +50% → 0 à 1

    # Score combiné (pondéré)
    # AUC: 40%, ROI: 40%, Calibration: 20%
    score = 0.4 * auc + 0.4 * roi_normalized + 0.2 * (1 - brier)

    return score


# ============================================================================
# OPTIMISATION OPTUNA
# ============================================================================


def create_objective(X: np.ndarray, y: np.ndarray, splits: list):
    """Crée la fonction objectif pour Optuna"""

    def objective(trial: optuna.Trial) -> float:
        # Hyperparamètres à optimiser
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": CONFIG["random_state"],
            "verbosity": 0,
            # Structure de l'arbre
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            # Apprentissage
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            # Régularisation
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            # Subsampling
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        }

        # Pondération des classes (équilibrage)
        if CONFIG["boosted_scale_pos_weight"]:
            # Calculer le ratio négatifs/positifs
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            base_weight = neg_count / pos_count

            # Ajuster avec un facteur
            weight_factor = trial.suggest_float("scale_pos_weight_factor", 0.3, 1.5)
            params["scale_pos_weight"] = base_weight * weight_factor

        # Validation croisée temporelle
        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Entraînement
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=params["n_estimators"],
                evals=[(dval, "val")],
                early_stopping_rounds=CONFIG["early_stopping"],
                verbose_eval=False,
            )

            # Prédictions
            y_prob = model.predict(dval)

            # Score combiné
            fold_score = combined_score(y_val, y_prob)
            scores.append(fold_score)

        return np.mean(scores)

    return objective


def optimize_hyperparameters(
    X: np.ndarray, y: np.ndarray, n_trials: int = 100, timeout: int = 7200
) -> tuple[dict, optuna.Study]:
    """Lance l'optimisation Optuna"""

    logger.info(f"🔍 OPTIMISATION OPTUNA ({n_trials} essais, timeout {timeout}s)")
    logger.info("=" * 60)

    # Créer les splits temporels
    splits = split_temporal(X, y, CONFIG["n_splits"])
    logger.info(f"   {CONFIG['n_splits']} splits temporels créés")

    # Créer l'étude Optuna
    sampler = TPESampler(seed=CONFIG["random_state"])
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name="xgboost_safe_optimization"
    )

    # Lancer l'optimisation
    objective = create_objective(X, y, splits)

    start_time = time.time()

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        callbacks=[
            lambda study, trial: logger.info(f"   Trial {trial.number}: Score={trial.value:.4f}")
            if trial.number % 10 == 0
            else None
        ],
    )

    elapsed = time.time() - start_time

    logger.info(f"\n⏱️  Optimisation terminée en {elapsed/60:.1f} minutes")
    logger.info(f"🏆 Meilleur score: {study.best_value:.4f}")
    logger.info("📊 Meilleurs hyperparamètres:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            logger.info(f"   • {k}: {v:.4f}")
        else:
            logger.info(f"   • {k}: {v}")

    return study.best_params, study


# ============================================================================
# ENTRAÎNEMENT FINAL
# ============================================================================


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: dict,
) -> xgb.Booster:
    """Entraîne le modèle final avec les meilleurs hyperparamètres"""

    logger.info("🚀 ENTRAÎNEMENT DU MODÈLE FINAL")
    logger.info("=" * 60)

    # Construire les paramètres complets
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": CONFIG["random_state"],
        "verbosity": 0,
    }
    params.update(best_params)

    # Calculer scale_pos_weight si nécessaire
    if "scale_pos_weight_factor" in params:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        base_weight = neg_count / pos_count
        params["scale_pos_weight"] = base_weight * params.pop("scale_pos_weight_factor")

    n_estimators = params.pop("n_estimators", 1000)

    # Entraînement
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    start_time = time.time()

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    train_time = time.time() - start_time

    logger.info(f"⏱️  Temps d'entraînement: {train_time:.1f}s")
    logger.info(f"🌳 Nombre d'arbres: {model.best_iteration}")

    # Évaluation
    y_prob_train = model.predict(dtrain)
    y_prob_val = model.predict(dval)

    auc_train = roc_auc_score(y_train, y_prob_train)
    auc_val = roc_auc_score(y_val, y_prob_val)
    brier_val = brier_score_loss(y_val, y_prob_val)
    roi_val = calculate_roi(y_val, y_prob_val)

    logger.info("📊 Métriques:")
    logger.info(f"   🎯 AUC Train: {auc_train:.4f}")
    logger.info(f"   🎯 AUC Val: {auc_val:.4f}")
    logger.info(f"   📉 Brier Score: {brier_val:.4f}")
    logger.info(f"   💰 ROI simulé: {roi_val:+.1f}%")

    return model


# ============================================================================
# SAUVEGARDE
# ============================================================================


def save_optimized_model(
    model: xgb.Booster,
    scaler: RobustScaler,
    feature_names: list[str],
    best_params: dict,
    study: optuna.Study,
    output_dir: str = None,
):
    """Sauvegarde le modèle optimisé et tous les artefacts"""

    if output_dir is None:
        output_dir = CONFIG["output_dir"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 SAUVEGARDE: {output_path}")

    # 1. Modèle
    joblib.dump(model, output_path / "xgboost_model.pkl")
    logger.info("   ✅ Modèle")

    # 2. Scaler
    joblib.dump(scaler, output_path / "feature_scaler.pkl")
    logger.info("   ✅ Scaler")

    # 3. Features
    with open(output_path / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("   ✅ Features")

    # 4. Hyperparamètres optimaux
    with open(output_path / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    logger.info("   ✅ Hyperparamètres")

    # 5. Metadata
    metadata = {
        "model_type": "xgboost_optimized",
        "version": "optimized_v1.0",
        "created_at": datetime.now().isoformat(),
        "best_params": best_params,
        "n_trials": len(study.trials),
        "best_score": study.best_value,
        "n_features": len(feature_names),
        "description": "Modèle optimisé par Optuna sur 5 ans de données",
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("   ✅ Metadata")

    # 6. Historique Optuna
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_path / "optuna_trials.csv", index=False)
    logger.info("   ✅ Historique Optuna")

    # 7. Copier l'imputer
    champion_imputer = Path("data/models/champion/feature_imputer.pkl")
    if champion_imputer.exists():
        import shutil

        shutil.copy(champion_imputer, output_path / "feature_imputer.pkl")
        logger.info("   ✅ Imputer")

    return output_path


# ============================================================================
# MAIN
# ============================================================================


def main(n_trials: int = None, timeout: int = None):
    """Pipeline principal d'optimisation"""

    if n_trials is None:
        n_trials = CONFIG["n_trials"]
    if timeout is None:
        timeout = CONFIG["timeout"]

    logger.info("🎯 OPTIMISATION COMPLÈTE DU MODÈLE SAFE")
    logger.info("=" * 70)
    logger.info(f"🕐 Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Essais: {n_trials}")
    logger.info(f"   Timeout: {timeout/60:.0f} min")

    try:
        # 1. Charger les données
        df = load_all_data()

        # 2. Préparer les données
        X, y, feature_names, scaler = prepare_data(df)

        # 3. Split train/val pour l'évaluation finale
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info("\n📊 Split final:")
        logger.info(f"   Train: {len(X_train):,}")
        logger.info(f"   Val: {len(X_val):,}")

        # 4. Optimisation Optuna
        best_params, study = optimize_hyperparameters(
            X_train, y_train, n_trials=n_trials, timeout=timeout
        )

        # 5. Entraînement final
        model = train_final_model(X_train, y_train, X_val, y_val, best_params)

        # 6. Sauvegarde
        output_path = save_optimized_model(model, scaler, feature_names, best_params, study)

        # 7. Résumé
        logger.info("\n" + "=" * 70)
        logger.info("📋 RÉSUMÉ - MODÈLE OPTIMISÉ")
        logger.info("=" * 70)
        logger.info(f"🏆 Meilleur score combiné: {study.best_value:.4f}")
        logger.info(f"📊 Essais réalisés: {len(study.trials)}")
        logger.info(f"📁 Modèle sauvegardé: {output_path}")

        logger.info("\n🔧 Hyperparamètres optimaux:")
        for k, v in best_params.items():
            if isinstance(v, float):
                logger.info(f"   • {k}: {v:.4f}")
            else:
                logger.info(f"   • {k}: {v}")

        return model, best_params, study

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimisation complète du modèle avec Optuna")
    parser.add_argument(
        "--n-trials", "-n", type=int, default=100, help="Nombre d'essais Optuna (défaut: 100)"
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=7200, help="Timeout en secondes (défaut: 7200 = 2h)"
    )
    args = parser.parse_args()

    main(n_trials=args.n_trials, timeout=args.timeout)
