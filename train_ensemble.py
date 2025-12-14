#!/usr/bin/env python3
"""
train_ensemble.py - Ensemble Learning (Stacking) pour prédiction victoires PMU

Phase 6 : Optimisation & Ensemble
Combine RandomForest + XGBoost + LightGBM avec méta-modèle LogisticRegression.

Usage:
    python train_ensemble.py --input-dir data/normalized --optimization-dir data/optimization --output-dir data/models

Auteur: Phase 6 ML Pipeline
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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CHARGEMENT DONNÉES
# ============================================================================

def load_data(input_dir: Path):
    """Charge les datasets train/val/test."""
    logger.info("📂 Chargement des données...")
    
    df_train = pd.read_csv(input_dir / "train.csv")
    df_val = pd.read_csv(input_dir / "val.csv")
    df_test = pd.read_csv(input_dir / "test.csv")
    
    logger.info(f"   ✅ Train: {len(df_train):,} lignes")
    logger.info(f"   ✅ Val:   {len(df_val):,} lignes")
    logger.info(f"   ✅ Test:  {len(df_test):,} lignes")
    
    return df_train, df_val, df_test


def prepare_features(df: pd.DataFrame, feature_names: list = None):
    """Prépare X et y en excluant metadata."""
    exclude_patterns = [
        'date_course', 'hippodrome', 'numero_course', 'nom_cheval',
        'nom_jockey', 'nom_entraineur', 'proprietaire', 
        'victoire', 'place', 'ecart', 'id_', 'sexe_cheval',
        'position_arrivee'
    ]
    
    if feature_names is None:
        feature_cols = [
            col for col in df.columns 
            if not any(pattern in col.lower() for pattern in exclude_patterns)
            and df[col].dtype in ['int64', 'float64', 'bool']
        ]
    else:
        feature_cols = feature_names
    
    X = df[feature_cols].values
    y = df['victoire'].values
    
    return X, y, feature_cols


# ============================================================================
# MODÈLES
# ============================================================================

def load_best_params(optimization_dir: Path):
    """Charge les meilleurs hyperparamètres optimisés."""
    logger.info("📋 Chargement hyperparamètres optimisés...")
    
    # XGBoost
    xgb_path = optimization_dir / "best_params_xgboost.json"
    if xgb_path.exists():
        with open(xgb_path, 'r') as f:
            xgb_results = json.load(f)
            xgb_params = xgb_results['best_params']
        logger.info(f"   ✅ XGBoost: ROC-AUC CV = {xgb_results['best_score']:.4f}")
    else:
        xgb_params = None
        logger.warning("   ⚠️  XGBoost params non trouvés, utilisation valeurs par défaut")
    
    # LightGBM
    lgb_path = optimization_dir / "best_params_lightgbm.json"
    if lgb_path.exists():
        with open(lgb_path, 'r') as f:
            lgb_results = json.load(f)
            lgb_params = lgb_results['best_params']
        logger.info(f"   ✅ LightGBM: ROC-AUC CV = {lgb_results['best_score']:.4f}")
    else:
        lgb_params = None
        logger.warning("   ⚠️  LightGBM params non trouvés, utilisation valeurs par défaut")
    
    return xgb_params, lgb_params


def create_base_models(X_train, y_train, xgb_params=None, lgb_params=None):
    """Crée les modèles de base pour l'ensemble."""
    logger.info("")
    logger.info("🤖 Création des modèles de base...")
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    # RandomForest (baseline Phase 4)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    logger.info("   ✅ RandomForest configuré")
    
    # XGBoost (optimisé)
    if xgb_params:
        xgb_params_full = xgb_params.copy()
        xgb_params_full['scale_pos_weight'] = scale_pos_weight
        xgb_params_full['random_state'] = 42
        xgb_params_full['n_jobs'] = -1
        xgb_params_full['verbosity'] = 0
        # IMPORTANT: Pas d'early_stopping pour Stacking (pas d'eval_set fourni)
        xgb_params_full.pop('early_stopping_rounds', None)
    else:
        xgb_params_full = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    xgb_model = xgb.XGBClassifier(**xgb_params_full)
    logger.info("   ✅ XGBoost configuré")
    
    # LightGBM (optimisé)
    if lgb_params:
        lgb_params_full = lgb_params.copy()
        lgb_params_full['scale_pos_weight'] = scale_pos_weight
        lgb_params_full['random_state'] = 42
        lgb_params_full['n_jobs'] = -1
        lgb_params_full['verbose'] = -1
    else:
        lgb_params_full = {
            'n_estimators': 500,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    lgb_model = lgb.LGBMClassifier(**lgb_params_full)
    logger.info("   ✅ LightGBM configuré")
    
    return [
        ('rf', rf),
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ]


# ============================================================================
# ENTRAÎNEMENT ENSEMBLE
# ============================================================================

def train_stacking(X_train, y_train, X_val, y_val, base_models):
    """Entraîne un Stacking Classifier."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏗️  ENTRAÎNEMENT STACKING CLASSIFIER")
    logger.info("=" * 80)
    
    # Méta-modèle: LogisticRegression
    meta_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        n_jobs=-1
    )
    
    # Créer Stacking
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3,  # 3-fold interne pour générer méta-features
        n_jobs=-1
    )
    
    logger.info("⏳ Entraînement en cours...")
    start_time = time.time()
    
    stacking.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Entraînement terminé en {elapsed:.1f}s")
    
    return stacking


def train_voting(base_models):
    """Crée un Voting Classifier (soft voting)."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("🗳️  CRÉATION VOTING CLASSIFIER")
    logger.info("=" * 80)
    
    voting = VotingClassifier(
        estimators=base_models,
        voting='soft',  # Moyenne des probabilités
        n_jobs=-1
    )
    
    logger.info("✅ Voting Classifier configuré")
    
    return voting


# ============================================================================
# ÉVALUATION
# ============================================================================

def evaluate_model(model, X, y, model_name, dataset_name="Test"):
    """Évalue un modèle."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    logger.info("")
    logger.info(f"📊 {model_name} - {dataset_name}")
    logger.info("-" * 80)
    logger.info(f"   ROC-AUC  : {metrics['roc_auc']:.4f} ⭐")
    logger.info(f"   Accuracy : {metrics['accuracy']:.4f} ({100*metrics['accuracy']:.2f}%)")
    logger.info(f"   Precision: {metrics['precision']:.4f}")
    logger.info(f"   Recall   : {metrics['recall']:.4f}")
    logger.info(f"   F1-Score : {metrics['f1']:.4f}")
    logger.info(f"   CM: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Learning (Stacking + Voting) pour prédiction victoires"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/normalized',
        help='Répertoire données normalisées'
    )
    parser.add_argument(
        '--optimization-dir',
        type=str,
        default='data/optimization',
        help='Répertoire params optimisés'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models',
        help='Répertoire sauvegarde modèles'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    optimization_dir = Path(args.optimization_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎯 ENSEMBLE LEARNING - PHASE 6")
    logger.info("=" * 80)
    logger.info(f"   📂 Input:        {input_dir}")
    logger.info(f"   📂 Optimization: {optimization_dir}")
    logger.info(f"   📂 Output:       {output_dir}")
    logger.info("")
    
    try:
        # 1. Charger données
        df_train, df_val, df_test = load_data(input_dir)
        X_train, y_train, feature_names = prepare_features(df_train)
        X_val, y_val, _ = prepare_features(df_val, feature_names)
        X_test, y_test, _ = prepare_features(df_test, feature_names)
        
        # 2. Charger params optimisés
        xgb_params, lgb_params = load_best_params(optimization_dir)
        
        # 3. Créer modèles de base
        base_models = create_base_models(X_train, y_train, xgb_params, lgb_params)
        
        # 4. Entraîner Stacking
        stacking = train_stacking(X_train, y_train, X_val, y_val, base_models)
        
        # 5. Évaluer Stacking
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 ÉVALUATION STACKING")
        logger.info("=" * 80)
        
        metrics_stacking_val = evaluate_model(stacking, X_val, y_val, "Stacking", "Validation")
        metrics_stacking_test = evaluate_model(stacking, X_test, y_test, "Stacking", "Test")
        
        # 6. Entraîner et évaluer Voting
        logger.info("")
        logger.info("=" * 80)
        logger.info("🗳️  ENTRAÎNEMENT VOTING CLASSIFIER")
        logger.info("=" * 80)
        
        voting = train_voting(base_models)
        logger.info("⏳ Entraînement en cours...")
        start_time = time.time()
        voting.fit(X_train, y_train)
        elapsed = time.time() - start_time
        logger.info(f"✅ Entraînement terminé en {elapsed:.1f}s")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 ÉVALUATION VOTING")
        logger.info("=" * 80)
        
        metrics_voting_val = evaluate_model(voting, X_val, y_val, "Voting", "Validation")
        metrics_voting_test = evaluate_model(voting, X_test, y_test, "Voting", "Test")
        
        # 7. Sauvegarder modèles
        logger.info("")
        logger.info("💾 Sauvegarde des modèles...")
        
        with open(output_dir / "ensemble_stacking.pkl", 'wb') as f:
            pickle.dump(stacking, f)
        logger.info(f"   ✅ Stacking: {output_dir / 'ensemble_stacking.pkl'}")
        
        with open(output_dir / "ensemble_voting.pkl", 'wb') as f:
            pickle.dump(voting, f)
        logger.info(f"   ✅ Voting: {output_dir / 'ensemble_voting.pkl'}")
        
        # 8. Sauvegarder métriques
        results = {
            'stacking': {
                'validation': metrics_stacking_val,
                'test': metrics_stacking_test
            },
            'voting': {
                'validation': metrics_voting_val,
                'test': metrics_voting_test
            }
        }
        
        with open(output_dir / "ensemble_metrics.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"   ✅ Métriques: {output_dir / 'ensemble_metrics.json'}")
        
        # 9. Résumé final
        logger.info("")
        logger.info("=" * 80)
        logger.info("🏆 RÉSUMÉ FINAL - ENSEMBLES")
        logger.info("=" * 80)
        logger.info(f"   Stacking Val:  ROC-AUC = {metrics_stacking_val['roc_auc']:.4f}")
        logger.info(f"   Stacking Test: ROC-AUC = {metrics_stacking_test['roc_auc']:.4f}")
        logger.info(f"   Voting Val:    ROC-AUC = {metrics_voting_val['roc_auc']:.4f}")
        logger.info(f"   Voting Test:   ROC-AUC = {metrics_voting_test['roc_auc']:.4f}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
