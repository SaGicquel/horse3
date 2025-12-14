#!/usr/bin/env python3
"""
================================================================================
BASELINE RANDOM FOREST - PHASE 4
================================================================================

Description : Modèle baseline RandomForest pour prédire les victoires

Objectifs :
  1. Établir une baseline de performance
  2. Identifier les features les plus importantes
  3. Valider le pipeline ML complet
  4. Comparer avec modèles avancés (XGBoost, LSTM)

Métriques :
  • Accuracy : % prédictions correctes
  • ROC-AUC : qualité discrimination (0.5=random, 1.0=parfait)
  • Precision/Recall/F1 : performance par classe
  • Feature importances : top 20 predicteurs

Target attendue :
  • Baseline accuracy : 70-75% (classe majoritaire ~90% non-victoires)
  • ROC-AUC : 0.65-0.70 (meilleur que random 0.5)
  • Top features : forme_5c, cote_turfbzh, elo_cheval, prediction_ia_gagnant

Usage :
  python train_baseline_rf.py --input-dir data/normalized

================================================================================
"""

import argparse
import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve
)

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BaselineModel:
    """Modèle baseline RandomForest"""
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced',  # Handle imbalanced classes
            verbose=1
        )
        self.feature_names = []
        self.metrics = {}
    
    def load_data(self, input_dir: Path):
        """Charge train/val/test normalisés"""
        logger.info("📂 Chargement des données...")
        
        # Lire Parquet (plus rapide que CSV)
        train_path = input_dir / 'train.parquet'
        val_path = input_dir / 'val.parquet'
        
        if not train_path.exists():
            train_path = input_dir / 'train.csv'
            val_path = input_dir / 'val.csv'
        
        df_train = pd.read_parquet(train_path) if train_path.suffix == '.parquet' else pd.read_csv(train_path)
        df_val = pd.read_parquet(val_path) if val_path.suffix == '.parquet' else pd.read_csv(val_path)
        
        logger.info(f"   ✅ Train: {len(df_train):,} lignes")
        logger.info(f"   ✅ Val: {len(df_val):,} lignes")
        logger.info("")
        
        # Séparer features et target
        exclude_cols = ['victoire', 'place', 'position_arrivee', 'date_course', 
                       'id_performance', 'id_course', 'id_cheval', 'id_jockey', 
                       'id_entraineur', 'id_hippodrome', 'nom_hippodrome', 
                       'hippodrome_ville', 'musique', 'non_partant', 'ecart',
                       'sexe_cheval', 'meteo', 'etat_piste', 'type_piste',
                       'discipline', 'an_naissance', 'hippodrome_top20']
        
        # Sélectionner UNIQUEMENT les colonnes numériques (pas les strings)
        feature_cols = [c for c in df_train.columns 
                       if c not in exclude_cols 
                       and df_train[c].dtype in ['int64', 'float64', 'bool']]
        self.feature_names = feature_cols
        
        X_train = df_train[feature_cols].values
        y_train = df_train['victoire'].values
        
        X_val = df_val[feature_cols].values
        y_val = df_val['victoire'].values
        
        logger.info(f"📊 Features : {len(feature_cols)}")
        logger.info(f"🎯 Target distribution train : {y_train.sum():,} victoires ({100*y_train.mean():.2f}%)")
        logger.info(f"🎯 Target distribution val : {y_val.sum():,} victoires ({100*y_val.mean():.2f}%)")
        logger.info("")
        
        return X_train, y_train, X_val, y_val
    
    def train(self, X_train, y_train):
        """Entraîne le modèle"""
        logger.info("=" * 80)
        logger.info("🚀 ENTRAÎNEMENT RANDOM FOREST")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"   Estimators : {self.model.n_estimators}")
        logger.info(f"   Max depth  : {self.model.max_depth}")
        logger.info(f"   Class weight : {self.model.class_weight}")
        logger.info("")
        
        self.model.fit(X_train, y_train)
        logger.info("   ✅ Entraînement terminé")
        logger.info("")
    
    def evaluate(self, X_val, y_val):
        """Évalue sur validation set"""
        logger.info("=" * 80)
        logger.info("📊 ÉVALUATION SUR VALIDATION")
        logger.info("=" * 80)
        logger.info("")
        
        # Prédictions
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Métriques
        accuracy = accuracy_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba)
        
        logger.info(f"✅ Accuracy : {accuracy:.4f} ({100*accuracy:.2f}%)")
        logger.info(f"✅ ROC-AUC  : {roc_auc:.4f}")
        logger.info("")
        
        # Classification report
        logger.info("📋 CLASSIFICATION REPORT:")
        logger.info("")
        print(classification_report(y_val, y_pred, target_names=['Non-victoire', 'Victoire']))
        logger.info("")
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        logger.info("🔢 CONFUSION MATRIX:")
        logger.info(f"   TN: {cm[0,0]:5,}  FP: {cm[0,1]:5,}")
        logger.info(f"   FN: {cm[1,0]:5,}  TP: {cm[1,1]:5,}")
        logger.info("")
        
        # Stocker métriques
        self.metrics = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'n_samples_val': int(len(y_val)),
            'n_victoires_val': int(y_val.sum())
        }
        
        return y_pred, y_proba
    
    def feature_importance(self, top_n=20):
        """Analyse feature importance"""
        logger.info("=" * 80)
        logger.info(f"📊 TOP {top_n} FEATURES IMPORTANTES")
        logger.info("=" * 80)
        logger.info("")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        logger.info(f"{'Rang':<5} {'Feature':<40} {'Importance':>12}")
        logger.info("-" * 80)
        
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            logger.info(f"{i+1:<5} {self.feature_names[idx]:<40} {importances[idx]:>12.6f}")
        
        logger.info("")
        
        # Stocker
        self.metrics['feature_importances'] = {
            'names': [self.feature_names[i] for i in indices[:top_n]],
            'values': importances[indices[:top_n]].tolist()
        }
    
    def save_model(self, output_dir: Path):
        """Sauvegarde modèle et métriques"""
        logger.info("💾 Sauvegarde...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modèle
        model_path = output_dir / 'baseline_rf.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"   ✅ {model_path}")
        
        # Métriques
        self.metrics['trained_at'] = datetime.now().isoformat()
        self.metrics['model_type'] = 'RandomForestClassifier'
        self.metrics['n_estimators'] = self.model.n_estimators
        self.metrics['max_depth'] = self.model.max_depth
        self.metrics['n_features'] = len(self.feature_names)
        
        metrics_path = output_dir / 'baseline_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"   ✅ {metrics_path}")
        logger.info("")


def main():
    parser = argparse.ArgumentParser(description='Train baseline RandomForest')
    parser.add_argument('--input-dir', default='data/normalized',
                       help='Répertoire avec données normalisées')
    parser.add_argument('--output-dir', default='data/models',
                       help='Répertoire de sortie')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Nombre d\'arbres')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Profondeur max')
    
    args = parser.parse_args()
    
    try:
        # Initialiser
        model = BaselineModel(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )
        
        # Charger
        X_train, y_train, X_val, y_val = model.load_data(Path(args.input_dir))
        
        # Entraîner
        model.train(X_train, y_train)
        
        # Évaluer
        model.evaluate(X_val, y_val)
        
        # Feature importance
        model.feature_importance(top_n=20)
        
        # Sauvegarder
        model.save_model(Path(args.output_dir))
        
        logger.info("=" * 80)
        logger.info("✅ BASELINE RANDOM FOREST TERMINÉ")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"📁 Modèle : {Path(args.output_dir) / 'baseline_rf.pkl'}")
        logger.info(f"📊 Métriques : {Path(args.output_dir) / 'baseline_metrics.json'}")
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        raise


if __name__ == '__main__':
    main()
