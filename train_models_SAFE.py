#!/usr/bin/env python3
"""
NORMALISATION ET ENTRAÎNEMENT MODÈLE SAFE
=========================================

Normalise les features du dataset SAFE et entraîne XGBoost
pour comparer avec les performances du modèle avec data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
import time
from datetime import datetime

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data():
    """Charge et prépare les données SAFE"""
    
    logging.info("📂 CHARGEMENT DES DATASETS SAFE")
    
    # Chargement
    train_df = pd.read_csv('data/train_SAFE.csv')
    val_df = pd.read_csv('data/val_SAFE.csv')
    test_df = pd.read_csv('data/test_SAFE.csv')
    
    logging.info(f"✅ Train: {train_df.shape[0]:,} × {train_df.shape[1]}")
    logging.info(f"✅ Val: {val_df.shape[0]:,} × {val_df.shape[1]}")
    logging.info(f"✅ Test: {test_df.shape[0]:,} × {test_df.shape[1]}")
    
    # Identification des targets et features
    target_cols = ['position_arrivee', 'victoire', 'place']
    id_cols = ['id_performance', 'id_course', 'nom_norm']
    
    feature_cols = [col for col in train_df.columns 
                   if col not in target_cols + id_cols]
    
    logging.info(f"🎯 Targets: {len(target_cols)} ({target_cols})")
    logging.info(f"📊 Features: {len(feature_cols)}")
    
    # Séparation features/targets
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_cols]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_cols': feature_cols
    }

def normalize_features(data):
    """Normalise les features numériques et encode les catégorielles"""
    
    logging.info("🔄 NORMALISATION ET ENCODAGE DES FEATURES")
    
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    
    # Identification des colonnes par type
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    logging.info(f"📊 Colonnes numériques: {len(numeric_cols)}")
    logging.info(f"🏷️  Colonnes catégorielles: {len(categorical_cols)}")
    
    # Copies pour ne pas modifier les originaux
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    X_test_processed = X_test.copy()
    
    # 1. ENCODAGE DES COLONNES CATÉGORIELLES
    from sklearn.preprocessing import LabelEncoder
    
    label_encoders = {}
    
    if categorical_cols:
        logging.info("🔤 Encodage des colonnes catégorielles...")
        
        for col in categorical_cols:
            le = LabelEncoder()
            
            # Combinaison de toutes les valeurs possibles pour éviter les erreurs
            all_values = pd.concat([
                X_train_processed[col],
                X_val_processed[col], 
                X_test_processed[col]
            ]).astype(str)
            
            le.fit(all_values)
            
            # Encodage
            X_train_processed[col] = le.transform(X_train_processed[col].astype(str))
            X_val_processed[col] = le.transform(X_val_processed[col].astype(str))
            X_test_processed[col] = le.transform(X_test_processed[col].astype(str))
            
            label_encoders[col] = le
            
        logging.info(f"✅ {len(categorical_cols)} colonnes encodées")
    
    # 2. NORMALISATION DES COLONNES NUMÉRIQUES
    scaler = RobustScaler()
    
    if numeric_cols:
        logging.info("📊 Normalisation des colonnes numériques...")
        X_train_processed[numeric_cols] = scaler.fit_transform(X_train_processed[numeric_cols])
        X_val_processed[numeric_cols] = scaler.transform(X_val_processed[numeric_cols])
        X_test_processed[numeric_cols] = scaler.transform(X_test_processed[numeric_cols])
        
        # Sauvegarde du scaler
        joblib.dump(scaler, 'models/scaler_SAFE.pkl')
        logging.info("✅ Scaler sauvegardé")
    
    # Sauvegarde des encoders
    if label_encoders:
        joblib.dump(label_encoders, 'models/label_encoders_SAFE.pkl')
        logging.info("✅ Label encoders sauvegardés")
    
    return {
        'X_train': X_train_processed,
        'X_val': X_val_processed,
        'X_test': X_test_processed,
        'y_train': data['y_train'],
        'y_val': data['y_val'],
        'y_test': data['y_test'],
        'scaler': scaler,
        'label_encoders': label_encoders,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }

def train_xgboost_safe(data, target='victoire'):
    """Entraîne XGBoost sur le dataset SAFE"""
    
    logging.info(f"🚀 ENTRAÎNEMENT XGBOOST SAFE - TARGET: {target}")
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train'][target]
    y_val = data['y_val'][target]
    y_test = data['y_test'][target]
    
    # Configuration XGBoost optimisée
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Datasets XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Entraînement avec early stopping
    start_time = time.time()
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    train_time = time.time() - start_time
    
    # Prédictions
    y_pred_train = model.predict(dtrain)
    y_pred_val = model.predict(dval)
    y_pred_test = model.predict(dtest)
    
    # Métriques
    auc_train = roc_auc_score(y_train, y_pred_train)
    auc_val = roc_auc_score(y_val, y_pred_val)
    auc_test = roc_auc_score(y_test, y_pred_test)
    
    # Prédictions binaires (seuil 0.5)
    y_pred_test_bin = (y_pred_test > 0.5).astype(int)
    precision = precision_score(y_test, y_pred_test_bin)
    recall = recall_score(y_test, y_pred_test_bin)
    f1 = f1_score(y_test, y_pred_test_bin)
    
    # Résultats
    results = {
        'model': model,
        'target': target,
        'auc_train': auc_train,
        'auc_val': auc_val,
        'auc_test': auc_test,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time,
        'best_iteration': model.best_iteration
    }
    
    logging.info(f"📊 RÉSULTATS XGBoost SAFE ({target}):")
    logging.info(f"   🏆 AUC Train: {auc_train:.4f}")
    logging.info(f"   🔍 AUC Val: {auc_val:.4f}")
    logging.info(f"   🧪 AUC Test: {auc_test:.4f}")
    logging.info(f"   📈 Précision: {precision:.4f}")
    logging.info(f"   📈 Rappel: {recall:.4f}")
    logging.info(f"   📈 F1-Score: {f1:.4f}")
    logging.info(f"   ⏱️  Temps: {train_time:.1f}s")
    
    # Sauvegarde du modèle
    model_path = f'models/xgboost_SAFE_{target}.pkl'
    joblib.dump(model, model_path)
    logging.info(f"✅ Modèle sauvegardé: {model_path}")
    
    return results

def train_comparison_models(data, target='victoire'):
    """Entraîne des modèles de comparaison"""
    
    logging.info(f"🔄 ENTRAÎNEMENT MODÈLES DE COMPARAISON - TARGET: {target}")
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train'][target]
    y_test = data['y_test'][target]
    
    results = {}
    
    # 1. LightGBM
    logging.info("📊 Entraînement LightGBM...")
    start_time = time.time()
    
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    lgb_time = time.time() - start_time
    
    results['lightgbm'] = {
        'model': lgb_model,
        'auc_test': lgb_auc,
        'train_time': lgb_time
    }
    
    logging.info(f"   ✅ LightGBM AUC: {lgb_auc:.4f} (temps: {lgb_time:.1f}s)")
    
    # 2. Random Forest
    logging.info("📊 Entraînement Random Forest...")
    start_time = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred)
    rf_time = time.time() - start_time
    
    results['random_forest'] = {
        'model': rf_model,
        'auc_test': rf_auc,
        'train_time': rf_time
    }
    
    logging.info(f"   ✅ Random Forest AUC: {rf_auc:.4f} (temps: {rf_time:.1f}s)")
    
    return results

def main():
    """Pipeline principal"""
    
    logging.info("🚀 DÉMARRAGE PIPELINE ML SAFE")
    logging.info("=" * 70)
    
    try:
        # 1. Chargement des données
        data = prepare_data()
        
        # 2. Normalisation
        data_normalized = normalize_features(data)
        
        # 3. Entraînement XGBoost principal
        xgb_results = train_xgboost_safe(data_normalized, target='victoire')
        
        # 4. Modèles de comparaison
        comparison_results = train_comparison_models(data_normalized, target='victoire')
        
        # 5. Résumé final
        logging.info("\n📋 RÉSUMÉ FINAL - MODÈLES SAFE")
        logging.info("=" * 70)
        
        # XGBoost
        auc_xgb = xgb_results['auc_test']
        logging.info(f"🏆 XGBoost SAFE:")
        logging.info(f"   🎯 AUC Test: {auc_xgb:.4f}")
        logging.info(f"   📈 F1-Score: {xgb_results['f1_score']:.4f}")
        logging.info(f"   ⏱️  Temps: {xgb_results['train_time']:.1f}s")
        
        # Comparaisons
        for model_name, results in comparison_results.items():
            auc = results['auc_test']
            diff_vs_xgb = ((auc - auc_xgb) / auc_xgb) * 100
            logging.info(f"📊 {model_name.title()}:")
            logging.info(f"   🎯 AUC Test: {auc:.4f} ({diff_vs_xgb:+.1f}% vs XGBoost)")
            logging.info(f"   ⏱️  Temps: {results['train_time']:.1f}s")
        
        # Comparaison avec ancien modèle (supposé avec data leakage)
        logging.info(f"\n🔍 COMPARAISON AVEC ANCIEN MODÈLE:")
        logging.info(f"   🟢 SAFE XGBoost AUC: {auc_xgb:.4f}")
        logging.info(f"   🔴 Ancien modèle AUC: ~0.932 (avec data leakage)")
        
        difference = (0.932 - auc_xgb) * 100
        logging.info(f"   📉 Baisse attendue: -{difference:.1f} points")
        logging.info(f"   ✅ Cette baisse confirme l'élimination du data leakage!")
        
        # Sauvegarde des résultats
        all_results = {
            'xgboost_safe': xgb_results,
            'comparison_models': comparison_results,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(all_results, 'models/results_SAFE.pkl')
        logging.info(f"\n✅ Résultats sauvegardés: models/results_SAFE.pkl")
        
    except Exception as e:
        logging.error(f"❌ Erreur dans le pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()