#!/usr/bin/env python3
"""
VALIDATION DES PRÉDICTIONS PHASE D5
==================================

Valide et analyse les prédictions générées par XGBoost SAFE
sur tout l'historique.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_predictions():
    """Valide et analyse les prédictions générées"""
    
    logger.info("🔍 VALIDATION DES PRÉDICTIONS D5")
    logger.info("=" * 70)
    
    # Chargement des prédictions
    logger.info("📂 Chargement du fichier de prédictions...")
    df = pd.read_csv('data/backtest_predictions.csv')
    
    logger.info(f"✅ Prédictions chargées: {len(df):,} lignes × {df.shape[1]} colonnes")
    
    # 1. VALIDATION DE LA STRUCTURE
    logger.info("\n📋 VALIDATION STRUCTURE:")
    print("-" * 50)
    
    # Colonnes attendues
    expected_cols = ['race_key', 'id_cheval', 'date_course', 'p_model_win', 
                    'is_win', 'place', 'cote_sp', 'split', 'position_arrivee']
    
    missing_cols = [col for col in expected_cols if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_cols + ['cote_pm']]
    
    if missing_cols:
        print(f"❌ Colonnes manquantes: {missing_cols}")
    else:
        print("✅ Toutes les colonnes attendues présentes")
    
    if extra_cols:
        print(f"ℹ️  Colonnes supplémentaires: {extra_cols}")
    
    print(f"📊 Colonnes disponibles: {list(df.columns)}")
    
    # 2. VALIDATION TEMPORELLE
    logger.info("\n📅 VALIDATION TEMPORELLE:")
    print("-" * 50)
    
    df['date_course'] = pd.to_datetime(df['date_course'])
    date_min = df['date_course'].min()
    date_max = df['date_course'].max()
    
    print(f"📅 Période couverte: {date_min.strftime('%Y-%m-%d')} à {date_max.strftime('%Y-%m-%d')}")
    
    # Distribution par année
    df['annee'] = df['date_course'].dt.year
    yearly_dist = df['annee'].value_counts().sort_index()
    
    print(f"\n📊 RÉPARTITION ANNUELLE:")
    total = len(df)
    for year, count in yearly_dist.items():
        pct = (count / total) * 100
        print(f"   {year}: {count:,} ({pct:.1f}%)")
    
    # 3. VALIDATION DES SPLITS
    logger.info("\n🎯 VALIDATION SPLITS:")
    print("-" * 50)
    
    split_dist = df['split'].value_counts()
    
    for split_name, count in split_dist.items():
        pct = (count / total) * 100
        print(f"📊 {split_name.upper()}: {count:,} ({pct:.1f}%)")
    
    # Vérification cohérence temporelle des splits
    split_years = df.groupby('split')['annee'].agg(['min', 'max'])
    print(f"\n🔍 COHÉRENCE TEMPORELLE DES SPLITS:")
    for split_name, row in split_years.iterrows():
        print(f"   {split_name.upper()}: {row['min']}-{row['max']}")
    
    # 4. VALIDATION DES PRÉDICTIONS
    logger.info("\n🎯 VALIDATION PRÉDICTIONS:")
    print("-" * 50)
    
    # Statistiques des probabilités
    pred_stats = df['p_model_win'].describe()
    print(f"📈 STATISTIQUES p_model_win:")
    print(f"   Min: {pred_stats['min']:.6f}")
    print(f"   25%: {pred_stats['25%']:.4f}")
    print(f"   Médiane: {pred_stats['50%']:.4f}")
    print(f"   75%: {pred_stats['75%']:.4f}")
    print(f"   Max: {pred_stats['max']:.6f}")
    print(f"   Moyenne: {pred_stats['mean']:.4f}")
    print(f"   Écart-type: {pred_stats['std']:.4f}")
    
    # Validation des bornes
    valid_probs = (df['p_model_win'] >= 0) & (df['p_model_win'] <= 1)
    invalid_count = (~valid_probs).sum()
    
    if invalid_count > 0:
        print(f"❌ Probabilités invalides (hors [0,1]): {invalid_count}")
    else:
        print("✅ Toutes les probabilités dans [0,1]")
    
    # Valeurs nulles
    null_preds = df['p_model_win'].isnull().sum()
    if null_preds > 0:
        print(f"⚠️  Prédictions nulles: {null_preds}")
    else:
        print("✅ Aucune prédiction nulle")
    
    # 5. VALIDATION DES TARGETS
    logger.info("\n🏆 VALIDATION TARGETS:")
    print("-" * 50)
    
    # Victoires
    total_wins = df['is_win'].sum()
    win_rate = (total_wins / len(df)) * 100
    print(f"🏆 Victoires totales: {total_wins:,} ({win_rate:.1f}%)")
    
    # Places
    total_places = df['place'].sum()
    place_rate = (total_places / len(df)) * 100
    print(f"🥉 Places totales: {total_places:,} ({place_rate:.1f}%)")
    
    # Cohérence victoire/position
    win_pos_1 = ((df['is_win'] == 1) & (df['position_arrivee'] == 1)).sum()
    total_pos_1 = (df['position_arrivee'] == 1).sum()
    
    print(f"🔍 Victoires en position 1: {win_pos_1:,}")
    print(f"🔍 Total positions 1: {total_pos_1:,}")
    
    if win_pos_1 == total_pos_1:
        print("✅ Cohérence victoire/position validée")
    else:
        print(f"❌ Incohérence victoire/position: {total_pos_1 - win_pos_1} écart")
    
    # 6. VALIDATION DES COTES
    logger.info("\n💰 VALIDATION COTES:")
    print("-" * 50)
    
    cote_stats = df['cote_sp'].describe()
    cotes_disponibles = (~df['cote_sp'].isnull()).sum()
    cote_coverage = (cotes_disponibles / len(df)) * 100
    
    print(f"📊 Cotes disponibles: {cotes_disponibles:,} ({cote_coverage:.1f}%)")
    print(f"📈 Cote min: {cote_stats['min']:.1f}")
    print(f"📈 Cote médiane: {cote_stats['50%']:.1f}")
    print(f"📈 Cote max: {cote_stats['max']:.1f}")
    
    # 7. ANALYSE DE PERFORMANCE PAR SPLIT
    logger.info("\n📊 PERFORMANCE PAR SPLIT:")
    print("-" * 50)
    
    from sklearn.metrics import roc_auc_score
    
    for split_name in ['train', 'val', 'test']:
        split_data = df[df['split'] == split_name]
        if len(split_data) > 0:
            try:
                auc = roc_auc_score(split_data['is_win'], split_data['p_model_win'])
                win_rate = split_data['is_win'].mean() * 100
                avg_prob = split_data['p_model_win'].mean()
                
                print(f"📊 {split_name.upper()}:")
                print(f"   🎯 AUC: {auc:.4f}")
                print(f"   🏆 Taux victoire: {win_rate:.1f}%")
                print(f"   📈 Prob. moyenne: {avg_prob:.4f}")
                
            except Exception as e:
                print(f"❌ Erreur calcul AUC pour {split_name}: {e}")
    
    # 8. ÉCHANTILLONS REPRÉSENTATIFS
    logger.info("\n🔍 ÉCHANTILLONS REPRÉSENTATIFS:")
    print("-" * 50)
    
    # Top prédictions (favoris du modèle)
    top_preds = df.nlargest(5, 'p_model_win')[['race_key', 'id_cheval', 'date_course', 
                                                'p_model_win', 'is_win', 'cote_sp']]
    print("🏆 TOP 5 PRÉDICTIONS:")
    for _, row in top_preds.iterrows():
        result = "✅ GAGNÉ" if row['is_win'] == 1 else "❌ PERDU"
        print(f"   {row['date_course']} - {row['id_cheval']}: {row['p_model_win']:.4f} - Cote {row['cote_sp']:.1f} - {result}")
    
    # Prédictions récentes
    recent = df[df['date_course'] >= '2025-12-01'].head(5)[['race_key', 'id_cheval', 'date_course', 
                                                           'p_model_win', 'is_win', 'cote_sp']]
    if len(recent) > 0:
        print("\n📅 ÉCHANTILLON RÉCENT:")
        for _, row in recent.iterrows():
            result = "✅ GAGNÉ" if row['is_win'] == 1 else "❌ PERDU" 
            print(f"   {row['date_course']} - {row['id_cheval']}: {row['p_model_win']:.4f} - Cote {row['cote_sp']:.1f} - {result}")
    
    # 9. RÉSUMÉ FINAL
    logger.info("\n📝 RÉSUMÉ VALIDATION D5:")
    print("=" * 70)
    
    validation_score = 0
    max_score = 8
    
    # Critères de validation
    if len(df) > 600000:
        validation_score += 1
        print("✅ [1/8] Volume de données suffisant (>600K)")
    else:
        print(f"❌ [0/8] Volume insuffisant ({len(df):,} < 600K)")
    
    if not missing_cols:
        validation_score += 1
        print("✅ [1/8] Structure complète")
    else:
        print("❌ [0/8] Structure incomplète")
    
    if (date_max - date_min).days > 1500:  # ~4+ années
        validation_score += 1
        print("✅ [1/8] Période historique étendue")
    else:
        print("❌ [0/8] Période historique insuffisante")
    
    if len(split_dist) == 3:
        validation_score += 1
        print("✅ [1/8] Splits train/val/test présents")
    else:
        print("❌ [0/8] Splits incomplets")
    
    if invalid_count == 0:
        validation_score += 1
        print("✅ [1/8] Probabilités valides")
    else:
        print("❌ [0/8] Probabilités invalides")
    
    if win_pos_1 == total_pos_1:
        validation_score += 1
        print("✅ [1/8] Cohérence targets")
    else:
        print("❌ [0/8] Incohérence targets")
    
    if 8 <= win_rate <= 12:  # Taux de victoire réaliste hippisme
        validation_score += 1
        print("✅ [1/8] Taux de victoire réaliste")
    else:
        print("❌ [0/8] Taux de victoire anormal")
    
    if cote_coverage > 80:
        validation_score += 1
        print("✅ [1/8] Couverture cotes suffisante")
    else:
        print("❌ [0/8] Couverture cotes insuffisante")
    
    print(f"\n🏆 SCORE VALIDATION: {validation_score}/{max_score}")
    
    if validation_score >= 7:
        print("🟢 PRÉDICTIONS VALIDÉES - Prêtes pour backtesting!")
    elif validation_score >= 5:
        print("🟡 PRÉDICTIONS PARTIELLEMENT VALIDÉES - Vérifications recommandées")
    else:
        print("🔴 PRÉDICTIONS NON VALIDÉES - Corrections nécessaires")
    
    return {
        'total_predictions': len(df),
        'date_range': f"{date_min.strftime('%Y-%m-%d')} à {date_max.strftime('%Y-%m-%d')}",
        'validation_score': validation_score,
        'max_score': max_score,
        'win_rate': win_rate,
        'splits': dict(split_dist)
    }

if __name__ == "__main__":
    validate_predictions()