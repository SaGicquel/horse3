#!/usr/bin/env python3
"""
RÉSUMÉ FINAL PHASE D6 - CALIBRATION
===================================

Analyse finale des résultats de calibration et validation des métriques.
"""

import pandas as pd
import numpy as np
import json
import logging

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_phase_d6():
    """Validation finale de la phase D6"""
    
    logger.info("🎯 VALIDATION FINALE PHASE D6 - CALIBRATION")
    logger.info("=" * 70)
    
    # 1. VÉRIFICATION DU FICHIER DE SORTIE
    logger.info("📂 Vérification du fichier de sortie...")
    
    try:
        df = pd.read_csv('data/backtest_predictions_calibrated.csv')
        logger.info(f"✅ Fichier chargé: {len(df):,} lignes × {df.shape[1]} colonnes")
        
        # Colonnes attendues
        expected_cols = ['race_key', 'id_cheval', 'date_course', 'p_model_win', 
                        'p_model_norm', 'p_calibrated', 'p_final', 'is_win', 'place', 
                        'position_arrivee', 'cote_sp', 'split']
        
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Colonnes manquantes: {missing_cols}")
        else:
            logger.info("✅ Toutes les colonnes attendues présentes")
            
    except Exception as e:
        logger.error(f"❌ Erreur lecture fichier: {e}")
        return False
    
    # 2. VÉRIFICATION DES MÉTRIQUES DE CALIBRATION
    logger.info("\n🎯 Vérification des métriques de calibration...")
    
    try:
        with open('calibration/calibration_report_20251208_163949.json', 'r') as f:
            report = json.load(f)
        
        metrics = report['metrics']['calibration']
        brier = metrics['brier_score']
        ece = metrics['ece']
        
        logger.info(f"📊 Brier Score: {brier:.4f}")
        logger.info(f"📊 ECE (Expected Calibration Error): {ece:.4f}")
        
        # Validation des seuils
        brier_ok = brier <= 0.12  # Seuil acceptable
        ece_ok = ece <= 0.03      # Seuil acceptable
        
        if brier_ok:
            logger.info(f"✅ Brier Score acceptable (≤ 0.12): {brier:.4f}")
        else:
            logger.warning(f"⚠️  Brier Score élevé (> 0.12): {brier:.4f}")
            
        if ece_ok:
            logger.info(f"✅ ECE acceptable (≤ 0.03): {ece:.4f}")
        else:
            logger.warning(f"⚠️  ECE élevé (> 0.03): {ece:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lecture métriques: {e}")
        return False
    
    # 3. VÉRIFICATION NORMALISATION DES PROBABILITÉS
    logger.info("\n🔄 Vérification de la normalisation...")
    
    race_sums = df.groupby('race_key')['p_final'].sum()
    perfect_normalization = (np.abs(race_sums - 1.0) < 1e-3).sum()
    total_races = len(race_sums)
    pct_normalized = (perfect_normalization / total_races) * 100
    
    logger.info(f"📊 Courses parfaitement normalisées: {perfect_normalization:,}/{total_races:,} ({pct_normalized:.1f}%)")
    
    if pct_normalized >= 99.0:
        logger.info("✅ Normalisation excellente (≥ 99%)")
    elif pct_normalized >= 95.0:
        logger.info("✅ Normalisation bonne (≥ 95%)")
    else:
        logger.warning(f"⚠️  Normalisation problématique (< 95%)")
    
    # 4. COMPARAISON AVANT/APRÈS CALIBRATION
    logger.info("\n📈 Comparaison avant/après calibration...")
    
    # Statistiques des probabilités
    for col in ['p_model_win', 'p_model_norm', 'p_calibrated', 'p_final']:
        if col in df.columns:
            stats = df[col].describe()
            logger.info(f"📊 {col}:")
            logger.info(f"   Min: {stats['min']:.6f}, Médiane: {stats['50%']:.4f}, Max: {stats['max']:.4f}")
    
    # Corrélations
    if all(col in df.columns for col in ['p_model_win', 'p_final']):
        corr_orig_final = df['p_model_win'].corr(df['p_final'])
        logger.info(f"🔗 Corrélation p_model_win vs p_final: {corr_orig_final:.4f}")
        
        if corr_orig_final >= 0.95:
            logger.info("✅ Forte corrélation conservée (≥ 0.95)")
        elif corr_orig_final >= 0.90:
            logger.info("✅ Bonne corrélation conservée (≥ 0.90)")
        else:
            logger.warning(f"⚠️  Corrélation faible (< 0.90)")
    
    # 5. VALIDATION TEMPORELLE
    logger.info("\n📅 Validation temporelle...")
    
    df['date_course'] = pd.to_datetime(df['date_course'])
    date_min = df['date_course'].min()
    date_max = df['date_course'].max()
    
    logger.info(f"📊 Période couverte: {date_min.strftime('%Y-%m-%d')} à {date_max.strftime('%Y-%m-%d')}")
    
    # Distribution par split
    split_dist = df['split'].value_counts()
    logger.info(f"📊 Distribution splits:")
    for split_name, count in split_dist.items():
        pct = (count / len(df)) * 100
        logger.info(f"   {split_name}: {count:,} ({pct:.1f}%)")
    
    # 6. ARTEFACTS DE CALIBRATION
    logger.info("\n📦 Vérification des artefacts...")
    
    import os
    artifacts_dir = "calibration"
    expected_artifacts = [
        "calibration_report_20251208_163949.json",
        "scaler_temperature_20251208_163949.pkl",
        "calibrator_isotonic_20251208_163949.pkl",
        "health.json"
    ]
    
    for artifact in expected_artifacts:
        path = os.path.join(artifacts_dir, artifact)
        if os.path.exists(path):
            size = os.path.getsize(path)
            logger.info(f"✅ {artifact}: {size:,} bytes")
        else:
            logger.warning(f"⚠️  {artifact}: manquant")
    
    # 7. SCORE FINAL
    logger.info("\n🏆 SCORE FINAL PHASE D6:")
    logger.info("=" * 50)
    
    score = 0
    max_score = 7
    
    # Critères de validation
    if len(df) >= 650000:
        score += 1
        logger.info("✅ [1/7] Volume de données suffisant")
    else:
        logger.info("❌ [0/7] Volume de données insuffisant")
    
    if not missing_cols:
        score += 1
        logger.info("✅ [1/7] Structure complète")
    else:
        logger.info("❌ [0/7] Structure incomplète")
    
    if brier_ok:
        score += 1
        logger.info("✅ [1/7] Brier Score acceptable")
    else:
        logger.info("❌ [0/7] Brier Score problématique")
    
    if ece_ok:
        score += 1
        logger.info("✅ [1/7] ECE acceptable")
    else:
        logger.info("❌ [0/7] ECE problématique")
    
    if pct_normalized >= 95:
        score += 1
        logger.info("✅ [1/7] Normalisation correcte")
    else:
        logger.info("❌ [0/7] Normalisation problématique")
    
    if 'corr_orig_final' in locals() and corr_orig_final >= 0.90:
        score += 1
        logger.info("✅ [1/7] Corrélation conservée")
    else:
        logger.info("❌ [0/7] Corrélation dégradée")
    
    artifacts_present = sum(1 for artifact in expected_artifacts 
                           if os.path.exists(os.path.join(artifacts_dir, artifact)))
    if artifacts_present >= len(expected_artifacts) - 1:  # Au moins 3/4
        score += 1
        logger.info("✅ [1/7] Artefacts présents")
    else:
        logger.info("❌ [0/7] Artefacts manquants")
    
    logger.info(f"\n🎯 SCORE FINAL: {score}/{max_score}")
    
    if score >= 6:
        logger.info("🟢 PHASE D6 RÉUSSIE - Calibration excellente!")
        status = "✅ SUCCÈS"
    elif score >= 4:
        logger.info("🟡 PHASE D6 PARTIELLEMENT RÉUSSIE - Calibration acceptable")
        status = "⚠️ PARTIEL"
    else:
        logger.info("🔴 PHASE D6 ÉCHOUÉE - Calibration problématique")
        status = "❌ ÉCHEC"
    
    # 8. RÉSUMÉ EXÉCUTIF
    logger.info(f"\n📋 RÉSUMÉ EXÉCUTIF PHASE D6:")
    logger.info("=" * 50)
    logger.info(f"🎯 Status: {status}")
    logger.info(f"📊 Données traitées: {len(df):,} lignes sur 5+ années")
    logger.info(f"🎯 Brier Score: {brier:.4f} {'✅' if brier_ok else '⚠️'}")
    logger.info(f"📈 ECE: {ece:.4f} {'✅' if ece_ok else '⚠️'}")
    logger.info(f"🔄 Normalisation: {pct_normalized:.1f}% courses {'✅' if pct_normalized >= 95 else '⚠️'}")
    logger.info(f"📂 Fichier final: data/backtest_predictions_calibrated.csv")
    logger.info(f"📦 Artefacts: calibration/ (température, calibrateur, blender)")
    
    return score >= 4

if __name__ == "__main__":
    validate_phase_d6()