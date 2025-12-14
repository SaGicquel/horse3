#!/usr/bin/env python3
"""
SPLIT TRAIN/VAL/TEST DU DATASET SAFE
===================================

Division temporelle du dataset SAFE en ensembles d'entraînement,
validation et test pour éviter le data leakage temporel.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_dataset_safe():
    """Divise le dataset SAFE en train/val/test de manière temporelle"""
    
    logging.info("🔄 DÉMARRAGE SPLIT TRAIN/VAL/TEST DATASET SAFE")
    logging.info("=" * 70)
    
    try:
        # Chargement du dataset SAFE
        logging.info("📂 Chargement dataset SAFE...")
        df = pd.read_csv('data/ml_features_SAFE.csv')
        
        logging.info(f"✅ Dataset chargé: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
        
        # Conversion de la date
        logging.info("📅 Analyse des dates...")
        # Extraction de la date depuis l'ID de course (format: YYYY-MM-DD|R1|C1|VIN)
        df['date_extracted'] = df['date_course'].str.split('|').str[0]
        df['date_course'] = pd.to_datetime(df['date_extracted'])
        
        # Statistiques temporelles
        date_min = df['date_course'].min()
        date_max = df['date_course'].max()
        nb_annees = (date_max - date_min).days / 365.25
        
        logging.info(f"📊 Période couverte: {date_min.strftime('%Y-%m-%d')} à {date_max.strftime('%Y-%m-%d')}")
        logging.info(f"📊 Durée: {nb_annees:.1f} années")
        
        # Distribution par année
        df['annee'] = df['date_course'].dt.year
        repartition_annuelle = df['annee'].value_counts().sort_index()
        
        logging.info("\n📈 RÉPARTITION ANNUELLE:")
        for annee, count in repartition_annuelle.items():
            pct = (count / len(df)) * 100
            logging.info(f"   {annee}: {count:,} courses ({pct:.1f}%)")
        
        # Stratégie de split temporel
        # Train: 2020-2022 (3 années complètes)
        # Validation: 2023 (1 année)
        # Test: 2024-2025 (données récentes)
        
        logging.info("\n🎯 STRATÉGIE DE SPLIT TEMPOREL:")
        logging.info("   📚 Train: 2020-2022 (3 années)")
        logging.info("   🔍 Validation: 2023 (1 année)")
        logging.info("   🧪 Test: 2024-2025 (données récentes)")
        
        # Création des masques
        mask_train = (df['annee'] >= 2020) & (df['annee'] <= 2022)
        mask_val = (df['annee'] == 2023)
        mask_test = (df['annee'] >= 2024)
        
        # Extraction des ensembles
        df_train = df[mask_train].copy()
        df_val = df[mask_val].copy()
        df_test = df[mask_test].copy()
        
        # Statistiques des splits
        logging.info(f"\n📊 RÉSULTATS DU SPLIT:")
        logging.info(f"   📚 Train: {len(df_train):,} lignes ({len(df_train)/len(df)*100:.1f}%)")
        logging.info(f"   🔍 Val: {len(df_val):,} lignes ({len(df_val)/len(df)*100:.1f}%)")
        logging.info(f"   🧪 Test: {len(df_test):,} lignes ({len(df_test)/len(df)*100:.1f}%)")
        logging.info(f"   📊 Total: {len(df_train) + len(df_val) + len(df_test):,} lignes")
        
        # Vérification de la cohérence
        assert len(df_train) + len(df_val) + len(df_test) == len(df), "Perte de données dans le split!"
        
        # Distribution des targets dans chaque ensemble
        logging.info(f"\n🎯 DISTRIBUTION DES TARGETS:")
        
        for dataset_name, dataset in [('TRAIN', df_train), ('VAL', df_val), ('TEST', df_test)]:
            victoires_pct = (dataset['victoire'].sum() / len(dataset)) * 100
            places_pct = (dataset['place'].sum() / len(dataset)) * 100
            
            logging.info(f"   {dataset_name}:")
            logging.info(f"      🏆 Victoires: {dataset['victoire'].sum():,} ({victoires_pct:.1f}%)")
            logging.info(f"      🥉 Places: {dataset['place'].sum():,} ({places_pct:.1f}%)")
        
        # Sauvegarde des ensembles
        logging.info(f"\n💾 SAUVEGARDE DES ENSEMBLES:")
        
        # Suppression des colonnes temporaires
        for df_temp in [df_train, df_val, df_test]:
            for col in ['annee', 'date_extracted']:
                if col in df_temp.columns:
                    df_temp.drop(col, axis=1, inplace=True)
        
        # Sauvegarde
        df_train.to_csv('data/train_SAFE.csv', index=False)
        logging.info(f"   ✅ Train sauvé: data/train_SAFE.csv")
        
        df_val.to_csv('data/val_SAFE.csv', index=False)
        logging.info(f"   ✅ Validation sauvé: data/val_SAFE.csv")
        
        df_test.to_csv('data/test_SAFE.csv', index=False)
        logging.info(f"   ✅ Test sauvé: data/test_SAFE.csv")
        
        # Résumé final
        logging.info(f"\n📋 RÉSUMÉ FINAL:")
        logging.info(f"   📂 Fichiers générés: 3 datasets")
        logging.info(f"   📊 Colonnes par dataset: {df_train.shape[1]}")
        logging.info(f"   🎯 Targets disponibles: position_arrivee, victoire, place")
        logging.info(f"   📅 Split temporel: Garantie anti-leakage")
        
        return {
            'train_size': len(df_train),
            'val_size': len(df_val),
            'test_size': len(df_test),
            'nb_features': df_train.shape[1] - 3,  # -3 pour les targets
            'periode_train': '2020-2022',
            'periode_val': '2023',
            'periode_test': '2024-2025'
        }
        
    except Exception as e:
        logging.error(f"❌ Erreur lors du split: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    split_dataset_safe()