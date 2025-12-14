#!/usr/bin/env python3
"""
================================================================================
PREPARE ML FEATURES - VERSION RAPIDE (VECTORISÉE)
================================================================================

Version optimisée avec calculs vectorisés pour traiter rapidement l'historique complet.

Usage :
  python prepare_ml_features_fast.py --output data/ml_features_fast.csv

================================================================================
"""

import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from db_connection import get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_data():
    """Extrait et prépare les données de base"""
    logger.info("📥 Extraction des données...")
    
    conn = get_connection()
    
    query = """
    SELECT 
        ROW_NUMBER() OVER (ORDER BY annee, race_key, numero_dossard) as id_performance,
        race_key,
        nom_norm,
        annee,
        place_finale as position_arrivee,
        CASE WHEN place_finale = 1 THEN 1 ELSE 0 END as victoire,
        CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as place,
        numero_dossard as numero_corde,
        cote_finale as cote_sp,
        cote_matin as cote_pm,
        distance_m as distance,
        discipline,
        etat_piste,
        meteo,
        temperature_c,
        vent_kmh,
        nombre_partants,
        allocation_totale as allocation,
        hippodrome_nom,
        driver_jockey,
        entraineur,
        gains_course,
        age,
        sexe
    FROM cheval_courses_seen
    WHERE annee IS NOT NULL
      AND place_finale IS NOT NULL  
      AND place_finale > 0
      AND non_partant != 1
      AND disqualifie != 1
    ORDER BY annee, race_key, numero_dossard
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Date simplifiée (année + jour aléatoire)
    np.random.seed(42)  # Pour la reproductibilité
    df['date_course'] = pd.to_datetime(df['annee'].astype(str) + '-01-01') + \
                       pd.to_timedelta(np.random.randint(0, 365, len(df)), unit='D')
    
    logger.info(f"   ✅ {len(df):,} performances extraites")
    return df


def calculate_cheval_stats(df):
    """Calcule les statistiques par cheval (vectorisé)"""
    logger.info("🐎 Calcul des stats chevaux (vectorisé)...")
    
    # Trier par cheval et date
    df = df.sort_values(['nom_norm', 'date_course']).reset_index(drop=True)
    
    # Stats de base par cheval
    cheval_stats = df.groupby('nom_norm').agg({
        'victoire': ['count', 'sum', 'mean'],
        'place': ['sum', 'mean'],
        'position_arrivee': ['mean', 'std']
    }).reset_index()
    
    cheval_stats.columns = [
        'nom_norm', 'nb_courses_total', 'nb_victoires_total', 'taux_victoires_total',
        'nb_places_total', 'taux_places_total', 'position_moyenne', 'regularite_total'
    ]
    
    # Merge avec le dataset principal
    df = df.merge(cheval_stats, on='nom_norm', how='left')
    
    # Features approximatives (plus rapides à calculer)
    df['forme_5c'] = df['taux_victoires_total']  # Approximation
    df['forme_10c'] = df['taux_victoires_total'] 
    df['nb_courses_12m'] = df['nb_courses_total'] * 0.3  # Approximation 30% sur 12m
    df['nb_victoires_12m'] = df['nb_victoires_total'] * 0.3
    df['taux_places_12m'] = df['taux_places_total']
    df['regularite'] = 1.0 / (1.0 + df['regularite_total'].fillna(3))
    df['jours_depuis_derniere'] = 30  # Valeur par défaut
    
    # Aptitudes approximatives
    distance_stats = df.groupby(['nom_norm', pd.cut(df['distance'], 5)])['victoire'].mean().reset_index()
    distance_stats['aptitude_distance'] = distance_stats['victoire']
    
    df['aptitude_distance'] = df['taux_victoires_total']  # Approximation
    df['aptitude_piste'] = df['taux_victoires_total']
    df['aptitude_hippodrome'] = df['taux_victoires_total']
    
    logger.info("   ✅ Stats chevaux calculées")
    return df


def calculate_jockey_entraineur_stats(df):
    """Calcule les stats jockey/entraineur"""
    logger.info("👤 Calcul des stats jockey/entraineur...")
    
    # Stats jockey
    jockey_stats = df.groupby('driver_jockey').agg({
        'victoire': ['count', 'sum'],
        'place': 'sum'
    }).reset_index()
    jockey_stats.columns = ['driver_jockey', 'jockey_courses', 'jockey_victoires', 'jockey_places']
    jockey_stats['taux_victoires_jockey'] = jockey_stats['jockey_victoires'] / jockey_stats['jockey_courses']
    jockey_stats['taux_places_jockey'] = jockey_stats['jockey_places'] / jockey_stats['jockey_courses']
    
    df = df.merge(jockey_stats[['driver_jockey', 'taux_victoires_jockey', 'taux_places_jockey']], 
                 on='driver_jockey', how='left')
    
    # Stats entraineur
    entraineur_stats = df.groupby('entraineur').agg({
        'victoire': ['count', 'sum'],
        'place': 'sum'
    }).reset_index()
    entraineur_stats.columns = ['entraineur', 'entraineur_courses', 'entraineur_victoires', 'entraineur_places']
    entraineur_stats['taux_victoires_entraineur'] = entraineur_stats['entraineur_victoires'] / entraineur_stats['entraineur_courses']
    entraineur_stats['taux_places_entraineur'] = entraineur_stats['entraineur_places'] / entraineur_stats['entraineur_courses']
    
    df = df.merge(entraineur_stats[['entraineur', 'taux_victoires_entraineur', 'taux_places_entraineur']], 
                 on='entraineur', how='left')
    
    # Synergies
    df['synergie_jockey_cheval'] = df['taux_victoires_jockey'] * df['forme_5c']
    df['synergie_entraineur_cheval'] = df['taux_victoires_entraineur'] * df['forme_5c']
    
    # Remplir les NaN
    for col in ['taux_victoires_jockey', 'taux_places_jockey', 'taux_victoires_entraineur', 
                'taux_places_entraineur', 'synergie_jockey_cheval', 'synergie_entraineur_cheval']:
        df[col] = df[col].fillna(0)
    
    logger.info("   ✅ Stats jockey/entraineur calculées")
    return df


def calculate_course_features(df):
    """Calcule les features de course"""
    logger.info("🏁 Calcul des features course...")
    
    # Normalisation distance
    df['distance_norm'] = (df['distance'] - df['distance'].mean()) / df['distance'].std()
    
    # Niveau moyen concurrent par course
    course_quality = df.groupby('race_key')['forme_5c'].mean().reset_index()
    course_quality.columns = ['race_key', 'niveau_moyen_concurrent']
    df = df.merge(course_quality, on='race_key', how='left')
    
    logger.info("   ✅ Features course calculées")
    return df


def calculate_market_features(df):
    """Calcule les features de marché"""
    logger.info("💰 Calcul des features marché...")
    
    # Rang des cotes par course
    df['rang_cote_sp'] = df.groupby('race_key')['cote_sp'].rank(method='min')
    
    # Features marché par défaut
    df['rang_cote_turfbzh'] = df['rang_cote_sp']  # Approximation
    df['ecart_cote_ia'] = 0
    df['prediction_ia_gagnant'] = 1.0 / df['rang_cote_sp']  # Approximation
    df['elo_cheval'] = 1000 + (df['taux_victoires_total'] * 500)  # Approximation
    
    logger.info("   ✅ Features marché calculées")
    return df


def add_encodings_and_interactions(df):
    """Ajoute encodages et interactions"""
    logger.info("🔢 Encodages et interactions...")
    
    # Encodages
    df['discipline_Plat'] = (df['discipline'] == 'Plat').astype(int)
    df['discipline_Trot'] = (df['discipline'] == 'Trot').astype(int)
    df['sexe_H'] = (df['sexe'] == 'H').astype(int)
    df['sexe_M'] = (df['sexe'] == 'M').astype(int)
    
    # États de piste simplifiés
    for etat in ['Bon léger', 'Souple', 'PSF']:
        col_name = f"etat_{etat.replace(' ', '_').replace('é', 'e')}"
        df[col_name] = (df['etat_piste'] == etat).astype(int)
    
    # Top hippodromes
    top_hippos = df['hippodrome_nom'].value_counts().head(10).index
    df['hippodrome_top20'] = df['hippodrome_nom'].isin(top_hippos).astype(int)
    
    for i, hippo in enumerate(top_hippos[:5]):  # Top 5
        col_name = f"hippodrome_TOP_{i+1}"
        df[col_name] = (df['hippodrome_nom'] == hippo).astype(int)
    
    # Interactions
    df['interaction_forme_jockey'] = df['forme_5c'] * df['taux_victoires_jockey']
    df['interaction_aptitude_distance'] = df['aptitude_distance'] * df['distance_norm']
    df['interaction_elo_niveau'] = df['elo_cheval'] * df['niveau_moyen_concurrent']
    df['interaction_cote_ia'] = df['rang_cote_sp'] * df['prediction_ia_gagnant']
    df['interaction_synergie_forme'] = df['synergie_jockey_cheval'] * df['forme_10c']
    df['interaction_victoires_jockey'] = df['taux_victoires_jockey'] * df['nb_victoires_12m']
    
    # Popularité hippodrome
    hippo_counts = df['hippodrome_nom'].value_counts().to_dict()
    df['popularite_hippodrome'] = df['hippodrome_nom'].map(hippo_counts)
    
    df['interaction_aptitude_popularite'] = df['aptitude_hippodrome'] * df['popularite_hippodrome']
    df['interaction_regularite_volume'] = df['regularite'] * df['nb_courses_12m']
    
    logger.info("   ✅ Encodages et interactions ajoutés")
    return df


def prepare_final_dataset(df):
    """Prépare le dataset final"""
    logger.info("🧹 Finalisation du dataset...")
    
    # Colonnes essentielles (similaires au script original)
    essential_cols = [
        'id_performance', 'race_key', 'nom_norm', 'date_course',
        'position_arrivee', 'victoire', 'place',
        'numero_corde', 'cote_sp', 'cote_pm', 'distance', 'discipline',
        'etat_piste', 'meteo', 'temperature_c', 'vent_kmh', 'nombre_partants',
        'allocation', 'hippodrome_nom', 'driver_jockey', 'entraineur'
    ]
    
    # Features calculées
    feature_cols = [
        'forme_5c', 'forme_10c', 'nb_courses_12m', 'nb_victoires_12m',
        'taux_places_12m', 'regularite', 'jours_depuis_derniere',
        'aptitude_distance', 'aptitude_piste', 'aptitude_hippodrome',
        'taux_victoires_jockey', 'taux_places_jockey',
        'taux_victoires_entraineur', 'taux_places_entraineur',
        'synergie_jockey_cheval', 'synergie_entraineur_cheval',
        'distance_norm', 'niveau_moyen_concurrent',
        'rang_cote_sp', 'rang_cote_turfbzh', 'ecart_cote_ia',
        'prediction_ia_gagnant', 'elo_cheval'
    ]
    
    # Encodages
    encoding_cols = [col for col in df.columns if 'discipline_' in col or 'sexe_' in col or 
                    'etat_' in col or 'hippodrome_' in col]
    
    # Interactions
    interaction_cols = [col for col in df.columns if 'interaction_' in col or col == 'popularite_hippodrome']
    
    # Toutes les colonnes finales
    final_cols = essential_cols + feature_cols + encoding_cols + interaction_cols
    final_cols = [col for col in final_cols if col in df.columns]
    
    df_final = df[final_cols].copy()
    
    # Remplacer les valeurs manquantes
    df_final = df_final.fillna(0)
    
    # Remplacer les infinis
    df_final = df_final.replace([np.inf, -np.inf], 0)
    
    logger.info(f"   ✅ Dataset final : {len(df_final):,} lignes, {len(df_final.columns):,} colonnes")
    return df_final


def main():
    parser = argparse.ArgumentParser(description='Génère les features ML rapidement')
    parser.add_argument('--output', default='data/ml_features_fast.csv', help='Fichier de sortie')
    
    args = parser.parse_args()
    
    try:
        # Pipeline complet
        df = extract_data()
        df = calculate_cheval_stats(df)
        df = calculate_jockey_entraineur_stats(df)
        df = calculate_course_features(df)
        df = calculate_market_features(df)
        df = add_encodings_and_interactions(df)
        df_final = prepare_final_dataset(df)
        
        # Sauvegarde
        logger.info(f"💾 Sauvegarde dans {args.output}...")
        df_final.to_csv(args.output, index=False)
        
        # Résumé
        logger.info("🎉 GÉNÉRATION TERMINÉE !")
        logger.info(f"   📁 Fichier : {args.output}")
        logger.info(f"   📊 Performances : {len(df_final):,}")
        logger.info(f"   🏇 Chevaux uniques : {df_final['nom_norm'].nunique():,}")
        logger.info(f"   🏁 Courses uniques : {df_final['race_key'].nunique():,}")
        logger.info(f"   📈 Features : {len(df_final.columns):,}")
        logger.info(f"   📅 Période : 2020-2025")
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        raise


if __name__ == '__main__':
    main()