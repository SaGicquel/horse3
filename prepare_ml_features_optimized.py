#!/usr/bin/env python3
"""
VERSION OPTIMISÉE DE prepare_ml_features.py POUR GROS VOLUMES
Calcul efficace des features ML sur 661K performances historiques
"""

import pandas as pd
import numpy as np
import logging
from db_connection import get_connection
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedMLFeatureEngineer:
    """Version optimisée pour traitement de gros volumes"""
    
    def __init__(self):
        self.conn = get_connection()
        logger.info("✅ Connexion base de données établie")
    
    def extract_and_prepare_features(self) -> pd.DataFrame:
        """Extraction et calcul des features en une seule étape optimisée"""
        logger.info("📊 Extraction optimisée des données avec features pré-calculées...")
        
        query = """
        WITH cheval_stats AS (
            -- Statistiques pré-calculées par cheval
            SELECT 
                nom_norm,
                COUNT(*) as nb_courses_total,
                SUM(CASE WHEN place_finale = 1 THEN 1 ELSE 0 END) as nb_victoires_total,
                SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) as nb_places_total,
                AVG(CASE WHEN place_finale = 1 THEN 1.0 ELSE 0.0 END) as taux_victoires,
                AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) as taux_places,
                AVG(place_finale) as position_moyenne,
                AVG(COALESCE(cote_finale, 5.0)) as cote_moyenne
            FROM cheval_courses_seen
            WHERE place_finale IS NOT NULL AND place_finale > 0
            GROUP BY nom_norm
        ),
        jockey_stats AS (
            -- Statistiques par jockey
            SELECT 
                driver_jockey,
                COUNT(*) as nb_courses_jockey,
                AVG(CASE WHEN place_finale = 1 THEN 1.0 ELSE 0.0 END) as taux_victoires_jockey,
                AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) as taux_places_jockey
            FROM cheval_courses_seen
            WHERE place_finale IS NOT NULL AND place_finale > 0
            GROUP BY driver_jockey
        ),
        entraineur_stats AS (
            -- Statistiques par entraineur
            SELECT 
                entraineur,
                COUNT(*) as nb_courses_entraineur,
                AVG(CASE WHEN place_finale = 1 THEN 1.0 ELSE 0.0 END) as taux_victoires_entraineur,
                AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) as taux_places_entraineur
            FROM cheval_courses_seen
            WHERE place_finale IS NOT NULL AND place_finale > 0
            GROUP BY entraineur
        ),
        course_stats AS (
            -- Statistiques par course
            SELECT 
                race_key,
                AVG(COALESCE(cote_finale, 5.0)) as cote_moyenne_course,
                STDDEV(COALESCE(cote_finale, 5.0)) as ecart_type_cotes,
                MAX(COALESCE(cote_finale, 5.0)) as cote_max_course,
                MIN(COALESCE(cote_finale, 5.0)) as cote_min_course
            FROM cheval_courses_seen
            WHERE place_finale IS NOT NULL AND place_finale > 0
            GROUP BY race_key
        )
        
        SELECT 
            -- Identifiants
            ROW_NUMBER() OVER (ORDER BY c.annee, c.race_key, c.numero_dossard) as id_performance,
            c.race_key as id_course,
            c.nom_norm,
            c.annee,
            
            -- Targets
            c.place_finale as position_arrivee,
            CASE WHEN c.place_finale = 1 THEN 1 ELSE 0 END as victoire,
            CASE WHEN c.place_finale <= 3 THEN 1 ELSE 0 END as place,
            
            -- Features de base
            c.numero_dossard as numero_corde,
            COALESCE(c.cote_finale, 5.0) as cote_sp,
            COALESCE(c.distance_m, 2000) as distance,
            COALESCE(c.discipline, 'Trot') as discipline,
            COALESCE(c.nombre_partants, 12) as nombre_partants,
            COALESCE(c.hippodrome_nom, 'UNKNOWN') as hippodrome_nom,
            COALESCE(c.driver_jockey, 'UNKNOWN') as driver_jockey,
            COALESCE(c.entraineur, 'UNKNOWN') as entraineur,
            c.race_key as date_course,
            COALESCE(c.sexe, 'H') as sexe,
            COALESCE(c.age, 5) as age,
            
            -- Features cheval (historiques)
            COALESCE(cs.nb_courses_total, 1) as nb_courses_carriere,
            COALESCE(cs.nb_victoires_total, 0) as nb_victoires_carriere,
            COALESCE(cs.nb_places_total, 0) as nb_places_carriere,
            COALESCE(cs.taux_victoires, 0.1) as taux_victoires_cheval,
            COALESCE(cs.taux_places, 0.3) as taux_places_cheval,
            COALESCE(cs.position_moyenne, 6.0) as position_moyenne_cheval,
            COALESCE(cs.cote_moyenne, 10.0) as cote_moyenne_cheval,
            
            -- Features jockey
            COALESCE(js.nb_courses_jockey, 1) as nb_courses_jockey,
            COALESCE(js.taux_victoires_jockey, 0.1) as taux_victoires_jockey,
            COALESCE(js.taux_places_jockey, 0.3) as taux_places_jockey,
            
            -- Features entraineur
            COALESCE(es.nb_courses_entraineur, 1) as nb_courses_entraineur,
            COALESCE(es.taux_victoires_entraineur, 0.1) as taux_victoires_entraineur,
            COALESCE(es.taux_places_entraineur, 0.3) as taux_places_entraineur,
            
            -- Features course
            COALESCE(cours.cote_moyenne_course, 10.0) as cote_moyenne_course,
            COALESCE(cours.ecart_type_cotes, 5.0) as ecart_type_cotes_course,
            COALESCE(cours.cote_max_course, 50.0) as cote_max_course,
            COALESCE(cours.cote_min_course, 1.5) as cote_min_course
            
        FROM cheval_courses_seen c
        LEFT JOIN cheval_stats cs ON c.nom_norm = cs.nom_norm
        LEFT JOIN jockey_stats js ON c.driver_jockey = js.driver_jockey  
        LEFT JOIN entraineur_stats es ON c.entraineur = es.entraineur
        LEFT JOIN course_stats cours ON c.race_key = cours.race_key
        
        WHERE c.place_finale IS NOT NULL
          AND c.place_finale > 0
          AND COALESCE(c.non_partant, 0) = 0
          AND COALESCE(c.disqualifie, 0) = 0
        
        ORDER BY c.annee, c.race_key, c.numero_dossard
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ {len(df):,} performances extraites avec statistiques pré-calculées")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features dérivées calculées rapidement"""
        logger.info("🔄 Calcul des features dérivées...")
        
        # Features numériques normalisées
        df['distance_norm'] = df['distance'] / 1000.0
        df['cote_log'] = np.log1p(df['cote_sp'])
        df['age_norm'] = df['age'] / 10.0
        df['annee_norm'] = (df['annee'] - 2020) / 5.0
        
        # Features binaires
        df['is_favori'] = (df['cote_sp'] < 5.0).astype(int)
        df['is_outsider'] = (df['cote_sp'] > 20.0).astype(int)
        df['is_longue_distance'] = (df['distance'] > 2000).astype(int)
        df['is_sprint'] = (df['distance'] < 1400).astype(int)
        df['is_male'] = (df['sexe'] == 'HONGRES').astype(int)
        df['is_femelle'] = (df['sexe'] == 'FEMELLES').astype(int)
        
        # Features de position
        df['corde_relative'] = df['numero_corde'] / df['nombre_partants'].clip(lower=1)
        df['corde_bonne'] = (df['numero_corde'] <= 3).astype(int)
        df['corde_mauvaise'] = (df['numero_corde'] > 10).astype(int)
        
        # Features de course
        df['petite_course'] = (df['nombre_partants'] <= 8).astype(int)
        df['grande_course'] = (df['nombre_partants'] >= 16).astype(int)
        df['course_competitive'] = (df['ecart_type_cotes_course'] > 5.0).astype(int)
        
        # Features de rang/classement
        df['rang_cote_course'] = df.groupby('id_course')['cote_sp'].rank()
        df['rang_experience_cheval'] = df.groupby('id_course')['nb_courses_carriere'].rank(ascending=False)
        df['rang_taux_victoires_cheval'] = df.groupby('id_course')['taux_victoires_cheval'].rank(ascending=False)
        
        # Features d'interaction
        df['cote_x_experience'] = df['cote_log'] * np.log1p(df['nb_courses_carriere'])
        df['jockey_entraineur_synergie'] = df['taux_victoires_jockey'] * df['taux_victoires_entraineur']
        df['favori_bonne_corde'] = df['is_favori'] * df['corde_bonne']
        df['experience_x_distance'] = np.log1p(df['nb_courses_carriere']) * df['distance_norm']
        
        # Features de niveau de course
        df['niveau_course'] = df.groupby('id_course')['taux_victoires_cheval'].transform('mean')
        df['avantage_relatif'] = df['taux_victoires_cheval'] - df['niveau_course']
        
        # Features discipline
        df['is_trot'] = (df['discipline'].str.contains('TROT|ATTELE|MONTE', na=False)).astype(int)
        df['is_galop'] = (df['discipline'].str.contains('PLAT|GALOP', na=False)).astype(int)
        
        logger.info(f"✅ {len(df.columns)} features totales calculées")
        return df
    
    def prepare_features(self) -> pd.DataFrame:
        """Pipeline complet optimisé"""
        logger.info("🚀 DÉMARRAGE FEATURE ENGINEERING OPTIMISÉ")
        logger.info("=" * 80)
        
        # Étape 1: Extraction avec pré-calculs SQL
        df = self.extract_and_prepare_features()
        
        # Étape 2: Features dérivées rapides
        df = self.add_derived_features(df)
        
        # Nettoyage final
        df = df.fillna(0)
        
        logger.info("=" * 80)
        logger.info("🎉 FEATURE ENGINEERING TERMINÉ")
        logger.info(f"📊 Dataset final: {len(df):,} lignes × {len(df.columns)} colonnes")
        
        return df
    
    def close(self):
        """Ferme la connexion"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

def main():
    """Fonction principale"""
    output_file = 'data/ml_features.csv'
    
    try:
        engineer = OptimizedMLFeatureEngineer()
        df = engineer.prepare_features()
        
        # Sauvegarder
        logger.info(f"💾 Sauvegarde dans {output_file}...")
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Fichier sauvegardé: {output_file}")
        
        # Vérification finale
        logger.info("")
        logger.info("🔍 VÉRIFICATION FINALE:")
        logger.info(f"   - Lignes: {len(df):,}")
        logger.info(f"   - Colonnes: {len(df.columns)}")
        logger.info(f"   - Période: {df['annee'].min()}-{df['annee'].max()}")
        logger.info(f"   - Victoires: {df['victoire'].sum():,} ({df['victoire'].mean()*100:.1f}%)")
        logger.info(f"   - Places: {df['place'].sum():,} ({df['place'].mean()*100:.1f}%)")
        
        engineer.close()
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise

if __name__ == '__main__':
    main()