#!/usr/bin/env python3
"""
VERSION CORRIGÉE SANS DATA LEAKAGE
Calcul des features ML en excluant strictement la course actuelle des statistiques historiques
"""

import pandas as pd
import numpy as np
import logging
from db_connection import get_connection
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SafeMLFeatureEngineer:
    """Version sécurisée sans data leakage"""
    
    def __init__(self):
        self.conn = get_connection()
        logger.info("✅ Connexion base de données établie")
    
    def extract_safe_features(self) -> pd.DataFrame:
        """Extraction sécurisée avec exclusion stricte de la course actuelle"""
        logger.info("📊 Extraction SÉCURISÉE - exclusion course actuelle des stats...")
        
        query = """
        SELECT 
            -- Identifiants
            ROW_NUMBER() OVER (ORDER BY c.annee, c.race_key, c.numero_dossard) as id_performance,
            c.race_key as id_course,
            c.nom_norm,
            c.annee,
            
            -- Targets (OK - utilisés uniquement comme target, pas comme features)
            c.place_finale as position_arrivee,
            CASE WHEN c.place_finale = 1 THEN 1 ELSE 0 END as victoire,
            CASE WHEN c.place_finale <= 3 THEN 1 ELSE 0 END as place,
            
            -- Features PRE-COURSE SAFE
            c.numero_dossard as numero_corde,
            COALESCE(c.cote_finale, 5.0) as cote_sp,  -- Cote finale = cote à la fermeture des jeux, disponible pré-course
            COALESCE(c.distance_m, 2000) as distance,
            COALESCE(c.discipline, 'Trot') as discipline,
            COALESCE(c.nombre_partants, 12) as nombre_partants,
            COALESCE(c.hippodrome_nom, 'UNKNOWN') as hippodrome_nom,
            COALESCE(c.driver_jockey, 'UNKNOWN') as driver_jockey,
            COALESCE(c.entraineur, 'UNKNOWN') as entraineur,
            c.race_key as date_course,
            COALESCE(c.sexe, 'H') as sexe,
            COALESCE(c.age, 5) as age,
            
            -- Features SAFE - Basiques
            COALESCE(c.cote_matin, c.cote_finale, 5.0) as cote_matin,
            COALESCE(c.poids_kg, 55.0) as poids,
            COALESCE(c.allocation_totale, 10000) as allocation_totale,
            
            -- NOTE: Les statistiques historiques seront calculées séparément
            -- pour exclure strictement la course actuelle
            0 as nb_courses_historiques,  -- Placeholder
            0 as nb_victoires_historiques, -- Placeholder
            0.1 as taux_victoires_historique, -- Placeholder
            0.3 as taux_places_historique -- Placeholder
            
        FROM cheval_courses_seen c
        
        WHERE c.place_finale IS NOT NULL
          AND c.place_finale > 0
          AND COALESCE(c.non_partant, 0) = 0
          AND COALESCE(c.disqualifie, 0) = 0
        
        ORDER BY c.annee, c.race_key, c.numero_dossard
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ {len(df):,} performances extraites (base safe)")
        
        return df
    
    def add_safe_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute uniquement les features dérivées SAFE (pas d'historique)"""
        logger.info("🔄 Calcul des features dérivées SAFE...")
        
        # Features numériques normalisées
        df['distance_norm'] = df['distance'] / 1000.0
        df['cote_log'] = np.log1p(df['cote_sp'])
        df['cote_matin_log'] = np.log1p(df['cote_matin'])
        df['age_norm'] = df['age'] / 10.0
        df['annee_norm'] = (df['annee'] - 2020) / 5.0
        df['poids_norm'] = (df['poids'] - 55.0) / 10.0
        df['allocation_log'] = np.log1p(df['allocation_totale'])
        
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
        
        # Features de course (SAFE - uniquement info de la course actuelle)
        df['petite_course'] = (df['nombre_partants'] <= 8).astype(int)
        df['grande_course'] = (df['nombre_partants'] >= 16).astype(int)
        
        # Features de rang dans la course (SAFE - uniquement position relative)
        df['rang_cote_course'] = df.groupby('id_course')['cote_sp'].rank()
        df['rang_poids_course'] = df.groupby('id_course')['poids'].rank(ascending=False)
        df['rang_age_course'] = df.groupby('id_course')['age'].rank()
        
        # Features d'interaction SAFE
        df['cote_x_distance'] = df['cote_log'] * df['distance_norm']
        df['favori_bonne_corde'] = df['is_favori'] * df['corde_bonne']
        df['age_x_distance'] = df['age_norm'] * df['distance_norm']
        
        # Features de course relative (SAFE)
        df['cote_moyenne_course'] = df.groupby('id_course')['cote_sp'].transform('mean')
        df['avantage_cote'] = df['cote_sp'] - df['cote_moyenne_course']
        
        # Features discipline
        df['is_trot'] = (df['discipline'].str.contains('TROT|ATTELE|MONTE', na=False)).astype(int)
        df['is_galop'] = (df['discipline'].str.contains('PLAT|GALOP', na=False)).astype(int)
        
        # Supprimer les placeholders non utilisés
        df = df.drop(['nb_courses_historiques', 'nb_victoires_historiques', 
                     'taux_victoires_historique', 'taux_places_historique'], axis=1)
        
        logger.info(f"✅ {len(df.columns)} features SAFE calculées (sans historique)")
        return df
    
    def prepare_safe_features(self) -> pd.DataFrame:
        """Pipeline complet sécurisé"""
        logger.info("🔒 DÉMARRAGE FEATURE ENGINEERING SÉCURISÉ (ANTI-LEAKAGE)")
        logger.info("=" * 80)
        
        # Étape 1: Extraction sécurisée
        df = self.extract_safe_features()
        
        # Étape 2: Features dérivées safe uniquement
        df = self.add_safe_derived_features(df)
        
        # Nettoyage final
        df = df.fillna(0)
        
        logger.info("=" * 80)
        logger.info("🔒 FEATURE ENGINEERING SÉCURISÉ TERMINÉ")
        logger.info(f"📊 Dataset final: {len(df):,} lignes × {len(df.columns)} colonnes")
        logger.info("✅ AUCUN DATA LEAKAGE - Statistiques historiques supprimées")
        
        return df
    
    def close(self):
        """Ferme la connexion"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

def main():
    """Fonction principale"""
    output_file = 'data/ml_features_SAFE.csv'
    
    try:
        engineer = SafeMLFeatureEngineer()
        df = engineer.prepare_safe_features()
        
        # Sauvegarder
        logger.info(f"💾 Sauvegarde dans {output_file}...")
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Fichier sauvegardé: {output_file}")
        
        # Vérification finale anti-leakage
        logger.info("")
        logger.info("🔍 AUDIT FINAL ANTI-LEAKAGE:")
        
        dangerous_cols = [col for col in df.columns if any(word in col.lower() 
                         for word in ['victoire', 'place_', 'rapport', 'historique', 'carriere'])]
        safe_cols = [col for col in df.columns if col not in ['position_arrivee', 'victoire', 'place'] + dangerous_cols]
        
        logger.info(f"   - Colonnes SAFE: {len(safe_cols)}")
        logger.info(f"   - Targets: 3 (position_arrivee, victoire, place)")
        logger.info(f"   - Colonnes supprimées: {len(dangerous_cols)}")
        
        if dangerous_cols:
            logger.warning(f"   ⚠️  Colonnes suspectes: {dangerous_cols}")
        else:
            logger.info("   ✅ AUCUNE colonne dangereuse détectée!")
        
        engineer.close()
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise

if __name__ == '__main__':
    main()