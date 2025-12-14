#!/usr/bin/env python3
"""
================================================================================
MIGRATION MASSIVE - CHEVAL_COURSES_SEEN → TABLES NORMALISÉES
================================================================================

Description : Migre TOUTES les données de cheval_courses_seen vers les tables
normalisées (chevaux, jockeys, entraineurs, hippodromes, courses, performances)

ATTENTION : Ce script va traiter 800K+ enregistrements, cela peut prendre 
du temps (estimé : 30-60 minutes)

Usage :
  python migrate_full_dataset.py --year 2020-2025  # Par années
  python migrate_full_dataset.py --all             # Tout d'un coup
  python migrate_full_dataset.py --test            # Test sur 1000 lignes

================================================================================
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import psycopg2
from db_connection import get_connection

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FullDatasetMigrator:
    """Migre les données de cheval_courses_seen vers les tables normalisées"""
    
    def __init__(self):
        self.conn = get_connection()
        self.cursor = self.conn.cursor()
        
        # Caches pour éviter les doublons
        self.cache_chevaux = {}      # nom_norm -> id_cheval
        self.cache_jockeys = {}      # nom -> id_jockey  
        self.cache_entraineurs = {}  # nom -> id_entraineur
        self.cache_hippodromes = {}  # nom -> id_hippodrome
        self.cache_courses = {}      # (date, hippodrome, numero) -> id_course
        
    def get_or_create_cheval(self, nom_norm: str, race: str = None, 
                           an_naissance: int = None, sexe: str = None) -> int:
        """Récupère ou crée un cheval"""
        if nom_norm in self.cache_chevaux:
            return self.cache_chevaux[nom_norm]
        
        # Chercher dans la DB
        self.cursor.execute("""
            SELECT id_cheval FROM chevaux WHERE nom_cheval = %s
        """, (nom_norm,))
        result = self.cursor.fetchone()
        
        if result:
            cheval_id = result[0]
        else:
            # Créer nouveau cheval
            self.cursor.execute("""
                INSERT INTO chevaux (nom_cheval, race_cheval, an_naissance, sexe_cheval)
                VALUES (%s, %s, %s, %s)
                RETURNING id_cheval
            """, (nom_norm, race, an_naissance, sexe))
            cheval_id = self.cursor.fetchone()[0]
        
        self.cache_chevaux[nom_norm] = cheval_id
        return cheval_id
    
    def get_or_create_jockey(self, nom_jockey: str) -> Optional[int]:
        """Récupère ou crée un jockey"""
        if not nom_jockey or nom_jockey.strip() == "":
            return None
            
        nom_clean = nom_jockey.strip()
        if nom_clean in self.cache_jockeys:
            return self.cache_jockeys[nom_clean]
        
        # Chercher dans la DB
        self.cursor.execute("""
            SELECT id_jockey FROM jockeys WHERE nom_jockey = %s
        """, (nom_clean,))
        result = self.cursor.fetchone()
        
        if result:
            jockey_id = result[0]
        else:
            # Créer nouveau jockey
            self.cursor.execute("""
                INSERT INTO jockeys (nom_jockey)
                VALUES (%s)
                RETURNING id_jockey
            """, (nom_clean,))
            jockey_id = self.cursor.fetchone()[0]
        
        self.cache_jockeys[nom_clean] = jockey_id
        return jockey_id
    
    def get_or_create_entraineur(self, nom_entraineur: str) -> Optional[int]:
        """Récupère ou crée un entraineur"""
        if not nom_entraineur or nom_entraineur.strip() == "":
            return None
            
        nom_clean = nom_entraineur.strip()
        if nom_clean in self.cache_entraineurs:
            return self.cache_entraineurs[nom_clean]
        
        # Chercher dans la DB
        self.cursor.execute("""
            SELECT id_entraineur FROM entraineurs WHERE nom_entraineur = %s
        """, (nom_clean,))
        result = self.cursor.fetchone()
        
        if result:
            entraineur_id = result[0]
        else:
            # Créer nouveau entraineur
            self.cursor.execute("""
                INSERT INTO entraineurs (nom_entraineur)
                VALUES (%s)
                RETURNING id_entraineur
            """, (nom_clean,))
            entraineur_id = self.cursor.fetchone()[0]
        
        self.cache_entraineurs[nom_clean] = entraineur_id
        return entraineur_id
    
    def get_or_create_hippodrome(self, nom_hippodrome: str, ville: str = None) -> int:
        """Récupère ou crée un hippodrome"""
        nom_clean = nom_hippodrome.strip()
        if nom_clean in self.cache_hippodromes:
            return self.cache_hippodromes[nom_clean]
        
        # Chercher dans la DB
        self.cursor.execute("""
            SELECT id_hippodrome FROM hippodromes WHERE nom_hippodrome = %s
        """, (nom_clean,))
        result = self.cursor.fetchone()
        
        if result:
            hippodrome_id = result[0]
        else:
            # Créer nouvel hippodrome
            self.cursor.execute("""
                INSERT INTO hippodromes (nom_hippodrome, ville)
                VALUES (%s, %s)
                RETURNING id_hippodrome
            """, (nom_clean, ville))
            hippodrome_id = self.cursor.fetchone()[0]
        
        self.cache_hippodromes[nom_clean] = hippodrome_id
        return hippodrome_id
    
    def get_or_create_course(self, date_course: datetime, hippodrome_id: int, 
                           numero_course: int, **course_data) -> int:
        """Récupère ou crée une course"""
        cache_key = (date_course.strftime('%Y-%m-%d'), hippodrome_id, numero_course)
        if cache_key in self.cache_courses:
            return self.cache_courses[cache_key]
        
        # Chercher dans la DB
        self.cursor.execute("""
            SELECT id_course FROM courses 
            WHERE date_course = %s AND id_hippodrome = %s AND numero_course = %s
        """, (date_course, hippodrome_id, numero_course))
        result = self.cursor.fetchone()
        
        if result:
            course_id = result[0]
        else:
            # Créer nouvelle course
            self.cursor.execute("""
                INSERT INTO courses (
                    date_course, id_hippodrome, numero_course,
                    distance, discipline, etat_piste, meteo,
                    temperature_c, vent_kmh, nombre_partants, allocation
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id_course
            """, (
                date_course, hippodrome_id, numero_course,
                course_data.get('distance_m'),
                course_data.get('discipline'),
                course_data.get('etat_piste'),
                course_data.get('meteo'),
                course_data.get('temperature_c'),
                course_data.get('vent_kmh'),
                course_data.get('nombre_partants'),
                course_data.get('allocation_totale')
            ))
            course_id = self.cursor.fetchone()[0]
        
        self.cache_courses[cache_key] = course_id
        return course_id
    
    def convert_raw_date(self, annee: int, heure_depart: str = None) -> datetime:
        """Convertit année + heure en datetime"""
        # Pour l'instant, on utilise le 1er janvier de l'année
        # TODO: Améliorer avec des infos de date plus précises si disponibles
        base_date = datetime(annee, 1, 1)
        
        # Si on a une heure, l'ajouter (format approximatif)
        if heure_depart:
            try:
                # Extraire l'heure si possible
                if ':' in str(heure_depart):
                    hour_parts = str(heure_depart).split(':')
                    hour = int(hour_parts[0])
                    minute = int(hour_parts[1]) if len(hour_parts) > 1 else 0
                    base_date = base_date.replace(hour=hour, minute=minute)
            except:
                pass  # Utiliser la date de base
        
        return base_date
    
    def migrate_batch(self, limit: int = None, offset: int = 0, 
                     year_filter: str = None) -> int:
        """Migre un batch de données"""
        
        # Construction de la requête
        where_clause = "WHERE annee IS NOT NULL"
        params = []
        
        if year_filter:
            if '-' in year_filter:  # Range d'années
                start_year, end_year = year_filter.split('-')
                where_clause += " AND annee BETWEEN %s AND %s"
                params.extend([int(start_year), int(end_year)])
            else:  # Année spécifique
                where_clause += " AND annee = %s"
                params.append(int(year_filter))
        
        limit_clause = f"LIMIT {limit}" if limit else ""
        offset_clause = f"OFFSET {offset}" if offset > 0 else ""
        
        query = f"""
        SELECT 
            nom_norm, race, annee, sexe, age,
            driver_jockey, entraineur,
            hippodrome_nom, reunion_numero, course_numero,
            distance_m, discipline, etat_piste, meteo,
            temperature_c, vent_kmh, nombre_partants, allocation_totale,
            heure_depart, numero_dossard,
            place_finale, temps_sec, cote_finale,
            gains_course, non_partant, disqualifie,
            race_key
        FROM cheval_courses_seen
        {where_clause}
        ORDER BY annee, race_key
        {limit_clause} {offset_clause}
        """
        
        logger.info(f"📥 Extraction des données brutes (offset={offset}, limit={limit})...")
        
        self.cursor.execute(query, params)
        raw_data = self.cursor.fetchall()
        
        if not raw_data:
            logger.info("   ℹ️  Aucune donnée à traiter")
            return 0
        
        logger.info(f"   ✅ {len(raw_data):,} lignes extraites")
        
        migrated_count = 0
        for i, row in enumerate(raw_data):
            try:
                (nom_norm, race, annee, sexe, age,
                 jockey_nom, entraineur_nom,
                 hippodrome_nom, reunion_numero, course_numero,
                 distance_m, discipline, etat_piste, meteo,
                 temperature_c, vent_kmh, nombre_partants, allocation_totale,
                 heure_depart, numero_dossard,
                 place_finale, temps_sec, cote_finale,
                 gains_course, non_partant, disqualifie, race_key) = row
                
                # Calculer année de naissance approximative
                an_naissance = annee - age if age else None
                
                # Créer/récupérer les entités
                cheval_id = self.get_or_create_cheval(nom_norm, race, an_naissance, sexe)
                jockey_id = self.get_or_create_jockey(jockey_nom)
                entraineur_id = self.get_or_create_entraineur(entraineur_nom)
                hippodrome_id = self.get_or_create_hippodrome(hippodrome_nom)
                
                # Construire la date de course
                date_course = self.convert_raw_date(annee, heure_depart)
                
                # Créer/récupérer la course
                course_id = self.get_or_create_course(
                    date_course, hippodrome_id, course_numero,
                    distance_m=distance_m, discipline=discipline,
                    etat_piste=etat_piste, meteo=meteo,
                    temperature_c=temperature_c, vent_kmh=vent_kmh,
                    nombre_partants=nombre_partants, allocation_totale=allocation_totale
                )
                
                # Créer la performance
                try:
                    self.cursor.execute("""
                        INSERT INTO performances (
                            id_course, id_cheval, id_jockey, id_entraineur,
                            numero_corde, position_arrivee, temps_total,
                            cote_sp, gains, non_partant, disqualifie
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        course_id, cheval_id, jockey_id, entraineur_id,
                        numero_dossard, place_finale, temps_sec,
                        cote_finale, gains_course, 
                        1 if non_partant else 0, 1 if disqualifie else 0
                    ))
                    migrated_count += 1
                    
                except psycopg2.IntegrityError:
                    # Performance déjà existante, on passe
                    self.conn.rollback()
                    continue
                
                # Commit toutes les 1000 lignes
                if i > 0 and i % 1000 == 0:
                    self.conn.commit()
                    logger.info(f"   💾 Batch commit ({i:,}/{len(raw_data):,} lignes traitées)")
                
            except Exception as e:
                logger.error(f"❌ Erreur ligne {i}: {e}")
                self.conn.rollback()
                continue
        
        # Commit final
        self.conn.commit()
        logger.info(f"✅ Batch terminé : {migrated_count:,} performances migrées")
        
        return migrated_count
    
    def migrate_all(self, year_filter: str = None, test_mode: bool = False) -> int:
        """Migre toutes les données par batches"""
        
        logger.info("🚀 Début de la migration massive...")
        logger.info(f"   Filtre années : {year_filter or 'Toutes'}")
        logger.info(f"   Mode test : {test_mode}")
        
        total_migrated = 0
        batch_size = 1000 if test_mode else 10000
        offset = 0
        
        while True:
            limit = batch_size if not test_mode or total_migrated < 1000 else 0
            if limit == 0:
                break
                
            migrated = self.migrate_batch(limit=batch_size, offset=offset, 
                                        year_filter=year_filter)
            
            if migrated == 0:
                break
                
            total_migrated += migrated
            offset += batch_size
            
            logger.info(f"📊 Progression : {total_migrated:,} performances migrées au total")
            
            if test_mode and total_migrated >= 1000:
                logger.info("🧪 Mode test : arrêt à 1000 lignes")
                break
        
        logger.info(f"🎉 Migration terminée : {total_migrated:,} performances migrées")
        return total_migrated
    
    def close(self):
        """Ferme la connexion"""
        self.cursor.close()
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Migration massive des données historiques')
    parser.add_argument('--year', help='Année(s) à migrer (ex: 2021 ou 2020-2023)')
    parser.add_argument('--all', action='store_true', help='Migrer toutes les années')
    parser.add_argument('--test', action='store_true', help='Mode test (1000 lignes)')
    
    args = parser.parse_args()
    
    if not args.year and not args.all and not args.test:
        logger.error("❌ Spécifiez --year, --all ou --test")
        return
    
    try:
        migrator = FullDatasetMigrator()
        
        year_filter = None
        if args.year:
            year_filter = args.year
        elif not args.all and not args.test:
            year_filter = "2020-2025"  # Par défaut
        
        total = migrator.migrate_all(year_filter=year_filter, test_mode=args.test)
        migrator.close()
        
        logger.info(f"✅ Migration réussie : {total:,} performances")
        
    except Exception as e:
        logger.error(f"❌ Erreur de migration : {e}")
        raise


if __name__ == '__main__':
    main()