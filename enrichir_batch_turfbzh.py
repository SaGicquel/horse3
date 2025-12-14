#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enrichissement Batch Turf.bzh
==============================

Enrichit toutes les courses d'une période avec les données Turf.bzh

Usage:
    python enrichir_batch_turfbzh.py --date-range 2025-10-12 2025-11-11
    python enrichir_batch_turfbzh.py --days 7

Auteur : Système d'enrichissement PMU
Date : 12 novembre 2025
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Tuple
from scraper_turfbzh import ScraperTurfBzh
from db_connection import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_courses_from_db(date_debut: str, date_fin: str) -> List[Tuple[str, int, int]]:
    """
    Récupère toutes les courses de la période depuis la base
    
    Args:
        date_debut: Date YYYY-MM-DD
        date_fin: Date YYYY-MM-DD
        
    Returns:
        Liste de tuples (date, reunion, course)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # Convertir dates en format BDD
        date_debut_db = datetime.strptime(date_debut, '%Y-%m-%d').strftime('%Y%m%d')
        date_fin_db = datetime.strptime(date_fin, '%Y-%m-%d').strftime('%Y%m%d')
        
        query = """
            SELECT DISTINCT
                SUBSTRING(id_course, 1, 8) as date_course,
                CAST(SUBSTRING(id_course FROM 'R(\\d+)_') AS INTEGER) as reunion,
                CAST(SUBSTRING(id_course FROM 'C(\\d+)$') AS INTEGER) as course
            FROM courses
            WHERE SUBSTRING(id_course, 1, 8) BETWEEN %s AND %s
            ORDER BY date_course, reunion, course
        """
        
        cur.execute(query, (date_debut_db, date_fin_db))
        results = cur.fetchall()
        
        # Convertir dates format YYYYMMDD → YYYY-MM-DD
        courses = []
        for date_db, reunion, course in results:
            date_str = f"{date_db[0:4]}-{date_db[4:6]}-{date_db[6:8]}"
            courses.append((date_str, reunion, course))
        
        return courses
        
    finally:
        cur.close()
        conn.close()


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description='Enrichissement batch Turf.bzh pour une période'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        '--date-range',
        nargs=2,
        metavar=('START', 'END'),
        help='Période au format YYYY-MM-DD YYYY-MM-DD'
    )
    
    group.add_argument(
        '--days',
        type=int,
        help='Nombre de jours en arrière depuis aujourd\'hui'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        default=0,
        help='Nombre de courses à sauter (pour reprendre après interruption)'
    )
    
    parser.add_argument(
        '--max-courses',
        type=int,
        help='Nombre maximum de courses à traiter (pour tests)'
    )
    
    args = parser.parse_args()
    
    # Calculer la période
    if args.date_range:
        date_debut = args.date_range[0]
        date_fin = args.date_range[1]
    else:
        date_fin = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        date_debut = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    logger.info("=" * 90)
    logger.info(f"🚀 ENRICHISSEMENT BATCH TURF.BZH")
    logger.info(f"   Période : {date_debut} → {date_fin}")
    logger.info("=" * 90)
    print()
    
    # Récupérer les courses
    courses = get_courses_from_db(date_debut, date_fin)
    logger.info(f"📋 {len(courses)} courses à enrichir")
    print()
    
    if not courses:
        logger.warning("⚠️  Aucune course trouvée pour cette période")
        return
    
    # Créer le scraper
    scraper = ScraperTurfBzh()
    
    # Statistiques
    total_courses = len(courses)
    courses_enrichies = 0
    perfs_enrichies_total = 0
    perfs_total = 0
    
    # Appliquer skip si demandé
    if args.skip > 0:
        logger.info(f"⏭️  Saut des {args.skip} premières courses")
        courses = courses[args.skip:]
        total_courses = len(courses)
    
    # Limiter le nombre de courses si demandé
    if args.max_courses:
        logger.info(f"🔢 Limitation à {args.max_courses} courses")
        courses = courses[:args.max_courses]
        total_courses = len(courses)
    
    # Enrichir chaque course
    for i, (date, reunion, course) in enumerate(courses, 1):
        actual_index = i + args.skip if args.skip > 0 else i
        logger.info(f"[{actual_index}/{total_courses + args.skip}] 📅 {date} R{reunion}C{course}")
        
        try:
            enriched, total = scraper.enrich_course(date, reunion, course)
            
            if enriched > 0:
                courses_enrichies += 1
                perfs_enrichies_total += enriched
                perfs_total += total
                logger.info(f"   ✅ {enriched}/{total} performances enrichies")
            else:
                logger.info(f"   ⏭️  Aucune donnée Turf.bzh disponible")
        
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Interruption utilisateur (Ctrl+C)")
            logger.info(f"💡 Pour reprendre : --skip {actual_index}")
            raise
        except Exception as e:
            logger.error(f"   ❌ Erreur inattendue : {e}")
            logger.info(f"   ⏭️  Passage à la course suivante")
        
        print()
    
    # Rapport final
    logger.info("=" * 90)
    logger.info("📊 RAPPORT FINAL")
    logger.info("=" * 90)
    logger.info(f"   Courses traitées        : {total_courses}")
    logger.info(f"   Courses enrichies       : {courses_enrichies} ({courses_enrichies/total_courses*100:.1f}%)")
    logger.info(f"   Performances enrichies  : {perfs_enrichies_total}/{perfs_total} ({perfs_enrichies_total/perfs_total*100:.1f}% si > 0 else 0)")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
