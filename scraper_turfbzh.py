#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scraper Turf.bzh - Extraction des cotes PMU en temps réel
==========================================================

Objectif : Récupérer les cotes live depuis Turf.bzh pour enrichir la base de données

Format URL : https://www.turf.bzh/pronostics-pmu-DDMMYYYY-R{reunion}C{course}.html

Sources de données :
- Cote PMU live (cote)
- Cote précédente (prev_cote)  
- Prédictions IA (Top1IA, Top2IA, etc.)
- Classement ELO (pointsCheval, pointsJockey, pointsTrainer)
- Stats popularité (fans, PrevPop)

Auteur : Système d'enrichissement PMU
Date : 12 novembre 2025
"""

import re
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from db_connection import get_connection

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScraperTurfBzh:
    """Scraper pour récupérer les cotes et données depuis Turf.bzh"""
    
    BASE_URL = "https://www.turf.bzh"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        })
        # Timeouts par défaut (connexion, lecture)
        self.default_timeout = (10, 30)  # 10s connexion, 30s lecture
        self.max_retries = 2
        
    def format_date_url(self, date_str: str) -> str:
        """
        Convertit une date YYYY-MM-DD en format DDMMYYYY pour l'URL
        
        Args:
            date_str: Date au format YYYY-MM-DD
            
        Returns:
            Date au format DDMMYYYY (ex: 12112025)
        """
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%d%m%Y')
    
    def build_url(self, date_str: str, reunion: int, course: int) -> str:
        """
        Construit l'URL Turf.bzh pour une course donnée
        
        Args:
            date_str: Date YYYY-MM-DD
            reunion: Numéro de réunion (1-14)
            course: Numéro de course (1-8)
            
        Returns:
            URL complète (ex: https://www.turf.bzh/pronostics-pmu-12112025-R1C1.html)
        """
        date_formatted = self.format_date_url(date_str)
        return f"{self.BASE_URL}/pronostics-pmu-{date_formatted}-R{reunion}C{course}.html"
    
    def extract_table_data(self, html_content: str) -> Optional[List[Dict]]:
        """
        Extrait l'objet JavaScript tableData depuis le HTML
        
        Args:
            html_content: Contenu HTML de la page
            
        Returns:
            Liste des partants avec toutes leurs données, ou None si erreur
        """
        try:
            # Chercher le pattern tableData: [...] jusqu'au prochain "," à la racine
            # On doit gérer les objets imbriqués et les virgules dans les strings
            pattern = r'tableData:\s*\[(.*?)\],\s*\w+:'
            match = re.search(pattern, html_content, re.DOTALL)
            
            if not match:
                logger.warning("Impossible de trouver tableData dans le HTML")
                return None
            
            # Reconstruire le tableau JSON complet
            json_str = '[' + match.group(1) + ']'
            
            # Remplacer les séquences d'échappement problématiques
            # Gérer les &#039; (apostrophes HTML)
            json_str = json_str.replace('&#039;', "'")
            json_str = json_str.replace('&quot;', '"')
            
            # Parser le JSON
            data = json.loads(json_str)
            
            logger.info(f"✅ {len(data)} partants extraits")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erreur de parsing JSON : {e}")
            # Sauvegarder le JSON problématique pour debug
            with open('/tmp/turf_bzh_debug.json', 'w', encoding='utf-8') as f:
                f.write(json_str if 'json_str' in locals() else html_content[:5000])
            logger.error("JSON sauvegardé dans /tmp/turf_bzh_debug.json pour analyse")
            return None
        except Exception as e:
            logger.error(f"❌ Erreur extraction tableData : {e}")
            return None
    
    def scrape_course(self, date_str: str, reunion: int, course: int) -> Optional[List[Dict]]:
        """
        Scrape une course spécifique sur Turf.bzh avec retry automatique
        
        Args:
            date_str: Date YYYY-MM-DD
            reunion: Numéro de réunion
            course: Numéro de course
            
        Returns:
            Liste des partants avec leurs données, ou None si erreur
        """
        url = self.build_url(date_str, reunion, course)
        logger.info(f"🔍 Scraping : {url}")
        
        # Tentatives avec retry
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.default_timeout)
                
                if response.status_code == 404:
                    logger.warning(f"⚠️  404 - Course non trouvée : R{reunion}C{course}")
                    return None
                
                if response.status_code != 200:
                    logger.error(f"❌ Erreur HTTP {response.status_code}")
                    return None
                
                # Extraire les données
                data = self.extract_table_data(response.text)
                
                if data:
                    logger.info(f"✅ Course R{reunion}C{course} : {len(data)} partants")
                    # Afficher quelques infos pour debug
                    for partant in data[:3]:  # 3 premiers seulement
                        num = partant.get('numPmu', '?')
                        nom = partant.get('nom', '?')
                        cote = partant.get('cote', '-')
                        prev_cote = partant.get('prev_cote', '-')
                        logger.debug(f"   N°{num} {nom} - Cote: {cote} (Préc: {prev_cote})")
                
                return data
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    logger.warning(f"⏱️  Timeout (tentative {attempt}/{self.max_retries}), retry dans 2s...")
                    import time
                    time.sleep(2)
                else:
                    logger.error(f"❌ Timeout après {self.max_retries} tentatives : {url}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    logger.warning(f"⚠️  Erreur réseau (tentative {attempt}/{self.max_retries}): {e}, retry...")
                    import time
                    time.sleep(2)
                else:
                    logger.error(f"❌ Erreur réseau après {self.max_retries} tentatives : {e}")
                    return None
        
        return None
    
    def get_id_course_from_db(self, date_str: str, reunion: int, course: int) -> Optional[str]:
        """
        Récupère l'id_course depuis la base de données
        
        Args:
            date_str: Date YYYY-MM-DD
            reunion: Numéro de réunion
            course: Numéro de course
            
        Returns:
            id_course (format: YYYYMMDD_CODE_R1_C1) ou None
        """
        conn = get_connection()
        cur = conn.cursor()
        
        try:
            # Convertir date en format base de données
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            date_db = date_obj.strftime('%Y%m%d')
            
            # Chercher la course avec pattern LIKE
            query = """
                SELECT id_course 
                FROM courses 
                WHERE id_course LIKE %s 
                  AND id_course LIKE %s
                LIMIT 1
            """
            
            cur.execute(query, (f"{date_db}%", f"%_R{reunion}_C{course}"))
            result = cur.fetchone()
            
            if result:
                return result[0]
            else:
                logger.warning(f"⚠️  Course non trouvée en base : {date_db} R{reunion}C{course}")
                return None
                
        finally:
            cur.close()
            conn.close()
    
    def enrich_course(self, date_str: str, reunion: int, course: int) -> Tuple[int, int]:
        """
        Enrichit une course avec les données Turf.bzh
        
        Args:
            date_str: Date YYYY-MM-DD
            reunion: Numéro de réunion
            course: Numéro de course
            
        Returns:
            Tuple (performances_enrichies, performances_totales)
        """
        # 1. Scraper Turf.bzh
        data = self.scrape_course(date_str, reunion, course)
        if not data:
            return (0, 0)
        
        # 2. Récupérer id_course depuis BDD
        id_course = self.get_id_course_from_db(date_str, reunion, course)
        if not id_course:
            logger.error(f"❌ Impossible de trouver la course en base : R{reunion}C{course}")
            return (0, 0)
        
        # 3. Enrichir les performances
        conn = get_connection()
        cur = conn.cursor()
        
        enriched = 0
        total = len(data)
        
        try:
            for partant in data:
                num_pmu = partant.get('numPmu')
                cote = partant.get('cote', '-')
                prev_cote = partant.get('prev_cote')
                
                if not num_pmu:
                    continue
                
                # Convertir cote "-" en NULL
                cote_value = None if cote == '-' else cote
                prev_cote_value = None if not prev_cote else prev_cote
                
                # Données supplémentaires intéressantes
                top1_ia = partant.get('Top1IA')
                points_cheval = partant.get('pointsCheval')
                fans = partant.get('fans')
                
                # UPDATE avec nouvelles colonnes
                update_query = """
                    UPDATE performances
                    SET 
                        cote_turfbzh = %s,
                        cote_turfbzh_precedente = %s,
                        prediction_ia_gagnant = %s,
                        elo_cheval = %s,
                        popularite = %s,
                        updated_at = NOW()
                    WHERE id_course = %s 
                      AND numero_corde = %s
                """
                
                cur.execute(update_query, (
                    cote_value,
                    prev_cote_value,
                    top1_ia,
                    points_cheval,
                    fans,
                    id_course,
                    num_pmu
                ))
                
                if cur.rowcount > 0:
                    enriched += 1
            
            conn.commit()
            logger.info(f"✅ {enriched}/{total} performances enrichies pour {id_course}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erreur lors de l'enrichissement : {e}")
            
        finally:
            cur.close()
            conn.close()
        
        return (enriched, total)


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description='Scraper Turf.bzh pour enrichir les cotes PMU'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Date au format YYYY-MM-DD (ex: 2025-11-12)',
        required=True
    )
    
    parser.add_argument(
        '--reunion',
        type=int,
        help='Numéro de réunion (1-14)',
        required=True
    )
    
    parser.add_argument(
        '--course',
        type=int,
        help='Numéro de course (1-8)',
        required=True
    )
    
    args = parser.parse_args()
    
    # Créer le scraper
    scraper = ScraperTurfBzh()
    
    # Enrichir la course
    logger.info("=" * 80)
    logger.info(f"🚀 ENRICHISSEMENT TURF.BZH - {args.date} R{args.reunion}C{args.course}")
    logger.info("=" * 80)
    
    enriched, total = scraper.enrich_course(args.date, args.reunion, args.course)
    
    logger.info("=" * 80)
    logger.info(f"✅ TERMINÉ : {enriched}/{total} performances enrichies")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
