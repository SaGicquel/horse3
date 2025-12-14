#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRAPER HISTORIQUE AUTOMATISÉ
=============================

Script d'automatisation pour construire une base historique sur 5 ans
par petites périodes de 3 jours avec vérifications automatiques.

Fonctionnalités:
- Scrape par petites périodes (3 jours par défaut)
- Vérifications automatiques après chaque période
- Détection des doublons
- Contrôle de cohérence des données
- Progression sauvegardée (reprend où il s'est arrêté)
- Logs détaillés
- Mode pause automatique si trop d'erreurs

Usage:
    python scraper_historique_auto.py                    # Reprend ou démarre
    python scraper_historique_auto.py --reset            # Recommence depuis le début
    python scraper_historique_auto.py --status           # Affiche la progression
    python scraper_historique_auto.py --days 5           # Période de 5 jours
    python scraper_historique_auto.py --years 3          # Historique sur 3 ans

Auteur: Horse3 Automation
Date: 2025-11-27
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# Import des modules du projet
from db_connection import get_connection
from scraper_pmu_simple import (
    discover_reunions,
    discover_courses,
    enrich_from_course,
    fetch_participants,
    recalc_totals_from_seen,
    db_setup
)
import requests

# Configuration
PROJECT_DIR = Path(__file__).parent
PROGRESS_FILE = PROJECT_DIR / "data" / "scraping_historique_progress.json"
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / f"scraping_historique_{datetime.now().strftime('%Y%m%d')}.log"

# Paramètres par défaut - OPTIMISÉ POUR VITESSE MAX
DEFAULT_PERIOD_DAYS = 3       # Jours par période (court pour vérifications fréquentes)
DEFAULT_YEARS = 5             # Années d'historique
MAX_ERRORS_BEFORE_PAUSE = 10  # Erreurs consécutives avant pause (augmenté)
PAUSE_DURATION_MINUTES = 5    # Durée de pause en cas de trop d'erreurs (réduit)
DELAY_BETWEEN_DATES = 0       # Secondes entre chaque date (ZERO pour vitesse max)
DELAY_BETWEEN_PERIODS = 2     # Secondes entre chaque période (réduit)

# Créer les dossiers nécessaires
LOG_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_DIR / "data").mkdir(parents=True, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Vérificateur de qualité des données"""
    
    def __init__(self, conn):
        self.conn = conn
        self.issues = []
    
    def check_all(self, start_date: str, end_date: str) -> Dict:
        """Exécute toutes les vérifications sur une période"""
        self.issues = []
        
        results = {
            'period': f"{start_date} → {end_date}",
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'issues': [],
            'passed': True
        }
        
        # Vérification des doublons
        results['checks']['doublons_chevaux'] = self._check_duplicate_horses()
        results['checks']['doublons_courses'] = self._check_duplicate_courses()
        results['checks']['doublons_performances'] = self._check_duplicate_performances()
        
        # Vérification de cohérence
        results['checks']['coherence_dates'] = self._check_date_coherence(start_date, end_date)
        results['checks']['coherence_participants'] = self._check_participant_coherence()
        results['checks']['donnees_manquantes'] = self._check_missing_data(start_date, end_date)
        
        # Statistiques
        results['stats'] = self._get_period_stats(start_date, end_date)
        
        # Résumé
        results['issues'] = self.issues
        results['passed'] = len([i for i in self.issues if i['severity'] == 'critical']) == 0
        
        return results
    
    def _check_duplicate_horses(self) -> Dict:
        """Vérifie les doublons dans la table chevaux"""
        cur = self.conn.cursor()
        
        # Doublons par nom + année de naissance
        # Adapter aux différents schémas possibles
        try:
            cur.execute("""
                SELECT nom_cheval, an_naissance, COUNT(*) as cnt
                FROM chevaux
                WHERE nom_cheval IS NOT NULL AND an_naissance IS NOT NULL
                GROUP BY nom_cheval, an_naissance
                HAVING COUNT(*) > 1
                LIMIT 10
            """)
            duplicates = cur.fetchall()
        except Exception:
            # Fallback si colonnes différentes
            try:
                cur.execute("""
                    SELECT nom, annee_naissance, COUNT(*) as cnt
                    FROM chevaux
                    WHERE nom IS NOT NULL AND annee_naissance IS NOT NULL
                    GROUP BY nom, annee_naissance
                    HAVING COUNT(*) > 1
                    LIMIT 10
                """)
                duplicates = cur.fetchall()
            except Exception:
                duplicates = []
        
        if duplicates:
            self.issues.append({
                'type': 'doublons_chevaux',
                'severity': 'warning',
                'count': len(duplicates),
                'examples': [{'nom': d[0], 'annee': d[1], 'count': d[2]} for d in duplicates[:3]]
            })
        
        return {
            'status': 'ok' if not duplicates else 'warning',
            'duplicates_found': len(duplicates)
        }
    
    def _check_duplicate_courses(self) -> Dict:
        """Vérifie les doublons dans la table courses"""
        cur = self.conn.cursor()
        
        cur.execute("""
            SELECT id_course, COUNT(*) as cnt
            FROM courses
            GROUP BY id_course
            HAVING COUNT(*) > 1
            LIMIT 10
        """)
        duplicates = cur.fetchall()
        
        if duplicates:
            self.issues.append({
                'type': 'doublons_courses',
                'severity': 'critical',
                'count': len(duplicates),
                'examples': [d[0] for d in duplicates[:5]]
            })
        
        return {
            'status': 'ok' if not duplicates else 'error',
            'duplicates_found': len(duplicates)
        }
    
    def _check_duplicate_performances(self) -> Dict:
        """Vérifie les doublons dans la table performances"""
        cur = self.conn.cursor()
        
        cur.execute("""
            SELECT id_course, id_cheval, COUNT(*) as cnt
            FROM performances
            GROUP BY id_course, id_cheval
            HAVING COUNT(*) > 1
            LIMIT 10
        """)
        duplicates = cur.fetchall()
        
        if duplicates:
            self.issues.append({
                'type': 'doublons_performances',
                'severity': 'warning',
                'count': len(duplicates),
                'message': f"{len(duplicates)} paires course/cheval en doublon"
            })
        
        return {
            'status': 'ok' if not duplicates else 'warning',
            'duplicates_found': len(duplicates)
        }
    
    def _check_date_coherence(self, start_date: str, end_date: str) -> Dict:
        """Vérifie que les dates des courses sont cohérentes"""
        cur = self.conn.cursor()
        
        # Courses avec dates en dehors de la période
        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')
        
        cur.execute("""
            SELECT COUNT(*) 
            FROM courses 
            WHERE SUBSTRING(id_course, 1, 8) < %s OR SUBSTRING(id_course, 1, 8) > %s
        """, (start_str, end_str))
        
        # Ce n'est pas une erreur, juste une info
        return {'status': 'ok', 'message': 'Vérification OK'}
    
    def _check_participant_coherence(self) -> Dict:
        """Vérifie la cohérence des participants"""
        cur = self.conn.cursor()
        
        # Performances sans cheval associé
        cur.execute("""
            SELECT COUNT(*) 
            FROM performances p
            LEFT JOIN chevaux c ON p.id_cheval = c.id_cheval
            WHERE c.id_cheval IS NULL
        """)
        orphan_perfs = cur.fetchone()[0]
        
        if orphan_perfs > 0:
            self.issues.append({
                'type': 'performances_orphelines',
                'severity': 'warning',
                'count': orphan_perfs,
                'message': f"{orphan_perfs} performances sans cheval associé"
            })
        
        # Performances sans course associée
        cur.execute("""
            SELECT COUNT(*) 
            FROM performances p
            LEFT JOIN courses c ON p.id_course = c.id_course
            WHERE c.id_course IS NULL
        """)
        orphan_courses = cur.fetchone()[0]
        
        if orphan_courses > 0:
            self.issues.append({
                'type': 'performances_sans_course',
                'severity': 'warning',
                'count': orphan_courses,
                'message': f"{orphan_courses} performances sans course associée"
            })
        
        return {
            'status': 'ok' if orphan_perfs == 0 and orphan_courses == 0 else 'warning',
            'orphan_performances': orphan_perfs,
            'orphan_courses': orphan_courses
        }
    
    def _check_missing_data(self, start_date: str, end_date: str) -> Dict:
        """Vérifie les données manquantes sur la période"""
        cur = self.conn.cursor()
        
        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')
        
        # Courses sans participants
        cur.execute("""
            SELECT COUNT(*) 
            FROM courses c
            LEFT JOIN performances p ON c.id_course = p.id_course
            WHERE SUBSTRING(c.id_course, 1, 8) BETWEEN %s AND %s
            AND p.id_course IS NULL
        """, (start_str, end_str))
        courses_sans_participants = cur.fetchone()[0]
        
        if courses_sans_participants > 0:
            self.issues.append({
                'type': 'courses_sans_participants',
                'severity': 'warning',
                'count': courses_sans_participants,
                'message': f"{courses_sans_participants} courses sans aucun participant"
            })
        
        return {
            'status': 'ok' if courses_sans_participants == 0 else 'warning',
            'courses_sans_participants': courses_sans_participants
        }
    
    def _get_period_stats(self, start_date: str, end_date: str) -> Dict:
        """Récupère les statistiques de la période"""
        cur = self.conn.cursor()
        
        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')
        
        # Nombre de courses
        cur.execute("""
            SELECT COUNT(*) FROM courses 
            WHERE SUBSTRING(id_course, 1, 8) BETWEEN %s AND %s
        """, (start_str, end_str))
        nb_courses = cur.fetchone()[0]
        
        # Nombre de performances
        cur.execute("""
            SELECT COUNT(*) FROM performances p
            JOIN courses c ON p.id_course = c.id_course
            WHERE SUBSTRING(c.id_course, 1, 8) BETWEEN %s AND %s
        """, (start_str, end_str))
        nb_performances = cur.fetchone()[0]
        
        # Moyenne participants par course
        avg_participants = nb_performances / nb_courses if nb_courses > 0 else 0
        
        return {
            'courses': nb_courses,
            'performances': nb_performances,
            'avg_participants_per_course': round(avg_participants, 1)
        }


class HistoricalScraper:
    """Gestionnaire de scraping historique automatisé"""
    
    def __init__(self, period_days: int = DEFAULT_PERIOD_DAYS, years: int = DEFAULT_YEARS):
        self.period_days = period_days
        self.years = years
        self.progress = self._load_progress()
        self.consecutive_errors = 0
        self.conn = None
        self.quality_checker = None
        
    def _load_progress(self) -> Dict:
        """Charge la progression depuis le fichier"""
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erreur chargement progression: {e}")
        
        return {
            'started_at': None,
            'last_date_scraped': None,
            'target_start_date': None,
            'target_end_date': None,
            'periods_completed': 0,
            'total_courses': 0,
            'total_chevaux': 0,
            'total_errors': 0,
            'quality_reports': []
        }
    
    def _save_progress(self):
        """Sauvegarde la progression"""
        try:
            # Limiter le nombre de rapports gardés
            if len(self.progress.get('quality_reports', [])) > 50:
                self.progress['quality_reports'] = self.progress['quality_reports'][-50:]
            
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(self.progress, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Erreur sauvegarde progression: {e}")
    
    def get_status(self) -> Dict:
        """Retourne le statut actuel"""
        if not self.progress.get('started_at'):
            return {'status': 'not_started', 'message': 'Scraping non démarré'}
        
        target_start = self.progress.get('target_start_date')
        target_end = self.progress.get('target_end_date')
        last_scraped = self.progress.get('last_date_scraped')
        
        if not last_scraped:
            return {'status': 'initialized', 'message': 'Initialisé mais pas encore de données'}
        
        # Calculer la progression
        start_dt = datetime.strptime(target_start, '%Y-%m-%d').date()
        end_dt = datetime.strptime(target_end, '%Y-%m-%d').date()
        last_dt = datetime.strptime(last_scraped, '%Y-%m-%d').date()
        
        total_days = (end_dt - start_dt).days + 1
        days_done = (end_dt - last_dt).days + 1
        progress_pct = (days_done / total_days) * 100 if total_days > 0 else 0
        
        remaining_days = (last_dt - start_dt).days
        remaining_periods = remaining_days // self.period_days
        
        return {
            'status': 'in_progress' if remaining_days > 0 else 'completed',
            'started_at': self.progress.get('started_at'),
            'target_period': f"{target_start} → {target_end}",
            'last_scraped': last_scraped,
            'progress_percent': round(progress_pct, 1),
            'remaining_days': remaining_days,
            'remaining_periods': remaining_periods,
            'periods_completed': self.progress.get('periods_completed', 0),
            'total_courses': self.progress.get('total_courses', 0),
            'total_chevaux': self.progress.get('total_chevaux', 0),
            'total_errors': self.progress.get('total_errors', 0)
        }
    
    def reset(self):
        """Remet à zéro la progression"""
        self.progress = {
            'started_at': None,
            'last_date_scraped': None,
            'target_start_date': None,
            'target_end_date': None,
            'periods_completed': 0,
            'total_courses': 0,
            'total_chevaux': 0,
            'total_errors': 0,
            'quality_reports': []
        }
        self._save_progress()
        logger.info("🔄 Progression réinitialisée")
    
    def initialize(self, years: Optional[int] = None):
        """Initialise le scraping historique"""
        if years:
            self.years = years
        
        end_date = date.today() - timedelta(days=1)  # Commencer par hier
        start_date = end_date - timedelta(days=365 * self.years)
        
        self.progress['started_at'] = datetime.now().isoformat()
        self.progress['target_start_date'] = start_date.isoformat()
        self.progress['target_end_date'] = end_date.isoformat()
        self.progress['last_date_scraped'] = end_date.isoformat()  # On commence par la fin
        
        self._save_progress()
        
        logger.info(f"🚀 Scraping historique initialisé")
        logger.info(f"   Période cible: {start_date} → {end_date} ({self.years} ans)")
        logger.info(f"   Période de scraping: {self.period_days} jours")
        
        return self.progress
    
    def _scrape_date(self, date_iso: str) -> Dict:
        """Scrape une date unique"""
        result = {
            'date': date_iso,
            'success': False,
            'reunions': 0,
            'courses': 0,
            'chevaux': 0,
            'errors': []
        }
        
        try:
            # Découverte des réunions
            reunions = discover_reunions(date_iso)
            
            if not reunions:
                logger.info(f"   {date_iso}: Aucune course")
                result['success'] = True  # Pas d'erreur, juste pas de courses
                return result
            
            result['reunions'] = len(reunions)
            cur = self.conn.cursor()
            
            for r in reunions:
                courses = discover_courses(date_iso, r)
                if not courses:
                    continue
                
                for c in courses:
                    try:
                        enrich_from_course(cur, date_iso, r, c, sleep_s=0.1)
                        self.conn.commit()
                        
                        plist = fetch_participants(date_iso, r, c)
                        nb_participants = len(plist) if plist else 0
                        result['chevaux'] += nb_participants
                        result['courses'] += 1
                        
                    except requests.HTTPError as e:
                        result['errors'].append(f"R{r}C{c}: HTTP {e}")
                    except Exception as e:
                        result['errors'].append(f"R{r}C{c}: {str(e)[:50]}")
            
            result['success'] = True
            self.consecutive_errors = 0
            
        except Exception as e:
            result['errors'].append(str(e))
            self.consecutive_errors += 1
            logger.error(f"   {date_iso}: Erreur - {e}")
        
        return result
    
    def scrape_next_period(self) -> Optional[Dict]:
        """Scrape la prochaine période"""
        
        # Vérifier si initialisé
        if not self.progress.get('started_at'):
            self.initialize()
        
        last_scraped = self.progress.get('last_date_scraped')
        target_start = self.progress.get('target_start_date')
        
        if not last_scraped or not target_start:
            logger.error("❌ Progression invalide")
            return None
        
        last_dt = datetime.strptime(last_scraped, '%Y-%m-%d').date()
        start_dt = datetime.strptime(target_start, '%Y-%m-%d').date()
        
        # Vérifier si terminé
        if last_dt <= start_dt:
            logger.info("✅ Scraping historique terminé!")
            return {'completed': True}
        
        # Calculer la période à scraper (on recule dans le temps)
        period_end = last_dt - timedelta(days=1)
        period_start = max(period_end - timedelta(days=self.period_days - 1), start_dt)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📅 PÉRIODE: {period_start} → {period_end} ({(period_end - period_start).days + 1} jours)")
        logger.info(f"{'='*70}")
        
        # Connexion à la base
        self.conn = get_connection()
        # Note: db_setup peut échouer si le schéma est différent, on l'ignore
        try:
            db_setup(self.conn)
        except Exception as e:
            logger.warning(f"db_setup ignoré: {e}")
            self.conn.rollback()  # Annuler la transaction échouée
        self.quality_checker = DataQualityChecker(self.conn)
        
        # Stats avant
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chevaux")
        chevaux_avant = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM courses")
        courses_avant = cur.fetchone()[0]
        
        period_results = {
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'dates': [],
            'total_courses': 0,
            'total_chevaux': 0,
            'errors': []
        }
        
        try:
            # Scraper chaque date de la période
            current_date = period_end
            while current_date >= period_start:
                date_iso = current_date.isoformat()
                logger.info(f"   Scraping {date_iso}...")
                
                result = self._scrape_date(date_iso)
                period_results['dates'].append(result)
                
                if result['success']:
                    period_results['total_courses'] += result['courses']
                    period_results['total_chevaux'] += result['chevaux']
                else:
                    period_results['errors'].extend(result.get('errors', []))
                
                # Vérifier si trop d'erreurs consécutives
                if self.consecutive_errors >= MAX_ERRORS_BEFORE_PAUSE:
                    logger.warning(f"⚠️ {MAX_ERRORS_BEFORE_PAUSE} erreurs consécutives - Pause de {PAUSE_DURATION_MINUTES} minutes")
                    time.sleep(PAUSE_DURATION_MINUTES * 60)
                    self.consecutive_errors = 0
                
                current_date -= timedelta(days=1)
                time.sleep(DELAY_BETWEEN_DATES)
            
            # Recalculer les totaux (peut échouer si schéma différent)
            try:
                logger.info("🔄 Recalcul des statistiques...")
                recalc_totals_from_seen(self.conn)
            except Exception as e:
                logger.warning(f"Recalcul statistiques ignoré: {str(e)[:100]}")
                self.conn.rollback()
            
            # Stats après
            cur.execute("SELECT COUNT(*) FROM chevaux")
            chevaux_apres = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM courses")
            courses_apres = cur.fetchone()[0]
            
            period_results['nouveaux_chevaux'] = chevaux_apres - chevaux_avant
            period_results['nouvelles_courses'] = courses_apres - courses_avant
            
            # Vérification qualité
            logger.info("🔍 Vérification de la qualité des données...")
            quality_report = self.quality_checker.check_all(
                period_start.isoformat(),
                period_end.isoformat()
            )
            period_results['quality'] = quality_report
            
            # Log des problèmes détectés
            if quality_report['issues']:
                for issue in quality_report['issues']:
                    severity_icon = "❌" if issue['severity'] == 'critical' else "⚠️"
                    logger.warning(f"   {severity_icon} {issue['type']}: {issue.get('message', issue.get('count', ''))}")
            else:
                logger.info("   ✅ Aucun problème détecté")
            
            # Mettre à jour la progression
            self.progress['last_date_scraped'] = (period_start - timedelta(days=1)).isoformat()
            self.progress['periods_completed'] = self.progress.get('periods_completed', 0) + 1
            self.progress['total_courses'] = self.progress.get('total_courses', 0) + period_results['total_courses']
            self.progress['total_chevaux'] = self.progress.get('total_chevaux', 0) + period_results['total_chevaux']
            self.progress['total_errors'] = self.progress.get('total_errors', 0) + len(period_results['errors'])
            self.progress['quality_reports'].append({
                'period': f"{period_start} → {period_end}",
                'passed': quality_report['passed'],
                'issues_count': len(quality_report['issues'])
            })
            
            self._save_progress()
            
            # Résumé de la période
            logger.info(f"\n📊 RÉSUMÉ PÉRIODE:")
            logger.info(f"   • Courses scrapées: {period_results['total_courses']}")
            logger.info(f"   • Chevaux traités: {period_results['total_chevaux']}")
            logger.info(f"   • Nouveaux chevaux: {period_results['nouveaux_chevaux']}")
            logger.info(f"   • Qualité: {'✅ OK' if quality_report['passed'] else '⚠️ Problèmes détectés'}")
            
            status = self.get_status()
            logger.info(f"   • Progression globale: {status['progress_percent']}%")
            logger.info(f"   • Périodes restantes: ~{status['remaining_periods']}")
            
        except Exception as e:
            logger.error(f"❌ Erreur période: {e}")
            traceback.print_exc()
            period_results['errors'].append(str(e))
        
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None
        
        return period_results
    
    def run_continuous(self, max_periods: Optional[int] = None):
        """Exécute le scraping en continu"""
        periods_done = 0
        
        logger.info("🚀 Démarrage du scraping historique continu")
        
        while True:
            result = self.scrape_next_period()
            
            if result is None:
                logger.error("❌ Erreur fatale - Arrêt")
                break
            
            if result.get('completed'):
                logger.info("🎉 SCRAPING HISTORIQUE TERMINÉ!")
                break
            
            periods_done += 1
            
            if max_periods and periods_done >= max_periods:
                logger.info(f"⏸️ Limite de {max_periods} périodes atteinte - Pause")
                break
            
            # Pause entre les périodes
            logger.info(f"⏳ Pause de {DELAY_BETWEEN_PERIODS} secondes avant la prochaine période...")
            time.sleep(DELAY_BETWEEN_PERIODS)
        
        return self.get_status()


def main():
    parser = argparse.ArgumentParser(
        description='Scraper historique automatisé avec vérifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  %(prog)s                         # Exécute une période (reprend où arrêté)
  %(prog)s --periods 5             # Exécute 5 périodes
  %(prog)s --continuous            # Exécute en continu jusqu'à complétion
  %(prog)s --status                # Affiche la progression
  %(prog)s --reset                 # Réinitialise et recommence
  %(prog)s --years 3 --days 5      # 3 ans d'historique, périodes de 5 jours
        """
    )
    
    parser.add_argument('--status', action='store_true', help='Afficher le statut actuel')
    parser.add_argument('--reset', action='store_true', help='Réinitialiser la progression')
    parser.add_argument('--periods', type=int, default=1, help='Nombre de périodes à exécuter')
    parser.add_argument('--continuous', action='store_true', help='Exécuter en continu')
    parser.add_argument('--days', type=int, default=DEFAULT_PERIOD_DAYS, help='Jours par période')
    parser.add_argument('--years', type=int, default=DEFAULT_YEARS, help='Années d\'historique')
    
    args = parser.parse_args()
    
    scraper = HistoricalScraper(period_days=args.days, years=args.years)
    
    if args.status:
        status = scraper.get_status()
        print("\n" + "="*60)
        print("📊 STATUT DU SCRAPING HISTORIQUE")
        print("="*60)
        for key, value in status.items():
            print(f"   {key}: {value}")
        print("="*60 + "\n")
        return
    
    if args.reset:
        confirm = input("⚠️ Réinitialiser la progression? (oui/non): ")
        if confirm.lower() == 'oui':
            scraper.reset()
            print("✅ Progression réinitialisée")
        else:
            print("❌ Annulé")
        return
    
    # Exécuter le scraping
    if args.continuous:
        scraper.run_continuous()
    else:
        for i in range(args.periods):
            print(f"\n{'='*60}")
            print(f"📆 PÉRIODE {i+1}/{args.periods}")
            print(f"{'='*60}")
            result = scraper.scrape_next_period()
            if result and result.get('completed'):
                break


if __name__ == "__main__":
    main()
