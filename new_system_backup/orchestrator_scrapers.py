#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ORCHESTRATEUR - Lance plusieurs scrapers en séquence
Permet de lancer tous les scrapers disponibles pour enrichir la base.
"""

import sys
import time
from datetime import date, timedelta
from typing import List, Dict

# Import des scrapers
from scrapers.metadata_course import MetadataCourseScraper
from scrapers.cheval_details import ChevalDetailsScraper
from scrapers.perf_stats import PerfStatsScraper
from scrapers.conditions_jour import ConditionsJourScraper
from scrapers.cotes_marche import CotesMarcheScraper
from scrapers.connections import ConnectionsScraper
from scrapers.features_ml import FeaturesMLScraper


class ScraperOrchestrator:
    """
    Orchestrateur pour lancer plusieurs scrapers en séquence.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.scrapers = []
        self.register_scrapers()
    
    def register_scrapers(self):
        """Enregistre les scrapers disponibles"""
        self.scrapers = [
            {
                "name": "SCRAPER 1 - Métadonnées course",
                "class": MetadataCourseScraper,
                "emoji": "📋",
                "description": "Identifiants, horaires, météo, stalles",
            },
            {
                "name": "SCRAPER 2 - Détails cheval",
                "class": ChevalDetailsScraper,
                "emoji": "🐴",
                "description": "Handicap, équipement, forme récente",
            },
            {
                "name": "SCRAPER 3 - Statistiques performances",
                "class": PerfStatsScraper,
                "emoji": "📊",
                "description": "RK, SF, forme, consistance, records",
            },
            {
                "name": "SCRAPER 4 - Conditions jour",
                "class": ConditionsJourScraper,
                "emoji": "🏁",
                "description": "Biais stalles, pace, topographie, pénétrométrie",
            },
            {
                "name": "SCRAPER 5 - Cotes marché",
                "class": CotesMarcheScraper,
                "emoji": "💰",
                "description": "Cotes, évolution, value bets",
            },
            {
                "name": "SCRAPER 8 - Connections",
                "class": ConnectionsScraper,
                "emoji": "🤝",
                "description": "Winrates entraîneurs, jockeys, combos",
            },
            {
                "name": "SCRAPER 7 - Features ML",
                "class": FeaturesMLScraper,
                "emoji": "🤖",
                "description": "Features avancées, probas, Kelly",
            },
        ]
    
    def run_date(self, date_iso: str, scrapers_to_run: List[int] = None) -> Dict:
        """
        Lance les scrapers pour une date donnée.
        
        Args:
            date_iso: Date ISO (YYYY-MM-DD)
            scrapers_to_run: Liste des indices de scrapers à lancer (None = tous)
            
        Returns:
            Dict avec les stats globales
        """
        print(f"\n{'='*80}")
        print(f"🚀 ORCHESTRATEUR - Enrichissement {date_iso}")
        print(f"{'='*80}\n")
        
        # Déterminer quels scrapers lancer
        if scrapers_to_run is None:
            scrapers_to_run = list(range(len(self.scrapers)))
        
        # Stats globales
        global_stats = {
            "date": date_iso,
            "scrapers": [],
            "total_time": 0,
            "success": True,
        }
        
        start_time = time.time()
        
        # Lancer chaque scraper
        for idx in scrapers_to_run:
            if idx >= len(self.scrapers):
                continue
            
            scraper_info = self.scrapers[idx]
            
            print(f"{scraper_info['emoji']} {scraper_info['name']}")
            print(f"   → {scraper_info['description']}")
            print()
            
            scraper_start = time.time()
            
            try:
                # Instancier et lancer
                scraper_class = scraper_info["class"]
                with scraper_class(verbose=self.verbose) as scraper:
                    stats = scraper.scrape_date(date_iso)
                
                scraper_time = time.time() - scraper_start
                
                # Stats scraper
                scraper_stats = {
                    "name": scraper_info["name"],
                    "success": True,
                    "time": scraper_time,
                    "stats": stats,
                }
                
                global_stats["scrapers"].append(scraper_stats)
                
                print(f"   ✅ Terminé en {scraper_time:.1f}s")
                print()
                
            except Exception as e:
                scraper_time = time.time() - scraper_start
                
                print(f"   ❌ ERREUR: {e}")
                print()
                
                scraper_stats = {
                    "name": scraper_info["name"],
                    "success": False,
                    "time": scraper_time,
                    "error": str(e),
                }
                
                global_stats["scrapers"].append(scraper_stats)
                global_stats["success"] = False
        
        # Temps total
        global_stats["total_time"] = time.time() - start_time
        
        # Afficher résumé
        self.print_summary(global_stats)
        
        return global_stats
    
    def print_summary(self, stats: Dict):
        """Affiche le résumé des scrapers"""
        print(f"{'='*80}")
        print(f"📊 RÉSUMÉ GLOBAL")
        print(f"{'='*80}")
        print(f"Date: {stats['date']}")
        print(f"Temps total: {stats['total_time']:.1f}s")
        print(f"Statut: {'✅ SUCCÈS' if stats['success'] else '❌ ÉCHEC'}")
        print()
        
        for scraper_stats in stats["scrapers"]:
            status = "✅" if scraper_stats["success"] else "❌"
            print(f"{status} {scraper_stats['name']}")
            print(f"   Temps: {scraper_stats['time']:.1f}s")
            
            if scraper_stats["success"]:
                s = scraper_stats["stats"]
                if "reunions" in s:
                    print(f"   Réunions: {s['reunions']}")
                if "courses" in s:
                    print(f"   Courses: {s['courses']}")
                if "participants_updated" in s:
                    print(f"   Participants: {s['participants_updated']}")
                if "chevaux_updated" in s:
                    print(f"   Chevaux: {s['chevaux_updated']}")
                if "erreurs" in s and s["erreurs"] > 0:
                    print(f"   ⚠️  Erreurs: {s['erreurs']}")
            else:
                print(f"   Erreur: {scraper_stats.get('error', 'Inconnue')}")
            print()
        
        print(f"{'='*80}\n")
    
    def run_date_range(self, start_date: str, end_date: str, scrapers_to_run: List[int] = None):
        """
        Lance les scrapers sur une période de dates.
        
        Args:
            start_date: Date début (YYYY-MM-DD)
            end_date: Date fin (YYYY-MM-DD)
            scrapers_to_run: Liste des scrapers à lancer
        """
        from datetime import datetime
        
        d1 = datetime.strptime(start_date, "%Y-%m-%d").date()
        d2 = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        current = d1
        all_stats = []
        
        while current <= d2:
            date_iso = current.isoformat()
            stats = self.run_date(date_iso, scrapers_to_run)
            all_stats.append(stats)
            
            current += timedelta(days=1)
        
        # Résumé global période
        self.print_period_summary(all_stats)
    
    def print_period_summary(self, all_stats: List[Dict]):
        """Affiche le résumé pour une période"""
        print(f"\n{'='*80}")
        print(f"📈 RÉSUMÉ PÉRIODE")
        print(f"{'='*80}")
        print(f"Dates traitées: {len(all_stats)}")
        print(f"Succès: {sum(1 for s in all_stats if s['success'])}/{len(all_stats)}")
        print(f"Temps total: {sum(s['total_time'] for s in all_stats):.1f}s")
        print(f"{'='*80}\n")


def main():
    """Point d'entrée CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Orchestrateur de scrapers PMU")
    parser.add_argument("--date", help="Date à scraper (YYYY-MM-DD, défaut: aujourd'hui)")
    parser.add_argument("--start", help="Date début pour une période (YYYY-MM-DD)")
    parser.add_argument("--end", help="Date fin pour une période (YYYY-MM-DD)")
    parser.add_argument("--scrapers", help="Scrapers à lancer (ex: 1,2 ou 'all')", default="all")
    parser.add_argument("--quiet", action="store_true", help="Mode silencieux")
    
    args = parser.parse_args()
    
    # Déterminer date(s)
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
        mode = "period"
    elif args.date:
        target_date = args.date
        mode = "single"
    else:
        target_date = date.today().isoformat()
        mode = "single"
    
    # Déterminer scrapers
    if args.scrapers == "all":
        scrapers = None
    else:
        try:
            scrapers = [int(x) - 1 for x in args.scrapers.split(",")]  # 1-indexed → 0-indexed
        except ValueError:
            print("❌ Format scrapers invalide. Utilisez: 1,2 ou 'all'")
            sys.exit(1)
    
    # Lancer
    orchestrator = ScraperOrchestrator(verbose=not args.quiet)
    
    if mode == "period":
        orchestrator.run_date_range(start_date, end_date, scrapers)
    else:
        orchestrator.run_date(target_date, scrapers)


if __name__ == "__main__":
    main()
