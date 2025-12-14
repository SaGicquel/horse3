#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scraper PMU Interactif
Interface simple pour scraper des courses PMU

Mis à jour : Utilise PostgreSQL via db_connection.py
"""

from scraper_dates import (
    parse_date, generate_date_range, scrape_single_date
)
from scraper_pmu_simple import db_setup
from db_connection import get_connection
from datetime import date, timedelta
import time
import sys

def print_menu():
    """Affiche le menu principal"""
    print("\n" + "=" * 80)
    print("🏇 SCRAPER PMU - MODE INTERACTIF")
    print("=" * 80)
    print("\nChoisissez une option:")
    print()
    print("  1️⃣  Scraper AUJOURD'HUI")
    print("  2️⃣  Scraper HIER")
    print("  3️⃣  Scraper les 7 DERNIERS JOURS")
    print("  4️⃣  Scraper les 30 DERNIERS JOURS")
    print("  5️⃣  Scraper une DATE PRÉCISE")
    print("  6️⃣  Scraper une PLAGE DE DATES")
    print("  7️⃣  Scraper plusieurs DATES SPÉCIFIQUES")
    print("  0️⃣  QUITTER")
    print()
    print("=" * 80)

def get_user_choice():
    """Demande le choix de l'utilisateur"""
    while True:
        try:
            choice = input("\n👉 Votre choix (0-7): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7']:
                return choice
            print("❌ Choix invalide. Veuillez entrer un nombre entre 0 et 7.")
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir !")
            sys.exit(0)

def get_date_input(prompt="Date (YYYY-MM-DD): "):
    """Demande une date à l'utilisateur"""
    while True:
        try:
            date_str = input(prompt).strip()
            return parse_date(date_str)
        except ValueError as e:
            print(f"❌ {e}")
        except KeyboardInterrupt:
            print("\n\n👋 Opération annulée")
            sys.exit(0)

def confirm_action(message):
    """Demande confirmation à l'utilisateur"""
    while True:
        response = input(f"\n{message} (o/n): ").strip().lower()
        if response in ['o', 'oui', 'y', 'yes']:
            return True
        elif response in ['n', 'non', 'no']:
            return False
        print("❌ Veuillez répondre par 'o' (oui) ou 'n' (non)")

def main():
    """Programme principal interactif"""
    
    print("\n" + "🐴" * 40)
    print("Bienvenue dans le scraper PMU interactif !")
    print("🐴" * 40)
    
    while True:
        print_menu()
        choice = get_user_choice()
        
        dates_to_scrape = []
        
        if choice == '0':
            print("\n👋 Au revoir !")
            break
        
        elif choice == '1':
            # Aujourd'hui
            today = date.today()
            dates_to_scrape = [today]
            print(f"\n📅 Scraping des courses d'aujourd'hui ({today})")
        
        elif choice == '2':
            # Hier
            yesterday = date.today() - timedelta(days=1)
            dates_to_scrape = [yesterday]
            print(f"\n📅 Scraping des courses d'hier ({yesterday})")
        
        elif choice == '3':
            # 7 derniers jours
            end_date = date.today()
            start_date = end_date - timedelta(days=7)
            dates_to_scrape = generate_date_range(start_date, end_date)
            print(f"\n📅 Scraping des 7 derniers jours ({start_date} → {end_date})")
            print(f"   Cela représente {len(dates_to_scrape)} jours")
        
        elif choice == '4':
            # 30 derniers jours
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            dates_to_scrape = generate_date_range(start_date, end_date)
            print(f"\n📅 Scraping des 30 derniers jours ({start_date} → {end_date})")
            print(f"   Cela représente {len(dates_to_scrape)} jours")
            print("   ⚠️  Attention: Cela peut prendre plusieurs minutes")
        
        elif choice == '5':
            # Date précise
            print("\n📅 Scraping d'une date précise")
            print("   Format: YYYY-MM-DD (ex: 2024-01-15)")
            selected_date = get_date_input("\n👉 ")
            dates_to_scrape = [selected_date]
            print(f"\n✅ Date sélectionnée: {selected_date}")
        
        elif choice == '6':
            # Plage de dates
            print("\n📅 Scraping d'une plage de dates")
            print("   Format: YYYY-MM-DD (ex: 2024-01-15)")
            print()
            start = get_date_input("👉 Date de début: ")
            end = get_date_input("👉 Date de fin: ")
            
            if start > end:
                print("❌ Erreur: La date de début doit être avant la date de fin")
                continue
            
            dates_to_scrape = generate_date_range(start, end)
            print(f"\n✅ Plage sélectionnée: {start} → {end}")
            print(f"   Cela représente {len(dates_to_scrape)} jours")
        
        elif choice == '7':
            # Dates spécifiques
            print("\n📅 Scraping de dates spécifiques")
            print("   Format: YYYY-MM-DD,YYYY-MM-DD,... (ex: 2024-01-15,2024-01-20,2024-01-25)")
            print()
            dates_input = input("👉 Entrez les dates séparées par des virgules: ").strip()
            
            try:
                date_strings = dates_input.split(',')
                dates_to_scrape = [parse_date(d.strip()) for d in date_strings]
                print(f"\n✅ {len(dates_to_scrape)} date(s) sélectionnée(s):")
                for d in dates_to_scrape:
                    print(f"   • {d}")
            except ValueError as e:
                print(f"❌ Erreur: {e}")
                continue
        
        # Confirmation avant de lancer
        if not confirm_action(f"\n🚀 Lancer le scraping de {len(dates_to_scrape)} date(s) ?"):
            print("❌ Opération annulée")
            continue
        
        print("\n" + "🏁" * 40)
        print("Démarrage du scraping...")
        print("🏁" * 40 + "\n")
        
        # Connexion à la base de données PostgreSQL
        con = get_connection()
        db_setup(con)
        cur = con.cursor()
        
        # Statistiques globales
        results = []
        total_reunions = 0
        total_courses = 0
        total_chevaux = 0
        
        try:
            for i, d in enumerate(dates_to_scrape, 1):
                date_iso = d.isoformat()
                
                result = scrape_single_date(con, cur, date_iso)
                results.append(result)
                
                if result['success']:
                    total_reunions += result['reunions']
                    total_courses += result['courses']
                    total_chevaux += result['chevaux']
                
                # Petit délai entre les dates
                if i < len(dates_to_scrape):
                    time.sleep(2)
        
        finally:
            cur.close()
            con.close()
        
        # Récapitulatif global
        print("\n" + "=" * 80)
        print("🏆 RÉCAPITULATIF GLOBAL")
        print("=" * 80)
        print(f"📅 Dates scrapées: {len(dates_to_scrape)}")
        print(f"✅ Dates avec courses: {sum(1 for r in results if r['success'])}")
        print(f"❌ Dates sans courses: {sum(1 for r in results if not r['success'])}")
        print(f"📍 Total réunions: {total_reunions}")
        print(f"🏁 Total courses: {total_courses}")
        print(f"🐴 Total chevaux traités: {total_chevaux}")
        print("=" * 80)
        
        # Pause avant de revenir au menu
        input("\n✨ Appuyez sur Entrée pour revenir au menu...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Programme interrompu. Au revoir !")
        sys.exit(0)
