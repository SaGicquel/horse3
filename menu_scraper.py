#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Menu interactif pour le scraper PMU multi-threadé
Facilite le choix entre les différents modes et options
"""

import sys
from datetime import date, timedelta


def print_header():
    print("=" * 70)
    print("🏇 SCRAPER PMU - MENU PRINCIPAL")
    print("=" * 70)
    print()


def print_menu():
    print("Choisissez une option:")
    print()
    print("  1. 🚀 Scraper aujourd'hui (MULTI-THREADÉ - RAPIDE)")
    print("  2. 🐢 Scraper aujourd'hui (séquentiel - debug)")
    print("  3. 📊 Benchmark de performance (compare les deux modes)")
    print("  4. 🧪 Test rapide (3 réunions seulement)")
    print("  5. 📅 Scraper une date spécifique")
    print("  6. 📆 Scraper plusieurs dates")
    print("  7. 📈 Voir les statistiques de la base")
    print("  8. 🔍 Vérifier l'intégrité de la base (doublons, anomalies)")
    print("  9. 🔧 Corriger automatiquement les problèmes détectés")
    print("  10. ℹ️  Aide et documentation")
    print("  0. ❌ Quitter")
    print()


def scrape_today(multi_thread=True):
    """Scrape aujourd'hui"""
    from scraper_pmu_simple import run
    import time

    today = date.today().isoformat()
    mode = "MULTI-THREADÉ" if multi_thread else "SÉQUENTIEL"

    print(f"\n🏇 Scraping du {today} ({mode})")
    print("=" * 70)

    start = time.time()
    run(today, recalc_after=True, use_threading=multi_thread)
    elapsed = time.time() - start

    print(f"\n✅ Terminé en {elapsed:.1f}s")

    # Proposer la vérification
    verify = (
        input("\n💡 Voulez-vous vérifier l'intégrité de la base maintenant ? (O/n): ")
        .strip()
        .lower()
    )
    if verify != "n":
        verify_database()
    else:
        input("\n▶️  Appuyez sur Entrée pour continuer...")


def run_benchmark():
    """Lance le benchmark"""
    import subprocess

    print("\n📊 Lancement du benchmark...")
    print("=" * 70)
    subprocess.run([sys.executable, "benchmark_scraper.py"])
    input("\n▶️  Appuyez sur Entrée pour continuer...")


def run_test():
    """Lance le test rapide"""
    import subprocess

    print("\n🧪 Lancement du test rapide...")
    print("=" * 70)
    subprocess.run([sys.executable, "test_multi_thread.py"])
    input("\n▶️  Appuyez sur Entrée pour continuer...")


def scrape_date():
    """Scrape une date spécifique"""
    from scraper_pmu_simple import run
    import time

    print("\n📅 Scraper une date spécifique")
    print("=" * 70)

    date_str = input("\nEntrez la date (YYYY-MM-DD) ou 'h' pour hier: ").strip()

    if date_str.lower() == "h":
        date_iso = (date.today() - timedelta(days=1)).isoformat()
    else:
        date_iso = date_str

    multi_thread = input("Mode multi-threadé ? (O/n): ").strip().lower() != "n"

    print(f"\n🏇 Scraping du {date_iso}")
    print("=" * 70)

    start = time.time()
    run(date_iso, recalc_after=True, use_threading=multi_thread)
    elapsed = time.time() - start

    print(f"\n✅ Terminé en {elapsed:.1f}s")
    input("\n▶️  Appuyez sur Entrée pour continuer...")


def scrape_multiple():
    """Scrape plusieurs dates"""
    import subprocess

    print("\n📆 Scraper plusieurs dates")
    print("=" * 70)
    print()
    print("Options:")
    print("  1. Les 7 derniers jours")
    print("  2. Les 30 derniers jours")
    print("  3. Plage personnalisée")
    print()

    choice = input("Votre choix: ").strip()

    if choice == "1":
        subprocess.run([sys.executable, "scraper_dates.py", "--last-week"])
    elif choice == "2":
        subprocess.run([sys.executable, "scraper_dates.py", "--last-month"])
    elif choice == "3":
        start = input("Date de début (YYYY-MM-DD): ").strip()
        end = input("Date de fin (YYYY-MM-DD): ").strip()
        subprocess.run([sys.executable, "scraper_dates.py", start, end])

    input("\n▶️  Appuyez sur Entrée pour continuer...")


def show_stats():
    """Affiche les statistiques de la base"""
    from db_connection import get_connection

    print("\n📈 Statistiques de la base de données")
    print("=" * 70)

    con = get_connection()
    cur = con.cursor()

    # Stats générales
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_courses = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT race_key) FROM cheval_courses_seen")
    nb_courses_unique = cur.fetchone()[0]

    # Stats 2025
    cur.execute(
        "SELECT COALESCE(SUM(nombre_courses_2025), 0), COALESCE(SUM(nombre_victoires_2025), 0) FROM chevaux"
    )
    courses_2025, victoires_2025 = cur.fetchone()

    # Dernière course
    cur.execute("""
        SELECT race_key FROM cheval_courses_seen
        ORDER BY race_key DESC LIMIT 1
    """)
    last_race = cur.fetchone()

    cur.close()
    con.close()

    print()
    print(f"🐴 Nombre total de chevaux: {nb_chevaux:,}")
    print(f"🏁 Courses enregistrées: {nb_courses:,}")
    print(f"📊 Courses uniques: {nb_courses_unique:,}")
    print()
    print(f"📅 Courses 2025: {courses_2025 or 0:,}")
    print(f"🏆 Victoires 2025: {victoires_2025 or 0:,}")
    print()
    if last_race:
        print(f"🕐 Dernière course: {last_race[0]}")
    print()

    input("▶️  Appuyez sur Entrée pour continuer...")


def verify_database():
    """Lance la vérification de l'intégrité de la base"""
    import subprocess

    print("\n🔍 Vérification de l'intégrité de la base...")
    print("=" * 70)
    result = subprocess.run([sys.executable, "verify_database.py"])

    if result.returncode == 0:
        print("\n✅ Base de données vérifiée avec succès")
    else:
        print("\n⚠️  Des problèmes ont été détectés")
        fix = input("\n💡 Voulez-vous corriger automatiquement ? (o/N): ").strip().lower()
        if fix == "o":
            fix_database()
            return

    input("\n▶️  Appuyez sur Entrée pour continuer...")


def fix_database():
    """Lance la correction automatique"""
    import subprocess

    print("\n🔧 Correction automatique des problèmes...")
    print("=" * 70)
    subprocess.run([sys.executable, "fix_database.py"])
    input("\n▶️  Appuyez sur Entrée pour continuer...")


def show_help():
    """Affiche l'aide"""
    print("\nℹ️  AIDE ET DOCUMENTATION")
    print("=" * 70)
    print()
    print("📚 Documentation disponible:")
    print()
    print("  • MULTI_THREADING_README.md")
    print("    → Guide complet du multi-threading")
    print()
    print("  • OPTIMISATION_SUMMARY.md")
    print("    → Résumé des optimisations")
    print()
    print("  • scraper_config.ini")
    print("    → Configuration des paramètres")
    print()
    print("🎯 Performances attendues:")
    print()
    print("  Mode séquentiel:    ~100s pour 90 courses")
    print("  Mode multi-threadé:  ~16s pour 90 courses")
    print("  Gain moyen:          6x plus rapide")
    print()
    print("⚙️  Configuration:")
    print()
    print("  Threads parallèles: 8 (modifiable dans scraper_pmu_simple.py)")
    print("  Délai entre requêtes: 0.1s")
    print()
    print("💡 Commandes directes:")
    print()
    print("  python scraper_pmu_simple.py           # Scrape aujourd'hui")
    print("  python scraper_pmu_simple.py --no-threads  # Mode debug")
    print("  python benchmark_scraper.py            # Benchmark")
    print("  python scraper_dates.py --last-week    # 7 derniers jours")
    print()

    input("▶️  Appuyez sur Entrée pour continuer...")


def main():
    while True:
        print("\033[2J\033[H")  # Clear screen
        print_header()
        print_menu()

        choice = input("Votre choix: ").strip()

        if choice == "1":
            scrape_today(multi_thread=True)
        elif choice == "2":
            scrape_today(multi_thread=False)
        elif choice == "3":
            run_benchmark()
        elif choice == "4":
            run_test()
        elif choice == "5":
            scrape_date()
        elif choice == "6":
            scrape_multiple()
        elif choice == "7":
            show_stats()
        elif choice == "8":
            verify_database()
        elif choice == "9":
            fix_database()
        elif choice == "10":
            show_help()
        elif choice == "0":
            print("\n👋 Au revoir !")
            break
        else:
            print("\n❌ Choix invalide")
            input("▶️  Appuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Au revoir !")
        sys.exit(0)
