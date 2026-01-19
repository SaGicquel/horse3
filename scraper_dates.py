#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scraper PMU - Dates multiples
Permet de scraper plusieurs jours de courses en spécifiant:
- Une date précise
- Une plage de dates (date_debut -> date_fin)
- Une liste de dates

Usage:
    python scraper_dates.py 2024-01-15                           # Une seule date
    python scraper_dates.py 2024-01-15 2024-01-20               # Plage de dates
    python scraper_dates.py 2024-01-15,2024-01-20,2024-01-25    # Liste de dates
"""

from scraper_pmu_simple import *
from datetime import date, datetime, timedelta
import sys
import argparse
import time


def parse_date(date_str):
    """Parse une date au format YYYY-MM-DD"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"Format de date invalide: {date_str}. Utilisez YYYY-MM-DD (ex: 2024-01-15)"
        )


def generate_date_range(start_date, end_date):
    """Génère toutes les dates entre start_date et end_date (inclus)"""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def scrape_single_date(con, cur, date_iso):
    """Scrape une seule date"""

    print("=" * 80)
    print(f"🏇 SCRAPER PMU - COURSES DU {date_iso}")
    print("=" * 80)
    print()

    # Statistiques avant scraping
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux_avant = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_courses_avant = cur.fetchone()[0]

    # Découverte et scraping des réunions
    print(f"🔍 Recherche des réunions du {date_iso}...")
    reunions = discover_reunions(date_iso)

    if not reunions:
        print(f"❌ Aucune course trouvée pour {date_iso}")
        return {"date": date_iso, "success": False, "reunions": 0, "courses": 0, "chevaux": 0}

    print(f"✅ {len(reunions)} réunion(s) trouvée(s): R{', R'.join(map(str, reunions))}")
    print()

    # Scraping de chaque réunion
    total_courses = 0
    total_chevaux_scraped = 0

    for r in reunions:
        courses = discover_courses(date_iso, r)
        if not courses:
            continue

        print(f"📍 RÉUNION R{r} - {len(courses)} course(s)")

        for c in courses:
            try:
                print(f"   🏁 Course C{c}...", end=" ", flush=True)

                # Scraping de la course
                enrich_from_course(cur, date_iso, r, c, sleep_s=0.5)
                con.commit()

                # Compter les chevaux de cette course
                plist = fetch_participants(date_iso, r, c)
                nb_participants = len(plist) if plist else 0
                total_chevaux_scraped += nb_participants
                total_courses += 1

                print(f"✓ {nb_participants} chevaux")

            except requests.HTTPError as e:
                print(f"❌ Erreur HTTP: {e}")
            except Exception as e:
                print(f"❌ Erreur: {e}")

        print()

    # Recalcul des totaux
    print("🔄 Recalcul des statistiques...")
    recalc_totals_from_seen(con)

    # Statistiques après scraping
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux_apres = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_courses_apres = cur.fetchone()[0]

    # Récapitulatif
    nouveaux_chevaux = nb_chevaux_apres - nb_chevaux_avant
    nouvelles_courses = nb_courses_apres - nb_courses_avant

    print()
    print("=" * 80)
    print(f"📊 RÉCAPITULATIF - {date_iso}")
    print("=" * 80)
    print(f"   • Réunions scrapées: {len(reunions)}")
    print(f"   • Courses scrapées: {total_courses}")
    print(f"   • Chevaux traités: {total_chevaux_scraped}")
    print(f"   • Nouveaux chevaux en base: {nouveaux_chevaux}")
    print(f"   • Nouvelles participations: {nouvelles_courses}")
    print(f"   • Total chevaux en base: {nb_chevaux_apres}")
    print("=" * 80)
    print()

    return {
        "date": date_iso,
        "success": True,
        "reunions": len(reunions),
        "courses": total_courses,
        "chevaux": total_chevaux_scraped,
        "nouveaux_chevaux": nouveaux_chevaux,
        "nouvelles_courses": nouvelles_courses,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scraper PMU pour des dates multiples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  %(prog)s 2024-01-15                           # Une seule date
  %(prog)s 2024-01-15 2024-01-20               # Plage de dates (inclus)
  %(prog)s 2024-01-15,2024-01-20,2024-01-25    # Liste de dates séparées par virgule
  %(prog)s --last-week                          # 7 derniers jours
  %(prog)s --last-month                         # 30 derniers jours
        """,
    )

    parser.add_argument("dates", nargs="*", help="Date(s) au format YYYY-MM-DD")
    parser.add_argument("--last-week", action="store_true", help="Scraper les 7 derniers jours")
    parser.add_argument("--last-month", action="store_true", help="Scraper les 30 derniers jours")

    args = parser.parse_args()

    # Déterminer les dates à scraper
    dates_to_scrape = []

    if args.last_week:
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        dates_to_scrape = generate_date_range(start_date, end_date)
        print(f"🗓️  Mode: 7 derniers jours ({start_date} → {end_date})")

    elif args.last_month:
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        dates_to_scrape = generate_date_range(start_date, end_date)
        print(f"🗓️  Mode: 30 derniers jours ({start_date} → {end_date})")

    elif not args.dates:
        parser.print_help()
        sys.exit(1)

    elif len(args.dates) == 1:
        # Vérifier si c'est une liste séparée par des virgules
        if "," in args.dates[0]:
            date_strings = args.dates[0].split(",")
            dates_to_scrape = [parse_date(d.strip()) for d in date_strings]
            print(f"🗓️  Mode: Liste de {len(dates_to_scrape)} dates")
        else:
            # Une seule date
            dates_to_scrape = [parse_date(args.dates[0])]
            print("🗓️  Mode: Date unique")

    elif len(args.dates) == 2:
        # Plage de dates
        start = parse_date(args.dates[0])
        end = parse_date(args.dates[1])
        if start > end:
            print("❌ Erreur: La date de début doit être avant la date de fin")
            sys.exit(1)
        dates_to_scrape = generate_date_range(start, end)
        print(f"🗓️  Mode: Plage de dates ({start} → {end}, {len(dates_to_scrape)} jours)")

    else:
        print("❌ Erreur: Format invalide. Utilisez --help pour voir les exemples")
        sys.exit(1)

    print(f"📅 {len(dates_to_scrape)} date(s) à scraper")
    print()

    # Connexion à la base de données
    con = sqlite3.connect(DB_PATH)
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
            print(f"\n{'='*80}")
            print(f"📆 DATE {i}/{len(dates_to_scrape)}: {date_iso}")
            print(f"{'='*80}\n")

            result = scrape_single_date(con, cur, date_iso)
            results.append(result)

            if result["success"]:
                total_reunions += result["reunions"]
                total_courses += result["courses"]
                total_chevaux += result["chevaux"]

            # Petit délai entre les dates pour ne pas surcharger l'API
            if i < len(dates_to_scrape):
                time.sleep(2)

    finally:
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

    # Détail par date
    if len(results) > 1:
        print("\n📊 Détail par date:")
        print("-" * 80)
        for r in results:
            status = "✅" if r["success"] else "❌"
            if r["success"]:
                print(
                    f"  {status} {r['date']}: {r['reunions']} réunions, {r['courses']} courses, {r['chevaux']} chevaux"
                )
            else:
                print(f"  {status} {r['date']}: Aucune course")
        print("-" * 80)


if __name__ == "__main__":
    main()
