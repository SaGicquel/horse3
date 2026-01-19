#!/usr/bin/env python3
"""
SCRAPING HISTORIQUE PMU - 30 JOURS
Récupère automatiquement les 30 derniers jours de courses PMU.
"""

import sys
from datetime import datetime, timedelta
from scraper_pmu_adapter import PMUToSchemaAdapter


def scrape_last_n_days(n_days=30):
    """Scrape les N derniers jours."""
    print(f"\n🏇 SCRAPING HISTORIQUE - {n_days} DERNIERS JOURS")
    print("=" * 70)

    adapter = PMUToSchemaAdapter()
    adapter.connect_db()

    # Calculer les dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)

    print(f"📅 Période : {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"⏱️  Démarrage : {datetime.now().strftime('%H:%M:%S')}\n")

    try:
        current_date = start_date
        day_count = 0

        while current_date <= end_date:
            day_count += 1
            date_iso = current_date.strftime("%Y-%m-%d")

            print(f"\n{'='*70}")
            print(f"📆 JOUR {day_count}/{n_days+1} : {date_iso}")
            print(f"{'='*70}")

            try:
                adapter.scrape_date(date_iso)
            except KeyboardInterrupt:
                print("\n\n⚠️  Interruption utilisateur")
                break
            except Exception as e:
                print(f"❌ Erreur jour {date_iso}: {e}")

            current_date += timedelta(days=1)

        # Afficher stats finales
        adapter.show_stats()

        print(f"\n⏱️  Fin scraping : {datetime.now().strftime('%H:%M:%S')}")
        print(f"\n✅ Scraping historique terminé : {day_count} jours traités")

        # Calculer les statistiques automatiquement
        print("\n" + "=" * 70)
        print("📊 CALCUL DES STATISTIQUES")
        print("=" * 70)

        import subprocess

        result = subprocess.run(
            ["python", "calcul_stats.py", "--all"], capture_output=True, text=True
        )

        if result.returncode == 0:
            print("✅ Statistiques calculées avec succès")
            print(result.stdout)
        else:
            print("⚠️  Erreur calcul statistiques:")
            print(result.stderr)

        print(f"\n⏱️  Fin totale : {datetime.now().strftime('%H:%M:%S')}")

    except Exception as e:
        print(f"\n❌ Erreur globale : {e}")
        import traceback

        traceback.print_exc()
    finally:
        adapter.close_db()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scraping historique PMU")
    parser.add_argument(
        "--days", type=int, default=30, help="Nombre de jours à scraper (défaut: 30)"
    )

    args = parser.parse_args()
    scrape_last_n_days(args.days)


if __name__ == "__main__":
    main()
