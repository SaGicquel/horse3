#!/usr/bin/env python3
"""
ENRICHISSEMENT HISTORIQUE ZONE-TURF

Télécharge et enrichit automatiquement les N derniers jours depuis Zone-Turf.

Usage:
    python enrichir_historique_zoneturf.py --days 30
    python enrichir_historique_zoneturf.py --days 7
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

from enrichir_zoneturf import EnrichisseurZoneTurf


def main():
    parser = argparse.ArgumentParser(description="Enrichissement historique Zone-Turf")
    parser.add_argument(
        "--days", type=int, default=30, help="Nombre de jours à enrichir (défaut: 30)"
    )
    parser.add_argument("--start-date", type=str, help="Date de départ optionnelle (YYYY-MM-DD)")

    args = parser.parse_args()

    # Calculer la plage de dates
    if args.start_date:
        end_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=args.days - 1)

    print("\n" + "=" * 70)
    print("🏇 ENRICHISSEMENT HISTORIQUE ZONE-TURF")
    print("=" * 70)
    print(f"📅 Période : {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"⏱️  Démarrage : {datetime.now().strftime('%H:%M:%S')}")
    print("\n")

    # Créer l'enrichisseur
    enrichisseur = EnrichisseurZoneTurf()
    enrichisseur.connect_db()

    stats_globales = {
        "jours_traites": 0,
        "jours_reussis": 0,
        "jours_sans_donnees": 0,
        "jours_erreur": 0,
        "courses_enrichies": 0,
        "performances_enrichies": 0,
    }

    try:
        current_date = start_date
        jour_num = 1

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")

            print("=" * 70)
            print(f"📆 JOUR {jour_num}/{args.days} : {date_str}")
            print("=" * 70)
            print()

            # Enrichir cette date
            try:
                enrichisseur.enrich_date(date_str)

                stats_globales["jours_traites"] += 1

                if enrichisseur.stats["courses_enrichies"] > 0:
                    stats_globales["jours_reussis"] += 1
                    stats_globales["courses_enrichies"] += enrichisseur.stats["courses_enrichies"]
                    stats_globales["performances_enrichies"] += enrichisseur.stats[
                        "performances_enrichies"
                    ]
                else:
                    stats_globales["jours_sans_donnees"] += 1
                    print(f"   ⚠️  Pas de données pour {date_str}")

                # Réinitialiser les stats pour le prochain jour
                enrichisseur.stats = {
                    "courses_enrichies": 0,
                    "performances_enrichies": 0,
                    "temps_ajoutes": 0,
                    "ecarts_ajoutes": 0,
                    "musiques_ajoutees": 0,
                    "cotes_ajoutees": 0,
                    "courses_introuvables": 0,
                    "chevaux_introuvables": 0,
                }

            except Exception as e:
                print(f"   ❌ Erreur pour {date_str} : {e}")
                stats_globales["jours_erreur"] += 1

            print()

            current_date += timedelta(days=1)
            jour_num += 1

    finally:
        enrichisseur.close_db()

    # Afficher les stats globales
    print("\n" + "=" * 70)
    print("📊 STATISTIQUES GLOBALES")
    print("=" * 70)
    print(f"   jours_traites             : {stats_globales['jours_traites']:6d}")
    print(f"   jours_reussis             : {stats_globales['jours_reussis']:6d}")
    print(f"   jours_sans_donnees        : {stats_globales['jours_sans_donnees']:6d}")
    print(f"   jours_erreur              : {stats_globales['jours_erreur']:6d}")
    print(f"   courses_enrichies         : {stats_globales['courses_enrichies']:6d}")
    print(f"   performances_enrichies    : {stats_globales['performances_enrichies']:6d}")
    print("=" * 70)

    print(f"\n⏱️  Fin : {datetime.now().strftime('%H:%M:%S')}")
    print("\n✅ Enrichissement historique terminé !\n")


if __name__ == "__main__":
    main()
