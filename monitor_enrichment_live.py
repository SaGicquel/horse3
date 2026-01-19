#!/usr/bin/env python3
"""
Script de monitoring en temps réel de l'enrichissement Phase 2A.
Affiche la progression minute par minute.
"""

from db_connection import get_connection
import time
import sys


def get_stats():
    """Récupère les statistiques d'enrichissement."""
    conn = get_connection()
    cur = conn.cursor()

    # Total et dates
    cur.execute("""
        SELECT
            COUNT(DISTINCT c.id_course) as nb_courses,
            COUNT(p.id_performance) as nb_perfs,
            COUNT(CASE WHEN p.musique IS NOT NULL THEN 1 END) as musique_ok,
            COUNT(CASE WHEN p.temps_total IS NOT NULL THEN 1 END) as temps_ok,
            MIN(SUBSTRING(c.id_course, 1, 8)) as date_min,
            MAX(SUBSTRING(c.id_course, 1, 8)) as date_max
        FROM performances p
        JOIN courses c ON p.id_course = c.id_course
        WHERE SUBSTRING(c.id_course, 1, 4) = '2025'
    """)

    courses, perfs, musique, temps, date_min, date_max = cur.fetchone()

    # Stats 2024 (ancien)
    cur.execute("""
        SELECT
            COUNT(p.id_performance) as nb_perfs,
            COUNT(CASE WHEN p.musique IS NOT NULL THEN 1 END) as musique_ok
        FROM performances p
        JOIN courses c ON p.id_course = c.id_course
        WHERE SUBSTRING(c.id_course, 1, 4) = '2024'
    """)
    perfs_2024, mus_2024 = cur.fetchone()

    cur.close()
    conn.close()

    return {
        "courses": courses,
        "perfs": perfs,
        "musique": musique,
        "temps": temps,
        "date_min": date_min,
        "date_max": date_max,
        "perfs_2024": perfs_2024,
        "mus_2024": mus_2024,
    }


def display_progress(stats, previous=None):
    """Affiche les statistiques avec indication de progression."""
    print("\033[2J\033[H")  # Clear screen
    print("=" * 90)
    print("🔄 MONITORING ENRICHISSEMENT PHASE 2A - EN TEMPS RÉEL")
    print("=" * 90)
    print()

    # Infos générales
    print(f"📅 Période  : {stats['date_min']} → {stats['date_max']}")
    print(f"📊 Courses  : {stats['courses']:,}")
    print(f"🏇 Perfs    : {stats['perfs']:,}")
    print()

    # Enrichissement 2025 (données cibles)
    musique_pct = (stats["musique"] / stats["perfs"] * 100) if stats["perfs"] > 0 else 0
    temps_pct = (stats["temps"] / stats["perfs"] * 100) if stats["perfs"] > 0 else 0

    print("🎯 ENRICHISSEMENT 2025 (données actuelles)")
    print("-" * 90)
    print(f"   Musique : {stats['musique']:6,} / {stats['perfs']:6,} ({musique_pct:6.2f}%)", end="")
    if previous and previous["musique"] < stats["musique"]:
        delta = stats["musique"] - previous["musique"]
        print(f"  ⬆️ +{delta:,}", end="")
    print()

    print(f"   Temps   : {stats['temps']:6,} / {stats['perfs']:6,} ({temps_pct:6.2f}%)", end="")
    if previous and previous["temps"] < stats["temps"]:
        delta = stats["temps"] - previous["temps"]
        print(f"  ⬆️ +{delta:,}", end="")
    print()
    print()

    # Barre de progression
    bar_length = 60
    musique_bar = int(musique_pct / 100 * bar_length)
    temps_bar = int(temps_pct / 100 * bar_length)

    print("   Musique : [" + "█" * musique_bar + "░" * (bar_length - musique_bar) + "]")
    print("   Temps   : [" + "█" * temps_bar + "░" * (bar_length - temps_bar) + "]")
    print()

    # Données 2024 (référence)
    if stats["perfs_2024"] > 0:
        mus_2024_pct = stats["mus_2024"] / stats["perfs_2024"] * 100
        print("📌 RÉFÉRENCE 2024 (ancien test)")
        print("-" * 90)
        print(
            f"   Musique : {stats['mus_2024']:6,} / {stats['perfs_2024']:6,} ({mus_2024_pct:6.2f}%)"
        )
        print()

    # Objectifs
    objectif_musique = 95
    objectif_temps = 60

    print("🎯 OBJECTIFS PHASE 2A")
    print("-" * 90)

    if musique_pct >= objectif_musique:
        print(f"   ✅ Musique : {objectif_musique}% atteint ({musique_pct:.1f}%)")
    else:
        manque_mus = int((objectif_musique - musique_pct) / 100 * stats["perfs"])
        print(f"   ⏳ Musique : {objectif_musique}% - Manque {manque_mus:,} perfs")

    if temps_pct >= objectif_temps:
        print(f"   ✅ Temps   : {objectif_temps}% atteint ({temps_pct:.1f}%)")
    else:
        manque_tps = int((objectif_temps - temps_pct) / 100 * stats["perfs"])
        print(f"   ⏳ Temps   : {objectif_temps}% - Manque {manque_tps:,} perfs")

    print()
    print("=" * 90)
    print("⏳ Rafraîchissement dans 10s... (Ctrl+C pour arrêter)")
    print()


if __name__ == "__main__":
    print("🚀 Démarrage du monitoring...\n")

    try:
        previous_stats = None
        while True:
            stats = get_stats()
            display_progress(stats, previous_stats)
            previous_stats = stats
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring arrêté.\n")

        # Afficher un résumé final
        final_stats = get_stats()
        musique_pct = (
            (final_stats["musique"] / final_stats["perfs"] * 100) if final_stats["perfs"] > 0 else 0
        )
        temps_pct = (
            (final_stats["temps"] / final_stats["perfs"] * 100) if final_stats["perfs"] > 0 else 0
        )

        print("=" * 90)
        print("📊 RÉSUMÉ FINAL")
        print("=" * 90)
        print(
            f"Musique : {final_stats['musique']:,} / {final_stats['perfs']:,} ({musique_pct:.2f}%)"
        )
        print(f"Temps   : {final_stats['temps']:,} / {final_stats['perfs']:,} ({temps_pct:.2f}%)")
        print("=" * 90)
        print()
