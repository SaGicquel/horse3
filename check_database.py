#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de vérification rapide de la base de données
Affiche les statistiques essentielles pour valider le scraping
"""

from db_connection import get_connection


def check_database_stats():
    """Affiche les statistiques de la base de données"""

    print("\n" + "=" * 80)
    print("📊 STATISTIQUES BASE DE DONNÉES")
    print("=" * 80 + "\n")

    conn = get_connection()
    cur = conn.cursor()

    # Nombre de chevaux
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux = cur.fetchone()[0]
    print(f"🐴 Chevaux enregistrés: {nb_chevaux:,}")

    # Nombre de courses vues
    cur.execute("SELECT COUNT(DISTINCT race_key) FROM cheval_courses_seen")
    nb_courses = cur.fetchone()[0]
    print(f"🏁 Courses enregistrées: {nb_courses:,}")

    # Nombre de participations
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_participations = cur.fetchone()[0]
    print(f"📋 Participations totales: {nb_participations:,}")

    # Nombre d'entraîneurs
    cur.execute("SELECT COUNT(*) FROM entraineurs")
    nb_entraineurs = cur.fetchone()[0]
    print(f"👔 Entraîneurs: {nb_entraineurs:,}")

    # Nombre de jockeys/drivers
    cur.execute("SELECT COUNT(*) FROM drivers")
    nb_drivers = cur.fetchone()[0]
    print(f"🏇 Jockeys/Drivers: {nb_drivers:,}")

    # Nombre de propriétaires
    cur.execute("SELECT COUNT(*) FROM proprietaires")
    nb_proprietaires = cur.fetchone()[0]
    print(f"💼 Propriétaires: {nb_proprietaires:,}")

    print("\n" + "-" * 80)
    print("📅 DONNÉES PAR DATE")
    print("-" * 80 + "\n")

    # Top 10 dates avec le plus de courses
    cur.execute("""
        SELECT
            LEFT(race_key, 10) as date,
            COUNT(DISTINCT race_key) as nb_courses,
            COUNT(*) as nb_participations
        FROM cheval_courses_seen
        GROUP BY LEFT(race_key, 10)
        ORDER BY date DESC
        LIMIT 10
    """)

    print(f"{'Date':<12} {'Courses':<10} {'Participations':<15}")
    print("-" * 40)
    for row in cur.fetchall():
        date, nb_courses_date, nb_part = row
        print(f"{date:<12} {nb_courses_date:<10} {nb_part:<15}")

    print("\n" + "-" * 80)
    print("🎯 QUALITÉ DES DONNÉES")
    print("-" * 80 + "\n")

    # Participations avec cote finale
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE cote_finale IS NOT NULL) as with_cote,
            COUNT(*) as total,
            ROUND(100.0 * COUNT(*) FILTER (WHERE cote_finale IS NOT NULL) / COUNT(*), 1) as pct
        FROM cheval_courses_seen
    """)
    with_cote, total, pct = cur.fetchone()
    print(f"💰 Avec cote finale: {with_cote:,} / {total:,} ({pct}%)")

    # Participations avec gains carrière
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE gains_carriere IS NOT NULL AND gains_carriere > 0) as with_gains,
            COUNT(*) as total,
            ROUND(100.0 * COUNT(*) FILTER (WHERE gains_carriere IS NOT NULL AND gains_carriere > 0) / COUNT(*), 1) as pct
        FROM cheval_courses_seen
    """)
    with_gains, total, pct = cur.fetchone()
    print(f"💸 Avec gains carrière: {with_gains:,} / {total:,} ({pct}%)")

    # Participations avec musique
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE musique IS NOT NULL AND musique != '') as with_musique,
            COUNT(*) as total,
            ROUND(100.0 * COUNT(*) FILTER (WHERE musique IS NOT NULL AND musique != '') / COUNT(*), 1) as pct
        FROM cheval_courses_seen
    """)
    with_musique, total, pct = cur.fetchone()
    print(f"🎵 Avec musique: {with_musique:,} / {total:,} ({pct}%)")

    # Participations avec entraîneur winrate
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE entraineur_winrate_90j IS NOT NULL) as with_winrate,
            COUNT(*) as total,
            ROUND(100.0 * COUNT(*) FILTER (WHERE entraineur_winrate_90j IS NOT NULL) / COUNT(*), 1) as pct
        FROM cheval_courses_seen
    """)
    with_winrate, total, pct = cur.fetchone()
    print(f"📊 Avec winrate entraîneur: {with_winrate:,} / {total:,} ({pct}%)")

    print("\n" + "-" * 80)
    print("🏆 DONNÉES RÉCENTES")
    print("-" * 80 + "\n")

    # Nombre de chevaux avec données pour le jour sélectionné
    cur.execute("""
        SELECT COUNT(DISTINCT nom_norm)
        FROM cheval_courses_seen
        WHERE race_key LIKE '2025-11-04%'
    """)
    nb_chevaux_jour = cur.fetchone()[0]
    print(f"🐴 Chevaux du 04/11/2025: {nb_chevaux_jour}")

    # Courses avec le plus de participants
    cur.execute("""
        SELECT
            race_key,
            COUNT(*) as nb_participants
        FROM cheval_courses_seen
        WHERE race_key LIKE '2025-11-04%'
        GROUP BY race_key
        ORDER BY nb_participants DESC
        LIMIT 5
    """)

    print("\n📋 Top 5 courses (nb participants):\n")
    for race_key, nb_part in cur.fetchall():
        print(f"  • {race_key}: {nb_part} participants")

    print("\n" + "=" * 80 + "\n")

    cur.close()
    conn.close()


if __name__ == "__main__":
    try:
        check_database_stats()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
