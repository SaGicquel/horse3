#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de monitoring pour suivre l'avancement du scraping d'octobre
"""

import psycopg2
import psycopg2.extras
from datetime import datetime


def connect_db():
    """Connexion à la base"""
    return psycopg2.connect(
        host="localhost",
        port=54624,
        database="pmubdd",
        user="postgres",
        password="okokok",
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def check_october_data():
    """Vérifie les données enrichies pour octobre 2024"""

    conn = connect_db()
    cur = conn.cursor()

    print("=" * 80)
    print("📊 MONITORING SCRAPING OCTOBRE 2024")
    print("=" * 80)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Nombre total de participations octobre
    cur.execute("""
        SELECT COUNT(*) as total
        FROM cheval_courses_seen
        WHERE race_key LIKE '2024-10-%'
    """)
    total = cur.fetchone()["total"]
    print(f"📋 Total participations octobre 2024: {total:,}")

    # 2. Nombre de participations par jour
    cur.execute("""
        SELECT
            SUBSTRING(race_key, 1, 10) as date_course,
            COUNT(*) as nb_participations
        FROM cheval_courses_seen
        WHERE race_key LIKE '2024-10-%'
        GROUP BY SUBSTRING(race_key, 1, 10)
        ORDER BY SUBSTRING(race_key, 1, 10)
    """)
    days = cur.fetchall()
    print(f"\n📅 Jours traités: {len(days)}/31 jours")

    if days:
        print(f"   Premier jour: {days[0]['date_course']}")
        print(f"   Dernier jour: {days[-1]['date_course']}")

    # 3. Vérifier l'enrichissement par scraper
    print("\n🔍 ÉTAT DE L'ENRICHISSEMENT PAR SCRAPER:\n")

    # SCRAPER 1 - Métadonnées (course_id)
    cur.execute("""
        SELECT COUNT(*) as enrichi
        FROM cheval_courses_seen
        WHERE race_key LIKE '2024-10-%'
        AND course_id IS NOT NULL
    """)
    meta = cur.fetchone()["enrichi"]
    print(
        f"✅ SCRAPER 1 (Métadonnées):     {meta:,}/{total:,} ({meta*100//total if total > 0 else 0}%)"
    )

    # SCRAPER 2 - Détails cheval (handicap_distance déjà présent dans base)
    cur.execute("""
        SELECT COUNT(*) as enrichi
        FROM cheval_courses_seen
        WHERE race_key LIKE '2024-10-%'
        AND handicap_distance IS NOT NULL
    """)
    details = cur.fetchone()["enrichi"]
    print(
        f"✅ SCRAPER 2 (Détails cheval):  {details:,}/{total:,} ({details*100//total if total > 0 else 0}%)"
    )

    # SCRAPER 3 - Statistiques (nb_places_top3_12m)
    cur.execute("""
        SELECT COUNT(*) as enrichi
        FROM cheval_courses_seen
        WHERE race_key LIKE '2024-10-%'
        AND nb_places_top3_12m IS NOT NULL
    """)
    stats = cur.fetchone()["enrichi"]
    print(
        f"✅ SCRAPER 3 (Statistiques):    {stats:,}/{total:,} ({stats*100//total if total > 0 else 0}%)"
    )

    # SCRAPER 4 - Conditions (biais_stalle)
    cur.execute("""
        SELECT COUNT(*) as enrichi
        FROM cheval_courses_seen
        WHERE race_key LIKE '2024-10-%'
        AND biais_stalle IS NOT NULL
    """)
    conditions = cur.fetchone()["enrichi"]
    print(
        f"✅ SCRAPER 4 (Conditions jour):  {conditions:,}/{total:,} ({conditions*100//total if total > 0 else 0}%)"
    )

    # SCRAPER 5 - Cotes (cote_finale)
    cur.execute("""
        SELECT COUNT(*) as enrichi
        FROM cheval_courses_seen
        WHERE race_key LIKE '2024-10-%'
        AND cote_finale IS NOT NULL
    """)
    cotes = cur.fetchone()["enrichi"]
    print(
        f"✅ SCRAPER 5 (Cotes marché):    {cotes:,}/{total:,} ({cotes*100//total if total > 0 else 0}%)"
    )

    # SCRAPER 7 - ML (score_composite)
    cur.execute("""
        SELECT COUNT(*) as enrichi
        FROM cheval_courses_seen
        WHERE race_key LIKE '2024-10-%'
        AND score_composite IS NOT NULL
    """)
    ml = cur.fetchone()["enrichi"]
    print(
        f"✅ SCRAPER 7 (Features ML):     {ml:,}/{total:,} ({ml*100//total if total > 0 else 0}%)"
    )

    # SCRAPER 8 - Connections (winrate_entraineur non trouvé, utiliser entraineur_winrate_90j)
    cur.execute("""
        SELECT COUNT(*) as enrichi
        FROM cheval_courses_seen
        WHERE race_key LIKE '2024-10-%'
        AND entraineur_winrate_90j IS NOT NULL
    """)
    conn_data = cur.fetchone()["enrichi"]
    print(
        f"✅ SCRAPER 8 (Connections):     {conn_data:,}/{total:,} ({conn_data*100//total if total > 0 else 0}%)"
    )

    # 4. Colonnes vides (à identifier)
    print("\n🔍 DÉTECTION COLONNES NON ENRICHIES:\n")

    # Liste des colonnes importantes à vérifier
    colonnes_cles = [
        ("classe_course", "Phase 1"),
        ("meteo_code", "Phase 1"),
        ("allocations_1er", "Phase 1"),
        ("nombre_places_payees", "Phase 2"),
        ("gain_place_1", "Phase 2"),
        ("indice_performance", "Phase 2"),
    ]

    for colonne, phase in colonnes_cles:
        cur.execute(f"""
            SELECT COUNT(*) as vides
            FROM cheval_courses_seen
            WHERE race_key LIKE '2024-10-%'
            AND {colonne} IS NULL
        """)
        vides = cur.fetchone()["vides"]
        taux_enrichi = 100 - (vides * 100 // total if total > 0 else 0)

        if vides > 0:
            status = "⚠️ " if taux_enrichi < 50 else "✅"
            print(f"{status} {colonne:30s} ({phase:7s}): {taux_enrichi:3d}% enrichi")

    print(f"\n{'='*80}\n")

    cur.close()
    conn.close()


if __name__ == "__main__":
    try:
        check_october_data()
    except Exception as e:
        print(f"❌ Erreur: {e}")
