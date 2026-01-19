#!/usr/bin/env python3
"""
Visualiseur de statistiques de la base de données.
Affiche un rapport complet de l'état de la BDD.

Usage:
    python view_stats.py [--full]
"""

import argparse
from datetime import datetime
from db_connection import get_connection


def print_header(title: str):
    """Affiche un titre formaté."""
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    print("=" * 70)


def view_summary():
    """Affiche un résumé du contenu de la BDD."""
    conn = get_connection()
    cur = conn.cursor()

    print_header("RÉSUMÉ DE LA BASE DE DONNÉES")

    tables = [
        "hippodromes",
        "courses",
        "chevaux",
        "personnes",
        "performances",
        "cotes_historiques",
        "temps_sectionnels",
        "stats_chevaux",
        "stats_personnes",
    ]

    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        status = "✅" if count > 0 else "⚪"
        print(f"   {status} {table:25s} : {count:8d} lignes")

    cur.close()
    conn.close()


def view_hippodromes():
    """Affiche les hippodromes."""
    conn = get_connection()
    cur = conn.cursor()

    print_header("HIPPODROMES")

    cur.execute("""
        SELECT
            h.nom_hippodrome,
            h.code_pmu,
            h.pays,
            COUNT(c.id_course) as nb_courses
        FROM hippodromes h
        LEFT JOIN courses c ON h.id_hippodrome = c.id_hippodrome
        GROUP BY h.id_hippodrome, h.nom_hippodrome, h.code_pmu, h.pays
        ORDER BY nb_courses DESC
    """)

    rows = cur.fetchall()
    if rows:
        for row in rows:
            print(f"   {row[0]:30s} ({row[1]:4s}) - {row[2]:2s} : {row[3]:4d} courses")
    else:
        print("   (aucun hippodrome)")

    cur.close()
    conn.close()


def view_courses_by_discipline():
    """Affiche la répartition des courses par discipline."""
    conn = get_connection()
    cur = conn.cursor()

    print_header("COURSES PAR DISCIPLINE")

    cur.execute("""
        SELECT
            discipline,
            COUNT(*) as nb_courses,
            MIN(date_course) as premiere_course,
            MAX(date_course) as derniere_course
        FROM courses
        GROUP BY discipline
        ORDER BY nb_courses DESC
    """)

    rows = cur.fetchall()
    if rows:
        for row in rows:
            print(f"   {row[0]:15s} : {row[1]:4d} courses (du {row[2]} au {row[3]})")
    else:
        print("   (aucune course)")

    cur.close()
    conn.close()


def view_top_chevaux():
    """Affiche les meilleurs chevaux."""
    conn = get_connection()
    cur = conn.cursor()

    print_header("TOP 10 CHEVAUX (Taux de victoire)")

    cur.execute("""
        SELECT
            ch.nom_cheval,
            ch.sexe_cheval,
            ch.an_naissance,
            sc.nb_courses_total,
            sc.nb_victoires,
            sc.tx_victoire,
            sc.forme_5c
        FROM stats_chevaux sc
        JOIN chevaux ch ON sc.id_cheval = ch.id_cheval
        WHERE sc.nb_courses_total >= 1
        ORDER BY sc.tx_victoire DESC, sc.nb_victoires DESC
        LIMIT 10
    """)

    rows = cur.fetchall()
    if rows:
        print(
            f"\n   {'Cheval':25s} {'S':1s} {'Année':4s} {'Courses':7s} {'Vict':5s} {'Tx%':6s} {'Forme':6s}"
        )
        print(f"   {'-'*25} {'-'*1} {'-'*4} {'-'*7} {'-'*5} {'-'*6} {'-'*6}")
        for row in rows:
            forme = f"{row[6]:.1f}" if row[6] else "N/A"
            print(
                f"   {row[0]:25s} {row[1]:1s} {row[2]:4d} {row[3]:7d} {row[4]:5d} {row[5]:6.1f} {forme:6s}"
            )
    else:
        print("   (aucune statistique calculée)")

    cur.close()
    conn.close()


def view_top_jockeys():
    """Affiche les meilleurs jockeys."""
    conn = get_connection()
    cur = conn.cursor()

    print_header("TOP 10 JOCKEYS (12 mois)")

    cur.execute("""
        SELECT
            per.nom_complet,
            sp.nb_courses,
            sp.nb_victoires,
            sp.tx_victoire,
            sp.nb_places,
            sp.tx_place
        FROM stats_personnes sp
        JOIN personnes per ON sp.id_personne = per.id_personne
        WHERE per.type = 'JOCKEY'
        AND sp.periode = '12M'
        AND sp.nb_courses >= 1
        ORDER BY sp.tx_victoire DESC, sp.nb_victoires DESC
        LIMIT 10
    """)

    rows = cur.fetchall()
    if rows:
        print(
            f"\n   {'Jockey':30s} {'Courses':7s} {'Vict':5s} {'Tx%':6s} {'Places':6s} {'TxP%':6s}"
        )
        print(f"   {'-'*30} {'-'*7} {'-'*5} {'-'*6} {'-'*6} {'-'*6}")
        for row in rows:
            print(
                f"   {row[0]:30s} {row[1]:7d} {row[2]:5d} {row[3]:6.1f} {row[4]:6d} {row[5]:6.1f}"
            )
    else:
        print("   (aucune statistique calculée)")

    cur.close()
    conn.close()


def view_top_entraineurs():
    """Affiche les meilleurs entraîneurs."""
    conn = get_connection()
    cur = conn.cursor()

    print_header("TOP 10 ENTRAÎNEURS (12 mois)")

    cur.execute("""
        SELECT
            per.nom_complet,
            sp.nb_courses,
            sp.nb_victoires,
            sp.tx_victoire,
            sp.nb_places,
            sp.tx_place
        FROM stats_personnes sp
        JOIN personnes per ON sp.id_personne = per.id_personne
        WHERE per.type = 'ENTRAINEUR'
        AND sp.periode = '12M'
        AND sp.nb_courses >= 1
        ORDER BY sp.tx_victoire DESC, sp.nb_victoires DESC
        LIMIT 10
    """)

    rows = cur.fetchall()
    if rows:
        print(
            f"\n   {'Entraîneur':30s} {'Courses':7s} {'Vict':5s} {'Tx%':6s} {'Places':6s} {'TxP%':6s}"
        )
        print(f"   {'-'*30} {'-'*7} {'-'*5} {'-'*6} {'-'*6} {'-'*6}")
        for row in rows:
            print(
                f"   {row[0]:30s} {row[1]:7d} {row[2]:5d} {row[3]:6.1f} {row[4]:6d} {row[5]:6.1f}"
            )
    else:
        print("   (aucune statistique calculée)")

    cur.close()
    conn.close()


def view_recent_performances():
    """Affiche les performances récentes."""
    conn = get_connection()
    cur = conn.cursor()

    print_header("10 DERNIÈRES PERFORMANCES")

    cur.execute("""
        SELECT
            c.date_course,
            h.nom_hippodrome,
            c.discipline,
            ch.nom_cheval,
            p.position_arrivee,
            p.cote_sp,
            j.nom_complet as jockey
        FROM performances p
        JOIN courses c ON p.id_course = c.id_course
        JOIN chevaux ch ON p.id_cheval = ch.id_cheval
        JOIN hippodromes h ON c.id_hippodrome = h.id_hippodrome
        LEFT JOIN personnes j ON p.id_jockey = j.id_personne
        WHERE p.non_partant = FALSE
        ORDER BY c.date_course DESC, c.num_reunion DESC, c.num_course DESC
        LIMIT 10
    """)

    rows = cur.fetchall()
    if rows:
        print(
            f"\n   {'Date':10s} {'Hippodrome':15s} {'Disc':8s} {'Cheval':20s} {'Pos':3s} {'Cote':6s} {'Jockey':20s}"
        )
        print(f"   {'-'*10} {'-'*15} {'-'*8} {'-'*20} {'-'*3} {'-'*6} {'-'*20}")
        for row in rows:
            pos = str(row[4]) if row[4] else "NP"
            cote = f"{row[5]:.1f}" if row[5] else "N/A"
            jockey = row[6][:20] if row[6] else "N/A"
            print(
                f"   {str(row[0]):10s} {row[1]:15s} {row[2]:8s} {row[3]:20s} {pos:3s} {cote:6s} {jockey:20s}"
            )
    else:
        print("   (aucune performance)")

    cur.close()
    conn.close()


def view_full_report():
    """Affiche un rapport complet."""
    print("\n" + "🏇" * 35)
    print("🏇" + " " * 33 + "🏇")
    print("🏇" + " " * 7 + "RAPPORT STATISTIQUES PMU" + " " * 8 + "🏇")
    print("🏇" + " " * 33 + "🏇")
    print("🏇" * 35)
    print(f"\n📅 Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    view_summary()
    view_hippodromes()
    view_courses_by_discipline()
    view_top_chevaux()
    view_top_jockeys()
    view_top_entraineurs()
    view_recent_performances()

    print("\n" + "=" * 70)
    print("✅ Rapport terminé")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualiseur de statistiques PMU")
    parser.add_argument("--full", action="store_true", help="Rapport complet")

    args = parser.parse_args()

    try:
        if args.full:
            view_full_report()
        else:
            view_summary()
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
