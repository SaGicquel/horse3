#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour scraper le 16 janvier et analyser les résultats
"""

import sys
from datetime import datetime
from scraper_pmu_simple import run
from db_connection import get_connection


def analyse_avant_apres():
    """Analyse l'état avant et après scraping"""
    con = get_connection()
    cur = con.cursor()

    # État AVANT
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux_avant = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_courses_avant = cur.fetchone()[0]

    # Vérifier les colonnes
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'chevaux'
        ORDER BY ordinal_position
    """)
    cols_chevaux_avant = [r[0] for r in cur.fetchall()]

    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'cheval_courses_seen'
        ORDER BY ordinal_position
    """)
    cols_courses_avant = [r[0] for r in cur.fetchall()]

    con.close()

    return {
        "nb_chevaux_avant": nb_chevaux_avant,
        "nb_courses_avant": nb_courses_avant,
        "cols_chevaux_avant": cols_chevaux_avant,
        "cols_courses_avant": cols_courses_avant,
    }


def analyse_apres(date_iso):
    """Analyse l'état après scraping"""
    con = get_connection()
    cur = con.cursor()

    # État APRÈS
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux_apres = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_courses_apres = cur.fetchone()[0]

    # Nouvelles données pour cette date
    cur.execute(
        """
        SELECT COUNT(*)
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
    """,
        (f"{date_iso}|%",),
    )
    nb_courses_date = cur.fetchone()[0]

    # Statistiques par discipline
    cur.execute(
        """
        SELECT
            discipline,
            COUNT(*) as nb_courses,
            COUNT(DISTINCT id_cheval) as nb_chevaux
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        GROUP BY discipline
    """,
        (f"{date_iso}|%",),
    )
    stats_discipline = cur.fetchall()

    # Nouvelles colonnes utilisées (non NULL)
    cur.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE rapport_quarte IS NOT NULL) as avec_rapport_quarte,
            COUNT(*) FILTER (WHERE rapport_quinte IS NOT NULL) as avec_rapport_quinte,
            COUNT(*) FILTER (WHERE vitesse_moyenne IS NOT NULL) as avec_vitesse,
            COUNT(*) FILTER (WHERE ecart_premier IS NOT NULL) as avec_ecart,
            COUNT(*) FILTER (WHERE allocation_premier IS NOT NULL) as avec_allocation,
            COUNT(*) FILTER (WHERE pays_hippodrome IS NOT NULL) as avec_pays_hippo,
            COUNT(*) FILTER (WHERE id_driver_pmu IS NOT NULL) as avec_id_driver
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
    """,
        (f"{date_iso}|%",),
    )
    stats_nouvelles_cols = cur.fetchone()

    # Exemples de données enrichies
    cur.execute(
        """
        SELECT
            nom_norm,
            discipline,
            rapport_quarte,
            rapport_quinte,
            vitesse_moyenne,
            ecart_premier,
            allocation_premier,
            pays_hippodrome,
            couleurs_casaque_driver
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        AND (rapport_quarte IS NOT NULL
             OR rapport_quinte IS NOT NULL
             OR vitesse_moyenne IS NOT NULL
             OR allocation_premier IS NOT NULL)
        LIMIT 5
    """,
        (f"{date_iso}|%",),
    )
    exemples = cur.fetchall()

    # Statistiques chevaux enrichis
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE nombre_courses_trot IS NOT NULL) as avec_stats_trot,
            COUNT(*) FILTER (WHERE nombre_courses_plat IS NOT NULL) as avec_stats_plat,
            COUNT(*) FILTER (WHERE gains_annuels_par_annee IS NOT NULL) as avec_gains_annuels
        FROM chevaux
    """)
    stats_chevaux_enrichis = cur.fetchone()

    con.close()

    return {
        "nb_chevaux_apres": nb_chevaux_apres,
        "nb_courses_apres": nb_courses_apres,
        "nb_courses_date": nb_courses_date,
        "stats_discipline": stats_discipline,
        "stats_nouvelles_cols": stats_nouvelles_cols,
        "exemples": exemples,
        "stats_chevaux_enrichis": stats_chevaux_enrichis,
    }


def main():
    date_iso = "2025-01-16"

    print("=" * 80)
    print(f"📊 ANALYSE DU SCRAPING POUR LE {date_iso}")
    print("=" * 80)
    print()

    # 1. État avant
    print("📋 ÉTAT AVANT SCRAPING")
    print("-" * 80)
    etat_avant = analyse_avant_apres()
    print(f"  • Chevaux en base: {etat_avant['nb_chevaux_avant']}")
    print(f"  • Courses en base: {etat_avant['nb_courses_avant']}")
    print(f"  • Colonnes chevaux: {len(etat_avant['cols_chevaux_avant'])}")
    print(f"  • Colonnes courses: {len(etat_avant['cols_courses_avant'])}")
    print()

    # 2. Lancement du scraping
    print("🚀 LANCEMENT DU SCRAPING")
    print("-" * 80)
    try:
        run(date_iso, recalc_after=True, use_threading=True)
        print("✅ Scraping terminé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du scraping: {e}")
        import traceback

        traceback.print_exc()
        return 1
    print()

    # 3. État après
    print("📊 ÉTAT APRÈS SCRAPING")
    print("-" * 80)
    etat_apres = analyse_apres(date_iso)

    print(
        f"  • Chevaux en base: {etat_avant['nb_chevaux_avant']} → {etat_apres['nb_chevaux_apres']} (+{etat_apres['nb_chevaux_apres'] - etat_avant['nb_chevaux_avant']})"
    )
    print(
        f"  • Courses en base: {etat_avant['nb_courses_avant']} → {etat_apres['nb_courses_apres']} (+{etat_apres['nb_courses_apres'] - etat_avant['nb_courses_avant']})"
    )
    print(f"  • Courses pour {date_iso}: {etat_apres['nb_courses_date']}")
    print()

    # 4. Statistiques par discipline
    print("🏇 STATISTIQUES PAR DISCIPLINE")
    print("-" * 80)
    if etat_apres["stats_discipline"]:
        for disc, nb_c, nb_ch in etat_apres["stats_discipline"]:
            print(f"  • {disc or 'N/A'}: {nb_c} courses, {nb_ch} chevaux distincts")
    else:
        print("  ⚠️  Aucune course trouvée pour cette date")
    print()

    # 5. Utilisation des nouvelles colonnes
    print("🆕 UTILISATION DES NOUVELLES COLONNES")
    print("-" * 80)
    stats = etat_apres["stats_nouvelles_cols"]
    total = etat_apres["nb_courses_date"]

    if total > 0:
        print(f"  Sur {total} courses scrapées:")
        print(f"    • {stats[0]} avec rapport Quarté+ ({stats[0]*100//total if total > 0 else 0}%)")
        print(f"    • {stats[1]} avec rapport Quinté+ ({stats[1]*100//total if total > 0 else 0}%)")
        print(f"    • {stats[2]} avec vitesse moyenne ({stats[2]*100//total if total > 0 else 0}%)")
        print(f"    • {stats[3]} avec écart premier ({stats[3]*100//total if total > 0 else 0}%)")
        print(f"    • {stats[4]} avec allocation 1er ({stats[4]*100//total if total > 0 else 0}%)")
        print(f"    • {stats[5]} avec pays hippodrome ({stats[5]*100//total if total > 0 else 0}%)")
        print(f"    • {stats[6]} avec ID driver PMU ({stats[6]*100//total if total > 0 else 0}%)")
    else:
        print("  ⚠️  Aucune donnée disponible")
    print()

    # 6. Chevaux enrichis
    print("🐴 CHEVAUX ENRICHIS")
    print("-" * 80)
    stats_ch = etat_apres["stats_chevaux_enrichis"]
    total_ch = etat_apres["nb_chevaux_apres"]

    if total_ch > 0:
        print(f"  Sur {total_ch} chevaux:")
        print(
            f"    • {stats_ch[0]} avec stats trot ({stats_ch[0]*100//total_ch if total_ch > 0 else 0}%)"
        )
        print(
            f"    • {stats_ch[1]} avec stats plat ({stats_ch[1]*100//total_ch if total_ch > 0 else 0}%)"
        )
        print(
            f"    • {stats_ch[2]} avec gains annuels ({stats_ch[2]*100//total_ch if total_ch > 0 else 0}%)"
        )
    print()

    # 7. Exemples de données enrichies
    print("📝 EXEMPLES DE DONNÉES ENRICHIES")
    print("-" * 80)
    if etat_apres["exemples"]:
        for i, ex in enumerate(etat_apres["exemples"][:3], 1):
            print(f"  Exemple {i}:")
            print(f"    • Cheval: {ex[0]}")
            print(f"    • Discipline: {ex[1]}")
            if ex[2]:
                print(f"    • Rapport Quarté+: {ex[2]}")
            if ex[3]:
                print(f"    • Rapport Quinté+: {ex[3]}")
            if ex[4]:
                print(f"    • Vitesse moyenne: {ex[4]:.2f} km/h")
            if ex[5]:
                print(f"    • Écart 1er: {ex[5]:.2f}s")
            if ex[6]:
                print(f"    • Allocation 1er: {ex[6]}€")
            if ex[7]:
                print(f"    • Pays hippodrome: {ex[7]}")
            if ex[8]:
                print(f"    • Couleurs casaque: {ex[8]}")
            print()
    else:
        print("  ⚠️  Aucun exemple disponible (données pas encore enrichies)")

    print("=" * 80)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
