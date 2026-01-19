#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyses avancées des données PMU
Exploite au maximum vos données existantes
"""

import sqlite3
import json
from collections import defaultdict, Counter
from datetime import datetime

DB_PATH = "data/database.db"


def analyze_top_performers():
    """Top 20 chevaux par taux de victoire."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🏆 TOP 20 CHEVAUX PAR TAUX DE VICTOIRE                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        SELECT nom,
               nombre_victoires_total,
               nombre_courses_total,
               ROUND(CAST(nombre_victoires_total AS REAL) * 100 / nombre_courses_total, 2) as taux_victoire,
               race
        FROM chevaux
        WHERE nombre_courses_total >= 10
        ORDER BY taux_victoire DESC
        LIMIT 20
    """)

    print(
        f"{'Rang':<5} | {'Nom':<30} | {'Victoires':<10} | {'Courses':<8} | {'Taux':<8} | {'Race':<20}"
    )
    print(f"{'-'*5} | {'-'*30} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*20}")

    for i, (nom, vict, courses, taux, race) in enumerate(cur.fetchall(), 1):
        print(
            f"{i:<5} | {nom:<30} | {vict:<10} | {courses:<8} | {taux:<8.2f}% | {race or 'N/A':<20}"
        )

    con.close()


def analyze_2025_form():
    """Chevaux en forme en 2025."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🔥 TOP 20 CHEVAUX EN FORME (2025)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        SELECT nom,
               nombre_courses_2025,
               nombre_victoires_2025,
               ROUND(CAST(nombre_victoires_2025 AS REAL) * 100 / nombre_courses_2025, 2) as forme_2025,
               dernier_resultat
        FROM chevaux
        WHERE nombre_courses_2025 >= 5
        ORDER BY forme_2025 DESC, nombre_victoires_2025 DESC
        LIMIT 20
    """)

    print(
        f"{'Rang':<5} | {'Nom':<30} | {'Courses':<8} | {'Victoires':<10} | {'Taux':<8} | {'Musique':<15}"
    )
    print(f"{'-'*5} | {'-'*30} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*15}")

    for i, (nom, courses, vict, taux, musique) in enumerate(cur.fetchall(), 1):
        musique_short = (
            (musique[:15] + "...") if musique and len(musique) > 15 else (musique or "N/A")
        )
        print(
            f"{i:<5} | {nom:<30} | {courses:<8} | {vict:<10} | {taux:<8.2f}% | {musique_short:<15}"
        )

    con.close()


def analyze_regular_horses():
    """Chevaux réguliers sans problèmes."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ⭐ CHEVAUX RÉGULIERS (Sans D/A/T dans musique)                ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        SELECT nom, dernier_resultat, nombre_courses_total, nombre_victoires_total
        FROM chevaux
        WHERE dernier_resultat IS NOT NULL
          AND LENGTH(dernier_resultat) > 10
          AND race = 'TROTTEUR FRANCAIS'
        ORDER BY nombre_courses_total DESC
        LIMIT 500
    """)

    reguliers = []
    for nom, musique, courses, vict in cur.fetchall():
        problemes = musique.count("D") + musique.count("A") + musique.count("T")
        if problemes == 0:
            reguliers.append((nom, musique, courses, vict))

    print(f"Trouvés: {len(reguliers)} chevaux réguliers\n")
    print(f"{'Rang':<5} | {'Nom':<30} | {'Courses':<8} | {'Victoires':<10} | {'Musique':<20}")
    print(f"{'-'*5} | {'-'*30} | {'-'*8} | {'-'*10} | {'-'*20}")

    for i, (nom, musique, courses, vict) in enumerate(reguliers[:20], 1):
        musique_short = (musique[:20] + "...") if len(musique) > 20 else musique
        print(f"{i:<5} | {nom:<30} | {courses:<8} | {vict:<10} | {musique_short:<20}")

    con.close()


def analyze_progression():
    """Chevaux en progression (meilleur taux 2025 que total)."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              📈 CHEVAUX EN PROGRESSION (2025 > Total)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        SELECT nom,
               nombre_courses_total,
               nombre_victoires_total,
               nombre_courses_2025,
               nombre_victoires_2025,
               ROUND(CAST(nombre_victoires_2025 AS REAL) * 100 / nombre_courses_2025, 2) as taux_2025,
               ROUND(CAST(nombre_victoires_total AS REAL) * 100 / nombre_courses_total, 2) as taux_total
        FROM chevaux
        WHERE nombre_courses_2025 >= 5
          AND nombre_courses_total >= 20
          AND CAST(nombre_victoires_2025 AS REAL) / nombre_courses_2025 >
              CAST(nombre_victoires_total AS REAL) / nombre_courses_total
        ORDER BY taux_2025 DESC
        LIMIT 20
    """)

    print(f"{'Rang':<5} | {'Nom':<30} | {'Taux 2025':<11} | {'Taux Total':<12} | {'Progrès':<10}")
    print(f"{'-'*5} | {'-'*30} | {'-'*11} | {'-'*12} | {'-'*10}")

    for i, (nom, ct, vt, c25, v25, taux25, taux_tot) in enumerate(cur.fetchall(), 1):
        progres = taux25 - taux_tot
        print(f"{i:<5} | {nom:<30} | {taux25:<11.2f}% | {taux_tot:<12.2f}% | +{progres:<9.2f}%")

    con.close()


def analyze_recent_winners():
    """Chevaux avec victoires dans les 5 dernières courses."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🎯 CHEVAUX AVEC VICTOIRES RÉCENTES                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        SELECT nom, dernieres_performances, nombre_courses_total
        FROM chevaux
        WHERE dernieres_performances IS NOT NULL
          AND race = 'TROTTEUR FRANCAIS'
    """)

    winners = []
    for nom, perfs_json, courses_total in cur.fetchall():
        try:
            perfs = json.loads(perfs_json)
            victoires_recentes = sum(1 for p in perfs if p.get("is_win"))

            if victoires_recentes >= 2:
                winners.append((nom, victoires_recentes, len(perfs), courses_total))
        except:
            pass

    winners.sort(key=lambda x: x[1], reverse=True)

    print(f"Trouvés: {len(winners)} chevaux avec 2+ victoires récentes\n")
    print(
        f"{'Rang':<5} | {'Nom':<30} | {'Vict. Récentes':<15} | {'Sur':<5} | {'Total Courses':<15}"
    )
    print(f"{'-'*5} | {'-'*30} | {'-'*15} | {'-'*5} | {'-'*15}")

    for i, (nom, vict_rec, total_rec, courses) in enumerate(winners[:20], 1):
        print(f"{i:<5} | {nom:<30} | {vict_rec:<15} | {total_rec:<5} | {courses:<15}")

    con.close()


def analyze_trotters_summary():
    """Résumé des trotteurs."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              📊 RÉSUMÉ STATISTIQUE - TROTTEURS FRANÇAIS                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Stats générales
    cur.execute("SELECT COUNT(*) FROM chevaux WHERE race='TROTTEUR FRANCAIS'")
    total = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM chevaux WHERE race='TROTTEUR FRANCAIS' AND nombre_courses_2025 > 0"
    )
    actifs_2025 = cur.fetchone()[0]

    cur.execute(
        "SELECT SUM(nombre_courses_total), SUM(nombre_victoires_total) FROM chevaux WHERE race='TROTTEUR FRANCAIS'"
    )
    total_courses, total_vict = cur.fetchone()

    cur.execute(
        "SELECT AVG(CAST(nombre_victoires_total AS REAL) * 100 / nombre_courses_total) FROM chevaux WHERE race='TROTTEUR FRANCAIS' AND nombre_courses_total > 0"
    )
    taux_moyen = cur.fetchone()[0]

    # Distribution par sexe
    cur.execute("""
        SELECT sexe, COUNT(*)
        FROM chevaux
        WHERE race='TROTTEUR FRANCAIS'
        GROUP BY sexe
    """)
    sexes = dict(cur.fetchall())

    print("📊 Statistiques Globales")
    print(f"{'─'*80}")
    print(f"Total trotteurs:              {total:,}")
    print(f"Actifs en 2025:               {actifs_2025:,} ({actifs_2025/total*100:.1f}%)")
    print(f"Total courses enregistrées:   {total_courses:,}")
    print(f"Total victoires:              {total_vict:,}")
    print(f"Taux de victoire moyen:       {taux_moyen:.2f}%")
    print()

    print("👥 Répartition par sexe")
    print(f"{'─'*80}")
    for sexe, count in sorted(sexes.items(), key=lambda x: x[1], reverse=True):
        sexe_label = sexe if sexe else "Non renseigné"
        print(f"{sexe_label:20s} : {count:5,} ({count/total*100:5.1f}%)")

    con.close()


def main_menu():
    """Menu principal."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              📊 ANALYSES AVANCÉES - DONNÉES PMU                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Exploitez au maximum vos 39,565 chevaux en base !

Analyses disponibles:
  1. 🏆 Top 20 chevaux par taux de victoire
  2. 🔥 Top 20 chevaux en forme (2025)
  3. ⭐ Chevaux réguliers (sans problèmes)
  4. 📈 Chevaux en progression
  5. 🎯 Chevaux avec victoires récentes
  6. 📊 Résumé statistique trotteurs
  7. ✨ TOUTES les analyses

  0. Quitter
""")

    choice = input("Votre choix (0-7): ").strip()

    if choice == "1":
        analyze_top_performers()
    elif choice == "2":
        analyze_2025_form()
    elif choice == "3":
        analyze_regular_horses()
    elif choice == "4":
        analyze_progression()
    elif choice == "5":
        analyze_recent_winners()
    elif choice == "6":
        analyze_trotters_summary()
    elif choice == "7":
        analyze_trotters_summary()
        print("\n" + "=" * 80 + "\n")
        analyze_top_performers()
        print("\n" + "=" * 80 + "\n")
        analyze_2025_form()
        print("\n" + "=" * 80 + "\n")
        analyze_regular_horses()
        print("\n" + "=" * 80 + "\n")
        analyze_progression()
        print("\n" + "=" * 80 + "\n")
        analyze_recent_winners()
    elif choice == "0":
        print("\nAu revoir ! 👋")
        return
    else:
        print("\n❌ Choix invalide")

    print("\n" + "=" * 80)
    input("\nAppuyez sur Entrée pour continuer...")
    main_menu()


if __name__ == "__main__":
    main_menu()
