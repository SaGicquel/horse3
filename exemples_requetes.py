#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exemples de requêtes sur la base de données
"""

import sqlite3

DB_PATH = "data/database.db"


def rechercher_cheval(nom_partiel):
    """Recherche un cheval par son nom (partiel accepté)"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    print(f"🔍 Recherche de '{nom_partiel}'...\n")

    cur.execute(
        """
        SELECT nom, date_naissance, sexe, race,
               nombre_courses_total, nombre_victoires_total,
               entraineur_courant, dernier_resultat
        FROM chevaux
        WHERE LOWER(nom) LIKE LOWER(?)
        ORDER BY nombre_courses_total DESC
    """,
        (f"%{nom_partiel}%",),
    )

    results = cur.fetchall()

    if not results:
        print(f"❌ Aucun cheval trouvé pour '{nom_partiel}'")
        con.close()
        return

    print(f"✅ {len(results)} résultat(s) trouvé(s):\n")

    for i, row in enumerate(results, 1):
        nom, date_naiss, sexe, race, nb_courses, nb_victoires, entraineur, musique = row

        taux = (nb_victoires / nb_courses * 100) if nb_courses and nb_victoires else 0

        print(f"{'─' * 70}")
        print(f"🏇 Cheval #{i}: {nom.upper()}")
        print(f"{'─' * 70}")
        print(f"   📅 Date de naissance: {date_naiss or 'N/A'}")
        print(f"   ⚥  Sexe: {sexe or 'N/A'}")
        print(f"   🎯 Race: {race or 'N/A'}")
        print(f"   👨 Entraîneur: {entraineur or 'N/A'}")
        print(
            f"   📊 Statistiques: {nb_courses or 0} courses, {nb_victoires or 0} victoires ({taux:.1f}%)"
        )
        print(f"   🎵 Musique: {musique or 'N/A'}")
        print()

    con.close()


def top_performers(limite=10):
    """Top des meilleurs chevaux (ratio victoires/courses)"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    print(f"🏆 TOP {limite} DES MEILLEURS CHEVAUX (ratio victoires/courses)\n")

    cur.execute(
        """
        SELECT nom, date_naissance, sexe, race,
               nombre_courses_total, nombre_victoires_total,
               ROUND(CAST(nombre_victoires_total AS FLOAT) / nombre_courses_total * 100, 1) as taux
        FROM chevaux
        WHERE nombre_courses_total >= 10
        AND nombre_victoires_total > 0
        ORDER BY taux DESC, nombre_victoires_total DESC
        LIMIT ?
    """,
        (limite,),
    )

    print(f"{'Rang':<5} {'Nom':<25} {'Né':<12} {'Courses':<8} {'Victoires':<10} {'Taux':<8}")
    print("─" * 80)

    for i, row in enumerate(cur.fetchall(), 1):
        nom, date_naiss, sexe, race, nb_courses, nb_victoires, taux = row
        print(
            f"{i:<5} {nom[:24]:<25} {date_naiss or 'N/A':<12} {nb_courses:<8} {nb_victoires:<10} {taux:.1f}%"
        )

    print()
    con.close()


def chevaux_par_annee(annee):
    """Liste les chevaux nés une année donnée"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    print(f"🎂 CHEVAUX NÉS EN {annee}\n")

    cur.execute(
        """
        SELECT nom, sexe, race, nombre_courses_total, nombre_victoires_total
        FROM chevaux
        WHERE date_naissance LIKE ?
        ORDER BY nombre_courses_total DESC
        LIMIT 20
    """,
        (f"{annee}%",),
    )

    results = cur.fetchall()

    if not results:
        print(f"❌ Aucun cheval trouvé pour l'année {annee}")
        con.close()
        return

    print(f"✅ {len(results)} chevaux trouvés (affichage des 20 premiers par nombre de courses):\n")
    print(f"{'Nom':<30} {'Sexe':<6} {'Race':<20} {'Courses':<8} {'Victoires'}")
    print("─" * 80)

    for row in results:
        nom, sexe, race, nb_courses, nb_victoires = row
        print(
            f"{nom[:29]:<30} {sexe or 'N/A':<6} {(race or 'N/A')[:19]:<20} {nb_courses or 0:<8} {nb_victoires or 0}"
        )

    print()
    con.close()


def stats_generales():
    """Statistiques générales sur la base"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    print("📊 STATISTIQUES GÉNÉRALES\n")
    print("─" * 80)

    # Total chevaux
    cur.execute("SELECT COUNT(*) FROM chevaux")
    total = cur.fetchone()[0]
    print(f"📝 Total de chevaux: {total}")

    # Avec date de naissance
    cur.execute("SELECT COUNT(*) FROM chevaux WHERE date_naissance IS NOT NULL")
    avec_date = cur.fetchone()[0]
    print(f"📅 Avec date de naissance: {avec_date} ({avec_date/total*100:.1f}%)")

    # Courses distinctes
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    courses = cur.fetchone()[0]
    print(f"🏁 Courses distinctes enregistrées: {courses}")

    # Répartition par sexe
    print("\n⚥ Répartition par sexe:")
    cur.execute("""
        SELECT sexe, COUNT(*) as nb
        FROM chevaux
        WHERE sexe IS NOT NULL
        GROUP BY sexe
        ORDER BY nb DESC
    """)
    for sexe, nb in cur.fetchall():
        sexe_label = {"H": "Hongres", "M": "Mâles", "F": "Femelles"}.get(sexe, sexe)
        pct = nb / total * 100
        print(f"   • {sexe_label:10s}: {nb:5d} ({pct:5.1f}%)")

    # Races les plus courantes
    print("\n🏇 Top 5 des races:")
    cur.execute("""
        SELECT race, COUNT(*) as nb
        FROM chevaux
        WHERE race IS NOT NULL
        GROUP BY race
        ORDER BY nb DESC
        LIMIT 5
    """)
    for race, nb in cur.fetchall():
        pct = nb / total * 100
        print(f"   • {race[:30]:30s}: {nb:5d} ({pct:5.1f}%)")

    # Chevaux les plus actifs
    print("\n🔥 Chevaux les plus actifs (nombre de courses):")
    cur.execute("""
        SELECT nom, nombre_courses_total, nombre_victoires_total
        FROM chevaux
        WHERE nombre_courses_total IS NOT NULL
        ORDER BY nombre_courses_total DESC
        LIMIT 5
    """)
    for nom, nb_c, nb_v in cur.fetchall():
        taux = (nb_v / nb_c * 100) if nb_c and nb_v else 0
        print(f"   • {nom[:30]:30s}: {nb_c:3d} courses, {nb_v:2d} victoires ({taux:5.1f}%)")

    print("\n" + "─" * 80 + "\n")
    con.close()


if __name__ == "__main__":
    print("=" * 80)
    print("🏇 EXEMPLES DE REQUÊTES SUR LA BASE DE DONNÉES")
    print("=" * 80)
    print()

    # 1. Statistiques générales
    stats_generales()

    # 2. Recherche d'un cheval
    rechercher_cheval("black saxon")

    # 3. Top performers
    top_performers(10)

    # 4. Chevaux nés en 2020
    chevaux_par_annee(2020)

    print("=" * 80)
    print("💡 Pour utiliser ces fonctions dans vos propres scripts:")
    print("   from exemples_requetes import rechercher_cheval, top_performers")
    print("   rechercher_cheval('nom_du_cheval')")
    print("=" * 80)
