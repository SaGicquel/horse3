#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualisation des données scrapées
"""

import sqlite3
from datetime import date

DB_PATH = "data/database.db"

def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    print("=" * 80)
    print("📊 ANALYSE DES DONNÉES SCRAPÉES")
    print("=" * 80)
    print()
    
    # Statistiques globales
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM chevaux WHERE date_naissance IS NOT NULL")
    nb_avec_date = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT nom) FROM chevaux")
    nb_noms_uniques = cur.fetchone()[0]
    
    print(f"📈 Statistiques générales:")
    print(f"   • Total de chevaux: {nb_chevaux}")
    print(f"   • Chevaux avec date de naissance: {nb_avec_date}")
    print(f"   • Noms uniques: {nb_noms_uniques}")
    print(f"   • Doublons potentiels: {nb_chevaux - nb_noms_uniques}")
    print()
    
    # Répartition par année de naissance
    print("🎂 Répartition par année de naissance:")
    cur.execute("""
        SELECT SUBSTR(date_naissance, 1, 4) as annee, COUNT(*) as nb
        FROM chevaux 
        WHERE date_naissance IS NOT NULL
        GROUP BY annee
        ORDER BY annee DESC
    """)
    for row in cur.fetchall():
        annee, nb = row
        barre = "█" * (nb // 20)
        print(f"   {annee}: {nb:4d} {barre}")
    print()
    
    # Répartition par sexe
    print("⚥ Répartition par sexe:")
    cur.execute("""
        SELECT sexe, COUNT(*) as nb
        FROM chevaux 
        WHERE sexe IS NOT NULL
        GROUP BY sexe
        ORDER BY nb DESC
    """)
    for row in cur.fetchall():
        sexe, nb = row
        sexe_label = {"H": "Hongre", "M": "Mâle", "F": "Femelle"}.get(sexe, sexe)
        print(f"   {sexe_label:10s}: {nb:5d}")
    print()
    
    # Répartition par race
    print("🏇 Répartition par race (top 10):")
    cur.execute("""
        SELECT race, COUNT(*) as nb
        FROM chevaux 
        WHERE race IS NOT NULL
        GROUP BY race
        ORDER BY nb DESC
        LIMIT 10
    """)
    for row in cur.fetchall():
        race, nb = row
        print(f"   {race[:30]:30s}: {nb:5d}")
    print()
    
    # Chevaux avec le plus de courses
    print("🏆 Chevaux avec le plus de courses (top 10):")
    cur.execute("""
        SELECT nom, date_naissance, nombre_courses_total, nombre_victoires_total, race
        FROM chevaux 
        WHERE nombre_courses_total IS NOT NULL
        ORDER BY nombre_courses_total DESC
        LIMIT 10
    """)
    for row in cur.fetchall():
        nom, date_naiss, nb_courses, nb_victoires, race = row
        taux = (nb_victoires / nb_courses * 100) if nb_courses and nb_victoires else 0
        print(f"   {nom[:25]:25s} ({date_naiss or 'N/A':10s}): {nb_courses:3d} courses, {nb_victoires:2d} victoires ({taux:5.1f}%)")
    print()
    
    # Exemples de doublons (même nom, dates différentes)
    print("👥 Exemples de doublons gérés (même nom, dates différentes):")
    cur.execute("""
        SELECT nom, COUNT(*) as nb_doublons
        FROM chevaux
        GROUP BY LOWER(nom)
        HAVING nb_doublons > 1
        LIMIT 5
    """)
    doublons = cur.fetchall()
    if doublons:
        for nom, nb in doublons:
            print(f"\n   📌 {nom.upper()} ({nb} chevaux):")
            cur.execute("""
                SELECT date_naissance, sexe, race, nombre_courses_total
                FROM chevaux
                WHERE LOWER(nom) = LOWER(?)
                ORDER BY date_naissance DESC
            """, (nom,))
            for dn, sx, rc, nbc in cur.fetchall():
                print(f"      • Né: {dn or 'N/A':10s} | Sexe: {sx or 'N/A'} | Race: {(rc or 'N/A')[:20]:20s} | Courses: {nbc or 0}")
    else:
        print("   Aucun doublon trouvé")
    print()
    
    con.close()
    
    print("=" * 80)

if __name__ == "__main__":
    main()
