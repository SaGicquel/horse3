#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualiser les dernières performances d'un cheval
"""

import sqlite3
import json
import sys

DB_PATH = "data/database.db"

def voir_performances(nom_cheval):
    """Affiche les dernières performances d'un cheval"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    print("=" * 80)
    print(f"🏇 PERFORMANCES DE: {nom_cheval.upper()}")
    print("=" * 80)
    print()
    
    # Rechercher le cheval
    cur.execute("""
        SELECT nom, date_naissance, sexe, race, entraineur_courant,
               nombre_courses_total, nombre_victoires_total,
               dernier_resultat, dernieres_performances
        FROM chevaux
        WHERE LOWER(nom) LIKE LOWER(?)
        ORDER BY date_naissance DESC
    """, (f"%{nom_cheval}%",))
    
    results = cur.fetchall()
    
    if not results:
        print(f"❌ Aucun cheval trouvé pour '{nom_cheval}'")
        con.close()
        return
    
    for i, row in enumerate(results, 1):
        nom, date_naiss, sexe, race, entraineur, nb_courses, nb_victoires, musique, perfs_json = row
        
        taux = (nb_victoires / nb_courses * 100) if nb_courses and nb_victoires else 0
        
        print(f"{'─' * 80}")
        print(f"🏇 Cheval #{i}: {nom.upper()}")
        print(f"{'─' * 80}")
        print(f"   📅 Date de naissance: {date_naiss or 'N/A'}")
        print(f"   ⚥  Sexe: {sexe or 'N/A'}")
        print(f"   🎯 Race: {race or 'N/A'}")
        print(f"   👨 Entraîneur: {entraineur or 'N/A'}")
        print(f"   📊 Statistiques: {nb_courses or 0} courses, {nb_victoires or 0} victoires ({taux:.1f}%)")
        print(f"   🎵 Musique: {musique or 'N/A'}")
        print()
        
        # Afficher les dernières performances
        if perfs_json:
            try:
                perfs = json.loads(perfs_json)
                if perfs:
                    print(f"   📜 Dernières performances:")
                    print(f"   {'─' * 76}")
                    print(f"   {'#':<4} {'Date':<12} {'Hippodrome':<25} {'Place':<8} {'Victoire'}")
                    print(f"   {'─' * 76}")
                    
                    for j, p in enumerate(perfs[:10], 1):
                        date_course = p.get('date', '?')
                        hippo = p.get('hippodrome', '?')[:24]
                        place = p.get('place', '?')
                        is_win = "🏆" if p.get('is_win') else ""
                        
                        print(f"   {j:<4} {date_course:<12} {hippo:<25} {place:<8} {is_win}")
                    
                    print(f"   {'─' * 76}")
                    print()
            except json.JSONDecodeError:
                print(f"   ⚠️  Erreur de décodage des performances")
                print()
        
        # Afficher les courses actuelles (celles du jour)
        cur.execute("""
            SELECT race_key
            FROM cheval_courses_seen
            WHERE nom_norm = LOWER(?)
            ORDER BY race_key DESC
        """, (nom,))
        
        courses_actuelles = cur.fetchall()
        if courses_actuelles:
            print(f"   🏁 Courses actuelles (scrapées):")
            for course in courses_actuelles:
                # Parse race_key: DATE|R#|C#|HIPPO
                parts = course[0].split('|')
                if len(parts) >= 4:
                    date_c, reunion, course_n, hippo = parts[0], parts[1], parts[2], '|'.join(parts[3:])
                    print(f"      • {date_c} - {reunion} {course_n} à {hippo}")
            print()
    
    con.close()
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        nom = ' '.join(sys.argv[1:])
    else:
        nom = input("🔍 Nom du cheval à rechercher: ")
    
    voir_performances(nom)
