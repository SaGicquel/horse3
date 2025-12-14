#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de vérification de l'intégrité de la base de données PMU
Détecte les doublons, anomalies et incohérences après un scraping
"""

import sqlite3
from datetime import date
import sys

DB_PATH = "data/database.db"

def print_section(title):
    """Affiche un titre de section"""
    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)

def check_duplicate_horses(con):
    """Vérifie les chevaux en double (même nom + date de naissance)"""
    print_section("VÉRIFICATION DES DOUBLONS DE CHEVAUX")
    
    cur = con.cursor()
    
    # Chevaux avec même nom ET même date de naissance
    cur.execute("""
        SELECT LOWER(nom) as nom_norm, date_naissance, COUNT(*) as cnt
        FROM chevaux
        WHERE date_naissance IS NOT NULL
        GROUP BY nom_norm, date_naissance
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC
    """)
    
    doublons_strict = cur.fetchall()
    
    if doublons_strict:
        print(f"\n❌ {len(doublons_strict)} doublons STRICTS trouvés (même nom + date de naissance):")
        for nom, dn, cnt in doublons_strict[:10]:
            print(f"   • {nom} (né le {dn}): {cnt} entrées")
            # Afficher les IDs
            cur.execute("""
                SELECT id_cheval, nom, date_naissance, race, sexe
                FROM chevaux
                WHERE LOWER(nom) = ? AND date_naissance = ?
            """, (nom, dn))
            for row in cur.fetchall():
                print(f"      → ID {row[0]}: {row[1]} | Race: {row[3]} | Sexe: {row[4]}")
        
        if len(doublons_strict) > 10:
            print(f"   ... et {len(doublons_strict) - 10} autres")
    else:
        print("\n✅ Aucun doublon strict détecté")
    
    # Chevaux avec même nom mais dates différentes (peut être normal)
    cur.execute("""
        SELECT LOWER(nom) as nom_norm, COUNT(*) as cnt, 
               COUNT(DISTINCT date_naissance) as nb_dates
        FROM chevaux
        GROUP BY nom_norm
        HAVING COUNT(*) > 1 AND COUNT(DISTINCT date_naissance) > 1
        ORDER BY cnt DESC
        LIMIT 20
    """)
    
    homonymes = cur.fetchall()
    
    if homonymes:
        print(f"\n⚠️  {len(homonymes)} homonymes trouvés (même nom, dates différentes):")
        print("    (Ceci peut être normal - plusieurs chevaux portent le même nom)")
        for nom, cnt, nb_dates in homonymes[:5]:
            print(f"   • {nom}: {cnt} chevaux, {nb_dates} dates différentes")
        
        if len(homonymes) > 5:
            print(f"   ... ({len(homonymes) - 5} autres homonymes)")
    
    return len(doublons_strict)

def check_duplicate_courses(con):
    """Vérifie les courses en double dans cheval_courses_seen"""
    print_section("VÉRIFICATION DES DOUBLONS DE COURSES")
    
    cur = con.cursor()
    
    # Compter les doublons (ne devrait pas exister avec PRIMARY KEY)
    cur.execute("""
        SELECT nom_norm, race_key, COUNT(*) as cnt
        FROM cheval_courses_seen
        GROUP BY nom_norm, race_key
        HAVING COUNT(*) > 1
    """)
    
    doublons = cur.fetchall()
    
    if doublons:
        print(f"\n❌ {len(doublons)} doublons de courses trouvés:")
        for nom, race_key, cnt in doublons[:10]:
            print(f"   • {nom} | {race_key}: {cnt} entrées")
    else:
        print("\n✅ Aucun doublon de course détecté")
    
    return len(doublons)

def check_data_coherence(con):
    """Vérifie la cohérence des données"""
    print_section("VÉRIFICATION DE LA COHÉRENCE DES DONNÉES")
    
    cur = con.cursor()
    issues = []
    
    # 1. Chevaux avec victoires > courses
    cur.execute("""
        SELECT id_cheval, nom, nombre_courses_total, nombre_victoires_total
        FROM chevaux
        WHERE nombre_victoires_total > nombre_courses_total
        AND nombre_courses_total IS NOT NULL
        AND nombre_victoires_total IS NOT NULL
    """)
    
    incoherent = cur.fetchall()
    if incoherent:
        print(f"\n❌ {len(incoherent)} chevaux avec plus de victoires que de courses:")
        for id_ch, nom, nbc, nbv in incoherent[:5]:
            print(f"   • {nom} (ID {id_ch}): {nbv} victoires > {nbc} courses")
        issues.append(("victoires > courses", len(incoherent)))
    else:
        print("\n✅ Cohérence victoires/courses: OK")
    
    # 2. Chevaux avec victoires 2025 > total victoires
    cur.execute("""
        SELECT id_cheval, nom, nombre_victoires_total, nombre_victoires_2025
        FROM chevaux
        WHERE nombre_victoires_2025 > nombre_victoires_total
        AND nombre_victoires_total IS NOT NULL
        AND nombre_victoires_2025 IS NOT NULL
    """)
    
    incoherent_2025 = cur.fetchall()
    if incoherent_2025:
        print(f"\n❌ {len(incoherent_2025)} chevaux: victoires 2025 > victoires totales:")
        for id_ch, nom, nbv_tot, nbv_2025 in incoherent_2025[:5]:
            print(f"   • {nom} (ID {id_ch}): 2025={nbv_2025} > total={nbv_tot}")
        issues.append(("victoires 2025 > total", len(incoherent_2025)))
    else:
        print("\n✅ Cohérence victoires 2025: OK")
    
    # 3. Chevaux sans nom
    cur.execute("SELECT COUNT(*) FROM chevaux WHERE nom IS NULL OR nom = ''")
    sans_nom = cur.fetchone()[0]
    if sans_nom:
        print(f"\n❌ {sans_nom} chevaux sans nom")
        issues.append(("sans nom", sans_nom))
    else:
        print("\n✅ Tous les chevaux ont un nom")
    
    # 4. Courses avec année incohérente
    cur.execute("""
        SELECT race_key, annee
        FROM cheval_courses_seen
        WHERE annee < 2020 OR annee > 2025
        LIMIT 10
    """)
    
    annees_bizarres = cur.fetchall()
    if annees_bizarres:
        print(f"\n⚠️  Courses avec années suspectes:")
        for race_key, annee in annees_bizarres:
            print(f"   • {race_key}: année {annee}")
        issues.append(("années suspectes", len(annees_bizarres)))
    else:
        print("\n✅ Toutes les années sont cohérentes")
    
    return len(issues)

def check_race_normalization(con):
    """Vérifie la normalisation des races"""
    print_section("VÉRIFICATION DE LA NORMALISATION DES RACES")
    
    cur = con.cursor()
    
    # Races avec variations suspectes
    cur.execute("""
        SELECT race, COUNT(*) as cnt
        FROM chevaux
        WHERE race IS NOT NULL
        GROUP BY race
        ORDER BY cnt DESC
    """)
    
    races = cur.fetchall()
    
    print(f"\n📋 {len(races)} races différentes trouvées")
    
    # Chercher des variations
    variations = {}
    for race, cnt in races:
        race_clean = race.replace('-', ' ').replace('*', '').strip().upper()
        if race_clean not in variations:
            variations[race_clean] = []
        variations[race_clean].append((race, cnt))
    
    problemes = [(k, v) for k, v in variations.items() if len(v) > 1]
    
    if problemes:
        print(f"\n⚠️  {len(problemes)} races avec variations détectées:")
        for race_norm, variants in problemes[:5]:
            print(f"\n   • {race_norm}:")
            for race_orig, cnt in variants:
                print(f"      → '{race_orig}': {cnt} chevaux")
    else:
        print("\n✅ Normalisation des races: OK")
    
    # Top 10 des races
    print("\n📊 Top 10 des races:")
    for race, cnt in races[:10]:
        print(f"   • {race}: {cnt} chevaux")
    
    return len(problemes)

def check_orphan_records(con):
    """Vérifie les enregistrements orphelins"""
    print_section("VÉRIFICATION DES ENREGISTREMENTS ORPHELINS")
    
    cur = con.cursor()
    
    # Courses pour des chevaux qui n'existent pas
    cur.execute("""
        SELECT DISTINCT ccs.nom_norm
        FROM cheval_courses_seen ccs
        LEFT JOIN chevaux c ON LOWER(c.nom) = ccs.nom_norm
        WHERE c.id_cheval IS NULL
        LIMIT 20
    """)
    
    orphelins = cur.fetchall()
    
    if orphelins:
        print(f"\n❌ {len(orphelins)} chevaux dans courses_seen mais pas dans chevaux:")
        for (nom,) in orphelins[:10]:
            print(f"   • {nom}")
        
        if len(orphelins) > 10:
            print(f"   ... et {len(orphelins) - 10} autres")
        
        return len(orphelins)
    else:
        print("\n✅ Aucun enregistrement orphelin")
        return 0

def check_recent_scraping(con):
    """Vérifie les données du scraping récent"""
    print_section("VÉRIFICATION DU SCRAPING RÉCENT")
    
    cur = con.cursor()
    today = date.today().isoformat()
    
    # Courses d'aujourd'hui
    cur.execute("""
        SELECT COUNT(DISTINCT race_key)
        FROM cheval_courses_seen
        WHERE race_key LIKE ?
    """, (f"{today}%",))
    
    nb_courses_today = cur.fetchone()[0]
    
    # Chevaux participant aujourd'hui
    cur.execute("""
        SELECT COUNT(DISTINCT nom_norm)
        FROM cheval_courses_seen
        WHERE race_key LIKE ?
    """, (f"{today}%",))
    
    nb_chevaux_today = cur.fetchone()[0]
    
    print(f"\n📅 Courses du {today}:")
    print(f"   • {nb_courses_today} courses scrapées")
    print(f"   • {nb_chevaux_today} chevaux participants")
    
    if nb_courses_today == 0:
        print(f"\n⚠️  Aucune course scrapée pour aujourd'hui")
        print("   (Normal si pas de programme ou pas encore scrapé)")
    else:
        # Moyenne de chevaux par course
        avg = nb_chevaux_today / nb_courses_today if nb_courses_today > 0 else 0
        print(f"   • Moyenne: {avg:.1f} chevaux/course")
        
        if avg < 5:
            print(f"\n⚠️  Moyenne de chevaux/course suspicieusement basse ({avg:.1f})")
        elif avg > 20:
            print(f"\n⚠️  Moyenne de chevaux/course suspicieusement haute ({avg:.1f})")
        else:
            print(f"\n✅ Moyenne de chevaux/course normale")

def generate_stats(con):
    """Génère des statistiques générales"""
    print_section("STATISTIQUES GÉNÉRALES")
    
    cur = con.cursor()
    
    # Total chevaux
    cur.execute("SELECT COUNT(*) FROM chevaux")
    total_chevaux = cur.fetchone()[0]
    
    # Total courses
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    total_courses = cur.fetchone()[0]
    
    # Courses uniques
    cur.execute("SELECT COUNT(DISTINCT race_key) FROM cheval_courses_seen")
    courses_uniques = cur.fetchone()[0]
    
    # Chevaux avec date de naissance
    cur.execute("SELECT COUNT(*) FROM chevaux WHERE date_naissance IS NOT NULL")
    chevaux_avec_dn = cur.fetchone()[0]
    
    # Chevaux avec race
    cur.execute("SELECT COUNT(*) FROM chevaux WHERE race IS NOT NULL")
    chevaux_avec_race = cur.fetchone()[0]
    
    print(f"""
🐴 Total chevaux: {total_chevaux:,}
   • Avec date de naissance: {chevaux_avec_dn:,} ({100*chevaux_avec_dn/total_chevaux:.1f}%)
   • Avec race: {chevaux_avec_race:,} ({100*chevaux_avec_race/total_chevaux:.1f}%)

🏁 Total enregistrements courses: {total_courses:,}
   • Courses uniques: {courses_uniques:,}
   • Moyenne participations/cheval: {total_courses/total_chevaux:.1f}

📅 Données 2025:
""")
    
    cur.execute("""
        SELECT 
            SUM(nombre_courses_2025) as courses,
            SUM(nombre_victoires_2025) as victoires
        FROM chevaux
    """)
    courses_2025, victoires_2025 = cur.fetchone()
    
    if courses_2025:
        print(f"   • Courses: {courses_2025:,}")
        print(f"   • Victoires: {victoires_2025:,}")
        print(f"   • Taux de victoire: {100*victoires_2025/courses_2025:.2f}%")

def fix_suggestions(issues):
    """Propose des corrections"""
    if not issues:
        return
    
    print_section("💡 SUGGESTIONS DE CORRECTION")
    
    print("\nPour corriger les problèmes détectés, vous pouvez:")
    print("\n1. Recalculer les totaux depuis l'historique:")
    print("   → Relance `recalc_totals_from_seen()` dans scraper_pmu_simple.py")
    
    print("\n2. Supprimer les doublons stricts:")
    print("   → Gardez le cheval avec le plus de données")
    
    print("\n3. Normaliser les races:")
    print("   → Utilisez normalize_races.py")

def main():
    print("=" * 80)
    print("🔍 VÉRIFICATION DE L'INTÉGRITÉ DE LA BASE DE DONNÉES PMU")
    print("=" * 80)
    
    try:
        con = sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"\n❌ Erreur de connexion à la base: {e}")
        return 1
    
    issues = []
    
    try:
        # Statistiques générales
        generate_stats(con)
        
        # Vérifications
        nb_doublons_chevaux = check_duplicate_horses(con)
        nb_doublons_courses = check_duplicate_courses(con)
        nb_incoherences = check_data_coherence(con)
        nb_races_pb = check_race_normalization(con)
        nb_orphelins = check_orphan_records(con)
        
        # Scraping récent
        check_recent_scraping(con)
        
        # Résumé
        print_section("📋 RÉSUMÉ")
        
        total_issues = (nb_doublons_chevaux + nb_doublons_courses + 
                       nb_incoherences + nb_races_pb + nb_orphelins)
        
        if total_issues == 0:
            print("\n✅ Aucun problème détecté ! Base de données en bon état.")
            print("\n🎉 Vous pouvez utiliser les données en toute confiance.")
        else:
            print(f"\n⚠️  {total_issues} problème(s) détecté(s):")
            if nb_doublons_chevaux:
                print(f"   • {nb_doublons_chevaux} doublons de chevaux")
            if nb_doublons_courses:
                print(f"   • {nb_doublons_courses} doublons de courses")
            if nb_incoherences:
                print(f"   • {nb_incoherences} incohérences de données")
            if nb_races_pb:
                print(f"   • {nb_races_pb} variations de races")
            if nb_orphelins:
                print(f"   • {nb_orphelins} enregistrements orphelins")
            
            fix_suggestions(issues)
        
        print("\n" + "=" * 80)
        
        return 0 if total_issues == 0 else 1
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la vérification: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        con.close()

if __name__ == "__main__":
    sys.exit(main())
