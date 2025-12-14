#!/usr/bin/env python3
"""
Script de vérification automatique de la cohérence des données après scraping
Génère un rapport détaillé sur l'état des métadonnées
"""

import sys
from datetime import datetime
from db_connection import get_connection

def verifier_coherence(date_str=None):
    """Vérifie la cohérence des données pour une date donnée"""
    
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    conn = get_connection()
    cur = conn.cursor()
    
    print(f"\n{'='*70}")
    print(f"  VÉRIFICATION DE COHÉRENCE - {date_str}")
    print(f"{'='*70}\n")
    
    # 1. Statistiques générales
    print("1️⃣  STATISTIQUES GÉNÉRALES")
    print("-" * 70)
    
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT race_key) as nb_courses,
            COUNT(DISTINCT hippodrome_code) as nb_hippodromes,
            COUNT(CASE WHEN hippodrome_code IS NULL THEN 1 END) as null_hippo,
            COUNT(CASE WHEN course_id IS NULL THEN 1 END) as null_course_id,
            COUNT(CASE WHEN meeting_id IS NULL THEN 1 END) as null_meeting_id
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
    """, (f"{date_str}|%",))
    
    stats = cur.fetchone()
    
    if stats[0] == 0:
        print(f"   ⚠️  Aucune donnée trouvée pour le {date_str}")
        cur.close()
        conn.close()
        return
    
    total, nb_courses, nb_hippo, null_hippo, null_course, null_meeting = stats
    
    print(f"   Total participations: {total}")
    print(f"   Nombre de courses: {nb_courses}")
    print(f"   Nombre d'hippodromes: {nb_hippo}")
    print(f"   Hippodrome NULL: {null_hippo} ({null_hippo/total*100:.1f}%)")
    print(f"   course_id NULL: {null_course} ({null_course/total*100:.1f}%)")
    print(f"   meeting_id NULL: {null_meeting} ({null_meeting/total*100:.1f}%)")
    
    # 2. Répartition par hippodrome
    print(f"\n2️⃣  RÉPARTITION PAR HIPPODROME")
    print("-" * 70)
    
    cur.execute("""
        SELECT 
            hippodrome_code,
            LEFT(hippodrome_nom, 25) as hippo_short,
            COUNT(DISTINCT race_key) as nb_courses,
            COUNT(*) as nb_participations,
            COUNT(CASE WHEN course_id IS NOT NULL THEN 1 END) as avec_course_id,
            COUNT(CASE WHEN meeting_id IS NOT NULL THEN 1 END) as avec_meeting_id
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        GROUP BY hippodrome_code, hippo_short
        ORDER BY hippodrome_code
    """, (f"{date_str}|%",))
    
    hippodromes = cur.fetchall()
    
    for code, nom, courses, parts, with_course, with_meeting in hippodromes:
        code_display = code if code else 'NULL'
        nom_display = nom if nom else 'N/A'
        pct_course = (with_course / parts * 100) if parts > 0 else 0
        pct_meeting = (with_meeting / parts * 100) if parts > 0 else 0
        
        print(f"   {code_display:4s} {nom_display:25s} {courses:2d} courses, {parts:3d} parts")
        print(f"        → course_id: {pct_course:5.1f}%, meeting_id: {pct_meeting:5.1f}%")
    
    # 3. Courses sans métadonnées
    print(f"\n3️⃣  COURSES SANS MÉTADONNÉES COMPLÈTES")
    print("-" * 70)
    
    cur.execute("""
        SELECT 
            race_key,
            hippodrome_code,
            LEFT(course_nom, 30) as course_short,
            COUNT(*) as nb_chevaux,
            MAX(course_id) as course_id_sample,
            MAX(meeting_id) as meeting_id_sample
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        AND (course_id IS NULL OR meeting_id IS NULL)
        GROUP BY race_key, hippodrome_code, course_short
        ORDER BY race_key
        LIMIT 20
    """, (f"{date_str}|%",))
    
    courses_incomplete = cur.fetchall()
    
    if not courses_incomplete:
        print("   ✅ Toutes les courses ont leurs métadonnées complètes !")
    else:
        print(f"   ⚠️  {len(courses_incomplete)} courses sans métadonnées complètes")
        print()
        for race_key, code, course_nom, nb_chev, course_id, meeting_id in courses_incomplete[:10]:
            code_display = code if code else 'NULL'
            course_display = course_nom if course_nom else 'N/A'
            print(f"   {race_key} - {code_display}")
            print(f"      Course: {course_display}")
            print(f"      Chevaux: {nb_chev}, course_id: {course_id}, meeting_id: {meeting_id}")
    
    # 4. Doublons potentiels
    print(f"\n4️⃣  VÉRIFICATION DES DOUBLONS")
    print("-" * 70)
    
    cur.execute("""
        SELECT race_key, COUNT(*) as cnt
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        GROUP BY race_key
        HAVING COUNT(*) > 20  -- Plus de 20 chevaux = suspect
        ORDER BY cnt DESC
        LIMIT 10
    """, (f"{date_str}|%",))
    
    doublons = cur.fetchall()
    
    if not doublons:
        print("   ✅ Aucun doublon détecté (courses avec >20 partants)")
    else:
        print(f"   ⚠️  {len(doublons)} courses avec un nombre inhabituel de partants:")
        for race_key, cnt in doublons:
            print(f"      {race_key}: {cnt} partants")
    
    # 5. Score de qualité global
    print(f"\n5️⃣  SCORE DE QUALITÉ GLOBAL")
    print("-" * 70)
    
    score_hippo = ((total - null_hippo) / total * 100) if total > 0 else 0
    score_course_id = ((total - null_course) / total * 100) if total > 0 else 0
    score_meeting_id = ((total - null_meeting) / total * 100) if total > 0 else 0
    score_global = (score_hippo + score_course_id + score_meeting_id) / 3
    
    def get_emoji(score):
        if score >= 95: return "✅"
        elif score >= 75: return "🟢"
        elif score >= 50: return "🟡"
        elif score >= 25: return "🟠"
        else: return "🔴"
    
    print(f"   hippodrome_code: {get_emoji(score_hippo)} {score_hippo:5.1f}%")
    print(f"   course_id:       {get_emoji(score_course_id)} {score_course_id:5.1f}%")
    print(f"   meeting_id:      {get_emoji(score_meeting_id)} {score_meeting_id:5.1f}%")
    print(f"   {'─'*30}")
    print(f"   SCORE GLOBAL:    {get_emoji(score_global)} {score_global:5.1f}%")
    
    # 6. Recommandations
    print(f"\n6️⃣  RECOMMANDATIONS")
    print("-" * 70)
    
    actions = []
    
    if null_course > 0:
        nb_courses_incomplete = len(courses_incomplete)
        actions.append(f"   🔧 Exécuter MetadataCourseScraper pour {nb_courses_incomplete} courses")
        actions.append(f"      → python3 completer_metadata_{date_str}.py")
    
    if null_hippo > 0:
        actions.append(f"   🔧 {null_hippo} participations sans hippodrome_code")
        actions.append(f"      → Vérifier le scraper principal")
    
    if score_global >= 95:
        actions.append("   ✅ Qualité excellente - Aucune action requise")
    elif not actions:
        actions.append("   ✅ Données cohérentes - Monitoring standard")
    
    for action in actions:
        print(action)
    
    cur.close()
    conn.close()
    
    print(f"\n{'='*70}")
    print("✅ Vérification terminée")
    print(f"{'='*70}\n")

def main():
    date_str = None
    
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    
    verifier_coherence(date_str)

if __name__ == '__main__':
    main()
