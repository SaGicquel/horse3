#!/usr/bin/env python3
"""
Script pour compléter les course_id et meeting_id manquants pour le 2025-11-03
Propage les métadonnées via MetadataCourseScraper pour toutes les courses de la journée
"""

import sys
import psycopg2
from db_connection import get_connection
from scrapers.metadata_course import MetadataCourseScraper

def get_courses_sans_metadata(date_str='2025-11-03'):
    """Récupère la liste des courses sans course_id/meeting_id"""
    conn = get_connection()
    cur = conn.cursor()
    
    query = """
        SELECT DISTINCT 
            reunion_numero,
            course_numero,
            hippodrome_code,
            hippodrome_nom,
            course_nom,
            COUNT(*) as nb_chevaux
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        AND course_id IS NULL
        GROUP BY reunion_numero, course_numero, hippodrome_code, hippodrome_nom, course_nom
        ORDER BY reunion_numero, course_numero
    """
    
    cur.execute(query, (f"{date_str}|%",))
    courses = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return courses

def propager_metadata(date_str, reunion_num, course_num):
    """Propage les métadonnées pour une course donnée"""
    try:
        MetadataCourseScraper.scrape_course(date_str, reunion_num, course_num)
        return True
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def main():
    date_str = '2025-11-03'
    
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    
    print(f"{'='*70}")
    print(f"  PROPAGATION DES MÉTADONNÉES POUR LE {date_str}")
    print(f"{'='*70}\n")
    
    # 1. Récupérer les courses à traiter
    print(f"📊 Recherche des courses sans metadata pour {date_str}...")
    courses = get_courses_sans_metadata(date_str)
    
    if not courses:
        print(f"✅ Toutes les courses du {date_str} ont déjà leurs métadonnées !")
        return
    
    print(f"   → {len(courses)} courses à traiter\n")
    
    # 2. Afficher la liste
    print("Liste des courses à traiter:")
    print("-" * 70)
    for reunion_num, course_num, hippo_code, hippo_nom, course_nom, nb_chevaux in courses:
        hippo_display = hippo_nom[:30] if hippo_nom else 'N/A'
        course_display = course_nom[:30] if course_nom else 'N/A'
        print(f"  R{reunion_num}C{course_num} - {hippo_code} ({hippo_display})")
        print(f"    Course: {course_display}")
        print(f"    Chevaux: {nb_chevaux}")
        print()
    
    # 3. Demander confirmation
    print("-" * 70)
    reponse = input(f"\n⚠️  Propager les métadonnées pour {len(courses)} courses ? (o/N): ")
    
    if reponse.lower() not in ['o', 'oui', 'y', 'yes']:
        print("❌ Opération annulée")
        return
    
    # 4. Traiter chaque course
    print(f"\n🔄 Propagation en cours...\n")
    
    conn = get_connection()
    success_count = 0
    error_count = 0
    
    for reunion_num, course_num, hippo_code, hippo_nom, course_nom, nb_chevaux in courses:
        hippo_display = hippo_nom[:25] if hippo_nom else 'N/A'
        print(f"  R{reunion_num}C{course_num} - {hippo_code} ({hippo_display})...", end=' ')
        
        if propager_metadata(date_str, reunion_num, course_num):
            conn.commit()
            success_count += 1
            print("✅")
        else:
            conn.rollback()
            error_count += 1
            print("❌")
    
    conn.close()
    
    # 5. Rapport final
    print(f"\n{'='*70}")
    print(f"  RAPPORT FINAL")
    print(f"{'='*70}")
    print(f"  ✅ Succès: {success_count}/{len(courses)}")
    print(f"  ❌ Erreurs: {error_count}/{len(courses)}")
    
    # 6. Vérification post-traitement
    print(f"\n📊 Vérification post-traitement...")
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN course_id IS NOT NULL THEN 1 END) as avec_course_id,
            COUNT(CASE WHEN meeting_id IS NOT NULL THEN 1 END) as avec_meeting_id
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
    """, (f"{date_str}|%",))
    
    total, avec_course, avec_meeting = cur.fetchone()
    
    pct_course = (avec_course / total * 100) if total > 0 else 0
    pct_meeting = (avec_meeting / total * 100) if total > 0 else 0
    
    print(f"  Total entrées: {total}")
    print(f"  Avec course_id: {avec_course} ({pct_course:.1f}%)")
    print(f"  Avec meeting_id: {avec_meeting} ({pct_meeting:.1f}%)")
    
    cur.close()
    conn.close()
    
    print(f"\n✅ Traitement terminé !")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
