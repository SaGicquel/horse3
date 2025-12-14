#!/usr/bin/env python3
"""
VÉRIFICATION COMPLÈTE DE LA BASE DE DONNÉES PMU
Analyse la qualité et la complétude des données.
"""

from db_connection import get_connection
from datetime import datetime

def check_database():
    """Vérifie la complétude et qualité des données."""
    conn = get_connection()
    cur = conn.cursor()
    
    print("\n" + "="*70)
    print("🔍 VÉRIFICATION COMPLÈTE BASE DE DONNÉES PMU")
    print("="*70)
    print(f"📅 Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. VOLUMÉTRIE
    print("="*70)
    print("📊 VOLUMÉTRIE")
    print("="*70)
    
    tables = ['hippodromes', 'courses', 'chevaux', 'personnes', 
              'performances', 'stats_chevaux', 'stats_personnes']
    
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"   {table:25s} : {count:8d} lignes")
    
    # 2. PÉRIODE COUVERTE
    print(f"\n{'='*70}")
    print("📅 PÉRIODE COUVERTE")
    print("="*70)
    
    cur.execute("""
        SELECT 
            MIN(date_course) as premiere_course,
            MAX(date_course) as derniere_course,
            COUNT(DISTINCT date_course) as nb_jours,
            COUNT(*) as nb_courses
        FROM courses
    """)
    row = cur.fetchone()
    if row[0]:
        print(f"   Première course : {row[0]}")
        print(f"   Dernière course : {row[1]}")
        print(f"   Jours distincts : {row[2]}")
        print(f"   Total courses   : {row[3]}")
    
    # 3. RÉPARTITION PAR DISCIPLINE
    print(f"\n{'='*70}")
    print("🏇 RÉPARTITION PAR DISCIPLINE")
    print("="*70)
    
    cur.execute("""
        SELECT 
            discipline,
            COUNT(*) as nb_courses,
            SUM(nombre_partants) as total_partants
        FROM courses
        GROUP BY discipline
        ORDER BY nb_courses DESC
    """)
    
    for row in cur.fetchall():
        print(f"   {row[0]:15s} : {row[1]:4d} courses, {row[2]:5d} partants")
    
    # 4. TOP HIPPODROMES
    print(f"\n{'='*70}")
    print("🏟️  TOP 10 HIPPODROMES")
    print("="*70)
    
    cur.execute("""
        SELECT 
            h.nom_hippodrome,
            h.code_pmu,
            COUNT(*) as nb_courses
        FROM courses c
        JOIN hippodromes h ON c.id_hippodrome = h.id_hippodrome
        GROUP BY h.id_hippodrome, h.nom_hippodrome, h.code_pmu
        ORDER BY nb_courses DESC
        LIMIT 10
    """)
    
    for row in cur.fetchall():
        print(f"   {row[0]:35s} ({row[1]}) : {row[2]:4d} courses")
    
    # 5. QUALITÉ DES DONNÉES
    print(f"\n{'='*70}")
    print("✅ QUALITÉ DES DONNÉES")
    print("="*70)
    
    # Performances avec résultats
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN position_arrivee IS NOT NULL THEN 1 ELSE 0 END) as avec_position,
            SUM(CASE WHEN cote_sp IS NOT NULL THEN 1 ELSE 0 END) as avec_cote,
            SUM(CASE WHEN musique IS NOT NULL THEN 1 ELSE 0 END) as avec_musique,
            SUM(CASE WHEN temps_total IS NOT NULL THEN 1 ELSE 0 END) as avec_temps
        FROM performances
    """)
    
    row = cur.fetchone()
    total = row[0]
    print(f"   Performances totales        : {total:6d}")
    print(f"   Avec position arrivée       : {row[1]:6d} ({row[1]*100/total if total > 0 else 0:.1f}%)")
    print(f"   Avec cote                   : {row[2]:6d} ({row[2]*100/total if total > 0 else 0:.1f}%)")
    print(f"   Avec musique                : {row[3]:6d} ({row[3]*100/total if total > 0 else 0:.1f}%)")
    print(f"   Avec temps                  : {row[4]:6d} ({row[4]*100/total if total > 0 else 0:.1f}%)")
    
    # Chevaux avec pedigree
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN nom_pere IS NOT NULL THEN 1 ELSE 0 END) as avec_pere,
            SUM(CASE WHEN nom_mere IS NOT NULL THEN 1 ELSE 0 END) as avec_mere,
            SUM(CASE WHEN proprietaire IS NOT NULL THEN 1 ELSE 0 END) as avec_proprio
        FROM chevaux
    """)
    
    row = cur.fetchone()
    total = row[0]
    print(f"\n   Chevaux totaux              : {total:6d}")
    print(f"   Avec nom père               : {row[1]:6d} ({row[1]*100/total if total > 0 else 0:.1f}%)")
    print(f"   Avec nom mère               : {row[2]:6d} ({row[2]*100/total if total > 0 else 0:.1f}%)")
    print(f"   Avec propriétaire           : {row[3]:6d} ({row[3]*100/total if total > 0 else 0:.1f}%)")
    
    # 6. STATISTIQUES CALCULÉES
    print(f"\n{'='*70}")
    print("📈 STATISTIQUES CALCULÉES")
    print("="*70)
    
    cur.execute("SELECT COUNT(*) FROM stats_chevaux WHERE nb_courses_total > 0")
    nb_stats_chevaux = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM stats_personnes WHERE nb_courses > 0")
    nb_stats_personnes = cur.fetchone()[0]
    
    print(f"   Chevaux avec stats          : {nb_stats_chevaux:6d}")
    print(f"   Personnes avec stats        : {nb_stats_personnes:6d}")
    
    # 7. INTÉGRITÉ RÉFÉRENTIELLE
    print(f"\n{'='*70}")
    print("🔗 INTÉGRITÉ RÉFÉRENTIELLE")
    print("="*70)
    
    # Performances orphelines
    cur.execute("""
        SELECT COUNT(*) FROM performances p
        LEFT JOIN chevaux c ON p.id_cheval = c.id_cheval
        WHERE c.id_cheval IS NULL
    """)
    orphan_perf = cur.fetchone()[0]
    
    status = "✅" if orphan_perf == 0 else "⚠️ "
    print(f"   {status} Performances sans cheval    : {orphan_perf}")
    
    # Courses orphelines
    cur.execute("""
        SELECT COUNT(*) FROM courses c
        LEFT JOIN hippodromes h ON c.id_hippodrome = h.id_hippodrome
        WHERE h.id_hippodrome IS NULL
    """)
    orphan_courses = cur.fetchone()[0]
    
    status = "✅" if orphan_courses == 0 else "⚠️ "
    print(f"   {status} Courses sans hippodrome     : {orphan_courses}")
    
    # 8. RECOMMANDATIONS
    print(f"\n{'='*70}")
    print("💡 RECOMMANDATIONS")
    print("="*70)
    
    recommendations = []
    
    # Vérifier si stats à recalculer
    cur.execute("""
        SELECT COUNT(*) FROM chevaux ch
        WHERE NOT EXISTS (SELECT 1 FROM stats_chevaux sc WHERE sc.id_cheval = ch.id_cheval)
        AND EXISTS (SELECT 1 FROM performances p WHERE p.id_cheval = ch.id_cheval)
    """)
    chevaux_sans_stats = cur.fetchone()[0]
    
    if chevaux_sans_stats > 0:
        recommendations.append(f"Recalculer stats pour {chevaux_sans_stats} chevaux : python calcul_stats.py --all")
    
    # Vérifier période récente
    cur.execute("SELECT MAX(date_course) FROM courses")
    last_date = cur.fetchone()[0]
    if last_date:
        from datetime import date
        days_ago = (date.today() - last_date).days
        if days_ago > 1:
            recommendations.append(f"Mettre à jour données (dernier scraping : {days_ago} jours)")
            recommendations.append(f"Commande : python scraper_pmu_adapter.py --date today")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("   ✅ Aucune recommandation - Base à jour")
    
    print("\n" + "="*70)
    print("✅ VÉRIFICATION TERMINÉE")
    print("="*70 + "\n")
    
    cur.close()
    conn.close()

if __name__ == '__main__':
    check_database()
