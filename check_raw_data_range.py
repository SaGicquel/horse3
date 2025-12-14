#!/usr/bin/env python3
"""
Script pour vérifier la plage de dates dans les données brutes (cheval_courses_seen).
"""

import sys
from db_connection import get_connection

def check_raw_data_range():
    """Vérifie la plage de dates dans cheval_courses_seen (données brutes)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # D'abord, construisons une date à partir de l'année et d'autres champs
        query = """
        SELECT 
            MIN(annee) as annee_min,
            MAX(annee) as annee_max,
            COUNT(*) as total_records,
            COUNT(DISTINCT annee) as total_annees,
            COUNT(DISTINCT race_key) as total_courses_uniques
        FROM cheval_courses_seen
        WHERE annee IS NOT NULL;
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            annee_min, annee_max, total_records, total_annees, total_courses = result
            print(f"📅 Plage des données brutes (cheval_courses_seen) :")
            print(f"   Année minimum : {annee_min}")
            print(f"   Année maximum : {annee_max}")
            print(f"   Total enregistrements : {total_records:,}")
            print(f"   Années uniques : {total_annees:,}")
            print(f"   Courses uniques (race_key) : {total_courses:,}")
        
        # Vérifions aussi la répartition par année
        query_by_year = """
        SELECT 
            annee,
            COUNT(*) as nb_performances,
            COUNT(DISTINCT race_key) as nb_courses
        FROM cheval_courses_seen
        WHERE annee IS NOT NULL
        GROUP BY annee
        ORDER BY annee;
        """
        
        cursor.execute(query_by_year)
        results_by_year = cursor.fetchall()
        
        print(f"\n📊 Répartition par année :")
        for annee, nb_perf, nb_courses in results_by_year:
            print(f"   {annee}: {nb_perf:,} performances, {nb_courses:,} courses")
        
        # Comparons avec les tables normalisées
        query_normalized = """
        SELECT 
            EXTRACT(YEAR FROM date_course) as annee,
            COUNT(*) as nb_performances,
            COUNT(DISTINCT c.id_course) as nb_courses
        FROM performances p
        JOIN courses c ON p.id_course = c.id_course
        WHERE p.non_partant = FALSE
          AND p.position_arrivee IS NOT NULL
          AND p.position_arrivee > 0
        GROUP BY EXTRACT(YEAR FROM date_course)
        ORDER BY annee;
        """
        
        cursor.execute(query_normalized)
        results_normalized = cursor.fetchall()
        
        print(f"\n🔄 Tables normalisées (pour comparaison) :")
        for annee, nb_perf, nb_courses in results_normalized:
            print(f"   {int(annee)}: {nb_perf:,} performances, {nb_courses:,} courses")
        
        cursor.close()
        conn.close()
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification : {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    check_raw_data_range()