#!/usr/bin/env python3
"""
Script pour vérifier la plage de dates disponible dans la BDD.
"""

import sys
from db_connection import get_connection

def check_date_range():
    """Vérifie la plage de dates dans cheval_courses_seen"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Vérifier la plage de dates
        query = """
        SELECT 
            MIN(date_course) as date_min,
            MAX(date_course) as date_max,
            COUNT(*) as total_records,
            COUNT(DISTINCT date_course) as total_days
        FROM cheval_courses_seen;
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            date_min, date_max, total_records, total_days = result
            print(f"📅 Plage de dates dans la BDD :")
            print(f"   Date minimum : {date_min}")
            print(f"   Date maximum : {date_max}")
            print(f"   Total enregistrements : {total_records:,}")
            print(f"   Total jours uniques : {total_days:,}")
            
            # Calculer la différence en jours
            if date_min and date_max:
                diff_days = (date_max - date_min).days
                print(f"   Période couverte : {diff_days:,} jours")
                
        cursor.close()
        conn.close()
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification : {e}")
        return None

if __name__ == "__main__":
    check_date_range()