#!/usr/bin/env python3
"""
Script ultra-simple pour vérifier les données disponibles
"""

from db_connection import get_connection

def main():
    print("🔍 Vérification rapide des données...")
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Juste compter - pas de pandas
        cursor.execute("""
        SELECT COUNT(*) 
        FROM cheval_courses_seen
        WHERE annee IS NOT NULL
          AND place_finale IS NOT NULL
          AND place_finale > 0
          AND COALESCE(non_partant, 0) = 0
        """)
        
        total = cursor.fetchone()[0]
        print(f"📊 Total performances valides : {total:,}")
        
        # Par année
        cursor.execute("""
        SELECT annee, COUNT(*)
        FROM cheval_courses_seen
        WHERE annee IS NOT NULL
          AND place_finale IS NOT NULL
          AND place_finale > 0
          AND COALESCE(non_partant, 0) = 0
        GROUP BY annee
        ORDER BY annee
        """)
        
        print("\n📅 Par année :")
        for annee, nb in cursor.fetchall():
            print(f"   {annee}: {nb:,}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == '__main__':
    main()