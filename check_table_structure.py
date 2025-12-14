#!/usr/bin/env python3
"""
Script pour vérifier la structure de la table et la plage de dates.
"""

import sys
from db_connection import get_connection

def check_table_structure():
    """Vérifie la structure de la table cheval_courses_seen"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Vérifier la structure de la table
        query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'cheval_courses_seen'
        ORDER BY ordinal_position;
        """
        
        cursor.execute(query)
        columns = cursor.fetchall()
        
        print(f"📋 Structure de la table cheval_courses_seen :")
        for col_name, col_type in columns:
            print(f"   {col_name}: {col_type}")
        
        # Chercher les colonnes qui pourraient contenir des dates
        date_columns = [col[0] for col in columns if 'date' in col[0].lower() or col[1] in ('date', 'timestamp', 'timestamp without time zone')]
        
        print(f"\n📅 Colonnes de type date trouvées : {date_columns}")
        
        # Si on trouve des colonnes de date, vérifier leur contenu
        if date_columns:
            for date_col in date_columns:
                try:
                    query_range = f"""
                    SELECT 
                        MIN({date_col}) as date_min,
                        MAX({date_col}) as date_max,
                        COUNT(*) as total_records
                    FROM cheval_courses_seen
                    WHERE {date_col} IS NOT NULL;
                    """
                    
                    cursor.execute(query_range)
                    result = cursor.fetchone()
                    
                    if result:
                        date_min, date_max, total_records = result
                        print(f"\n📊 Plage pour {date_col} :")
                        print(f"   Date minimum : {date_min}")
                        print(f"   Date maximum : {date_max}")
                        print(f"   Enregistrements avec cette date : {total_records:,}")
                        
                except Exception as e:
                    print(f"   Erreur pour {date_col}: {e}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification : {e}")
        return None

if __name__ == "__main__":
    check_table_structure()