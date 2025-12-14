#!/usr/bin/env python3
"""
Script pour vérifier la structure des tables normalisées.
"""

import sys
from db_connection import get_connection

def check_table_schemas():
    """Vérifie la structure des tables normalisées"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        tables = ['chevaux', 'jockeys', 'entraineurs', 'hippodromes', 'courses', 'performances']
        
        for table in tables:
            try:
                query = f"""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
                ORDER BY ordinal_position;
                """
                
                cursor.execute(query)
                columns = cursor.fetchall()
                
                print(f"📋 Table {table} :")
                if columns:
                    for col_name, col_type, nullable in columns:
                        nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
                        print(f"   {col_name}: {col_type} ({nullable_str})")
                else:
                    print(f"   ❌ Table {table} n'existe pas ou est vide")
                print()
                
            except Exception as e:
                print(f"   ❌ Erreur pour {table}: {e}")
                print()
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_table_schemas()