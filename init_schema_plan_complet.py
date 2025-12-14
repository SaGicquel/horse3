#!/usr/bin/env python3
"""
Script d'initialisation du nouveau schéma conforme au plan complet.
Crée toutes les tables, index, vues, fonctions et triggers.

Usage:
    python init_schema_plan_complet.py
"""

import sys
from pathlib import Path
from db_connection import get_connection

def init_schema():
    """Initialise le schéma complet depuis le fichier SQL."""
    
    schema_file = Path(__file__).parent / "schema_plan_complet_v1.sql"
    
    if not schema_file.exists():
        print(f"❌ Fichier schema introuvable : {schema_file}")
        return False
    
    print(f"📋 Lecture du schéma : {schema_file}")
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    print("🔌 Connexion à la base de données...")
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        print("🏗️  Exécution du schéma...")
        cur.execute(schema_sql)
        
        conn.commit()
        
        print("\n✅ Schéma créé avec succès !")
        
        # Vérification des tables créées
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = cur.fetchall()
        
        print(f"\n📊 {len(tables)} tables créées :")
        for table in tables:
            print(f"   - {table[0]}")
        
        # Vérification des vues
        cur.execute("""
            SELECT table_name 
            FROM information_schema.views 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        views = cur.fetchall()
        
        if views:
            print(f"\n👁️  {len(views)} vues créées :")
            for view in views:
                print(f"   - {view[0]}")
        
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la création du schéma :")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_schema():
    """Vérifie que le schéma est bien en place."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Tables attendues (Phase 1)
        expected_tables = [
            'hippodromes',
            'courses',
            'chevaux',
            'personnes',
            'performances',
            'cotes_historiques',
            'temps_sectionnels',
            'stats_chevaux',
            'stats_personnes'
        ]
        
        print("\n🔍 Vérification du schéma...")
        
        for table in expected_tables:
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = '{table}'
            """)
            count = cur.fetchone()[0]
            
            status = "✅" if count > 0 else "❌"
            print(f"   {status} Table '{table}'")
        
        cur.close()
        conn.close()
        
        print("\n✅ Vérification terminée")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la vérification : {e}")
        return False

def show_summary():
    """Affiche un résumé du contenu de la base."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        print("\n📊 Résumé du contenu de la base :")
        
        tables = [
            'hippodromes',
            'courses',
            'chevaux',
            'personnes',
            'performances',
            'cotes_historiques',
            'temps_sectionnels',
            'stats_chevaux',
            'stats_personnes'
        ]
        
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"   {table:25s} : {count:6d} lignes")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"\n⚠️  Impossible d'afficher le résumé : {e}")

if __name__ == '__main__':
    print("=" * 70)
    print("🏇 INITIALISATION SCHEMA PMU - VERSION PLAN COMPLET")
    print("=" * 70)
    
    if init_schema():
        verify_schema()
        show_summary()
        print("\n🎉 Base de données prête à l'emploi !")
        sys.exit(0)
    else:
        print("\n💥 Échec de l'initialisation")
        sys.exit(1)
