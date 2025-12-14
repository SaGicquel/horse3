#!/usr/bin/env python3
"""
Script pour mettre à jour le nombre total de courses de chaque cheval
dans la table 'chevaux' en comptant les apparitions dans 'cheval_courses_seen'.

Usage:
    python update_nombre_courses.py
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "data/database.db"

def update_nombre_courses():
    """
    Met à jour la colonne nombre_courses_total dans la table chevaux
    en comptant les apparitions dans cheval_courses_seen.
    """
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Erreur : La base de données {DB_PATH} n'existe pas")
        return
    
    print(f"🔄 Connexion à la base de données : {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    try:
        # Vérifier que les tables existent
        cur.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('chevaux', 'cheval_courses_seen')
        """)
        tables = [row[0] for row in cur.fetchall()]
        
        if 'chevaux' not in tables:
            print("❌ Erreur : La table 'chevaux' n'existe pas")
            return
        
        if 'cheval_courses_seen' not in tables:
            print("❌ Erreur : La table 'cheval_courses_seen' n'existe pas")
            return
        
        # Vérifier si la colonne nombre_courses_total existe
        cur.execute("PRAGMA table_info(chevaux)")
        columns = [row[1] for row in cur.fetchall()]
        
        if 'nombre_courses_total' not in columns:
            print("⚠️  La colonne 'nombre_courses_total' n'existe pas, création...")
            cur.execute("ALTER TABLE chevaux ADD COLUMN nombre_courses_total INTEGER DEFAULT 0")
            con.commit()
        
        # Compter le nombre de chevaux avant mise à jour
        cur.execute("SELECT COUNT(*) FROM chevaux")
        total_chevaux = cur.fetchone()[0]
        print(f"📊 Nombre de chevaux dans la table : {total_chevaux}")
        
        # Compter le nombre d'entrées dans cheval_courses_seen
        cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
        total_courses_seen = cur.fetchone()[0]
        print(f"📊 Nombre d'entrées dans cheval_courses_seen : {total_courses_seen}")
        
        # Mise à jour du nombre de courses pour chaque cheval
        print("\n🔄 Mise à jour du nombre de courses...")
        
        cur.execute("""
            UPDATE chevaux
            SET nombre_courses_total = (
                SELECT COUNT(*) 
                FROM cheval_courses_seen s
                WHERE s.nom_norm = LOWER(chevaux.nom)
            )
        """)
        
        rows_updated = cur.rowcount
        con.commit()
        
        print(f"✅ Mise à jour terminée : {rows_updated} chevaux mis à jour")
        
        # Afficher quelques statistiques
        print("\n📈 Statistiques après mise à jour :")
        
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(nombre_courses_total) as moyenne,
                MAX(nombre_courses_total) as maximum,
                MIN(nombre_courses_total) as minimum
            FROM chevaux
            WHERE nombre_courses_total > 0
        """)
        
        stats = cur.fetchone()
        if stats and stats[0] > 0:
            print(f"  - Chevaux avec courses : {stats[0]}")
            print(f"  - Moyenne de courses : {stats[1]:.2f}")
            print(f"  - Maximum de courses : {stats[2]}")
            print(f"  - Minimum de courses : {stats[3]}")
        
        # Afficher le top 10 des chevaux avec le plus de courses
        print("\n🏆 Top 10 des chevaux avec le plus de courses :")
        cur.execute("""
            SELECT nom, nombre_courses_total
            FROM chevaux
            WHERE nombre_courses_total > 0
            ORDER BY nombre_courses_total DESC
            LIMIT 10
        """)
        
        for i, (nom, nb_courses) in enumerate(cur.fetchall(), 1):
            print(f"  {i}. {nom} : {nb_courses} courses")
        
        # Chevaux sans courses
        cur.execute("""
            SELECT COUNT(*) 
            FROM chevaux 
            WHERE nombre_courses_total IS NULL OR nombre_courses_total = 0
        """)
        chevaux_sans_courses = cur.fetchone()[0]
        if chevaux_sans_courses > 0:
            print(f"\n⚠️  {chevaux_sans_courses} chevaux n'ont aucune course enregistrée")
        
    except sqlite3.Error as e:
        print(f"❌ Erreur SQL : {e}")
        con.rollback()
    
    finally:
        con.close()
        print(f"\n✅ Script terminé à {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    print("=" * 60)
    print("🐴 Mise à jour du nombre de courses par cheval")
    print("=" * 60)
    print()
    
    update_nombre_courses()
