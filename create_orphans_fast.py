#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script rapide pour créer les chevaux orphelins
"""

import sqlite3

DB_PATH = "data/database.db"

print("=" * 80)
print("✨ CRÉATION RAPIDE DES CHEVAUX ORPHELINS")
print("=" * 80)

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

# Créer un index temporaire pour accélérer
print("\n📊 Préparation...")
cur.execute("CREATE INDEX IF NOT EXISTS idx_chevaux_nom_lower ON chevaux(LOWER(nom))")
con.commit()

# Trouver les noms orphelins (simple et rapide)
print("🔍 Recherche des orphelins...")
cur.execute("""
    SELECT DISTINCT nom_norm 
    FROM cheval_courses_seen
    WHERE nom_norm NOT IN (SELECT LOWER(nom) FROM chevaux)
""")

orphelins = [row[0] for row in cur.fetchall()]

if not orphelins:
    print("✅ Aucun orphelin trouvé !")
    con.close()
    exit(0)

print(f"\n⚠️  {len(orphelins)} chevaux orphelins trouvés\n")

# Créer chaque cheval avec ses stats
for i, nom in enumerate(orphelins, 1):
    # Récupérer les stats
    cur.execute("""
        SELECT 
            COUNT(*),
            SUM(is_win),
            SUM(CASE WHEN annee = 2025 THEN 1 ELSE 0 END),
            SUM(CASE WHEN annee = 2025 AND is_win = 1 THEN 1 ELSE 0 END)
        FROM cheval_courses_seen
        WHERE nom_norm = ?
    """, (nom,))
    
    nb_courses, nb_vict, nb_2025, nbv_2025 = cur.fetchone()
    
    print(f"[{i}/{len(orphelins)}] {nom}: {nb_courses} courses, {nb_vict or 0} victoires")
    
    # Créer le cheval
    cur.execute("""
        INSERT INTO chevaux (
            nom,
            nombre_courses_total,
            nombre_victoires_total,
            nombre_courses_2025,
            nombre_victoires_2025,
            created_at
        ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (nom, nb_courses, nb_vict or 0, nb_2025, nbv_2025 or 0))

con.commit()
con.close()

print(f"\n✅ {len(orphelins)} chevaux créés avec succès !")
print("=" * 80)
