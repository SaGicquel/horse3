#!/usr/bin/env python3
"""
Script ultra-simple pour extraire TOUTES les données historiques
"""
import sys
sys.path.append('/Users/gicquelsacha/horse3')

from db_connection import get_connection
import csv

def extract_all_data():
    print("🚀 Extraction ultra-simple des données complètes...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Simple requête pour tout extraire
    query = """
    SELECT 
        race_key,
        nom_norm,
        annee,
        place_finale,
        numero_dossard,
        cote_finale,
        distance_m,
        discipline,
        hippodrome_nom,
        driver_jockey,
        entraineur,
        nombre_partants
    FROM cheval_courses_seen
    WHERE annee IS NOT NULL
      AND place_finale IS NOT NULL
      AND place_finale > 0
      AND COALESCE(non_partant, 0) = 0
    ORDER BY annee, race_key
    """
    
    print("📊 Exécution de la requête...")
    cursor.execute(query)
    
    # Écrire directement dans un CSV
    print("💾 Écriture du CSV...")
    with open('data/ml_features_all_raw.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # En-têtes
        writer.writerow([
            'race_key', 'nom_norm', 'annee', 'place_finale', 'numero_dossard',
            'cote_finale', 'distance_m', 'discipline', 'hippodrome_nom', 
            'driver_jockey', 'entraineur', 'nombre_partants'
        ])
        
        # Données par chunks pour éviter de saturer la mémoire
        rows_written = 0
        while True:
            rows = cursor.fetchmany(10000)  # Par chunks de 10K
            if not rows:
                break
            
            for row in rows:
                writer.writerow(row)
                rows_written += 1
                
            if rows_written % 50000 == 0:
                print(f"   {rows_written:,} lignes écrites...")
    
    cursor.close()
    conn.close()
    
    print(f"✅ Terminé ! {rows_written:,} lignes extraites")
    print(f"📁 Fichier: data/ml_features_all_raw.csv")

if __name__ == '__main__':
    extract_all_data()