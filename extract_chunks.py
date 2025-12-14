#!/usr/bin/env python3
"""
Extraction par chunks pour éviter la surcharge mémoire
"""

from db_connection import get_connection
import csv

def main():
    print("🚀 Extraction par chunks...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Extraction par chunks de 10k lignes
    offset = 0
    chunk_size = 10000
    total_processed = 0
    
    # Ouvrir le fichier CSV en écriture
    with open('data/ml_features_complete.csv', 'w', newline='', encoding='utf-8') as f:
        writer = None
        
        while True:
            print(f"📥 Chunk {offset//chunk_size + 1} (offset {offset})...")
            
            # Requête simple sans calculs complexes
            query = f"""
            SELECT 
                ROW_NUMBER() OVER (ORDER BY annee, race_key, numero_dossard) + {offset} as id_performance,
                race_key as id_course,
                nom_norm,
                annee,
                place_finale as position_arrivee,
                CASE WHEN place_finale = 1 THEN 1 ELSE 0 END as victoire,
                CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as place,
                numero_dossard as numero_corde,
                COALESCE(cote_finale, 5.0) as cote_sp,
                COALESCE(distance_m, 2000) as distance,
                COALESCE(discipline, 'Trot') as discipline,
                COALESCE(nombre_partants, 12) as nombre_partants,
                hippodrome_nom,
                driver_jockey,
                entraineur
            FROM cheval_courses_seen
            WHERE annee IS NOT NULL
              AND place_finale IS NOT NULL
              AND place_finale > 0
              AND COALESCE(non_partant, 0) = 0
            ORDER BY annee, race_key, numero_dossard
            LIMIT {chunk_size} OFFSET {offset}
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                break
            
            # Première fois : écrire les headers
            if writer is None:
                headers = [
                    'id_performance', 'id_course', 'nom_norm', 'annee', 'position_arrivee',
                    'victoire', 'place', 'numero_corde', 'cote_sp', 'distance', 
                    'discipline', 'nombre_partants', 'hippodrome_nom', 'driver_jockey', 'entraineur'
                ]
                writer = csv.writer(f)
                writer.writerow(headers)
            
            # Écrire les données
            writer.writerows(rows)
            
            total_processed += len(rows)
            print(f"   ✅ {len(rows):,} lignes traitées (total: {total_processed:,})")
            
            offset += chunk_size
            
            # Sécurité : arrêt si on dépasse 200k pour les tests
            if total_processed >= 200000:
                print("🛑 Arrêt à 200k pour test")
                break
    
    cursor.close()
    conn.close()
    
    print(f"✅ Extraction terminée : {total_processed:,} lignes")

if __name__ == '__main__':
    main()