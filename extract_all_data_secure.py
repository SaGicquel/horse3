#!/usr/bin/env python3
"""
Extraction COMPLETE et SÉCURISÉE de toutes les données 2020-2025
Traitement par très petits chunks pour éviter tout crash
"""

import sys
import os
import time
import psycopg2
import csv
from datetime import datetime

# Configuration de la base de données
DB_CONFIG = {
    "host": "localhost",
    "port": 54624,
    "database": "pmu_database",
    "user": "pmu_user",
    "password": "pmu_secure_password_2025",
}


def extract_all_data_safe():
    """
    Extraction de TOUTES les données 2020-2025 par très petits chunks
    Ultra-sécurisé pour éviter tout crash du terminal
    """

    print("🚀 Début de l'extraction COMPLÈTE des données 2020-2025")
    print("⚡ Méthode ultra-sécurisée par petits chunks")

    output_file = "data/ml_features_COMPLETE_2020_2025.csv"

    # Connexion à la base
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✅ Connexion à la base de données réussie")
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return

    # 1. Compter le total disponible
    print("📊 Comptage des données disponibles...")
    cursor.execute("""
        SELECT COUNT(*)
        FROM cheval_courses_seen
        WHERE annee >= 2020
        AND annee <= 2025
        AND place_finale IS NOT NULL
    """)
    total_count = cursor.fetchone()[0]
    print(f"📈 TOTAL à extraire: {total_count:,} performances")

    # 2. Extraction par chunks de 2000 lignes (très petit pour sécurité)
    chunk_size = 2000
    offset = 0
    chunk_num = 0

    # Créer le fichier CSV
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = None

        while offset < total_count:
            chunk_num += 1

            print(f"🔄 Chunk {chunk_num} - Offset: {offset:,} ({offset/total_count*100:.1f}%)")

            # Requête pour ce chunk
            cursor.execute(
                """
                SELECT
                    row_number() OVER (ORDER BY annee, race_key) as id_performance,
                    race_key as id_course,
                    nom_norm,
                    annee,
                    place_finale as position_arrivee,
                    CASE WHEN place_finale = 1 THEN 1 ELSE 0 END as victoire,
                    CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as place,
                    numero_dossard as numero_corde,
                    cote_finale as cote_sp,
                    distance_m as distance,
                    discipline,
                    nombre_partants,
                    hippodrome_nom,
                    driver_jockey,
                    entraineur,
                    race_key as date_course,
                    sexe,
                    age
                FROM cheval_courses_seen
                WHERE annee >= 2020
                AND annee <= 2025
                AND place_finale IS NOT NULL
                ORDER BY annee, race_key
                LIMIT %s OFFSET %s
            """,
                (chunk_size, offset),
            )

            rows = cursor.fetchall()

            if not rows:
                print("✅ Fin des données")
                break

            # Écrire le header au premier chunk
            if writer is None:
                fieldnames = [
                    "id_performance",
                    "id_course",
                    "nom_norm",
                    "annee",
                    "position_arrivee",
                    "victoire",
                    "place",
                    "numero_corde",
                    "cote_sp",
                    "distance",
                    "discipline",
                    "nombre_partants",
                    "hippodrome_nom",
                    "driver_jockey",
                    "entraineur",
                    "date_course",
                    "sexe",
                    "age",
                ]
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                print("📝 Header CSV créé")

            # Écrire les données
            for row in rows:
                writer.writerow(row)

            offset += len(rows)

            print(f"✅ Chunk {chunk_num} terminé: {len(rows)} lignes - Total: {offset:,}")

            # Pause sécurité entre les chunks
            time.sleep(0.1)

            # Commit périodique pour libérer la mémoire
            if chunk_num % 10 == 0:
                csvfile.flush()
                print(f"💾 Flush fichier - Progression: {offset/total_count*100:.1f}%")

    # Fermeture propre
    cursor.close()
    conn.close()

    print("🎉 EXTRACTION COMPLÈTE TERMINÉE !")
    print(f"📄 Fichier créé: {output_file}")
    print(f"📊 Total extrait: {offset:,} performances")

    # Vérification finale
    print("🔍 Vérification finale du fichier...")
    try:
        with open(output_file, "r") as f:
            line_count = sum(1 for _ in f) - 1  # -1 pour le header
        print(f"✅ Vérification OK: {line_count:,} lignes dans le fichier")
    except Exception as e:
        print(f"⚠️  Erreur de vérification: {e}")


if __name__ == "__main__":
    extract_all_data_safe()
