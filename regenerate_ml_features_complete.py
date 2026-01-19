#!/usr/bin/env python3
"""
D1 - Régénération complète de ml_features.csv sur TOUTE la période historique 2020-2025
Traitement sécurisé par chunks avec features ML complètes
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path

# Configuration de la base de données
DB_CONFIG = {
    "host": "localhost",
    "port": 54624,
    "database": "pmu_database",
    "user": "pmu_user",
    "password": "pmu_secure_password_2025",
}


def regenerate_ml_features_complete():
    """
    Régénère ml_features.csv avec TOUTES les données 2020-2025
    Inclut toutes les features ML essentielles
    """

    print("🧮 D1 - RÉGÉNÉRATION COMPLÈTE DES FEATURES ML")
    print("📅 Période: 2020-2025 (661K performances)")
    print("⚡ Traitement ultra-sécurisé par chunks")

    # Fichiers de travail
    base_file = "data/ml_features_COMPLETE_2020_2025.csv"
    output_file = "data/ml_features.csv"
    temp_file = "data/ml_features_temp.csv"

    # Vérifier que le fichier de base existe
    if not Path(base_file).exists():
        print(f"❌ Fichier de base {base_file} introuvable")
        return

    print("📊 Lecture du dataset complet...")

    # Traitement par chunks pour éviter la surcharge mémoire
    chunk_size = 10000  # Chunks de 10K lignes pour sécurité
    chunk_num = 0
    total_processed = 0

    # Créer le fichier temporaire
    first_chunk = True

    print("🔄 Début du traitement par chunks...")

    for chunk_df in pd.read_csv(base_file, chunksize=chunk_size):
        chunk_num += 1

        print(f"📊 Traitement chunk {chunk_num} - {len(chunk_df)} lignes")

        # Ajouter toutes les features ML
        chunk_enhanced = add_complete_ml_features(chunk_df)

        # Sauvegarder le chunk
        mode = "w" if first_chunk else "a"
        header = first_chunk

        chunk_enhanced.to_csv(temp_file, mode=mode, header=header, index=False)

        total_processed += len(chunk_df)
        first_chunk = False

        print(
            f"✅ Chunk {chunk_num} traité. Total: {total_processed:,} lignes ({total_processed/661063*100:.1f}%)"
        )

        # Pause sécurité
        time.sleep(0.1)

        # Flush périodique
        if chunk_num % 20 == 0:
            print(f"💾 Checkpoint - {total_processed:,} lignes traitées")

    # Renommer le fichier temporaire
    if Path(temp_file).exists():
        Path(output_file).unlink(missing_ok=True)  # Supprimer l'ancien
        Path(temp_file).rename(output_file)
        print(f"✅ Fichier renommé: {output_file}")

    print("🎉 RÉGÉNÉRATION TERMINÉE !")
    print(f"📄 Nouveau ml_features.csv créé avec {total_processed:,} performances")
    print("📈 Période complète: 2020-2025")

    # Vérification finale
    verify_final_file(output_file)


def add_complete_ml_features(df):
    """
    Ajoute TOUTES les features ML essentielles au chunk
    """

    # Copie du dataframe pour éviter les modifications
    df = df.copy()

    # 1. FEATURES NUMÉRIQUES DE BASE
    df["distance_norm"] = df["distance"] / 1000.0  # Distance en km
    df["cote_log"] = np.log1p(df["cote_sp"].fillna(10.0))  # Log de la cote
    df["age_norm"] = df["age"].fillna(5) / 10.0  # Age normalisé

    # 2. FEATURES BINAIRES
    df["is_favori"] = (df["cote_sp"] < 5.0).astype(int)
    df["is_outsider"] = (df["cote_sp"] > 20.0).astype(int)
    df["is_longue_distance"] = (df["distance"] > 2000).astype(int)
    df["is_sprint"] = (df["distance"] < 1400).astype(int)
    df["is_male"] = (df["sexe"] == "HONGRES").astype(int)
    df["is_femelle"] = (df["sexe"] == "FEMELLES").astype(int)

    # 3. FEATURES DE POSITION
    df["corde_relative"] = df["numero_corde"] / df["nombre_partants"].clip(lower=1)
    df["corde_bonne"] = (df["numero_corde"] <= 3).astype(int)
    df["corde_mauvaise"] = (df["numero_corde"] > 10).astype(int)

    # 4. FEATURES TEMPORELLES
    df["annee_norm"] = (df["annee"] - 2020) / 5.0
    df["is_recent"] = (df["annee"] >= 2024).astype(int)

    # 5. FEATURES DE COURSE
    df["petite_course"] = (df["nombre_partants"] <= 8).astype(int)
    df["grande_course"] = (df["nombre_partants"] >= 16).astype(int)

    # 6. FEATURES COMBINÉES
    df["cote_corde_interaction"] = df["cote_log"] * df["corde_relative"]
    df["distance_age_interaction"] = df["distance_norm"] * df["age_norm"]
    df["favori_bonne_corde"] = df["is_favori"] * df["corde_bonne"]

    # 7. FEATURES DE DISCIPLINE (si pas déjà présentes)
    df["is_trot_attele"] = (df["discipline"] == "ATTELE").astype(int)
    df["is_trot_monte"] = (df["discipline"] == "MONTE").astype(int)
    df["is_galop"] = (df["discipline"] == "PLAT").astype(int)

    # 8. NETTOYAGE DES VALEURS MANQUANTES
    df["cote_sp"] = df["cote_sp"].fillna(10.0)
    df["age"] = df["age"].fillna(5)
    df["nombre_partants"] = df["nombre_partants"].fillna(12)

    return df


def verify_final_file(filepath):
    """Vérifie le fichier final"""
    try:
        # Compter les lignes
        with open(filepath, "r") as f:
            line_count = sum(1 for _ in f) - 1  # -1 pour le header

        # Lire un échantillon
        sample_df = pd.read_csv(filepath, nrows=5)

        print("🔍 Vérification finale:")
        print(f"   - Lignes: {line_count:,}")
        print(f"   - Colonnes: {len(sample_df.columns)}")
        print(f"   - Features ajoutées: {len(sample_df.columns) - 18} nouvelles colonnes")
        print("   - Échantillon OK: ✅")

    except Exception as e:
        print(f"⚠️  Erreur de vérification: {e}")


if __name__ == "__main__":
    regenerate_ml_features_complete()
