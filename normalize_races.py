#!/usr/bin/env python3
"""
Script pour normaliser les noms de races dans la base de données.
Unifie les variations comme "PUR SANG"/"PUR-SANG" et "ANGLO-ARABE"/"*ANGLO-ARABE*"
"""

import sqlite3


def normalize_race_name(race_name):
    """
    Normalise un nom de race en enlevant les caractères spéciaux
    et en standardisant les tirets/espaces.
    """
    if not race_name:
        return None

    # Enlever les astérisques
    normalized = race_name.strip().replace("*", "")

    # Normaliser les tirets et espaces
    # Remplacer les tirets par des espaces pour uniformiser
    normalized = normalized.replace("-", " ")

    # Gérer les cas spéciaux pour standardiser
    # "PUR SANG" devient la forme standard (sans tiret)
    # "ANGLO ARABE" devient la forme standard (sans tiret)

    return normalized.strip()


def normalize_database_races(db_path="data/database.db"):
    """
    Normalise toutes les races dans la base de données.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("🔍 Analyse des races actuelles...")

    # Récupérer toutes les races distinctes
    cur.execute(
        "SELECT DISTINCT race, COUNT(*) as count FROM chevaux WHERE race IS NOT NULL GROUP BY race ORDER BY count DESC"
    )
    races = cur.fetchall()

    print("\n📊 Races trouvées avant normalisation:")
    for race, count in races:
        normalized = normalize_race_name(race)
        if race != normalized:
            print(f"   ❌ '{race}' ({count} chevaux) → '{normalized}'")
        else:
            print(f"   ✓ '{race}' ({count} chevaux) [déjà normalisé]")

    # Créer un mapping des races à normaliser
    race_mapping = {}
    for race, _ in races:
        normalized = normalize_race_name(race)
        if race != normalized:
            race_mapping[race] = normalized

    if not race_mapping:
        print("\n✅ Aucune race à normaliser!")
        conn.close()
        return

    print(f"\n🔧 Normalisation de {len(race_mapping)} races...")

    # Appliquer les normalisations
    total_updated = 0
    for old_race, new_race in race_mapping.items():
        cur.execute("UPDATE chevaux SET race = ? WHERE race = ?", (new_race, old_race))
        updated = cur.rowcount
        total_updated += updated
        print(f"   ✓ '{old_race}' → '{new_race}' ({updated} chevaux mis à jour)")

    conn.commit()

    print(f"\n✅ {total_updated} chevaux mis à jour au total")

    # Vérifier le résultat
    print("\n📊 Races après normalisation:")
    cur.execute(
        "SELECT DISTINCT race, COUNT(*) as count FROM chevaux WHERE race IS NOT NULL GROUP BY race ORDER BY count DESC"
    )
    races_after = cur.fetchall()

    for race, count in races_after:
        print(f"   • {race}: {count} chevaux")

    conn.close()


if __name__ == "__main__":
    print("🐴 Normalisation des noms de races\n")
    normalize_database_races()
    print("\n✅ Normalisation terminée!")
