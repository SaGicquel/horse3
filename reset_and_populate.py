import sqlite3
import csv
import os
from datetime import datetime


def reset_database():
    """Supprime toutes les tables et recrée seulement la table chevaux"""

    # Connexion à la base SQLite
    conn = sqlite3.connect("data/database.db")
    cursor = conn.cursor()

    print("🗑️  Suppression de toutes les tables existantes...")

    # Récupérer la liste de toutes les tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Supprimer toutes les tables (sauf sqlite_sequence)
    for table in tables:
        table_name = table[0]
        if table_name != "sqlite_sequence":
            print(f"   - Suppression de la table: {table_name}")
            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")

    print("✅ Toutes les tables supprimées")

    # Créer la table chevaux
    print("\n🏗️  Création de la table chevaux...")
    cursor.execute("""
    CREATE TABLE chevaux (
        id_cheval INTEGER PRIMARY KEY AUTOINCREMENT,
        nom TEXT NOT NULL UNIQUE,
        race TEXT,
        sexe TEXT,
        robe TEXT,
        date_naissance DATE,
        pays_naissance TEXT,
        entraineur_courant TEXT,
        jockey_habituel TEXT,
        nombre_courses_total INTEGER DEFAULT 0,
        nombre_victoires_total INTEGER DEFAULT 0,
        dernier_poids_couru REAL,
        dernier_resultat TEXT,
        meilleur_temps_distance TEXT,
        surface_preferee TEXT,
        distance_preferee TEXT,
        date_derniere_course DATE,
        condition_physique TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    print("✅ Table chevaux créée")

    # Races acceptées pour chevaux de COURSE
    RACES_COURSE = [
        "PUR SANG",
        "AQPS",
        "TROTTEUR FRANCAIS",
        "TROTTEUR ANGLO-NORMAND",
        "ANGLO-ARABE",
        "POSTIER BRETON",
    ]

    print("\n📥 Importation des chevaux depuis le fichier CSV...")
    print("🔍 Critères de filtrage:")
    print("   ❌ CHE_COCONSO = 'O' (destiné à la consommation)")
    print("   ❌ DATE_DE_DECES non vide (cheval mort)")
    print("   ❌ Date naissance avant 2005")
    print("   ✅ Race = course hippique")
    print()

    imported = 0
    skipped_conso = 0
    skipped_mort = 0
    skipped_race = 0
    skipped_age = 0

    try:
        with open("fichier-des-equides.csv", "r", encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            for row_num, row in enumerate(csv_reader, start=2):
                # Filtre 1 : Rejeter si destiné à la consommation
                if row.get("CHE_COCONSO", "").strip() == "O":
                    skipped_conso += 1
                    continue

                # Filtre 2 : Rejeter si mort
                date_mort = row.get("DATE_DE_DECES", "").strip()
                if date_mort and date_mort != "":
                    skipped_mort += 1
                    continue

                # Filtre 3 : Accepter seulement les races de course
                race = row.get("RACE", "").strip().upper()
                if not any(r in race for r in RACES_COURSE):
                    skipped_race += 1
                    continue

                # Filtre 4 : Exclure chevaux nés avant 2005
                date_naissance_str = row.get("DATE_DE_NAISSANCE", "").strip()
                if date_naissance_str:
                    try:
                        date_naissance = datetime.strptime(date_naissance_str, "%d/%m/%Y")
                    except ValueError:
                        # Date invalide, ignorer ce cheval
                        skipped_age += 1
                        continue
                    if date_naissance.year < 2005:
                        skipped_age += 1
                        continue
                else:
                    # Pas de date de naissance, ignorer ce cheval
                    skipped_age += 1
                    continue

                # Insérer le cheval
                try:
                    cursor.execute(
                        """
                    INSERT OR IGNORE INTO chevaux
                    (nom, race, sexe, robe, date_naissance, pays_naissance)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            row.get("NOM", "").strip(),
                            race,
                            row.get("SEXE", "").strip(),
                            row.get("ROBE", "").strip(),
                            date_naissance_str or None,
                            row.get("PAYS_DE_NAISSANCE", "").strip(),
                        ),
                    )
                    imported += 1

                    # Afficher la progression tous les 1000 chevaux
                    if imported % 1000 == 0:
                        print(f"  ✓ {imported} chevaux importés...")

                except Exception as e:
                    print(f"⚠️  Erreur ligne {row_num}: {e}")
                    continue

            conn.commit()

            # Statistiques
            print(f"\n{'='*60}")
            print("✅ IMPORTATION TERMINÉE")
            print(f"{'='*60}")
            print(f"✓ Chevaux importés (races de course)    : {imported}")
            print(f"✗ Rejetés (destiné consommation)       : {skipped_conso}")
            print(f"✗ Rejetés (chevaux décédés)             : {skipped_mort}")
            print(f"✗ Rejetés (races non compatibles)       : {skipped_race}")
            print(f"✗ Rejetés (nés avant 2005 ou date invalide): {skipped_age}")
            print(f"{'='*60}")

    except FileNotFoundError:
        print("❌ Fichier 'fichier-des-equides.csv' non trouvé")
        print("\n📥 Télécharge-le depuis :")
        print("   https://www.data.gouv.fr/datasets/fichier-des-equides/")
        return False

    # Vérification finale
    print("\n🔍 Vérification de la base:")
    cursor.execute("SELECT COUNT(*) FROM chevaux")
    total = cursor.fetchone()[0]
    print(f"   Total chevaux en base: {total}")

    cursor.execute("SELECT sexe, COUNT(*) as nb FROM chevaux GROUP BY sexe")
    for sexe, nb in cursor.fetchall():
        print(f"   - Sexe {sexe}: {nb}")

    cursor.execute(
        "SELECT race, COUNT(*) as nb FROM chevaux GROUP BY race ORDER BY nb DESC LIMIT 5"
    )
    print("\n   Top 5 races:")
    for race, nb in cursor.fetchall():
        print(f"   - {race}: {nb}")

    conn.close()
    print("\n🎯 Base de données réinitialisée avec succès !")
    return True


if __name__ == "__main__":
    reset_database()
