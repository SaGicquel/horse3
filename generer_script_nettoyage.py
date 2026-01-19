#!/usr/bin/env python3
"""
Génère un script SQL de nettoyage sécurisé
Analyse les colonnes et propose suppressions
"""

import psycopg2
import os
from datetime import datetime

# Config DB
conn = psycopg2.connect(
    host="localhost", port=54624, database="pmubdd", user="postgres", password="okokok"
)
cur = conn.cursor()

print("=" * 80)
print("🔧 GÉNÉRATION SCRIPT NETTOYAGE SQL")
print("=" * 80)

# Seuils de décision
SEUIL_VIDE = 0.5  # <0.5% considéré vide
SEUIL_PEU_REMPLI = 5.0  # <5% peu rempli

# Colonnes à TOUJOURS garder (critiques)
COLONNES_CRITIQUES = {
    "chevaux": ["id_cheval", "nom", "sexe", "race", "date_naissance", "created_at"],
    "cheval_courses_seen": [
        "id_cheval",
        "race_key",
        "date_course",
        "position_arrivee",
        "created_at",
    ],
}

# Colonnes utilisées par scrapers (détectées dans audit)
COLONNES_SCRAPERS = {
    "cheval_courses_seen": [
        "autostart_ligne",
        "classe_course",
        "code_hippodrome",
        "course_id",
        "heure_locale",
        "meeting_id",
        "nom_hippodrome",
        "pays_course",
        "profil_piste",
        "reduction_km",
        "rend_m",
        "temps_total_s",
        "type_course",
    ],
    "chevaux": [],
}

# Colonnes calculées (Jours 2-5) à TOUJOURS GARDER même si peu remplies
COLONNES_CALCULEES = {
    "chevaux": [
        # Jour 2 - Stats basiques
        "nb_courses_total",
        "nb_victoires",
        "nb_places",
        "taux_victoire",
        "taux_place",
        "gains_total",
        "gains_moyen",
        "derniere_course",
        "age_debut_carriere",
        "indice_consistance_12m",
        # Jour 3 - Temporelles
        "jours_depuis_derniere_course",
        "forme_recente",
        "regularite_courses",
        "progression_distance",
        "performance_saison",
        "tendance_last_5",
        # Jour 4 - Comparaisons
        "ecart_median_gagnant",
        "meilleur_temps_relatif",
        "rank_gains_generation",
        "performance_vs_moyenne_course",
        "taux_victoire_vs_concurrent",
        "handicap_optimal",
        # Jour 5 - ML Features
        "momentum_score",
        "distance_preference_score",
        "terrain_preference_score",
        "consistency_score",
        "improvement_rate",
        "versatility_score",
    ],
    "cheval_courses_seen": [],
}

script_sql = []
script_sql.append("-- =====================================================")
script_sql.append("-- SCRIPT DE NETTOYAGE BASE DE DONNÉES")
script_sql.append(f"-- Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
script_sql.append("-- =====================================================")
script_sql.append("")
script_sql.append("BEGIN;")
script_sql.append("")

statistiques = {
    "chevaux": {"total": 0, "a_supprimer": 0, "vides": 0, "peu_remplies": 0},
    "cheval_courses_seen": {"total": 0, "a_supprimer": 0, "vides": 0, "peu_remplies": 0},
}

for table in ["chevaux", "cheval_courses_seen"]:
    print(f"\n📋 Analyse table: {table}")
    print("-" * 80)

    # Compter lignes total
    cur.execute(f"SELECT COUNT(*) FROM {table};")
    total_lignes = cur.fetchone()[0]

    # Lister colonnes
    cur.execute(f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{table}'
        ORDER BY ordinal_position;
    """)
    colonnes = cur.fetchall()
    statistiques[table]["total"] = len(colonnes)

    script_sql.append("-- =====================================================")
    script_sql.append(f"-- TABLE: {table.upper()}")
    script_sql.append(f"-- Colonnes totales: {len(colonnes)}")
    script_sql.append("-- =====================================================")
    script_sql.append("")

    colonnes_a_supprimer = []
    colonnes_a_garder = []

    for col_name, col_type in colonnes:
        # Vérifier si colonne critique
        if col_name in COLONNES_CRITIQUES.get(table, []):
            colonnes_a_garder.append((col_name, "CRITIQUE"))
            continue

        # Vérifier si colonne calculée (Jours 2-5)
        if col_name in COLONNES_CALCULEES.get(table, []):
            colonnes_a_garder.append((col_name, "CALCULÉE"))
            continue

        # Vérifier si utilisée par scraper
        if col_name in COLONNES_SCRAPERS.get(table, []):
            colonnes_a_garder.append((col_name, "SCRAPER"))
            continue

        # Calculer taux remplissage
        cur.execute(f"""
            SELECT
                COUNT(*) as total,
                COUNT({col_name}) as non_null
            FROM {table}
            LIMIT 10000;  -- Échantillon
        """)
        total, non_null = cur.fetchone()
        taux = (non_null / total * 100) if total > 0 else 0

        # Décision
        if taux < SEUIL_VIDE:
            colonnes_a_supprimer.append((col_name, taux, "VIDE"))
            statistiques[table]["vides"] += 1
        elif taux < SEUIL_PEU_REMPLI:
            colonnes_a_supprimer.append((col_name, taux, "PEU_REMPLI"))
            statistiques[table]["peu_remplies"] += 1
        else:
            colonnes_a_garder.append((col_name, f"REMPLI {taux:.1f}%"))

    statistiques[table]["a_supprimer"] = len(colonnes_a_supprimer)

    # Générer commandes DROP
    if colonnes_a_supprimer:
        script_sql.append(f"-- Supprimer {len(colonnes_a_supprimer)} colonnes inutiles")
        script_sql.append(f"ALTER TABLE {table}")

        for i, (col, taux, raison) in enumerate(colonnes_a_supprimer):
            virgule = "," if i < len(colonnes_a_supprimer) - 1 else ";"
            script_sql.append(
                f"  DROP COLUMN IF EXISTS {col} CASCADE{virgule}  -- {raison} ({taux:.1f}%)"
            )

        script_sql.append("")

    # Afficher résumé console
    print(f"✅ À GARDER: {len(colonnes_a_garder)} colonnes")
    for col, raison in colonnes_a_garder[:5]:
        print(f"   - {col:40} ({raison})")
    if len(colonnes_a_garder) > 5:
        print(f"   ... et {len(colonnes_a_garder)-5} autres")

    print(f"\n❌ À SUPPRIMER: {len(colonnes_a_supprimer)} colonnes")
    for col, taux, raison in colonnes_a_supprimer[:5]:
        print(f"   - {col:40} {taux:5.1f}% ({raison})")
    if len(colonnes_a_supprimer) > 5:
        print(f"   ... et {len(colonnes_a_supprimer)-5} autres")

# Vérifications finales
script_sql.append("")
script_sql.append("-- =====================================================")
script_sql.append("-- VÉRIFICATIONS FINALES")
script_sql.append("-- =====================================================")
script_sql.append("")
script_sql.append("-- Vérifier structure après nettoyage")
script_sql.append("SELECT table_name, COUNT(*) as nb_colonnes")
script_sql.append("FROM information_schema.columns")
script_sql.append("WHERE table_name IN ('chevaux', 'cheval_courses_seen')")
script_sql.append("GROUP BY table_name;")
script_sql.append("")
script_sql.append("-- Vérifier données toujours présentes")
script_sql.append("SELECT 'chevaux' as table_name, COUNT(*) as nb_lignes FROM chevaux")
script_sql.append("UNION ALL")
script_sql.append("SELECT 'cheval_courses_seen', COUNT(*) FROM cheval_courses_seen;")
script_sql.append("")
script_sql.append("COMMIT;")
script_sql.append("")
script_sql.append("-- =====================================================")
script_sql.append("-- FIN DU SCRIPT")
script_sql.append("-- =====================================================")

# Sauvegarder script
filename = f"script_nettoyage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
with open(filename, "w") as f:
    f.write("\n".join(script_sql))

print("\n" + "=" * 80)
print("📊 STATISTIQUES GLOBALES")
print("=" * 80)

total_colonnes = sum(s["total"] for s in statistiques.values())
total_supprimer = sum(s["a_supprimer"] for s in statistiques.values())
pct_suppression = (total_supprimer / total_colonnes * 100) if total_colonnes > 0 else 0

for table, stats in statistiques.items():
    print(f"\n{table.upper()}:")
    print(f"  Total colonnes:      {stats['total']}")
    print(
        f"  À supprimer:         {stats['a_supprimer']} ({stats['a_supprimer']/stats['total']*100:.1f}%)"
    )
    print(f"    - Vides (<0.5%):   {stats['vides']}")
    print(f"    - Peu remplies:    {stats['peu_remplies']}")
    print(f"  À conserver:         {stats['total'] - stats['a_supprimer']}")

print(f"\n{'='*80}")
print("TOTAL:")
print(f"  Colonnes actuelles:  {total_colonnes}")
print(f"  Colonnes à supprimer: {total_supprimer} ({pct_suppression:.1f}%)")
print(f"  Colonnes finales:    {total_colonnes - total_supprimer}")
print(f"{'='*80}")

print(f"\n✅ Script SQL généré: {filename}")
print("\n⚠️  AVANT D'EXÉCUTER:")
print("   1. Vérifier le script manuellement")
print("   2. Faire un backup: pg_dump -U postgres -d pmubdd > backup.sql")
print("   3. Tester sur copie base si possible")
print(f"   4. Exécuter: psql -U postgres -d pmubdd -f {filename}")

cur.close()
conn.close()
