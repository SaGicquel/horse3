#!/usr/bin/env python3
"""
AUDIT RAPIDE - Structure et utilisation des colonnes
Analyse intelligente sans scanner toutes les lignes
"""

import psycopg2
from collections import defaultdict
import os
import re

# Connexion DB
conn = psycopg2.connect(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "54624")),
    database=os.getenv("PGDATABASE", "pmubdd"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", "okokok"),
)
cur = conn.cursor()

print("=" * 80)
print("🔍 AUDIT RAPIDE DES TABLES")
print("=" * 80)

# 1. STRUCTURE DES TABLES
print("\n📋 ÉTAPE 1: STRUCTURE DES TABLES")
print("-" * 80)

for table in ["chevaux", "cheval_courses_seen"]:
    cur.execute(f"""
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = '{table}'
        ORDER BY ordinal_position;
    """)
    cols = cur.fetchall()
    print(f"\n{table.upper()}: {len(cols)} colonnes")

# 2. ANALYSE RAPIDE DE REMPLISSAGE (échantillon)
print("\n📊 ÉTAPE 2: TAUX DE REMPLISSAGE (échantillon 1000 lignes)")
print("-" * 80)

resultats = {}

for table in ["chevaux", "cheval_courses_seen"]:
    print(f"\n{table.upper()}:")

    # Compter total
    cur.execute(f"SELECT COUNT(*) FROM {table};")
    total = cur.fetchone()[0]
    print(f"  Total lignes: {total:,}")

    # Lister colonnes
    cur.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table}'
        ORDER BY ordinal_position;
    """)
    colonnes = [r[0] for r in cur.fetchall()]

    # Échantillon 1000 lignes pour stats rapides
    sample_size = min(1000, total)

    stats = {}
    for col in colonnes:
        cur.execute(f"""
            SELECT
                COUNT(*) as total,
                COUNT({col}) as non_null,
                COUNT(DISTINCT {col}) as distinct_vals
            FROM (SELECT * FROM {table} LIMIT {sample_size}) t;
        """)
        r = cur.fetchone()
        pct = (r[1] / r[0] * 100) if r[0] > 0 else 0
        stats[col] = {"non_null": r[1], "pct": pct, "distinct": r[2]}

    resultats[table] = stats

# 3. ANALYSE DES SCRAPERS
print("\n📦 ÉTAPE 3: MAPPING SCRAPERS → COLONNES")
print("-" * 80)

# Chercher tous les fichiers scrapers
scraper_files = []
for root, dirs, files in os.walk("scrapers"):
    for f in files:
        if f.endswith(".py") and not f.startswith("__"):
            scraper_files.append(os.path.join(root, f))

# Patterns SQL courants
sql_patterns = [
    r"INSERT\s+INTO\s+(\w+)\s*\((.*?)\)",
    r"UPDATE\s+(\w+)\s+SET\s+(.*?)(?:WHERE|$)",
    r"cur\.execute\(['\"]INSERT INTO (\w+) \((.*?)\)",
    r"cur\.execute\(['\"]UPDATE (\w+) SET (.*?)(?:WHERE|$)",
]

colonnes_utilisees = defaultdict(set)

print("\nScrapers analysés:")
for sf in scraper_files:
    print(f"  - {sf}")
    with open(sf, "r", encoding="utf-8") as f:
        content = f.read()

        # Chercher INSERT/UPDATE
        for pattern in sql_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
                table = match.group(1)
                if table in ["chevaux", "cheval_courses_seen"]:
                    cols_text = match.group(2)
                    # Extraire noms colonnes
                    cols = re.findall(r"\b(\w+)\b", cols_text)
                    for c in cols:
                        if c not in ["SET", "VALUES", "WHERE", "AND", "OR"]:
                            colonnes_utilisees[table].add(c)

print("\n✅ Colonnes utilisées dans les scrapers:")
for table, cols in colonnes_utilisees.items():
    print(f"\n{table.upper()}: {len(cols)} colonnes utilisées")

# 4. RAPPORT FINAL
print("\n" + "=" * 80)
print("📊 RAPPORT FINAL")
print("=" * 80)

for table in ["chevaux", "cheval_courses_seen"]:
    print(f"\n{table.upper()}:")
    print("-" * 80)

    stats = resultats[table]
    cols_scraper = colonnes_utilisees.get(table, set())

    # Catégoriser
    vides = []
    peu_remplies = []
    bien_remplies = []
    non_utilisees = []

    for col, data in sorted(stats.items(), key=lambda x: x[1]["pct"]):
        pct = data["pct"]
        utilisee = col in cols_scraper

        if pct == 0:
            vides.append((col, utilisee))
        elif pct < 10:
            peu_remplies.append((col, pct, utilisee))
        else:
            bien_remplies.append((col, pct, utilisee))

        if not utilisee:
            non_utilisees.append(col)

    print(f"\n🔴 COLONNES VIDES ({len(vides)}):")
    for col, used in vides[:10]:  # Top 10
        flag = "✅" if used else "❌"
        print(f"  {flag} {col}")
    if len(vides) > 10:
        print(f"  ... et {len(vides)-10} autres")

    print(f"\n🟡 COLONNES PEU REMPLIES <10% ({len(peu_remplies)}):")
    for col, pct, used in peu_remplies[:10]:
        flag = "✅" if used else "❌"
        print(f"  {flag} {col}: {pct:.1f}%")
    if len(peu_remplies) > 10:
        print(f"  ... et {len(peu_remplies)-10} autres")

    print(f"\n🟢 COLONNES BIEN REMPLIES >10% ({len(bien_remplies)}):")
    for col, pct, used in bien_remplies[:10]:
        flag = "✅" if used else "❌"
        print(f"  {flag} {col}: {pct:.1f}%")
    if len(bien_remplies) > 10:
        print(f"  ... et {len(bien_remplies)-10} autres")

    print(f"\n⚠️  COLONNES NON UTILISÉES PAR SCRAPERS ({len(non_utilisees)}):")
    for col in non_utilisees[:15]:
        print(f"  - {col}")
    if len(non_utilisees) > 15:
        print(f"  ... et {len(non_utilisees)-15} autres")

print("\n" + "=" * 80)
print("✅ AUDIT TERMINÉ")
print("=" * 80)

# Sauvegarder résultats détaillés
with open("audit_resultats.txt", "w") as f:
    f.write("RÉSULTATS DÉTAILLÉS AUDIT\n")
    f.write("=" * 80 + "\n\n")

    for table in ["chevaux", "cheval_courses_seen"]:
        f.write(f"\n{table.upper()}\n")
        f.write("-" * 80 + "\n")

        stats = resultats[table]
        cols_scraper = colonnes_utilisees.get(table, set())

        for col in sorted(stats.keys()):
            data = stats[col]
            used = "OUI" if col in cols_scraper else "NON"
            f.write(
                f"{col:40} | {data['pct']:6.1f}% | Distinct: {data['distinct']:6} | Scraper: {used}\n"
            )

print("\n💾 Résultats détaillés sauvegardés dans: audit_resultats.txt")

cur.close()
conn.close()
