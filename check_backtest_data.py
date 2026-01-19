import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection

conn = get_connection()
cur = conn.cursor()

# Compter les courses avec résultats sur la période du backtest
cur.execute("""
    SELECT
        COUNT(DISTINCT race_key) as nb_races,
        COUNT(*) as nb_partants,
        SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as nb_winners
    FROM cheval_courses_seen
    WHERE SUBSTRING(race_key FROM 1 FOR 10) >= '2025-11-18'
    AND SUBSTRING(race_key FROM 1 FOR 10) <= '2026-01-18'
    AND cote_finale IS NOT NULL
    AND is_win IS NOT NULL
""")

row = cur.fetchone()
print("Période backtest (2025-11-18 à 2026-01-18):")
print(f"  Courses: {row[0]}")
print(f"  Partants totaux: {row[1]}")
print(f"  Gagnants: {row[2]}")
print()

# Stats sur les cotes
cur.execute("""
    SELECT
        AVG(cote_reference) as avg_ref,
        AVG(cote_finale) as avg_fin,
        AVG(cote_finale - cote_reference) as avg_drift,
        COUNT(CASE WHEN cote_reference IS NULL THEN 1 END) as missing_ref
    FROM cheval_courses_seen
    WHERE SUBSTRING(race_key FROM 1 FOR 10) >= '2025-11-18'
    AND SUBSTRING(race_key FROM 1 FOR 10) <= '2026-01-18'
    AND cote_finale IS NOT NULL
    AND cote_finale < 100
""")

row = cur.fetchone()
print("Statistiques cotes:")
avg_ref = row[0] if row[0] else 0
avg_fin = row[1] if row[1] else 0
avg_drift = row[2] if row[2] else 0
print(f"  Cote référence moyenne: {avg_ref:.2f}")
print(f"  Cote finale moyenne: {avg_fin:.2f}")
print(f"  Drift moyen: {avg_drift:.2f}")
print(f"  Cotes référence manquantes: {row[3]}")

conn.close()
