#!/usr/bin/env python3
"""Script simple de monitoring basique de l'enrichissement"""

import psycopg2
from datetime import datetime

conn = psycopg2.connect(
    host="localhost", port=54624, database="pmubdd", user="postgres", password="okokok"
)

cur = conn.cursor()

# Total participations octobre
cur.execute("""
    SELECT
        COUNT(*) as total,
        COUNT(DISTINCT SUBSTRING(race_key, 1, 10)) as jours,
        MIN(SUBSTRING(race_key, 1, 10)) as premier_jour,
        MAX(SUBSTRING(race_key, 1, 10)) as dernier_jour
    FROM cheval_courses_seen
    WHERE race_key LIKE '2024-10-%'
""")

row = cur.fetchone()
total, jours, premier, dernier = row

print(f"\n{'='*70}")
print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - MONITORING OCTOBRE 2024")
print(f"{'='*70}")
print(f"📋 Participations : {total:,}")
print(f"📅 Jours traités  : {jours}/31 ({int(jours*100/31)}%)")
print(f"   ├─ Premier : {premier}")
print(f"   └─ Dernier : {dernier}")

# Enrichissement (colonnes de base existantes)
cur.execute("""
    SELECT
        COUNT(CASE WHEN course_id IS NOT NULL THEN 1 END) as has_course_id,
        COUNT(CASE WHEN classe_course IS NOT NULL THEN 1 END) as has_classe,
        COUNT(CASE WHEN meteo_code IS NOT NULL THEN 1 END) as has_meteo,
        COUNT(CASE WHEN handicap_distance IS NOT NULL THEN 1 END) as has_handicap,
        COUNT(CASE WHEN entraineur_winrate_90j IS NOT NULL THEN 1 END) as has_winrate
    FROM cheval_courses_seen
    WHERE race_key LIKE '2024-10-%'
""")

row = cur.fetchone()
course_id, classe, meteo, handicap, winrate = row

print("\n🔍 ENRICHISSEMENT:")
print(
    f"   ├─ course_id        : {course_id:,}/{total:,} ({int(course_id*100/total if total > 0 else 0)}%)"
)
print(
    f"   ├─ classe_course    : {classe:,}/{total:,} ({int(classe*100/total if total > 0 else 0)}%)"
)
print(f"   ├─ meteo_code       : {meteo:,}/{total:,} ({int(meteo*100/total if total > 0 else 0)}%)")
print(
    f"   ├─ handicap_distance: {handicap:,}/{total:,} ({int(handicap*100/total if total > 0 else 0)}%)"
)
print(
    f"   └─ entraineur_wr_90j: {winrate:,}/{total:,} ({int(winrate*100/total if total > 0 else 0)}%)"
)

print(f"{'='*70}\n")

cur.close()
conn.close()
