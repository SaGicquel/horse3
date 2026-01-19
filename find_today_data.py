#!/usr/bin/env python3
import psycopg2
from datetime import date

conn = psycopg2.connect(
    host="localhost",
    port=54624,
    database="pmu_database",
    user="pmu_user",
    password="pmu_secure_password_2025",
)

cursor = conn.cursor()
today = "2026-01-19"

print(f"Recherche de données pour {today} dans toutes les tables...\n")

# Chercher dans toutes les tables
tables_to_check = [
    "courses",
    "cheval_courses_seen",
    "performances",
    "cotes_historiques",
    "user_bets",
]

for table in tables_to_check:
    try:
        cursor.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table}'
            AND column_name LIKE '%date%'
        """)
        date_cols = [r[0] for r in cursor.fetchall()]

        if date_cols:
            for col in date_cols:
                try:
                    cursor.execute(f"""
                        SELECT COUNT(*)
                        FROM {table}
                        WHERE {col} = '{today}'
                    """)
                    count = cursor.fetchone()[0]

                    if count > 0:
                        print(f"✅ {table}.{col}: {count} enregistrements pour {today}")
                    else:
                        # Dernière date
                        cursor.execute(f"SELECT MAX({col}) FROM {table}")
                        max_date = cursor.fetchone()[0]
                        print(f"❌ {table}.{col}: 0 pour {today}, dernière: {max_date}")
                except Exception as e:
                    print(f"⚠️  {table}.{col}: {e}")
    except Exception as e:
        print(f"⚠️  {table}: {e}")

cursor.close()
conn.close()
