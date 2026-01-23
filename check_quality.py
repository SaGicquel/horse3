import os
import sys
from datetime import datetime
import pandas as pd
from db_connection import get_connection


def check_today_data_quality():
    conn = get_connection()
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Checking data quality for {today}")

    query = f"""
    SELECT
        race_key,
        heure_depart,
        cote_reference,
        distance_m,
        age,
        poids_kg,
        hippodrome_code
    FROM cheval_courses_seen
    WHERE race_key LIKE '{today}%'
    """

    try:
        df = pd.read_sql(query, conn)
        print(f"Total rows: {len(df)}")

        if len(df) == 0:
            print("No rows found!")
            return

        print("\nDtypes:")
        print(df.dtypes)

        print("\nMissing values count:")
        print(df.isnull().sum())

        # Check non-null cote_reference count
        valid_cote = df[df["cote_reference"].notna() & (df["cote_reference"] > 0)]
        print(f"\nRows with valid cote_reference: {len(valid_cote)}")

        if len(valid_cote) > 0:
            print("\nDtypes in valid_cote subset:")
            print(valid_cote.dtypes)

            # Check distinct types in poids_kg for valid_cote
            print("\nUnique types in poids_kg:")
            print(valid_cote["poids_kg"].apply(type).value_counts())

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    check_today_data_quality()
