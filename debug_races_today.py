import os
import sys
from datetime import datetime
from db_connection import get_connection


def check_today_races():
    conn = get_connection()
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Checking races for {today}")

    query = f"""
    SELECT race_key, heure_depart, cote_reference
    FROM cheval_courses_seen
    WHERE race_key LIKE '{today}%'
    LIMIT 20
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            print(f"Found {len(rows)} rows sample.")
            for row in rows:
                print(f"Key: {row[0]}, Heure: {row[1]} (Type: {type(row[1])}), Cote Ref: {row[2]}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    check_today_races()
