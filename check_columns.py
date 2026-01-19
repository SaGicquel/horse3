import os
import sys
from db_connection import get_connection

try:
    conn = get_connection()
    cur = conn.cursor()

    print("--- Table chevaux ---")
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'chevaux'")
    print([row[0] for row in cur.fetchall()])

    print("\n--- Table stats_chevaux ---")
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'stats_chevaux'"
    )
    print([row[0] for row in cur.fetchall()])

    conn.close()
except Exception as e:
    print(e)
