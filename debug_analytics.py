
import os
import sys
from dotenv import load_dotenv
import pandas as pd

# Add parent dir to path to import db_connection
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from db_connection import get_connection
    print("Using db_connection")
    conn = get_connection()
    is_postgres = True
except ImportError:
    import sqlite3
    print("Using SQLite fallback")
    conn = sqlite3.connect('data/database.db')
    is_postgres = False

cur = conn.cursor()

print(f"Database type: {'PostgreSQL' if is_postgres else 'SQLite'}")

# Check total rows
try:
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    print(f"Total rows in cheval_courses_seen: {cur.fetchone()[0]}")
except Exception as e:
    print(f"Error querying cheval_courses_seen: {e}")

# Check Jockeys
print("\n--- Jockeys Analysis ---")
try:
    if is_postgres:
        query = """
            SELECT driver_jockey, COUNT(*) as c 
            FROM cheval_courses_seen 
            WHERE driver_jockey IS NOT NULL 
            GROUP BY driver_jockey 
            ORDER BY c DESC LIMIT 5
        """
    else:
        query = """
            SELECT jockey_habituel, COUNT(*) as c 
            FROM chevaux 
            GROUP BY jockey_habituel 
            ORDER BY c DESC LIMIT 5
        """
    cur.execute(query)
    print("Top 5 Jockeys by count:")
    for row in cur.fetchall():
        print(row)
        
    # Check threshold
    if is_postgres:
        cur.execute("SELECT COUNT(*) FROM (SELECT driver_jockey FROM cheval_courses_seen GROUP BY driver_jockey HAVING COUNT(*) >= 10) as t")
    else:
        cur.execute("SELECT COUNT(*) FROM (SELECT jockey_habituel FROM chevaux GROUP BY jockey_habituel HAVING COUNT(*) >= 3) as t")
    print(f"Number of jockeys meeting threshold: {cur.fetchone()[0]}")

except Exception as e:
    print(f"Error analyzing jockeys: {e}")

# Check Entraineurs
print("\n--- Entraineurs Analysis ---")
try:
    if is_postgres:
        query = """
            SELECT entraineur, COUNT(*) as c 
            FROM cheval_courses_seen 
            WHERE entraineur IS NOT NULL 
            GROUP BY entraineur 
            ORDER BY c DESC LIMIT 5
        """
    else:
        query = """
            SELECT entraineur_courant, COUNT(*) as c 
            FROM chevaux 
            GROUP BY entraineur_courant 
            ORDER BY c DESC LIMIT 5
        """
    cur.execute(query)
    print("Top 5 Entraineurs by count:")
    for row in cur.fetchall():
        print(row)

    # Check threshold
    if is_postgres:
        cur.execute("SELECT COUNT(*) FROM (SELECT entraineur FROM cheval_courses_seen GROUP BY entraineur HAVING COUNT(*) >= 10) as t")
    else:
        cur.execute("SELECT COUNT(*) FROM (SELECT entraineur_courant FROM chevaux GROUP BY entraineur_courant HAVING COUNT(*) >= 3) as t")
    print(f"Number of entraineurs meeting threshold: {cur.fetchone()[0]}")

except Exception as e:
    print(f"Error analyzing entraineurs: {e}")

# Check Top Performers
print("\n--- Top Performers Analysis ---")
try:
    if is_postgres:
        # The query used in main.py joins with stats_chevaux
        cur.execute("SELECT COUNT(*) FROM stats_chevaux")
        print(f"Rows in stats_chevaux: {cur.fetchone()[0]}")
        
        cur.execute("SELECT COUNT(*) FROM stats_chevaux WHERE nb_courses_total >= 10")
        print(f"Horses with >= 10 races in stats_chevaux: {cur.fetchone()[0]}")
    else:
        cur.execute("SELECT COUNT(*) FROM chevaux WHERE nombre_courses_total >= 10")
        print(f"Horses with >= 10 races in chevaux: {cur.fetchone()[0]}")

except Exception as e:
    print(f"Error analyzing performers: {e}")

conn.close()
