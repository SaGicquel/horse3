"""Initialisation de la nouvelle base (drop -> bootstrap -> schema_v1).

Usage:
    python init_new_schema.py [--no-drop]

Le script exécute séquentiellement:
 1. sql/drop_all.sql (sauf si --no-drop)
 2. sql/bootstrap.sql
 3. sql/schema_v1.sql

Il suppose que la connexion PostgreSQL est gérée par db_connection.get_connection().
"""
from pathlib import Path
import sys
from db_connection import get_connection

SQL_DIR = Path(__file__).parent / 'sql'


def read_sql(name: str) -> str:
    path = SQL_DIR / name
    return path.read_text(encoding='utf-8')


def execute_block(cur, sql: str, label: str):
    print(f"-- Exécution {label} ...")
    cur.execute(sql)
    print(f"-- OK {label}")


def main():
    do_drop = '--no-drop' not in sys.argv
    con = get_connection()
    con.autocommit = True
    cur = con.cursor()

    if do_drop:
        try:
            execute_block(cur, read_sql('drop_all.sql'), 'drop_all')
        except Exception as e:
            print(f"(Avertissement) drop_all a rencontré une erreur (ignorée): {e}")

    execute_block(cur, read_sql('bootstrap.sql'), 'bootstrap')
    execute_block(cur, read_sql('schema_v1.sql'), 'schema_v1')

    cur.close()
    con.close()
    print("✅ Initialisation terminée")


if __name__ == '__main__':
    main()
