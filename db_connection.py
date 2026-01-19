#!/usr/bin/env python3
"""
Module de connexion à la base de données.
Utilise PostgreSQL par défaut (conteneur Docker pmuBDD).
"""

import os
import psycopg2
import psycopg2.extras

# Configuration PostgreSQL par défaut
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "54624")),  # Port Docker du conteneur pmuBDD
    "database": os.getenv("PGDATABASE", "pmu_database"),
    "user": os.getenv("PGUSER", "pmu_user"),
    "password": os.getenv("PGPASSWORD", "pmu_secure_password_2025"),
}


def get_connection():
    """
    Retourne une connexion PostgreSQL.

    Returns:
        psycopg2.connection: Connexion à la base de données PostgreSQL
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Erreur de connexion PostgreSQL : {e}")
        print(f"   Config : {DB_CONFIG}")
        raise


def get_cursor(conn=None, dict_cursor=True):
    """
    Retourne un curseur pour la connexion.

    Args:
        conn: Connexion existante (si None, en crée une nouvelle)
        dict_cursor: Si True, retourne un DictCursor (accès par nom de colonne)

    Returns:
        tuple: (connection, cursor)
    """
    if conn is None:
        conn = get_connection()

    if dict_cursor:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    else:
        cursor = conn.cursor()

    return conn, cursor


def test_connection():
    """Test rapide de la connexion."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.close()
        conn.close()
        print("✅ Connexion PostgreSQL OK")
        print(f"   Version : {version[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Test de connexion échoué : {e}")
        return False


if __name__ == "__main__":
    test_connection()
