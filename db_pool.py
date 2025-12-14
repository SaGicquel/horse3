"""
Module de connection pooling PostgreSQL pour le scraper PMU
Améliore les performances en réutilisant les connexions existantes
"""

import os
from psycopg2 import pool
from psycopg2.extras import DictCursor
import threading
from contextlib import contextmanager

# Configuration
DB_CONFIG = {
    'host': os.getenv('PGHOST', 'localhost'),
    'port': os.getenv('PGPORT', '54624'),
    'database': os.getenv('PGDATABASE', 'pmubdd'),
    'user': os.getenv('PGUSER', 'postgres'),
    'password': os.getenv('PGPASSWORD', 'okokok')
}

# Pool de connexions global
_connection_pool = None
_pool_lock = threading.Lock()

def initialize_pool(minconn=5, maxconn=20):
    """
    Initialise le pool de connexions
    
    Args:
        minconn: Nombre minimum de connexions à maintenir
        maxconn: Nombre maximum de connexions autorisées
    """
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is None:
            print(f"🏊 Initialisation du pool de connexions (min={minconn}, max={maxconn})")
            _connection_pool = pool.ThreadedConnectionPool(
                minconn,
                maxconn,
                **DB_CONFIG
            )
            print("✅ Pool de connexions initialisé")
        else:
            print("⚠️ Pool déjà initialisé")

def close_pool():
    """Ferme toutes les connexions du pool"""
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is not None:
            _connection_pool.closeall()
            _connection_pool = None
            print("🔒 Pool de connexions fermé")

@contextmanager
def get_connection():
    """
    Context manager pour obtenir une connexion du pool
    
    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
    """
    if _connection_pool is None:
        initialize_pool()
    
    conn = _connection_pool.getconn()
    try:
        yield conn
    finally:
        _connection_pool.putconn(conn)

@contextmanager
def get_cursor(cursor_factory=DictCursor):
    """
    Context manager pour obtenir directement un curseur
    
    Args:
        cursor_factory: Type de curseur (DictCursor par défaut)
    
    Usage:
        with get_cursor() as cur:
            cur.execute("SELECT ...")
            results = cur.fetchall()
    """
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()

def get_pool_stats():
    """Retourne les statistiques du pool"""
    if _connection_pool is None:
        return {"status": "non initialisé"}
    
    # Note: psycopg2.pool n'expose pas directement les stats
    # mais on peut vérifier qu'il fonctionne
    return {
        "status": "actif",
        "min_connections": _connection_pool.minconn,
        "max_connections": _connection_pool.maxconn
    }

def test_pool():
    """Test du pool de connexions"""
    print("\n🧪 Test du pool de connexions...")
    
    initialize_pool(minconn=2, maxconn=5)
    
    # Test 1: Connexion simple
    print("\n1️⃣ Test connexion simple...")
    with get_cursor() as cur:
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        print(f"   ✅ PostgreSQL version: {version[:50]}...")
    
    # Test 2: Connexions multiples
    print("\n2️⃣ Test connexions multiples...")
    import concurrent.futures
    
    def test_query(n):
        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chevaux")
            count = cur.fetchone()[0]
            return f"Thread {n}: {count} chevaux"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(test_query, i) for i in range(3)]
        for future in concurrent.futures.as_completed(futures):
            print(f"   ✅ {future.result()}")
    
    # Test 3: Statistiques
    print("\n3️⃣ Statistiques du pool...")
    stats = get_pool_stats()
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    close_pool()
    print("\n✅ Tests terminés")

if __name__ == "__main__":
    test_pool()
