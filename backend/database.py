"""
Database connection module for FastAPI backend.
Uses psycopg2 with connection pooling for PostgreSQL.
"""

import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "54624")),
    "database": os.getenv("PGDATABASE", "pmu_database"),
    "user": os.getenv("PGUSER", "pmu_user"),
    "password": os.getenv("PGPASSWORD", "pmu_secure_password_2025"),
}

_connection_pool: pool.ThreadedConnectionPool | None = None


def get_pool() -> pool.ThreadedConnectionPool:
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(minconn=2, maxconn=10, **DB_CONFIG)
    return _connection_pool


def close_pool() -> None:
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None


@contextmanager
def get_db() -> Generator[psycopg2.extensions.connection, None, None]:
    conn = get_pool().getconn()
    try:
        yield conn
    finally:
        get_pool().putconn(conn)


@contextmanager
def get_cursor(dict_cursor: bool = True):
    with get_db() as conn:
        cursor_factory = RealDictCursor if dict_cursor else None
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
