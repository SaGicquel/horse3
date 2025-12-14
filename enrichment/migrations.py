# -*- coding: utf-8 -*-
"""
Migrations SQLite pour l'enrichissement des données hippiques
Modèle : IFCE horses, PMU horses, performances enrichies, agrégats
"""

import sqlite3
from typing import Optional


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Vérifie si une colonne existe dans une table"""
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Vérifie si une table existe"""
    cur = conn.cursor()
    cur.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (table,))
    return cur.fetchone() is not None


def run_migrations(conn: sqlite3.Connection, enable_fts: bool = False):
    """
    Exécute toutes les migrations nécessaires (non destructives).
    
    Args:
        conn: Connexion SQLite
        enable_fts: Activer le Full-Text Search (optionnel, pour fuzzy matching)
    """
    cur = conn.cursor()
    
    print("🔧 Démarrage des migrations...")
    
    # ========== 1. Table IFCE horses ==========
    print("  [1/8] Table ifce_horses...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ifce_horses (
            horse_key INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            name_norm TEXT NOT NULL,
            sex TEXT,                -- H, M, F
            birth_date DATE,
            birth_year INTEGER,
            country TEXT,            -- Code pays (FR, GB, etc.)
            breed TEXT,              -- Race (TROTTEUR FRANÇAIS, PUR SANG, etc.)
            coat TEXT,               -- Robe
            trainer TEXT,            -- Entraîneur (si présent dans CSV)
            jockey TEXT,             -- Jockey (si présent dans CSV)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Index sur ifce_horses
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ifce_name_norm ON ifce_horses(name_norm)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ifce_birth_year ON ifce_horses(birth_year)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ifce_country ON ifce_horses(country)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ifce_sex ON ifce_horses(sex)")
    
    # ========== 2. Table PMU horses (chevaux observés via scraping) ==========
    print("  [2/8] Table pmu_horses...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pmu_horses (
            pmu_horse_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            name_norm TEXT NOT NULL,
            sex TEXT,
            birth_year INTEGER,
            country TEXT,
            breed TEXT,
            ifce_horse_key INTEGER,        -- FK vers ifce_horses (nullable)
            match_stage TEXT,              -- 'A', 'B', 'C', 'none', 'ambiguous'
            match_confidence REAL,         -- 0.0 - 1.0
            first_seen_date DATE,
            last_seen_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ifce_horse_key) REFERENCES ifce_horses(horse_key)
        )
    """)
    
    # Index sur pmu_horses
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pmu_name_norm ON pmu_horses(name_norm)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pmu_ifce ON pmu_horses(ifce_horse_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pmu_match_stage ON pmu_horses(match_stage)")
    
    # ========== 3. Table performances (enrichie) ==========
    print("  [3/8] Extension table performances...")
    
    # Vérifier si la table existe, sinon la créer
    if not table_exists(conn, 'performances'):
        cur.execute("""
            CREATE TABLE performances (
                performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                horse_key INTEGER NOT NULL,
                race_date DATE NOT NULL,
                race_code TEXT,
                venue TEXT,
                discipline TEXT,
                distance_m INTEGER,
                finish_position TEXT,
                finish_status TEXT,
                time_str TEXT,
                time_sec REAL,
                reduction_km_sec REAL,
                allocation_eur REAL,
                trainer_race TEXT,
                jockey_race TEXT,
                owner_race TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (horse_key) REFERENCES ifce_horses(horse_key)
            )
        """)
    else:
        # Ajouter colonnes manquantes si table existe déjà
        columns_to_add = [
            ('allocation_eur', 'REAL'),
            ('reduction_km_sec', 'REAL'),
            ('trainer_race', 'TEXT'),
            ('jockey_race', 'TEXT'),
            ('owner_race', 'TEXT'),
            ('discipline', 'TEXT'),
            ('time_str', 'TEXT'),
            ('time_sec', 'REAL'),
        ]
        
        for col, typ in columns_to_add:
            if not column_exists(conn, 'performances', col):
                cur.execute(f"ALTER TABLE performances ADD COLUMN {col} {typ}")
                print(f"    → Ajout colonne performances.{col}")
    
    # Index sur performances
    cur.execute("CREATE INDEX IF NOT EXISTS idx_perf_horse_date ON performances(horse_key, race_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_perf_race_date ON performances(race_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_perf_discipline ON performances(discipline)")
    
    # ========== 4. Table horse_year_stats (gains annuels) ==========
    print("  [4/8] Table horse_year_stats...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS horse_year_stats (
            horse_key INTEGER NOT NULL,
            year INTEGER NOT NULL,
            gains_annuels_eur REAL DEFAULT 0,
            nb_courses INTEGER DEFAULT 0,
            nb_victoires INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (horse_key, year),
            FOREIGN KEY (horse_key) REFERENCES ifce_horses(horse_key)
        )
    """)
    
    cur.execute("CREATE INDEX IF NOT EXISTS idx_year_stats_year ON horse_year_stats(year)")
    
    # ========== 5. Table horse_totals (agrégats globaux) ==========
    print("  [5/8] Table horse_totals...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS horse_totals (
            horse_key INTEGER PRIMARY KEY,
            gains_totaux_eur REAL DEFAULT 0,
            record_attele_sec REAL,
            record_attele_date DATE,
            record_attele_venue TEXT,
            record_attele_race TEXT,
            record_monte_sec REAL,
            record_monte_date DATE,
            record_monte_venue TEXT,
            record_monte_race TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (horse_key) REFERENCES ifce_horses(horse_key)
        )
    """)
    
    # ========== 6. Table horse_aliases (cache matching) ==========
    print("  [6/8] Table horse_aliases...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS horse_aliases (
            pmu_name_norm TEXT PRIMARY KEY,
            ifce_horse_key INTEGER NOT NULL,
            validated_by TEXT,
            validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ifce_horse_key) REFERENCES ifce_horses(horse_key)
        )
    """)
    
    cur.execute("CREATE INDEX IF NOT EXISTS idx_aliases_ifce ON horse_aliases(ifce_horse_key)")
    
    # ========== 7. FTS (Full-Text Search, optionnel) ==========
    if enable_fts:
        print("  [7/8] Activation FTS sur ifce_horses...")
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS ifce_horses_fts 
            USING fts5(name_norm, content=ifce_horses, content_rowid=horse_key)
        """)
        
        # Peupler FTS si vide
        cur.execute("SELECT COUNT(*) FROM ifce_horses_fts")
        if cur.fetchone()[0] == 0:
            cur.execute("""
                INSERT INTO ifce_horses_fts(rowid, name_norm)
                SELECT horse_key, name_norm FROM ifce_horses
            """)
            print("    → FTS peuplé avec données existantes")
        
        # Triggers pour maintenir FTS à jour
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS ifce_horses_ai AFTER INSERT ON ifce_horses
            BEGIN
                INSERT INTO ifce_horses_fts(rowid, name_norm) VALUES (new.horse_key, new.name_norm);
            END
        """)
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS ifce_horses_ad AFTER DELETE ON ifce_horses
            BEGIN
                INSERT INTO ifce_horses_fts(ifce_horses_fts, rowid, name_norm) 
                VALUES('delete', old.horse_key, old.name_norm);
            END
        """)
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS ifce_horses_au AFTER UPDATE ON ifce_horses
            BEGIN
                INSERT INTO ifce_horses_fts(ifce_horses_fts, rowid, name_norm) 
                VALUES('delete', old.horse_key, old.name_norm);
                INSERT INTO ifce_horses_fts(rowid, name_norm) VALUES (new.horse_key, new.name_norm);
            END
        """)
    else:
        print("  [7/8] FTS désactivé (fuzzy matching non disponible)")
    
    # ========== 8. Vues utiles ==========
    print("  [8/8] Création vues...")
    
    # Vue : chevaux avec matching
    cur.execute("""
        CREATE VIEW IF NOT EXISTS v_horses_matched AS
        SELECT 
            p.pmu_horse_id,
            p.name AS pmu_name,
            p.name_norm,
            p.birth_year AS pmu_birth_year,
            p.match_stage,
            p.match_confidence,
            i.horse_key AS ifce_key,
            i.name AS ifce_name,
            i.birth_year AS ifce_birth_year,
            i.sex,
            i.country,
            i.breed
        FROM pmu_horses p
        LEFT JOIN ifce_horses i ON p.ifce_horse_key = i.horse_key
    """)
    
    # Vue : performances enrichies
    cur.execute("""
        CREATE VIEW IF NOT EXISTS v_performances_enriched AS
        SELECT 
            p.*,
            i.name AS horse_name,
            i.breed,
            i.country
        FROM performances p
        LEFT JOIN ifce_horses i ON p.horse_key = i.horse_key
    """)
    
    # Vue : stats annuelles avec noms
    cur.execute("""
        CREATE VIEW IF NOT EXISTS v_year_stats AS
        SELECT 
            s.horse_key,
            i.name,
            i.breed,
            s.year,
            s.gains_annuels_eur,
            s.nb_courses,
            s.nb_victoires
        FROM horse_year_stats s
        JOIN ifce_horses i ON s.horse_key = i.horse_key
    """)
    
    conn.commit()
    print("✅ Migrations terminées avec succès!\n")


def rollback_migrations(conn: sqlite3.Connection):
    """
    Supprime les tables créées par les migrations (ATTENTION: destructif!)
    Utiliser uniquement pour tests ou réinitialisation complète.
    """
    cur = conn.cursor()
    
    print("⚠️  ROLLBACK : Suppression des tables enrichissement...")
    
    tables_to_drop = [
        'v_year_stats',
        'v_performances_enriched',
        'v_horses_matched',
        'ifce_horses_fts',
        'horse_aliases',
        'horse_totals',
        'horse_year_stats',
        'performances',
        'pmu_horses',
        'ifce_horses',
    ]
    
    for table in tables_to_drop:
        try:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute(f"DROP VIEW IF EXISTS {table}")
            print(f"  ✓ Supprimé : {table}")
        except Exception as e:
            print(f"  ✗ Erreur {table} : {e}")
    
    conn.commit()
    print("✅ Rollback terminé\n")


if __name__ == '__main__':
    import sys
    
    DB_PATH = "data/database.db"
    
    # Mode rollback si --rollback
    if '--rollback' in sys.argv:
        print("⚠️  MODE ROLLBACK ACTIVÉ\n")
        response = input("Voulez-vous vraiment supprimer les tables d'enrichissement? (oui/non): ")
        if response.lower() == 'oui':
            conn = sqlite3.connect(DB_PATH)
            rollback_migrations(conn)
            conn.close()
        else:
            print("Annulé.")
    else:
        # Mode normal : appliquer migrations
        enable_fts = '--enable-fts' in sys.argv
        
        print(f"Base de données : {DB_PATH}")
        print(f"FTS activé : {enable_fts}\n")
        
        conn = sqlite3.connect(DB_PATH)
        run_migrations(conn, enable_fts=enable_fts)
        conn.close()
        
        print("Pour activer le FTS (fuzzy matching) :")
        print(f"  python {sys.argv[0]} --enable-fts\n")
