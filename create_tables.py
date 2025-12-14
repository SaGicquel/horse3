import sqlite3
import os

# Créer le dossier data s'il n'existe pas
os.makedirs('data', exist_ok=True)

# Connexion à la base SQLite
conn = sqlite3.connect('data/database.db')
cursor = conn.cursor()

# Créer la table horses (avec race_id)
cursor.execute('''
CREATE TABLE IF NOT EXISTS horses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    number INTEGER,
    age INTEGER,
    gender TEXT,
    weight REAL,
    jockey TEXT,
    trainer TEXT,
    owner TEXT,
    distance INTEGER,
    race_time TEXT,
    allocation REAL,
    race_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id)
)
''')

# Créer la table races
cursor.execute('''
CREATE TABLE IF NOT EXISTS races (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    title TEXT,
    hour TEXT,
    status TEXT,
    race_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Créer la table race_results
cursor.execute('''
CREATE TABLE IF NOT EXISTS race_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER,
    horse_id INTEGER,
    position INTEGER,
    odds TEXT,
    FOREIGN KEY (race_id) REFERENCES races(id),
    FOREIGN KEY (horse_id) REFERENCES horses(id),
    UNIQUE(race_id, horse_id)
)
''')

# Créer la table ifce_horses (pour le matching)
cursor.execute('''
CREATE TABLE IF NOT EXISTS ifce_horses (
    horse_key INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    name_norm TEXT,
    birth_year INTEGER,
    sex TEXT,
    country TEXT,
    sire_name TEXT,
    dam_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Créer la table horse_aliases (pour le cache de matching)
cursor.execute('''
CREATE TABLE IF NOT EXISTS horse_aliases (
    pmu_name_norm TEXT PRIMARY KEY,
    ifce_horse_key INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ifce_horse_key) REFERENCES ifce_horses(horse_key)
)
''')

conn.commit()
print("✅ Tables créées avec succès dans data/database.db")
conn.close()
