#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Création du schéma étendu pour capturer TOUTES les infos PMU
- Réunions (calendrier, hippodromes)
- Courses (discipline, distance, allocations)
- Partants (dossards, équipement, résultats)
- Paris & Rapports
- Incidents
"""

import sqlite3
import sys

DB_PATH = "data/database.db"

def create_extended_schema(con):
    """Crée toutes les nouvelles tables pour l'enrichissement PMU"""
    cur = con.cursor()
    
    print("📋 Création du schéma étendu...")
    
    # ========================================
    # TABLE 1: race_meetings - Réunions
    # ========================================
    print("  • Création de race_meetings...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS race_meetings (
            meeting_id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_date DATE NOT NULL,
            meeting_number INTEGER NOT NULL,      -- R1, R2, R3...
            venue_code TEXT,                       -- Code hippodrome (ex: MAI)
            venue_name TEXT,                       -- Nom complet
            venue_country TEXT,                    -- Pays
            start_time TEXT,                       -- Heure de début
            weather TEXT,                          -- Météo
            track_condition TEXT,                  -- État piste
            status TEXT,                           -- scheduled, ongoing, completed
            pmu_meeting_id INTEGER,                -- ID interne PMU
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(race_date, meeting_number)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_meetings_date ON race_meetings(race_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_meetings_venue ON race_meetings(venue_code);")
    
    # ========================================
    # TABLE 2: races - Courses détaillées
    # ========================================
    print("  • Création de races...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS races (
            race_id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id INTEGER,
            race_date DATE NOT NULL,
            race_number INTEGER NOT NULL,          -- C1, C2, C3...
            race_code TEXT NOT NULL,               -- R1C1, R2C3...
            
            -- Discipline
            discipline TEXT,                       -- Trot, Plat, Obstacle
            specialty TEXT,                        -- Attelé, Monté, Haies, Steeple
            
            -- Caractéristiques
            distance_m INTEGER,                    -- Distance en mètres
            start_method TEXT,                     -- Autostart, Volte, Élastique
            rope_side TEXT,                        -- Gauche, Droite
            track_surface TEXT,                    -- Sable, Herbe, PSF, Fibresand
            track_condition TEXT,                  -- Bon, Souple, Collant, Lourd
            
            -- Dotation
            total_allocation INTEGER,              -- Dotation totale en €
            
            -- Infos supplémentaires
            race_name TEXT,                        -- Nom de la course
            race_conditions TEXT,                  -- Conditions (âge, sexe, gains)
            race_type TEXT,                        -- Handicap, Réclamation, Conditions
            start_time TEXT,                       -- Heure de départ
            num_runners INTEGER,                   -- Nombre de partants
            
            -- Statut
            status TEXT,                           -- scheduled, running, completed, cancelled
            
            -- IDs PMU
            pmu_race_id INTEGER,                   -- ID interne PMU
            pmu_meeting_id INTEGER,                -- ID réunion PMU
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (meeting_id) REFERENCES race_meetings(meeting_id),
            UNIQUE(race_date, race_code)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_races_code ON races(race_code);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_races_meeting ON races(meeting_id);")
    
    # ========================================
    # TABLE 3: race_participants - Partants
    # ========================================
    print("  • Création de race_participants...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS race_participants (
            participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id INTEGER NOT NULL,
            pmu_horse_id INTEGER,                  -- FK vers pmu_horses
            
            -- Identification cheval
            horse_name TEXT NOT NULL,
            horse_name_norm TEXT NOT NULL,
            horse_num_pmu INTEGER,                 -- numPmu (ID PMU)
            saddle_number INTEGER,                 -- Numéro de corde/dossard
            
            -- Personnel
            driver_jockey TEXT,
            trainer TEXT,
            owner TEXT,
            breeder TEXT,
            
            -- Caractéristiques cheval
            age INTEGER,
            sex TEXT,
            country TEXT,
            breed TEXT,
            coat TEXT,
            
            -- Pedigree
            sire TEXT,                             -- Père
            dam TEXT,                              -- Mère
            
            -- Statistiques pré-course
            career_earnings INTEGER,               -- Gains carrière
            career_starts INTEGER,                 -- Nombre de courses
            career_wins INTEGER,                   -- Nombre de victoires
            music TEXT,                            -- Musique
            
            -- Équipement & Handicap
            shoeing TEXT,                          -- Déferrage
            equipment TEXT,                        -- Œillères, etc.
            handicap_distance INTEGER,             -- Handicap en mètres
            weight_kg REAL,                        -- Poids porté
            
            -- Cotes
            morning_odds REAL,                     -- Cote du matin
            final_odds REAL,                       -- Cote finale
            
            -- Résultats
            finish_position INTEGER,               -- Place finale (1, 2, 3...)
            finish_status TEXT,                    -- Arrivé, DAI, NP, Distancé, Arrêté, Tombé
            finish_time_str TEXT,                  -- Temps brut (ex: 1'12"3)
            finish_time_sec REAL,                  -- Temps en secondes
            reduction_km_sec REAL,                 -- Réduction kilométrique
            gaps TEXT,                             -- Écarts (ex: "1L 1/4, 3/4")
            earnings_race INTEGER,                 -- Gains pour cette course
            
            -- Statut
            is_non_runner BOOLEAN DEFAULT 0,       -- Non-partant
            is_disqualified BOOLEAN DEFAULT 0,     -- Disqualifié
            pre_race_notes TEXT,                   -- Remarques avant course
            post_race_notes TEXT,                  -- Observations après course
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (race_id) REFERENCES races(race_id),
            FOREIGN KEY (pmu_horse_id) REFERENCES pmu_horses(pmu_horse_id)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_participants_race ON race_participants(race_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_participants_horse ON race_participants(pmu_horse_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_participants_name ON race_participants(horse_name_norm);")
    
    # ========================================
    # TABLE 4: race_betting - Paris & Rapports
    # ========================================
    print("  • Création de race_betting...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS race_betting (
            betting_id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id INTEGER NOT NULL,
            
            -- Type de pari
            bet_type TEXT NOT NULL,                -- Gagnant, Placé, Couplé, Trio, Quarté, Quinté, Multi, Pick5
            
            -- Résultat
            winning_combination TEXT,              -- Ex: "5-12-3" pour trio
            dividend REAL,                         -- Rapport pour 1€
            pool_total INTEGER,                    -- Montant total enjeux
            
            -- Détails supplémentaires
            base_stake REAL,                       -- Mise de base
            num_winners INTEGER,                   -- Nombre de gagnants
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (race_id) REFERENCES races(race_id)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_betting_race ON race_betting(race_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_betting_type ON race_betting(bet_type);")
    
    # ========================================
    # TABLE 5: race_incidents - Incidents
    # ========================================
    print("  • Création de race_incidents...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS race_incidents (
            incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id INTEGER NOT NULL,
            
            incident_type TEXT,                    -- Réclamation, Disqualification, Chute, Enquête
            description TEXT,
            affected_horses TEXT,                  -- Numéros des chevaux concernés
            stewards_decision TEXT,                -- Décision des commissaires
            incident_time TEXT,                    -- Moment de l'incident
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (race_id) REFERENCES races(race_id)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_incidents_race ON race_incidents(race_id);")
    
    con.commit()
    print("✅ Schéma étendu créé avec succès!")
    
    # Afficher un résumé
    print("\n📊 Résumé des nouvelles tables:")
    tables = [
        "race_meetings",
        "races", 
        "race_participants",
        "race_betting",
        "race_incidents"
    ]
    
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"   • {table}: {count} lignes")

def main():
    try:
        con = sqlite3.connect(DB_PATH)
        create_extended_schema(con)
        con.close()
        print("\n✅ Création du schéma terminée!")
        print(f"📁 Base de données: {DB_PATH}")
        return 0
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
