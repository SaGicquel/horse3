#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration des données de l'ancienne structure (table 'chevaux') 
vers la nouvelle structure (pmu_horses + performances)
"""

import sqlite3
import json
from datetime import datetime
from enrichment.normalization import normalize_name

DB_PATH = "data/database.db"

def migrate_chevaux_to_pmu_horses():
    """Migre la table chevaux vers pmu_horses et performances"""
    
    print("🔄 Migration de l'ancienne structure vers la nouvelle...\n")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Vérifier que les tables existent
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chevaux'")
    if not cur.fetchone():
        print("❌ Table 'chevaux' introuvable")
        conn.close()
        return
    
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pmu_horses'")
    if not cur.fetchone():
        print("❌ Table 'pmu_horses' introuvable. Lancez d'abord : python cli.py init-db")
        conn.close()
        return
    
    # Compter les chevaux à migrer
    cur.execute("SELECT COUNT(*) as total FROM chevaux WHERE nom IS NOT NULL")
    total = cur.fetchone()['total']
    print(f"📊 Chevaux à migrer : {total}")
    
    # Migration pmu_horses
    print("\n[1/2] Migration vers pmu_horses...")
    
    # Nettoyer d'abord pour éviter les doublons
    print("   🧹 Nettoyage des anciennes données...")
    cur.execute("DELETE FROM performances")
    cur.execute("DELETE FROM horse_year_stats")
    cur.execute("DELETE FROM horse_totals")
    cur.execute("DELETE FROM pmu_horses")
    conn.commit()
    
    print("   📥 Import des chevaux PMU...")
    cur.execute("""
        INSERT OR IGNORE INTO pmu_horses 
            (name, name_norm, sex, birth_year, country, breed, first_seen_date, last_seen_date)
        SELECT 
            nom,
            LOWER(REPLACE(REPLACE(REPLACE(nom, ' ', ''), '-', ''), '''', '')),
            sexe,
            CAST(substr(date_naissance, 1, 4) AS INTEGER),
            pays_naissance,
            race,
            date_derniere_course,
            date_derniere_course
        FROM chevaux
        WHERE nom IS NOT NULL
          AND nom != ''
    """)
    
    migrated_horses = cur.rowcount
    conn.commit()
    print(f"   ✅ {migrated_horses} chevaux migrés vers pmu_horses")
    
    # Migration performances (depuis dernieres_performances JSON)
    print("\n[2/3] Migration performances depuis dernieres_performances...")
    
    cur.execute("""
        SELECT 
            pmu_horse_id,
            name,
            name_norm
        FROM pmu_horses
    """)
    
    pmu_horses = {row['name_norm']: row['pmu_horse_id'] for row in cur.fetchall()}
    
    # Récupérer les performances depuis chevaux
    cur.execute("""
        SELECT 
            nom,
            dernieres_performances
        FROM chevaux
        WHERE dernieres_performances IS NOT NULL
          AND dernieres_performances != ''
    """)
    
    perf_count = 0
    error_count = 0
    
    for row in cur.fetchall():
        nom = row['nom']
        nom_norm = normalize_name(nom)
        
        if nom_norm not in pmu_horses:
            continue
        
        horse_key = pmu_horses[nom_norm]
        
        try:
            perfs = json.loads(row['dernieres_performances'])
            if not isinstance(perfs, list):
                continue
            
            for perf in perfs:
                if not isinstance(perf, dict):
                    continue
                
                # Extraction des données de performance
                race_date = perf.get('date')
                venue = perf.get('hippodrome')
                discipline = perf.get('discipline')
                distance = perf.get('distance')
                finish = perf.get('place') or perf.get('classement')
                allocation = perf.get('gain') or perf.get('allocation')
                
                if not race_date or race_date == '?':
                    continue
                
                # Insérer la performance
                cur.execute("""
                    INSERT OR IGNORE INTO performances 
                        (horse_key, race_date, venue, discipline, distance_m, 
                         finish_position, allocation_eur)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    horse_key,
                    race_date,
                    venue,
                    discipline,
                    distance,
                    str(finish) if finish else None,
                    float(allocation) if allocation else None
                ))
                
                if cur.rowcount > 0:
                    perf_count += 1
        
        except json.JSONDecodeError:
            error_count += 1
            continue
        except Exception as e:
            error_count += 1
            continue
    
    conn.commit()
    print(f"   ✅ {perf_count} performances migrées depuis JSON")
    if error_count > 0:
        print(f"   ⚠️  {error_count} erreurs")
    
    # Migration depuis cheval_courses_seen
    print("\n[3/3] Migration performances depuis cheval_courses_seen...")
    
    cur.execute("""
        SELECT 
            ccs.nom_norm,
            ccs.race_key,
            ccs.annee,
            ccs.is_win
        FROM cheval_courses_seen ccs
        WHERE ccs.race_key IS NOT NULL
    """)
    
    races_seen_count = 0
    
    for row in cur.fetchall():
        nom_norm = row['nom_norm']
        race_key = row['race_key']
        annee = row['annee']
        is_win = row['is_win']
        
        if nom_norm not in pmu_horses:
            continue
        
        horse_key = pmu_horses[nom_norm]
        
        # Parse race_key: format semble être "YYYY-MM-DD|R#|C#|VENUE"
        try:
            parts = race_key.split('|')
            if len(parts) >= 4:
                race_date = parts[0]
                venue = parts[3] if len(parts) > 3 else None
                finish = '1' if is_win else None
                
                cur.execute("""
                    INSERT OR IGNORE INTO performances 
                        (horse_key, race_date, venue, finish_position, finish_status)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    horse_key,
                    race_date,
                    venue,
                    finish,
                    '1' if is_win else None
                ))
                
                if cur.rowcount > 0:
                    races_seen_count += 1
        except Exception:
            continue
    
    conn.commit()
    print(f"   ✅ {races_seen_count} performances migrées depuis cheval_courses_seen")
    
    # Statistiques finales
    print("\n" + "=" * 60)
    print("📊 Résumé de migration")
    print("=" * 60)
    
    cur.execute("SELECT COUNT(*) as count FROM pmu_horses")
    pmu_count = cur.fetchone()['count']
    
    cur.execute("SELECT COUNT(*) as count FROM performances")
    perf_total = cur.fetchone()['count']
    
    print(f"✅ pmu_horses     : {pmu_count} chevaux")
    print(f"✅ performances   : {perf_total} enregistrements")
    print("=" * 60)
    
    print("\n💡 Prochaines étapes :")
    print("   1. python cli.py recompute")
    print("   2. python cli.py match-report")
    
    conn.close()


if __name__ == '__main__':
    migrate_chevaux_to_pmu_horses()
