#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration vers structure simplifiée : 2 tables seulement
- chevaux : toutes les infos sur les chevaux
- cheval_courses_seen : toutes les courses avec tous les détails

Ce script va :
1. Enrichir la table chevaux avec colonnes manquantes
2. Enrichir la table cheval_courses_seen avec toutes les infos de course
3. Migrer les données des 5 nouvelles tables vers ces 2 tables
4. Supprimer les 5 nouvelles tables
"""

import sqlite3
import sys

DB_PATH = "data/database.db"

def backup_database():
    """Créer un backup avant modification"""
    import shutil
    from datetime import datetime
    
    backup_path = f"data/database.db.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(DB_PATH, backup_path)
    print(f"✅ Backup créé: {backup_path}")
    return backup_path

def enrich_chevaux_table(cur):
    """Ajouter les colonnes manquantes à la table chevaux"""
    print("\n📋 Enrichissement de la table chevaux...")
    
    # Liste des colonnes à ajouter
    new_columns = [
        ("num_pmu", "INTEGER"),                    # ID PMU interne
        ("nom_pere", "TEXT"),                      # Père
        ("nom_mere", "TEXT"),                      # Mère
        ("proprietaire", "TEXT"),                  # Propriétaire principal
        ("eleveur", "TEXT"),                       # Éleveur
        ("musique_complete", "TEXT"),              # Musique complète (prioritaire)
    ]
    
    # Vérifier quelles colonnes existent déjà
    cur.execute("PRAGMA table_info(chevaux)")
    existing_cols = {row[1] for row in cur.fetchall()}
    
    # Ajouter les colonnes manquantes
    added = 0
    for col_name, col_type in new_columns:
        if col_name not in existing_cols:
            cur.execute(f"ALTER TABLE chevaux ADD COLUMN {col_name} {col_type}")
            print(f"  ✓ Ajout colonne: {col_name}")
            added += 1
    
    if added == 0:
        print("  ℹ️  Toutes les colonnes existent déjà")
    
    return added

def enrich_cheval_courses_seen_table(cur):
    """Transformer cheval_courses_seen en table complète de courses"""
    print("\n📋 Enrichissement de la table cheval_courses_seen...")
    
    # Liste COMPLÈTE des colonnes à ajouter
    new_columns = [
        # Infos réunion
        ("reunion_numero", "INTEGER"),
        ("course_numero", "INTEGER"),
        ("hippodrome_code", "TEXT"),
        ("hippodrome_nom", "TEXT"),
        ("meteo", "TEXT"),
        ("etat_piste", "TEXT"),
        
        # Infos course
        ("course_nom", "TEXT"),
        ("discipline", "TEXT"),
        ("specialite", "TEXT"),
        ("distance_m", "INTEGER"),
        ("type_depart", "TEXT"),
        ("corde", "TEXT"),
        ("type_piste", "TEXT"),
        ("allocation_totale", "INTEGER"),
        ("conditions_course", "TEXT"),
        ("type_course", "TEXT"),
        ("heure_depart", "TEXT"),
        
        # Infos participant (dans cette course)
        ("numero_dossard", "INTEGER"),
        ("num_pmu", "INTEGER"),
        ("driver_jockey", "TEXT"),
        ("entraineur", "TEXT"),
        ("proprietaire", "TEXT"),
        ("age", "INTEGER"),
        ("sexe", "TEXT"),
        ("poids_kg", "REAL"),
        
        # Équipement
        ("deferrage", "TEXT"),
        ("equipement", "TEXT"),
        ("handicap_distance", "INTEGER"),
        
        # Cotes
        ("cote_matin", "REAL"),
        ("cote_finale", "REAL"),
        
        # Résultats détaillés
        ("place_finale", "INTEGER"),
        ("statut_arrivee", "TEXT"),
        ("temps_str", "TEXT"),
        ("temps_sec", "REAL"),
        ("reduction_km_sec", "REAL"),
        ("ecarts", "TEXT"),
        ("gains_course", "INTEGER"),
        
        # Statuts spéciaux
        ("non_partant", "INTEGER DEFAULT 0"),
        ("disqualifie", "INTEGER DEFAULT 0"),
        ("observations", "TEXT"),
        
        # Rapports PMU (pour cette course)
        ("rapport_gagnant", "REAL"),
        ("rapport_place", "REAL"),
        ("rapport_couple", "TEXT"),
        ("rapport_trio", "TEXT"),
        
        # IDs techniques
        ("pmu_reunion_id", "INTEGER"),
        ("pmu_course_id", "INTEGER"),
    ]
    
    # Vérifier quelles colonnes existent déjà
    cur.execute("PRAGMA table_info(cheval_courses_seen)")
    existing_cols = {row[1] for row in cur.fetchall()}
    
    # Ajouter les colonnes manquantes
    added = 0
    for col_name, col_type in new_columns:
        if col_name not in existing_cols:
            cur.execute(f"ALTER TABLE cheval_courses_seen ADD COLUMN {col_name} {col_type}")
            print(f"  ✓ Ajout colonne: {col_name}")
            added += 1
    
    if added == 0:
        print("  ℹ️  Toutes les colonnes existent déjà")
    
    return added

def migrate_data_from_new_tables(cur):
    """Migrer les données des 5 nouvelles tables vers les 2 tables principales"""
    print("\n🔄 Migration des données...")
    
    # Stratégie simple : reconstruire cheval_courses_seen depuis les nouvelles tables
    # puis compléter avec les anciennes données
    
    print("  📦 Comptage des données à migrer...")
    
    cur.execute("SELECT COUNT(*) FROM race_participants")
    total_participants = cur.fetchone()[0]
    print(f"    • {total_participants} participants dans race_participants")
    
    if total_participants == 0:
        print("    ℹ️  Aucune donnée à migrer depuis les nouvelles tables")
        return 0
    
    # Migrer les données participant par participant
    print("  📦 Migration des participants...")
    
    cur.execute("""
        SELECT 
            rp.horse_name_norm,
            r.race_date,
            rm.meeting_number,
            r.race_number,
            rm.venue_code,
            rp.saddle_number,
            rp.horse_num_pmu,
            rp.driver_jockey,
            rp.trainer,
            rp.owner,
            rp.age,
            rp.sex,
            rp.weight_kg,
            rp.shoeing,
            rp.equipment,
            rp.handicap_distance,
            rp.morning_odds,
            rp.final_odds,
            rp.finish_position,
            rp.finish_status,
            rp.finish_time_str,
            rp.finish_time_sec,
            rp.reduction_km_sec,
            rp.gaps,
            rp.earnings_race,
            rp.is_non_runner,
            rp.is_disqualified,
            rp.post_race_notes,
            r.discipline,
            r.specialty,
            r.distance_m,
            r.start_method,
            r.rope_side,
            r.track_surface,
            r.total_allocation,
            r.race_name,
            r.race_conditions,
            r.race_type,
            r.start_time,
            rm.venue_name,
            rm.weather,
            rm.track_condition
        FROM race_participants rp
        JOIN races r ON rp.race_id = r.race_id
        JOIN race_meetings rm ON r.meeting_id = rm.meeting_id
    """)
    
    rows = cur.fetchall()
    migrated = 0
    
    for row in rows:
        (horse_name_norm, race_date, meeting_num, race_num, venue_code,
         saddle_num, num_pmu, driver, trainer, owner, age, sex, weight,
         shoeing, equipment, handicap, cote_matin, cote_finale,
         finish_pos, finish_status, finish_time_str, finish_time_sec,
         reduction_km, gaps, earnings, is_np, is_dq, observations,
         discipline, specialty, distance, start_method, rope_side, track_surface,
         allocation, race_name, race_conditions, race_type, start_time,
         venue_name, weather, track_condition) = row
        
        # Construire race_key au format attendu
        race_key = f"{race_date}|R{meeting_num}|C{race_num}|{venue_code or '?'}"
        annee = int(race_date[:4])
        is_win = 1 if finish_pos == 1 else 0
        
        # INSERT OR REPLACE dans cheval_courses_seen
        cur.execute("""
            INSERT OR REPLACE INTO cheval_courses_seen (
                nom_norm, race_key, annee, is_win,
                reunion_numero, course_numero, hippodrome_code, hippodrome_nom,
                meteo, etat_piste,
                course_nom, discipline, specialite, distance_m,
                type_depart, corde, type_piste, allocation_totale,
                conditions_course, type_course, heure_depart,
                numero_dossard, num_pmu, driver_jockey, entraineur, proprietaire,
                age, sexe, poids_kg,
                deferrage, equipement, handicap_distance,
                cote_matin, cote_finale,
                place_finale, statut_arrivee, temps_str, temps_sec,
                reduction_km_sec, ecarts, gains_course,
                non_partant, disqualifie, observations
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
        """, (
            horse_name_norm, race_key, annee, is_win,
            meeting_num, race_num, venue_code, venue_name,
            weather, track_condition,
            race_name, discipline, specialty, distance,
            start_method, rope_side, track_surface, allocation,
            race_conditions, race_type, start_time,
            saddle_num, num_pmu, driver, trainer, owner,
            age, sex, weight,
            shoeing, equipment, handicap,
            cote_matin, cote_finale,
            finish_pos, finish_status, finish_time_str, finish_time_sec,
            reduction_km, gaps, earnings,
            is_np, is_dq, observations
        ))
        
        migrated += 1
        
        if migrated % 100 == 0:
            print(f"    ... {migrated}/{total_participants}")
    
    print(f"    ✓ {migrated} participants migrés")
    
    # Migrer les infos pedigree vers chevaux
    print("  📦 Migration pedigree → chevaux...")
    
    cur.execute("""
        UPDATE chevaux
        SET num_pmu = (
            SELECT rp.horse_num_pmu
            FROM race_participants rp
            WHERE LOWER(rp.horse_name_norm) = LOWER(chevaux.nom)
            LIMIT 1
        ),
        nom_pere = (
            SELECT rp.sire
            FROM race_participants rp
            WHERE LOWER(rp.horse_name_norm) = LOWER(chevaux.nom)
              AND rp.sire IS NOT NULL
            LIMIT 1
        ),
        nom_mere = (
            SELECT rp.dam
            FROM race_participants rp
            WHERE LOWER(rp.horse_name_norm) = LOWER(chevaux.nom)
              AND rp.dam IS NOT NULL
            LIMIT 1
        ),
        proprietaire = (
            SELECT rp.owner
            FROM race_participants rp
            WHERE LOWER(rp.horse_name_norm) = LOWER(chevaux.nom)
              AND rp.owner IS NOT NULL
            LIMIT 1
        ),
        eleveur = (
            SELECT rp.breeder
            FROM race_participants rp
            WHERE LOWER(rp.horse_name_norm) = LOWER(chevaux.nom)
              AND rp.breeder IS NOT NULL
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM race_participants rp
            WHERE LOWER(rp.horse_name_norm) = LOWER(chevaux.nom)
        )
    """)
    
    pedigree_updated = cur.rowcount
    print(f"    ✓ {pedigree_updated} chevaux mis à jour")
    
    return migrated

def drop_new_tables(cur):
    """Supprimer les 5 nouvelles tables devenues inutiles"""
    print("\n🗑️  Suppression des tables inutiles...")
    
    tables_to_drop = [
        "race_meetings",
        "races",
        "race_participants",
        "race_betting",
        "race_incidents"
    ]
    
    for table in tables_to_drop:
        try:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"  ✓ Table supprimée: {table}")
        except Exception as e:
            print(f"  ⚠️  Erreur suppression {table}: {e}")

def main():
    print("=" * 60)
    print("MIGRATION VERS STRUCTURE SIMPLIFIÉE (2 TABLES)")
    print("=" * 60)
    
    # Confirmation
    print("\n⚠️  ATTENTION : Cette opération va :")
    print("  1. Enrichir les tables chevaux et cheval_courses_seen")
    print("  2. Migrer les données des 5 nouvelles tables")
    print("  3. Supprimer les 5 nouvelles tables")
    print("\n  Un backup sera créé automatiquement.")
    
    response = input("\n👉 Continuer ? (oui/non) : ").strip().lower()
    if response not in ["oui", "o", "yes", "y"]:
        print("❌ Annulé")
        return 1
    
    try:
        # Backup
        backup_path = backup_database()
        
        # Connexion
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        
        # Étape 1: Enrichir les tables
        chevaux_cols = enrich_chevaux_table(cur)
        courses_cols = enrich_cheval_courses_seen_table(cur)
        con.commit()
        
        # Étape 2: Migrer les données
        migrated = migrate_data_from_new_tables(cur)
        con.commit()
        
        # Étape 3: Supprimer les anciennes tables
        print("\n⚠️  Voulez-vous supprimer les 5 nouvelles tables maintenant ?")
        print("  (Vous pouvez garder les données migrées et supprimer plus tard)")
        response = input("👉 Supprimer ? (oui/non) : ").strip().lower()
        
        if response in ["oui", "o", "yes", "y"]:
            drop_new_tables(cur)
            con.commit()
        else:
            print("  ℹ️  Tables conservées (vous pouvez les supprimer plus tard)")
        
        con.close()
        
        print("\n" + "=" * 60)
        print("✅ MIGRATION TERMINÉE !")
        print("=" * 60)
        print(f"\n📊 Résumé :")
        print(f"  • Colonnes ajoutées à chevaux: {chevaux_cols}")
        print(f"  • Colonnes ajoutées à cheval_courses_seen: {courses_cols}")
        print(f"  • Lignes migrées: {migrated}")
        print(f"\n💾 Backup: {backup_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
