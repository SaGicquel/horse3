#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI unifié pour la gestion complète du système Horse3
Menu interactif pour scraping, vérification BDD, audit, nettoyage, etc.
"""

import sys
# import sqlite3  # REMOVED: Migration to PostgreSQL
import csv
import os
import subprocess
from datetime import datetime, timedelta, date
from pathlib import Path

# Import connexion PostgreSQL
from db_connection import get_connection, get_cursor

# Import des modules enrichissement
from enrichment.migrations import run_migrations
from enrichment.normalization import normalize_name, extract_birth_year, normalize_country, normalize_sex
from enrichment.matching import HorseMatcher
from enrichment.calculations import (
    compute_annual_gains, compute_total_gains, compute_records
)


# DB_PATH = "data/database.db" # REMOVED: Migration to PostgreSQL


def cmd_init_db(args):
    """Initialise la base de données PostgreSQL (tables, index)"""
    
    print("🔧 Initialisation de la base de données PostgreSQL...")
    
    try:
        # Utiliser le script create_tables.py existant s'il est adapté, 
        # ou exécuter les scripts SQL directement
        import create_tables
        # create_tables.main() # Si create_tables a une fonction main
        
        # Alternative: Exécuter le SQL directement
        conn = get_connection()
        cur = conn.cursor()
        
        # Vérifier si on doit exécuter un script SQL spécifique
        # Pour l'instant, on suppose que la structure est gérée par scraper_pmu_simple.db_setup
        # ou par des scripts SQL d'init.
        
        print("   ✅ Connexion PostgreSQL réussie.")
        print("   💡 Utilisez 'python scraper_pmu_simple.py' pour créer/mettre à jour le schéma automatiquement.")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Erreur : {e}")


def cmd_import_ifce(args):
    """Importe le CSV IFCE dans la table ifce_horses"""
    
    # Récupérer le chemin du CSV
    csv_path = None
    for i, arg in enumerate(args):
        if arg == '--path' and i + 1 < len(args):
            csv_path = args[i + 1]
            break
    
    if not csv_path:
        print("❌ Erreur : --path manquant")
        print("Usage : python cli.py import-ifce --path ./fichier-des-equides.csv")
        return
    
    if not Path(csv_path).exists():
        print(f"❌ Erreur : fichier introuvable : {csv_path}")
        return
    
    print(f"📥 Import du CSV IFCE : {csv_path}")
    
    print(f"📥 Import du CSV IFCE : {csv_path}")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Vérifier que la table existe
    cur.execute("SELECT to_regclass('public.ifce_horses')")
    if not cur.fetchone()[0]:
        print("❌ Erreur : table ifce_horses introuvable. Création de la table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ifce_horses (
                id_ifce SERIAL PRIMARY KEY,
                name VARCHAR(255),
                name_norm VARCHAR(255),
                sex VARCHAR(50),
                birth_date DATE,
                birth_year INTEGER,
                country VARCHAR(50),
                breed VARCHAR(100),
                coat VARCHAR(100),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name_norm, birth_year)
            );
            CREATE INDEX IF NOT EXISTS idx_ifce_name_norm ON ifce_horses(name_norm);
        """)
        conn.commit()
        print("   ✅ Table ifce_horses créée.")
    
    # Import CSV
    imported = 0
    skipped = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                name = row.get('NOM', '').strip()
                if not name:
                    skipped += 1
                    continue
                
                name_norm = normalize_name(name)
                if not name_norm:
                    skipped += 1
                    continue
                
                # Extraction des champs
                sex = normalize_sex(row.get('SEXE'))
                birth_date = row.get('DATE_DE_NAISSANCE', '').strip()
                birth_year = extract_birth_year(birth_date)
                country = normalize_country(row.get('PAYS_DE_NAISSANCE'))
                breed = row.get('RACE', '').strip().upper() if row.get('RACE') else None
                coat = row.get('ROBE', '').strip() if row.get('ROBE') else None
                
                # Insertion ou mise à jour (Syntaxe PostgreSQL)
                cur.execute("""
                    INSERT INTO ifce_horses (
                        name, name_norm, sex, birth_date, birth_year, 
                        country, breed, coat
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(name_norm, birth_year) DO UPDATE SET
                        sex = COALESCE(EXCLUDED.sex, ifce_horses.sex),
                        birth_date = COALESCE(EXCLUDED.birth_date, ifce_horses.birth_date),
                        country = COALESCE(EXCLUDED.country, ifce_horses.country),
                        breed = COALESCE(EXCLUDED.breed, ifce_horses.breed),
                        coat = COALESCE(EXCLUDED.coat, ifce_horses.coat),
                        updated_at = CURRENT_TIMESTAMP
                """, (name, name_norm, sex, birth_date or None, birth_year, 
                      country, breed, coat))
                
                imported += 1
                
                if imported % 1000 == 0:
                    print(f"  → {imported} chevaux importés...")
                    conn.commit()
        
        conn.commit()
        
        print(f"\n✅ Import terminé :")
        print(f"   • {imported} chevaux importés")
        print(f"   • {skipped} lignes ignorées (nom vide ou invalide)\n")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'import : {e}")
        conn.rollback()
    finally:
        conn.close()


def cmd_fetch(args):
    """Lance le scraping PMU pour une date donnée"""
    
    # Récupérer la date
    target_date = None
    for i, arg in enumerate(args):
        if arg == '--date' and i + 1 < len(args):
            target_date = args[i + 1]
            break
    
    if not target_date:
        target_date = date.today().isoformat()
    
    print(f"🏇 Scraping PMU pour : {target_date}")
    print("   → Utilisation de scraper_pmu_simple.py existant\n")
    
    # Importer et lancer le scraper existant
    try:
        import scraper_pmu_simple
        scraper_pmu_simple.run(target_date, recalc_after=True)
        print(f"\n✅ Scraping terminé pour {target_date}")
    except Exception as e:
        print(f"❌ Erreur lors du scraping : {e}")


def cmd_backfill(args):
    """Lance le scraping PMU pour une période (rattrapage historique)"""
    
    # Récupérer le nombre de jours
    days_back = 30
    for i, arg in enumerate(args):
        if arg == '--days-back' and i + 1 < len(args):
            days_back = int(args[i + 1])
            break
    
    print(f"🏇 Rattrapage historique : {days_back} derniers jours\n")
    
    # Générer les dates
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    
    current_date = start_date
    success_count = 0
    error_count = 0
    
    while current_date <= end_date:
        date_str = current_date.isoformat()
        print(f"  → {date_str}...")
        
        try:
            import scraper_pmu_simple
            scraper_pmu_simple.run(date_str, recalc_after=False)
            success_count += 1
        except Exception as e:
            print(f"    ✗ Erreur : {e}")
            error_count += 1
        
        current_date += timedelta(days=1)
    
    print(f"\n✅ Backfill terminé :")
    print(f"   • {success_count} jours scrapés")
    print(f"   • {error_count} erreurs\n")


def cmd_recompute(args):
    """Recalcule les gains et records depuis les performances
    
    Optimisé avec :
    - SQL agrégé pour gains/stats (évite boucles Python)
    - Batch processing pour records
    - Commit par batch de 500 chevaux
    """
    
    print("📊 Recalcul optimisé des gains et records...\n")
    
    conn = get_connection()
    # conn.row_factory = sqlite3.Row  # Pas nécessaire avec psycopg2 DictCursor ou accès par index
    cur = conn.cursor()
    
    # Vérifier que les tables existent
    cur.execute("SELECT to_regclass('public.performances')") # Ou cheval_courses_seen selon le schéma
    # Note: Le scraper utilise 'cheval_courses_seen', on va adapter pour utiliser cette table
    # Si 'performances' est l'ancienne table, on doit utiliser 'cheval_courses_seen'
    
    # On vérifie cheval_courses_seen car c'est la table principale du scraper actuel
    cur.execute("SELECT to_regclass('public.cheval_courses_seen')")
    if not cur.fetchone()[0]:
        print("❌ Erreur : table cheval_courses_seen introuvable")
        conn.close()
        return
    
    # ========================================
    # ÉTAPE 1 : Gains annuels (SQL agrégé)
    # ========================================
    print("   [1/3] Calcul gains annuels (SQL agrégé)...")
    
    # Adaptation pour PostgreSQL et table cheval_courses_seen
    # On suppose que horse_year_stats existe ou on la crée
    cur.execute("""
        CREATE TABLE IF NOT EXISTS horse_year_stats (
            horse_key VARCHAR(255),
            year INTEGER,
            gains_annuels_eur REAL,
            nb_courses INTEGER,
            nb_victoires INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (horse_key, year)
        )
    """)
    conn.commit()

    cur.execute("""
        INSERT INTO horse_year_stats 
            (horse_key, year, gains_annuels_eur, nb_courses, nb_victoires, updated_at)
        SELECT 
            nom_norm as horse_key,
            annee as year,
            SUM(CASE 
                WHEN is_win = 1 THEN COALESCE(allocation_premier, 0)
                WHEN place_finale = 2 THEN COALESCE(allocation_deuxieme, 0)
                WHEN place_finale = 3 THEN COALESCE(allocation_troisieme, 0)
                ELSE 0
            END) as gains_annuels_eur,
            COUNT(*) as nb_courses,
            SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as nb_victoires,
            CURRENT_TIMESTAMP
        FROM cheval_courses_seen
        WHERE nom_norm IS NOT NULL
          AND annee IS NOT NULL
        GROUP BY nom_norm, annee
        ON CONFLICT (horse_key, year) DO UPDATE SET
            gains_annuels_eur = EXCLUDED.gains_annuels_eur,
            nb_courses = EXCLUDED.nb_courses,
            nb_victoires = EXCLUDED.nb_victoires,
            updated_at = CURRENT_TIMESTAMP
    """)
    
    conn.commit()
    print(f"      ✅ {cur.rowcount} statistiques annuelles calculées")
    
    # ========================================
    # ÉTAPE 2 : Gains totaux (SQL agrégé)
    # ========================================
    print("   [2/3] Calcul gains totaux (SQL agrégé)...")
    
    # Création table si nécessaire
    cur.execute("""
        CREATE TABLE IF NOT EXISTS horse_totals (
            horse_key VARCHAR(255) PRIMARY KEY,
            gains_totaux_eur REAL,
            record_attele_sec REAL,
            record_attele_date DATE,
            record_attele_venue VARCHAR(100),
            record_attele_race VARCHAR(100),
            record_monte_sec REAL,
            record_monte_date DATE,
            record_monte_venue VARCHAR(100),
            record_monte_race VARCHAR(100),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    cur.execute("""
        INSERT INTO horse_totals (horse_key, gains_totaux_eur, updated_at)
        SELECT 
            nom_norm as horse_key,
            SUM(CASE 
                WHEN is_win = 1 THEN COALESCE(allocation_premier, 0)
                WHEN place_finale = 2 THEN COALESCE(allocation_deuxieme, 0)
                WHEN place_finale = 3 THEN COALESCE(allocation_troisieme, 0)
                ELSE 0
            END) as gains_totaux_eur,
            CURRENT_TIMESTAMP
        FROM cheval_courses_seen
        WHERE nom_norm IS NOT NULL
        GROUP BY nom_norm
        ON CONFLICT(horse_key) DO UPDATE SET
            gains_totaux_eur = EXCLUDED.gains_totaux_eur,
            updated_at = CURRENT_TIMESTAMP
    """)
    
    conn.commit()
    print(f"      ✅ {cur.rowcount} gains totaux calculés")
    
    # ========================================
    # ÉTAPE 3 : Records (traitement batch)
    # ========================================
    print("   [3/3] Calcul records par discipline (batch processing)...")
    
    # Récupérer chevaux avec réductions
    # Récupérer chevaux avec réductions
    cur.execute("""
        SELECT DISTINCT nom_norm 
        FROM cheval_courses_seen
        WHERE nom_norm IS NOT NULL
          AND reduction_km_sec IS NOT NULL
          AND reduction_km_sec > 0
    """)
    horse_keys = [row[0] for row in cur.fetchall()]
    total = len(horse_keys)
    
    if total == 0:
        print("      ⚠️  Aucun cheval avec réductions kilométriques")
        conn.close()
        return
    
    print(f"      Traitement de {total} chevaux avec records...")
    
    # Préparer requête optimisée pour records (1 requête par cheval)
    record_query = """
        SELECT 
            reduction_km_sec, discipline, race_key, hippodrome_code, course_nom
        FROM cheval_courses_seen
        WHERE nom_norm = %s
          AND reduction_km_sec IS NOT NULL
          AND reduction_km_sec > 0
        ORDER BY reduction_km_sec ASC
        LIMIT 50
    """
    
    update_query = """
        UPDATE horse_totals SET
            record_attele_sec = %s,
            record_attele_date = %s,
            record_attele_venue = %s,
            record_attele_race = %s,
            record_monte_sec = %s,
            record_monte_date = %s,
            record_monte_venue = %s,
            record_monte_race = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE horse_key = %s
    """
    
    batch_size = 500
    for idx, horse_key in enumerate(horse_keys, 1):
        # Récupérer top 50 meilleures perfs (au lieu de toutes)
        cur.execute(record_query, (horse_key,))
        perfs = []
        for row in cur.fetchall():
            # Extraction de la date depuis race_key (format: YYYY-MM-DD|...)
            race_key = row[2]
            try:
                race_date = race_key.split('|')[0]
            except:
                race_date = None

            perfs.append({
                'reduction_km_sec': row[0],
                'discipline': row[1],
                'race_date': race_date,
                'venue': row[3],
                'race_code': row[4],
            })
        
        if not perfs:
            continue
        
        # Calcul records (optimisé car liste triée et limitée)
        record_attele, record_monte = compute_records(perfs)
        
        # Update en batch
        cur.execute(update_query, (
            record_attele.reduction_km_sec if record_attele else None,
            record_attele.date if record_attele else None,
            record_attele.venue if record_attele else None,
            record_attele.race_code if record_attele else None,
            record_monte.reduction_km_sec if record_monte else None,
            record_monte.date if record_monte else None,
            record_monte.venue if record_monte else None,
            record_monte.race_code if record_monte else None,
            horse_key
        ))
        
        # Commit par batch + affichage progression
        if idx % batch_size == 0 or idx == total:
            conn.commit()
            pct = (idx / total) * 100
            print(f"      → {idx}/{total} chevaux ({pct:.1f}%)")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Recalcul terminé : {total} chevaux traités\n")


def cmd_match_report(args):
    """Génère un rapport KPI du matching PMU ↔ IFCE"""
    
    print("📊 Rapport de matching PMU ↔ IFCE\n")
    
    conn = get_connection()
    
    try:
        matcher = HorseMatcher(conn)
        report = matcher.generate_match_report()
        
        total = sum(report.values())
        
        print("=" * 60)
        print(f"{'Stage':<15} {'Nombre':<10} {'Pourcentage'}")
        print("=" * 60)
        
        for stage in ['A', 'B', 'C', 'ambiguous', 'none']:
            count = report.get(stage, 0)
            pct = (count / total * 100) if total > 0 else 0
            
            label = {
                'A': '✅ Strict',
                'B': '🟡 Souple',
                'C': '🔍 Fuzzy',
                'ambiguous': '❓ Ambigus',
                'none': '❌ Non trouvés'
            }.get(stage, stage)
            
            print(f"{label:<15} {count:<10} {pct:>5.1f}%")
        
        print("=" * 60)
        print(f"{'TOTAL':<15} {total:<10} 100.0%")
        print("=" * 60)
        
        # Taux de couverture
        matched = report.get('A', 0) + report.get('B', 0) + report.get('C', 0)
        coverage = (matched / total * 100) if total > 0 else 0
        
        print(f"\n📈 Taux de couverture : {coverage:.1f}%")
        print(f"   ({matched} chevaux matchés sur {total})\n")
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
    finally:
        conn.close()


def cmd_migrate(args):
    """Migre les données de l'ancienne structure (chevaux) vers la nouvelle (pmu_horses + performances)"""
    
    print("🔄 Migration de l'ancienne structure vers la nouvelle...\n")
    
    # Importer le module de migration
    try:
        import migrate_old_to_new
        migrate_old_to_new.migrate_chevaux_to_pmu_horses()
    except Exception as e:
        print(f"❌ Erreur lors de la migration : {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# NOUVELLES COMMANDES CLI (Phase 12+)
# ============================================================================

def cmd_calibrate(args):
    """
    Recalibre le modèle de probabilités (température/Platt) et exporte les artefacts.
    
    Usage: python cli.py calibrate [--days 30] [--sync-config]
    
    - Charge les données des N derniers jours
    - Entraîne la pipeline de calibration
    - Sauvegarde les artefacts dans calibration/
    - --sync-config: Met à jour config/pro_betting.yaml avec les nouveaux paramètres
    """
    import json
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    # Parser --days et --sync-config
    days = 30
    sync_config = False
    for i, arg in enumerate(args):
        if arg == '--days' and i + 1 < len(args):
            days = int(args[i + 1])
        elif arg == '--sync-config':
            sync_config = True
    
    print(f"🎯 CALIBRATION DU MODÈLE (derniers {days} jours)")
    print("=" * 60)
    
    # Créer le dossier calibration si nécessaire
    calibration_dir = Path("calibration")
    calibration_dir.mkdir(exist_ok=True)
    
    try:
        # Import de la pipeline
        from calibration_pipeline import CalibrationPipeline, CalibrationConfig, generate_synthetic_data
        
        # Configuration
        config = CalibrationConfig(
            artifacts_dir=str(calibration_dir),
            temperature_init=1.5,
            min_samples_cluster=100
        )
        
        # Charger les données depuis la BDD
        print("\n📊 Chargement des données historiques...")
        
        conn = get_connection()
        cur = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Récupérer les données avec logits et résultats
        # Note: La date est extraite de race_key (format: YYYY-MM-DD|...)
        cur.execute("""
            SELECT 
                c.race_key as race_id,
                c.nom_norm as horse_id,
                SPLIT_PART(c.race_key, '|', 1)::date as date,
                c.discipline,
                c.is_win as label_win,
                CASE WHEN c.place_finale <= 3 AND c.place_finale > 0 THEN 1 ELSE 0 END as label_place,
                c.cote_finale as odds_market_preoff,
                COALESCE(c.cote_reference, LOG(1.0 / NULLIF(c.cote_finale, 0) + 0.000001)) as logits_model
            FROM cheval_courses_seen c
            WHERE SPLIT_PART(c.race_key, '|', 1)::date >= %s
              AND c.cote_finale IS NOT NULL
              AND c.cote_finale > 1
              AND c.place_finale IS NOT NULL
            ORDER BY SPLIT_PART(c.race_key, '|', 1)::date
        """, (cutoff_date,))
        
        rows = cur.fetchall()
        conn.close()
        
        if len(rows) < 500:
            print(f"⚠️  Pas assez de données ({len(rows)} lignes). Utilisation de données synthétiques.")
            df = generate_synthetic_data(n_races=500, horses_per_race=12)
        else:
            # Construire le DataFrame
            df = pd.DataFrame(rows, columns=[
                'race_id', 'horse_id', 'date', 'discipline', 
                'label_win', 'label_place', 'odds_market_preoff', 'logits_model'
            ])
            
            # Si logits_model est None, le simuler à partir des cotes
            if df['logits_model'].isna().all():
                print("   Génération des logits depuis les cotes...")
                df['logits_model'] = np.log(1 / df['odds_market_preoff'] + 1e-6)
        
        print(f"   {len(df)} observations, {df['race_id'].nunique()} courses")
        
        # Entraîner la pipeline
        print("\n🚀 Entraînement de la pipeline de calibration...")
        pipeline = CalibrationPipeline(config)
        pipeline.fit(
            df,
            logits_col='logits_model',
            odds_col='odds_market_preoff',
            race_col='race_id',
            label_win_col='label_win',
            date_col='date',
            cluster_col='discipline'
        )
        
        # Sauvegarder le fichier health pour cmd_health
        health_data = {
            'last_calibration': datetime.now().isoformat(),
            'days_used': days,
            'n_samples': len(df),
            'n_races': int(df['race_id'].nunique()),
            'temperature': pipeline.scaler.temperature if pipeline.scaler else 1.0,
            'alpha': pipeline.blender.alpha if pipeline.blender else 0.5,
            'calibrator_type': pipeline.best_calibrator_type,
            'metrics': pipeline.metrics.get('calibration', {}),
            'profit_flat': pipeline.metrics.get('profit_flat', {}),
            'profit_kelly': pipeline.metrics.get('profit_kelly', {})
        }
        
        health_path = calibration_dir / "health.json"
        with open(health_path, 'w') as f:
            json.dump(health_data, f, indent=2, default=str)
        
        print(f"\n✅ Calibration terminée!")
        print(f"   Artefacts sauvegardés dans: {calibration_dir}/")
        print(f"   Health file: {health_path}")
        
        # --sync-config: Synchroniser le YAML avec les artefacts
        if sync_config:
            print(f"\n🔄 Synchronisation du YAML avec les artefacts...")
            try:
                from calibration.artifacts_loader import sync_yaml_from_artifacts
                result = sync_yaml_from_artifacts()
                if result['success']:
                    print(f"   ✅ {result['message']}")
                    for change in result.get('changes', []):
                        print(f"      - {change}")
                else:
                    print(f"   ⚠️  {result['message']}")
            except ImportError:
                print(f"   ⚠️  Module artifacts_loader non disponible")
            except Exception as e:
                print(f"   ❌ Erreur sync: {e}")
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("   Assurez-vous que calibration_pipeline.py est présent.")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


def cmd_health(args):
    """
    Affiche le health-check du système de calibration (ECE/Brier 7 jours).
    
    Usage: python cli.py health
    
    Affiche les valeurs exactes depuis calibration/health.json:
    - Temperature (T*)
    - Alpha (α)
    - Calibrator type
    - Métriques de calibration
    """
    import json
    from pathlib import Path
    from datetime import datetime, timedelta
    
    print("🏥 HEALTH CHECK - Système de Calibration")
    print("=" * 60)
    
    calibration_dir = Path("calibration")
    health_path = calibration_dir / "health.json"
    
    if not health_path.exists():
        print("❌ Aucun fichier health.json trouvé.")
        print("   Lancez 'python cli.py calibrate --days 30' d'abord.")
        return
    
    try:
        with open(health_path, 'r') as f:
            health = json.load(f)
        
        last_calib = datetime.fromisoformat(health.get('last_calibration', '1970-01-01'))
        age_days = (datetime.now() - last_calib).days
        
        print(f"\n📅 Dernière calibration: {last_calib.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Âge: {age_days} jour(s)")
        
        if age_days > 7:
            print("   ⚠️  ATTENTION: Calibration vieille de plus de 7 jours!")
        
        # Paramètres de calibration (SOURCE DE VÉRITÉ)
        temperature = health.get('temperature', 'N/A')
        alpha = health.get('alpha', 'N/A')
        calibrator = health.get('calibrator_type', 'N/A')
        
        print(f"\n⚙️  PARAMÈTRES DE CALIBRATION (source: artefacts)")
        print(f"   • Temperature: {temperature:.6f}" if isinstance(temperature, (int, float)) else f"   • Temperature: {temperature}")
        print(f"   • Alpha:       {alpha:.1f}" if isinstance(alpha, (int, float)) else f"   • Alpha:       {alpha}")
        print(f"   • Calibrator:  {calibrator}")
        
        # Vérifier si YAML est synchronisé
        try:
            from calibration.artifacts_loader import check_yaml_artifacts_mismatch
            mismatch = check_yaml_artifacts_mismatch()
            if mismatch['has_mismatch']:
                print(f"\n   ⚠️  YAML désynchronisé! Différences:")
                for diff in mismatch['mismatches']:
                    print(f"      - {diff}")
                print(f"   💡 Exécutez: python cli.py calibrate --sync-config")
            else:
                print(f"\n   ✅ YAML synchronisé avec les artefacts")
        except ImportError:
            pass
        
        # Métriques de calibration
        metrics = health.get('metrics', {})
        print(f"\n📊 Métriques de calibration:")
        print(f"   • Brier Score: {metrics.get('brier_score', 'N/A'):.6f}" if isinstance(metrics.get('brier_score'), (int, float)) else f"   • Brier Score: N/A")
        print(f"   • ECE:         {metrics.get('ece', 'N/A'):.6f}" if isinstance(metrics.get('ece'), (int, float)) else f"   • ECE: N/A")
        print(f"   • Log Loss:    {metrics.get('log_loss', 'N/A'):.6f}" if isinstance(metrics.get('log_loss'), (int, float)) else f"   • Log Loss: N/A")
        
        # Données utilisées
        print(f"\n📈 Données de calibration:")
        print(f"   • Jours utilisés: {health.get('days_used', 'N/A')}")
        print(f"   • Échantillons:   {health.get('n_samples', 'N/A')}")
        print(f"   • Courses:        {health.get('n_races', 'N/A')}")
        
        # Simulation de profit (si disponible)
        profit_flat = health.get('profit_flat', {})
        profit_kelly = health.get('profit_kelly', {})
        
        if profit_flat:
            print(f"\n💰 Simulation Flat Betting (test):")
            print(f"   • Paris: {profit_flat.get('n_bets', 0)}")
            print(f"   • ROI: {profit_flat.get('roi', 0):.1f}%")
        
        if profit_kelly:
            print(f"\n💰 Simulation Kelly (test):")
            print(f"   • Paris: {profit_kelly.get('n_bets', 0)}")
            print(f"   • ROI: {profit_kelly.get('roi', 0):.1f}%")
            print(f"   • Drawdown max: {profit_kelly.get('max_drawdown_pct', 0):.1f}%")
        
        # Status global
        brier = metrics.get('brier_score', 1.0)
        ece = metrics.get('ece', 1.0)
        
        print(f"\n📈 STATUS GLOBAL:")
        if isinstance(brier, (int, float)) and isinstance(ece, (int, float)):
            if brier < 0.15 and ece < 0.05 and age_days <= 7:
                print("   ✅ SYSTÈME SAIN - Calibration OK")
            elif brier < 0.20 and ece < 0.08:
                print("   🟡 ATTENTION - Calibration à surveiller")
            else:
                print("   🔴 ALERTE - Recalibration recommandée")
        else:
            print("   ❓ Métriques indisponibles")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du health check: {e}")


def cmd_pick(args):
    """
    Génère les pronostics du jour (JSON + portfolio YAML).
    
    Usage: python cli.py pick [--date YYYY-MM-DD]
    
    - Charge les courses du jour depuis la BDD
    - Génère les probabilités calibrées avec T/α depuis artefacts
    - Produit un JSON de pronostics et un YAML de portfolio
    """
    import json
    from datetime import datetime, date
    from pathlib import Path
    import yaml
    
    # Parser --date (TODAY par défaut)
    target_date = date.today().isoformat()
    for i, arg in enumerate(args):
        if arg == '--date' and i + 1 < len(args):
            val = args[i + 1]
            if val.upper() == 'TODAY':
                target_date = date.today().isoformat()
            else:
                target_date = val
            break
    
    print(f"🎯 GÉNÉRATION DES PRONOSTICS - {target_date}")
    print("=" * 60)
    
    # Charger l'état de calibration depuis les artefacts
    try:
        from calibration.artifacts_loader import load_calibration_state, log_calibration_init
        state = load_calibration_state(prefer_artifacts=True)
        init_msg = log_calibration_init(state)
        print(f"   {init_msg}")
    except ImportError:
        state = None
        print("   ⚠️  artifacts_loader non disponible, utilisation des défauts")
    
    output_dir = Path("data/picks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) Source canonique : backend /picks/today (probabilités calibrées champion + blend γ/α)
    try:
        import requests
        resp = requests.get("http://localhost:8000/picks/today", timeout=10)
        if resp.ok:
            data = resp.json()
            output_file = output_dir / f"picks_{target_date}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Picks (backend /picks/today) sauvegardés dans {output_file}")
            return
        else:
            print(f"⚠️ Backend /picks/today indisponible ({resp.status_code}), fallback heuristique.")
    except Exception as e:
        print(f"⚠️ Backend /picks/today non joignable ({e}), fallback heuristique.")
    
    try:
        # Import du générateur de pronostics
        from race_pronostic_generator import RacePronosticGenerator, blend_logit_odds
        import numpy as np
        
        # Configuration depuis l'état de calibration (artefacts comme source de vérité)
        if state:
            config = {
                'num_simulations': 20000,
                'max_stake_pct': 0.05,
                'kelly_fraction': 0.25,
                'temperature': state.temperature,
                'blend_alpha': state.alpha,
            }
        else:
            config = {
                'num_simulations': 20000,
                'max_stake_pct': 0.05,
                'kelly_fraction': 0.25,
                'temperature': 1.5,
                'blend_alpha': 0.5,
            }
        
        # Charger les courses du jour
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT DISTINCT race_key, course_nom, hippodrome_nom, discipline
            FROM cheval_courses_seen
            WHERE SPLIT_PART(race_key, '|', 1)::date = %s::date
        """, (target_date,))
        
        races = cur.fetchall()
        
        if not races:
            print(f"❌ Aucune course trouvée pour {target_date}")
            conn.close()
            return
        
        print(f"   {len(races)} courses trouvées")
        
        all_picks = []
        portfolio_bets = []
        
        # On utilise un générateur simplifié si RacePronosticGenerator n'est pas compatible
        for race_key, course_nom, hippodrome, discipline in races:
            # Charger les partants
            cur.execute("""
                SELECT nom_norm as nom, cote_finale as cote_preoff, cote_reference as probability_model, num_pmu as numero
                FROM cheval_courses_seen
                WHERE race_key = %s
                ORDER BY num_pmu
            """, (race_key,))
            
            partants = cur.fetchall()
            if len(partants) < 3:
                continue
            
            # Préparer les données
            horses = []
            odds = []
            model_probs = []
            
            for nom, cote, prob_model, numero in partants:
                horses.append(nom or f"Horse_{numero}")
                odds.append(cote if cote and cote > 1 else 10.0)
                
                if prob_model and prob_model > 0:
                    model_probs.append(prob_model)
                else:
                    # Fallback: probabilité implicite des cotes
                    model_probs.append(1 / cote if cote and cote > 1 else 0.1)
            
            # Normaliser
            model_probs = np.array(model_probs)
            model_probs = model_probs / model_probs.sum()
            
            # Probabilités marché (inversées des cotes normalisées)
            odds_arr = np.array(odds)
            p_market = (1.0 / odds_arr)
            p_market = p_market / p_market.sum()
            
            # Blend modèle/marché
            try:
                p_blend = blend_logit_odds(model_probs, p_market, alpha=config['blend_alpha'])
            except:
                p_blend = model_probs
            
            # Top 3 par probabilité
            sorted_idx = np.argsort(p_blend)[::-1]
            top3 = [{'horse': horses[i], 'prob': round(float(p_blend[i]) * 100, 1)} for i in sorted_idx[:3]]
            
            # Value bets (EV > 5%)
            value_bets = []
            for i, (p, o, h) in enumerate(zip(p_blend, odds, horses)):
                ev = p * o - 1  # EV = p * odds - 1
                if ev > 0.05:  # > 5%
                    # Kelly stake
                    q = 1 - p
                    b = o - 1
                    kelly = (p * b - q) / b if b > 0 else 0
                    stake = min(kelly * config['kelly_fraction'], config['max_stake_pct']) * 100  # En €
                    
                    value_bets.append({
                        'horse': h,
                        'prob': round(float(p) * 100, 1),
                        'odds': round(float(o), 2),
                        'ev': round(float(ev), 3),
                        'stake': round(stake, 2)
                    })
            
            pick = {
                'race_key': race_key,
                'course': course_nom,
                'hippodrome': hippodrome,
                'discipline': discipline,
                'top3': top3,
                'value_bets': value_bets
            }
            all_picks.append(pick)
            
            # Ajouter au portfolio les value bets
            for vb in value_bets:
                portfolio_bets.append({
                    'race': f"{hippodrome} - {course_nom}",
                    'horse': vb['horse'],
                    'stake': vb['stake'],
                    'odds': vb['odds'],
                    'ev_pct': round(vb['ev'] * 100, 1)
                })
        
        conn.close()
        
        # Sauvegarder le JSON des picks
        json_path = output_dir / f"picks_{target_date}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'date': target_date,
                'generated_at': datetime.now().isoformat(),
                'n_races': len(all_picks),
                'picks': all_picks
            }, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder le portfolio YAML
        yaml_path = output_dir / f"portfolio_{target_date}.yaml"
        portfolio = {
            'date': target_date,
            'generated_at': datetime.now().isoformat(),
            'total_stake': round(sum(b['stake'] for b in portfolio_bets), 2),
            'n_bets': len(portfolio_bets),
            'bets': portfolio_bets
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(portfolio, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n✅ Pronostics générés:")
        print(f"   • {len(all_picks)} courses analysées")
        print(f"   • {len(portfolio_bets)} paris value identifiés")
        print(f"   • JSON: {json_path}")
        print(f"   • YAML: {yaml_path}")
        
        # Afficher le top 5 des paris
        if portfolio_bets:
            print(f"\n🏆 Top 5 Value Bets:")
            sorted_bets = sorted(portfolio_bets, key=lambda x: x['ev_pct'], reverse=True)[:5]
            for i, bet in enumerate(sorted_bets, 1):
                print(f"   {i}. {bet['horse']} @ {bet['odds']:.1f} (EV: +{bet['ev_pct']}%)")
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


def cmd_exotic(args):
    """
    Génère les tickets exotiques (Trio/Quinté) pour une course.
    
    Usage: python cli.py exotic --race <race_id>
    
    - Produit 3 packs: SÛR, ÉQUILIBRÉ, RISQUÉ
    - N=20000 simulations Monte Carlo
    """
    import json
    from pathlib import Path
    from datetime import datetime
    
    # Parser --race
    race_id = None
    for i, arg in enumerate(args):
        if arg == '--race' and i + 1 < len(args):
            race_id = args[i + 1]
            break
    
    if not race_id:
        print("❌ Erreur: --race <race_id> requis")
        print("Usage: python cli.py exotic --race R00123")
        return
    
    print(f"🎰 GÉNÉRATION TICKETS EXOTIQUES - {race_id}")
    print("=" * 60)
    
    try:
        from exotic_ticket_generator import ExoticTicketGenerator, ExoticConfig
        import numpy as np
        
        # Charger les données de la course
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT nom_norm as nom, cote_finale as cote_preoff, cote_reference as probability_model, num_pmu as numero
            FROM cheval_courses_seen
            WHERE race_key = %s
            ORDER BY num_pmu
        """, (race_id,))
        
        partants = cur.fetchall()
        conn.close()
        
        if len(partants) < 5:
            print(f"❌ Pas assez de partants ({len(partants)}) pour un pari exotique")
            return
        
        # Préparer les données
        horses = []
        probabilities = []
        
        for nom, cote, prob_model, numero in partants:
            horses.append(nom)
            if prob_model and prob_model > 0:
                probabilities.append(prob_model)
            elif cote and cote > 1:
                probabilities.append(1 / cote)
            else:
                probabilities.append(0.1)
        
        # Normaliser
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        print(f"   {len(horses)} partants")
        
        # Configuration avec N=20000
        config = ExoticConfig(
            num_simulations=20000,
            budget=100.0,
            max_tickets=30,
            structure='trio_ordre'
        )
        
        # Générer les tickets
        generator = ExoticTicketGenerator(config)
        result = generator.generate(
            probabilities=probabilities.tolist(),
            horse_names=horses,
            structure='trio_ordre'
        )
        
        # Sauvegarder
        output_dir = Path("data/exotic")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = output_dir / f"exotic_{race_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Tickets générés: {json_path}")
        
        # Afficher les packs
        print(f"\n📦 PACKS DE TICKETS:")
        for pack in result.get('packs', []):
            label = pack.get('label', '?')
            n_tickets = pack.get('n_tickets', 0)
            total_stake = pack.get('total_stake', 0)
            ev_pct = pack.get('portfolio_ev_pct', 0)
            
            emoji = {'SÛR': '🛡️', 'ÉQUILIBRÉ': '⚖️', 'RISQUÉ': '🎲'}.get(label, '📦')
            print(f"\n   {emoji} {label}:")
            print(f"      • {n_tickets} tickets, {total_stake}€ misés")
            print(f"      • EV portefeuille: {ev_pct:+.1f}%")
            
            # Top 3 tickets du pack
            tickets = pack.get('tickets', [])[:3]
            for t in tickets:
                combo = ' - '.join(t.get('combo', []))
                ev = t.get('ev_pct', '?')
                print(f"      → {combo} (EV: {ev})")
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


def cmd_report(args):
    """
    Publie un rapport de métriques (turnover, EV, drawdown, value deciles).
    
    Usage: python cli.py report [--days 7]
    """
    import json
    from datetime import datetime, timedelta
    from pathlib import Path
    import numpy as np
    
    # Parser --days
    days = 7
    for i, arg in enumerate(args):
        if arg == '--days' and i + 1 < len(args):
            days = int(args[i + 1])
            break
    
    print(f"📊 RAPPORT DE PERFORMANCE ({days} derniers jours)")
    print("=" * 60)
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Récupérer les paris du paper trading log
        paper_log_path = Path("data/paper_trading_log.csv")
        
        if paper_log_path.exists():
            import pandas as pd
            df = pd.read_csv(paper_log_path)
            
            # Filtrer par date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] >= cutoff_date]
            
            if len(df) == 0:
                print(f"❌ Aucun pari trouvé sur les {days} derniers jours")
                conn.close()
                return
            
            print(f"\n📈 Statistiques sur {len(df)} paris:\n")
            
            # Turnover
            if 'stake' in df.columns:
                turnover = df['stake'].sum()
                print(f"   💰 Turnover total: {turnover:.2f}€")
            
            # P&L et ROI
            if 'pnl' in df.columns:
                total_pnl = df['pnl'].sum()
                staked = df['stake'].sum() if 'stake' in df.columns else len(df)
                roi = (total_pnl / staked * 100) if staked > 0 else 0
                print(f"   💵 P&L total: {total_pnl:+.2f}€")
                print(f"   📊 ROI: {roi:+.1f}%")
            
            # Drawdown
            if 'pnl' in df.columns:
                cumsum = df['pnl'].cumsum()
                running_max = cumsum.cummax()
                drawdown = cumsum - running_max
                max_drawdown = drawdown.min()
                print(f"   📉 Drawdown max: {max_drawdown:.2f}€")
            
            # Win rate
            if 'pnl' in df.columns:
                wins = (df['pnl'] > 0).sum()
                win_rate = wins / len(df) * 100
                print(f"   🎯 Win rate: {win_rate:.1f}% ({wins}/{len(df)})")
            
            # EV par décile de cotes
            if 'odds' in df.columns and 'pnl' in df.columns:
                print(f"\n   📊 Performance par décile de cotes:")
                df['odds_decile'] = pd.qcut(df['odds'], q=10, labels=False, duplicates='drop')
                
                for decile in sorted(df['odds_decile'].unique()):
                    subset = df[df['odds_decile'] == decile]
                    avg_odds = subset['odds'].mean()
                    ev = subset['pnl'].sum() / subset['stake'].sum() if 'stake' in subset.columns else 0
                    n = len(subset)
                    print(f"      D{decile+1} (cotes ~{avg_odds:.1f}): EV {ev*100:+.1f}% ({n} paris)")
        else:
            print("❌ Fichier paper_trading_log.csv non trouvé")
            print("   Lancez d'abord le paper trading.")
        
        # Métriques de calibration si disponibles
        health_path = Path("calibration/health.json")
        if health_path.exists():
            with open(health_path, 'r') as f:
                health = json.load(f)
            
            print(f"\n   🎯 Métriques de calibration:")
            metrics = health.get('metrics', {})
            print(f"      • Brier: {metrics.get('brier_score', 'N/A')}")
            print(f"      • ECE: {metrics.get('ece', 'N/A')}")
        
        conn.close()
        
        # Sauvegarder le rapport
        output_dir = Path("data/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_path, 'w') as f:
            f.write(f"Rapport de performance - {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"Période: {days} derniers jours\n")
            # TODO: écrire les métriques détaillées
        
        print(f"\n✅ Rapport sauvegardé: {report_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


def print_help():
    """Affiche l'aide"""
    print("""
🏇 CLI Enrichissement Horse3 - Aide
════════════════════════════════════════════════════════════════

COMMANDES DISPONIBLES :

  init-db [--enable-fts]
      Initialise la base de données avec les tables d'enrichissement.
      --enable-fts : Active le Full-Text Search (fuzzy matching)

  import-ifce --path <fichier.csv>
      Importe le CSV IFCE dans la base de données.
      Exemple : python cli.py import-ifce --path ./fichier-des-equides.csv

  fetch [--date YYYY-MM-DD]
      Lance le scraping PMU pour une date donnée (défaut: aujourd'hui).
      Exemple : python cli.py fetch --date 2025-11-01

  backfill [--days-back N]
      Rattrapage historique des N derniers jours (défaut: 30).
      Exemple : python cli.py backfill --days-back 7

  recompute
      Recalcule les gains annuels/totaux et records depuis les performances.
      À lancer après un scraping important.

  match-report
      Génère un rapport KPI du matching PMU ↔ IFCE.
      Affiche les taux de réussite par stratégie (A/B/C).

  migrate
      Migre les données de l'ancienne structure (table chevaux)
      vers la nouvelle structure (pmu_horses + performances).

COMMANDES CALIBRATION & PARIS (Phase 12+) :

  calibrate [--days N] [--sync-config]
      Recalibre le modèle de probabilités (température/Platt).
      Exporte les artefacts dans calibration/
      --sync-config: Met à jour config/pro_betting.yaml avec les artefacts
      Exemple : python cli.py calibrate --days 30 --sync-config

  health
      Affiche le health-check du système de calibration.
      Montre T, α, calibrator depuis health.json (source de vérité).
      Vérifie si YAML est synchronisé avec les artefacts.
      Exemple : python cli.py health

  pick [--date YYYY-MM-DD|TODAY]
      Génère les pronostics du jour (JSON + portfolio YAML).
      Utilise T/α depuis artefacts (log: "Calibration: T=5.0, α=0.0, source=artefacts")
      Exemple : python cli.py pick --date TODAY

  exotic --race <race_id>
      Génère les tickets exotiques (Trio/Quinté) pour une course.
      Produit 3 packs: SÛR, ÉQUILIBRÉ, RISQUÉ (N=20000 simulations).
      Exemple : python cli.py exotic --race R00123

  report [--days N]
      Publie un rapport de métriques (turnover, EV, drawdown).
      Analyse les performances par décile de cotes.
      Exemple : python cli.py report --days 7

  menu
      Lance le menu interactif avec toutes les options disponibles.

WORKFLOW TYPIQUE :

  1. Initialisation :
     python cli.py init-db

  2. Import données IFCE :
     python cli.py import-ifce --path ./fichier-des-equides.csv

  3. Scraping PMU :
     python cli.py fetch --date 2025-11-01

  4. Recalcul agrégats :
     python cli.py recompute

  5. Calibration (hebdomadaire) :
     python cli.py calibrate --days 30

  6. Génération quotidienne :
     python cli.py pick --date TODAY

  7. Vérification santé :
     python cli.py health

════════════════════════════════════════════════════════════════
    """)


# ============================================================================
# MENU INTERACTIF
# ============================================================================

def clear_screen():
    """Efface l'écran"""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_menu_header():
    """Affiche l'en-tête du menu"""
    clear_screen()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "🏇  HORSE3 - SYSTÈME DE GESTION HIPPIQUE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()


def print_main_menu():
    """Affiche le menu principal"""
    print_menu_header()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                            MENU PRINCIPAL                                   │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│                                                                             │")
    print("│  📊 1. SCRAPING & ENRICHISSEMENT                                            │")
    print("│  🔍 2. VÉRIFICATION & AUDIT BDD                                             │")
    print("│  🧹 3. NETTOYAGE & MAINTENANCE                                              │")
    print("│  🏗️  4. CONFIGURATION & INITIALISATION                                      │")
    print("│  📈 5. STATISTIQUES & RAPPORTS                                              │")
    print("│  🛠️  6. OUTILS AVANCÉS                                                       │")
    print("│                                                                             │")
    print("│  ❌ 0. QUITTER                                                              │")
    print("│                                                                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print()


def print_scraping_menu():
    """Menu scraping et enrichissement"""
    print_menu_header()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                     📊 SCRAPING & ENRICHISSEMENT                            │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│                                                                             │")
    print("│  1. Scraper une date spécifique                                            │")
    print("│  2. Scraper aujourd'hui                                                     │")
    print("│  3. Rattrapage historique (backfill)                                       │")
    print("│  4. Scraper avec orchestrateur (tous les scrapers)                         │")
    print("│  5. Normaliser les noms de races                                           │")
    print("│  6. Recalculer gains et records                                            │")
    print("│  7. Clean & Rescrape (nettoyer puis re-scraper)                            │")
    print("│                                                                             │")
    print("│  0. Retour au menu principal                                               │")
    print("│                                                                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print()


def print_audit_menu():
    """Menu vérification et audit"""
    print_menu_header()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                     🔍 VÉRIFICATION & AUDIT BDD                             │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│                                                                             │")
    print("│  1. Audit rapide des colonnes                                              │")
    print("│  2. Audit complet des colonnes BDD                                         │")
    print("│  3. Audit des scrapers (qualité code)                                      │")
    print("│  4. Vérifier les doublons                                                  │")
    print("│  5. Vérifier intégrité de la BDD                                           │")
    print("│  6. Analyse des champs manquants                                           │")
    print("│  7. Rapport sur les index DB                                               │")
    print("│                                                                             │")
    print("│  0. Retour au menu principal                                               │")
    print("│                                                                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print()


def print_cleaning_menu():
    """Menu nettoyage et maintenance"""
    print_menu_header()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                     🧹 NETTOYAGE & MAINTENANCE                              │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│                                                                             │")
    print("│  1. Nettoyer les orphelins                                                 │")
    print("│  2. Corriger les doublons                                                  │")
    print("│  3. Générer script de nettoyage SQL                                        │")
    print("│  4. Optimiser la base de données (VACUUM)                                  │")
    print("│  5. Créer backup de la BDD                                                 │")
    print("│  6. Réinitialiser et repeupler                                             │")
    print("│                                                                             │")
    print("│  0. Retour au menu principal                                               │")
    print("│                                                                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print()


def print_config_menu():
    """Menu configuration"""
    print_menu_header()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                  🏗️  CONFIGURATION & INITIALISATION                          │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│                                                                             │")
    print("│  1. Initialiser la base de données                                         │")
    print("│  2. Initialiser avec FTS (fuzzy matching)                                  │")
    print("│  3. Créer les tables                                                       │")
    print("│  4. Créer les index                                                        │")
    print("│  5. Importer CSV IFCE                                                      │")
    print("│  6. Migrer ancienne structure vers nouvelle                                │")
    print("│  7. Valider l'installation                                                 │")
    print("│                                                                             │")
    print("│  0. Retour au menu principal                                               │")
    print("│                                                                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print()


def print_stats_menu():
    """Menu statistiques"""
    print_menu_header()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                     📈 STATISTIQUES & RAPPORTS                              │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│                                                                             │")
    print("│  1. Statistiques générales                                                 │")
    print("│  2. Rapport de matching PMU ↔ IFCE                                         │")
    print("│  3. Résultats de scraping                                                  │")
    print("│  4. Analyses avancées                                                      │")
    print("│  5. Exemples de requêtes                                                   │")
    print("│  6. Visualiser les performances d'un cheval                                │")
    print("│  🎨 7. Générer graphiques d'analyses (10 graphiques)                        │")
    print("│                                                                             │")
    print("│  0. Retour au menu principal                                               │")
    print("│                                                                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print()


def print_tools_menu():
    """Menu outils avancés"""
    print_menu_header()
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                       🛠️  OUTILS AVANCÉS                                     │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│                                                                             │")
    print("│  1. Tests unitaires (enrichment)                                           │")
    print("│  2. Tests de validation (scrapers)                                         │")
    print("│  3. Test multi-thread                                                      │")
    print("│  4. Test optimisations                                                     │")
    print("│  5. Démonstration enrichissement                                           │")
    print("│  6. Voir les logs de scraping                                              │")
    print("│  7. Pool de connexions DB                                                  │")
    print("│                                                                             │")
    print("│  0. Retour au menu principal                                               │")
    print("│                                                                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print()


def run_script(script_name, *args):
    """Execute un script Python"""
    try:
        if not Path(script_name).exists():
            print(f"\n❌ Erreur : Le script '{script_name}' n'existe pas.")
            input("\nAppuyez sur Entrée pour continuer...")
            return
        
        print(f"\n🚀 Exécution de {script_name}...\n")
        print("─" * 80)
        
        cmd = [sys.executable, script_name] + list(args)
        result = subprocess.run(cmd, capture_output=False)
        
        print("─" * 80)
        if result.returncode == 0:
            print(f"\n✅ {script_name} terminé avec succès!")
        else:
            print(f"\n⚠️  {script_name} terminé avec code {result.returncode}")
        
        input("\nAppuyez sur Entrée pour continuer...")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution : {e}")
        input("\nAppuyez sur Entrée pour continuer...")


def get_input(prompt, default=None):
    """Récupère une saisie utilisateur avec valeur par défaut"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


# ============================================================================
# GESTIONNAIRES DE MENUS
# ============================================================================

def handle_scraping_menu():
    """Gère le menu scraping"""
    while True:
        print_scraping_menu()
        choice = input("Votre choix: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            target_date = get_input("Date à scraper (YYYY-MM-DD)", date.today().isoformat())
            cmd_fetch(['--date', target_date])
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "2":
            cmd_fetch([])
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "3":
            days = get_input("Nombre de jours à rattraper", "30")
            cmd_backfill(['--days-back', days])
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "4":
            target_date = get_input("Date pour orchestrateur (YYYY-MM-DD)", date.today().isoformat())
            run_script("orchestrator_scrapers.py", "--date", target_date)
        elif choice == "5":
            run_script("normalize_races.py")
        elif choice == "6":
            cmd_recompute([])
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "7":
            run_script("clean_and_rescrape.py")
        else:
            print("❌ Choix invalide")
            input("\nAppuyez sur Entrée pour continuer...")


def handle_audit_menu():
    """Gère le menu audit"""
    while True:
        print_audit_menu()
        choice = input("Votre choix: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            run_script("audit_rapide.py")
        elif choice == "2":
            run_script("audit_colonnes_bdd.py")
        elif choice == "3":
            run_script("audit_scrapers.py")
        elif choice == "4":
            run_script("fix_doublons.py")
        elif choice == "5":
            run_script("verify_database.py")
        elif choice == "6":
            print("\n📄 Fichier d'analyse disponible : ANALYSE_CHAMPS_MANQUANTS.md")
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "7":
            if Path("scrapers/index_analyzer.py").exists():
                run_script("scrapers/index_analyzer.py")
            else:
                print("\n❌ Script index_analyzer.py non trouvé")
                input("\nAppuyez sur Entrée pour continuer...")
        else:
            print("❌ Choix invalide")
            input("\nAppuyez sur Entrée pour continuer...")


def handle_cleaning_menu():
    """Gère le menu nettoyage"""
    while True:
        print_cleaning_menu()
        choice = input("Votre choix: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            run_script("clean_orphans.py")
        elif choice == "2":
            run_script("fix_doublons.py")
        elif choice == "3":
            run_script("generer_script_nettoyage.py")
        elif choice == "4":
            print("\n🔧 Optimisation de la base de données (VACUUM)...")
            try:
                conn = get_connection()
                conn.set_isolation_level(0) # AUTOCOMMIT pour VACUUM
                cur = conn.cursor()
                cur.execute("VACUUM")
                conn.close()
                print("✅ Base de données optimisée!")
            except Exception as e:
                print(f"❌ Erreur : {e}")
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "5":
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            print(f"\n💾 Création du backup PostgreSQL : {backup_name}...")
            try:
                # Pour PostgreSQL, on utilise pg_dump au lieu de copier un fichier
                import subprocess
                result = subprocess.run(
                    ["pg_dump", "-h", "localhost", "-U", "postgres", "-d", "horse3", "-f", backup_name],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"✅ Backup créé : {backup_name}")
                else:
                    print(f"❌ Erreur pg_dump : {result.stderr}")
            except Exception as e:
                print(f"❌ Erreur : {e}")
                print("   Note: Assurez-vous que pg_dump est installé et accessible.")
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "6":
            confirm = input("\n⚠️  ATTENTION : Cela va réinitialiser la BDD. Confirmer ? (oui/non): ")
            if confirm.lower() == "oui":
                run_script("reset_and_populate.py")
            else:
                print("❌ Opération annulée")
                input("\nAppuyez sur Entrée pour continuer...")
        else:
            print("❌ Choix invalide")
            input("\nAppuyez sur Entrée pour continuer...")


def handle_config_menu():
    """Gère le menu configuration"""
    while True:
        print_config_menu()
        choice = input("Votre choix: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            cmd_init_db([])
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "2":
            cmd_init_db(['--enable-fts'])
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "3":
            run_script("create_tables.py")
        elif choice == "4":
            if Path("add_indexes.sql").exists():
                print("\n🔧 Exécution du script d'index SQL...")
                try:
                    conn = get_connection()
                    cur = conn.cursor()
                    with open("add_indexes.sql", 'r') as f:
                        cur.execute(f.read())
                    conn.commit()
                    conn.close()
                    print("✅ Index créés!")
                except Exception as e:
                    print(f"❌ Erreur : {e}")
            else:
                print("❌ Fichier add_indexes.sql non trouvé")
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "5":
            csv_path = get_input("Chemin du fichier CSV IFCE", "./fichier-des-equides.csv")
            cmd_import_ifce(['--path', csv_path])
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "6":
            cmd_migrate([])
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "7":
            if Path("validate_installation.py").exists():
                run_script("validate_installation.py")
            else:
                print("❌ Script validate_installation.py non trouvé")
                input("\nAppuyez sur Entrée pour continuer...")
        else:
            print("❌ Choix invalide")
            input("\nAppuyez sur Entrée pour continuer...")


def handle_stats_menu():
    """Gère le menu statistiques"""
    while True:
        print_stats_menu()
        choice = input("Votre choix: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            run_script("stats.py")
        elif choice == "2":
            cmd_match_report([])
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "3":
            run_script("scraper_results.py")
        elif choice == "4":
            run_script("analyses_avancees.py")
        elif choice == "5":
            run_script("exemples_requetes.py")
        elif choice == "6":
            horse_name = get_input("Nom du cheval")
            if horse_name:
                # Créer un script temporaire pour visualiser les perfs
                print(f"\n🔍 Recherche des performances de '{horse_name}'...")
                try:
                    conn = get_connection()
                    cur = conn.cursor()
                    
                    # Recherche dans chevaux (table principale)
                    cur.execute("""
                        SELECT COUNT(*) FROM chevaux 
                        WHERE nom LIKE %s
                    """, (f"%{horse_name.lower()}%",))
                    count = cur.fetchone()[0]
                    
                    if count > 0:
                        cur.execute("""
                            SELECT nom, nombre_courses_total, nombre_victoires_total, dernier_resultat
                            FROM chevaux
                            WHERE nom LIKE %s
                            LIMIT 20
                        """, (f"%{horse_name.lower()}%",))
                        
                        print(f"\n📊 {count} cheval/chevaux trouvé(s)\n")
                        print(f"{'Nom':<25} {'Courses':<10} {'Victoires':<10} {'Musique'}")
                        print("─" * 80)
                        
                        for row in cur.fetchall():
                            nom, nbc, nbv, mus = row
                            print(f"{nom:<25} {nbc or 0:<10} {nbv or 0:<10} {mus or 'N/A'}")
                            
                        # Afficher détails courses si un seul cheval trouvé
                        if count == 1:
                            print("\nDétails des courses (cheval_courses_seen):")
                            cur.execute("""
                                SELECT course_nom, hippodrome_nom, annee, place_finale, allocation_totale
                                FROM cheval_courses_seen
                                WHERE nom_norm LIKE %s
                                ORDER BY annee DESC
                                LIMIT 10
                            """, (f"%{horse_name.lower()}%",))
                            for row in cur.fetchall():
                                print(f"  - {row[2]} {row[1]}: {row[0]} (Place: {row[3]})")
                    else:
                        print(f"\n❌ Aucune course trouvée pour '{horse_name}'")
                    
                    conn.close()
                except Exception as e:
                    print(f"❌ Erreur : {e}")
                input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "7":
            print("\n🎨 Génération des graphiques d'analyses...")
            print("   Cela peut prendre quelques instants...\n")
            run_script("graphiques_analyses.py")
        else:
            print("❌ Choix invalide")
            input("\nAppuyez sur Entrée pour continuer...")


def handle_tools_menu():
    """Gère le menu outils avancés"""
    while True:
        print_tools_menu()
        choice = input("Votre choix: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            run_script("test_enrichment.py")
        elif choice == "2":
            if Path("scrapers/tests_validation.py").exists():
                run_script("scrapers/tests_validation.py")
            else:
                print("❌ Script tests_validation.py non trouvé")
                input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "3":
            run_script("test_multi_thread.py")
        elif choice == "4":
            run_script("test_optimisations.py")
        elif choice == "5":
            run_script("demo_enrichment.py")
        elif choice == "6":
            if Path("scraping_log.txt").exists():
                print("\n📄 Affichage des dernières lignes du log...\n")
                os.system("tail -50 scraping_log.txt")
            else:
                print("❌ Fichier scraping_log.txt non trouvé")
            input("\nAppuyez sur Entrée pour continuer...")
        elif choice == "7":
            run_script("db_pool.py")
        else:
            print("❌ Choix invalide")
            input("\nAppuyez sur Entrée pour continuer...")


def interactive_menu():
    """Lance le menu interactif principal"""
    while True:
        print_main_menu()
        choice = input("Votre choix: ").strip()
        
        if choice == "0":
            print("\n👋 Au revoir!\n")
            break
        elif choice == "1":
            handle_scraping_menu()
        elif choice == "2":
            handle_audit_menu()
        elif choice == "3":
            handle_cleaning_menu()
        elif choice == "4":
            handle_config_menu()
        elif choice == "5":
            handle_stats_menu()
        elif choice == "6":
            handle_tools_menu()
        else:
            print("\n❌ Choix invalide")
            input("\nAppuyez sur Entrée pour continuer...")


# ============================================================================
# COMMANDES EXISTANTES (conservées pour compatibilité)
# ============================================================================


def main():
    # Si aucun argument ou --help, afficher l'aide
    if len(sys.argv) < 2:
        # Lancer le menu interactif par défaut
        interactive_menu()
        return
    
    if sys.argv[1] in ('--help', '-h', 'help'):
        print_help()
        return
    
    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    commands = {
        'init-db': cmd_init_db,
        'import-ifce': cmd_import_ifce,
        'fetch': cmd_fetch,
        'backfill': cmd_backfill,
        'recompute': cmd_recompute,
        'match-report': cmd_match_report,
        'migrate': cmd_migrate,
        'menu': lambda args: interactive_menu(),  # Menu interactif
        # Nouvelles commandes CLI Phase 12+
        'calibrate': cmd_calibrate,
        'health': cmd_health,
        'pick': cmd_pick,
        'exotic': cmd_exotic,
        'report': cmd_report,
    }
    
    if cmd in commands:
        commands[cmd](args)
    else:
        print(f"❌ Commande inconnue : {cmd}")
        print("Utilisez 'python cli.py --help' pour l'aide")
        print("Ou lancez 'python cli.py menu' pour le menu interactif")


if __name__ == '__main__':
    main()
