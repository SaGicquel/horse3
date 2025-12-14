#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de démonstration du système d'enrichissement
Montre toutes les fonctionnalités principales
"""

import sqlite3
from enrichment import (
    normalize_name,
    extract_birth_year,
    parse_time_str,
    compute_reduction_km,
    HorseMatcher,
    run_migrations
)


def demo_normalisation():
    """Démo normalisation de noms"""
    print("=" * 70)
    print("1️⃣  DÉMONSTRATION : Normalisation de noms")
    print("=" * 70)
    
    test_cases = [
        "Élégant D'Avril (FR)",
        "BLACK SAXON",
        "L'As-du-Jour",
        "  Saint  Martin  ",
        "Champion Star (GB)",
    ]
    
    for name in test_cases:
        normalized = normalize_name(name)
        print(f"  {name:30} → {normalized}")
    
    print()


def demo_parsing_temps():
    """Démo parsing temps hippiques"""
    print("=" * 70)
    print("2️⃣  DÉMONSTRATION : Parsing temps hippiques")
    print("=" * 70)
    
    test_times = [
        "1'12\"8",
        "1'11\"",
        "68.7",
        "1:12.8",
        "2'05\"5",
    ]
    
    for time_str in test_times:
        seconds = parse_time_str(time_str)
        print(f"  {time_str:12} → {seconds} secondes")
    
    print()


def demo_calculs():
    """Démo calculs réductions kilométriques"""
    print("=" * 70)
    print("3️⃣  DÉMONSTRATION : Réductions kilométriques")
    print("=" * 70)
    
    tests = [
        (72.8, 2400, "Course rapide"),
        (90.0, 2700, "Course moyenne"),
        (65.5, 2100, "Course sprint"),
    ]
    
    for time_sec, distance_m, desc in tests:
        reduction = compute_reduction_km(time_sec, distance_m)
        print(f"  {desc:20} : {time_sec}s / {distance_m}m = {reduction:.2f} s/km")
    
    print()


def demo_matching():
    """Démo matching PMU ↔ IFCE"""
    print("=" * 70)
    print("4️⃣  DÉMONSTRATION : Matching PMU ↔ IFCE")
    print("=" * 70)
    
    # Créer une base temporaire avec données de test
    import tempfile
    import os
    
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    conn = sqlite3.connect(db_path)
    
    # Migrations
    print("  → Initialisation base de données...")
    run_migrations(conn, enable_fts=False)
    
    # Insérer des chevaux de test
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO ifce_horses (name, name_norm, sex, birth_year, country, breed)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [
        ('Black Saxon', 'BLACK SAXON', 'H', 2018, 'FR', 'TROTTEUR FRANÇAIS'),
        ('Élégant D\'Avril', 'ELEGANT DAVRIL', 'M', 2019, 'FR', 'TROTTEUR FRANÇAIS'),
        ('Champion Star', 'CHAMPION STAR', 'M', 2020, 'GB', 'PUR SANG'),
    ])
    conn.commit()
    
    # Tester le matching
    matcher = HorseMatcher(conn, enable_fuzzy=False)
    
    test_cases = [
        ("BLACK SAXON", 2018, 'H', 'FR', "Match strict (A)"),
        ("Élégant d'Avril (FR)", 2019, None, None, "Match avec accents"),
        ("CHAMPION STAR", 2020, None, 'GB', "Match avec pays"),
        ("UNKNOWN HORSE", 2020, None, None, "Non trouvé"),
    ]
    
    for name, year, sex, country, desc in test_cases:
        result = matcher.match_horse(name, year, sex, country)
        status = "✅" if result.ifce_horse_key else "❌"
        print(f"  {status} {desc:25} : stage={result.match_stage}, confidence={result.confidence:.2f}")
    
    # Nettoyage
    conn.close()
    os.close(db_fd)
    os.unlink(db_path)
    
    print()


def demo_requetes_sql():
    """Démo requêtes SQL utiles"""
    print("=" * 70)
    print("5️⃣  EXEMPLES DE REQUÊTES SQL")
    print("=" * 70)
    
    queries = [
        ("Top 10 gains 2025", """
            SELECT 
                i.name,
                s.gains_annuels_eur,
                s.nb_courses,
                s.nb_victoires
            FROM horse_year_stats s
            JOIN ifce_horses i ON s.horse_key = i.horse_key
            WHERE s.year = 2025
            ORDER BY s.gains_annuels_eur DESC
            LIMIT 10;
        """),
        
        ("Records d'un cheval", """
            SELECT 
                i.name,
                t.record_attele_sec,
                t.record_attele_date,
                t.record_attele_venue,
                t.record_monte_sec,
                t.record_monte_date
            FROM horse_totals t
            JOIN ifce_horses i ON t.horse_key = i.horse_key
            WHERE i.name_norm = 'BLACK SAXON';
        """),
        
        ("Performances récentes", """
            SELECT 
                race_date,
                venue,
                discipline,
                finish_position,
                allocation_eur,
                reduction_km_sec
            FROM v_performances_enriched
            WHERE horse_name = 'BLACK SAXON'
            ORDER BY race_date DESC
            LIMIT 5;
        """),
        
        ("Taux de matching", """
            SELECT 
                match_stage,
                COUNT(*) as nb_chevaux,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM pmu_horses), 1) as pourcentage
            FROM pmu_horses
            GROUP BY match_stage
            ORDER BY nb_chevaux DESC;
        """),
    ]
    
    for title, query in queries:
        print(f"\n  📊 {title}")
        print("  " + "-" * 66)
        print("  " + query.strip().replace("\n", "\n  "))
    
    print()


def main():
    """Lance toutes les démonstrations"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  🏇 SYSTÈME D'ENRICHISSEMENT HORSE3 - DÉMONSTRATION".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    demo_normalisation()
    demo_parsing_temps()
    demo_calculs()
    demo_matching()
    demo_requetes_sql()
    
    print("=" * 70)
    print("✅ DÉMONSTRATION TERMINÉE")
    print("=" * 70)
    print()
    print("Pour utiliser le système :")
    print("  1. python cli.py init-db")
    print("  2. python cli.py import-ifce --path ./fichier-des-equides.csv")
    print("  3. python cli.py fetch")
    print("  4. python cli.py recompute")
    print("  5. python cli.py match-report")
    print()
    print("Documentation complète : README_ENRICHISSEMENT.md")
    print()


if __name__ == '__main__':
    main()
