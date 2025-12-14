#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark du scraper PMU - Compare les performances avec/sans multi-threading

Usage:
    python benchmark_scraper.py [date]
    
    Si aucune date n'est fournie, utilise aujourd'hui.
"""

import sys
import time
from datetime import date
import sqlite3
from scraper_pmu_simple import run, DB_PATH, discover_reunions, discover_courses

def count_tasks(date_iso):
    """Compte le nombre de tâches (courses) à scraper"""
    reunions = discover_reunions(date_iso)
    total = 0
    for r in reunions:
        courses = discover_courses(date_iso, r)
        total += len(courses)
    return len(reunions), total

def get_db_stats():
    """Récupère les stats de la DB"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_courses = cur.fetchone()[0]
    con.close()
    return nb_chevaux, nb_courses

def main():
    if len(sys.argv) > 1:
        date_iso = sys.argv[1]
    else:
        date_iso = date.today().isoformat()
    
    print("=" * 80)
    print(f"🏇 BENCHMARK SCRAPER PMU - {date_iso}")
    print("=" * 80)
    print()
    
    # Compter les tâches
    print("🔍 Analyse du programme...")
    nb_reunions, nb_courses = count_tasks(date_iso)
    
    if nb_courses == 0:
        print(f"❌ Aucune course trouvée pour {date_iso}")
        return
    
    print(f"   • {nb_reunions} réunions")
    print(f"   • {nb_courses} courses à scraper")
    print()
    
    # Stats initiales
    nb_chevaux_init, nb_courses_seen_init = get_db_stats()
    
    # Test 1: Version séquentielle
    print("=" * 80)
    print("🐢 TEST 1: VERSION SÉQUENTIELLE (classique)")
    print("=" * 80)
    print()
    
    start = time.time()
    run(date_iso, recalc_after=False, use_threading=False)
    time_sequential = time.time() - start
    
    nb_chevaux_seq, nb_courses_seen_seq = get_db_stats()
    chevaux_added = nb_chevaux_seq - nb_chevaux_init
    courses_added = nb_courses_seen_seq - nb_courses_seen_init
    
    print(f"\n⏱️  Temps séquentiel: {time_sequential:.2f}s")
    print(f"📊 Données ajoutées: {chevaux_added} chevaux, {courses_added} courses")
    print()
    
    # Pause entre les tests
    print("⏸️  Pause de 3 secondes avant le test multi-threadé...")
    time.sleep(3)
    
    # Test 2: Version multi-threadée
    print()
    print("=" * 80)
    print("🚀 TEST 2: VERSION MULTI-THREADÉE (optimisée)")
    print("=" * 80)
    print()
    
    start = time.time()
    run(date_iso, recalc_after=False, use_threading=True)
    time_threaded = time.time() - start
    
    nb_chevaux_thread, nb_courses_seen_thread = get_db_stats()
    
    print(f"\n⏱️  Temps multi-threadé: {time_threaded:.2f}s")
    print()
    
    # Comparaison
    print("=" * 80)
    print("📊 RÉSULTATS")
    print("=" * 80)
    print()
    print(f"⏱️  TEMPS D'EXÉCUTION:")
    print(f"   • Séquentiel:     {time_sequential:>8.2f}s")
    print(f"   • Multi-threadé:  {time_threaded:>8.2f}s")
    print()
    
    speedup = time_sequential / time_threaded
    gain_pct = ((time_sequential - time_threaded) / time_sequential) * 100
    
    print(f"🚀 GAIN DE PERFORMANCE:")
    print(f"   • Accélération:   {speedup:>8.2f}x plus rapide")
    print(f"   • Gain de temps:  {gain_pct:>8.1f}%")
    print(f"   • Temps économisé: {time_sequential - time_threaded:>7.1f}s")
    print()
    
    # Estimation pour des volumes plus importants
    if nb_courses < 50:
        estimated_100_seq = (time_sequential / nb_courses) * 100
        estimated_100_thread = (time_threaded / nb_courses) * 100
        print(f"📈 PROJECTION POUR 100 COURSES:")
        print(f"   • Séquentiel:     ~{estimated_100_seq:.0f}s ({estimated_100_seq/60:.1f} minutes)")
        print(f"   • Multi-threadé:  ~{estimated_100_thread:.0f}s ({estimated_100_thread/60:.1f} minutes)")
        print(f"   • Gain estimé:    ~{estimated_100_seq - estimated_100_thread:.0f}s")
        print()
    
    print("=" * 80)
    print("💡 RECOMMANDATION:")
    print("=" * 80)
    
    if speedup >= 4:
        print("✅ Le multi-threading apporte un gain TRÈS SIGNIFICATIF!")
        print("   → Utilisez TOUJOURS le mode multi-threadé en production")
    elif speedup >= 2:
        print("✅ Le multi-threading apporte un gain SIGNIFICATIF!")
        print("   → Recommandé pour la production")
    elif speedup >= 1.5:
        print("⚠️  Le multi-threading apporte un gain modéré")
        print("   → Peut-être utile pour de gros volumes")
    else:
        print("❌ Le multi-threading n'apporte pas de gain significatif")
        print("   → La version séquentielle suffit")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
