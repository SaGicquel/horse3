#!/usr/bin/env python3
"""
Script de test complet des optimisations PostgreSQL
Vérifie que tous les modules fonctionnent correctement
"""

import time
from db_pool import initialize_pool, get_cursor, close_pool, get_pool_stats
from db_batch import batch_insert_chevaux

def print_header(title):
    """Affiche un en-tête stylisé"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_connection_pool():
    """Test 1: Connection Pooling"""
    print_header("TEST 1: Connection Pooling")
    
    print("🔌 Initialisation du pool...")
    initialize_pool(minconn=3, maxconn=10)
    
    stats = get_pool_stats()
    print(f"✅ Pool créé: {stats['min_connections']}-{stats['max_connections']} connexions")
    
    # Test concurrent
    print("\n🧵 Test de 5 requêtes simultanées...")
    import concurrent.futures
    
    def query(n):
        with get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chevaux")
            return cur.fetchone()[0]
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(query, range(5)))
    elapsed = time.time() - start
    
    print(f"✅ 5 requêtes en parallèle: {elapsed:.3f}s")
    print(f"   Résultats: {results[0]} chevaux (x5 threads)")

def test_indexes():
    """Test 2: Index Performance"""
    print_header("TEST 2: Performance des Index")
    
    # Test sans EXPLAIN (requête rapide)
    print("🎯 Requête avec index (id_cheval + annee)...")
    start = time.time()
    with get_cursor() as cur:
        cur.execute("""
            SELECT COUNT(*), SUM(CASE WHEN is_win=1 THEN 1 ELSE 0 END)
            FROM cheval_courses_seen
            WHERE id_cheval = 1000000 AND annee = 2024
        """)
        result = cur.fetchone()
    elapsed = time.time() - start
    
    print(f"✅ Résultat en {elapsed*1000:.1f}ms")
    print(f"   {result[0]} courses, {result[1]} victoires")

def test_materialized_views():
    """Test 3: Vues Matérialisées"""
    print_header("TEST 3: Vues Matérialisées")
    
    print("📊 Lecture de mv_chevaux_stats...")
    start = time.time()
    with get_cursor() as cur:
        cur.execute("""
            SELECT id_cheval, nom, nb_courses, nb_victoires
            FROM mv_chevaux_stats
            WHERE nb_victoires > 5
            ORDER BY nb_victoires DESC
            LIMIT 5
        """)
        top_winners = cur.fetchall()
    elapsed = time.time() - start
    
    print(f"✅ Top 5 chevaux en {elapsed*1000:.1f}ms:")
    for i, row in enumerate(top_winners, 1):
        print(f"   {i}. {row['nom']}: {row['nb_victoires']} victoires sur {row['nb_courses']} courses")

def test_normalized_tables():
    """Test 4: Tables Normalisées"""
    print_header("TEST 4: Tables Normalisées")
    
    print("🔗 Statistiques des tables de référence...")
    with get_cursor() as cur:
        # Compter les entrées
        cur.execute("SELECT COUNT(*) FROM entraineurs")
        nb_entraineurs = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM drivers")
        nb_drivers = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM proprietaires")
        nb_proprietaires = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM hippodromes")
        nb_hippodromes = cur.fetchone()[0]
        
        # Top entraineurs
        cur.execute("""
            SELECT nom, nb_victoires 
            FROM entraineurs 
            ORDER BY nb_victoires DESC 
            LIMIT 3
        """)
        top_entraineurs = cur.fetchall()
    
    print(f"✅ Tables de référence:")
    print(f"   • {nb_entraineurs} entraineurs")
    print(f"   • {nb_drivers} drivers")
    print(f"   • {nb_proprietaires} proprietaires")
    print(f"   • {nb_hippodromes} hippodromes")
    
    print(f"\n🏆 Top 3 entraineurs:")
    for i, row in enumerate(top_entraineurs, 1):
        print(f"   {i}. {row['nom']}: {row['nb_victoires']} victoires")

def test_batch_inserts():
    """Test 5: Batch Inserts"""
    print_header("TEST 5: Batch Inserts")
    
    print("📦 Test d'insertion batch (100 chevaux)...")
    
    # Créer des données de test
    test_data = [
        {
            'id_cheval': 8888000 + i,
            'nom': f'BATCH_TEST_{i}',
            'sexe': 'M' if i % 2 == 0 else 'F',
            'race': 'PS',
            'nombre_courses_total': 0,
            'nombre_victoires_total': 0
        }
        for i in range(100)
    ]
    
    # Batch insert
    start = time.time()
    count = batch_insert_chevaux(test_data, page_size=100)
    elapsed = time.time() - start
    
    print(f"✅ {count} chevaux insérés en {elapsed:.3f}s ({count/elapsed:.0f} ops/s)")
    
    # Nettoyage
    with get_cursor() as cur:
        cur.execute("DELETE FROM chevaux WHERE id_cheval >= 8888000")
        deleted = cur.rowcount
        print(f"🧹 {deleted} chevaux de test supprimés")

def test_enriched_view():
    """Test 6: Vue Enrichie"""
    print_header("TEST 6: Vue Enrichie (v_courses_enrichies)")
    
    print("🔍 Requête sur la vue enrichie...")
    start = time.time()
    with get_cursor() as cur:
        cur.execute("""
            SELECT 
                nom_cheval,
                nom_entraineur,
                nom_driver,
                hippodrome_nom,
                race_key
            FROM v_courses_enrichies
            WHERE is_win = 1
            LIMIT 5
        """)
        top_winners = cur.fetchall()
    elapsed = time.time() - start
    
    print(f"✅ Top 5 gagnants en {elapsed*1000:.1f}ms:")
    for i, row in enumerate(top_winners, 1):
        if row['nom_cheval']:
            print(f"   {i}. {row['nom_cheval']}")
            print(f"      Entraineur: {row['nom_entraineur'] or 'N/A'}, Driver: {row['nom_driver'] or 'N/A'}")
            print(f"      Hippodrome: {row['hippodrome_nom'] or 'N/A'}")

def show_summary():
    """Affiche un résumé de la base"""
    print_header("RÉSUMÉ DE LA BASE")
    
    with get_cursor() as cur:
        # Statistiques principales
        cur.execute("SELECT COUNT(*) FROM chevaux")
        nb_chevaux = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
        nb_courses = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM mv_chevaux_stats")
        nb_stats = cur.fetchone()[0]
        
        # Calcul de l'espace disque
        cur.execute("""
            SELECT 
                pg_size_pretty(pg_total_relation_size('chevaux')) as chevaux_size,
                pg_size_pretty(pg_total_relation_size('cheval_courses_seen')) as courses_size,
                pg_size_pretty(pg_database_size('pmubdd')) as db_size
        """)
        sizes = cur.fetchone()
        
        # Nombre d'index
        cur.execute("""
            SELECT COUNT(*) 
            FROM pg_indexes 
            WHERE schemaname = 'public'
        """)
        nb_indexes = cur.fetchone()[0]
    
    print("📊 Données:")
    print(f"   • Chevaux: {nb_chevaux:,}")
    print(f"   • Courses: {nb_courses:,}")
    print(f"   • Stats (vue mat.): {nb_stats:,}")
    
    print("\n💾 Espace disque:")
    print(f"   • Table chevaux: {sizes['chevaux_size']}")
    print(f"   • Table courses: {sizes['courses_size']}")
    print(f"   • Base complète: {sizes['db_size']}")
    
    print(f"\n🎯 Index: {nb_indexes} créés")

def main():
    """Lance tous les tests"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🏇 TEST COMPLET DES OPTIMISATIONS POSTGRESQL            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    start_total = time.time()
    
    try:
        # Tests individuels
        test_connection_pool()
        test_indexes()
        test_materialized_views()
        test_normalized_tables()
        test_batch_inserts()
        test_enriched_view()
        
        # Résumé
        show_summary()
        
        # Temps total
        elapsed_total = time.time() - start_total
        
        print_header("RÉSULTAT FINAL")
        print(f"✅ Tous les tests ont réussi !")
        print(f"⏱️  Temps total: {elapsed_total:.2f}s")
        print(f"\n📚 Voir OPTIMISATIONS_IMPLEMENTEES.md pour les détails")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        close_pool()
        print("\n🔒 Pool de connexions fermé")

if __name__ == "__main__":
    main()
