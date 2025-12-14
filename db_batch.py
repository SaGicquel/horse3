"""
Module de batch inserts pour PostgreSQL
Utilise psycopg2.extras.execute_values() pour des inserts 5x plus rapides
"""

from psycopg2.extras import execute_values
from db_pool import get_cursor
import time
from typing import List, Dict, Any

def batch_insert_chevaux(chevaux_data: List[Dict[str, Any]], page_size=1000):
    """
    Insert en batch de chevaux
    
    Args:
        chevaux_data: Liste de dictionnaires avec les données des chevaux
        page_size: Nombre d'insertions par batch
    
    Returns:
        Nombre de chevaux insérés
    """
    if not chevaux_data:
        return 0
    
    start_time = time.time()
    
    # Préparer les tuples de données selon la vraie structure de la table
    values = []
    for cheval in chevaux_data:
        values.append((
            cheval.get('id_cheval'),
            cheval.get('nom'),
            cheval.get('sexe'),
            cheval.get('race'),
            cheval.get('robe'),
            cheval.get('pays_naissance'),
            cheval.get('eleveur'),
            cheval.get('nom_pere'),
            cheval.get('nom_mere'),
            cheval.get('proprietaire'),
            cheval.get('nombre_courses_total', 0),
            cheval.get('nombre_victoires_total', 0)
        ))
    
    # Query avec ON CONFLICT pour l'upsert
    query = """
        INSERT INTO chevaux (
            id_cheval, nom, sexe, race, robe, pays_naissance,
            eleveur, nom_pere, nom_mere, proprietaire,
            nombre_courses_total, nombre_victoires_total
        ) VALUES %s
        ON CONFLICT (id_cheval) DO UPDATE SET
            nombre_courses_total = GREATEST(chevaux.nombre_courses_total, EXCLUDED.nombre_courses_total),
            nombre_victoires_total = GREATEST(chevaux.nombre_victoires_total, EXCLUDED.nombre_victoires_total),
            race = COALESCE(EXCLUDED.race, chevaux.race),
            robe = COALESCE(EXCLUDED.robe, chevaux.robe),
            pays_naissance = COALESCE(EXCLUDED.pays_naissance, chevaux.pays_naissance),
            eleveur = COALESCE(EXCLUDED.eleveur, chevaux.eleveur),
            nom_pere = COALESCE(EXCLUDED.nom_pere, chevaux.nom_pere),
            nom_mere = COALESCE(EXCLUDED.nom_mere, chevaux.nom_mere),
            proprietaire = COALESCE(EXCLUDED.proprietaire, chevaux.proprietaire)
    """
    
    with get_cursor() as cur:
        execute_values(cur, query, values, page_size=page_size)
        inserted = cur.rowcount
    
    elapsed = time.time() - start_time
    print(f"✅ {inserted} chevaux insérés en {elapsed:.2f}s ({inserted/elapsed:.0f} chevaux/s)")
    
    return inserted

def batch_insert_courses(courses_data: List[Dict[str, Any]], page_size=500):
    """
    Insert en batch de courses
    
    Args:
        courses_data: Liste de dictionnaires avec les données des courses
        page_size: Nombre d'insertions par batch
    
    Returns:
        Nombre de courses insérées
    """
    if not courses_data:
        return 0
    
    start_time = time.time()
    
    # Préparer les tuples (56 colonnes !)
    values = []
    for course in courses_data:
        values.append((
            course.get('race_key'),
            course.get('date_course'),
            course.get('id_cheval'),
            course.get('numero'),
            course.get('nom'),
            course.get('sexe'),
            course.get('age'),
            course.get('poids'),
            course.get('distance_m'),
            course.get('musique'),
            course.get('cote'),
            course.get('place_finale'),
            course.get('temps_min'),
            course.get('gains_course'),
            course.get('driver_jockey'),
            course.get('entraineur'),
            course.get('proprietaire'),
            course.get('hippodrome_code'),
            course.get('hippodrome_nom'),
            course.get('nom_course'),
            course.get('discipline'),
            course.get('specialite'),
            course.get('conditions'),
            course.get('nombre_partants'),
            course.get('gains_carriere'),
            course.get('robe'),
            course.get('race'),
            course.get('pays_naissance'),
            course.get('eleveur'),
            course.get('nom_pere'),
            course.get('nom_mere'),
            course.get('deferre'),
            course.get('oeilleres'),
            course.get('nb_courses_annee'),
            course.get('nb_victoires_annee'),
            course.get('nb_places_annee'),
            course.get('annee'),
            course.get('mois'),
            course.get('is_win', False),
            course.get('is_place', False),
            course.get('non_partant', False),
            course.get('scratched', False),
            course.get('disqualifie', False),
            course.get('defaut_course', False),
            course.get('arrivee_ordre'),
            course.get('rapport_simple_gagnant'),
            course.get('rapport_simple_place'),
            course.get('ordre_arrivee_complet'),
            course.get('nombre_chevaux'),
            course.get('id_entraineur'),
            course.get('id_driver'),
            course.get('id_proprietaire'),
            course.get('nom_norm')
        ))
    
    query = """
        INSERT INTO cheval_courses_seen (
            race_key, date_course, id_cheval, numero, nom, sexe, age, poids,
            distance_m, musique, cote, place_finale, temps_min, gains_course,
            driver_jockey, entraineur, proprietaire, hippodrome_code,
            hippodrome_nom, nom_course, discipline, specialite, conditions,
            nombre_partants, gains_carriere, robe, race, pays_naissance,
            eleveur, nom_pere, nom_mere, deferre, oeilleres, nb_courses_annee,
            nb_victoires_annee, nb_places_annee, annee, mois, is_win, is_place,
            non_partant, scratched, disqualifie, defaut_course, arrivee_ordre,
            rapport_simple_gagnant, rapport_simple_place, ordre_arrivee_complet,
            nombre_chevaux, id_entraineur, id_driver, id_proprietaire, nom_norm
        ) VALUES %s
        ON CONFLICT (race_key, numero) DO UPDATE SET
            place_finale = EXCLUDED.place_finale,
            temps_min = EXCLUDED.temps_min,
            gains_course = EXCLUDED.gains_course,
            is_win = EXCLUDED.is_win,
            is_place = EXCLUDED.is_place,
            arrivee_ordre = EXCLUDED.arrivee_ordre,
            rapport_simple_gagnant = EXCLUDED.rapport_simple_gagnant,
            rapport_simple_place = EXCLUDED.rapport_simple_place,
            ordre_arrivee_complet = EXCLUDED.ordre_arrivee_complet
    """
    
    with get_cursor() as cur:
        execute_values(cur, query, values, page_size=page_size)
        inserted = cur.rowcount
    
    elapsed = time.time() - start_time
    print(f"✅ {inserted} courses insérées en {elapsed:.2f}s ({inserted/elapsed:.0f} courses/s)")
    
    return inserted

def benchmark_batch_vs_single():
    """Compare les performances batch vs single inserts"""
    print("\n🏁 Benchmark: Batch vs Single Inserts\n")
    
    # Données de test
    test_chevaux = [
        {
            'id_cheval': 9999000 + i,
            'nom': f'TEST_CHEVAL_{i}',
            'sexe': 'M',
            'race': 'PS',
            'nombre_courses_total': 0,
            'nombre_victoires_total': 0
        }
        for i in range(1000)
    ]
    
    # Test 1: Batch insert
    print("1️⃣ Batch insert (1000 chevaux)...")
    start = time.time()
    batch_insert_chevaux(test_chevaux, page_size=1000)
    batch_time = time.time() - start
    print(f"   ⏱️ Temps: {batch_time:.2f}s\n")
    
    # Test 2: Single inserts (simulation sur 100 chevaux pour pas attendre)
    print("2️⃣ Single inserts (100 chevaux)...")
    start = time.time()
    
    from db_pool import get_cursor
    with get_cursor() as cur:
        for cheval in test_chevaux[:100]:
            cur.execute("""
                INSERT INTO chevaux (id_cheval, nom, sexe, race)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id_cheval) DO NOTHING
            """, (
                cheval['id_cheval'],
                cheval['nom'],
                cheval['sexe'],
                cheval['race']
            ))
    
    single_time = time.time() - start
    print(f"   ⏱️ Temps: {single_time:.2f}s")
    
    # Extrapolation
    estimated_single = single_time * 10
    speedup = estimated_single / batch_time
    print(f"\n📊 Résultats:")
    print(f"   • Batch (1000): {batch_time:.2f}s")
    print(f"   • Single estimé (1000): {estimated_single:.2f}s")
    print(f"   • Gain de vitesse: {speedup:.1f}x")
    
    # Nettoyage
    with get_cursor() as cur:
        cur.execute("DELETE FROM chevaux WHERE id_cheval >= 9999000")
        deleted = cur.rowcount
        print(f"\n🧹 Nettoyage: {deleted} chevaux de test supprimés")

if __name__ == "__main__":
    from db_pool import initialize_pool, close_pool
    
    initialize_pool()
    benchmark_batch_vs_single()
    close_pool()
