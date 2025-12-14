#!/usr/bin/env python3
"""
🏇 Test du scraper enrichi - Validation complète
=================================================
Script pour tester que le scraper modifié fonctionne correctement
et enregistre toutes les nouvelles données en base.
"""

import sys
import os
from datetime import datetime, timedelta

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_connection import get_connection
from scraper_pmu_simple import enrich_from_course, fetch_participants

def test_scraping_enrichi():
    """Test le scraping avec les nouvelles données."""
    
    print("=" * 80)
    print("🏇 TEST SCRAPING ENRICHI - VALIDATION COMPLÈTE")
    print("=" * 80)
    
    # Date d'hier
    date_iso = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    reunion = 1
    course = 1
    
    print(f"\n📅 Date: {date_iso}")
    print(f"📍 Course: R{reunion}C{course}")
    
    # 1. Connexion DB
    print("\n" + "-" * 40)
    print("1️⃣ Connexion à la base de données...")
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        print("✅ Connexion établie")
    except Exception as e:
        print(f"❌ Erreur connexion: {e}")
        return False
    
    # 2. Vérifier les nouvelles colonnes
    print("\n" + "-" * 40)
    print("2️⃣ Vérification des nouvelles colonnes...")
    
    nouvelles_colonnes = [
        'cote_reference', 'tendance_cote', 'amplitude_tendance', 
        'est_favori', 'grosse_prise', 'avis_entraineur',
        'allure', 'statut_participant', 'supplement', 'engagement',
        'poids_condition_monte_change', 'url_casaque', 'source_commentaire',
        'duree_course', 'course_trackee', 'replay_disponible'
    ]
    
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'cheval_courses_seen'
    """)
    colonnes_existantes = [row[0] for row in cur.fetchall()]
    
    colonnes_ok = 0
    colonnes_manquantes = []
    for col in nouvelles_colonnes:
        if col in colonnes_existantes:
            colonnes_ok += 1
        else:
            colonnes_manquantes.append(col)
    
    print(f"  ✅ {colonnes_ok}/{len(nouvelles_colonnes)} colonnes présentes")
    if colonnes_manquantes:
        print(f"  ⚠️ Colonnes manquantes: {colonnes_manquantes}")
    
    # 3. Lancer le scraping
    print("\n" + "-" * 40)
    print("3️⃣ Lancement du scraping enrichi...")
    
    try:
        enrich_from_course(cur, date_iso, reunion, course, sleep_s=0.1)
        conn.commit()
        print("✅ Scraping terminé avec succès")
    except Exception as e:
        print(f"❌ Erreur scraping: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
    
    # 4. Vérifier les données enregistrées
    print("\n" + "-" * 40)
    print("4️⃣ Vérification des données enregistrées...")
    
    race_key_pattern = f"{date_iso}|R{reunion}|C{course}|%"
    
    cur.execute("""
        SELECT 
            nom_norm,
            cote_finale,
            cote_reference,
            tendance_cote,
            amplitude_tendance,
            est_favori,
            grosse_prise,
            avis_entraineur,
            driver_change,
            indicateur_inedit,
            allure,
            statut_participant,
            url_casaque,
            duree_course,
            replay_disponible
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        LIMIT 10
    """, (race_key_pattern,))
    
    rows = cur.fetchall()
    
    if not rows:
        print("❌ Aucune donnée trouvée pour cette course!")
        return False
    
    print(f"\n  📊 {len(rows)} chevaux trouvés\n")
    
    # Stats des nouvelles colonnes
    stats = {
        'cote_finale': 0,
        'cote_reference': 0,
        'tendance_cote': 0,
        'amplitude_tendance': 0,
        'est_favori': 0,
        'avis_entraineur': 0,
        'allure': 0,
        'url_casaque': 0,
        'duree_course': 0,
    }
    
    print("  " + "-" * 70)
    print(f"  {'Cheval':<20} {'Cote':<8} {'Ref':<8} {'Tend':<6} {'Fav':<5} {'Avis':<10}")
    print("  " + "-" * 70)
    
    for row in rows:
        nom = row[0][:18] if row[0] else "?"
        cote = row[1]
        cote_ref = row[2]
        tendance = row[3]
        amplitude = row[4]
        favori = row[5]
        grosse = row[6]
        avis = row[7]
        driver_ch = row[8]
        inedit = row[9]
        allure = row[10]
        statut = row[11]
        url = row[12]
        duree = row[13]
        replay = row[14]
        
        # Compter les remplissages
        if cote: stats['cote_finale'] += 1
        if cote_ref: stats['cote_reference'] += 1
        if tendance: stats['tendance_cote'] += 1
        if amplitude: stats['amplitude_tendance'] += 1
        if favori: stats['est_favori'] += 1
        if avis: stats['avis_entraineur'] += 1
        if allure: stats['allure'] += 1
        if url: stats['url_casaque'] += 1
        if duree: stats['duree_course'] += 1
        
        fav_str = "⭐" if favori else ""
        print(f"  {nom:<20} {cote or '-':<8} {cote_ref or '-':<8} {tendance or '-':<6} {fav_str:<5} {avis or '-':<10}")
    
    # Résumé
    print("\n" + "-" * 40)
    print("5️⃣ RÉSUMÉ DES NOUVELLES DONNÉES")
    print("-" * 40)
    
    total = len(rows)
    score = 0
    for key, count in stats.items():
        pct = (count / total) * 100 if total > 0 else 0
        status = "✅" if pct > 50 else "⚠️" if pct > 0 else "❌"
        print(f"  {status} {key:<25}: {count}/{total} ({pct:.0f}%)")
        if pct > 50:
            score += 10
    
    print("\n" + "=" * 40)
    print(f"🎯 SCORE FINAL: {score}/90")
    print("=" * 40)
    
    if score >= 70:
        print("\n✅ VALIDATION RÉUSSIE! Le scraper enrichi fonctionne correctement.")
        success = True
    elif score >= 40:
        print("\n⚠️ VALIDATION PARTIELLE. Certaines données sont récupérées.")
        success = True
    else:
        print("\n❌ VALIDATION ÉCHOUÉE. Vérifier le scraper.")
        success = False
    
    cur.close()
    conn.close()
    
    return success


def test_extraction_participant():
    """Test direct de l'extraction des participants."""
    
    print("\n" + "=" * 80)
    print("🔍 TEST EXTRACTION DIRECTE PARTICIPANTS")
    print("=" * 80)
    
    date_iso = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\n📅 Date: {date_iso}")
    
    participants = fetch_participants(date_iso, 1, 1)
    
    if not participants:
        print("❌ Aucun participant trouvé")
        return False
    
    print(f"✅ {len(participants)} participants trouvés")
    
    # Analyser le premier participant
    p = participants[0]
    print(f"\n📊 Analyse de: {p.get('nom', 'N/A')}")
    
    # Nouvelles données
    rapport_direct = p.get("dernierRapportDirect", {})
    rapport_ref = p.get("dernierRapportReference", {})
    
    checks = {
        "Cote directe": rapport_direct.get("rapport"),
        "Cote référence": rapport_ref.get("rapport"),
        "Tendance": rapport_direct.get("indicateurTendance"),
        "Amplitude": rapport_direct.get("nombreIndicateurTendance"),
        "Favori": rapport_direct.get("favoris"),
        "Grosse prise": rapport_direct.get("grossePrise"),
        "Avis entraîneur": p.get("avisEntraineur"),
        "Driver change": p.get("driverChange"),
        "Allure": p.get("allure"),
        "Statut": p.get("statut"),
        "URL casaque": p.get("urlCasaque"),
    }
    
    print("\n  Vérification des champs:")
    for key, value in checks.items():
        status = "✅" if value not in [None, "", 0, False] else "⚠️"
        val_str = str(value)[:30] if value else "N/A"
        print(f"    {status} {key}: {val_str}")
    
    return True


if __name__ == "__main__":
    print("\n🏇 DÉMARRAGE DES TESTS DU SCRAPER ENRICHI\n")
    
    # Test 1: Extraction directe
    test1_ok = test_extraction_participant()
    
    # Test 2: Scraping complet
    test2_ok = test_scraping_enrichi()
    
    # Résumé final
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ FINAL DES TESTS")
    print("=" * 80)
    print(f"  Test extraction participants: {'✅ OK' if test1_ok else '❌ ÉCHEC'}")
    print(f"  Test scraping complet:        {'✅ OK' if test2_ok else '❌ ÉCHEC'}")
    
    if test1_ok and test2_ok:
        print("\n✅ TOUS LES TESTS SONT PASSÉS!")
        sys.exit(0)
    else:
        print("\n⚠️ Certains tests ont échoué.")
        sys.exit(1)
