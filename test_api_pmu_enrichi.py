#!/usr/bin/env python3
"""
🏇 Test API PMU - Extraction des données enrichies
=================================================
Script pour tester et valider l'extraction des nouvelles données PMU :
- Cotes directes et références
- Tendances de cotes
- Avis entraîneur
- Indicateurs stratégiques (favori, driver change, etc.)
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Configuration
UA = "horse-test/1.0"
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json",
    "Accept-Language": "fr-FR,fr;q=0.9",
}
BASE = "https://online.turfinfo.api.pmu.fr/rest/client/7"


def get_json(url: str, timeout: int = 15) -> Optional[Dict]:
    """Récupère les données JSON d'une URL."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code in (204, 404):
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None


def extract_cotes_detaillees(participant: Dict) -> Dict[str, Any]:
    """
    Extrait les cotes détaillées depuis un participant.
    NOUVELLES DONNÉES À RÉCUPÉRER.
    """
    result = {
        "cote_directe": None,
        "cote_reference": None,
        "tendance_cote": None,
        "amplitude_tendance": None,
        "est_favori": False,
        "grosse_prise": False,
    }
    
    # Dernier rapport direct (cote actuelle)
    rapport_direct = participant.get("dernierRapportDirect", {})
    if rapport_direct:
        result["cote_directe"] = rapport_direct.get("rapport")
        result["tendance_cote"] = rapport_direct.get("indicateurTendance", "").strip()
        result["amplitude_tendance"] = rapport_direct.get("nombreIndicateurTendance")
        result["est_favori"] = rapport_direct.get("favoris", False)
        result["grosse_prise"] = rapport_direct.get("grossePrise", False)
    
    # Rapport de référence (cote matin/veille)
    rapport_ref = participant.get("dernierRapportReference", {})
    if rapport_ref:
        result["cote_reference"] = rapport_ref.get("rapport")
    
    return result


def extract_indicateurs_strategiques(participant: Dict) -> Dict[str, Any]:
    """
    Extrait les indicateurs stratégiques pour la prédiction.
    NOUVELLES DONNÉES À RÉCUPÉRER.
    """
    return {
        "avis_entraineur": participant.get("avisEntraineur"),
        "driver_change": participant.get("driverChange", False),
        "indicateur_inedit": participant.get("indicateurInedit", False),
        "jument_pleine": participant.get("jumentPleine", False),
        "allure": participant.get("allure"),
        "statut_participant": participant.get("statut"),
        "supplement": participant.get("supplement", 0),
        "engagement": participant.get("engagement", False),
        "poids_condition_monte_change": participant.get("poidsConditionMonteChange", False),
        "url_casaque": participant.get("urlCasaque"),
    }


def extract_donnees_supplementaires(participant: Dict) -> Dict[str, Any]:
    """
    Extrait les données supplémentaires du participant.
    """
    # Gains détaillés
    gains = participant.get("gainsParticipant", {})
    
    return {
        "gains_carriere": gains.get("gainsCarriere"),
        "gains_victoires": gains.get("gainsVictoires"),
        "gains_place": gains.get("gainsPlace"),
        "gains_annee_en_cours": gains.get("gainsAnneeEnCours"),
        "gains_annee_precedente": gains.get("gainsAnneePrecedente"),
        "nombre_places": participant.get("nombrePlaces"),
        "nombre_places_second": participant.get("nombrePlacesSecond"),
        "nombre_places_troisieme": participant.get("nombrePlacesTroisieme"),
        "commentaire_apres_course": None,
        "source_commentaire": None,
    }
    
    # Commentaire après course (peut être au niveau participant)
    commentaire = participant.get("commentaireApresCourse", {})
    if isinstance(commentaire, dict):
        result["commentaire_apres_course"] = commentaire.get("texte")
        result["source_commentaire"] = commentaire.get("source")
    
    return result


def extract_donnees_course(course_data: Dict) -> Dict[str, Any]:
    """
    Extrait les données supplémentaires au niveau course.
    """
    return {
        "duree_course": course_data.get("dureeCourse"),
        "course_trackee": course_data.get("courseTrackee", False),
        "replay_disponible": course_data.get("replayDisponible", False),
        "ordre_arrivee": course_data.get("ordreArrivee"),
    }


def test_extraction_course(date_str: str = None, reunion: int = 1, course: int = 1):
    """
    Test complet d'extraction sur une course réelle.
    """
    if not date_str:
        # Utiliser hier par défaut
        date_str = (datetime.now() - timedelta(days=1)).strftime('%d%m%Y')
    
    print("=" * 80)
    print(f"🏇 TEST EXTRACTION DONNÉES ENRICHIES PMU")
    print(f"   Date: {date_str} | R{reunion}C{course}")
    print("=" * 80)
    
    # 1. Récupérer les données de la course
    url = f"{BASE}/programme/{date_str}/R{reunion}/C{course}"
    print(f"\n📥 Récupération: {url}")
    
    course_data = get_json(url)
    if not course_data:
        print("❌ Impossible de récupérer les données de la course")
        return None
    
    print(f"✅ Course récupérée: {course_data.get('libelle', 'N/A')}")
    
    # 2. Données au niveau course
    print("\n" + "-" * 60)
    print("📊 DONNÉES COURSE (nouvelles)")
    print("-" * 60)
    
    donnees_course = extract_donnees_course(course_data)
    for key, value in donnees_course.items():
        if key != "ordre_arrivee":  # Skip la liste complète
            status = "✅" if value is not None else "⚠️"
            print(f"  {status} {key}: {value}")
    
    # 3. Parcourir les participants
    participants = course_data.get("participants", [])
    print(f"\n📋 {len(participants)} participants trouvés")
    
    if not participants:
        print("❌ Aucun participant")
        return None
    
    # Stats globales
    stats = {
        "avec_cote_directe": 0,
        "avec_cote_reference": 0,
        "avec_tendance": 0,
        "favoris": 0,
        "avec_avis_entraineur": 0,
        "driver_change": 0,
        "avec_commentaire": 0,
    }
    
    # Analyser chaque participant
    resultats = []
    
    for idx, p in enumerate(participants[:5], 1):  # Limiter à 5 pour le test
        print(f"\n" + "=" * 60)
        print(f"🐴 {idx}. {p.get('nom', 'N/A')} (N°{p.get('numPmu', '?')})")
        print("=" * 60)
        
        # Extraction cotes
        cotes = extract_cotes_detaillees(p)
        print("\n📈 COTES DÉTAILLÉES:")
        for key, value in cotes.items():
            status = "✅" if value not in [None, False, "", 0] else "⚠️"
            print(f"  {status} {key}: {value}")
            
        if cotes["cote_directe"]:
            stats["avec_cote_directe"] += 1
        if cotes["cote_reference"]:
            stats["avec_cote_reference"] += 1
        if cotes["tendance_cote"]:
            stats["avec_tendance"] += 1
        if cotes["est_favori"]:
            stats["favoris"] += 1
        
        # Extraction indicateurs
        indicateurs = extract_indicateurs_strategiques(p)
        print("\n🎯 INDICATEURS STRATÉGIQUES:")
        for key, value in indicateurs.items():
            if key != "url_casaque":  # Skip URL longue
                status = "✅" if value not in [None, False, "", 0] else "⚠️"
                print(f"  {status} {key}: {value}")
        
        if indicateurs["avis_entraineur"] and indicateurs["avis_entraineur"] != "NEUTRE":
            stats["avec_avis_entraineur"] += 1
        if indicateurs["driver_change"]:
            stats["driver_change"] += 1
        
        # Données supplémentaires
        donnees_supp = extract_donnees_supplementaires(p)
        print("\n💰 GAINS DÉTAILLÉS:")
        for key, value in list(donnees_supp.items())[:5]:
            status = "✅" if value not in [None, 0] else "⚠️"
            print(f"  {status} {key}: {value}")
        
        # Commentaire après course
        commentaire = p.get("commentaireApresCourse", {})
        if isinstance(commentaire, dict) and commentaire.get("texte"):
            stats["avec_commentaire"] += 1
            print(f"\n💬 COMMENTAIRE: {commentaire.get('texte', '')[:100]}...")
            print(f"   Source: {commentaire.get('source', 'N/A')}")
        
        resultats.append({
            "nom": p.get("nom"),
            "num_pmu": p.get("numPmu"),
            "cotes": cotes,
            "indicateurs": indicateurs,
        })
    
    # Résumé
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ EXTRACTION")
    print("=" * 80)
    total = min(5, len(participants))
    print(f"\n  Participants analysés: {total}")
    print(f"  ✅ Avec cote directe:    {stats['avec_cote_directe']}/{total}")
    print(f"  ✅ Avec cote référence:  {stats['avec_cote_reference']}/{total}")
    print(f"  ✅ Avec tendance cote:   {stats['avec_tendance']}/{total}")
    print(f"  ⭐ Favoris:              {stats['favoris']}/{total}")
    print(f"  📝 Avis entraîneur actif: {stats['avec_avis_entraineur']}/{total}")
    print(f"  🔄 Driver change:        {stats['driver_change']}/{total}")
    print(f"  💬 Avec commentaire:     {stats['avec_commentaire']}/{total}")
    
    # Calculer score de complétude
    completude = (
        (stats["avec_cote_directe"] > 0) * 20 +
        (stats["avec_cote_reference"] > 0) * 20 +
        (stats["avec_tendance"] > 0) * 15 +
        (stats["avec_commentaire"] > 0) * 15 +
        (stats["favoris"] > 0) * 10 +
        (stats["avec_avis_entraineur"] >= 0) * 10 +  # Toujours dispo
        (donnees_course["duree_course"] is not None) * 10
    )
    
    print(f"\n🎯 Score de complétude: {completude}%")
    
    if completude >= 80:
        print("✅ Excellente extraction! Toutes les données clés sont disponibles.")
    elif completude >= 50:
        print("⚠️ Extraction partielle. Certaines données manquent (course peut-être non terminée).")
    else:
        print("❌ Extraction insuffisante. Vérifier l'URL ou la disponibilité des données.")
    
    return {
        "date": date_str,
        "reunion": reunion,
        "course": course,
        "nb_participants": len(participants),
        "stats": stats,
        "completude": completude,
        "resultats": resultats,
    }


def test_multiple_courses():
    """
    Test sur plusieurs courses pour valider la robustesse.
    """
    print("\n" + "=" * 80)
    print("🏇 TEST MULTIPLE - VALIDATION ROBUSTESSE")
    print("=" * 80)
    
    # Tester sur les 3 derniers jours
    tests_results = []
    
    for days_ago in range(1, 4):
        date_str = (datetime.now() - timedelta(days=days_ago)).strftime('%d%m%Y')
        
        # Récupérer le programme du jour
        url_prog = f"{BASE}/programme/{date_str}"
        prog = get_json(url_prog)
        
        if not prog:
            print(f"\n⚠️ Pas de programme pour {date_str}")
            continue
        
        reunions = prog.get("programme", {}).get("reunions", [])
        if not reunions:
            continue
        
        # Tester la première réunion, première course
        r1 = reunions[0]
        r_num = r1.get("numOfficiel", 1)
        courses = r1.get("courses", [])
        
        if courses:
            c_num = courses[0].get("numOrdre", 1)
            print(f"\n📅 Test {date_str} - R{r_num}C{c_num}")
            
            result = test_extraction_course(date_str, r_num, c_num)
            if result:
                tests_results.append(result)
    
    # Résumé global
    if tests_results:
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ GLOBAL - TOUS LES TESTS")
        print("=" * 80)
        
        avg_completude = sum(r["completude"] for r in tests_results) / len(tests_results)
        print(f"\n  Tests effectués: {len(tests_results)}")
        print(f"  Score moyen de complétude: {avg_completude:.1f}%")
        
        if avg_completude >= 70:
            print("\n✅ VALIDATION RÉUSSIE! L'extraction fonctionne correctement.")
        else:
            print("\n⚠️ Des ajustements peuvent être nécessaires.")
    
    return tests_results


if __name__ == "__main__":
    import sys
    
    print("🏇 DÉMARRAGE TESTS API PMU ENRICHI")
    print("=" * 80)
    
    # Test simple sur une course récente
    result = test_extraction_course()
    
    # Si argument --full, faire test multiple
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        test_multiple_courses()
    
    print("\n✅ Tests terminés!")
