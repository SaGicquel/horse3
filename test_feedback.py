"""
🧪 Tests pour API Feedback - Phase 8 Online Learning
====================================================

Suite de tests pour valider les endpoints de feedback:
- POST /feedback : Soumettre résultats
- GET /feedback/stats : Statistiques
- GET /feedback/model-performance : Performance

Usage:
    pytest test_feedback.py -v
    python test_feedback.py  # Run direct

Auteur: Phase 8 - Online Learning
Date: 2025-11-13
"""

import sys
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any


# Configuration
API_URL = "http://localhost:8000"
TIMEOUT = 30


class Colors:
    """Codes ANSI pour couleurs terminal."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_colored(text: str, color: str):
    """Affiche texte coloré."""
    print(f"{color}{text}{Colors.ENDC}")


def print_section(title: str):
    """Affiche un titre de section."""
    print("\n" + "=" * 80)
    print_colored(f"  {title}", Colors.HEADER + Colors.BOLD)
    print("=" * 80)


def test_api_health() -> bool:
    """Test 1: Vérifier que l'API est démarrée."""
    print_section("TEST 1: Health Check API")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_colored(f"✅ API healthy: {data}", Colors.OKGREEN)
            return True
        else:
            print_colored(f"❌ API non healthy: {response.status_code}", Colors.FAIL)
            return False
            
    except requests.exceptions.ConnectionError:
        print_colored(f"❌ API non accessible sur {API_URL}", Colors.FAIL)
        print_colored("   Démarrer l'API avec: ./manage_api.sh start", Colors.WARNING)
        return False
    except Exception as e:
        print_colored(f"❌ Erreur: {e}", Colors.FAIL)
        return False


def test_submit_feedback() -> bool:
    """Test 2: Soumettre un feedback de course."""
    print_section("TEST 2: POST /feedback - Soumettre Résultats")
    
    # Créer feedback de test
    feedback = {
        "course_id": "TEST_VINCENNES_2025-11-13_R1C3",
        "date_course": "2025-11-13",
        "hippodrome": "VINCENNES",
        "resultats": [
            {
                "cheval_id": "CHEVAL_001",
                "numero_partant": 1,
                "position_arrivee": 3
            },
            {
                "cheval_id": "CHEVAL_002",
                "numero_partant": 2,
                "position_arrivee": 1  # Gagnant
            },
            {
                "cheval_id": "CHEVAL_003",
                "numero_partant": 3,
                "position_arrivee": 2
            },
            {
                "cheval_id": "CHEVAL_004",
                "numero_partant": 4,
                "position_arrivee": 5
            },
            {
                "cheval_id": "CHEVAL_005",
                "numero_partant": 5,
                "position_arrivee": 4
            }
        ]
    }
    
    try:
        print(f"📤 Envoi feedback pour course: {feedback['course_id']}")
        print(f"   Hippodrome: {feedback['hippodrome']}")
        print(f"   Nombre de chevaux: {len(feedback['resultats'])}")
        
        response = requests.post(
            f"{API_URL}/feedback",
            json=feedback,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_colored("✅ Feedback enregistré avec succès!", Colors.OKGREEN)
            print(f"   Status: {data['status']}")
            print(f"   Message: {data['message']}")
            print(f"   Nb résultats: {data['nb_resultats']}")
            print(f"   Timestamp: {data['timestamp']}")
            return True
        else:
            print_colored(f"❌ Erreur HTTP {response.status_code}", Colors.FAIL)
            print(f"   Détail: {response.text}")
            return False
            
    except Exception as e:
        print_colored(f"❌ Erreur: {e}", Colors.FAIL)
        return False


def test_submit_invalid_feedback() -> bool:
    """Test 3: Soumettre un feedback invalide (doit échouer)."""
    print_section("TEST 3: POST /feedback - Validation Données Invalides")
    
    # Feedback avec position_arrivee invalide
    invalid_feedback = {
        "course_id": "TEST_INVALID",
        "date_course": "2025-11-13",
        "hippodrome": "VINCENNES",
        "resultats": [
            {
                "cheval_id": "CHEVAL_001",
                "numero_partant": 1,
                "position_arrivee": 0  # Invalide: doit être >= 1
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json=invalid_feedback,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 422 or response.status_code == 400:
            print_colored("✅ Validation fonctionne: feedback invalide rejeté", Colors.OKGREEN)
            print(f"   Code HTTP: {response.status_code}")
            return True
        else:
            print_colored(f"❌ Devrait rejeter feedback invalide, reçu: {response.status_code}", Colors.FAIL)
            return False
            
    except Exception as e:
        print_colored(f"❌ Erreur: {e}", Colors.FAIL)
        return False


def test_feedback_stats() -> bool:
    """Test 4: Récupérer statistiques feedback."""
    print_section("TEST 4: GET /feedback/stats - Statistiques")
    
    try:
        response = requests.get(f"{API_URL}/feedback/stats", timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_colored("✅ Statistiques récupérées!", Colors.OKGREEN)
            print(f"   Total courses: {data['total_courses']}")
            print(f"   Total prédictions: {data['total_predictions']}")
            print(f"   Période: {data['periode_debut']} → {data['periode_fin']}")
            print(f"   Courses 7j: {data['nb_courses_last_7d']}")
            print(f"   Courses 30j: {data['nb_courses_last_30d']}")
            print(f"   Taux collection: {data['taux_collection']:.1%}")
            return True
        else:
            print_colored(f"❌ Erreur HTTP {response.status_code}", Colors.FAIL)
            return False
            
    except Exception as e:
        print_colored(f"❌ Erreur: {e}", Colors.FAIL)
        return False


def test_model_performance() -> bool:
    """Test 5: Analyser performance du modèle."""
    print_section("TEST 5: GET /feedback/model-performance - Performance Modèle")
    
    try:
        # Tester avec 7 jours
        response = requests.get(
            f"{API_URL}/feedback/model-performance?days=7",
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_colored("✅ Performance calculée!", Colors.OKGREEN)
            print(f"   Période: {data['periode']}")
            print(f"   Nb courses: {data['nb_courses']}")
            print(f"   Nb prédictions: {data['nb_predictions']}")
            print(f"   Accuracy top 1: {data['accuracy_top1']:.1%} ({data['nb_correct_top1']} corrects)")
            print(f"   Accuracy top 3: {data['accuracy_top3']:.1%} ({data['nb_correct_top3']} corrects)")
            print(f"   Brier Score: {data['brier_score']:.4f}")
            print(f"   ECE: {data['ece']:.4f}")
            return True
        else:
            print_colored(f"❌ Erreur HTTP {response.status_code}", Colors.FAIL)
            return False
            
    except Exception as e:
        print_colored(f"❌ Erreur: {e}", Colors.FAIL)
        return False


def test_model_performance_invalid_days() -> bool:
    """Test 6: Valider paramètre days invalide."""
    print_section("TEST 6: GET /feedback/model-performance - Validation Days")
    
    try:
        # Tester avec days > 90 (invalide)
        response = requests.get(
            f"{API_URL}/feedback/model-performance?days=365",
            timeout=TIMEOUT
        )
        
        if response.status_code == 400:
            print_colored("✅ Validation days fonctionne: paramètre invalide rejeté", Colors.OKGREEN)
            print(f"   Message: {response.json().get('detail', 'N/A')}")
            return True
        else:
            print_colored(f"❌ Devrait rejeter days=365, reçu: {response.status_code}", Colors.FAIL)
            return False
            
    except Exception as e:
        print_colored(f"❌ Erreur: {e}", Colors.FAIL)
        return False


def test_multiple_feedbacks() -> bool:
    """Test 7: Soumettre plusieurs feedbacks en série."""
    print_section("TEST 7: Soumettre Plusieurs Feedbacks")
    
    feedbacks = []
    for i in range(3):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        feedbacks.append({
            "course_id": f"TEST_MULTIPLE_{i}_{date}",
            "date_course": date,
            "hippodrome": f"HIPPODROME_{i}",
            "resultats": [
                {
                    "cheval_id": f"CHEVAL_{i}_{j}",
                    "numero_partant": j + 1,
                    "position_arrivee": j + 1
                }
                for j in range(5)
            ]
        })
    
    success_count = 0
    
    try:
        for i, feedback in enumerate(feedbacks, 1):
            response = requests.post(
                f"{API_URL}/feedback",
                json=feedback,
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                success_count += 1
                print(f"   ✅ Feedback {i}/3 enregistré: {feedback['course_id']}")
            else:
                print(f"   ❌ Feedback {i}/3 échoué: {response.status_code}")
        
        if success_count == len(feedbacks):
            print_colored(f"✅ Tous les feedbacks enregistrés ({success_count}/{len(feedbacks)})", Colors.OKGREEN)
            return True
        else:
            print_colored(f"⚠️ Seulement {success_count}/{len(feedbacks)} réussis", Colors.WARNING)
            return False
            
    except Exception as e:
        print_colored(f"❌ Erreur: {e}", Colors.FAIL)
        return False


def run_all_tests() -> Dict[str, Any]:
    """Exécute tous les tests et retourne résumé."""
    print_colored("\n" + "╔" + "=" * 78 + "╗", Colors.BOLD)
    print_colored("║" + " " * 20 + "🧪 TESTS API FEEDBACK - PHASE 8" + " " * 27 + "║", Colors.BOLD)
    print_colored("╚" + "=" * 78 + "╝", Colors.BOLD)
    
    start_time = time.time()
    
    # Liste des tests
    tests = [
        ("Health Check", test_api_health),
        ("Submit Feedback", test_submit_feedback),
        ("Invalid Feedback", test_submit_invalid_feedback),
        ("Feedback Stats", test_feedback_stats),
        ("Model Performance", test_model_performance),
        ("Invalid Days", test_model_performance_invalid_days),
        ("Multiple Feedbacks", test_multiple_feedbacks)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_colored(f"\n❌ Erreur fatale dans {test_name}: {e}", Colors.FAIL)
            results[test_name] = False
        
        # Pause entre tests
        time.sleep(0.5)
    
    # Calcul résumé
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    duration = time.time() - start_time
    
    # Affichage résumé
    print_section("📊 RÉSUMÉ DES TESTS")
    
    print(f"\n{'Test':<30} {'Statut':<15}")
    print("-" * 50)
    
    for test_name, passed in results.items():
        status = f"{Colors.OKGREEN}✅ PASS{Colors.ENDC}" if passed else f"{Colors.FAIL}❌ FAIL{Colors.ENDC}"
        print(f"{test_name:<30} {status}")
    
    print("\n" + "=" * 80)
    print(f"Total tests: {total_tests}")
    print_colored(f"✅ Réussis: {passed_tests}", Colors.OKGREEN)
    
    if failed_tests > 0:
        print_colored(f"❌ Échoués: {failed_tests}", Colors.FAIL)
    
    print(f"⏱️  Durée: {duration:.2f}s")
    print(f"📈 Taux de réussite: {success_rate:.1f}%")
    
    # Message final
    if success_rate == 100:
        print_colored("\n🎉 TOUS LES TESTS PASSENT! API FEEDBACK OPÉRATIONNELLE 🎉\n", Colors.OKGREEN + Colors.BOLD)
    elif success_rate >= 80:
        print_colored("\n⚠️  La plupart des tests passent, quelques ajustements nécessaires\n", Colors.WARNING)
    else:
        print_colored("\n❌ TESTS ÉCHOUÉS - Vérifier logs API\n", Colors.FAIL)
    
    return {
        "total": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "success_rate": success_rate,
        "duration": duration,
        "details": results
    }


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit code basé sur résultats
    sys.exit(0 if results["failed"] == 0 else 1)
