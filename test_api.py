"""
🏇 Script de Test - API Prédiction
===================================

Teste l'API avec des données réelles pour valider le déploiement.

Usage:
    python test_api.py --url http://localhost:8000 --verbose
"""

import sys
import time
import json
import argparse
from typing import Dict, Any
from datetime import datetime

import requests
import pandas as pd


def test_health(base_url: str) -> bool:
    """Teste l'endpoint /health."""
    print("\n" + "=" * 80)
    print("🏥 TEST HEALTHCHECK")
    print("=" * 80)

    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()

        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"   Modèle chargé: {data['model_loaded']}")
        print(f"   Version: {data['model_version']}")
        print(f"   Uptime: {data['uptime_seconds']:.1f}s")
        print(f"   Prédictions totales: {data['total_predictions']}")

        return data["status"] == "healthy" and data["model_loaded"]

    except Exception as e:
        print(f"❌ Erreur healthcheck: {e}")
        return False


def test_metrics(base_url: str) -> bool:
    """Teste l'endpoint /metrics."""
    print("\n" + "=" * 80)
    print("📊 TEST MÉTRIQUES PROMETHEUS")
    print("=" * 80)

    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        response.raise_for_status()

        metrics = response.text
        print(f"✅ Métriques récupérées ({len(metrics)} caractères)")

        # Parser les métriques
        for line in metrics.split("\n"):
            if line and not line.startswith("#"):
                print(f"   {line}")

        return True

    except Exception as e:
        print(f"❌ Erreur métriques: {e}")
        return False


def create_sample_course() -> Dict[str, Any]:
    """Crée une course d'exemple pour le test."""
    return {
        "course_id": "TEST_2025-11-13_VINCENNES_R1C3",
        "date_course": "2025-11-13",
        "hippodrome": "VINCENNES",
        "distance": 2700,
        "type_piste": "Plat",
        "partants": [
            {
                "cheval_id": "CHEVAL_001",
                "numero_partant": 1,
                "forme_5c": 0.8,
                "forme_10c": 0.75,
                "nb_courses_12m": 12,
                "nb_victoires_12m": 3,
                "nb_places_12m": 7,
                "recence": 15.0,
                "regularite": 0.7,
                "aptitude_distance": 0.85,
                "aptitude_piste": 0.8,
                "aptitude_hippodrome": 0.9,
                "taux_victoires_jockey": 0.25,
                "taux_places_jockey": 0.45,
                "taux_victoires_entraineur": 0.22,
                "taux_places_entraineur": 0.42,
                "synergie_jockey_cheval": 0.6,
                "synergie_entraineur_cheval": 0.65,
                "distance_norm": 0.5,
                "niveau_moyen_concurrent": 0.7,
                "nb_partants": 10,
                "cote_turfbzh": 3.5,
                "rang_cote_turfbzh": 1,
                "cote_sp": 3.2,
                "rang_cote_sp": 1,
                "prediction_ia_gagnant": 0.28,
                "elo_cheval": 1850.0,
                "ecart_cote_ia": 0.03,
            },
            {
                "cheval_id": "CHEVAL_002",
                "numero_partant": 2,
                "forme_5c": 0.6,
                "forme_10c": 0.65,
                "nb_courses_12m": 10,
                "nb_victoires_12m": 1,
                "nb_places_12m": 4,
                "recence": 25.0,
                "regularite": 0.5,
                "aptitude_distance": 0.7,
                "aptitude_piste": 0.65,
                "aptitude_hippodrome": 0.75,
                "taux_victoires_jockey": 0.18,
                "taux_places_jockey": 0.38,
                "taux_victoires_entraineur": 0.15,
                "taux_places_entraineur": 0.35,
                "synergie_jockey_cheval": 0.4,
                "synergie_entraineur_cheval": 0.45,
                "distance_norm": 0.5,
                "niveau_moyen_concurrent": 0.7,
                "nb_partants": 10,
                "cote_turfbzh": 8.5,
                "rang_cote_turfbzh": 3,
                "cote_sp": 9.0,
                "rang_cote_sp": 3,
                "prediction_ia_gagnant": 0.11,
                "elo_cheval": 1650.0,
                "ecart_cote_ia": -0.02,
            },
            {
                "cheval_id": "CHEVAL_003",
                "numero_partant": 3,
                "forme_5c": 0.7,
                "forme_10c": 0.68,
                "nb_courses_12m": 15,
                "nb_victoires_12m": 2,
                "nb_places_12m": 6,
                "recence": 20.0,
                "regularite": 0.6,
                "aptitude_distance": 0.75,
                "aptitude_piste": 0.72,
                "aptitude_hippodrome": 0.8,
                "taux_victoires_jockey": 0.20,
                "taux_places_jockey": 0.40,
                "taux_victoires_entraineur": 0.19,
                "taux_places_entraineur": 0.39,
                "synergie_jockey_cheval": 0.5,
                "synergie_entraineur_cheval": 0.55,
                "distance_norm": 0.5,
                "niveau_moyen_concurrent": 0.7,
                "nb_partants": 10,
                "cote_turfbzh": 5.5,
                "rang_cote_turfbzh": 2,
                "cote_sp": 5.8,
                "rang_cote_sp": 2,
                "prediction_ia_gagnant": 0.18,
                "elo_cheval": 1750.0,
                "ecart_cote_ia": 0.01,
            },
        ],
    }


def test_prediction(base_url: str, verbose: bool = False) -> bool:
    """Teste l'endpoint /predict."""
    print("\n" + "=" * 80)
    print("🔮 TEST PRÉDICTION")
    print("=" * 80)

    try:
        # Créer requête
        course = create_sample_course()

        if verbose:
            print("\n📥 Requête:")
            print(f"   Course: {course['course_id']}")
            print(f"   Hippodrome: {course['hippodrome']}")
            print(f"   Partants: {len(course['partants'])}")

        # Envoyer requête
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json=course,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        latency_ms = (time.time() - start_time) * 1000

        response.raise_for_status()
        data = response.json()

        # Afficher résultats
        print(f"\n✅ Prédiction réussie en {latency_ms:.1f}ms (API: {data['latence_ms']:.1f}ms)")
        print(f"   Modèle: {data['model_version']}")
        print(f"   Timestamp: {data['timestamp']}")

        print("\n🏆 TOP 3 PRÉDICTIONS:")
        print("   " + "-" * 76)
        print(f"   {'Rang':<6} {'N°':<4} {'Cheval':<15} {'Probabilité':<12} {'Confiance':<12}")
        print("   " + "-" * 76)

        for pred in data["top_3"]:
            print(
                f"   {pred['rang_prediction']:<6} "
                f"{pred['numero_partant']:<4} "
                f"{pred['cheval_id']:<15} "
                f"{pred['probabilite_victoire']:.4f} ({pred['probabilite_victoire']*100:5.2f}%)  "
                f"{pred['confiance']:<12}"
            )

        if verbose:
            print("\n📊 TOUTES LES PRÉDICTIONS:")
            print("   " + "-" * 76)
            for pred in data["toutes_predictions"]:
                print(
                    f"   {pred['rang_prediction']:<6} "
                    f"{pred['numero_partant']:<4} "
                    f"{pred['cheval_id']:<15} "
                    f"{pred['probabilite_victoire']:.4f} ({pred['probabilite_victoire']*100:5.2f}%)  "
                    f"{pred['confiance']:<12}"
                )

        # Vérifications
        assert len(data["top_3"]) == 3, "Top 3 doit contenir 3 chevaux"
        assert len(data["toutes_predictions"]) == len(
            course["partants"]
        ), "Toutes prédictions incohérentes"
        assert all(
            0 <= p["probabilite_victoire"] <= 1 for p in data["toutes_predictions"]
        ), "Probabilités invalides"

        # Vérifier que le top 3 est trié
        probas = [p["probabilite_victoire"] for p in data["top_3"]]
        assert probas == sorted(probas, reverse=True), "Top 3 mal trié"

        print("\n✅ Toutes les vérifications passées")

        return True

    except Exception as e:
        print(f"❌ Erreur prédiction: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False


def test_load(base_url: str, num_requests: int = 100) -> bool:
    """Teste les performances sous charge."""
    print("\n" + "=" * 80)
    print(f"⚡ TEST DE CHARGE ({num_requests} requêtes)")
    print("=" * 80)

    try:
        course = create_sample_course()
        latencies = []
        errors = 0

        print(f"\n🔄 Envoi de {num_requests} requêtes...")
        start_time = time.time()

        for i in range(num_requests):
            try:
                req_start = time.time()
                response = requests.post(f"{base_url}/predict", json=course, timeout=10)
                req_latency = (time.time() - req_start) * 1000

                if response.status_code == 200:
                    latencies.append(req_latency)
                else:
                    errors += 1

                # Afficher progression tous les 20%
                if (i + 1) % (num_requests // 5) == 0:
                    print(f"   Progression: {i+1}/{num_requests} ({100*(i+1)/num_requests:.0f}%)")

            except Exception as e:
                errors += 1

        total_time = time.time() - start_time

        # Statistiques
        print("\n📊 RÉSULTATS:")
        print(f"   Requêtes totales: {num_requests}")
        print(f"   Succès: {len(latencies)} ({100*len(latencies)/num_requests:.1f}%)")
        print(f"   Erreurs: {errors} ({100*errors/num_requests:.1f}%)")
        print(f"   Temps total: {total_time:.2f}s")
        print(f"   Throughput: {num_requests/total_time:.1f} req/s")

        if latencies:
            latencies_sorted = sorted(latencies)
            print("\n⏱️  LATENCES:")
            print(f"   Moyenne: {sum(latencies)/len(latencies):.1f}ms")
            print(f"   Médiane (P50): {latencies_sorted[len(latencies)//2]:.1f}ms")
            print(f"   P95: {latencies_sorted[int(len(latencies)*0.95)]:.1f}ms")
            print(f"   P99: {latencies_sorted[int(len(latencies)*0.99)]:.1f}ms")
            print(f"   Min: {min(latencies):.1f}ms")
            print(f"   Max: {max(latencies):.1f}ms")

        # Critères de succès
        success_rate = len(latencies) / num_requests
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        if success_rate >= 0.99 and avg_latency < 500:
            print(
                f"\n✅ Test de charge réussi (success rate: {100*success_rate:.1f}%, latence: {avg_latency:.1f}ms)"
            )
            return True
        else:
            print(
                f"\n⚠️  Performances insuffisantes (success rate: {100*success_rate:.1f}%, latence: {avg_latency:.1f}ms)"
            )
            return False

    except Exception as e:
        print(f"❌ Erreur test de charge: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test de l'API de prédiction")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="URL de base de l'API (défaut: http://localhost:8000)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Mode verbeux (affiche tous les détails)"
    )
    parser.add_argument(
        "--load-test", action="store_true", help="Effectuer un test de charge (100 requêtes)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Nombre de requêtes pour le test de charge (défaut: 100)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🏇 TEST API PRÉDICTION COURSES HIPPIQUES")
    print("=" * 80)
    print(f"URL: {args.url}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Exécuter tests
    results = {}

    results["health"] = test_health(args.url)
    time.sleep(1)

    results["metrics"] = test_metrics(args.url)
    time.sleep(1)

    results["prediction"] = test_prediction(args.url, verbose=args.verbose)
    time.sleep(1)

    if args.load_test:
        results["load"] = test_load(args.url, num_requests=args.num_requests)

    # Résumé final
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 80)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name.upper():<20} : {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 TOUS LES TESTS RÉUSSIS !")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
