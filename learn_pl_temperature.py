#!/usr/bin/env python3
"""
Apprentissage des Températures Plackett-Luce par Discipline
============================================================

Ce script analyse les courses historiques pour apprendre la température
optimale du modèle Plackett-Luce par discipline (plat, trot, obstacle).

La température contrôle la "concentration" de la distribution:
- T < 1: Favoris plus avantagés (distribution concentrée)
- T = 1: Standard Plackett-Luce
- T > 1: Plus d'incertitude (distribution uniforme)

Usage:
    python learn_pl_temperature.py --discipline plat --min-races 100
    python learn_pl_temperature.py --all --output results/temperatures.json

Auteur: Horse Racing AI System
Date: 2024-12
"""

import json
import numpy as np
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import argparse
from collections import defaultdict

# Imports locaux
try:
    from place_probability_estimator import (
        TemperatureLearner,
        CalibrationMetrics,
        PlackettLuceTemperatureSimulator,
    )
except ImportError:
    print("⚠️  Module place_probability_estimator.py requis")
    raise

try:
    from db_connection import get_db_path
except ImportError:

    def get_db_path():
        return Path(__file__).parent / "data" / "courses.db"


# =============================================================================
# EXTRACTION DES DONNÉES HISTORIQUES
# =============================================================================


def extract_historical_races(
    discipline: str = None,
    min_horses: int = 6,
    max_horses: int = 20,
    days_back: int = 365,
    limit: int = None,
) -> List[Dict]:
    """
    Extrait les courses historiques avec résultats pour l'apprentissage.

    Args:
        discipline: Filtrer par discipline ('plat', 'trot', 'obstacle')
        min_horses: Nombre minimum de partants
        max_horses: Nombre maximum de partants
        days_back: Nombre de jours à regarder en arrière
        limit: Limite le nombre de courses

    Returns:
        Liste de courses avec:
        - win_probs: probabilités de victoire (implicites des cotes)
        - actual_order: ordre d'arrivée réel
        - discipline: type de course
    """
    db_path = get_db_path()
    if not Path(db_path).exists():
        print(f"⚠️  Base de données non trouvée: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Date limite
    date_limit = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # Requête pour obtenir les courses avec résultats
    query = """
        SELECT DISTINCT
            c.race_id,
            c.date,
            c.discipline,
            c.distance,
            c.hippodrome
        FROM courses c
        WHERE c.date >= ?
        AND c.discipline IS NOT NULL
    """

    params = [date_limit]

    if discipline:
        query += " AND LOWER(c.discipline) LIKE ?"
        params.append(f"%{discipline.lower()}%")

    query += " ORDER BY c.date DESC"

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, params)
    races_meta = cursor.fetchall()

    print(f"📊 {len(races_meta)} courses trouvées")

    races = []
    for race_id, date, disc, distance, hippodrome in races_meta:
        # Récupérer les partants avec leurs cotes et positions d'arrivée
        cursor.execute(
            """
            SELECT
                p.numero,
                p.nom,
                p.cote_probable,
                p.position_arrivee
            FROM partants p
            WHERE p.race_id = ?
            AND p.cote_probable IS NOT NULL
            AND p.cote_probable > 1
            AND p.position_arrivee IS NOT NULL
            ORDER BY p.position_arrivee
        """,
            (race_id,),
        )

        partants = cursor.fetchall()

        if len(partants) < min_horses or len(partants) > max_horses:
            continue

        # Construire les probabilités implicites
        cotes = [p[2] for p in partants]
        positions = [p[3] for p in partants]

        # Vérifier que les positions sont cohérentes
        if not all(1 <= pos <= len(partants) for pos in positions):
            continue

        # Probabilités implicites (1/cote normalisées)
        raw_probs = [1 / c for c in cotes]
        total = sum(raw_probs)
        win_probs = [p / total for p in raw_probs]

        # Ordre d'arrivée (indices des chevaux triés par position)
        # positions[i] = position du cheval i
        # On veut: actual_order[k] = indice du cheval en k-ème position
        indexed = list(enumerate(positions))
        indexed.sort(key=lambda x: x[1])
        actual_order = [idx for idx, _ in indexed]

        # Normaliser la discipline
        disc_normalized = normalize_discipline(disc)

        races.append(
            {
                "race_id": race_id,
                "date": date,
                "discipline": disc_normalized,
                "distance": distance,
                "hippodrome": hippodrome,
                "n_horses": len(partants),
                "win_probs": win_probs,
                "actual_order": actual_order,
                "cotes": cotes,
            }
        )

    conn.close()

    print(f"✅ {len(races)} courses valides extraites")
    return races


def normalize_discipline(discipline: str) -> str:
    """Normalise le nom de discipline."""
    if not discipline:
        return "autre"

    disc = discipline.lower().strip()

    if any(x in disc for x in ["plat", "flat"]):
        return "plat"
    elif any(x in disc for x in ["trot", "attelé", "monte"]):
        return "trot"
    elif any(x in disc for x in ["obstacle", "haie", "steeple", "cross"]):
        return "obstacle"
    else:
        return "autre"


# =============================================================================
# APPRENTISSAGE DE LA TEMPÉRATURE
# =============================================================================


def learn_temperature_scipy(
    races: List[Dict], discipline: str, temp_range: Tuple[float, float] = (0.5, 2.0)
) -> Dict[str, Any]:
    """
    Apprend la température optimale via maximisation de la log-vraisemblance.

    Args:
        races: Courses historiques
        discipline: Discipline à filtrer
        temp_range: Intervalle de recherche (min, max)

    Returns:
        Dict avec température optimale et métriques
    """
    # Filtrer les courses de la discipline
    filtered = [r for r in races if r["discipline"] == discipline]

    if len(filtered) < 20:
        return {
            "discipline": discipline,
            "optimal_temperature": 1.0,
            "n_races": len(filtered),
            "status": "insufficient_data",
        }

    print(f"\n🎯 Apprentissage pour {discipline} ({len(filtered)} courses)")

    def compute_log_likelihood(temperature: float) -> float:
        """Calcule la log-vraisemblance négative pour une température."""
        total_ll = 0.0

        for race in filtered:
            win_probs = np.array(race["win_probs"])
            actual_order = race["actual_order"]

            # Appliquer la température
            log_probs = np.log(np.clip(win_probs, 1e-10, 1.0))
            scaled = softmax(log_probs / temperature)

            # Log-vraisemblance de l'ordre observé
            remaining = list(range(len(win_probs)))

            for pos, horse in enumerate(actual_order):
                if horse not in remaining:
                    continue

                p = scaled[remaining]
                p = p / p.sum()

                idx = remaining.index(horse)
                total_ll += np.log(max(p[idx], 1e-10))

                remaining.remove(horse)

                # On ne regarde que les 5 premières positions
                if pos >= 4:
                    break

        return -total_ll  # Négatif car on minimise

    # Optimisation
    print(f"   Recherche dans [{temp_range[0]}, {temp_range[1]}]...")

    result = minimize_scalar(
        compute_log_likelihood, bounds=temp_range, method="bounded", options={"xatol": 0.001}
    )

    optimal_temp = result.x
    optimal_ll = -result.fun

    # Baseline (T=1)
    baseline_ll = -compute_log_likelihood(1.0)

    # Calcul du Brier score pour validation
    brier_scores = []
    for race in filtered[:100]:  # Échantillon pour Brier
        win_probs = np.array(race["win_probs"])
        actual_order = race["actual_order"]

        sim = PlackettLuceTemperatureSimulator(win_probs, optimal_temp, seed=42)
        p_place, _ = sim.estimate_place_probs(1000, top_n=3)

        # Outcome réel: qui était placé
        placed = set(actual_order[:3])
        for i in range(len(win_probs)):
            actual = 1.0 if i in placed else 0.0
            brier_scores.append((p_place[i] - actual) ** 2)

    brier_place = np.mean(brier_scores) if brier_scores else 1.0

    return {
        "discipline": discipline,
        "optimal_temperature": round(optimal_temp, 4),
        "log_likelihood": round(optimal_ll, 2),
        "baseline_ll": round(baseline_ll, 2),
        "improvement_pct": round((optimal_ll - baseline_ll) / abs(baseline_ll) * 100, 2)
        if baseline_ll != 0
        else 0,
        "brier_place": round(brier_place, 4),
        "n_races": len(filtered),
        "status": "success",
    }


def cross_validate_temperature(
    races: List[Dict], discipline: str, n_folds: int = 5
) -> Dict[str, Any]:
    """
    Validation croisée pour la température.
    """
    filtered = [r for r in races if r["discipline"] == discipline]

    if len(filtered) < n_folds * 10:
        return {"error": "Pas assez de données pour validation croisée"}

    # Shuffle et split
    np.random.seed(42)
    np.random.shuffle(filtered)

    fold_size = len(filtered) // n_folds
    temperatures = []

    for fold in range(n_folds):
        # Test set
        test_start = fold * fold_size
        test_end = test_start + fold_size

        # Train set
        train = filtered[:test_start] + filtered[test_end:]

        # Apprendre
        result = learn_temperature_scipy(train, discipline)
        temperatures.append(result["optimal_temperature"])

    return {
        "discipline": discipline,
        "temperatures": temperatures,
        "mean_temperature": round(np.mean(temperatures), 4),
        "std_temperature": round(np.std(temperatures), 4),
        "cv_temperature": round(np.std(temperatures) / np.mean(temperatures), 4),
        "n_folds": n_folds,
    }


# =============================================================================
# RAPPORT FINAL
# =============================================================================


def generate_temperature_report(races: List[Dict], output_path: str = None) -> Dict[str, Any]:
    """
    Génère un rapport complet sur les températures par discipline.
    """
    print("\n" + "=" * 70)
    print("🌡️  APPRENTISSAGE DES TEMPÉRATURES PLACKETT-LUCE")
    print("=" * 70)

    # Statistiques globales
    disciplines = defaultdict(int)
    for race in races:
        disciplines[race["discipline"]] += 1

    print("\nDistribution des courses:")
    for disc, count in sorted(disciplines.items(), key=lambda x: -x[1]):
        print(f"  {disc}: {count} courses")

    # Apprendre pour chaque discipline
    results = {}

    for discipline in ["plat", "trot", "obstacle"]:
        if disciplines.get(discipline, 0) >= 20:
            result = learn_temperature_scipy(races, discipline)
            results[discipline] = result

            print(f"\n{discipline.upper()}:")
            print(f"  Température optimale: {result['optimal_temperature']}")
            print(f"  Log-likelihood: {result['log_likelihood']}")
            print(f"  Amélioration vs T=1: {result['improvement_pct']}%")
            print(f"  Brier (place): {result['brier_place']}")

    # Rapport final
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_races_total": len(races),
        "distribution": dict(disciplines),
        "temperatures": results,
        "recommendations": {
            "temperature_plat": results.get("plat", {}).get("optimal_temperature", 0.95),
            "temperature_trot": results.get("trot", {}).get("optimal_temperature", 1.05),
            "temperature_obstacle": results.get("obstacle", {}).get("optimal_temperature", 1.10),
        },
    }

    # Sauvegarder
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📁 Rapport sauvegardé: {output_path}")

    return report


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Apprentissage des températures Plackett-Luce par discipline"
    )
    parser.add_argument(
        "--discipline",
        "-d",
        choices=["plat", "trot", "obstacle", "all"],
        default="all",
        help="Discipline à analyser",
    )
    parser.add_argument("--days", "-n", type=int, default=365, help="Nombre de jours d'historique")
    parser.add_argument("--min-races", type=int, default=50, help="Nombre minimum de courses")
    parser.add_argument(
        "--output", "-o", default="results/pl_temperatures.json", help="Fichier de sortie"
    )
    parser.add_argument(
        "--cross-validate", action="store_true", help="Faire une validation croisée"
    )

    args = parser.parse_args()

    # Extraire les données
    print("📥 Extraction des données historiques...")
    races = extract_historical_races(
        discipline=None if args.discipline == "all" else args.discipline, days_back=args.days
    )

    if len(races) < args.min_races:
        print(f"❌ Pas assez de courses ({len(races)} < {args.min_races})")
        return

    if args.cross_validate:
        print("\n🔄 Validation croisée...")
        for disc in ["plat", "trot", "obstacle"]:
            cv_result = cross_validate_temperature(races, disc)
            if "error" not in cv_result:
                print(f"\n{disc.upper()}:")
                print(f"  Températures: {cv_result['temperatures']}")
                print(
                    f"  Moyenne: {cv_result['mean_temperature']} ± {cv_result['std_temperature']}"
                )

    # Générer le rapport
    report = generate_temperature_report(races, args.output)

    print("\n" + "=" * 70)
    print("📋 RECOMMANDATIONS POUR config/pro_betting.yaml:")
    print("=" * 70)
    print(
        """
place_estimators:
  temperature_plat: {temp_plat}
  temperature_trot: {temp_trot}
  temperature_obstacle: {temp_obstacle}
""".format(
            temp_plat=report["recommendations"]["temperature_plat"],
            temp_trot=report["recommendations"]["temperature_trot"],
            temp_obstacle=report["recommendations"]["temperature_obstacle"],
        )
    )


if __name__ == "__main__":
    main()
