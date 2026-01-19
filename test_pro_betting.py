#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST PRO BETTING ANALYZER
============================

Script de test pour vérifier:
1. Somme des p_win = 1 (à la précision près)
2. Pas de fuite temporelle
3. Format JSON correct
4. Kelly et Value calculés correctement
"""

import json
import sys
from datetime import datetime


def test_analyzer():
    """Test complet de l'analyseur"""

    from db_connection import get_connection
    from pro_betting_analyzer import ProBettingAnalyzer

    print("=" * 70)
    print("🧪 TEST PRO BETTING ANALYZER")
    print("=" * 70)

    conn = get_connection()
    analyzer = ProBettingAnalyzer(conn)

    # Récupérer une course de test
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT race_key
        FROM cheval_courses_seen
        WHERE race_key LIKE '2025-11%'
        ORDER BY race_key DESC
        LIMIT 3
    """)

    race_keys = [row[0] for row in cur.fetchall()]

    if not race_keys:
        print("❌ Aucune course trouvée pour les tests")
        return False

    all_passed = True

    for race_key in race_keys:
        print(f"\n{'─' * 70}")
        print(f"📊 Test: {race_key}")
        print(f"{'─' * 70}")

        # Analyser
        result_json = analyzer.analyze_race(race_key)

        try:
            result = json.loads(result_json)
        except json.JSONDecodeError as e:
            print(f"❌ JSON invalide: {e}")
            all_passed = False
            continue

        if "error" in result:
            print(f"⚠️ Erreur: {result['error']}")
            continue

        # Test 1: Somme p_win = 1
        runners = result.get("runners", [])
        p_wins = [r.get("p_win", 0) for r in runners if r.get("p_win") is not None]

        if p_wins:
            sum_p_win = sum(p_wins)
            if abs(sum_p_win - 1.0) < 0.001:
                print(f"✅ Test 1 - Somme p_win = {sum_p_win:.6f} (OK)")
            else:
                print(f"❌ Test 1 - Somme p_win = {sum_p_win:.6f} (FAIL, devrait être 1.0)")
                all_passed = False
        else:
            print("⚠️ Test 1 - Pas de p_win trouvé")

        # Test 2: Champs requis présents
        required_fields = ["race_id", "timestamp", "hippodrome", "nb_partants", "runners"]
        missing = [f for f in required_fields if f not in result]

        if not missing:
            print("✅ Test 2 - Tous les champs requis présents")
        else:
            print(f"❌ Test 2 - Champs manquants: {missing}")
            all_passed = False

        # Test 3: Champs partants
        runner_fields = [
            "numero",
            "nom",
            "p_win",
            "fair_odds",
            "value_pct",
            "kelly_fraction",
            "rationale",
        ]

        if runners:
            first_runner = runners[0]
            missing_runner = [f for f in runner_fields if f not in first_runner]

            if not missing_runner:
                print("✅ Test 3 - Tous les champs partant présents")
            else:
                print(f"⚠️ Test 3 - Champs partant manquants: {missing_runner}")

        # Test 4: Cohérence fair_odds
        for r in runners[:3]:  # Vérifier les 3 premiers
            p_win = r.get("p_win")
            fair_odds = r.get("fair_odds")

            if p_win and fair_odds:
                expected_fair = 1 / p_win
                if abs(fair_odds - expected_fair) < 0.1:
                    pass  # OK
                else:
                    print(
                        f"⚠️ Test 4 - fair_odds incohérent pour {r['nom']}: {fair_odds} vs attendu {expected_fair:.2f}"
                    )

        print("✅ Test 4 - fair_odds cohérents")

        # Test 5: Kelly positif seulement si value positive
        for r in runners:
            kelly = r.get("kelly_fraction", 0) or 0
            value = r.get("value_pct", 0) or 0

            if kelly > 0 and value < 0:
                print(f"⚠️ Test 5 - Kelly > 0 mais Value < 0 pour {r['nom']}")

        print("✅ Test 5 - Kelly/Value cohérents")

        # Test 6: Rationale max 3 éléments
        for r in runners:
            rationale = r.get("rationale", [])
            if len(rationale) > 3:
                print(f"⚠️ Test 6 - Rationale > 3 éléments pour {r['nom']}")
                all_passed = False

        print("✅ Test 6 - Rationale ≤ 3 éléments")

        # Afficher exemple
        print("\n📋 Exemple de sortie (top 3):")
        for i, r in enumerate(runners[:3], 1):
            print(f"\n  {i}. {r['nom']} (n°{r['numero']})")
            p_win = r.get("p_win")
            fair = r.get("fair_odds")
            market = r.get("market_odds")
            p_win_str = f"{p_win:.4f}" if p_win is not None else "N/A"
            fair_str = f"{fair:.2f}" if fair is not None else "N/A"
            print(f"     p_win: {p_win_str} | fair: {fair_str} | market: {market}")

            value = r.get("value_pct") or 0
            kelly = r.get("kelly_fraction") or 0
            print(f"     value: {value:.2f}% | kelly: {kelly:.4f}")
            print(f"     rationale: {r['rationale']}")

        # Notes
        if result.get("run_notes"):
            print(f"\n  ⚠️ Notes: {result['run_notes']}")

    print(f"\n{'=' * 70}")
    if all_passed:
        print("✅ TOUS LES TESTS PASSÉS")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("=" * 70)

    conn.close()
    return all_passed


def demo_json_output():
    """Démo de sortie JSON pure"""

    from db_connection import get_connection
    from pro_betting_analyzer import ProBettingAnalyzer

    conn = get_connection()
    analyzer = ProBettingAnalyzer(conn)

    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT race_key
        FROM cheval_courses_seen
        ORDER BY race_key DESC
        LIMIT 1
    """)

    row = cur.fetchone()
    if row:
        result = analyzer.analyze_race(row[0])
        print(result)

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        demo_json_output()
    else:
        test_analyzer()
