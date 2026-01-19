#!/usr/bin/env python3
"""
📊 COMPARAISON CONSEILS V1 vs V2
================================

Compare les résultats des deux systèmes:
- V1 (port 8000): Système actuel avec Agent IA
- V2 (port 8001): Algo brut optimisé (+71% ROI validé)

Usage:
    python3 compare_conseils.py --date 2025-11-01
"""

import requests
import json
from datetime import datetime, timedelta
import argparse
from typing import List, Dict
import pandas as pd

# URLs des APIs
API_V1_URL = "http://localhost:8000"
API_V2_URL = "http://localhost:8001"


def get_conseils_v1(date_str: str) -> List[Dict]:
    """Récupère les conseils de la V1 (Agent IA)."""
    try:
        response = requests.get(f"{API_V1_URL}/daily-advice", params={"date_str": date_str})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"⚠️  V1 erreur {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print(f"❌ V1 non disponible: {e}")
        return []


def get_conseils_v2(date_str: str) -> List[Dict]:
    """Récupère les conseils de la V2 (Algo brut)."""
    try:
        response = requests.get(f"{API_V2_URL}/daily-advice-v2", params={"date_str": date_str})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"⚠️  V2 erreur {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print(f"❌ V2 non disponible: {e}")
        print("   Démarrez l'API V2 avec: python3 user_app_api_v2.py")
        return []


def compare_conseils(date_str: str):
    """Compare les conseils des deux versions."""
    print("=" * 100)
    print(f"📊 COMPARAISON CONSEILS - {date_str}")
    print("=" * 100)

    # Récupérer conseils
    print("\n[1/3] Récupération des conseils...")
    conseils_v1 = get_conseils_v1(date_str)
    conseils_v2 = get_conseils_v2(date_str)

    print(f"  ✓ V1 (Agent IA): {len(conseils_v1)} conseils")
    print(f"  ✓ V2 (Algo brut): {len(conseils_v2)} conseils")

    # Affichage V1
    if conseils_v1:
        print("\n" + "=" * 100)
        print("🧠 VERSION 1 - SYSTÈME ACTUEL (Agent IA)")
        print("=" * 100)
        print(
            f"\n{'Cheval':<25} | {'#':<3} | {'Cote':<6} | {'Proba':<7} | {'Value':<7} | {'Mise':<8} | {'Profil':<15}"
        )
        print("-" * 100)

        for c in conseils_v1[:10]:  # Top 10
            print(
                f"{c['nom'][:24]:<25} | {c['numero']:<3} | {c.get('odds', 0):<6.1f} | "
                f"{c.get('p_final', 0):<6.1f}% | {c.get('value', 0):>+6.1f}% | "
                f"{c.get('mise_conseillee', 0):>7.2f}€ | {c.get('profil', 'N/A'):<15}"
            )

        if len(conseils_v1) > 10:
            print(f"\n... et {len(conseils_v1) - 10} autres conseils")

        # Stats V1
        total_mise_v1 = sum(c.get("mise_conseillee", 0) for c in conseils_v1)
        value_moy_v1 = (
            sum(c.get("value", 0) for c in conseils_v1) / len(conseils_v1) if conseils_v1 else 0
        )
        proba_moy_v1 = (
            sum(c.get("p_final", 0) for c in conseils_v1) / len(conseils_v1) if conseils_v1 else 0
        )

        print("\n📊 STATISTIQUES V1:")
        print(f"  - Mise totale: {total_mise_v1:.2f}€")
        print(f"  - Value moyenne: {value_moy_v1:+.1f}%")
        print(f"  - Proba moyenne: {proba_moy_v1:.1f}%")

    # Affichage V2
    if conseils_v2:
        print("\n" + "=" * 100)
        print("⚡ VERSION 2 - ALGO BRUT OPTIMISÉ (+71% ROI validé)")
        print("=" * 100)
        print(
            f"\n{'Cheval':<25} | {'#':<3} | {'Cote':<6} | {'Proba':<7} | {'Cote Placé':<10} | {'Mise':<8} | {'Gain pot.':<10}"
        )
        print("-" * 100)

        for c in conseils_v2[:10]:  # Top 10
            print(
                f"{c['nom'][:24]:<25} | {c['numero']:<3} | {c['cote']:<6.1f} | "
                f"{c['proba']:<6.1f}% | {c['cote_place']:<10.2f} | "
                f"{c['mise']:>7.2f}€ | {c['gain_potentiel']:>9.2f}€"
            )

        if len(conseils_v2) > 10:
            print(f"\n... et {len(conseils_v2) - 10} autres conseils")

        # Stats V2
        total_mise_v2 = sum(c["mise"] for c in conseils_v2)
        proba_moy_v2 = sum(c["proba"] for c in conseils_v2) / len(conseils_v2) if conseils_v2 else 0
        cote_moy_v2 = sum(c["cote"] for c in conseils_v2) / len(conseils_v2) if conseils_v2 else 0
        gain_pot_total_v2 = sum(c["gain_potentiel"] for c in conseils_v2)

        print("\n📊 STATISTIQUES V2:")
        print(f"  - Mise totale: {total_mise_v2:.2f}€")
        print(f"  - Proba moyenne: {proba_moy_v2:.1f}%")
        print(f"  - Cote moyenne: {cote_moy_v2:.1f}")
        print(f"  - Gain potentiel total: {gain_pot_total_v2:.2f}€")
        print(
            f"  - ROI potentiel si TOUS placés: {(gain_pot_total_v2 - total_mise_v2) / total_mise_v2 * 100:+.1f}%"
        )

    # Comparaison
    print("\n" + "=" * 100)
    print("🔍 ANALYSE COMPARATIVE")
    print("=" * 100)

    if conseils_v1 and conseils_v2:
        print("\n📌 Nombre de paris:")
        print(f"  - V1 (Agent IA): {len(conseils_v1)} paris")
        print(f"  - V2 (Algo brut): {len(conseils_v2)} paris")
        print(
            f"  → Différence: {abs(len(conseils_v1) - len(conseils_v2))} paris "
            f"({'V1 plus sélective' if len(conseils_v1) < len(conseils_v2) else 'V2 plus sélective'})"
        )

        print("\n💰 Mise totale:")
        print(f"  - V1 (Agent IA): {total_mise_v1:.2f}€")
        print(f"  - V2 (Algo brut): {total_mise_v2:.2f}€")
        print(f"  → Différence: {abs(total_mise_v1 - total_mise_v2):.2f}€")

        print("\n🎯 Probabilité moyenne:")
        print(f"  - V1 (Agent IA): {proba_moy_v1:.1f}%")
        print(f"  - V2 (Algo brut): {proba_moy_v2:.1f}%")

        # Chevaux en commun
        chevaux_v1 = {(c["nom"], c["numero"]) for c in conseils_v1}
        chevaux_v2 = {(c["nom"], c["numero"]) for c in conseils_v2}
        communs = chevaux_v1 & chevaux_v2

        print("\n🔗 Chevaux en commun:")
        print(f"  - {len(communs)} chevaux sélectionnés par les DEUX systèmes")
        if communs:
            print("  → Chevaux validés par V1 ET V2:")
            for nom, num in sorted(communs):
                print(f"     • {nom} (#{num})")

        uniques_v1 = chevaux_v1 - chevaux_v2
        uniques_v2 = chevaux_v2 - chevaux_v1

        if uniques_v1:
            print(f"\n  - {len(uniques_v1)} chevaux UNIQUEMENT V1 (Agent IA):")
            for nom, num in sorted(list(uniques_v1)[:5]):
                print(f"     • {nom} (#{num})")

        if uniques_v2:
            print(f"\n  - {len(uniques_v2)} chevaux UNIQUEMENT V2 (Algo brut):")
            for nom, num in sorted(list(uniques_v2)[:5]):
                print(f"     • {nom} (#{num})")

    elif not conseils_v1 and not conseils_v2:
        print("\n⚠️  Aucun conseil disponible pour les deux versions")
    elif not conseils_v1:
        print("\n⚠️  V1 (Agent IA) non disponible - Vérifiez que l'API tourne sur le port 8000")
    elif not conseils_v2:
        print("\n⚠️  V2 (Algo brut) non disponible - Lancez: python3 user_app_api_v2.py")

    print("\n" + "=" * 100)
    print("FIN DE LA COMPARAISON")
    print("=" * 100)


def compare_multiple_days(start_date: str, days: int = 7):
    """Compare les conseils sur plusieurs jours."""
    print("\n" + "=" * 100)
    print(f"📅 COMPARAISON SUR {days} JOURS")
    print("=" * 100)

    start = datetime.strptime(start_date, "%Y-%m-%d")

    summary = []

    for i in range(days):
        date = start + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")

        print(f"\n[{i+1}/{days}] {date_str}...")

        conseils_v1 = get_conseils_v1(date_str)
        conseils_v2 = get_conseils_v2(date_str)

        summary.append(
            {
                "date": date_str,
                "v1_count": len(conseils_v1),
                "v2_count": len(conseils_v2),
                "v1_mise": sum(c.get("mise_conseillee", 0) for c in conseils_v1),
                "v2_mise": sum(c["mise"] for c in conseils_v2),
            }
        )

    # Afficher résumé
    df = pd.DataFrame(summary)

    print("\n" + "=" * 100)
    print("📊 RÉSUMÉ")
    print("=" * 100)
    print(f"\n{df.to_string(index=False)}")

    print("\n🔢 TOTAUX:")
    print(f"  - V1 (Agent IA): {df['v1_count'].sum()} paris, {df['v1_mise'].sum():.2f}€")
    print(f"  - V2 (Algo brut): {df['v2_count'].sum()} paris, {df['v2_mise'].sum():.2f}€")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Conseils V1 vs V2")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date au format YYYY-MM-DD",
    )
    parser.add_argument("--days", type=int, default=1, help="Nombre de jours à comparer")

    args = parser.parse_args()

    if args.days == 1:
        compare_conseils(args.date)
    else:
        compare_multiple_days(args.date, args.days)
