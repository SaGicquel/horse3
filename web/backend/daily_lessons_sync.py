#!/usr/bin/env python3
"""
🔄 Script de synchronisation quotidienne des leçons Agent IA
============================================================

Ce script doit être exécuté après 23h quand toutes les courses sont terminées.
Il synchronise les outcomes (résultats) puis régénère les leçons apprises.

Usage:
    python daily_lessons_sync.py [--date YYYY-MM-DD] [--api-url URL]

Exemples:
    python daily_lessons_sync.py
    python daily_lessons_sync.py --date 2025-12-30
    python daily_lessons_sync.py --api-url http://192.168.1.10:8000
"""

import argparse
import sys
from datetime import datetime, date

import requests


def sync_daily_lessons(
    api_base: str = "http://localhost:8000", target_date: date | None = None
) -> bool:
    """
    Synchronise les outcomes puis génère les leçons.

    Args:
        api_base: URL de l'API backend
        target_date: Date cible (défaut: aujourd'hui)

    Returns:
        True si succès, False sinon
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Démarrage synchronisation...")

    date_str = target_date.strftime("%Y-%m-%d") if target_date else None

    # 1. Synchroniser les outcomes
    try:
        params = {"target_date": date_str} if date_str else {}
        resp = requests.post(f"{api_base}/agent/outcomes/sync", params=params, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            synced = data.get("synced", 0)
            errors = data.get("errors", 0)
            print(f"✅ Outcomes sync: {synced} nouveaux, {errors} erreurs")
        else:
            print(f"❌ Erreur sync outcomes: HTTP {resp.status_code}")
            print(f"   Réponse: {resp.text[:200]}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Impossible de se connecter à {api_base}")
        print("   Vérifiez que le backend est démarré.")
        return False
    except requests.exceptions.Timeout:
        print("❌ Timeout lors de la synchronisation (>60s)")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

    # 2. Générer les leçons
    try:
        resp = requests.post(f"{api_base}/agent/lessons/generate-all", timeout=120)

        if resp.status_code == 200:
            data = resp.json()
            lessons = data.get("lessons_created", 0)
            print(f"✅ Leçons générées: {lessons}")
        else:
            print(f"❌ Erreur génération leçons: HTTP {resp.status_code}")
            print(f"   Réponse: {resp.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Timeout lors de la génération des leçons (>120s)")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Synchronisation terminée avec succès"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Synchronise les outcomes et génère les leçons de l'Agent IA"
    )
    parser.add_argument(
        "--date", "-d", type=str, help="Date cible au format YYYY-MM-DD (défaut: aujourd'hui)"
    )
    parser.add_argument(
        "--api-url",
        "-u",
        type=str,
        default="http://localhost:8000",
        help="URL de l'API backend (défaut: http://localhost:8000)",
    )

    args = parser.parse_args()

    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"❌ Format de date invalide: {args.date} (attendu: YYYY-MM-DD)")
            sys.exit(1)

    success = sync_daily_lessons(api_base=args.api_url, target_date=target_date)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
