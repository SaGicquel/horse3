#!/usr/bin/env python3
"""Script de surveillance périodique - vérifie toutes les 10 minutes"""

import subprocess
import time
import os
from datetime import datetime


def check_process_running():
    """Vérifie si le processus de scraping tourne"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "orchestrator_scrapers.py --start 2024-10-01 --end 2024-10-31"],
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except:
        return False


def get_progress():
    """Récupère la progression actuelle"""
    try:
        result = subprocess.run(
            ["python3", "watch_progress.py"], capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except:
        return "❌ Erreur lors de la récupération de la progression"


def main():
    print("🔍 Surveillance automatique du scraping octobre 2024")
    print("=" * 70)
    print(f"Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Vérification toutes les 10 minutes")
    print("Ctrl+C pour arrêter\n")

    iteration = 0

    while True:
        iteration += 1

        # Vérifier si le processus tourne
        is_running = check_process_running()

        print(f"\n{'='*70}")
        print(f"🔄 Vérification #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")

        if is_running:
            print("✅ Processus en cours...\n")
            progress = get_progress()
            print(progress)

            print("\n⏳ Prochaine vérification dans 10 minutes...")
            print("   (Ctrl+C pour arrêter la surveillance)")

            # Attendre 10 minutes
            time.sleep(600)

        else:
            print("🎉 SCRAPING TERMINÉ !\n")
            progress = get_progress()
            print(progress)

            print("\n📊 Lancement de l'analyse finale...")

            # Lancer l'analyse
            try:
                subprocess.run(["python3", "analyse_finale_enrichissement.py"], timeout=300)
                print("\n✅ Analyse terminée !")
                print("📄 Rapport disponible : ANALYSE_ENRICHISSEMENT_OCTOBRE_2024.txt")
            except Exception as e:
                print(f"\n❌ Erreur lors de l'analyse : {e}")

            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Surveillance interrompue par l'utilisateur")
        print("Le scraping continue en arrière-plan (PID 47199)")
        print("Relancer ce script pour reprendre la surveillance\n")
