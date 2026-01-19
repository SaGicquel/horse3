#!/usr/bin/env python3
"""
Script de vérification de l'installation de Conseils V2
Vérifie que tous les fichiers sont en place et que l'API V2 fonctionne
"""

import os
import sys
import requests
from datetime import datetime

# Couleurs pour l'affichage
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check_file(filepath, description):
    """Vérifie qu'un fichier existe"""
    exists = os.path.exists(filepath)
    status = f"{GREEN}✅{RESET}" if exists else f"{RED}❌{RESET}"
    print(f"{status} {description}: {filepath}")
    return exists


def check_api(url, description):
    """Vérifie qu'une API répond"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"{GREEN}✅{RESET} {description}: {url}")
            return True
        else:
            print(f"{RED}❌{RESET} {description}: {url} (HTTP {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"{RED}❌{RESET} {description}: {url} ({e})")
        return False


def main():
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}🔍 Vérification installation Conseils V2 - Algo Brut{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    base_path = "/Users/gicquelsacha/horse3"
    all_checks = []

    # 1. Vérification des fichiers backend
    print(f"\n{YELLOW}📁 Fichiers Backend:{RESET}")
    all_checks.append(check_file(f"{base_path}/user_app_api_v2.py", "API V2 Backend"))
    all_checks.append(check_file(f"{base_path}/compare_conseils.py", "Script de comparaison"))
    all_checks.append(
        check_file(f"{base_path}/audit_ultimate_config.py", "Script d'audit validation")
    )

    # 2. Vérification des fichiers frontend
    print(f"\n{YELLOW}📁 Fichiers Frontend:{RESET}")
    all_checks.append(
        check_file(f"{base_path}/web/frontend/src/pages/Conseils2.jsx", "Page Conseils V2")
    )
    all_checks.append(check_file(f"{base_path}/web/frontend/src/App.jsx", "Routing App.jsx"))
    all_checks.append(
        check_file(f"{base_path}/web/frontend/src/components/Navigation.jsx", "Navigation.jsx")
    )

    # 3. Vérification documentation
    print(f"\n{YELLOW}📁 Documentation:{RESET}")
    all_checks.append(check_file(f"{base_path}/CONSEILS2_README.md", "Documentation Conseils V2"))

    # 4. Vérification APIs
    print(f"\n{YELLOW}🌐 APIs:{RESET}")
    all_checks.append(check_api("http://localhost:8001/health", "API V2 Health Check"))

    # Test prediction pour aujourd'hui
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        print(f"\n{YELLOW}🎯 Test Prediction (date={today}):{RESET}")
        response = requests.get(
            f"http://localhost:8001/daily-advice-v2?date_str={today}", timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            nb_paris = len(data) if isinstance(data, list) else 0
            print(f"{GREEN}✅{RESET} API retourne {nb_paris} paris pour aujourd'hui")

            if nb_paris > 0:
                print(f"\n{BLUE}Exemple de paris:{RESET}")
                for i, pari in enumerate(data[:3], 1):
                    print(
                        f"  {i}. {pari['nom']} (#{pari['numero']}) - Cote {pari['cote']:.1f} - Proba {pari['proba']:.1f}% - Mise {pari['mise']:.0f}€"
                    )
            all_checks.append(True)
        else:
            print(f"{RED}❌{RESET} API erreur HTTP {response.status_code}")
            all_checks.append(False)
    except Exception as e:
        print(f"{RED}❌{RESET} Erreur lors du test prediction: {e}")
        all_checks.append(False)

    # Résumé
    print(f"\n{BLUE}{'='*70}{RESET}")
    success_count = sum(all_checks)
    total_count = len(all_checks)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0

    if success_count == total_count:
        print(
            f"{GREEN}✅ Installation complète : {success_count}/{total_count} checks passés ({success_rate:.0f}%){RESET}"
        )
        print(f"\n{GREEN}🎉 Conseils V2 est prêt à l'emploi !{RESET}")
        print(f"{BLUE}📌 Accès : http://localhost:5173/conseils2{RESET}")
    else:
        print(
            f"{RED}❌ Installation incomplète : {success_count}/{total_count} checks passés ({success_rate:.0f}%){RESET}"
        )
        print(f"{YELLOW}⚠️  Vérifiez les erreurs ci-dessus{RESET}")
        return 1

    print(f"{BLUE}{'='*70}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
