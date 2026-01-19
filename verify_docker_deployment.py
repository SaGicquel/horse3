#!/usr/bin/env python3
"""
Script de vérification post-déploiement Docker
Vérifie que tout fonctionne après le rebuild des conteneurs
"""

import requests
import sys
from datetime import datetime

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check_endpoint(url, description):
    """Teste un endpoint HTTP"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"{GREEN}✅{RESET} {description}: {url} (HTTP {response.status_code})")
            return True
        else:
            print(f"{RED}❌{RESET} {description}: {url} (HTTP {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"{RED}❌{RESET} {description}: {url} ({e})")
        return False


def main():
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}🐳 Vérification Post-Déploiement Docker - Conseils V2{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    all_checks = []

    # 1. Frontend
    print(f"{YELLOW}🌐 Frontend (Nginx):{RESET}")
    all_checks.append(check_endpoint("http://localhost/", "Page d'accueil"))

    # 2. Backend V1 (via proxy Nginx)
    print(f"\n{YELLOW}🔌 Backend V1 (API Agent IA - Port 8000):{RESET}")
    all_checks.append(check_endpoint("http://localhost/api/health", "Health check V1 (via proxy)"))

    # 3. Backend V2 (direct)
    print(f"\n{YELLOW}🔌 Backend V2 (API Algo Brut - Port 8001):{RESET}")
    all_checks.append(check_endpoint("http://localhost:8001/health", "Health check V2 (direct)"))

    # Test prediction V2 pour aujourd'hui
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        print(f"\n{YELLOW}🎯 Test Prediction V2 (date={today}):{RESET}")
        response = requests.get(
            f"http://localhost:8001/daily-advice-v2?date_str={today}", timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            nb_paris = len(data) if isinstance(data, list) else 0
            print(f"{GREEN}✅{RESET} API V2 retourne {nb_paris} paris pour {today}")

            if nb_paris > 0:
                print(f"\n{BLUE}Paris du jour:{RESET}")
                for i, pari in enumerate(data[:5], 1):
                    print(
                        f"  {i}. {pari['nom']} (#{pari['numero']}) - Cote {pari['cote']:.1f} - Proba {pari['proba']:.1f}% - Mise {pari['mise']:.0f}€"
                    )

                # Calcul statistiques
                total_mise = sum(p["mise"] for p in data)
                total_gain = sum(p["gain_potentiel"] for p in data)
                roi_potentiel = (
                    ((total_gain - total_mise) / total_mise * 100) if total_mise > 0 else 0
                )

                print(f"\n{BLUE}Résumé:{RESET}")
                print(f"  📊 Total mise: {total_mise:.0f}€")
                print(f"  💰 Gain potentiel: {total_gain:.2f}€")
                print(f"  📈 ROI potentiel: {roi_potentiel:+.1f}%")
            all_checks.append(True)
        else:
            print(f"{RED}❌{RESET} API V2 erreur HTTP {response.status_code}")
            all_checks.append(False)
    except Exception as e:
        print(f"{RED}❌{RESET} Erreur test prediction: {e}")
        all_checks.append(False)

    # 4. Docker containers
    print(f"\n{YELLOW}🐳 Conteneurs Docker:{RESET}")
    import subprocess

    try:
        result = subprocess.run(
            [
                "docker-compose",
                "-f",
                "/Users/gicquelsacha/horse3/web/docker-compose.yml",
                "ps",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            import json

            containers = [json.loads(line) for line in result.stdout.strip().split("\n") if line]
            for container in containers:
                name = container.get("Name", "unknown")
                status = container.get("State", "unknown")
                health = container.get("Health", "N/A")

                if status == "running":
                    print(f"{GREEN}✅{RESET} {name}: {status} (health: {health})")
                    all_checks.append(True)
                else:
                    print(f"{RED}❌{RESET} {name}: {status}")
                    all_checks.append(False)
        else:
            print(f"{RED}❌{RESET} Impossible de vérifier les conteneurs Docker")
            all_checks.append(False)
    except Exception as e:
        print(f"{YELLOW}⚠️{RESET}  Vérification Docker manuelle: {e}")

    # Résumé
    print(f"\n{BLUE}{'='*70}{RESET}")
    success_count = sum(all_checks)
    total_count = len(all_checks)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0

    if success_count == total_count:
        print(
            f"{GREEN}✅ Déploiement réussi : {success_count}/{total_count} checks passés ({success_rate:.0f}%){RESET}"
        )
        print(f"\n{GREEN}🎉 Conseils V2 est opérationnel !{RESET}")
        print(f"{BLUE}📌 Accès : http://localhost/conseils2{RESET}")
        print(f"{BLUE}📌 API V2 : http://localhost:8001/daily-advice-v2{RESET}")
    else:
        print(
            f"{RED}❌ Déploiement incomplet : {success_count}/{total_count} checks passés ({success_rate:.0f}%){RESET}"
        )
        print(f"{YELLOW}⚠️  Vérifiez les erreurs ci-dessus{RESET}")
        return 1

    print(f"{BLUE}{'='*70}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
