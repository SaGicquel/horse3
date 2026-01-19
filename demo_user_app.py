#!/usr/bin/env python3
"""
HORSE3 USER APP - DÉMONSTRATION INTERACTIVE

Ce script démontre les capacités de l'application utilisateur
en simulant des scénarios d'utilisation réels.
"""

import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
import sys


class Horse3UserDemo:
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.session = requests.Session()

    def print_banner(self):
        """Affiche le banner de démo"""
        print("\n" + "=" * 80)
        print("🏇 HORSE3 USER APP - DÉMONSTRATION INTERACTIVE")
        print("=" * 80)
        print("🏆 Modèle Champion XGBoost v1.0")
        print("🎯 ROI +22.71% | Sharpe 3.599 | ECE 0.0112")
        print("💡 Stratégie Blend + Kelly Optimisée")
        print("=" * 80 + "\n")

    def check_api_health(self) -> bool:
        """Vérifie la santé de l'API"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Status: {data['status']}")
                print(f"🕒 Timestamp: {data['timestamp']}")
                print(f"🏆 Champion Model: {data['champion_model_configured']}")
                return True
            else:
                print(f"❌ API Health Check Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Erreur connexion API: {e}")
            print("💡 Assurez-vous que l'API est démarrée: python user_app_api.py")
            return False

    def demo_daily_advice(self, date_str: str = None):
        """Démontre la page Conseils du jour"""
        print("\n" + "🎯 DÉMONSTRATION - CONSEILS DU JOUR")
        print("-" * 50)

        if not date_str:
            date_str = "2025-12-08"

        try:
            response = self.session.get(f"{self.api_url}/daily-advice?date_str={date_str}")

            if response.status_code == 200:
                data = response.json()

                print(f"📅 Date: {data['date']}")
                print(f"🎯 Nombre de conseils: {len(data['conseils'])}")
                print(f"💰 Total des mises: {data['total_mise']:.2f}€")
                print(f"📊 EV moyenne: {data['ev_moyen']:.1f}%")

                print("\n🏇 TOP 5 CONSEILS:")
                for i, conseil in enumerate(data["conseils"][:5], 1):
                    profil_emoji = {"SÛR": "🛡️", "Standard": "⚖️", "Ambitieux": "🚀"}
                    emoji = profil_emoji.get(conseil["profil"], "❓")

                    print(f"\n{i}. {conseil['nom']} ({conseil['race_key']})")
                    print(f"   {emoji} Profil: {conseil['profil']}")
                    print(f"   🎯 Probabilité: {conseil['p_final']:.1f}%")
                    print(f"   💎 Value: +{conseil['value']:.1f}%")
                    print(f"   💰 Mise: {conseil['mise_conseillee']:.2f}€")
                    print(f"   📈 EV: +{conseil['ev_pct']:.1f}%")

            else:
                print(f"❌ Erreur: {response.status_code}")
                print(f"📝 Message: {response.text}")

        except Exception as e:
            print(f"❌ Erreur lors de la démo: {e}")

    def demo_portfolio(self, date_str: str = None):
        """Démontre la page Portefeuille"""
        print("\n" + "💼 DÉMONSTRATION - PORTEFEUILLE")
        print("-" * 50)

        if not date_str:
            date_str = "2025-12-08"

        try:
            response = self.session.get(f"{self.api_url}/portfolio?date_str={date_str}")

            if response.status_code == 200:
                data = response.json()

                print(f"📅 Date: {data['date']}")
                print(f"💰 Bankroll de référence: {data['bankroll_reference']:,.0f}€")
                print(f"🎯 Mise totale du jour: {data['mise_totale']:.2f}€")

                # Indicateur de risque avec couleurs
                risque = data["risque_pct"]
                if risque <= 10:
                    risque_status = f"✅ FAIBLE ({risque:.1f}%)"
                elif risque <= 25:
                    risque_status = f"⚠️  MODÉRÉ ({risque:.1f}%)"
                else:
                    risque_status = f"❌ ÉLEVÉ ({risque:.1f}%)"

                print(f"⚖️  Risque du jour: {risque_status}")
                print(f"🎲 Nombre de paris: {data['nombre_paris']}")

                # Répartition par profil
                profils = {}
                for pari in data["paris_details"]:
                    profil = pari["profil"]
                    if profil not in profils:
                        profils[profil] = {"count": 0, "mise": 0}
                    profils[profil]["count"] += 1
                    profils[profil]["mise"] += pari["mise_conseillee"]

                print("\n📊 RÉPARTITION PAR PROFIL:")
                profil_emojis = {"SÛR": "🛡️", "Standard": "⚖️", "Ambitieux": "🚀"}
                for profil, stats in profils.items():
                    emoji = profil_emojis.get(profil, "❓")
                    print(f"   {emoji} {profil}: {stats['count']} paris, {stats['mise']:.2f}€")

            else:
                print(f"❌ Erreur: {response.status_code}")

        except Exception as e:
            print(f"❌ Erreur lors de la démo: {e}")

    def demo_bankroll_update(self):
        """Démontre la mise à jour de la bankroll"""
        print("\n" + "⚙️  DÉMONSTRATION - GESTION BANKROLL")
        print("-" * 50)

        # Sauvegarde la bankroll actuelle
        try:
            portfolio_response = self.session.get(f"{self.api_url}/portfolio")
            current_bankroll = 1000  # défaut
            if portfolio_response.status_code == 200:
                current_bankroll = portfolio_response.json()["bankroll_reference"]

            print(f"💰 Bankroll actuelle: {current_bankroll:,.0f}€")

            # Test mise à jour
            new_bankroll = 1500
            print(f"🔄 Mise à jour vers: {new_bankroll:,.0f}€")

            update_response = self.session.post(
                f"{self.api_url}/update-bankroll?bankroll={new_bankroll}"
            )

            if update_response.status_code == 200:
                data = update_response.json()
                print("✅ Mise à jour réussie!")
                print(f"📝 Message: {data['message']}")
                print(f"💰 Nouvelle bankroll: {data['nouvelle_bankroll']:,.0f}€")

                # Vérification
                time.sleep(1)
                verify_response = self.session.get(f"{self.api_url}/portfolio")
                if verify_response.status_code == 200:
                    verify_data = verify_response.json()
                    new_risk = verify_data["risque_pct"]
                    print(f"⚖️  Nouveau risque du jour: {new_risk:.1f}%")

                # Restaure la bankroll originale
                print("\n🔄 Restauration de la bankroll originale...")
                restore_response = self.session.post(
                    f"{self.api_url}/update-bankroll?bankroll={current_bankroll}"
                )
                if restore_response.status_code == 200:
                    print("✅ Bankroll restaurée")

            else:
                print(f"❌ Erreur mise à jour: {update_response.status_code}")

        except Exception as e:
            print(f"❌ Erreur lors de la démo: {e}")

    def demo_historical_stats(self):
        """Démontre la page Historique & Stats"""
        print("\n" + "📊 DÉMONSTRATION - HISTORIQUE & STATS")
        print("-" * 50)

        try:
            response = self.session.get(f"{self.api_url}/historical-stats")

            if response.status_code == 200:
                data = response.json()

                print("📅 Période: 6 derniers mois")
                print(f"🎯 Total paris: {data['nb_paris_total']:,}")
                print(f"📈 ROI moyen: {data.get('roi_moyen', 0):.1f}%")
                print(f"📉 Drawdown actuel: {data['drawdown_actuel']:.1f}%")
                print(f"📉 Drawdown max: {data['drawdown_max']:.1f}%")
                print(f"🔥 Série gagnante: {data['serie_gagnante']}")
                print(f"❄️  Série perdante: {data['serie_perdante']}")

                # ROI mensuel
                print("\n📊 ROI MENSUEL (6 derniers mois):")
                roi_mensuel = data.get("roi_mensuel", {})
                for mois, roi in roi_mensuel.items():
                    if roi >= 0:
                        print(f"   📈 {mois}: +{roi:.1f}%")
                    else:
                        print(f"   📉 {mois}: {roi:.1f}%")

                # Évolution bankroll
                evolution = data.get("bankroll_evolution", [])
                if evolution:
                    print("\n💰 ÉVOLUTION BANKROLL (30 derniers jours):")
                    first_day = evolution[0]["bankroll"]
                    last_day = evolution[-1]["bankroll"]
                    evolution_pct = ((last_day - first_day) / first_day) * 100
                    print(f"   🚀 Début période: {first_day:,.0f}€")
                    print(f"   🎯 Fin période: {last_day:,.0f}€")
                    print(f"   📊 Évolution: {evolution_pct:+.1f}%")

            else:
                print(f"❌ Erreur: {response.status_code}")

        except Exception as e:
            print(f"❌ Erreur lors de la démo: {e}")

    def demo_api_endpoints(self):
        """Démontre tous les endpoints de l'API"""
        print("\n" + "🔗 DÉMONSTRATION - ENDPOINTS API")
        print("-" * 50)

        endpoints = [
            ("GET /", "Status API"),
            ("GET /health", "Health check"),
            ("GET /daily-advice", "Conseils du jour"),
            ("GET /portfolio", "Portefeuille"),
            ("GET /historical-stats", "Stats historiques"),
            ("POST /update-bankroll", "Mise à jour bankroll"),
        ]

        print("📋 ENDPOINTS DISPONIBLES:")
        for endpoint, description in endpoints:
            print(f"   {endpoint:<25} - {description}")

        print(f"\n🌐 URL Base API: {self.api_url}")
        print("📖 Documentation: http://localhost:8001/docs")
        print("🔧 Redoc: http://localhost:8001/redoc")

    def run_complete_demo(self):
        """Lance la démonstration complète"""
        self.print_banner()

        # Vérification API
        if not self.check_api_health():
            return

        print("\n🎬 DÉMARRAGE DE LA DÉMONSTRATION...")
        input("Appuyez sur Entrée pour continuer...")

        # Démo 1: Conseils du jour
        self.demo_daily_advice()
        input("\nAppuyez sur Entrée pour continuer...")

        # Démo 2: Portefeuille
        self.demo_portfolio()
        input("\nAppuyez sur Entrée pour continuer...")

        # Démo 3: Gestion bankroll
        self.demo_bankroll_update()
        input("\nAppuyez sur Entrée pour continuer...")

        # Démo 4: Stats historiques
        self.demo_historical_stats()
        input("\nAppuyez sur Entrée pour continuer...")

        # Démo 5: Endpoints API
        self.demo_api_endpoints()

        print("\n" + "=" * 80)
        print("🎉 DÉMONSTRATION TERMINÉE !")
        print("🏆 L'application utilisateur Horse3 est prête à maximiser vos gains !")
        print("📖 Guide complet: USER_APP_GUIDE.md")
        print("🚀 Pour démarrer: python user_app_api.py")
        print("=" * 80 + "\n")

    def run_quick_demo(self):
        """Lance une démonstration rapide"""
        self.print_banner()

        if not self.check_api_health():
            return

        print("⚡ DÉMONSTRATION RAPIDE\n")

        # Tests rapides de tous les endpoints
        self.demo_daily_advice()
        self.demo_portfolio()
        self.demo_historical_stats()

        print("\n🎉 Démo rapide terminée ! Utilisez --full pour la démo complète.")


def main():
    """Point d'entrée principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Démonstration Horse3 User App")
    parser.add_argument("--full", action="store_true", help="Démonstration complète interactive")
    parser.add_argument(
        "--url",
        default="http://localhost:8001",
        help="URL de l'API (défaut: http://localhost:8001)",
    )

    args = parser.parse_args()

    demo = Horse3UserDemo(api_url=args.url)

    if args.full:
        demo.run_complete_demo()
    else:
        demo.run_quick_demo()


if __name__ == "__main__":
    main()
