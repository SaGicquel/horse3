#!/usr/bin/env python3
"""
🚀 LANCEUR APP UTILISATEUR
==========================

Script pour démarrer l'application utilisateur Horse3 complète :
- API backend (port 8001)
- Pages : Conseils du jour, Portefeuille, Historique & Stats
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Vérifie que les dépendances sont installées."""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        print("✅ Dépendances Python OK")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("💡 Installez avec: pip install fastapi uvicorn pandas numpy")
        return False

def check_champion_model():
    """Vérifie que le modèle champion est disponible."""
    champion_path = Path("data/models/champion/xgboost_model.pkl")
    calibration_path = Path("calibration/champion/calibration_report.json")
    
    if champion_path.exists() and calibration_path.exists():
        print("✅ Modèle champion disponible")
        return True
    else:
        print("❌ Modèle champion manquant")
        print("💡 Exécutez d'abord: python validate_champion_model.py")
        return False

def check_picks_data():
    """Vérifie qu'il y a des picks pour aujourd'hui."""
    today = time.strftime("%Y-%m-%d")
    picks_file = Path(f"data/picks/picks_{today}.json")
    
    if picks_file.exists():
        print(f"✅ Picks disponibles pour {today}")
        return True
    else:
        print(f"⚠️ Pas de picks pour {today}")
        print("💡 Générez avec: python cli.py pick")
        return False

def start_user_api():
    """Lance l'API utilisateur."""
    print("🚀 Démarrage API utilisateur (port 8001)...")
    
    try:
        # Utiliser l'environnement virtuel si disponible
        python_cmd = sys.executable
        if Path(".venv/bin/python").exists():
            python_cmd = ".venv/bin/python"
        elif Path("venv/bin/python").exists():
            python_cmd = "venv/bin/python"
        
        # Lancer l'API
        process = subprocess.Popen([
            python_cmd, "user_app_api.py"
        ])
        
        print(f"✅ API démarrée (PID: {process.pid})")
        print("🌐 Accès: http://localhost:8001")
        print("📋 Docs API: http://localhost:8001/docs")
        
        return process
        
    except Exception as e:
        print(f"❌ Erreur démarrage API: {e}")
        return None

def print_endpoints():
    """Affiche les endpoints disponibles."""
    print("\n" + "="*60)
    print("🎯 ENDPOINTS API UTILISATEUR")
    print("="*60)
    
    endpoints = [
        ("GET  /", "Statut de l'API"),
        ("GET  /daily-advice", "Conseils du jour (avec p_final, value, mise)"),
        ("GET  /portfolio", "Portefeuille (recap mises, bankroll, risque)"),
        ("GET  /historical-stats", "Stats historiques (ROI, drawdown, séries)"),
        ("POST /update-bankroll", "Mise à jour bankroll de référence"),
        ("GET  /health", "Health check")
    ]
    
    for endpoint, description in endpoints:
        print(f"  {endpoint:<25} {description}")
    
    print("\n🔧 EXEMPLES D'UTILISATION:")
    print("  curl http://localhost:8001/daily-advice")
    print("  curl http://localhost:8001/portfolio") 
    print("  curl http://localhost:8001/historical-stats")
    print("  curl -X POST 'http://localhost:8001/update-bankroll?bankroll=1500'")

def print_frontend_integration():
    """Affiche des infos d'intégration frontend."""
    print("\n" + "="*60)
    print("🖥️ INTÉGRATION FRONTEND")
    print("="*60)
    
    print("📁 Pages React créées:")
    pages = [
        "src/pages/DailyAdvice.jsx",
        "src/pages/Portfolio.jsx", 
        "src/pages/HistoricalStats.jsx",
        "src/pages/UserDashboard.jsx"
    ]
    
    for page in pages:
        if Path(f"web/frontend/{page}").exists():
            print(f"  ✅ {page}")
        else:
            print(f"  ❌ {page}")
    
    print("\n🔗 Pour intégrer au routing React:")
    print("  1. Ajoutez les routes dans App.jsx")
    print("  2. Configurez l'API base URL vers localhost:8001") 
    print("  3. Installez les dépendances: recharts, lucide-react")

def main():
    """Fonction principale."""
    print("🏇 HORSE3 - DÉMARRAGE APPLICATION UTILISATEUR")
    print("=" * 60)
    
    # Vérifications préalables
    if not check_requirements():
        return False
        
    if not check_champion_model():
        return False
        
    check_picks_data()  # Warning seulement
    
    # Démarrage API
    api_process = start_user_api()
    
    if not api_process:
        return False
    
    # Informations
    print_endpoints()
    print_frontend_integration()
    
    print("\n" + "="*60)
    print("🎉 APPLICATION UTILISATEUR DÉMARRÉE!")
    print("="*60)
    print("⚡ API Backend: http://localhost:8001") 
    print("📚 Documentation: http://localhost:8001/docs")
    print("🎯 Pages disponibles:")
    print("   • Conseils du jour (p_final, value, mise, profil)")
    print("   • Portefeuille (bankroll, risque, recap mises)")
    print("   • Historique & Stats (ROI mensuel, drawdown, séries)")
    print("\n🛑 Appuyez sur Ctrl+C pour arrêter")
    
    try:
        # Attendre interruption utilisateur
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par utilisateur")
        api_process.terminate()
        api_process.wait()
        print("✅ API arrêtée proprement")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)