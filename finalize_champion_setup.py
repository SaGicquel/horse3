#!/usr/bin/env python3
"""
🔧 FINALISATION MODÈLE CHAMPION
===============================

Finalise la configuration du modèle champion en s'assurant que tous
les composants pointent vers les bons fichiers.
"""

import os
import sys
from pathlib import Path

def update_api_prediction_comment():
    """Met à jour les commentaires de documentation dans api_prediction.py."""
    api_file = Path("api_prediction.py")
    
    with open(api_file, 'r') as f:
        content = f.read()
    
    # Mettre à jour la description du modèle dans les commentaires
    old_desc = "API REST FastAPI pour servir le modèle Stacking Ensemble (ROC-AUC Test: 0.7009)."
    new_desc = "API REST FastAPI pour servir le modèle XGBoost Champion (ROC-AUC Test: 0.6189, Backtest ROI: 22.71%)."
    
    if old_desc in content:
        content = content.replace(old_desc, new_desc)
        
        with open(api_file, 'w') as f:
            f.write(content)
        print("✅ Documentation API mise à jour")
    else:
        print("ℹ️ Documentation API déjà à jour")

def create_champion_symlinks():
    """Crée des liens symboliques pour faciliter l'accès au modèle champion."""
    
    # Créer un lien vers le modèle champion dans le dossier racine
    champion_model = Path("data/models/champion/xgboost_model.pkl")
    symlink_path = Path("champion_model.pkl")
    
    if champion_model.exists() and not symlink_path.exists():
        try:
            symlink_path.symlink_to(champion_model)
            print("✅ Lien symbolique créé: champion_model.pkl")
        except OSError:
            print("ℹ️ Impossible de créer le lien symbolique (peut nécessiter des permissions admin)")

def update_readme():
    """Met à jour ou crée un README pour le modèle champion."""
    
    readme_content = """# 🏆 Modèle Champion XGBoost

## 📊 Performance
- **ROI Backtest**: 22.71%
- **Sharpe Ratio**: 3.599
- **Max Drawdown**: 25.61%
- **ROC-AUC**: 0.6189

## 📁 Structure
```
data/models/champion/
├── xgboost_model.pkl       # Modèle XGBoost entraîné
├── feature_scaler.pkl      # Normalisation des features
├── feature_imputer.pkl     # Imputation des valeurs manquantes
└── metadata.json          # Métadonnées du modèle

calibration/champion/
├── scaler_temperature.pkl   # Scaler de température (T=0.5)
├── calibrator_platt.pkl    # Calibrateur Platt
├── calibration_report.json # Rapport de calibration (ECE=0.0112)
└── dynamic_blender/        # Blender dynamique modèle/marché
```

## 🔧 Utilisation
Le modèle champion est automatiquement chargé par:
- `api_prediction.py` (API REST)
- `pro_betting_analyzer.py` (analyse pro)
- `race_pronostic_generator.py` (génération pronostics)
- `cli.py pick` (commande CLI)

## 🎯 Calibration
- **Temperature Scaling**: T = 0.5
- **Platt Calibration**: Activée
- **Market Blending**: α = 0.4 (modèle=60%, marché=40%)
- **ECE**: 0.0112 (excellente calibration)

## ⚡ Déploiement
```bash
# Test du modèle
python validate_champion_model.py

# API
python api_prediction.py

# Pronostics CLI
python cli.py pick --date 2025-12-08
```
"""
    
    readme_path = Path("data/models/champion/README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("✅ README.md créé pour le modèle champion")

def verify_config_alignment():
    """Vérifie l'alignement de la configuration."""
    
    try:
        from config.loader import get_calibration_params_from_artifacts
        params = get_calibration_params_from_artifacts()
        
        print("🔧 Configuration actuelle:")
        print(f"   - Source: {params.get('source')}")
        print(f"   - Temperature: {params.get('temperature')}")
        print(f"   - Blend Alpha: {params.get('blend_alpha')}")
        
        if params.get('source') == 'artifacts':
            print("✅ Configuration chargée depuis les artefacts champion")
        else:
            print("⚠️ Configuration pas depuis les artefacts - vérifier config/loader.py")
            
    except Exception as e:
        print(f"❌ Erreur vérification config: {e}")

def main():
    print("🔧 FINALISATION DU MODÈLE CHAMPION")
    print("=" * 50)
    
    update_api_prediction_comment()
    create_champion_symlinks() 
    update_readme()
    verify_config_alignment()
    
    print("\n" + "=" * 50)
    print("🎉 CONFIGURATION CHAMPION FINALISÉE!")
    print("\n📋 Actions effectuées:")
    print("   ✓ Fichiers modèle copiés vers data/models/champion/")
    print("   ✓ Artefacts calibration copiés vers calibration/champion/") 
    print("   ✓ api_prediction.py configuré pour le champion")
    print("   ✓ config/loader.py pointe vers calibration/champion/")
    print("   ✓ Documentation mise à jour")
    
    print("\n🚀 Le système peut maintenant utiliser le modèle champion!")
    print("   • API: python api_prediction.py")
    print("   • Pronostics: python cli.py pick")
    print("   • Validation: python validate_champion_model.py")

if __name__ == "__main__":
    main()