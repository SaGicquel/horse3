#!/usr/bin/env python3
"""
🧠 AI Model Optimizer - Utilise Gemini pour optimiser tes scripts d'entraînement
================================================================================
Ce script analyse ton modèle actuel et tes données, puis utilise l'IA pour:
- Analyser les métriques et les features
- Suggérer de nouvelles features à créer
- Proposer des hyperparamètres optimaux
- Identifier les points faibles du modèle
- Générer du code d'amélioration

Usage:
    python ai_model_optimizer.py --analyze         # Analyse complète
    python ai_model_optimizer.py --suggest-features  # Suggestions de features
    python ai_model_optimizer.py --optimize-params   # Optimisation hyperparamètres
    python ai_model_optimizer.py --full-report       # Rapport complet
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_DIR = Path("data/models/champion")
OPTIMIZED_DIR = Path("data/models/optimized")

# Vérifier la clé API
if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY non définie. Export-la d'abord:")
    print("   export GOOGLE_API_KEY=AIzaSyBi_o5kQJ9Nl1OR-_y2axv2Iv5nLKI9f7w")
    sys.exit(1)

# Import Gemini - nouveau SDK
try:
    from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)
    MODEL_NAME = "gemini-2.0-flash"  # Modèle gratuit et rapide
except ImportError:
    print("❌ Package google-genai non installé. Run: pip install google-genai")
    sys.exit(1)

# Import pour l'analyse
import numpy as np
import pandas as pd


class AIModelOptimizer:
    """Optimiseur de modèle assisté par IA."""

    def __init__(self):
        self.client = client
        self.model_name = MODEL_NAME
        self.current_metrics = {}
        self.feature_names = []
        self.feature_importances = {}
        self.metadata = {}

    def load_model_info(self) -> Dict[str, Any]:
        """Charge les informations du modèle actuel."""
        info = {}

        # Charger les métadonnées
        meta_path = MODEL_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
                info["metadata"] = self.metadata
                print(f"✅ Métadonnées chargées: {self.metadata.get('version', 'unknown')}")

        # Charger les noms de features
        features_path = MODEL_DIR / "feature_names.json"
        if features_path.exists():
            with open(features_path) as f:
                self.feature_names = json.load(f)
                info["n_features"] = len(self.feature_names)
                info["features"] = self.feature_names
                print(f"✅ {len(self.feature_names)} features chargées")

        # Charger aussi depuis optimized si disponible
        opt_meta_path = OPTIMIZED_DIR / "metadata.json"
        if opt_meta_path.exists():
            with open(opt_meta_path) as f:
                opt_meta = json.load(f)
                info["optimized_params"] = opt_meta.get("best_params", {})
                info["optimized_score"] = opt_meta.get("best_score", 0)
                info["n_trials"] = opt_meta.get("n_trials", 0)

        return info

    def analyze_training_scripts(self) -> str:
        """Analyse les scripts d'entraînement existants."""
        scripts = []

        # Chercher les scripts de training
        training_files = [
            "train_xgboost.py",
            "train_models_SAFE.py",
            "train_model_conservative.py",
            "optimize_model_full.py",
            "prepare_ml_features.py",
        ]

        for script_name in training_files:
            script_path = Path(script_name)
            if script_path.exists():
                with open(script_path) as f:
                    content = f.read()
                    # Prendre les 200 premières lignes pour éviter les tokens excessifs
                    lines = content.split("\n")[:200]
                    total_lines = len(content.split("\n"))
                    scripts.append(
                        {
                            "name": script_name,
                            "content": "\n".join(lines),
                            "total_lines": total_lines,
                        }
                    )
                    print(f"✅ Script analysé: {script_name} ({total_lines} lignes)")

        return scripts

    def get_ai_analysis(self, prompt: str, context: Dict = None) -> str:
        """Envoie une requête à Gemini et retourne la réponse."""
        full_prompt = f"""Tu es un expert en Machine Learning spécialisé dans la prédiction de courses hippiques.

CONTEXTE:
{json.dumps(context, indent=2, ensure_ascii=False) if context else 'Aucun contexte fourni'}

TÂCHE:
{prompt}

Réponds en français avec des suggestions concrètes et du code Python si pertinent.
Utilise le format markdown pour structurer ta réponse.
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=full_prompt
            )
            return response.text
        except Exception as e:
            return f"Erreur Gemini: {e}"

    def analyze_model(self) -> str:
        """Analyse complète du modèle actuel."""
        print("\n🔍 Chargement des informations du modèle...")
        model_info = self.load_model_info()

        print("\n📊 Analyse avec Gemini...")
        prompt = """Analyse ce modèle de prédiction hippique et fournis:

1. **Évaluation des Métriques** - Que penses-tu du score AUC et des paramètres?
2. **Analyse des Features** - Quelles features semblent les plus importantes? Lesquelles manquent?
3. **Points Faibles Potentiels** - Où le modèle pourrait-il s'améliorer?
4. **Recommandations Prioritaires** - Top 3 des améliorations à faire

Sois concis et actionnable."""

        return self.get_ai_analysis(prompt, model_info)

    def suggest_new_features(self) -> str:
        """Suggère de nouvelles features à créer."""
        model_info = self.load_model_info()

        prompt = """En analysant les features existantes du modèle hippique, suggère:

1. **Nouvelles Features Statistiques** - Basées sur les données existantes
2. **Features Temporelles** - Tendances, momentum, saisonnalité
3. **Features Contextuelles** - Météo, terrain, distance
4. **Features de Ranking** - Comparaisons entre chevaux

Pour chaque suggestion, donne:
- Le nom de la feature
- La formule de calcul
- Un snippet de code Python pour la créer

Limite-toi à 5-7 features vraiment impactantes."""

        return self.get_ai_analysis(prompt, model_info)

    def optimize_hyperparameters(self) -> str:
        """Suggère des hyperparamètres optimaux."""
        model_info = self.load_model_info()

        prompt = """En analysant les hyperparamètres actuels du modèle XGBoost:

1. **Analyse des Paramètres Actuels** - Sont-ils cohérents?
2. **Suggestions d'Optimisation** - Quels paramètres ajuster?
3. **Plages de Recherche** - Quelles valeurs tester en priorité?
4. **Code Optuna** - Génère un objectif Optuna optimisé

Prends en compte:
- La prévention de l'overfitting
- L'équilibre vitesse/précision
- La stabilité des prédictions"""

        return self.get_ai_analysis(prompt, model_info)

    def analyze_scripts_with_ai(self) -> str:
        """Analyse les scripts d'entraînement et suggère des améliorations."""
        print("\n📂 Analyse des scripts d'entraînement...")
        scripts = self.analyze_training_scripts()

        if not scripts:
            return "Aucun script d'entraînement trouvé."

        # Prendre seulement les noms et un résumé pour économiser les tokens
        scripts_summary = [{"name": s["name"], "lines": s["total_lines"]} for s in scripts]

        # Analyser le script principal
        main_script = next(
            (s for s in scripts if s["name"] == "optimize_model_full.py"), scripts[0]
        )

        prompt = f"""Analyse ce script d'entraînement de modèle hippique et suggère des améliorations:

SCRIPT: {main_script["name"]}
```python
{main_script["content"][:3000]}
```

Fournis:
1. **Points Forts** - Ce qui est bien fait
2. **Améliorations de Code** - Bugs potentiels, optimisations
3. **Améliorations de Méthodologie** - Cross-validation, feature engineering
4. **Nouveau Code** - Snippets à ajouter ou modifier"""

        context = {"scripts_disponibles": scripts_summary}
        return self.get_ai_analysis(prompt, context)

    def generate_full_report(self) -> str:
        """Génère un rapport complet d'optimisation."""
        print("\n" + "=" * 60)
        print("🧠 RAPPORT D'OPTIMISATION IA")
        print("=" * 60)

        sections = []

        # 1. Analyse du modèle
        print("\n📊 Section 1: Analyse du modèle...")
        sections.append("# 📊 Analyse du Modèle\n" + self.analyze_model())

        # 2. Suggestions de features
        print("\n🔧 Section 2: Nouvelles features...")
        sections.append("\n---\n# 🔧 Nouvelles Features Suggérées\n" + self.suggest_new_features())

        # 3. Hyperparamètres
        print("\n⚙️ Section 3: Hyperparamètres...")
        sections.append(
            "\n---\n# ⚙️ Optimisation Hyperparamètres\n" + self.optimize_hyperparameters()
        )

        report = "\n".join(sections)

        # Sauvegarder le rapport
        report_path = Path("ai_optimization_report.md")
        with open(report_path, "w") as f:
            f.write("# 🧠 Rapport d'Optimisation IA\n")
            f.write(f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
            f.write(report)

        print(f"\n✅ Rapport sauvegardé: {report_path}")
        return report


def main():
    parser = argparse.ArgumentParser(
        description="🧠 AI Model Optimizer - Optimise tes scripts avec Gemini"
    )
    parser.add_argument("--analyze", action="store_true", help="Analyse le modèle actuel")
    parser.add_argument(
        "--suggest-features", action="store_true", help="Suggère de nouvelles features"
    )
    parser.add_argument(
        "--optimize-params", action="store_true", help="Optimise les hyperparamètres"
    )
    parser.add_argument(
        "--analyze-scripts", action="store_true", help="Analyse les scripts de training"
    )
    parser.add_argument("--full-report", action="store_true", help="Génère un rapport complet")

    args = parser.parse_args()

    optimizer = AIModelOptimizer()

    if args.analyze:
        print(optimizer.analyze_model())
    elif args.suggest_features:
        print(optimizer.suggest_new_features())
    elif args.optimize_params:
        print(optimizer.optimize_hyperparameters())
    elif args.analyze_scripts:
        print(optimizer.analyze_scripts_with_ai())
    elif args.full_report:
        print(optimizer.generate_full_report())
    else:
        parser.print_help()
        print("\n💡 Exemple: python ai_model_optimizer.py --full-report")


if __name__ == "__main__":
    main()
