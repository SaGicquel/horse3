#!/usr/bin/env python3
"""
🔍 AI Error Analyzer - Analyse les erreurs de prédiction avec Gemini
=====================================================================
Charge les prédictions passées, identifie les erreurs, et utilise Gemini
pour analyser pourquoi le modèle s'est trompé.

Usage:
    python ai_error_analyzer.py --zone micro
    python ai_error_analyzer.py --zone full --limit 50
    python ai_error_analyzer.py --all-zones
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY non définie")
    sys.exit(1)

# Import Gemini
from google import genai

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# Répertoires
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models" / "zones"
REPORTS_DIR = BASE_DIR / "reports" / "ai_analysis"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class AIErrorAnalyzer:
    """Analyse les erreurs de prédiction avec Gemini."""

    def __init__(self, zone: str):
        self.zone = zone
        self.zone_dir = MODELS_DIR / zone
        self.errors = []
        self.insights = []

        print(f"\n{'='*60}")
        print(f"🔍 Analyse des erreurs - Zone {zone.upper()}")
        print(f"{'='*60}")

    def load_predictions(self, limit: int = 100) -> pd.DataFrame:
        """Charge les prédictions récentes avec leurs résultats."""
        print("\n📊 Chargement des prédictions...")

        # Essayer de charger depuis un fichier de prédictions
        pred_files = [
            DATA_DIR / f"predictions_{self.zone}.parquet",
            DATA_DIR / "predictions_history.parquet",
            DATA_DIR / "betting_history.csv",
        ]

        for f in pred_files:
            if f.exists():
                print(f"   Fichier: {f}")
                if f.suffix == ".parquet":
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f)
                return df.tail(limit)

        # Sinon charger depuis la BDD
        return self._load_from_db(limit)

    def _load_from_db(self, limit: int) -> pd.DataFrame:
        """Charge les prédictions depuis PostgreSQL."""
        try:
            from db_connection import get_connection

            conn = get_connection()
            query = f"""
                SELECT
                    b.id, b.created_at, b.selection, b.bet_type,
                    b.stake, b.odds, b.status, b.pnl,
                    b.race_key, b.hippodrome
                FROM user_bets b
                WHERE b.status IN ('WIN', 'LOSE')
                ORDER BY b.created_at DESC
                LIMIT {limit}
            """
            df = pd.read_sql(query, conn)
            conn.close()
            print(f"   Chargés: {len(df)} paris")
            return df
        except Exception as e:
            print(f"⚠️  Pas de données en BDD: {e}")
            return self._generate_sample_data(limit)

    def _generate_sample_data(self, n: int = 50) -> pd.DataFrame:
        """Génère des données de test si pas de vraies données."""
        print("   ⚠️  Génération de données de test...")

        np.random.seed(42)

        data = []
        for i in range(n):
            is_win = np.random.random() < 0.35  # 35% win rate simulé
            odds = np.random.uniform(1.5, 8.0)
            stake = np.random.choice([5, 10, 15, 20, 25])
            predicted_proba = np.random.uniform(0.1, 0.4)

            data.append(
                {
                    "id": i + 1,
                    "date": f"2024-12-{(i % 30) + 1:02d}",
                    "cheval": f"Cheval_{np.random.randint(1, 100)}",
                    "hippodrome": np.random.choice(
                        ["Vincennes", "Longchamp", "Auteuil", "Chantilly"]
                    ),
                    "discipline": np.random.choice(["Trot", "Galop", "Obstacle"]),
                    "bet_type": np.random.choice(["PLACE", "WIN", "E_P"]),
                    "cote": round(odds, 2),
                    "stake": stake,
                    "status": "WIN" if is_win else "LOSE",
                    "pnl": round(stake * (odds - 1) if is_win else -stake, 2),
                    "predicted_proba": round(predicted_proba, 3),
                    "actual_place": np.random.randint(1, 15),
                    "participants_count": np.random.randint(8, 16),
                    "favoris_rank": np.random.randint(1, 10),
                    "meteo": np.random.choice(["Soleil", "Nuageux", "Pluie", "Orageux"]),
                    "terrain": np.random.choice(["Bon", "Souple", "Lourd", "Très lourd"]),
                }
            )

        df = pd.DataFrame(data)
        print(f"   Générés: {len(df)} paris simulés")
        return df

    def identify_errors(self, df: pd.DataFrame) -> List[Dict]:
        """Identifie les erreurs de prédiction."""
        print("\n🔍 Identification des erreurs...")

        # Filtrer les erreurs (prédit gagnant mais perdu)
        if "status" in df.columns:
            losses = df[df["status"] == "LOSE"].copy()
        else:
            losses = df[df["pnl"] < 0].copy()

        print(f"   Total paris: {len(df)}")
        print(f"   Erreurs (pertes): {len(losses)}")

        # Convertir en liste de dicts
        self.errors = losses.to_dict("records")
        return self.errors

    def analyze_with_gemini(self, errors: List[Dict], batch_size: int = 10) -> str:
        """Analyse les erreurs avec Gemini."""
        print(f"\n🧠 Analyse avec Gemini ({len(errors)} erreurs)...")

        # Prendre un échantillon pour ne pas dépasser les limites
        sample = errors[:batch_size]

        # Formatter les erreurs pour Gemini
        errors_text = json.dumps(sample, indent=2, ensure_ascii=False, default=str)

        prompt = f"""Tu es un expert en analyse de données de courses hippiques.

Voici {len(sample)} paris PERDANTS (le modèle a prédit que ces chevaux seraient gagnants/placés mais ils ont perdu):

```json
{errors_text}
```

Analyse ces erreurs et fournis:

## 1. Patterns Communs
Quels points communs ont ces paris perdants? (météo, terrain, cotes, discipline, etc.)

## 2. Causes Probables
Pour chaque type d'erreur, explique pourquoi le modèle a pu se tromper.

## 3. Features Manquantes
Quelles informations auraient pu aider à éviter ces erreurs?

## 4. Recommandations Concrètes
Liste 5 améliorations spécifiques à apporter au modèle.

## 5. Code Suggéré
Propose du code Python pour ajouter une feature qui aurait pu détecter ces erreurs.

Sois concis et actionnable."""

        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text
        except Exception as e:
            return f"Erreur Gemini: {e}"

    def analyze_winning_patterns(self, df: pd.DataFrame) -> str:
        """Analyse les patterns des paris gagnants."""
        print("\n✅ Analyse des paris gagnants...")

        if "status" in df.columns:
            wins = df[df["status"] == "WIN"].copy()
        else:
            wins = df[df["pnl"] > 0].copy()

        if len(wins) == 0:
            return "Pas de paris gagnants à analyser."

        sample = wins.head(10).to_dict("records")
        wins_text = json.dumps(sample, indent=2, ensure_ascii=False, default=str)

        prompt = f"""Voici {len(sample)} paris GAGNANTS:

```json
{wins_text}
```

Analyse ces succès et identifie:

## 1. Patterns de Succès
Quels facteurs ont contribué à ces victoires?

## 2. Conditions Favorables
Dans quelles conditions le modèle performe le mieux?

## 3. Features Importantes
Quelles features semblent les plus prédictives?

Sois concis."""

        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text
        except Exception as e:
            return f"Erreur Gemini: {e}"

    def generate_v2_recommendations(self, error_analysis: str, win_analysis: str) -> str:
        """Génère des recommandations pour le modèle V2."""
        print("\n🚀 Génération des recommandations V2...")

        prompt = f"""Basé sur ces analyses de paris hippiques:

## ANALYSE DES ERREURS:
{error_analysis[:2000]}

## ANALYSE DES SUCCÈS:
{win_analysis[:2000]}

Génère un plan d'amélioration pour le modèle V2:

## 1. Nouvelles Features à Créer
Liste 5 features avec leur code Python.

## 2. Modifications des Hyperparamètres
Quels ajustements faire à XGBoost?

## 3. Filtres à Ajouter
Quels paris éviter selon les patterns d'erreur?

## 4. Pondération
Comment ajuster les poids des features?

## 5. Code d'Implémentation
Fournis le code Python complet pour implémenter ces améliorations.

Sois très concret et donne du code exécutable."""

        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text
        except Exception as e:
            return f"Erreur Gemini: {e}"

    def save_report(self, error_analysis: str, win_analysis: str, v2_reco: str) -> Path:
        """Sauvegarde le rapport d'analyse."""
        report_path = (
            REPORTS_DIR / f"analysis_{self.zone}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        )

        content = f"""# 🔍 Analyse IA - Zone {self.zone.upper()}
*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}*

---

## Analyse des Erreurs
{error_analysis}

---

## Analyse des Succès
{win_analysis}

---

## Recommandations V2
{v2_reco}
"""

        with open(report_path, "w") as f:
            f.write(content)

        print(f"\n💾 Rapport sauvegardé: {report_path}")
        return report_path

    def run(self, limit: int = 100) -> Dict[str, Any]:
        """Exécute l'analyse complète."""
        # Charger les prédictions
        df = self.load_predictions(limit)

        if df is None or len(df) == 0:
            return {"success": False, "reason": "no_data"}

        # Identifier les erreurs
        errors = self.identify_errors(df)

        # Analyser les erreurs avec Gemini
        print("\n" + "-" * 40)
        error_analysis = self.analyze_with_gemini(errors)
        print(error_analysis[:500] + "..." if len(error_analysis) > 500 else error_analysis)

        # Analyser les succès
        print("\n" + "-" * 40)
        win_analysis = self.analyze_winning_patterns(df)

        # Générer recommandations V2
        print("\n" + "-" * 40)
        v2_recommendations = self.generate_v2_recommendations(error_analysis, win_analysis)

        # Sauvegarder le rapport
        report_path = self.save_report(error_analysis, win_analysis, v2_recommendations)

        return {
            "success": True,
            "zone": self.zone,
            "errors_count": len(errors),
            "report_path": str(report_path),
            "v2_recommendations": v2_recommendations[:1000],
        }


def main():
    parser = argparse.ArgumentParser(
        description="🔍 AI Error Analyzer - Analyse les erreurs avec Gemini"
    )
    parser.add_argument("--zone", choices=["micro", "small", "full"], help="Zone à analyser")
    parser.add_argument("--all-zones", action="store_true", help="Analyse toutes les zones")
    parser.add_argument("--limit", type=int, default=100, help="Nombre de paris à analyser")

    args = parser.parse_args()

    if args.all_zones:
        for zone in ["micro", "small", "full"]:
            analyzer = AIErrorAnalyzer(zone)
            analyzer.run(args.limit)
    elif args.zone:
        analyzer = AIErrorAnalyzer(args.zone)
        result = analyzer.run(args.limit)
        print(f"\n✅ Analyse terminée: {result}")
    else:
        parser.print_help()
        print("\n💡 Exemple: python ai_error_analyzer.py --zone full")


if __name__ == "__main__":
    main()
