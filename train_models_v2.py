#!/usr/bin/env python3
"""
🚀 Train Models V2 - Réentraînement avec recommandations Gemini
================================================================
Lit les recommandations générées par ai_error_analyzer.py et
réentraîne les modèles avec les améliorations suggérées.

Usage:
    python train_models_v2.py --zone micro
    python train_models_v2.py --all-zones
    python train_models_v2.py --zone full --apply-reco
"""

import os
import sys
import json
import pickle
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️  GOOGLE_API_KEY non définie - mode sans analyse IA")

# Import Gemini si disponible
try:
    from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
    MODEL_NAME = "gemini-2.0-flash"
except ImportError:
    client = None

# Répertoires
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models" / "zones"
V2_DIR = DATA_DIR / "models" / "zones_v2"
V2_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = BASE_DIR / "reports" / "ai_analysis"


class ModelV2Trainer:
    """Entraîne les modèles V2 avec les insights IA."""

    def __init__(self, zone: str):
        self.zone = zone
        self.v1_dir = MODELS_DIR / zone
        self.v2_dir = V2_DIR / zone
        self.v2_dir.mkdir(parents=True, exist_ok=True)

        self.v1_metadata = {}
        self.recommendations = {}
        self.new_features = []
        self.adjusted_params = {}

        print(f"\n{'='*60}")
        print(f"🚀 Modèle V2 - Zone {zone.upper()}")
        print(f"{'='*60}")

    def load_v1_metadata(self) -> Dict[str, Any]:
        """Charge les métadonnées du modèle V1."""
        meta_path = self.v1_dir / "metadata.json"

        if not meta_path.exists():
            print(f"⚠️  Pas de modèle V1 trouvé pour {self.zone}")
            return {}

        with open(meta_path) as f:
            self.v1_metadata = json.load(f)

        print(f"✅ V1 chargé: AUC={self.v1_metadata.get('metrics', {}).get('auc', 'N/A')}")
        return self.v1_metadata

    def load_recommendations(self) -> Optional[str]:
        """Charge les recommandations du dernier rapport d'analyse."""
        # Chercher le dernier rapport
        pattern = f"analysis_{self.zone}_*.md"
        reports = sorted(REPORTS_DIR.glob(pattern), reverse=True)

        if not reports:
            print("⚠️  Pas de rapport d'analyse trouvé")
            return None

        latest = reports[0]
        print(f"📄 Rapport: {latest.name}")

        with open(latest) as f:
            content = f.read()

        # Extraire la section recommandations
        if "## Recommandations V2" in content:
            reco_section = content.split("## Recommandations V2")[1]
            self.recommendations = {"raw": reco_section}
            return reco_section

        return content

    def parse_feature_suggestions(self, recommendations: str) -> List[Dict]:
        """Parse les suggestions de features depuis les recommandations."""
        print("\n🔧 Extraction des features suggérées...")

        # Chercher les blocs de code Python
        code_blocks = re.findall(r"```python\n(.*?)```", recommendations, re.DOTALL)

        features = []
        for code in code_blocks:
            # Identifier les fonctions de feature
            if "def " in code or "df[" in code:
                features.append({"code": code.strip(), "source": "gemini"})

        print(f"   {len(features)} suggestions de code trouvées")
        self.new_features = features
        return features

    def adjust_hyperparameters(self) -> Dict[str, Any]:
        """Ajuste les hyperparamètres basé sur l'analyse."""
        print("\n⚙️ Ajustement des hyperparamètres...")

        # Charger les params V1
        v1_params = self.v1_metadata.get("config", {}).get("xgb_params", {})

        # Améliorer basé sur les recommandations Gemini
        if client and GOOGLE_API_KEY:
            prompt = f"""Basé sur les métriques V1:
AUC: {self.v1_metadata.get('metrics', {}).get('auc', 'N/A')}
Precision: {self.v1_metadata.get('metrics', {}).get('precision', 'N/A')}
Recall: {self.v1_metadata.get('metrics', {}).get('recall', 'N/A')}

Params actuels: {json.dumps(v1_params, indent=2)}

Suggère des ajustements pour améliorer. Réponds en JSON uniquement:
{{"max_depth": int, "learning_rate": float, "n_estimators": int, ...}}"""

            try:
                response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
                # Extraire le JSON
                json_match = re.search(r"\{[^{}]+\}", response.text)
                if json_match:
                    suggested = json.loads(json_match.group())
                    v1_params.update(suggested)
                    print("   ✅ Params ajustés par Gemini")
            except Exception as e:
                print(f"   ⚠️ Erreur Gemini: {e}")

        # Ajustements par défaut pour V2
        v2_adjustments = {
            "learning_rate": v1_params.get("learning_rate", 0.1) * 0.9,  # Légèrement plus bas
            "n_estimators": int(v1_params.get("n_estimators", 300) * 1.2),  # Plus d'arbres
            "reg_alpha": v1_params.get("reg_alpha", 0.5) * 1.2,  # Plus de régularisation
        }

        v1_params.update(v2_adjustments)
        self.adjusted_params = v1_params

        print(f"   learning_rate: {self.adjusted_params.get('learning_rate', 'N/A'):.4f}")
        print(f"   n_estimators: {self.adjusted_params.get('n_estimators', 'N/A')}")

        return self.adjusted_params

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Charge et prépare les données avec nouvelles features."""
        print("\n📊 Préparation des données V2 (table cheval_courses_seen - 812k)...")

        # Charger depuis la BDD avec la VRAIE table historique
        from db_connection import get_connection

        conn = get_connection()
        cur = conn.cursor()

        # Utiliser cheval_courses_seen (812k lignes) au lieu de performances (28k)
        query = """
            SELECT
                nom_norm, race_key, annee, is_win,
                hippodrome_nom, meteo, etat_piste,
                discipline, specialite, distance_m,
                type_piste, type_course, classe_course,
                numero_dossard, age, sexe, poids_kg,
                cote_matin, cote_finale, place_finale,
                nombre_partants, gains_course, gains_carriere,
                temps_sec, reduction_km_sec, vitesse_moyenne,
                non_partant, disqualifie
            FROM cheval_courses_seen
            WHERE place_finale IS NOT NULL
              AND (non_partant IS NULL OR non_partant = 0)
              AND cote_finale IS NOT NULL
              AND cote_finale > 0
            ORDER BY race_key
        """

        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        conn.close()

        df = pd.DataFrame(rows, columns=columns)

        # Convertir les colonnes numériques en float (PostgreSQL renvoie des Decimal)
        numeric_cols = [
            "cote_matin",
            "cote_finale",
            "poids_kg",
            "distance_m",
            "nombre_partants",
            "gains_course",
            "gains_carriere",
            "temps_sec",
            "reduction_km_sec",
            "vitesse_moyenne",
            "age",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # Renommer cote_finale en cote pour compatibilité
        df["cote"] = df["cote_finale"]

        print(f"   Lignes: {len(df):,}")

        # Appliquer les nouvelles features
        df = self.apply_new_features(df)

        # Target - utiliser is_win existant dans cheval_courses_seen
        if "is_win" in df.columns:
            df["is_winner"] = df["is_win"].astype(int)
        elif "place_finale" in df.columns:
            df["is_winner"] = (df["place_finale"] == 1).astype(int)

        # Sélectionner les features numériques
        exclude_cols = [
            "nom_norm",
            "race_key",
            "is_win",
            "is_winner",
            "place_finale",
            "hippodrome_nom",
            "discipline",
            "specialite",
            "type_piste",
            "type_course",
            "classe_course",
            "sexe",
            "meteo",
            "etat_piste",
            "cote_finale",
            "non_partant",
            "disqualifie",
        ]

        feature_cols = [
            c
            for c in df.columns
            if c not in exclude_cols and df[c].dtype in ["int64", "float64", "int32", "float32"]
        ]

        # Remplir les NaN avec la médiane
        df_subset = df[feature_cols + ["is_winner"]].copy()
        for col in feature_cols:
            if df_subset[col].isna().any():
                median_val = df_subset[col].median()
                df_subset[col] = df_subset[col].fillna(median_val if pd.notna(median_val) else 0)

        df_clean = df_subset.dropna(subset=["is_winner"])

        X = df_clean[feature_cols].values
        y = df_clean["is_winner"].values

        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(y):,}")
        print(f"   Features V2: {feature_cols}")

        return X, y, feature_cols

    def apply_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique les nouvelles features optimisées pour V2."""
        print("   Ajout des features V2 optimisées...")

        added_features = []

        # 1. Proba implicite (recommandé par Gemini)
        if "cote" in df.columns:
            df["proba_implicite"] = 1 / df["cote"].replace(0, 1)
            added_features.append("proba_implicite")

        # 2. Cote relative - cote par rapport à la moyenne de la course
        if "cote" in df.columns and "race_key" in df.columns:
            mean_odds = df.groupby("race_key")["cote"].transform("mean")
            df["cote_relative"] = df["cote"] / mean_odds.replace(0, 1)
            added_features.append("cote_relative")

        # 3. Avantage cote - écart à la moyenne
        if "cote" in df.columns and "race_key" in df.columns:
            mean_odds = df.groupby("race_key")["cote"].transform("mean")
            df["avantage_cote"] = (mean_odds - df["cote"]) / mean_odds.replace(0, 1)
            added_features.append("avantage_cote")

        # 4. Discipline encoded - Trot=0, Plat=1, Obstacle=2
        if "discipline" in df.columns:
            discipline_map = {
                "TROT_ATTELE": 0,
                "TROT_MONTE": 0,
                "TROT": 0,
                "trot": 0,
                "PLAT": 1,
                "plat": 1,
                "galop": 1,
                "GALOP": 1,
                "OBSTACLE": 2,
                "obstacle": 2,
                "HAIES": 2,
                "STEEPLECHASE": 2,
            }
            df["discipline_encoded"] = df["discipline"].map(discipline_map).fillna(1)
            added_features.append("discipline_encoded")

        # 5. Risk level (catégorisation de la cote)
        if "cote" in df.columns:
            conditions = [(df["cote"] <= 2), (df["cote"] <= 4), (df["cote"] <= 8), (df["cote"] > 8)]
            choices = [1, 2, 3, 4]
            df["risk_level"] = np.select(conditions, choices, default=2).astype(float)
            added_features.append("risk_level")

        # 6. Compétitivité (1/nombre_partants)
        if "nombre_partants" in df.columns:
            df["competitivite"] = 1 / df["nombre_partants"].replace(0, 1)
            added_features.append("competitivite")

        # 7. Gains relatifs (gains_course / gains_carriere)
        if "gains_course" in df.columns and "gains_carriere" in df.columns:
            df["gains_ratio"] = df["gains_course"] / df["gains_carriere"].replace(0, 1)
            added_features.append("gains_ratio")

        # 8. Performance vitesse (vitesse_moyenne normalisée)
        if "vitesse_moyenne" in df.columns:
            df["perf_vitesse"] = df["vitesse_moyenne"].fillna(0)
            added_features.append("perf_vitesse")

        print(f"   ✅ {len(added_features)} features V2 ajoutées: {added_features}")

        return df

    def train_v2(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Entraîne le modèle V2."""
        print("\n🚀 Entraînement V2...")

        # Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Scaler
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # Params V2
        params = self.adjusted_params.copy()
        params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "use_label_encoder": False,
                "random_state": 42,
            }
        )

        # Entraînement
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)

        # Évaluation
        y_pred_proba = model.predict_proba(X_val_s)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "auc": roc_auc_score(y_val, y_pred_proba),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
        }

        # Comparer avec V1
        v1_auc = self.v1_metadata.get("metrics", {}).get("auc", 0)
        improvement = ((metrics["auc"] - v1_auc) / v1_auc * 100) if v1_auc > 0 else 0

        print("\n📈 Métriques V2:")
        print(f"   AUC: {metrics['auc']:.4f} (V1: {v1_auc:.4f}, {improvement:+.1f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1: {metrics['f1']:.4f}")

        # Sauvegarder
        self._save_model(model, scaler, feature_names, metrics)

        return {
            "success": True,
            "metrics_v2": metrics,
            "metrics_v1": self.v1_metadata.get("metrics", {}),
            "improvement_auc": improvement,
        }

    def _save_model(self, model, scaler, feature_names, metrics):
        """Sauvegarde le modèle V2."""
        print(f"\n💾 Sauvegarde dans {self.v2_dir}...")

        # Modèle
        with open(self.v2_dir / "xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Scaler
        with open(self.v2_dir / "feature_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # Features
        with open(self.v2_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

        # Metadata
        metadata = {
            "zone": self.zone,
            "version": "v2.0",
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics,
            "based_on": "v1.0",
            "improvements": self.new_features,
            "params": self.adjusted_params,
        }
        with open(self.v2_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print("   ✅ Modèle V2 sauvegardé")

    def run(self) -> Dict[str, Any]:
        """Exécute le pipeline complet."""
        # Charger V1
        self.load_v1_metadata()

        # Charger recommandations
        reco = self.load_recommendations()
        if reco:
            self.parse_feature_suggestions(reco)

        # Ajuster les params
        self.adjust_hyperparameters()

        # Préparer les données
        X, y, features = self.load_and_prepare_data()

        # Entraîner V2
        result = self.train_v2(X, y, features)

        return result


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Train Models V2 - Réentraînement avec insights IA"
    )
    parser.add_argument("--zone", choices=["micro", "small", "full"], help="Zone à réentraîner")
    parser.add_argument("--all-zones", action="store_true", help="Réentraîne toutes les zones")

    args = parser.parse_args()

    if args.all_zones:
        for zone in ["micro", "small", "full"]:
            trainer = ModelV2Trainer(zone)
            trainer.run()
    elif args.zone:
        trainer = ModelV2Trainer(args.zone)
        result = trainer.run()
        print(f"\n✅ Résultat: {json.dumps(result, indent=2)}")
    else:
        parser.print_help()
        print("\n💡 Exemple: python train_models_v2.py --zone full")


if __name__ == "__main__":
    main()
