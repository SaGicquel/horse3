#!/usr/bin/env python3
"""
🎯 Train Zone Models - Entraînement de 3 modèles par zone de bankroll
======================================================================
Entraîne des modèles XGBoost optimisés pour chaque zone:
- MICRO (<50€): Paris sûrs, cotes basses, minimiser pertes
- SMALL (50-500€): Équilibre risque/rendement
- FULL (>500€): ROI maximum, toutes cotes

Usage:
    python train_zone_models.py --zone micro
    python train_zone_models.py --zone small
    python train_zone_models.py --zone full
    python train_zone_models.py --all  # Entraîne les 3
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss

# Configuration des répertoires
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models" / "zones"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration des zones
ZONE_CONFIGS = {
    "micro": {
        "name": "MICRO",
        "description": "Bankroll <50€ - Paris très sûrs",
        "max_odds": 4.0,  # Cotes max
        "min_proba": 0.25,  # Proba min (25%)
        "bet_types": ["PLACE"],  # Uniquement placé
        "objective": "minimize_loss",
        "xgb_params": {
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_weight": 10,
            "gamma": 0.5,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "scale_pos_weight": 1.5,
        },
    },
    "small": {
        "name": "SMALL",
        "description": "Bankroll 50-500€ - Équilibre risque/rendement",
        "max_odds": 8.0,
        "min_proba": 0.15,
        "bet_types": ["PLACE", "E_P"],
        "objective": "balanced_roi",
        "xgb_params": {
            "max_depth": 6,
            "learning_rate": 0.08,
            "n_estimators": 300,
            "min_child_weight": 5,
            "gamma": 0.3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "scale_pos_weight": 2.0,
        },
    },
    "full": {
        "name": "FULL",
        "description": "Bankroll >500€ - ROI maximum",
        "max_odds": 15.0,
        "min_proba": 0.08,
        "bet_types": ["PLACE", "E_P", "WIN"],
        "objective": "maximize_roi",
        "xgb_params": {
            "max_depth": 8,
            "learning_rate": 0.12,
            "n_estimators": 400,
            "min_child_weight": 3,
            "gamma": 0.2,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.3,
            "reg_lambda": 0.5,
            "scale_pos_weight": 3.0,
        },
    },
}


class ZoneModelTrainer:
    """Entraîneur de modèle pour une zone spécifique."""

    def __init__(self, zone: str):
        if zone not in ZONE_CONFIGS:
            raise ValueError(f"Zone inconnue: {zone}. Choix: {list(ZONE_CONFIGS.keys())}")

        self.zone = zone
        self.config = ZONE_CONFIGS[zone]
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.metrics = {}

        # Répertoire de sortie
        self.output_dir = MODELS_DIR / zone
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"🎯 Zone: {self.config['name']} - {self.config['description']}")
        print(f"{'='*60}")

    def load_data(self) -> pd.DataFrame:
        """Charge les données d'entraînement."""
        print("\n📊 Chargement des données...")

        # Chercher les fichiers de données
        possible_files = [
            DATA_DIR / "training_data.parquet",
            DATA_DIR / "ml_features.parquet",
            DATA_DIR / "features_ml.csv",
            BASE_DIR / "training_data.csv",
        ]

        data_file = None
        for f in possible_files:
            if f.exists():
                data_file = f
                break

        if data_file is None:
            # Essayer de charger depuis la base PostgreSQL
            print("📥 Chargement depuis PostgreSQL...")
            return self._load_from_db()

        print(f"   Fichier: {data_file}")

        if data_file.suffix == ".parquet":
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_csv(data_file)

        print(f"   Lignes: {len(df):,}")
        return df

    def _load_from_db(self) -> pd.DataFrame:
        """Charge les données depuis PostgreSQL."""
        try:
            from db_connection import get_connection
            import psycopg2

            conn = get_connection()
            cur = conn.cursor()

            # Utiliser le bon schéma: performances, courses, chevaux
            query = """
                SELECT
                    p.id_performance, p.id_course, p.id_cheval, p.id_jockey, p.id_entraineur,
                    p.numero_corde, p.numero_dossard, p.poids_porte,
                    p.cote_pm, p.cote_sp, p.position_arrivee, p.place,
                    p.disqualifie, p.non_partant,
                    c.date_course as date, c.discipline, c.distance, c.nombre_partants,
                    h.nom_hippodrome as hippodrome,
                    ch.nom_cheval as cheval_nom,
                    ch.sexe_cheval as sexe,
                    EXTRACT(YEAR FROM NOW()) - ch.an_naissance as age,
                    ch.forme_recente_30j, ch.forme_recente_60j,
                    ch.nombre_victoires_total, ch.nombre_courses_total
                FROM performances p
                JOIN courses c ON p.id_course = c.id_course
                LEFT JOIN chevaux ch ON p.id_cheval = ch.id_cheval
                LEFT JOIN hippodromes h ON c.id_hippodrome = h.id_hippodrome
                WHERE c.date_course >= NOW() - INTERVAL '2 years'
                  AND p.place IS NOT NULL
                  AND p.non_partant = false
                ORDER BY c.date_course
            """

            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            conn.close()

            df = pd.DataFrame(rows, columns=columns)

            # Renommer cote_sp en cote pour compatibilité
            if "cote_sp" in df.columns:
                df["cote"] = df["cote_sp"]

            print(f"   Lignes chargées: {len(df):,}")
            return df
        except Exception as e:
            print(f"❌ Erreur DB: {e}")
            raise

    def filter_for_zone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtre les données selon la configuration de la zone."""
        print(f"\n🔍 Filtrage pour zone {self.config['name']}...")

        initial_count = len(df)

        # Filtrer par cote max
        if "cote" in df.columns:
            df = df[df["cote"] <= self.config["max_odds"]]
            print(f"   Après filtre cotes ≤{self.config['max_odds']}: {len(df):,}")

        # Filtrer par probabilité implicite min (1/cote)
        if "cote" in df.columns:
            df = df[(1 / df["cote"]) >= self.config["min_proba"]]
            print(f"   Après filtre proba ≥{self.config['min_proba']*100:.0f}%: {len(df):,}")

        print(f"   Final: {len(df):,} lignes ({len(df)/initial_count*100:.1f}% des données)")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prépare les features et la target."""
        print("\n🔧 Préparation des features...")

        # Définir la target selon l'objectif
        if self.config["objective"] == "minimize_loss":
            # Pour micro: target = gagnant (car les favoris sont souvent placés)
            target_col = "is_winner"
            if target_col not in df.columns:
                df[target_col] = (df["place"] == 1).astype(int)
        else:
            # Pour small/full: target = gagnant
            target_col = "is_winner"
            if target_col not in df.columns:
                df[target_col] = (df["place"] == 1).astype(int)

        print(f"   Target: {target_col}")
        print(f"   Distribution: {df[target_col].value_counts().to_dict()}")

        # Sélectionner les features numériques
        exclude_cols = [
            "id_performance",
            "id_course",
            "id_cheval",
            "id_jockey",
            "id_entraineur",
            "place",
            "position_arrivee",
            "is_winner",
            "is_place",
            "date",
            "nom",
            "cheval_nom",
            "hippodrome",
            "discipline",
            "sexe",
            "disqualifie",
            "non_partant",
        ]

        feature_cols = [
            c
            for c in df.columns
            if c not in exclude_cols and df[c].dtype in ["int64", "float64", "int32", "float32"]
        ]

        # Nettoyer les données avec fillna au lieu de dropna
        df_subset = df[feature_cols + [target_col]].copy()

        # Remplir les NaN avec des valeurs neutres
        for col in feature_cols:
            if df_subset[col].isna().any():
                median_val = df_subset[col].median()
                df_subset[col] = df_subset[col].fillna(median_val if pd.notna(median_val) else 0)

        # Supprimer seulement les lignes où la target est manquante
        df_clean = df_subset.dropna(subset=[target_col])

        X = df_clean[feature_cols].values
        y = df_clean[target_col].values

        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(y):,}")
        print(f"   Features utilisées: {feature_cols}")

        self.feature_names = feature_cols
        return X, y, feature_cols

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entraîne le modèle XGBoost."""
        print(f"\n🚀 Entraînement du modèle {self.config['name']}...")

        # Split temporel
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False,  # Pas de shuffle pour respecter l'ordre temporel
        )

        print(f"   Train: {len(y_train):,} | Val: {len(y_val):,}")

        # Scaler
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Paramètres XGBoost
        params = self.config["xgb_params"].copy()
        params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "use_label_encoder": False,
                "random_state": 42,
                "n_jobs": -1,
            }
        )

        # Entraînement
        self.model = xgb.XGBClassifier(**params)

        self.model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

        # Évaluation
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        self.metrics = {
            "auc": roc_auc_score(y_val, y_pred_proba),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "brier": brier_score_loss(y_val, y_pred_proba),
            "train_samples": len(y_train),
            "val_samples": len(y_val),
        }

        print(f"\n📈 Métriques {self.config['name']}:")
        print(f"   AUC: {self.metrics['auc']:.4f}")
        print(f"   Precision: {self.metrics['precision']:.4f}")
        print(f"   Recall: {self.metrics['recall']:.4f}")
        print(f"   F1: {self.metrics['f1']:.4f}")
        print(f"   Brier: {self.metrics['brier']:.4f}")

    def save(self) -> Dict[str, Path]:
        """Sauvegarde le modèle et les artefacts."""
        print(f"\n💾 Sauvegarde dans {self.output_dir}...")

        paths = {}

        # Modèle
        model_path = self.output_dir / "xgboost_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        paths["model"] = model_path

        # Scaler
        scaler_path = self.output_dir / "feature_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        paths["scaler"] = scaler_path

        # Feature names
        features_path = self.output_dir / "feature_names.json"
        with open(features_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)
        paths["features"] = features_path

        # Metadata
        metadata = {
            "zone": self.zone,
            "name": self.config["name"],
            "description": self.config["description"],
            "version": "v1.0",
            "trained_at": datetime.now().isoformat(),
            "metrics": self.metrics,
            "config": self.config,
            "n_features": len(self.feature_names),
        }
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        paths["metadata"] = meta_path

        print(f"   ✅ Modèle: {model_path.name}")
        print(f"   ✅ Scaler: {scaler_path.name}")
        print(f"   ✅ Features: {features_path.name}")
        print(f"   ✅ Metadata: {meta_path.name}")

        return paths

    def run(self) -> Dict[str, Any]:
        """Exécute le pipeline complet d'entraînement."""
        # Charger les données
        df = self.load_data()

        # Filtrer pour la zone
        df_filtered = self.filter_for_zone(df)

        if len(df_filtered) < 1000:
            print(f"⚠️  Pas assez de données pour la zone {self.zone} ({len(df_filtered)} < 1000)")
            return {"success": False, "reason": "insufficient_data"}

        # Préparer les features
        X, y, feature_names = self.prepare_features(df_filtered)

        # Entraîner
        self.train(X, y)

        # Sauvegarder
        paths = self.save()

        return {
            "success": True,
            "zone": self.zone,
            "metrics": self.metrics,
            "paths": {k: str(v) for k, v in paths.items()},
        }


def train_all_zones() -> Dict[str, Any]:
    """Entraîne les modèles pour toutes les zones."""
    print("\n" + "=" * 60)
    print("🎯 ENTRAÎNEMENT DES 3 MODÈLES PAR ZONE")
    print("=" * 60)

    results = {}

    for zone in ["micro", "small", "full"]:
        try:
            trainer = ZoneModelTrainer(zone)
            result = trainer.run()
            results[zone] = result
        except Exception as e:
            print(f"❌ Erreur zone {zone}: {e}")
            results[zone] = {"success": False, "error": str(e)}

    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)

    for zone, result in results.items():
        if result.get("success"):
            metrics = result["metrics"]
            print(f"   {zone.upper()}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
        else:
            print(f"   {zone.upper()}: ❌ {result.get('reason', result.get('error', 'unknown'))}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="🎯 Train Zone Models - Entraîne des modèles par zone de bankroll"
    )
    parser.add_argument("--zone", choices=["micro", "small", "full"], help="Zone à entraîner")
    parser.add_argument("--all", action="store_true", help="Entraîne les 3 zones")

    args = parser.parse_args()

    if args.all:
        train_all_zones()
    elif args.zone:
        trainer = ZoneModelTrainer(args.zone)
        trainer.run()
    else:
        parser.print_help()
        print("\n💡 Exemple: python train_zone_models.py --all")


if __name__ == "__main__":
    main()
