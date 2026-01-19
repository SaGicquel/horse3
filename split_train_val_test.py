#!/usr/bin/env python3
"""
================================================================================
SPLIT TRAIN/VAL/TEST - PHASE 4
================================================================================

Description : Découpe temporelle des données ML en train/validation/test

Split :
  • 70% Train (courses les plus anciennes)
  • 15% Validation
  • 15% Test (courses les plus récentes)

Le split est TEMPOREL pour éviter le data leakage :
  - Train : apprend sur historique ancien
  - Val : tune sur période intermédiaire
  - Test : évalue sur courses futures (simulation production)

Usage :
  python split_train_val_test.py --input data/ml_features_complete.csv

================================================================================
"""

import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Gère le découpage des données en train/val/test"""

    def __init__(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.001:
            raise ValueError("Les ratios doivent sommer à 1.0")

    def split_temporal(self, df: pd.DataFrame) -> tuple:
        """
        Split temporel basé sur date_course

        Args:
            df: DataFrame avec colonne 'date_course'

        Returns:
            (df_train, df_val, df_test)
        """
        logger.info("=" * 80)
        logger.info("📊 SPLIT TEMPOREL TRAIN/VAL/TEST")
        logger.info("=" * 80)
        logger.info("")

        # Vérifier colonne date
        if "date_course" not in df.columns:
            raise ValueError("Colonne 'date_course' manquante")

        # Convertir en datetime si besoin
        if df["date_course"].dtype == "object":
            df["date_course"] = pd.to_datetime(df["date_course"])

        # Trier par date
        df_sorted = df.sort_values("date_course").reset_index(drop=True)

        # Calculer indices de split
        n = len(df_sorted)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        # Découper
        df_train = df_sorted.iloc[:train_end].copy()
        df_val = df_sorted.iloc[train_end:val_end].copy()
        df_test = df_sorted.iloc[val_end:].copy()

        # Statistiques
        logger.info("📈 TAILLES:")
        logger.info(f"   Train      : {len(df_train):6,} lignes ({100*len(df_train)/n:.1f}%)")
        logger.info(f"   Validation : {len(df_val):6,} lignes ({100*len(df_val)/n:.1f}%)")
        logger.info(f"   Test       : {len(df_test):6,} lignes ({100*len(df_test)/n:.1f}%)")
        logger.info(f"   TOTAL      : {n:6,} lignes")
        logger.info("")

        logger.info("📅 PÉRIODES:")
        logger.info(
            f"   Train      : {df_train['date_course'].min()} → {df_train['date_course'].max()}"
        )
        logger.info(
            f"   Validation : {df_val['date_course'].min()} → {df_val['date_course'].max()}"
        )
        logger.info(
            f"   Test       : {df_test['date_course'].min()} → {df_test['date_course'].max()}"
        )
        logger.info("")

        # Distribution target
        logger.info("🎯 DISTRIBUTION VICTOIRES:")
        logger.info(
            f"   Train      : {df_train['victoire'].sum():5,} / {len(df_train):6,} ({100*df_train['victoire'].mean():.2f}%)"
        )
        logger.info(
            f"   Validation : {df_val['victoire'].sum():5,} / {len(df_val):6,} ({100*df_val['victoire'].mean():.2f}%)"
        )
        logger.info(
            f"   Test       : {df_test['victoire'].sum():5,} / {len(df_test):6,} ({100*df_test['victoire'].mean():.2f}%)"
        )
        logger.info("")

        return df_train, df_val, df_test

    def save_splits(self, df_train, df_val, df_test, output_dir: str):
        """Sauvegarde les splits en CSV"""
        logger.info("💾 SAUVEGARDE:")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sauvegarder CSV
        train_path = output_path / "train.csv"
        val_path = output_path / "val.csv"
        test_path = output_path / "test.csv"

        df_train.to_csv(train_path, index=False)
        logger.info(f"   ✅ {train_path}")

        df_val.to_csv(val_path, index=False)
        logger.info(f"   ✅ {val_path}")

        df_test.to_csv(test_path, index=False)
        logger.info(f"   ✅ {test_path}")
        logger.info("")

        # Metadata JSON
        metadata = {
            "created_at": datetime.now().isoformat(),
            "n_features": len(df_train.columns),
            "train": {
                "n_samples": len(df_train),
                "n_victoires": int(df_train["victoire"].sum()),
                "taux_victoires": float(df_train["victoire"].mean()),
                "date_min": df_train["date_course"].min().isoformat(),
                "date_max": df_train["date_course"].max().isoformat(),
                "file": str(train_path.name),
            },
            "val": {
                "n_samples": len(df_val),
                "n_victoires": int(df_val["victoire"].sum()),
                "taux_victoires": float(df_val["victoire"].mean()),
                "date_min": df_val["date_course"].min().isoformat(),
                "date_max": df_val["date_course"].max().isoformat(),
                "file": str(val_path.name),
            },
            "test": {
                "n_samples": len(df_test),
                "n_victoires": int(df_test["victoire"].sum()),
                "taux_victoires": float(df_test["victoire"].mean()),
                "date_min": df_test["date_course"].min().isoformat(),
                "date_max": df_test["date_course"].max().isoformat(),
                "file": str(test_path.name),
            },
            "ratios": {"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio},
        }

        metadata_path = output_path / "split_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"📄 Metadata : {metadata_path}")
        logger.info("")


def main():
    parser = argparse.ArgumentParser(description="Split train/val/test temporel")
    parser.add_argument("--input", required=True, help="Fichier CSV avec features")
    parser.add_argument("--output-dir", default="data/splits", help="Répertoire de sortie")
    parser.add_argument(
        "--train-ratio", type=float, default=0.70, help="Ratio train (défaut: 0.70)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Ratio validation (défaut: 0.15)"
    )
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Ratio test (défaut: 0.15)")

    args = parser.parse_args()

    try:
        # Charger données
        logger.info(f"📂 Chargement : {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"   ✅ {len(df):,} lignes, {len(df.columns)} colonnes")
        logger.info("")

        # Split
        splitter = DataSplitter(
            train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio
        )

        df_train, df_val, df_test = splitter.split_temporal(df)

        # Sauvegarder
        splitter.save_splits(df_train, df_val, df_test, args.output_dir)

        logger.info("=" * 80)
        logger.info("✅ SPLIT TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        raise


if __name__ == "__main__":
    main()
