#!/usr/bin/env python3
"""
================================================================================
NORMALIZE FEATURES - PHASE 4
================================================================================

Description : Normalisation des features ML avec StandardScaler

Pipeline :
  1. Charge train/val/test splits
  2. Identifie features numériques à normaliser
  3. Fit StandardScaler sur train UNIQUEMENT
  4. Transform train/val/test
  5. Sauvegarde scaler.pkl pour production
  6. Export CSV/Parquet/NumPy

Features normalisées :
  • Forme (forme_5c, nb_courses_12m, etc.)
  • Aptitudes (aptitude_distance, aptitude_piste, etc.)
  • Relationnelles (taux_victoires_jockey, etc.)
  • Gains (gains_carriere, gains_12m, etc.)
  • Interactions
  
Features NON normalisées :
  • One-hot encoded (déjà 0/1)
  • Identifiants (id_cheval, id_course, etc.)
  • Dates
  • Target (victoire, place)

Usage :
  python normalize_features.py --input-dir data/splits --output-dir data/normalized

================================================================================
"""

import argparse
import logging
import pickle
import json
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """Gère la normalisation des features ML"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.numeric_features = []
        self.categorical_features = []
        self.excluded_features = []
    
    def identify_features(self, df: pd.DataFrame):
        """Identifie quelles colonnes normaliser"""
        logger.info("🔍 Identification des features...")
        
        # Features à exclure (ID, dates, target, écart avec "tête")
        exclude_patterns = [
            'id_', 'date_', 'victoire', 'place', 'position_arrivee',
            'non_partant', 'nom_', 'musique', 'hippodrome_ville', 'ecart'
        ]
        
        # One-hot encoded (déjà normalisés 0/1)
        onehot_patterns = [
            'discipline_', 'sexe_', 'piste_', 'etat_', 'hippodrome_'
        ]
        
        for col in df.columns:
            # Exclure ID/dates/target
            if any(col.startswith(pat) or pat in col for pat in exclude_patterns):
                self.excluded_features.append(col)
            # One-hot déjà normalisés
            elif any(col.startswith(pat) for pat in onehot_patterns):
                self.categorical_features.append(col)
            # Features numériques à normaliser - VÉRIFIER que vraiment numérique
            elif df[col].dtype in ['int64', 'float64']:
                # Double vérification: vérifier qu'il n'y a pas de strings cachés
                try:
                    pd.to_numeric(df[col], errors='raise')
                    self.numeric_features.append(col)
                except (ValueError, TypeError):
                    self.excluded_features.append(col)
            else:
                self.excluded_features.append(col)
        
        logger.info(f"   📊 Features numériques : {len(self.numeric_features)}")
        logger.info(f"   🔢 Features catégorielles (one-hot) : {len(self.categorical_features)}")
        logger.info(f"   ⏭️  Features exclues : {len(self.excluded_features)}")
        logger.info("")
    
    def fit(self, df_train: pd.DataFrame):
        """Fit scaler + imputer sur train uniquement"""
        logger.info("🎓 Fit StandardScaler sur train...")
        
        if not self.numeric_features:
            raise ValueError("Aucune feature numérique identifiée")
        
        X_train_numeric = df_train[self.numeric_features]
        
        # Imputer puis scaler
        X_imputed = self.imputer.fit_transform(X_train_numeric)
        self.scaler.fit(X_imputed)
        
        # Stocker les features réellement conservées après imputation
        # (l'imputer peut supprimer les colonnes all-NaN)
        self.fitted_features = [f for f in self.numeric_features 
                               if f not in (self.imputer.statistics_ == 0).nonzero()[0]]
        
        logger.info(f"   ✅ Scaler fitted sur {X_imputed.shape[1]} features ({len(self.numeric_features)} demandées)")
        logger.info(f"   📊 Stats train : mean={self.scaler.mean_[:5]}, std={self.scaler.scale_[:5]}")
        logger.info("")
    
    def transform(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """Transform un split (train/val/test)"""
        logger.info(f"🔄 Transform {split_name}...")
        
        df_normalized = df.copy()
        
        # Normaliser features numériques
        X_numeric = df[self.numeric_features]
        X_imputed = self.imputer.transform(X_numeric)
        X_scaled = self.scaler.transform(X_imputed)
        
        # FIX: Utiliser les colonnes qui ont vraiment été transformées
        # (l'imputer skip les colonnes all-NaN)
        transformed_cols = [col for col in self.numeric_features 
                           if df[col].notna().any()]
        
        if len(transformed_cols) == X_scaled.shape[1]:
            df_scaled = pd.DataFrame(X_scaled, index=df.index, columns=transformed_cols)
            # Remplacer dans DataFrame
            for col in transformed_cols:
                df_normalized[col] = df_scaled[col]
        else:
            # Fallback: assigner directement sans noms de colonnes
            logger.warning(f"   ⚠️  Mismatch colonnes: {len(transformed_cols)} vs {X_scaled.shape[1]}")
            for i, col in enumerate(transformed_cols[:X_scaled.shape[1]]):
                df_normalized[col] = X_scaled[:, i]
        
        logger.info(f"   ✅ {len(self.numeric_features)} features normalisées")
        
        # Vérifier valeurs aberrantes
        for col in self.numeric_features[:5]:  # Check 5 premières
            vals = df_normalized[col].dropna()
            if len(vals) > 0:
                logger.info(f"      {col:<30} : mean={vals.mean():.3f}, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}")
        
        logger.info("")
        return df_normalized
    
    def save_artifacts(self, output_dir: Path):
        """Sauvegarde scaler + métadonnées"""
        logger.info("💾 Sauvegarde artifacts...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scaler
        scaler_path = output_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"   ✅ {scaler_path}")
        
        # Imputer
        imputer_path = output_dir / 'imputer.pkl'
        with open(imputer_path, 'wb') as f:
            pickle.dump(self.imputer, f)
        logger.info(f"   ✅ {imputer_path}")
        
        # Metadata
        metadata = {
            'scaler_type': 'StandardScaler',
            'imputer_strategy': 'median',
            'n_numeric_features': len(self.numeric_features),
            'n_categorical_features': len(self.categorical_features),
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'excluded_features': self.excluded_features,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist()
        }
        
        metadata_path = output_dir / 'normalization_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"   ✅ {metadata_path}")
        logger.info("")


def export_multi_format(df: pd.DataFrame, output_path: Path, split_name: str):
    """Export en CSV, Parquet, NumPy"""
    logger.info(f"📦 Export multi-format {split_name}...")
    
    # CSV
    csv_path = output_path / f'{split_name}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"   ✅ CSV : {csv_path}")
    
    # Parquet (50-80% plus compact) - optionnel si pyarrow installé
    try:
        parquet_path = output_path / f'{split_name}.parquet'
        df.to_parquet(parquet_path, index=False, compression='snappy')
        logger.info(f"   ✅ Parquet : {parquet_path}")
    except Exception as e:
        logger.warning(f"   ⚠️  Parquet skip (install pyarrow): {e}")
    
    # NumPy arrays (X features, y target)
    feature_cols = [c for c in df.columns if c not in ['victoire', 'place', 'position_arrivee']]
    X = df[feature_cols].values
    y_victoire = df['victoire'].values
    y_place = df['place'].values
    
    np.save(output_path / f'X_{split_name}.npy', X)
    np.save(output_path / f'y_{split_name}_victoire.npy', y_victoire)
    np.save(output_path / f'y_{split_name}_place.npy', y_place)
    
    logger.info(f"   ✅ NumPy : X_{split_name}.npy, y_{split_name}_*.npy")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(description='Normalise les features ML')
    parser.add_argument('--input-dir', default='data/splits',
                       help='Répertoire avec train/val/test.csv')
    parser.add_argument('--output-dir', default='data/normalized',
                       help='Répertoire de sortie')
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("🔧 NORMALISATION DES FEATURES")
        logger.info("=" * 80)
        logger.info("")
        
        input_path = Path(args.input_dir)
        output_path = Path(args.output_dir)
        
        # Charger splits
        logger.info("📂 Chargement des splits...")
        df_train = pd.read_csv(input_path / 'train.csv')
        df_val = pd.read_csv(input_path / 'val.csv')
        df_test = pd.read_csv(input_path / 'test.csv')
        logger.info(f"   ✅ Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")
        logger.info("")
        
        # Normalizer
        normalizer = FeatureNormalizer()
        normalizer.identify_features(df_train)
        normalizer.fit(df_train)
        
        # Transform
        df_train_norm = normalizer.transform(df_train, 'train')
        df_val_norm = normalizer.transform(df_val, 'val')
        df_test_norm = normalizer.transform(df_test, 'test')
        
        # Sauvegarder
        normalizer.save_artifacts(output_path)
        
        # Export multi-format
        export_multi_format(df_train_norm, output_path, 'train')
        export_multi_format(df_val_norm, output_path, 'val')
        export_multi_format(df_test_norm, output_path, 'test')
        
        logger.info("=" * 80)
        logger.info("✅ NORMALISATION TERMINÉE")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📁 Fichiers générés:")
        logger.info(f"   • {output_path}/train.csv, val.csv, test.csv")
        logger.info(f"   • {output_path}/train.parquet, val.parquet, test.parquet")
        logger.info(f"   • {output_path}/X_*.npy, y_*_victoire.npy, y_*_place.npy")
        logger.info(f"   • {output_path}/scaler.pkl, imputer.pkl")
        logger.info(f"   • {output_path}/normalization_metadata.json")
        
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        raise


if __name__ == '__main__':
    main()
