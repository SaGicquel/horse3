#!/usr/bin/env python3
"""
Ré-entraînement du modèle Champion XGBoost avec artifacts cohérents
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("🚀 RÉ-ENTRAÎNEMENT MODÈLE CHAMPION XGBOOST")
print("=" * 80)

DATA_DIR = Path("/Users/gicquelsacha/horse3/data/normalized")
OUTPUT_DIR = Path("/Users/gicquelsacha/horse3/data/models/champion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n📂 Chargement des données...")
df_train = pd.read_csv(DATA_DIR / "train.csv")
df_val = pd.read_csv(DATA_DIR / "val.csv")
df_test = pd.read_csv(DATA_DIR / "test.csv")

print(f"   Train: {len(df_train):,} lignes")
print(f"   Val:   {len(df_val):,} lignes")
print(f"   Test:  {len(df_test):,} lignes")

EXCLUDE_COLS = [
    "id_performance",
    "id_course",
    "nom_norm",
    "annee",
    "position_arrivee",
    "victoire",
    "place",
    "hippodrome_nom",
    "driver_jockey",
    "entraineur",
    "discipline",
    "sexe",
]

feature_cols = [
    col
    for col in df_train.columns
    if col not in EXCLUDE_COLS and df_train[col].dtype in ["int64", "float64"]
]

print(f"\n📊 Features: {len(feature_cols)}")

X_train = df_train[feature_cols].values.astype(np.float32)
y_train = df_train["victoire"].values
X_val = df_val[feature_cols].values.astype(np.float32)
y_val = df_val["victoire"].values
X_test = df_test[feature_cols].values.astype(np.float32)
y_test = df_test["victoire"].values

print(f"🎯 Train: {100*y_train.mean():.2f}% victoires")

print("\n🔧 Imputation + Standardisation...")
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("\n⏳ Entraînement XGBoost...")
scale_pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    early_stopping_rounds=20,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"   ✅ {model.best_iteration} arbres")

y_pred_val = model.predict_proba(X_val)[:, 1]
y_pred_test = model.predict_proba(X_test)[:, 1]
roc_test = roc_auc_score(y_test, y_pred_test)
brier_test = brier_score_loss(y_test, y_pred_test)

print(f"\n📊 ROC-AUC Test: {roc_test:.4f}")
print(f"📊 Brier Test: {brier_test:.4f}")

print("\n💾 Sauvegarde...")
with open(OUTPUT_DIR / "xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(OUTPUT_DIR / "feature_imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)
with open(OUTPUT_DIR / "feature_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(OUTPUT_DIR / "feature_names.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

metadata = {
    "model_type": "xgboost",
    "champion_since": "2025-12-14",
    "performance_metrics": {"roc_auc_test": round(roc_test, 4), "brier_test": round(brier_test, 4)},
    "files": {
        "model": "xgboost_model.pkl",
        "feature_scaler": "feature_scaler.pkl",
        "feature_imputer": "feature_imputer.pkl",
        "feature_names": "feature_names.json",
    },
    "features_count": len(feature_cols),
    "training_samples": len(df_train),
    "best_iteration": model.best_iteration,
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ MODÈLE CHAMPION RÉ-ENTRAÎNÉ !")
print(f"   Features: {len(feature_cols)} (cohérents)")
print(f"   ROC-AUC: {roc_test:.4f}")
