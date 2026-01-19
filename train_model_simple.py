#!/usr/bin/env python3
"""
Entraînement modèle optimisé - Version adaptée aux colonnes réelles
"""

import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection
import numpy as np
import pandas as pd
import json
import pickle
import os
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

print("=" * 120)
print("EXTRACTION DES DONNÉES HISTORIQUES")
print("=" * 120)

conn = get_connection()

# Requête adaptée aux vraies colonnes
query = """
SELECT
    nom_norm,
    race_key,
    annee,
    hippodrome_code,

    -- Target
    CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as target_place,
    is_win::int as target_win,
    place_finale,

    -- Cotes
    cote_reference,
    cote_finale,

    -- Course
    discipline,
    distance_m,
    numero_dossard,

    -- Cheval
    age,
    sexe,
    poids_kg,
    handicap_distance,

    -- Conditions
    etat_piste,
    meteo_code,
    temperature_c,
    vent_kmh,

    -- Stats historiques si disponibles
    musique_last_5,
    total_gains_euros,
    nombre_victoires,
    nombre_places,
    taux_reussite,

    -- Jockey/Driver
    driver_jockey,
    entraineur

FROM cheval_courses_seen
WHERE cote_reference IS NOT NULL
  AND cote_reference > 0
  AND place_finale IS NOT NULL
  AND annee >= 2023
ORDER BY race_key ASC
"""

print("\nChargement des données...")
try:
    df = pd.read_sql(query, conn)
except Exception as e:
    print(f"Erreur requête initiale: {e}")
    print("\nEssai avec requête minimale...")

    query = """
    SELECT
        nom_norm,
        race_key,
        annee,
        CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as target_place,
        is_win::int as target_win,
        place_finale,
        cote_reference,
        cote_finale,
        discipline,
        distance_m,
        age,
        sexe,
        hippodrome_code
    FROM cheval_courses_seen
    WHERE cote_reference IS NOT NULL
      AND cote_reference > 0
      AND place_finale IS NOT NULL
      AND annee >= 2023
    ORDER BY race_key ASC
    """
    df = pd.read_sql(query, conn)

conn.close()

print(f"✓ {len(df):,} courses chargées")
print(f"  {df['race_key'].nunique():,} courses uniques")
print(f"  Placés: {df['target_place'].sum():,} ({df['target_place'].mean()*100:.1f}%)")
print(f"  Gagnants: {df['target_win'].sum():,} ({df['target_win'].mean()*100:.1f}%)")

print("\n" + "=" * 120)
print("PRÉPARATION DES FEATURES")
print("=" * 120)

# Extraire date depuis race_key (format: YYYY-MM-DD|...)
df["date"] = pd.to_datetime(df["race_key"].str.split("|").str[0])

# Features numériques simples
numeric_cols = ["cote_reference", "cote_finale", "distance_m", "age"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
            df[col].median() if col in df.columns else 0
        )

# Encodage catégorielles
from sklearn.preprocessing import LabelEncoder

categorical_cols = ["discipline", "sexe", "hippodrome_code"]
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN")
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Features dérivées
df["cote_log"] = np.log1p(df["cote_reference"])
df["cote_drift"] = df["cote_finale"] - df["cote_reference"]
df["categorie_cote"] = pd.cut(
    df["cote_reference"],
    bins=[0, 5, 10, 20, 1000],
    labels=["favori", "moyen", "outsider", "longshot"],
)
df["categorie_cote_enc"] = LabelEncoder().fit_transform(df["categorie_cote"])

# Features finales
feature_cols = [
    "cote_reference",
    "cote_finale",
    "distance_m",
    "age",
    "cote_log",
    "cote_drift",
    "discipline_enc",
    "sexe_enc",
    "hippodrome_code_enc",
    "categorie_cote_enc",
]

feature_cols = [c for c in feature_cols if c in df.columns]
print(f"✓ {len(feature_cols)} features: {feature_cols}")

print("\n" + "=" * 120)
print("SPLIT TEMPOREL")
print("=" * 120)

# Split par date
train_df = df[df["date"] < "2025-11-01"].copy()
val_df = df[(df["date"] >= "2025-11-01") & (df["date"] < "2025-12-15")].copy()
test_df = df[df["date"] >= "2025-12-15"].copy()

print(f"Train: {len(train_df):,} courses")
print(f"Val:   {len(val_df):,} courses")
print(f"Test:  {len(test_df):,} courses")

X_train = train_df[feature_cols].values
y_train = train_df["target_place"].values

X_val = val_df[feature_cols].values
y_val = val_df["target_place"].values

X_test = test_df[feature_cols].values
y_test = test_df["target_place"].values

print("\n" + "=" * 120)
print("ENTRAÎNEMENT DES MODÈLES")
print("=" * 120)

results = {}

# XGBoost
print("\n[1/3] XGBoost...")
try:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred_val = model.predict_proba(X_val)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]

    auc_val = roc_auc_score(y_val, pred_val)
    auc_test = roc_auc_score(y_test, pred_test)

    print(f"  Val AUC:  {auc_val:.4f}")
    print(f"  Test AUC: {auc_test:.4f}")

    results["xgb"] = {
        "model": model,
        "val_auc": auc_val,
        "test_auc": auc_test,
        "pred_test": pred_test,
    }
except Exception as e:
    print(f"  ❌ {e}")

# LightGBM
print("\n[2/3] LightGBM...")
try:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    pred_val = model.predict_proba(X_val)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]

    auc_val = roc_auc_score(y_val, pred_val)
    auc_test = roc_auc_score(y_test, pred_test)

    print(f"  Val AUC:  {auc_val:.4f}")
    print(f"  Test AUC: {auc_test:.4f}")

    results["lgb"] = {
        "model": model,
        "val_auc": auc_val,
        "test_auc": auc_test,
        "pred_test": pred_test,
    }
except Exception as e:
    print(f"  ❌ {e}")

# CatBoost
print("\n[3/3] CatBoost...")
try:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=300, random_seed=42, verbose=False
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    pred_val = model.predict_proba(X_val)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]

    auc_val = roc_auc_score(y_val, pred_val)
    auc_test = roc_auc_score(y_test, pred_test)

    print(f"  Val AUC:  {auc_val:.4f}")
    print(f"  Test AUC: {auc_test:.4f}")

    results["cat"] = {
        "model": model,
        "val_auc": auc_val,
        "test_auc": auc_test,
        "pred_test": pred_test,
    }
except Exception as e:
    print(f"  ❌ {e}")

# Ensemble
if len(results) >= 2:
    print("\n[ENSEMBLE] Moyenne des modèles...")
    pred_test_ensemble = np.mean([r["pred_test"] for r in results.values()], axis=0)
    auc_test = roc_auc_score(y_test, pred_test_ensemble)
    print(f"  Test AUC: {auc_test:.4f}")

    results["ensemble"] = {
        "model": None,
        "val_auc": 0,
        "test_auc": auc_test,
        "pred_test": pred_test_ensemble,
    }

print("\n" + "=" * 120)
print("RÉSULTATS")
print("=" * 120)

print(f"\n{'Modèle':<15} {'Val AUC':<12} {'Test AUC':<12}")
print("-" * 40)
for name, res in results.items():
    print(f"{name.upper():<15} {res['val_auc']:.4f}      {res['test_auc']:.4f}")

best_name = max(results.keys(), key=lambda k: results[k]["test_auc"]) if results else "none"
if not results:
    print("\n❌ Aucun modèle entraîné avec succès")
    sys.exit(1)

best_model = results[best_name]

print(f"\n🏆 MEILLEUR: {best_name.upper()} (AUC Test = {best_model['test_auc']:.4f})")

# Sauvegarde
model_dir = "/Users/gicquelsacha/horse3/models_optimized"
os.makedirs(model_dir, exist_ok=True)

if best_model["model"] is not None:
    path = f"{model_dir}/{best_name}_best.pkl"
    with open(path, "wb") as f:
        pickle.dump(best_model["model"], f)
    print(f"\n✓ Sauvegardé: {path}")

# Tester stratégie
print("\n" + "=" * 120)
print("SIMULATION STRATÉGIE (données test)")
print("=" * 120)

test_df_copy = test_df.copy()
test_df_copy["pred"] = best_model["pred_test"]

for threshold in [0.3, 0.35, 0.4, 0.45]:
    mask = test_df_copy["pred"] >= threshold
    selected = test_df_copy[mask]

    if len(selected) == 0:
        continue

    stake = 10
    total_stake = len(selected) * stake
    total_return = (selected["target_place"] * selected["cote_finale"] * stake).sum()
    roi = ((total_return - total_stake) / total_stake * 100) if total_stake > 0 else 0

    print(
        f"\nSeuil {threshold:.2f}: {len(selected)} paris, Win rate: {selected['target_place'].mean()*100:.1f}%, ROI: {roi:+.2f}%"
    )

print("\n✅ TERMINÉ\n")
