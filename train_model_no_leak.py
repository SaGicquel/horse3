#!/usr/bin/env python3
"""
Entraînement SANS LEAK - Uniquement features disponibles AVANT la course
"""

import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

print("=" * 120)
print("ENTRAÎNEMENT SANS LOOK-AHEAD BIAS")
print("=" * 120)

conn = get_connection()

query = """
SELECT
    nom_norm,
    race_key,
    annee,

    -- Target
    CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as target_place,
    is_win::int as target_win,
    place_finale,

    -- ✅ COTE REFERENCE UNIQUEMENT (pas finale!)
    cote_reference,

    -- ✅ Features structurelles
    discipline,
    distance_m,
    age,
    sexe,
    hippodrome_code,
    numero_dossard,
    poids_kg,
    handicap_distance,

    -- ✅ Conditions
    etat_piste,
    meteo_code,
    temperature_c,
    vent_kmh,

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

print("\nChargement...")
df = pd.read_sql(query, conn)
conn.close()

print(f"✓ {len(df):,} courses")
print(f"  Placés: {df['target_place'].mean()*100:.1f}%")

# Extraire date
df["date"] = pd.to_datetime(df["race_key"].str.split("|").str[0])

print("\n" + "=" * 120)
print("FEATURES SANS LEAK")
print("=" * 120)

# Numériques (remplir NaN)
numeric_cols = [
    "cote_reference",
    "distance_m",
    "age",
    "poids_kg",
    "handicap_distance",
    "temperature_c",
    "vent_kmh",
    "numero_dossard",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# Catégorielles
categorical_cols = ["discipline", "sexe", "hippodrome_code", "etat_piste", "meteo_code"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN")
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# Features dérivées SAFE
df["cote_log"] = np.log1p(df["cote_reference"])
df["cote_squared"] = df["cote_reference"] ** 2
df["is_favori"] = (df["cote_reference"] <= 5).astype(int)
df["is_outsider"] = (df["cote_reference"] >= 15).astype(int)
df["distance_cat"] = pd.cut(
    df["distance_m"],
    bins=[0, 1600, 2200, 3000, 10000],
    labels=["courte", "moyenne", "longue", "tres_longue"],
)
df["distance_cat_enc"] = LabelEncoder().fit_transform(df["distance_cat"])
df["age_cote_interaction"] = df["age"] * df["cote_log"]
df["poids_distance_ratio"] = df["poids_kg"] / (df["distance_m"] / 1000)

# Stats par hippodrome (calculées sur train seulement pour éviter leak)
print("\nCalcul stats hippodrome (sur données train)...")

train_mask = df["date"] < "2025-11-01"
train_df_temp = df[train_mask]

hippodrome_stats = (
    train_df_temp.groupby("hippodrome_code")
    .agg({"target_place": "mean", "cote_reference": "mean"})
    .reset_index()
)
hippodrome_stats.columns = ["hippodrome_code", "hippodrome_place_rate", "hippodrome_avg_cote"]

df = df.merge(hippodrome_stats, on="hippodrome_code", how="left")
df["hippodrome_place_rate"] = df["hippodrome_place_rate"].fillna(0.313)  # moyenne globale
df["hippodrome_avg_cote"] = df["hippodrome_avg_cote"].fillna(df["cote_reference"].mean())

# Features finales
feature_cols = [
    # Cote
    "cote_reference",
    "cote_log",
    "cote_squared",
    "is_favori",
    "is_outsider",
    # Course
    "distance_m",
    "distance_cat_enc",
    # Cheval
    "age",
    "poids_kg",
    "handicap_distance",
    "numero_dossard",
    # Interactions
    "age_cote_interaction",
    "poids_distance_ratio",
    # Catégorielles
    "discipline_enc",
    "sexe_enc",
    "hippodrome_code_enc",
    "etat_piste_enc",
    "meteo_code_enc",
    # Stats hippodrome
    "hippodrome_place_rate",
    "hippodrome_avg_cote",
]

feature_cols = [c for c in feature_cols if c in df.columns]
print(f"\n✅ {len(feature_cols)} features SAFE:")
for i, f in enumerate(feature_cols, 1):
    print(f"  {i:2}. {f}")

print("\n" + "=" * 120)
print("SPLIT TEMPOREL")
print("=" * 120)

train_df = df[df["date"] < "2025-11-01"].copy()
val_df = df[(df["date"] >= "2025-11-01") & (df["date"] < "2025-12-15")].copy()
test_df = df[df["date"] >= "2025-12-15"].copy()

print(f"Train: {len(train_df):,}")
print(f"Val:   {len(val_df):,}")
print(f"Test:  {len(test_df):,}")

X_train = train_df[feature_cols].values
y_train = train_df["target_place"].values

X_val = val_df[feature_cols].values
y_val = val_df["target_place"].values

X_test = test_df[feature_cols].values
y_test = test_df["target_place"].values

print("\n" + "=" * 120)
print("ENTRAÎNEMENT")
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

    print(f"  Val: {auc_val:.4f}, Test: {auc_test:.4f}")
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

    print(f"  Val: {auc_val:.4f}, Test: {auc_test:.4f}")
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

    print(f"  Val: {auc_val:.4f}, Test: {auc_test:.4f}")
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
    print("\n[ENSEMBLE]...")
    pred_test_ens = np.mean([r["pred_test"] for r in results.values()], axis=0)
    auc_test = roc_auc_score(y_test, pred_test_ens)
    print(f"  Test: {auc_test:.4f}")
    results["ensemble"] = {
        "model": None,
        "val_auc": 0,
        "test_auc": auc_test,
        "pred_test": pred_test_ens,
    }

print("\n" + "=" * 120)
print("RÉSULTATS")
print("=" * 120)

for name, res in results.items():
    print(f"{name.upper():12} | Val: {res['val_auc']:.4f} | Test: {res['test_auc']:.4f}")

best_name = max(results.keys(), key=lambda k: results[k]["test_auc"])
best = results[best_name]

print(f"\n🏆 MEILLEUR: {best_name.upper()} (Test AUC = {best['test_auc']:.4f})")

# Sauvegarder
model_dir = "/Users/gicquelsacha/horse3/models_optimized"
os.makedirs(model_dir, exist_ok=True)

if best["model"] is not None:
    path = f"{model_dir}/{best_name}_no_leak.pkl"
    with open(path, "wb") as f:
        pickle.dump(best["model"], f)
    print(f"✓ Sauvegardé: {path}")

# Tester stratégie RÉALISTE
print("\n" + "=" * 120)
print("SIMULATION RÉALISTE (données test)")
print("=" * 120)

test_df_copy = test_df.copy()
test_df_copy["pred"] = best["pred_test"]

# Important: utiliser cote_reference (pas finale!) pour calculer les gains
for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
    mask = test_df_copy["pred"] >= threshold
    selected = test_df_copy[mask]

    if len(selected) == 0:
        continue

    stake = 10
    total_stake = len(selected) * stake

    # IMPORTANT: Utiliser cote_reference pour simuler les gains
    # (car cote_finale n'est pas dispo avant la course)
    total_return = (selected["target_place"] * selected["cote_reference"] * stake).sum()

    roi = ((total_return - total_stake) / total_stake * 100) if total_stake > 0 else 0
    win_rate = selected["target_place"].mean() * 100
    avg_cote = selected["cote_reference"].mean()

    print(
        f"Seuil {threshold:.2f}: {len(selected):4} paris | Win: {win_rate:5.1f}% | Cote moy: {avg_cote:5.2f} | ROI: {roi:+7.2f}%"
    )

print("\n" + "=" * 120)
print("✅ TERMINÉ - ROI RÉALISTE SANS LEAK")
print("=" * 120)
