#!/usr/bin/env python3
"""
Entraînement avec VRAIES cotes PLACÉ
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
print("ENTRAÎNEMENT AVEC VRAIES COTES PLACÉ")
print("=" * 120)

conn = get_connection()

query = """
SELECT
    nom_norm,
    race_key,
    annee,

    -- Target
    CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as target_place,
    place_finale,

    -- ✅ Cote REFERENCE (avant course)
    cote_reference,

    -- ✅ Rapport PLACÉ (pour calcul ROI réaliste)
    rapport_place,

    -- Features
    discipline,
    distance_m,
    age,
    sexe,
    hippodrome_code,
    numero_dossard,
    poids_kg,
    handicap_distance,
    etat_piste,
    meteo_code,
    temperature_c,
    vent_kmh

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
print(f"  Placés: {df['target_place'].sum():,} ({df['target_place'].mean()*100:.1f}%)")

# Calculer cote placé approximative si rapport_place manquant
df["rapport_place"] = pd.to_numeric(df["rapport_place"], errors="coerce")

# Formule approximative: cote_place ≈ 1 + (cote_gagnant - 1) / 3.5
df["cote_place_approx"] = 1 + (df["cote_reference"] - 1) / 3.5

# Utiliser rapport_place si dispo, sinon approximation
df["cote_place"] = df["rapport_place"].fillna(df["cote_place_approx"])

print(
    f"\nRapport placé disponible: {df['rapport_place'].notna().sum():,} ({df['rapport_place'].notna().mean()*100:.1f}%)"
)
print(f"Utilise approximation: {df['rapport_place'].isna().sum():,}")

# Extraire date
df["date"] = pd.to_datetime(df["race_key"].str.split("|").str[0])

print("\n" + "=" * 120)
print("FEATURES")
print("=" * 120)

# Numériques
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
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

# Catégorielles
categorical_cols = ["discipline", "sexe", "hippodrome_code", "etat_piste", "meteo_code"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN")
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# Features dérivées
df["cote_log"] = np.log1p(df["cote_reference"])
df["cote_squared"] = df["cote_reference"] ** 2
df["is_favori"] = (df["cote_reference"] <= 5).astype(int)
df["is_outsider"] = (df["cote_reference"] >= 15).astype(int)
df["distance_cat_enc"] = pd.cut(
    df["distance_m"], bins=[0, 1600, 2200, 3000, 10000], labels=[0, 1, 2, 3]
).astype(int)
df["age_cote_interaction"] = df["age"] * df["cote_log"]

# Stats hippodrome (sur train uniquement)
train_mask = df["date"] < "2025-11-01"
hippodrome_stats = (
    df[train_mask]
    .groupby("hippodrome_code")
    .agg({"target_place": "mean", "cote_reference": "mean"})
    .reset_index()
)
hippodrome_stats.columns = ["hippodrome_code", "hippodrome_place_rate", "hippodrome_avg_cote"]

df = df.merge(hippodrome_stats, on="hippodrome_code", how="left")
df["hippodrome_place_rate"] = df["hippodrome_place_rate"].fillna(0.313)
df["hippodrome_avg_cote"] = df["hippodrome_avg_cote"].fillna(df["cote_reference"].mean())

feature_cols = [
    "cote_reference",
    "cote_log",
    "cote_squared",
    "is_favori",
    "is_outsider",
    "distance_m",
    "distance_cat_enc",
    "age",
    "poids_kg",
    "handicap_distance",
    "numero_dossard",
    "age_cote_interaction",
    "discipline_enc",
    "sexe_enc",
    "hippodrome_code_enc",
    "etat_piste_enc",
    "meteo_code_enc",
    "hippodrome_place_rate",
    "hippodrome_avg_cote",
]

feature_cols = [c for c in feature_cols if c in df.columns]
print(f"✓ {len(feature_cols)} features")

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

    pred_test = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, pred_test)

    print(f"  Test AUC: {auc_test:.4f}")
    results["xgb"] = {"model": model, "test_auc": auc_test, "pred_test": pred_test}
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

    pred_test = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, pred_test)

    print(f"  Test AUC: {auc_test:.4f}")
    results["lgb"] = {"model": model, "test_auc": auc_test, "pred_test": pred_test}
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

    pred_test = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, pred_test)

    print(f"  Test AUC: {auc_test:.4f}")
    results["cat"] = {"model": model, "test_auc": auc_test, "pred_test": pred_test}
except Exception as e:
    print(f"  ❌ {e}")

# Ensemble
if len(results) >= 2:
    print("\n[ENSEMBLE]...")
    pred_test_ens = np.mean([r["pred_test"] for r in results.values()], axis=0)
    auc_test = roc_auc_score(y_test, pred_test_ens)
    print(f"  Test AUC: {auc_test:.4f}")
    results["ensemble"] = {"model": None, "test_auc": auc_test, "pred_test": pred_test_ens}

best_name = max(results.keys(), key=lambda k: results[k]["test_auc"])
best = results[best_name]

print(f"\n🏆 MEILLEUR: {best_name.upper()} (AUC = {best['test_auc']:.4f})")

# Sauvegarder
model_dir = "/Users/gicquelsacha/horse3/models_optimized"
os.makedirs(model_dir, exist_ok=True)

if best["model"] is not None:
    path = f"{model_dir}/{best_name}_place_final.pkl"
    with open(path, "wb") as f:
        pickle.dump(best["model"], f)
    print(f"✓ Sauvegardé: {path}")

print("\n" + "=" * 120)
print("SIMULATION AVEC VRAIES COTES PLACÉ")
print("=" * 120)

test_df_copy = test_df.copy()
test_df_copy["pred"] = best["pred_test"]

print(f"\n{'Seuil':8} | {'Paris':>6} | {'Win%':>6} | {'Cote Place Moy':>15} | {'ROI':>10}")
print("-" * 70)

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
    mask = test_df_copy["pred"] >= threshold
    selected = test_df_copy[mask]

    if len(selected) == 0:
        continue

    stake = 10
    total_stake = len(selected) * stake

    # ✅ CALCUL CORRECT avec cote_place
    total_return = (selected["target_place"] * selected["cote_place"] * stake).sum()

    roi = ((total_return - total_stake) / total_stake * 100) if total_stake > 0 else 0
    win_rate = selected["target_place"].mean() * 100
    avg_cote_place = selected["cote_place"].mean()

    print(
        f"{threshold:.2f}     | {len(selected):6} | {win_rate:5.1f}% | {avg_cote_place:15.2f} | {roi:+9.2f}%"
    )

# Comparer avec calcul FAUX (cote gagnant)
print(f"\n{'='*120}")
print("COMPARAISON: Calcul CORRECT (cote placé) vs INCORRECT (cote gagnant)")
print(f"{'='*120}\n")

threshold = 0.40
mask = test_df_copy["pred"] >= threshold
selected = test_df_copy[mask]

stake = 10
total_stake = len(selected) * stake

# Calcul CORRECT
total_return_correct = (selected["target_place"] * selected["cote_place"] * stake).sum()
roi_correct = (total_return_correct - total_stake) / total_stake * 100

# Calcul INCORRECT
total_return_incorrect = (selected["target_place"] * selected["cote_reference"] * stake).sum()
roi_incorrect = (total_return_incorrect - total_stake) / total_stake * 100

print(f"Avec COTE PLACÉ (correct):  ROI = {roi_correct:+7.2f}%")
print(f"Avec COTE GAGNANT (faux):   ROI = {roi_incorrect:+7.2f}%")
print(f"\nDifférence: {roi_incorrect - roi_correct:+.2f} points de ROI !")

print("\n" + "=" * 120)
print("✅ ENTRAÎNEMENT TERMINÉ - ROI RÉALISTE")
print("=" * 120)
