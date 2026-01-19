#!/usr/bin/env python3
"""
Entraînement d'un modèle optimisé sur TOUTES les données historiques
Objectif: Maximiser la précision des prédictions de placement
"""

import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import warnings

warnings.filterwarnings("ignore")

print("=" * 120)
print("EXTRACTION DES DONNÉES HISTORIQUES COMPLÈTES")
print("=" * 120)

conn = get_connection()

# Extraire TOUTES les données disponibles
query = """
SELECT
    -- Identifiants
    nom_norm,
    race_key,
    date_course,
    hippodrome_code,

    -- Target (victoire ou placé top 3)
    CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as target_place,
    is_win::int as target_win,
    place_finale,

    -- Cotes
    cote_reference,
    cote_finale,

    -- Performances récentes du cheval
    nb_courses_30j,
    nb_victoires_30j,
    nb_places_30j,
    moyenne_place_30j,
    nb_courses_90j,
    nb_victoires_90j,
    nb_places_90j,
    moyenne_place_90j,
    nb_courses_365j,
    nb_victoires_365j,
    nb_places_365j,
    moyenne_place_365j,

    -- Stats de gains
    total_gains,
    gains_12_derniers_mois,
    gains_course_precedente,

    -- Forme récente
    jours_depuis_derniere_course,
    place_derniere_course,
    moyenne_recency_5_courses,

    -- Caractéristiques du cheval
    age,
    sexe_code,
    poids_cheval_kg,
    handicap_distance,
    handicap_poids,

    -- Jockey/Entraîneur
    jockey_nom,
    jockey_victoires_saison,
    jockey_taux_victoire,
    entraineur_nom,
    entraineur_victoires_saison,
    entraineur_taux_victoire,

    -- Course
    discipline,
    distance_m,
    nombre_partants,
    numero_corde,

    -- Conditions
    etat_piste,
    meteo_code,
    temperature_c,
    vent_kmh,

    -- Stats hippodrome
    hippodrome_taux_favoris,

    -- Position dans le peloton
    CASE
        WHEN cote_reference <= 5 THEN 'favori'
        WHEN cote_reference <= 10 THEN 'moyen'
        WHEN cote_reference <= 20 THEN 'outsider'
        ELSE 'longshot'
    END as categorie_cote

FROM cheval_courses_seen
WHERE date_course IS NOT NULL
  AND cote_reference IS NOT NULL
  AND cote_reference > 0
  AND place_finale IS NOT NULL
  AND date_course >= '2023-01-01'
ORDER BY date_course ASC
"""

print("\nChargement des données...")
df = pd.read_sql(query, conn)
conn.close()

print(f"✓ {len(df):,} courses chargées")
print(f"  Période: {df['date_course'].min()} à {df['date_course'].max()}")
print(f"  {df['race_key'].nunique():,} courses uniques")
print(f"  {df['nom_norm'].nunique():,} chevaux uniques")

# Stats target
print("\nDistribution target:")
print(f"  Placés (top 3): {df['target_place'].sum():,} ({df['target_place'].mean()*100:.1f}%)")
print(f"  Gagnants: {df['target_win'].sum():,} ({df['target_win'].mean()*100:.1f}%)")

print("\n" + "=" * 120)
print("PRÉPARATION DES FEATURES")
print("=" * 120)

# Features numériques
numeric_features = [
    "cote_reference",
    "cote_finale",
    "nb_courses_30j",
    "nb_victoires_30j",
    "nb_places_30j",
    "moyenne_place_30j",
    "nb_courses_90j",
    "nb_victoires_90j",
    "nb_places_90j",
    "moyenne_place_90j",
    "nb_courses_365j",
    "nb_victoires_365j",
    "nb_places_365j",
    "moyenne_place_365j",
    "total_gains",
    "gains_12_derniers_mois",
    "gains_course_precedente",
    "jours_depuis_derniere_course",
    "place_derniere_course",
    "moyenne_recency_5_courses",
    "age",
    "poids_cheval_kg",
    "handicap_distance",
    "handicap_poids",
    "jockey_victoires_saison",
    "jockey_taux_victoire",
    "entraineur_victoires_saison",
    "entraineur_taux_victoire",
    "distance_m",
    "nombre_partants",
    "numero_corde",
    "temperature_c",
    "vent_kmh",
    "hippodrome_taux_favoris",
]

# Features catégorielles
categorical_features = [
    "hippodrome_code",
    "sexe_code",
    "discipline",
    "etat_piste",
    "meteo_code",
    "categorie_cote",
]

# Remplir les valeurs manquantes
print("\nTraitement des valeurs manquantes...")
for col in numeric_features:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN")

# Encodage des catégorielles
print("Encodage des variables catégorielles...")
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Features finales
feature_cols = [col for col in numeric_features if col in df.columns]
feature_cols += [col + "_encoded" for col in categorical_features if col in df.columns]

print(f"\n✓ {len(feature_cols)} features prêtes")

# Créer features dérivées
print("\nCréation de features dérivées...")

# Ratio de forme
df["ratio_victoires_30j"] = df["nb_victoires_30j"] / (df["nb_courses_30j"] + 1)
df["ratio_places_30j"] = df["nb_places_30j"] / (df["nb_courses_30j"] + 1)
df["ratio_victoires_90j"] = df["nb_victoires_90j"] / (df["nb_courses_90j"] + 1)

# Gains par course
df["gains_par_course"] = df["total_gains"] / (df["nb_courses_365j"] + 1)

# Activité récente
df["courses_par_mois"] = df["nb_courses_90j"] / 3.0

# Avantage cote
df["cote_drift"] = df["cote_finale"] - df["cote_reference"]
df["cote_log"] = np.log1p(df["cote_reference"])

# Compétitivité jockey/entraîneur
df["jockey_entraineur_score"] = df["jockey_taux_victoire"] * df["entraineur_taux_victoire"]

# Features position
df["position_relative"] = df["numero_corde"] / df["nombre_partants"]

feature_cols += [
    "ratio_victoires_30j",
    "ratio_places_30j",
    "ratio_victoires_90j",
    "gains_par_course",
    "courses_par_mois",
    "cote_drift",
    "cote_log",
    "jockey_entraineur_score",
    "position_relative",
]

print(f"✓ {len(feature_cols)} features totales (avec dérivées)")

print("\n" + "=" * 120)
print("SPLIT TEMPOREL DES DONNÉES")
print("=" * 120)

# Split temporel (pas aléatoire !)
# Train: avant 2025-11-01
# Val: 2025-11-01 à 2025-12-15
# Test: après 2025-12-15

train_df = df[df["date_course"] < "2025-11-01"].copy()
val_df = df[(df["date_course"] >= "2025-11-01") & (df["date_course"] < "2025-12-15")].copy()
test_df = df[df["date_course"] >= "2025-12-15"].copy()

print(
    f"Train: {len(train_df):,} courses ({train_df['date_course'].min()} à {train_df['date_course'].max()})"
)
print(
    f"Val:   {len(val_df):,} courses ({val_df['date_course'].min()} à {val_df['date_course'].max()})"
)
print(
    f"Test:  {len(test_df):,} courses ({test_df['date_course'].min()} à {test_df['date_course'].max()})"
)

X_train = train_df[feature_cols].values
y_train_place = train_df["target_place"].values
y_train_win = train_df["target_win"].values

X_val = val_df[feature_cols].values
y_val_place = val_df["target_place"].values
y_val_win = val_df["target_win"].values

X_test = test_df[feature_cols].values
y_test_place = test_df["target_place"].values
y_test_win = test_df["target_win"].values

print("\n" + "=" * 120)
print("ENTRAÎNEMENT DES MODÈLES")
print("=" * 120)

results = {}

# ============ XGBOOST ============
print("\n[1/3] XGBoost...")
try:
    import xgboost as xgb

    # Modèle pour PLACÉ (top 3)
    print("  Entraînement XGBoost PLACÉ...")
    xgb_place = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    xgb_place.fit(
        X_train,
        y_train_place,
        eval_set=[(X_val, y_val_place)],
        early_stopping_rounds=50,
        verbose=False,
    )

    # Prédictions
    pred_val_place = xgb_place.predict_proba(X_val)[:, 1]
    pred_test_place = xgb_place.predict_proba(X_test)[:, 1]

    auc_val = roc_auc_score(y_val_place, pred_val_place)
    auc_test = roc_auc_score(y_test_place, pred_test_place)

    print(f"    Val AUC:  {auc_val:.4f}")
    print(f"    Test AUC: {auc_test:.4f}")

    results["xgb_place"] = {
        "model": xgb_place,
        "val_auc": auc_val,
        "test_auc": auc_test,
        "predictions_val": pred_val_place,
        "predictions_test": pred_test_place,
    }

except Exception as e:
    print(f"  ❌ Erreur XGBoost: {e}")

# ============ LIGHTGBM ============
print("\n[2/3] LightGBM...")
try:
    import lightgbm as lgb

    print("  Entraînement LightGBM PLACÉ...")
    lgb_place = lgb.LGBMClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    lgb_place.fit(
        X_train,
        y_train_place,
        eval_set=[(X_val, y_val_place)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    pred_val_place = lgb_place.predict_proba(X_val)[:, 1]
    pred_test_place = lgb_place.predict_proba(X_test)[:, 1]

    auc_val = roc_auc_score(y_val_place, pred_val_place)
    auc_test = roc_auc_score(y_test_place, pred_test_place)

    print(f"    Val AUC:  {auc_val:.4f}")
    print(f"    Test AUC: {auc_test:.4f}")

    results["lgb_place"] = {
        "model": lgb_place,
        "val_auc": auc_val,
        "test_auc": auc_test,
        "predictions_val": pred_val_place,
        "predictions_test": pred_test_place,
    }

except Exception as e:
    print(f"  ❌ Erreur LightGBM: {e}")

# ============ CATBOOST ============
print("\n[3/3] CatBoost...")
try:
    from catboost import CatBoostClassifier

    print("  Entraînement CatBoost PLACÉ...")
    cat_place = CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=500, l2_leaf_reg=3.0, random_seed=42, verbose=False
    )

    cat_place.fit(
        X_train,
        y_train_place,
        eval_set=(X_val, y_val_place),
        early_stopping_rounds=50,
        verbose=False,
    )

    pred_val_place = cat_place.predict_proba(X_val)[:, 1]
    pred_test_place = cat_place.predict_proba(X_test)[:, 1]

    auc_val = roc_auc_score(y_val_place, pred_val_place)
    auc_test = roc_auc_score(y_test_place, pred_test_place)

    print(f"    Val AUC:  {auc_val:.4f}")
    print(f"    Test AUC: {auc_test:.4f}")

    results["cat_place"] = {
        "model": cat_place,
        "val_auc": auc_val,
        "test_auc": auc_test,
        "predictions_val": pred_val_place,
        "predictions_test": pred_test_place,
    }

except Exception as e:
    print(f"  ❌ Erreur CatBoost: {e}")

# ============ ENSEMBLE ============
print("\n[BONUS] Ensemble (moyenne des 3 modèles)...")
if len(results) >= 2:
    pred_val_ensemble = np.mean([results[k]["predictions_val"] for k in results.keys()], axis=0)
    pred_test_ensemble = np.mean([results[k]["predictions_test"] for k in results.keys()], axis=0)

    auc_val = roc_auc_score(y_val_place, pred_val_ensemble)
    auc_test = roc_auc_score(y_test_place, pred_test_ensemble)

    print(f"    Val AUC:  {auc_val:.4f}")
    print(f"    Test AUC: {auc_test:.4f}")

    results["ensemble"] = {
        "model": None,
        "val_auc": auc_val,
        "test_auc": auc_test,
        "predictions_val": pred_val_ensemble,
        "predictions_test": pred_test_ensemble,
    }

print("\n" + "=" * 120)
print("COMPARAISON DES MODÈLES")
print("=" * 120)

comparison = []
for name, res in results.items():
    comparison.append(
        {
            "Modèle": name.upper(),
            "Val AUC": f"{res['val_auc']:.4f}",
            "Test AUC": f"{res['test_auc']:.4f}",
            "Delta": f"{res['test_auc'] - res['val_auc']:+.4f}",
        }
    )

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))

# Meilleur modèle
best_model_name = max(results.keys(), key=lambda k: results[k]["test_auc"])
best_model = results[best_model_name]

print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name.upper()}")
print(f"   Test AUC: {best_model['test_auc']:.4f}")

# Sauvegarder
print("\n" + "=" * 120)
print("SAUVEGARDE DU MODÈLE")
print("=" * 120)

model_dir = "/Users/gicquelsacha/horse3/models_optimized"
import os

os.makedirs(model_dir, exist_ok=True)

# Sauvegarder le meilleur modèle
if best_model["model"] is not None:
    model_path = f"{model_dir}/{best_model_name}_best.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model["model"], f)
    print(f"✓ Modèle sauvegardé: {model_path}")

# Sauvegarder les features
features_path = f"{model_dir}/features.json"
with open(features_path, "w") as f:
    json.dump(
        {
            "feature_names": feature_cols,
            "label_encoders": {k: list(v.classes_) for k, v in label_encoders.items()},
            "best_model": best_model_name,
            "test_auc": best_model["test_auc"],
        },
        f,
        indent=2,
    )
print(f"✓ Features sauvegardées: {features_path}")

# Sauvegarder tous les modèles
for name, res in results.items():
    if res["model"] is not None:
        path = f"{model_dir}/{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(res["model"], f)

print(f"\n✓ {len(results)} modèles sauvegardés dans {model_dir}/")

print("\n" + "=" * 120)
print("ANALYSE DE CALIBRATION")
print("=" * 120)

# Binning des prédictions
test_pred = best_model["predictions_test"]
bins = np.percentile(test_pred, [0, 20, 40, 60, 80, 100])
bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

test_df_copy = test_df.copy()
test_df_copy["pred_proba"] = test_pred
test_df_copy["bin"] = pd.cut(test_pred, bins=bins, labels=bin_labels, include_lowest=True)

calibration = (
    test_df_copy.groupby("bin")
    .agg({"target_place": ["count", "mean"], "pred_proba": "mean"})
    .round(3)
)

print("\nCalibration par quantile de prédiction:")
print(calibration)

print("\n" + "=" * 120)
print("SIMULATION DE STRATÉGIE DE PARIS")
print("=" * 120)

# Tester différents seuils
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]

for threshold in thresholds:
    # Filtrer les prédictions au-dessus du seuil
    mask = test_pred >= threshold

    if mask.sum() == 0:
        continue

    selected = test_df_copy[mask].copy()

    # Calculer ROI si on parie flat 10€
    stake = 10
    selected["gain"] = selected.apply(
        lambda row: stake * row["cote_finale"] if row["target_place"] == 1 else 0, axis=1
    )

    total_stake = len(selected) * stake
    total_return = selected["gain"].sum()
    profit = total_return - total_stake
    roi = (profit / total_stake * 100) if total_stake > 0 else 0

    win_rate = selected["target_place"].mean() * 100

    print(f"\nSeuil {threshold:.2f}:")
    print(f"  Paris: {len(selected)}")
    print(f"  Taux réussite: {win_rate:.1f}%")
    print(f"  Cote moyenne: {selected['cote_finale'].mean():.2f}")
    print(f"  Misé: {total_stake}€")
    print(f"  Retour: {total_return:.2f}€")
    print(f"  ROI: {roi:+.2f}%")

print("\n" + "=" * 120)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("=" * 120)
print(f"\nMeilleur modèle: {best_model_name.upper()} (AUC={best_model['test_auc']:.4f})")
print(f"Fichiers dans: {model_dir}/")
