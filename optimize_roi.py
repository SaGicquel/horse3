#!/usr/bin/env python3
"""
OPTIMISATION COMPLÈTE - Recherche de la meilleure configuration pour maximiser le ROI
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
print("OPTIMISATION COMPLÈTE - RECHERCHE MEILLEUR ROI")
print("=" * 120)

# Charger les données
conn = get_connection()

query = """
SELECT
    nom_norm,
    race_key,
    annee,
    CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as target_place,
    place_finale,
    cote_reference,
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

print("\nChargement des données...")
df = pd.read_sql(query, conn)
conn.close()

print(f"✓ {len(df):,} courses")

# Préparer les données de base
df["date"] = pd.to_datetime(df["race_key"].str.split("|").str[0])
df["cote_place"] = 1 + (df["cote_reference"] - 1) / 3.5

# Features numériques
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

# Split
train_df = df[df["date"] < "2025-11-01"].copy()
val_df = df[(df["date"] >= "2025-11-01") & (df["date"] < "2025-12-15")].copy()
test_df = df[df["date"] >= "2025-12-15"].copy()

print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# Stats hippodrome sur train
hippodrome_stats = (
    train_df.groupby("hippodrome_code")
    .agg({"target_place": "mean", "cote_reference": "mean"})
    .reset_index()
)
hippodrome_stats.columns = ["hippodrome_code", "hippodrome_place_rate", "hippodrome_avg_cote"]

for split_df in [train_df, val_df, test_df]:
    split_df = split_df.merge(hippodrome_stats, on="hippodrome_code", how="left")
    split_df["hippodrome_place_rate"] = split_df["hippodrome_place_rate"].fillna(0.313)
    split_df["hippodrome_avg_cote"] = split_df["hippodrome_avg_cote"].fillna(
        df["cote_reference"].mean()
    )

# Réassigner après merge
train_df = df[df["date"] < "2025-11-01"].merge(hippodrome_stats, on="hippodrome_code", how="left")
val_df = df[(df["date"] >= "2025-11-01") & (df["date"] < "2025-12-15")].merge(
    hippodrome_stats, on="hippodrome_code", how="left"
)
test_df = df[df["date"] >= "2025-12-15"].merge(hippodrome_stats, on="hippodrome_code", how="left")

for split_df in [train_df, val_df, test_df]:
    split_df["hippodrome_place_rate"] = split_df["hippodrome_place_rate"].fillna(0.313)
    split_df["hippodrome_avg_cote"] = split_df["hippodrome_avg_cote"].fillna(
        df["cote_reference"].mean()
    )

print("\n" + "=" * 120)
print("TEST DE DIFFÉRENTES CONFIGURATIONS")
print("=" * 120)

results_all = []

# CONFIGURATION 1: Features minimales (baseline)
print("\n[CONFIG 1] Features minimales...")

features_minimal = ["cote_reference", "age", "distance_m"]
train_df["cote_log"] = np.log1p(train_df["cote_reference"])
val_df["cote_log"] = np.log1p(val_df["cote_reference"])
test_df["cote_log"] = np.log1p(test_df["cote_reference"])
features_minimal.append("cote_log")

X_train = train_df[features_minimal].values
y_train = train_df["target_place"].values
X_test = test_df[features_minimal].values
y_test = test_df["target_place"].values

try:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        max_depth=4, learning_rate=0.1, n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    pred = model.predict_proba(X_test)[:, 1]

    for threshold in [0.35, 0.40, 0.45]:
        mask = pred >= threshold
        if mask.sum() > 0:
            selected = test_df[mask]
            stake = 10
            total_stake = len(selected) * stake
            total_return = (selected["target_place"] * selected["cote_place"] * stake).sum()
            roi = (total_return - total_stake) / total_stake * 100

            results_all.append(
                {
                    "config": "Minimal",
                    "features": len(features_minimal),
                    "threshold": threshold,
                    "nb_paris": len(selected),
                    "win_rate": selected["target_place"].mean() * 100,
                    "roi": roi,
                    "auc": roc_auc_score(y_test, pred),
                }
            )
    print(f"  ✓ AUC: {roc_auc_score(y_test, pred):.4f}")
except Exception as e:
    print(f"  ❌ {e}")

# CONFIGURATION 2: Features complètes
print("\n[CONFIG 2] Features complètes...")

for split_df in [train_df, val_df, test_df]:
    split_df["cote_squared"] = split_df["cote_reference"] ** 2
    split_df["is_favori"] = (split_df["cote_reference"] <= 5).astype(int)
    split_df["is_outsider"] = (split_df["cote_reference"] >= 15).astype(int)
    split_df["distance_cat_enc"] = pd.cut(
        split_df["distance_m"], bins=[0, 1600, 2200, 3000, 10000], labels=[0, 1, 2, 3]
    ).astype(int)
    split_df["age_cote_interaction"] = split_df["age"] * split_df["cote_log"]

features_full = [
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

features_full = [f for f in features_full if f in test_df.columns]

X_train = train_df[features_full].values
X_test = test_df[features_full].values

try:
    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=300, subsample=0.8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    pred = model.predict_proba(X_test)[:, 1]

    for threshold in [0.35, 0.40, 0.45]:
        mask = pred >= threshold
        if mask.sum() > 0:
            selected = test_df[mask]
            stake = 10
            total_stake = len(selected) * stake
            total_return = (selected["target_place"] * selected["cote_place"] * stake).sum()
            roi = (total_return - total_stake) / total_stake * 100

            results_all.append(
                {
                    "config": "Complet",
                    "features": len(features_full),
                    "threshold": threshold,
                    "nb_paris": len(selected),
                    "win_rate": selected["target_place"].mean() * 100,
                    "roi": roi,
                    "auc": roc_auc_score(y_test, pred),
                }
            )
    print(f"  ✓ AUC: {roc_auc_score(y_test, pred):.4f}")
except Exception as e:
    print(f"  ❌ {e}")

# CONFIGURATION 3-5: Différents modèles
print("\n[CONFIG 3-5] Test de différents modèles...")

models_config = [("LightGBM", "lgb"), ("CatBoost", "cat"), ("XGBoost-Optimisé", "xgb_opt")]

for model_name, model_type in models_config:
    print(f"\n  {model_name}...")
    try:
        if model_type == "lgb":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                max_depth=7,
                learning_rate=0.03,
                n_estimators=400,
                subsample=0.9,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        elif model_type == "cat":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                depth=7, learning_rate=0.03, iterations=400, random_seed=42, verbose=False
            )
        else:  # xgb_opt
            model = xgb.XGBClassifier(
                max_depth=7,
                learning_rate=0.03,
                n_estimators=400,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
            )

        model.fit(X_train, y_train, verbose=False)
        pred = model.predict_proba(X_test)[:, 1]

        for threshold in [0.35, 0.40, 0.45, 0.50]:
            mask = pred >= threshold
            if mask.sum() > 0:
                selected = test_df[mask]
                stake = 10
                total_stake = len(selected) * stake
                total_return = (selected["target_place"] * selected["cote_place"] * stake).sum()
                roi = (total_return - total_stake) / total_stake * 100

                results_all.append(
                    {
                        "config": model_name,
                        "features": len(features_full),
                        "threshold": threshold,
                        "nb_paris": len(selected),
                        "win_rate": selected["target_place"].mean() * 100,
                        "roi": roi,
                        "auc": roc_auc_score(y_test, pred),
                    }
                )
        print(f"    ✓ AUC: {roc_auc_score(y_test, pred):.4f}")
    except Exception as e:
        print(f"    ❌ {e}")

# CONFIGURATION 6: Ensemble
print("\n[CONFIG 6] Ensemble de modèles...")

try:
    # Entraîner 3 modèles
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    model1 = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=-1
    )
    model2 = lgb.LGBMClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=-1, verbose=-1
    )
    model3 = CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=300, random_seed=42, verbose=False
    )

    model1.fit(X_train, y_train, verbose=False)
    model2.fit(X_train, y_train, verbose=False)
    model3.fit(X_train, y_train, verbose=False)

    pred1 = model1.predict_proba(X_test)[:, 1]
    pred2 = model2.predict_proba(X_test)[:, 1]
    pred3 = model3.predict_proba(X_test)[:, 1]

    pred_ensemble = (pred1 + pred2 + pred3) / 3

    for threshold in [0.35, 0.40, 0.45, 0.50]:
        mask = pred_ensemble >= threshold
        if mask.sum() > 0:
            selected = test_df[mask]
            stake = 10
            total_stake = len(selected) * stake
            total_return = (selected["target_place"] * selected["cote_place"] * stake).sum()
            roi = (total_return - total_stake) / total_stake * 100

            results_all.append(
                {
                    "config": "Ensemble",
                    "features": len(features_full),
                    "threshold": threshold,
                    "nb_paris": len(selected),
                    "win_rate": selected["target_place"].mean() * 100,
                    "roi": roi,
                    "auc": roc_auc_score(y_test, pred_ensemble),
                }
            )
    print(f"  ✓ AUC: {roc_auc_score(y_test, pred_ensemble):.4f}")
except Exception as e:
    print(f"  ❌ {e}")

# CONFIGURATION 7: Filtrage par cote
print("\n[CONFIG 7] Filtrage par tranches de cotes...")

try:
    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=300, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    pred = model.predict_proba(X_test)[:, 1]

    for cote_min, cote_max in [(0, 5), (3, 10), (5, 15)]:
        for threshold in [0.35, 0.40, 0.45]:
            mask = (
                (pred >= threshold)
                & (test_df["cote_reference"] >= cote_min)
                & (test_df["cote_reference"] <= cote_max)
            )
            if mask.sum() > 0:
                selected = test_df[mask]
                stake = 10
                total_stake = len(selected) * stake
                total_return = (selected["target_place"] * selected["cote_place"] * stake).sum()
                roi = (total_return - total_stake) / total_stake * 100

                results_all.append(
                    {
                        "config": f"Cote {cote_min}-{cote_max}",
                        "features": len(features_full),
                        "threshold": threshold,
                        "nb_paris": len(selected),
                        "win_rate": selected["target_place"].mean() * 100,
                        "roi": roi,
                        "auc": roc_auc_score(y_test, pred),
                    }
                )
    print("  ✓ Tests sur tranches de cotes terminés")
except Exception as e:
    print(f"  ❌ {e}")

print("\n" + "=" * 120)
print("RÉSULTATS - TOP 20 MEILLEURES CONFIGURATIONS")
print("=" * 120)

df_results = pd.DataFrame(results_all)
df_results = df_results.sort_values("roi", ascending=False).head(20)

print(
    f"\n{'Rang':4} | {'Config':20} | {'Seuil':6} | {'Paris':6} | {'Win%':6} | {'ROI':10} | {'AUC':6}"
)
print("-" * 85)

for i, row in enumerate(df_results.itertuples(), 1):
    print(
        f"{i:4} | {row.config:20} | {row.threshold:6.2f} | {row.nb_paris:6} | {row.win_rate:5.1f}% | {row.roi:+9.2f}% | {row.auc:.4f}"
    )

# Meilleure config
best = df_results.iloc[0]

print("\n" + "=" * 120)
print("🏆 MEILLEURE CONFIGURATION TROUVÉE")
print("=" * 120)

print(f"\nConfiguration: {best['config']}")
print(f"Seuil optimal: {best['threshold']:.2f}")
print(f"Nombre de paris: {best['nb_paris']}")
print(f"Taux de réussite: {best['win_rate']:.1f}%")
print(f"ROI: {best['roi']:+.2f}%")
print(f"AUC: {best['auc']:.4f}")

print("\n" + "=" * 120)
print("✅ OPTIMISATION TERMINÉE")
print("=" * 120)
