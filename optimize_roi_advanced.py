#!/usr/bin/env python3
"""
OPTIMISATION AVANCÉE - Test sur multiples périodes et configurations
Pour trouver la configuration la plus robuste et optimale
"""

import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from itertools import product
import warnings

warnings.filterwarnings("ignore")

print("=" * 120)
print("OPTIMISATION AVANCÉE - MULTIPLES PÉRIODES & CONFIGURATIONS")
print("=" * 120)

# Charger toutes les données
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

print(f"✓ {len(df):,} courses chargées")

# Préparation
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

# Features dérivées
df["cote_log"] = np.log1p(df["cote_reference"])
df["cote_squared"] = df["cote_reference"] ** 2
df["is_favori"] = (df["cote_reference"] <= 5).astype(int)
df["is_outsider"] = (df["cote_reference"] >= 15).astype(int)
df["distance_cat_enc"] = pd.cut(
    df["distance_m"], bins=[0, 1600, 2200, 3000, 10000], labels=[0, 1, 2, 3]
).astype(int)
df["age_cote_interaction"] = df["age"] * df["cote_log"]

print("\n" + "=" * 120)
print("DÉFINITION DES PÉRIODES DE TEST")
print("=" * 120)

# Définir plusieurs périodes de test différentes
periods = [
    ("2025-11-01", "2025-11-30", "Nov 2025"),
    ("2025-12-01", "2025-12-31", "Dec 2025"),
    ("2025-11-01", "2025-12-15", "Nov-MiDec 2025"),
    ("2025-12-15", "2026-01-18", "MiDec-Jan 2026"),
    ("2025-10-01", "2025-10-31", "Oct 2025"),
    ("2025-09-01", "2025-09-30", "Sep 2025"),
    ("2025-08-01", "2025-08-31", "Aout 2025"),
    ("2025-11-01", "2026-01-18", "Nov-Jan 2026"),  # Période longue
]

print(f"\n✓ {len(periods)} périodes de test définies")

print("\n" + "=" * 120)
print("DÉFINITION DES CONFIGURATIONS À TESTER")
print("=" * 120)

# Configurations de features
feature_configs = {
    "minimal": ["cote_reference", "cote_log", "age", "distance_m"],
    "base": [
        "cote_reference",
        "cote_log",
        "age",
        "distance_m",
        "poids_kg",
        "discipline_enc",
        "sexe_enc",
        "hippodrome_code_enc",
    ],
    "complet": [
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
    ],
}

# Hyperparamètres XGBoost
xgb_configs = [
    {"max_depth": 4, "lr": 0.1, "n_est": 100},
    {"max_depth": 5, "lr": 0.05, "n_est": 200},
    {"max_depth": 6, "lr": 0.05, "n_est": 300},
    {"max_depth": 7, "lr": 0.03, "n_est": 400},
    {"max_depth": 8, "lr": 0.02, "n_est": 500},
]

# Filtres de cotes
cote_filters = [
    (0, 100, "Toutes"),
    (0, 5, "Favoris"),
    (3, 10, "Moyennes"),
    (5, 15, "Moyennes+"),
    (10, 100, "Outsiders"),
    (2, 8, "Équilibrées"),
]

# Seuils de prédiction
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

print(f"\n✓ {len(feature_configs)} sets de features")
print(f"✓ {len(xgb_configs)} configs XGBoost")
print(f"✓ {len(cote_filters)} filtres de cotes")
print(f"✓ {len(thresholds)} seuils")

total_configs = (
    len(periods) * len(feature_configs) * len(xgb_configs) * len(cote_filters) * len(thresholds)
)
print(f"\n➜ TOTAL: {total_configs:,} configurations à tester")

print("\n" + "=" * 120)
print("LANCEMENT DES TESTS (cela peut prendre plusieurs minutes)")
print("=" * 120)

import xgboost as xgb

results = []
config_counter = 0

for period_start, period_end, period_name in periods:
    print(f"\n[PÉRIODE: {period_name}]")

    # Préparer train/test pour cette période
    train_df = df[df["date"] < period_start].copy()
    test_df = df[(df["date"] >= period_start) & (df["date"] <= period_end)].copy()

    if len(test_df) < 100:
        print(f"  ⚠️  Période trop courte ({len(test_df)} courses), skip")
        continue

    # Stats hippodrome sur train
    hippodrome_stats = (
        train_df.groupby("hippodrome_code")
        .agg({"target_place": "mean", "cote_reference": "mean"})
        .reset_index()
    )
    hippodrome_stats.columns = ["hippodrome_code", "hippodrome_place_rate", "hippodrome_avg_cote"]

    train_df = train_df.merge(hippodrome_stats, on="hippodrome_code", how="left")
    test_df = test_df.merge(hippodrome_stats, on="hippodrome_code", how="left")

    train_df["hippodrome_place_rate"] = train_df["hippodrome_place_rate"].fillna(0.313)
    train_df["hippodrome_avg_cote"] = train_df["hippodrome_avg_cote"].fillna(
        df["cote_reference"].mean()
    )
    test_df["hippodrome_place_rate"] = test_df["hippodrome_place_rate"].fillna(0.313)
    test_df["hippodrome_avg_cote"] = test_df["hippodrome_avg_cote"].fillna(
        df["cote_reference"].mean()
    )

    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    for feat_name, features in feature_configs.items():
        # Ajouter stats hippodrome si pas dans features minimales
        if feat_name != "minimal":
            features_used = features + ["hippodrome_place_rate", "hippodrome_avg_cote"]
        else:
            features_used = features

        features_used = [f for f in features_used if f in test_df.columns]

        for xgb_cfg in xgb_configs:
            # Entraîner le modèle
            X_train = train_df[features_used].values
            y_train = train_df["target_place"].values
            X_test = test_df[features_used].values
            y_test = test_df["target_place"].values

            try:
                model = xgb.XGBClassifier(
                    max_depth=xgb_cfg["max_depth"],
                    learning_rate=xgb_cfg["lr"],
                    n_estimators=xgb_cfg["n_est"],
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train, verbose=False)
                pred = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, pred)

            except Exception as e:
                print(f"  ❌ Erreur modèle: {e}")
                continue

            # Tester toutes les combinaisons de filtres et seuils
            for cote_min, cote_max, cote_name in cote_filters:
                for threshold in thresholds:
                    config_counter += 1

                    mask = (
                        (pred >= threshold)
                        & (test_df["cote_reference"] >= cote_min)
                        & (test_df["cote_reference"] <= cote_max)
                    )

                    if mask.sum() < 10:  # Au moins 10 paris
                        continue

                    selected = test_df[mask]
                    stake = 10
                    total_stake = len(selected) * stake
                    total_return = (selected["target_place"] * selected["cote_place"] * stake).sum()
                    roi = (total_return - total_stake) / total_stake * 100

                    results.append(
                        {
                            "periode": period_name,
                            "features": feat_name,
                            "n_features": len(features_used),
                            "max_depth": xgb_cfg["max_depth"],
                            "lr": xgb_cfg["lr"],
                            "n_est": xgb_cfg["n_est"],
                            "cote_filter": cote_name,
                            "cote_min": cote_min,
                            "cote_max": cote_max,
                            "threshold": threshold,
                            "nb_paris": len(selected),
                            "win_rate": selected["target_place"].mean() * 100,
                            "roi": roi,
                            "auc": auc,
                        }
                    )

    print(f"  ✓ {config_counter} configs testées")

print("\n" + "=" * 120)
print("ANALYSE DES RÉSULTATS")
print("=" * 120)

df_results = pd.DataFrame(results)
print(f"\n✓ {len(df_results):,} résultats collectés")

# Top 30 meilleures configs
print("\n" + "=" * 120)
print("TOP 30 MEILLEURES CONFIGURATIONS (ROI)")
print("=" * 120)

top30 = df_results.nlargest(30, "roi")

print(
    f"\n{'#':3} | {'Période':15} | {'Features':8} | {'Depth':5} | {'LR':5} | {'Cote':12} | {'Seuil':6} | {'Paris':6} | {'Win%':6} | {'ROI':10} | {'AUC':6}"
)
print("-" * 125)

for i, row in enumerate(top30.itertuples(), 1):
    print(
        f"{i:3} | {row.periode:15} | {row.features:8} | {row.max_depth:5} | {row.lr:5.2f} | {row.cote_filter:12} | {row.threshold:6.2f} | {row.nb_paris:6} | {row.win_rate:5.1f}% | {row.roi:+9.2f}% | {row.auc:.4f}"
    )

# Analyse de robustesse: chercher les configs qui performent sur PLUSIEURS périodes
print("\n" + "=" * 120)
print("ANALYSE DE ROBUSTESSE - Configs performantes sur PLUSIEURS périodes")
print("=" * 120)

# Créer un identifiant de config
df_results["config_id"] = (
    df_results["features"]
    + "_"
    + df_results["max_depth"].astype(str)
    + "_"
    + df_results["lr"].astype(str)
    + "_"
    + df_results["cote_filter"]
    + "_"
    + df_results["threshold"].astype(str)
)

# Agréger par config
config_agg = (
    df_results.groupby("config_id")
    .agg(
        {
            "roi": ["mean", "std", "min", "max", "count"],
            "nb_paris": "sum",
            "win_rate": "mean",
            "auc": "mean",
        }
    )
    .reset_index()
)

config_agg.columns = [
    "config_id",
    "roi_mean",
    "roi_std",
    "roi_min",
    "roi_max",
    "n_periodes",
    "total_paris",
    "win_rate_mean",
    "auc_mean",
]

# Filtrer: au moins 3 périodes testées
robust_configs = config_agg[config_agg["n_periodes"] >= 3].copy()
robust_configs = robust_configs.sort_values("roi_mean", ascending=False).head(20)

print("\nConfigurations testées sur au moins 3 périodes :")
print(
    f"\n{'#':3} | {'Config':45} | {'Périodes':8} | {'ROI Moy':9} | {'ROI Std':8} | {'ROI Min':9} | {'ROI Max':9} | {'Paris':7} | {'Win%':6} | {'AUC':6}"
)
print("-" * 140)

for i, row in enumerate(robust_configs.itertuples(), 1):
    print(
        f"{i:3} | {row.config_id[:45]:45} | {int(row.n_periodes):8} | {row.roi_mean:+8.2f}% | {row.roi_std:7.2f}% | {row.roi_min:+8.2f}% | {row.roi_max:+8.2f}% | {int(row.total_paris):7} | {row.win_rate_mean:5.1f}% | {row.auc_mean:.4f}"
    )

# Meilleure config robuste
if len(robust_configs) > 0:
    best_robust = robust_configs.iloc[0]

    print("\n" + "=" * 120)
    print("🏆 MEILLEURE CONFIGURATION ROBUSTE")
    print("=" * 120)

    print(f"\nConfig: {best_robust['config_id']}")
    print(f"Testée sur {int(best_robust['n_periodes'])} périodes")
    print(f"ROI moyen: {best_robust['roi_mean']:+.2f}%")
    print(f"ROI écart-type: {best_robust['roi_std']:.2f}%")
    print(f"ROI min: {best_robust['roi_min']:+.2f}%")
    print(f"ROI max: {best_robust['roi_max']:+.2f}%")
    print(f"Total paris: {int(best_robust['total_paris'])}")
    print(f"Taux de réussite moyen: {best_robust['win_rate_mean']:.1f}%")
    print(f"AUC moyen: {best_robust['auc_mean']:.4f}")

    # Détail par période
    print("\nPerformance détaillée par période:")
    print(f"{'Période':15} | {'Paris':6} | {'Win%':6} | {'ROI':10}")
    print("-" * 45)

    detail = df_results[df_results["config_id"] == best_robust["config_id"]].sort_values("periode")
    for row in detail.itertuples():
        print(f"{row.periode:15} | {row.nb_paris:6} | {row.win_rate:5.1f}% | {row.roi:+9.2f}%")

print("\n" + "=" * 120)
print("✅ OPTIMISATION AVANCÉE TERMINÉE")
print("=" * 120)

# Sauvegarder les résultats
df_results.to_csv("optimization_results_detailed.csv", index=False)
print("\n✓ Résultats détaillés sauvegardés dans: optimization_results_detailed.csv")
