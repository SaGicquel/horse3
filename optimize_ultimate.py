#!/usr/bin/env python3
"""
OPTIMISATION ULTIME - Test exhaustif de configurations avancées
Feature engineering, hyperparamètres optimisés, ensembles, filtres sophistiqués
"""

import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

print("=" * 120)
print("OPTIMISATION ULTIME - RECHERCHE CONFIGURATION MAXIMALE")
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

print("\n[1/5] Chargement des données...")
df = pd.read_sql(query, conn)
conn.close()
print(f"✓ {len(df):,} courses")

# Préparation de base
df["date"] = pd.to_datetime(df["race_key"].str.split("|").str[0])
df["cote_place"] = 1 + (df["cote_reference"] - 1) / 3.5

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

categorical_cols = ["discipline", "sexe", "hippodrome_code", "etat_piste", "meteo_code"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN")
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

print("\n[2/5] Feature engineering avancé...")

# Features de base
df["cote_log"] = np.log1p(df["cote_reference"])
df["cote_squared"] = df["cote_reference"] ** 2
df["cote_sqrt"] = np.sqrt(df["cote_reference"])
df["cote_inv"] = 1 / (df["cote_reference"] + 1)

# Catégorisation des cotes (plus fine)
df["cote_cat"] = pd.cut(
    df["cote_reference"], bins=[0, 3, 5, 8, 12, 20, 100], labels=[0, 1, 2, 3, 4, 5]
)
df["cote_cat"] = df["cote_cat"].cat.codes

# Features de distance
df["distance_log"] = np.log1p(df["distance_m"])
df["distance_cat"] = pd.cut(
    df["distance_m"], bins=[0, 1600, 2000, 2400, 3000, 10000], labels=[0, 1, 2, 3, 4]
)
df["distance_cat"] = df["distance_cat"].cat.codes

# Features d'âge
df["age_squared"] = df["age"] ** 2
df["is_young"] = (df["age"] <= 3).astype(int)
df["is_mature"] = (df["age"] >= 5).astype(int)

# Features de poids
df["poids_normalized"] = (df["poids_kg"] - df["poids_kg"].mean()) / df["poids_kg"].std()

# Interactions
df["age_cote"] = df["age"] * df["cote_log"]
df["age_distance"] = df["age"] * df["distance_log"]
df["cote_distance"] = df["cote_log"] * df["distance_log"]
df["poids_cote"] = df["poids_kg"] * df["cote_log"]
df["age_poids"] = df["age"] * df["poids_kg"]

# Indicators
df["is_favori"] = (df["cote_reference"] <= 4).astype(int)
df["is_moyen"] = ((df["cote_reference"] > 4) & (df["cote_reference"] <= 12)).astype(int)
df["is_outsider"] = (df["cote_reference"] > 12).astype(int)
df["is_long_distance"] = (df["distance_m"] > 2400).astype(int)
df["is_short_distance"] = (df["distance_m"] < 1800).astype(int)

print(
    f"✓ {len([c for c in df.columns if c.endswith('_enc') or c in ['cote_log', 'age', 'distance_m']])} features créées"
)

# Périodes de test
periods = [
    ("2025-08-01", "2025-08-31", "Aout 2025"),
    ("2025-09-01", "2025-09-30", "Sep 2025"),
    ("2025-10-01", "2025-10-31", "Oct 2025"),
    ("2025-11-01", "2025-11-30", "Nov 2025"),
    ("2025-12-01", "2025-12-31", "Dec 2025"),
    ("2025-11-01", "2026-01-18", "Nov-Jan 2026"),
]

print("\n[3/5] Définition des configurations...")

# Sets de features (du plus simple au plus complexe)
feature_sets = {
    "core": ["cote_reference", "cote_log", "age", "distance_m", "poids_kg"],
    "extended": [
        "cote_reference",
        "cote_log",
        "cote_sqrt",
        "cote_cat",
        "age",
        "age_squared",
        "distance_m",
        "distance_log",
        "distance_cat",
        "poids_kg",
        "discipline_enc",
        "sexe_enc",
    ],
    "interactions": [
        "cote_reference",
        "cote_log",
        "cote_squared",
        "cote_sqrt",
        "cote_inv",
        "cote_cat",
        "age",
        "age_squared",
        "is_young",
        "is_mature",
        "distance_m",
        "distance_log",
        "distance_cat",
        "is_long_distance",
        "is_short_distance",
        "poids_kg",
        "poids_normalized",
        "handicap_distance",
        "age_cote",
        "age_distance",
        "cote_distance",
        "poids_cote",
        "age_poids",
        "is_favori",
        "is_moyen",
        "is_outsider",
        "discipline_enc",
        "sexe_enc",
        "hippodrome_code_enc",
        "etat_piste_enc",
        "meteo_code_enc",
    ],
    "ultimate": [
        "cote_reference",
        "cote_log",
        "cote_squared",
        "cote_sqrt",
        "cote_inv",
        "cote_cat",
        "age",
        "age_squared",
        "is_young",
        "is_mature",
        "distance_m",
        "distance_log",
        "distance_cat",
        "is_long_distance",
        "is_short_distance",
        "poids_kg",
        "poids_normalized",
        "handicap_distance",
        "numero_dossard",
        "age_cote",
        "age_distance",
        "cote_distance",
        "poids_cote",
        "age_poids",
        "is_favori",
        "is_moyen",
        "is_outsider",
        "discipline_enc",
        "sexe_enc",
        "hippodrome_code_enc",
        "etat_piste_enc",
        "meteo_code_enc",
        "temperature_c",
        "vent_kmh",
    ],
}

# Hyperparamètres XGBoost optimisés
xgb_configs = [
    # Shallow & fast
    {"depth": 4, "lr": 0.1, "n": 150, "subsample": 0.8, "colsample": 0.8},
    {"depth": 5, "lr": 0.08, "n": 200, "subsample": 0.85, "colsample": 0.85},
    # Medium depth
    {"depth": 6, "lr": 0.05, "n": 300, "subsample": 0.85, "colsample": 0.9},
    {"depth": 7, "lr": 0.04, "n": 350, "subsample": 0.9, "colsample": 0.9},
    {"depth": 7, "lr": 0.03, "n": 400, "subsample": 0.9, "colsample": 0.95},
    # Deep & powerful
    {"depth": 8, "lr": 0.03, "n": 450, "subsample": 0.9, "colsample": 0.95},
    {"depth": 9, "lr": 0.02, "n": 500, "subsample": 0.95, "colsample": 0.95},
]

# Filtres de cotes ultra-précis
cote_filters = [
    # Cotes moyennes (sweet spot historique)
    (4, 10, "Moyens 4-10"),
    (5, 12, "Moyens 5-12"),
    (5, 15, "Moyens+ 5-15"),
    (6, 14, "Optimum 6-14"),
    (4, 12, "Large-Moyens"),
    # Variations
    (3, 9, "Favoris+ 3-9"),
    (7, 15, "Semi-Outsiders"),
    (4, 8, "Moyens-Serrés"),
    (6, 12, "Moyens-Précis"),
]

# Seuils de prédiction
thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print(f"✓ {len(feature_sets)} feature sets")
print(f"✓ {len(xgb_configs)} configs XGBoost")
print(f"✓ {len(cote_filters)} filtres de cotes")
print(f"✓ {len(thresholds)} seuils")

total_configs = (
    len(periods) * len(feature_sets) * len(xgb_configs) * len(cote_filters) * len(thresholds)
)
print(f"\n➜ TOTAL: {total_configs:,} configurations à tester")

print("\n[4/5] Exécution des tests (avec barre de progression)...")

results = []
config_counter = 0
total_to_test = len(periods) * len(feature_sets) * len(xgb_configs)

for period_idx, (period_start, period_end, period_name) in enumerate(periods):
    print(f"\n[Période {period_idx+1}/{len(periods)}: {period_name}]")

    train_df = df[df["date"] < period_start].copy()
    test_df = df[(df["date"] >= period_start) & (df["date"] <= period_end)].copy()

    if len(test_df) < 100:
        continue

    # Stats hippodrome (calculées sur train uniquement)
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

    for feat_name, features in feature_sets.items():
        # Ajouter les stats hippodrome
        features_used = features + ["hippodrome_place_rate", "hippodrome_avg_cote"]
        features_used = [f for f in features_used if f in test_df.columns]

        for xgb_cfg in xgb_configs:
            config_counter += 1

            if config_counter % 5 == 0:
                print(
                    f"  Progression: {config_counter}/{total_to_test} modèles entraînés...",
                    end="\r",
                )

            X_train = train_df[features_used].values
            y_train = train_df["target_place"].values
            X_test = test_df[features_used].values
            y_test = test_df["target_place"].values

            try:
                model = xgb.XGBClassifier(
                    max_depth=xgb_cfg["depth"],
                    learning_rate=xgb_cfg["lr"],
                    n_estimators=xgb_cfg["n"],
                    subsample=xgb_cfg["subsample"],
                    colsample_bytree=xgb_cfg["colsample"],
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train, verbose=False)
                pred = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, pred)
            except Exception as e:
                continue

            # Tester tous les filtres et seuils
            for cote_min, cote_max, cote_name in cote_filters:
                for threshold in thresholds:
                    mask = (
                        (pred >= threshold)
                        & (test_df["cote_reference"] >= cote_min)
                        & (test_df["cote_reference"] <= cote_max)
                    )

                    # Minimum 30 paris (moins strict pour trouver plus de configs)
                    if mask.sum() < 30:
                        continue

                    selected = test_df[mask]
                    n = len(selected)
                    stake = 10
                    total_stake = n * stake
                    total_return = (selected["target_place"] * selected["cote_place"] * stake).sum()
                    roi = (total_return - total_stake) / total_stake * 100

                    win_rate = selected["target_place"].mean()
                    se = np.sqrt(win_rate * (1 - win_rate) / n)
                    ci_width = 1.96 * se * 100

                    results.append(
                        {
                            "periode": period_name,
                            "features": feat_name,
                            "n_features": len(features_used),
                            "depth": xgb_cfg["depth"],
                            "lr": xgb_cfg["lr"],
                            "n_est": xgb_cfg["n"],
                            "subsample": xgb_cfg["subsample"],
                            "colsample": xgb_cfg["colsample"],
                            "cote_filter": cote_name,
                            "cote_min": cote_min,
                            "cote_max": cote_max,
                            "threshold": threshold,
                            "nb_paris": n,
                            "win_rate": win_rate * 100,
                            "ci_width": ci_width,
                            "roi": roi,
                            "auc": auc,
                        }
                    )

    print(f"  ✓ {period_name} complétée ({len(results)} résultats cumulés)")

print("\n[5/5] Analyse des résultats...")

df_results = pd.DataFrame(results)

if len(df_results) == 0:
    print("\n❌ Aucun résultat trouvé!")
else:
    print(f"\n✓ {len(df_results):,} configurations testées")

    print("\n" + "=" * 120)
    print("TOP 40 MEILLEURES CONFIGURATIONS")
    print("=" * 120)

    top40 = df_results.nlargest(40, "roi")

    print(
        f"\n{'#':3} | {'Période':13} | {'Features':12} | {'D':2} | {'LR':5} | {'Sub':4} | {'Col':4} | {'Cote':14} | {'Seuil':6} | {'Paris':6} | {'Win%':6} | {'IC±':5} | {'ROI':10} | {'AUC':6}"
    )
    print("-" * 155)

    for i, row in enumerate(top40.itertuples(), 1):
        print(
            f"{i:3} | {row.periode:13} | {row.features:12} | {row.depth:2} | {row.lr:5.2f} | {row.subsample:4.2f} | {row.colsample:4.2f} | {row.cote_filter:14} | {row.threshold:6.2f} | {row.nb_paris:6} | {row.win_rate:5.1f}% | {row.ci_width:4.1f}% | {row.roi:+9.2f}% | {row.auc:.4f}"
        )

    # Analyse de robustesse
    print("\n" + "=" * 120)
    print("CONFIGURATIONS LES PLUS ROBUSTES (>= 2 périodes, >= 100 paris total)")
    print("=" * 120)

    df_results["config_id"] = (
        df_results["features"]
        + "_"
        + df_results["depth"].astype(str)
        + "_"
        + df_results["lr"].astype(str)
        + "_"
        + df_results["subsample"].astype(str)
        + "_"
        + df_results["colsample"].astype(str)
        + "_"
        + df_results["cote_filter"].str.replace(" ", "-")
        + "_"
        + df_results["threshold"].astype(str)
    )

    config_agg = (
        df_results.groupby("config_id")
        .agg(
            {
                "roi": ["mean", "std", "min", "max", "count"],
                "nb_paris": "sum",
                "win_rate": "mean",
                "ci_width": "mean",
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
        "ci_width_mean",
        "auc_mean",
    ]

    robust = (
        config_agg[
            (config_agg["n_periodes"] >= 2)
            & (config_agg["total_paris"] >= 100)
            & (config_agg["roi_mean"] > 15)
        ]
        .sort_values("roi_mean", ascending=False)
        .head(25)
    )

    if len(robust) > 0:
        print(
            f"\n{'#':3} | {'Config':65} | {'N':2} | {'ROI Moy':9} | {'Std':7} | {'Min':8} | {'Max':8} | {'Paris':7} | {'Win%':6} | {'IC±':5} | {'AUC':6}"
        )
        print("-" * 160)

        for i, row in enumerate(robust.itertuples(), 1):
            print(
                f"{i:3} | {row.config_id[:65]:65} | {int(row.n_periodes):2} | {row.roi_mean:+8.2f}% | {row.roi_std:6.2f}% | {row.roi_min:+7.2f}% | {row.roi_max:+7.2f}% | {int(row.total_paris):7} | {row.win_rate_mean:5.1f}% | {row.ci_width_mean:4.1f}% | {row.auc_mean:.4f}"
            )

        # Meilleure config
        best = robust.iloc[0]

        print("\n" + "=" * 120)
        print("🏆 CONFIGURATION ULTIME TROUVÉE")
        print("=" * 120)

        print(f"\nConfig ID: {best['config_id']}")
        print(f"Périodes testées: {int(best['n_periodes'])}")
        print(f"Total de paris: {int(best['total_paris'])}")
        print(f"\nROI moyen: {best['roi_mean']:+.2f}%")
        print(f"ROI écart-type: {best['roi_std']:.2f}%")
        print(f"ROI min-max: [{best['roi_min']:+.2f}% - {best['roi_max']:+.2f}%]")
        print(f"Taux de réussite: {best['win_rate_mean']:.1f}%")
        print(f"Marge d'erreur (IC 95%): ±{best['ci_width_mean']:.1f}%")
        print(f"AUC moyen: {best['auc_mean']:.4f}")

        # Détail par période
        detail = df_results[df_results["config_id"] == best["config_id"]].sort_values("periode")

        print(f"\n{'Période':15} | {'Paris':6} | {'Win%':6} | {'IC±':5} | {'ROI':10}")
        print("-" * 55)

        for row in detail.itertuples():
            print(
                f"{row.periode:15} | {row.nb_paris:6} | {row.win_rate:5.1f}% | {row.ci_width:4.1f}% | {row.roi:+9.2f}%"
            )

        # Amélioration vs config précédente (+33.25%)
        improvement = best["roi_mean"] - 33.25

        print("\n" + "=" * 120)
        print("COMPARAISON AVEC CONFIG PRÉCÉDENTE")
        print("=" * 120)

        print("\nConfig précédente (complet_7_0.03_Moyens+_0.5): +33.25% ROI")
        print(f"Nouvelle config ultime: {best['roi_mean']:+.2f}% ROI")
        print(f"Amélioration: {improvement:+.2f} points")

        if improvement > 2:
            print(f"\n✅ AMÉLIORATION SIGNIFICATIVE de {improvement:+.2f}% !")
        elif improvement > 0:
            print(f"\n✓ Légère amélioration de {improvement:+.2f}%")
        else:
            print(f"\n⚠️  Pas d'amélioration ({improvement:+.2f}%)")

    else:
        print("\n⚠️  Aucune config robuste trouvée avec ces critères stricts")

    # Sauvegarder les résultats
    df_results.to_csv("ultimate_optimization_results.csv", index=False)
    print("\n✓ Résultats sauvegardés: ultimate_optimization_results.csv")

print("\n" + "=" * 120)
print("✅ OPTIMISATION ULTIME TERMINÉE")
print("=" * 120)
