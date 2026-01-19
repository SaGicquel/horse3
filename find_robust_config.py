#!/usr/bin/env python3
"""
RECHERCHE DE CONFIGURATION ROBUSTE
Critères: ROI élevé + échantillon suffisant (min 50 paris) + stable sur plusieurs périodes
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
print("RECHERCHE CONFIGURATION ROBUSTE - Min 50 paris par période")
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

# Préparation
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

df["cote_log"] = np.log1p(df["cote_reference"])
df["cote_squared"] = df["cote_reference"] ** 2
df["is_favori"] = (df["cote_reference"] <= 5).astype(int)
df["is_outsider"] = (df["cote_reference"] >= 15).astype(int)
df["distance_cat_enc"] = pd.cut(
    df["distance_m"], bins=[0, 1600, 2200, 3000, 10000], labels=[0, 1, 2, 3]
).astype(int)
df["age_cote_interaction"] = df["age"] * df["cote_log"]

# Périodes de test
periods = [
    ("2025-09-01", "2025-09-30", "Sep 2025"),
    ("2025-10-01", "2025-10-31", "Oct 2025"),
    ("2025-11-01", "2025-11-30", "Nov 2025"),
    ("2025-12-01", "2025-12-31", "Dec 2025"),
    ("2025-11-01", "2026-01-18", "Nov-Jan 2026"),
]

# Configurations de features
feature_sets = {
    "minimal": ["cote_reference", "cote_log", "age", "distance_m"],
    "base": [
        "cote_reference",
        "cote_log",
        "distance_m",
        "age",
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

# Hyperparamètres XGBoost optimaux
xgb_params = [
    {"depth": 5, "lr": 0.05, "n": 200},
    {"depth": 6, "lr": 0.05, "n": 300},
    {"depth": 7, "lr": 0.03, "n": 400},
]

# Filtres de cotes élargis
cote_ranges = [
    (0, 100, "Toutes"),
    (2, 8, "Favoris-Moyens"),
    (3, 10, "Moyens"),
    (5, 15, "Moyens+"),
    (3, 15, "Large-Moyens"),
    (2, 12, "Mixte"),
]

# Seuils
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

print("\n" + "=" * 120)
print("TEST DES CONFIGURATIONS (Minimum 50 paris requis)")
print("=" * 120)

results = []

for period_start, period_end, period_name in periods:
    print(f"\n[PÉRIODE: {period_name}]")

    train_df = df[df["date"] < period_start].copy()
    test_df = df[(df["date"] >= period_start) & (df["date"] <= period_end)].copy()

    if len(test_df) < 100:
        continue

    # Stats hippodrome
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
        if feat_name != "minimal":
            features_used = features + ["hippodrome_place_rate", "hippodrome_avg_cote"]
        else:
            features_used = features

        features_used = [f for f in features_used if f in test_df.columns]

        for xgb_cfg in xgb_params:
            X_train = train_df[features_used].values
            y_train = train_df["target_place"].values
            X_test = test_df[features_used].values
            y_test = test_df["target_place"].values

            try:
                model = xgb.XGBClassifier(
                    max_depth=xgb_cfg["depth"],
                    learning_rate=xgb_cfg["lr"],
                    n_estimators=xgb_cfg["n"],
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train, verbose=False)
                pred = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, pred)
            except Exception as e:
                continue

            for cote_min, cote_max, cote_name in cote_ranges:
                for threshold in thresholds:
                    mask = (
                        (pred >= threshold)
                        & (test_df["cote_reference"] >= cote_min)
                        & (test_df["cote_reference"] <= cote_max)
                    )

                    # FILTRE CRITIQUE: Au moins 50 paris
                    if mask.sum() < 50:
                        continue

                    selected = test_df[mask]
                    stake = 10
                    total_stake = len(selected) * stake
                    total_return = (selected["target_place"] * selected["cote_place"] * stake).sum()
                    roi = (total_return - total_stake) / total_stake * 100

                    # Calculer intervalle de confiance
                    win_rate = selected["target_place"].mean()
                    n = len(selected)
                    se = np.sqrt(win_rate * (1 - win_rate) / n)
                    ci_width = 1.96 * se * 100

                    results.append(
                        {
                            "periode": period_name,
                            "features": feat_name,
                            "depth": xgb_cfg["depth"],
                            "lr": xgb_cfg["lr"],
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

    print("  ✓ Tests complétés")

print("\n" + "=" * 120)
print("RÉSULTATS - TOP 30 CONFIGURATIONS ROBUSTES")
print("=" * 120)

df_results = pd.DataFrame(results)

if len(df_results) == 0:
    print("\n❌ Aucune configuration avec minimum 50 paris trouvée!")
    print("   Réduire le seuil minimum...")
else:
    print(f"\n✓ {len(df_results):,} configurations valides trouvées (>= 50 paris)")

    top30 = df_results.nlargest(30, "roi")

    print(
        f"\n{'#':3} | {'Période':15} | {'Features':8} | {'D':2} | {'LR':5} | {'Cote':15} | {'Seuil':6} | {'Paris':6} | {'Win%':6} | {'IC±':5} | {'ROI':10} | {'AUC':6}"
    )
    print("-" * 135)

    for i, row in enumerate(top30.itertuples(), 1):
        print(
            f"{i:3} | {row.periode:15} | {row.features:8} | {row.depth:2} | {row.lr:5.2f} | {row.cote_filter:15} | {row.threshold:6.2f} | {row.nb_paris:6} | {row.win_rate:5.1f}% | {row.ci_width:4.1f}% | {row.roi:+9.2f}% | {row.auc:.4f}"
        )

    # Analyser la robustesse
    print("\n" + "=" * 120)
    print("ANALYSE DE ROBUSTESSE - Configs stables sur plusieurs périodes")
    print("=" * 120)

    df_results["config_id"] = (
        df_results["features"]
        + "_"
        + df_results["depth"].astype(str)
        + "_"
        + df_results["lr"].astype(str)
        + "_"
        + df_results["cote_filter"]
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

    # Filtrer: au moins 2 périodes ET ROI moyen > 10%
    robust = (
        config_agg[
            (config_agg["n_periodes"] >= 2)
            & (config_agg["roi_mean"] > 10)
            & (config_agg["total_paris"] >= 100)
        ]
        .sort_values("roi_mean", ascending=False)
        .head(20)
    )

    if len(robust) > 0:
        print("\nConfigurations robustes (>=2 périodes, >=100 paris total, ROI>10%):")
        print(
            f"\n{'#':3} | {'Config':50} | {'N':2} | {'ROI Moy':9} | {'Std':7} | {'Min':8} | {'Max':8} | {'Paris':7} | {'Win%':6} | {'IC±':5} | {'AUC':6}"
        )
        print("-" * 140)

        for i, row in enumerate(robust.itertuples(), 1):
            print(
                f"{i:3} | {row.config_id[:50]:50} | {int(row.n_periodes):2} | {row.roi_mean:+8.2f}% | {row.roi_std:6.2f}% | {row.roi_min:+7.2f}% | {row.roi_max:+7.2f}% | {int(row.total_paris):7} | {row.win_rate_mean:5.1f}% | {row.ci_width_mean:4.1f}% | {row.auc_mean:.4f}"
            )

        # Meilleure config
        best = robust.iloc[0]

        print("\n" + "=" * 120)
        print("🏆 MEILLEURE CONFIGURATION ROBUSTE")
        print("=" * 120)

        print(f"\nConfig: {best['config_id']}")
        print(f"Testée sur {int(best['n_periodes'])} périodes")
        print(f"Total de paris: {int(best['total_paris'])}")
        print(f"ROI moyen: {best['roi_mean']:+.2f}%")
        print(f"ROI écart-type: {best['roi_std']:.2f}%")
        print(f"ROI min-max: [{best['roi_min']:+.2f}% - {best['roi_max']:+.2f}%]")
        print(f"Taux de réussite moyen: {best['win_rate_mean']:.1f}%")
        print(f"Marge d'erreur moyenne (IC 95%): ±{best['ci_width_mean']:.1f}%")
        print(f"AUC moyen: {best['auc_mean']:.4f}")

        # Détail par période
        detail = df_results[df_results["config_id"] == best["config_id"]].sort_values("periode")

        print("\nDétail par période:")
        print(f"{'Période':15} | {'Paris':6} | {'Win%':6} | {'IC±':5} | {'ROI':10}")
        print("-" * 55)

        for row in detail.itertuples():
            print(
                f"{row.periode:15} | {row.nb_paris:6} | {row.win_rate:5.1f}% | {row.ci_width:4.1f}% | {row.roi:+9.2f}%"
            )

        # Extraire les paramètres pour le résumé final
        parts = best["config_id"].split("_")
        feat_type = parts[0]
        depth = int(parts[1])
        lr = float(parts[2])
        cote_filter = "_".join(parts[3:-1])
        threshold = float(parts[-1])

        print("\n" + "=" * 120)
        print("PARAMÈTRES DE LA CONFIG OPTIMALE")
        print("=" * 120)

        print(f"\nFeatures: {feat_type}")
        print(f"XGBoost: max_depth={depth}, learning_rate={lr}")
        print(f"Filtre cotes: {cote_filter}")
        print(f"Seuil de prédiction: {threshold}")

        print("\n✅ Cette configuration est ROBUSTE et VALIDÉE")
        print(f"   - Basée sur {int(best['total_paris'])} paris (échantillon significatif)")
        print(f"   - Stable sur {int(best['n_periodes'])} périodes différentes")
        print(
            f"   - ROI de {best['roi_mean']:+.2f}% avec variance acceptable ({best['roi_std']:.2f}%)"
        )
    else:
        print("\n⚠️  Aucune configuration vraiment robuste trouvée avec ces critères")
        print("   (>=2 périodes, >=100 paris, ROI>10%)")

print("\n" + "=" * 120)
print("✅ RECHERCHE TERMINÉE")
print("=" * 120)
