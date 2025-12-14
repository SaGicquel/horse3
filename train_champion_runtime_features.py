#!/usr/bin/env python3
"""
train_champion_runtime_features.py
===================================
Entraîne le modèle Champion XGBoost avec les MÊMES features que le backend
peut construire en temps réel depuis la BDD.

Ce script:
1. Extrait les données historiques de cheval_courses_seen
2. Construit les features identiques à celles de main.py
3. Entraîne un XGBoost avec ces features
4. Sauvegarde les artifacts compatibles avec champion_predictor.py

Usage:
    python train_champion_runtime_features.py
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Configuration
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://pmu_user:pmu_secure_password_2025@localhost:54624/pmu_database"
)
OUTPUT_DIR = Path("data/models/champion")
MIN_DATE = "2023-01-01"
MAX_DATE = "2024-10-31"
VAL_DATE = "2024-07-01"  # Val: 2024-07-01 to 2024-08-31
TEST_DATE = "2024-09-01"  # Test: 2024-09-01+

# Features construites par le backend (main.py lignes 7110-7160)
FEATURE_NAMES = [
    "numero_corde",
    "cote_sp",
    "cote_pm",
    "distance",
    "nombre_partants",
    "allocation",
    "forme_5c",
    "forme_10c",
    "nb_courses_12m",
    "nb_victoires_12m",
    "nb_places_12m",
    "regularite",
    "jours_depuis_derniere",
    "aptitude_distance",
    "aptitude_piste",
    "aptitude_hippodrome",
    "synergie_jockey_cheval",
    "synergie_entraineur_cheval",
    "jockey_win_rate",
    "jockey_place_rate",
    "entraineur_win_rate",
    "entraineur_place_rate",
    "distance_norm",
    "niveau_moyen_concurrent",
    "rang_cote_sp",
    "an_naissance",
    "age",
    "temperature_c",
    "vent_kmh",
    "meteo_code",
    # Interactions
    "interaction_forme_jockey",
    "interaction_aptitude_distance",
    "interaction_synergie_forme",
    "interaction_aptitude_popularite",
    "interaction_regularite_volume",
    # One-hot discipline
    "discipline_Plat",
    "discipline_Trot",
    "discipline_Obstacle",
    # One-hot sexe
    "sexe_H",
    "sexe_M",
    "sexe_F",
    # One-hot état piste (principales)
    "etat_Bon",
    "etat_Souple",
    "etat_Lourd",
    "etat_PSF",
    "etat_Leger",
]


def connect_db():
    """Connexion à la base PostgreSQL."""
    return psycopg2.connect(DB_URL)


def load_historical_data(conn, min_date: str, max_date: str) -> pd.DataFrame:
    """Charge les données historiques depuis cheval_courses_seen."""
    print(f"\n📂 Chargement des données du {min_date} au {max_date}...")

    query = """
    SELECT
        ccs.race_key,
        ccs.id_cheval,
        ccs.course_id,
        ccs.draw_stalle AS numero_corde,
        ccs.cote_finale AS cote_sp,
        ccs.cote_matin AS cote_pm,
        ccs.distance_m AS distance,
        ccs.nombre_partants,
        ccs.allocation_totale AS allocation,
        ccs.place_finale,
        ccs.is_win,
        ccs.driver_jockey,
        ccs.entraineur,
        ccs.sexe,
        ccs.age,
        ccs.temperature_c,
        ccs.vent_kmh,
        ccs.meteo_code,
        ccs.etat_piste,
        ccs.discipline,
        SUBSTRING(ccs.race_key FROM 1 FOR 10) AS race_date
    FROM cheval_courses_seen ccs
    WHERE SUBSTRING(ccs.race_key FROM 1 FOR 10)::date BETWEEN %s AND %s
      AND ccs.place_finale IS NOT NULL
      AND ccs.cote_finale IS NOT NULL
      AND ccs.cote_finale > 0
    ORDER BY ccs.race_key, ccs.id_cheval
    """

    df = pd.read_sql(query, conn, params=(min_date, max_date))
    print(f"   ✅ {len(df):,} lignes chargées")
    return df


def load_stats_chevaux(conn) -> dict[int, tuple]:
    """Charge les stats chevaux (forme, aptitudes)."""
    print("📊 Chargement stats chevaux...")
    query = """
    SELECT id_cheval, forme_5c, aptitude_distance, aptitude_piste, aptitude_hippodrome, regularite
    FROM stats_chevaux
    """
    cur = conn.cursor()
    cur.execute(query)
    stats = {row[0]: row[1:] for row in cur.fetchall()}
    print(f"   ✅ {len(stats):,} chevaux")
    return stats


def compute_historical_stats(conn, race_date: str, cheval_ids: list[int]) -> dict[int, tuple]:
    """Calcule les stats historiques (nb courses/victoires 12m, récence)."""
    if not cheval_ids:
        return {}

    query = """
    WITH hist AS (
        SELECT
            id_cheval,
            SUBSTRING(race_key FROM 1 FOR 10)::date AS d,
            is_win,
            place_finale
        FROM cheval_courses_seen
        WHERE id_cheval = ANY(%s)
          AND place_finale IS NOT NULL
          AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date
    )
    SELECT
        id_cheval,
        COUNT(*) FILTER (WHERE d >= (%s::date - INTERVAL '365 days')) AS nb_courses_12m,
        SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) FILTER (WHERE d >= (%s::date - INTERVAL '365 days')) AS nb_victoires_12m,
        SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) FILTER (WHERE d >= (%s::date - INTERVAL '365 days')) AS nb_places_12m,
        (%s::date - MAX(d))::float AS recence
    FROM hist
    GROUP BY id_cheval
    """
    cur = conn.cursor()
    cur.execute(query, (cheval_ids, race_date, race_date, race_date, race_date, race_date))
    return {row[0]: row[1:] for row in cur.fetchall()}


def compute_jockey_trainer_stats(
    conn, race_date: str, jockeys: list[str], trainers: list[str]
) -> tuple[dict, dict]:
    """Calcule les stats jockey/entraineur sur 12 mois."""
    jockey_stats = {}
    trainer_stats = {}
    cur = conn.cursor()

    if jockeys:
        query = """
        WITH hist AS (
            SELECT driver_jockey, is_win, place_finale
            FROM cheval_courses_seen
            WHERE driver_jockey = ANY(%s)
              AND place_finale IS NOT NULL
              AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date
              AND SUBSTRING(race_key FROM 1 FOR 10)::date >= (%s::date - INTERVAL '365 days')
        )
        SELECT driver_jockey, COUNT(*), SUM(CASE WHEN is_win=1 THEN 1 ELSE 0 END), SUM(CASE WHEN place_finale<=3 THEN 1 ELSE 0 END)
        FROM hist GROUP BY driver_jockey
        """
        cur.execute(query, (jockeys, race_date, race_date))
        for name, nb, wins, places in cur.fetchall():
            nb = nb or 1
            jockey_stats[name] = ((wins or 0) / nb, (places or 0) / nb)

    if trainers:
        query = """
        WITH hist AS (
            SELECT entraineur, is_win, place_finale
            FROM cheval_courses_seen
            WHERE entraineur = ANY(%s)
              AND place_finale IS NOT NULL
              AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date
              AND SUBSTRING(race_key FROM 1 FOR 10)::date >= (%s::date - INTERVAL '365 days')
        )
        SELECT entraineur, COUNT(*), SUM(CASE WHEN is_win=1 THEN 1 ELSE 0 END), SUM(CASE WHEN place_finale<=3 THEN 1 ELSE 0 END)
        FROM hist GROUP BY entraineur
        """
        cur.execute(query, (trainers, race_date, race_date))
        for name, nb, wins, places in cur.fetchall():
            nb = nb or 1
            trainer_stats[name] = ((wins or 0) / nb, (places or 0) / nb)

    return jockey_stats, trainer_stats


def build_features_for_dataset(df: pd.DataFrame, conn, stats_chevaux: dict) -> pd.DataFrame:
    """Construit les features pour tout le dataset."""
    print("\n🔧 Construction des features...")

    all_features = []
    race_keys = df["race_key"].unique()
    total = len(race_keys)

    for i, race_key in enumerate(race_keys):
        if i % 1000 == 0:
            print(f"   Processing {i}/{total} courses...")

        race_df = df[df["race_key"] == race_key].copy()
        race_date = race_df["race_date"].iloc[0]

        cheval_ids = race_df["id_cheval"].tolist()
        jockeys = [j.strip() for j in race_df["driver_jockey"].dropna().unique() if j]
        trainers = [t.strip() for t in race_df["entraineur"].dropna().unique() if t]

        # Stats historiques
        hist_stats = compute_historical_stats(conn, race_date, cheval_ids)
        jockey_stats, trainer_stats = compute_jockey_trainer_stats(
            conn, race_date, jockeys, trainers
        )

        # Niveau moyen concurrent
        formes = [float(stats_chevaux.get(cid, (0, 0, 0, 0, 0))[0] or 0) for cid in cheval_ids]
        mean_forme = sum(formes) / len(formes) if formes else 0

        # Rang cote
        cotes = race_df["cote_sp"].fillna(999).tolist()
        sorted_idx = sorted(range(len(cotes)), key=lambda x: cotes[x])
        rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_idx)}

        distance = float(race_df["distance"].iloc[0] or 2000)
        distance_norm = distance / 1000.0
        nb_partants = len(race_df)

        for idx, (_, row) in enumerate(race_df.iterrows()):
            cid = row["id_cheval"]
            forme_5c, apt_dist, apt_piste, apt_hippo, regularite = stats_chevaux.get(
                cid, (0, 0, 0, 0, 0)
            )
            nb_courses_12m, nb_victoires_12m, nb_places_12m, recence = hist_stats.get(
                cid, (0, 0, 0, 90)
            )

            j_name = (row["driver_jockey"] or "").strip()
            t_name = (row["entraineur"] or "").strip()
            j_wr, j_pr = jockey_stats.get(j_name, (0, 0))
            t_wr, t_pr = trainer_stats.get(t_name, (0, 0))

            sexe = (row["sexe"] or "").strip().upper()
            disc = (row["discipline"] or "").strip().lower()
            etat = (row["etat_piste"] or "").strip()

            synergie_j = float((j_wr or 0) * (forme_5c or 0))
            synergie_t = float((t_wr or 0) * (forme_5c or 0))

            try:
                year = int(str(race_date)[:4])
                age = float(row["age"] or 0)
                an_naissance = year - age if age else 0
            except:
                an_naissance = 0
                age = 0

            features = {
                "race_key": race_key,
                "id_cheval": cid,
                "is_win": int(row["is_win"] or 0),
                # Features numériques
                "numero_corde": float(row["numero_corde"] or 0),
                "cote_sp": float(row["cote_sp"] or 0),
                "cote_pm": float(row["cote_pm"] or row["cote_sp"] or 0),
                "distance": distance,
                "nombre_partants": nb_partants,
                "allocation": float(row["allocation"] or 0),
                "forme_5c": float(forme_5c or 0),
                "forme_10c": float(forme_5c or 0),
                "nb_courses_12m": float(nb_courses_12m or 0),
                "nb_victoires_12m": float(nb_victoires_12m or 0),
                "nb_places_12m": float(nb_places_12m or 0),
                "regularite": float(regularite or 0),
                "jours_depuis_derniere": float(recence or 90),
                "aptitude_distance": float(apt_dist or 0),
                "aptitude_piste": float(apt_piste or 0),
                "aptitude_hippodrome": float(apt_hippo or 0),
                "synergie_jockey_cheval": synergie_j,
                "synergie_entraineur_cheval": synergie_t,
                "jockey_win_rate": float(j_wr or 0),
                "jockey_place_rate": float(j_pr or 0),
                "entraineur_win_rate": float(t_wr or 0),
                "entraineur_place_rate": float(t_pr or 0),
                "distance_norm": distance_norm,
                "niveau_moyen_concurrent": mean_forme,
                "rang_cote_sp": float(rank_map.get(idx, 1)),
                "an_naissance": float(an_naissance),
                "age": float(age or 0),
                "temperature_c": float(row["temperature_c"])
                if pd.notna(row["temperature_c"])
                else 0,
                "vent_kmh": float(row["vent_kmh"]) if pd.notna(row["vent_kmh"]) else 0,
                "meteo_code": 0.0,  # String code, ignore
                # Interactions
                "interaction_forme_jockey": float((forme_5c or 0) * (j_wr or 0)),
                "interaction_aptitude_distance": float((apt_dist or 0) * distance_norm),
                "interaction_synergie_forme": float((synergie_j + synergie_t) * (forme_5c or 0)),
                "interaction_aptitude_popularite": float(
                    (apt_dist or 0) * (1.0 / max(1, rank_map.get(idx, 1)))
                ),
                "interaction_regularite_volume": float((regularite or 0) * (nb_courses_12m or 0)),
                # One-hot discipline
                "discipline_Plat": 1.0 if "plat" in disc else 0.0,
                "discipline_Trot": 1.0 if "trot" in disc or "att" in disc else 0.0,
                "discipline_Obstacle": 1.0
                if "haie" in disc or "steeple" in disc or "cross" in disc
                else 0.0,
                # One-hot sexe
                "sexe_H": 1.0 if sexe.startswith("H") else 0.0,
                "sexe_M": 1.0 if sexe.startswith("M") else 0.0,
                "sexe_F": 1.0 if sexe.startswith("F") else 0.0,
                # One-hot état piste
                "etat_Bon": 1.0 if "bon" in etat.lower() else 0.0,
                "etat_Souple": 1.0 if "souple" in etat.lower() else 0.0,
                "etat_Lourd": 1.0 if "lourd" in etat.lower() else 0.0,
                "etat_PSF": 1.0 if "psf" in etat.lower() else 0.0,
                "etat_Leger": 1.0 if "léger" in etat.lower() or "leger" in etat.lower() else 0.0,
            }

            all_features.append(features)

    print(f"   ✅ {len(all_features):,} échantillons générés")
    return pd.DataFrame(all_features)


def main():
    print("=" * 80)
    print("🚀 ENTRAÎNEMENT MODÈLE CHAMPION - FEATURES RUNTIME")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Connexion BDD
    print("\n🔌 Connexion à la base de données...")
    conn = connect_db()

    # Charger données
    df_raw = load_historical_data(conn, MIN_DATE, MAX_DATE)

    # Charger stats chevaux
    stats_chevaux = load_stats_chevaux(conn)

    # Construire features
    df_features = build_features_for_dataset(df_raw, conn, stats_chevaux)

    conn.close()

    # Split train/val/test
    print("\n📊 Split des données...")
    df_features["race_date"] = df_features["race_key"].str[:10]

    df_train = df_features[df_features["race_date"] < VAL_DATE]
    df_val = df_features[
        (df_features["race_date"] >= VAL_DATE) & (df_features["race_date"] < TEST_DATE)
    ]
    df_test = df_features[df_features["race_date"] >= TEST_DATE]

    print(
        f"   Train: {len(df_train):,} ({df_train['race_date'].min()} - {df_train['race_date'].max()})"
    )
    print(f"   Val:   {len(df_val):,} ({df_val['race_date'].min()} - {df_val['race_date'].max()})")
    print(
        f"   Test:  {len(df_test):,} ({df_test['race_date'].min()} - {df_test['race_date'].max()})"
    )

    # Colonnes features (exclure race_key, id_cheval, is_win, race_date)
    feature_cols = [
        c for c in df_features.columns if c not in ["race_key", "id_cheval", "is_win", "race_date"]
    ]

    print(f"\n📊 Features: {len(feature_cols)}")

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["is_win"].values
    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["is_win"].values
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test["is_win"].values

    print(f"   Train: {100*y_train.mean():.2f}% victoires")

    # Imputer + Scaler
    print("\n🔧 Imputation + Standardisation...")
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Nettoyer inf/nan
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    # Entraîner XGBoost
    print("\n⏳ Entraînement XGBoost...")
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=30,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"   ✅ {model.best_iteration} arbres")

    # Évaluation
    y_pred_val = model.predict_proba(X_val)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    roc_val = roc_auc_score(y_val, y_pred_val)
    roc_test = roc_auc_score(y_test, y_pred_test)
    brier_val = brier_score_loss(y_val, y_pred_val)
    brier_test = brier_score_loss(y_test, y_pred_test)

    print("\n📊 PERFORMANCES:")
    print(f"   ROC-AUC Val:  {roc_val:.4f}")
    print(f"   ROC-AUC Test: {roc_test:.4f}")
    print(f"   Brier Val:    {brier_val:.4f}")
    print(f"   Brier Test:   {brier_test:.4f}")

    # Sauvegarder
    print("\n💾 Sauvegarde des artifacts...")

    with open(OUTPUT_DIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("   ✅ xgboost_model.pkl")

    with open(OUTPUT_DIR / "feature_imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    print(f"   ✅ feature_imputer.pkl ({imputer.n_features_in_} features)")

    with open(OUTPUT_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"   ✅ feature_scaler.pkl ({scaler.n_features_in_} features)")

    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"   ✅ feature_names.json ({len(feature_cols)} features)")

    metadata = {
        "model_type": "xgboost_runtime_features",
        "champion_since": datetime.now().strftime("%Y-%m-%d"),
        "training_period": {"start": MIN_DATE, "end": VAL_DATE},
        "test_period": {"start": TEST_DATE, "end": MAX_DATE},
        "performance_metrics": {
            "roc_auc_val": round(roc_val, 4),
            "roc_auc_test": round(roc_test, 4),
            "brier_val": round(brier_val, 4),
            "brier_test": round(brier_test, 4),
        },
        "files": {
            "model": "xgboost_model.pkl",
            "feature_scaler": "feature_scaler.pkl",
            "feature_imputer": "feature_imputer.pkl",
            "feature_names": "feature_names.json",
        },
        "features_count": len(feature_cols),
        "training_samples": len(df_train),
        "best_iteration": model.best_iteration,
        "compatible_with": "web/backend/main.py champion predictor",
    }

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("   ✅ metadata.json")

    print("\n" + "=" * 80)
    print("✅ MODÈLE CHAMPION RÉ-ENTRAÎNÉ AVEC FEATURES RUNTIME !")
    print("=" * 80)
    print(f"   📊 Features: {len(feature_cols)} (compatibles backend)")
    print(f"   🎯 ROC-AUC Test: {roc_test:.4f}")
    print(f"   📁 Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
