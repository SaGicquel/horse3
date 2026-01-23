import sys
import os
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import time
from datetime import datetime
from db_connection import get_connection

# Mock ALGO_CONFIG
ALGO_CONFIG = {
    "features_base": ["cote_reference", "cote_log", "distance_m", "age", "poids_kg"],
    "xgb_params": {
        "max_depth": 7,
        "learning_rate": 0.04,
        "n_estimators": 350,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
    },
    "cote_min": 7,
    "cote_max": 15,
    "threshold": 0.50,  # 50% de probabilité
    "mise_uniforme": 10.0,
}


def train_model_for_date(target_date: str):
    print(f"[TRAIN] Entraînement pour {target_date} - MOCK (loading minimal data)")
    # Instead of full training, just creating a dummy model or small training to allow prediction
    # Actually we need a real model to predict probabilities.
    # Let's try to train on a very small set or reuse existing logic if fast enough.

    conn = get_connection()
    query = f"""
    SELECT
        place_finale,
        cote_reference,
        distance_m,
        age,
        poids_kg,
        hippodrome_code
    FROM cheval_courses_seen
    WHERE cote_reference IS NOT NULL
      AND cote_reference > 0
      AND place_finale IS NOT NULL
      AND annee >= 2023
      AND race_key < '{target_date}'
    LIMIT 1000
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Simple feature engineering
    df["target_place"] = df["place_finale"].apply(lambda x: 1 if x <= 3 else 0)
    df["cote_log"] = np.log1p(df["cote_reference"])

    hippo_stats = (
        df.groupby("hippodrome_code")
        .agg({"target_place": "mean", "cote_reference": "mean"})
        .reset_index()
    )
    hippo_stats.columns = ["hippodrome_code", "hippodrome_place_rate", "hippodrome_avg_cote"]

    df = df.merge(hippo_stats, on="hippodrome_code", how="left")
    df["hippodrome_place_rate"] = df["hippodrome_place_rate"].fillna(0.313)
    df["hippodrome_avg_cote"] = df["hippodrome_avg_cote"].fillna(df["cote_reference"].mean())

    features = ALGO_CONFIG["features_base"] + ["hippodrome_place_rate", "hippodrome_avg_cote"]
    X = df[features].values
    y = df["target_place"].values

    model = xgb.XGBClassifier(**ALGO_CONFIG["xgb_params"])
    model.fit(X, y, verbose=False)

    return model, hippo_stats


def debug_daily_advice_v2():
    date_str = datetime.now().strftime("%Y-%m-%d")
    print(f"Date: {date_str}")

    conn = get_connection()
    query = f"""
    SELECT
        nom_norm,
        race_key,
        cote_reference,
        distance_m,
        age,
        poids_kg,
        hippodrome_code,
        numero_dossard,
        place_finale,
        heure_depart
    FROM cheval_courses_seen
    WHERE race_key LIKE '{date_str}%'
      AND cote_reference IS NOT NULL
      AND cote_reference > 0
    ORDER BY race_key ASC, numero_dossard ASC
    """
    df_races = pd.read_sql(query, conn)
    conn.close()

    print(f"Total races rows: {len(df_races)}")
    if len(df_races) == 0:
        print("No races found in DB.")
        return

    # Train (or mock)
    model, hippo_stats = train_model_for_date(date_str)

    # Feature engineering
    df_races["cote_log"] = np.log1p(df_races["cote_reference"])
    df_races = df_races.merge(hippo_stats, on="hippodrome_code", how="left")
    df_races["hippodrome_place_rate"] = df_races["hippodrome_place_rate"].fillna(0.313)
    df_races["hippodrome_avg_cote"] = df_races["hippodrome_avg_cote"].fillna(
        df_races["cote_reference"].mean()
    )

    features = ALGO_CONFIG["features_base"] + ["hippodrome_place_rate", "hippodrome_avg_cote"]
    X = df_races[features].values
    pred_proba = model.predict_proba(X)[:, 1] * 100
    df_races["proba"] = pred_proba

    # Filters analysis
    print("\n--- Filters Analysis ---")

    # 1. Cote Filter
    mask_cote = (df_races["cote_reference"] >= ALGO_CONFIG["cote_min"]) & (
        df_races["cote_reference"] <= ALGO_CONFIG["cote_max"]
    )
    print(
        f"Candidates passing Cote [{ALGO_CONFIG['cote_min']}-{ALGO_CONFIG['cote_max']}]: {mask_cote.sum()}"
    )

    if mask_cote.sum() > 0:
        print(
            "Sample passing cote:",
            df_races[mask_cote][["nom_norm", "cote_reference"]].head(3).values,
        )

    # 2. Proba Filter
    mask_proba = df_races["proba"] >= ALGO_CONFIG["threshold"] * 100
    print(f"Candidates passing Proba >={ALGO_CONFIG['threshold']*100}%: {mask_proba.sum()}")

    if mask_proba.sum() > 0:
        print("Sample passing proba:", df_races[mask_proba][["nom_norm", "proba"]].head(3).values)

    # Combined Cote + Proba
    mask_algo = mask_cote & mask_proba
    print(f"Candidates passing Algo (Cote + Proba): {mask_algo.sum()}")

    # 3. Time Filter
    current_ts = int(time.time() * 1000)
    print(f"Current TS: {current_ts}")

    def parse_timestamp(val):
        try:
            if pd.notna(val) and str(val).isdigit():
                return int(val)
            return 0
        except:
            return 0

    df_races["ts_depart"] = df_races["heure_depart"].apply(parse_timestamp)

    mask_time = df_races["ts_depart"] > current_ts
    print(f"Candidates passing Time > Current: {mask_time.sum()}")
    if mask_time.sum() > 0:
        # Show gap
        df_races["diff_min"] = (df_races["ts_depart"] - current_ts) / 1000 / 60
        print(
            f"Upcoming races delay (min): {df_races[mask_time]['diff_min'].min():.1f} to {df_races[mask_time]['diff_min'].max():.1f}"
        )

    # Sample non-passing time
    not_passing_time = ~mask_time
    if not_passing_time.sum() > 0:
        print(
            "Sample past races:",
            df_races[not_passing_time][["nom_norm", "ts_depart"]].head(3).values,
        )

    # FINAL
    mask_final = mask_algo & mask_time
    print(f"FINAL Candidates: {mask_final.sum()}")


if __name__ == "__main__":
    debug_daily_advice_v2()
