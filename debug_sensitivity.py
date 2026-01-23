import pandas as pd
import numpy as np
from datetime import datetime
from db_connection import get_connection


# Mock logic similar to api
def check_volume_sensitivity():
    conn = get_connection()
    date_str = datetime.now().strftime("%Y-%m-%d")

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
        heure_depart
    FROM cheval_courses_seen
    WHERE race_key LIKE '{date_str}%'
      AND cote_reference IS NOT NULL
      AND cote_reference > 0
    """
    df_races = pd.read_sql(query, conn)
    conn.close()

    # Simulate probas (reuse logic or load from log if possible, but here we simulate based on trends seen in log)
    # The log said: Max proba 49.61%, Mean 34.19%.
    # We can't perfectly reproduce the model output without the model object in memory or reloading it.
    # But wait, I can just import the train_model_for_date function from user_app_api_v2!

    sys.path.append(".")
    from user_app_api_v2 import train_model_for_date, ALGO_CONFIG

    # Need to monkeypatch or ensure ALGO_CONFIG is what we want to test?
    # No, I will just use the returned model to predict and then apply different filters in pandas.

    model, hippo_stats = train_model_for_date(date_str)

    # Feature eng
    for col in ALGO_CONFIG["features_base"]:
        if col in df_races.columns:
            df_races[col] = pd.to_numeric(df_races[col], errors="coerce")

    df_races["cote_log"] = np.log1p(df_races["cote_reference"])
    df_races = df_races.merge(hippo_stats, on="hippodrome_code", how="left")
    df_races["hippodrome_place_rate"] = df_races["hippodrome_place_rate"].fillna(0.313)
    df_races["hippodrome_avg_cote"] = df_races["hippodrome_avg_cote"].fillna(
        df_races["cote_reference"].mean()
    )

    features = ALGO_CONFIG["features_base"] + ["hippodrome_place_rate", "hippodrome_avg_cote"]
    X = df_races[features].values
    df_races["proba"] = model.predict_proba(X)[:, 1] * 100

    print(f"Total horses: {len(df_races)}")

    # Test thresholds
    thresholds = [40, 42, 45, 48, 50]
    cote_ranges = [(7, 15), (7, 20)]

    print("\n--- Sensitivity Analysis ---")
    for min_c, max_c in cote_ranges:
        print(f"\nCote Range: {min_c}-{max_c}")
        for th in thresholds:
            mask = (
                (df_races["cote_reference"] >= min_c)
                & (df_races["cote_reference"] <= max_c)
                & (df_races["proba"] >= th)
            )
            count = mask.sum()
            print(f"  Threshold {th}%: {count} bets")
            if count > 0 and count < 10:
                print(
                    f"    -> Top bets: {df_races[mask]['proba'].sort_values(ascending=False).tolist()}"
                )


import sys

if __name__ == "__main__":
    check_volume_sensitivity()
