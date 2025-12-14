#!/usr/bin/env python3
"""Train Champion model inside Docker with compatible sklearn version."""

import json
import os
import pickle

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://pmu_user:pmu_secure_password_2025@host.docker.internal:54624/pmu_database",
)
OUTPUT_DIR = "/tmp/champion_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    "interaction_forme_jockey",
    "interaction_aptitude_distance",
    "interaction_synergie_forme",
    "interaction_aptitude_popularite",
    "interaction_regularite_volume",
    "discipline_Plat",
    "discipline_Trot",
    "discipline_Obstacle",
    "sexe_H",
    "sexe_M",
    "sexe_F",
    "etat_Bon",
    "etat_Souple",
    "etat_Lourd",
    "etat_PSF",
    "etat_Leger",
]

print("Connecting to DB...")
conn = psycopg2.connect(DB_URL)

print("Loading data...")
query = """
SELECT race_key, id_cheval, draw_stalle, cote_finale, cote_matin, distance_m,
       nombre_partants, allocation_totale, place_finale, is_win, driver_jockey,
       entraineur, sexe, age, temperature_c, vent_kmh, etat_piste, discipline,
       SUBSTRING(race_key FROM 1 FOR 10) AS race_date
FROM cheval_courses_seen
WHERE SUBSTRING(race_key FROM 1 FOR 10)::date BETWEEN '2023-01-01' AND '2024-08-31'
  AND place_finale IS NOT NULL AND cote_finale IS NOT NULL AND cote_finale > 0
ORDER BY race_key, id_cheval
LIMIT 50000
"""
df_raw = pd.read_sql(query, conn, params=())
print(f"Loaded {len(df_raw)} rows")

# Stats chevaux
print("Loading stats...")
cur = conn.cursor()
cur.execute(
    "SELECT id_cheval, forme_5c, aptitude_distance, aptitude_piste, aptitude_hippodrome, regularite FROM stats_chevaux"
)
stats = {r[0]: r[1:] for r in cur.fetchall()}
print(f"Stats: {len(stats)} chevaux")
conn.close()

print("Building features...")
all_rows = []
race_keys = df_raw["race_key"].unique()

for i, rk in enumerate(race_keys[:2000]):
    rdf = df_raw[df_raw["race_key"] == rk]
    nb = len(rdf)
    dist = float(rdf["distance_m"].iloc[0] or 2000)
    dist_norm = dist / 1000

    cotes = rdf["cote_finale"].fillna(999).tolist()
    sorted_idx = sorted(range(len(cotes)), key=lambda x: cotes[x])
    rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_idx)}

    formes = [float(stats.get(cid, (0, 0, 0, 0, 0))[0] or 0) for cid in rdf["id_cheval"]]
    mean_forme = sum(formes) / len(formes) if formes else 0

    for idx, (_, row) in enumerate(rdf.iterrows()):
        cid = row["id_cheval"]
        f5, ad, ap, ah, reg = stats.get(cid, (0, 0, 0, 0, 0))
        disc = (row["discipline"] or "").lower()
        sexe = (row["sexe"] or "").upper()
        etat = (row["etat_piste"] or "").lower()

        features = {
            "is_win": int(row["is_win"] or 0),
            "numero_corde": float(row["draw_stalle"] or 0),
            "cote_sp": float(row["cote_finale"] or 0),
            "cote_pm": float(row["cote_matin"] or row["cote_finale"] or 0),
            "distance": dist,
            "nombre_partants": nb,
            "allocation": float(row["allocation_totale"] or 0),
            "forme_5c": float(f5 or 0),
            "forme_10c": float(f5 or 0),
            "nb_courses_12m": 5.0,
            "nb_victoires_12m": 0.5,
            "nb_places_12m": 1.5,
            "regularite": float(reg or 0),
            "jours_depuis_derniere": 30.0,
            "aptitude_distance": float(ad or 0),
            "aptitude_piste": float(ap or 0),
            "aptitude_hippodrome": float(ah or 0),
            "synergie_jockey_cheval": float((f5 or 0) * 0.1),
            "synergie_entraineur_cheval": float((f5 or 0) * 0.1),
            "jockey_win_rate": 0.1,
            "jockey_place_rate": 0.3,
            "entraineur_win_rate": 0.1,
            "entraineur_place_rate": 0.3,
            "distance_norm": dist_norm,
            "niveau_moyen_concurrent": mean_forme,
            "rang_cote_sp": float(rank_map.get(idx, 1)),
            "an_naissance": 2019.0,
            "age": float(row["age"] or 5),
            "temperature_c": float(row["temperature_c"] or 15)
            if pd.notna(row["temperature_c"])
            else 15,
            "vent_kmh": float(row["vent_kmh"] or 10) if pd.notna(row["vent_kmh"]) else 10,
            "meteo_code": 0.0,
            "interaction_forme_jockey": float((f5 or 0) * 0.1),
            "interaction_aptitude_distance": float((ad or 0) * dist_norm),
            "interaction_synergie_forme": 0.0,
            "interaction_aptitude_popularite": float((ad or 0) / max(1, rank_map.get(idx, 1))),
            "interaction_regularite_volume": float((reg or 0) * 5),
            "discipline_Plat": 1.0 if "plat" in disc else 0.0,
            "discipline_Trot": 1.0 if "trot" in disc or "att" in disc else 0.0,
            "discipline_Obstacle": 1.0 if "haie" in disc or "steeple" in disc else 0.0,
            "sexe_H": 1.0 if sexe.startswith("H") else 0.0,
            "sexe_M": 1.0 if sexe.startswith("M") else 0.0,
            "sexe_F": 1.0 if sexe.startswith("F") else 0.0,
            "etat_Bon": 1.0 if "bon" in etat else 0.0,
            "etat_Souple": 1.0 if "souple" in etat else 0.0,
            "etat_Lourd": 1.0 if "lourd" in etat else 0.0,
            "etat_PSF": 1.0 if "psf" in etat else 0.0,
            "etat_Leger": 1.0 if "léger" in etat or "leger" in etat else 0.0,
        }
        all_rows.append(features)

print(f"Built {len(all_rows)} samples")
df = pd.DataFrame(all_rows)
X = df[FEATURE_NAMES].values.astype(np.float32)
y = df["is_win"].values
print(f"X shape: {X.shape}, y mean: {y.mean():.3f}")

# Imputer + Scaler
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0).astype(np.float32)

# Train
print("Training XGBoost...")
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
scale_pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    early_stopping_rounds=20,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

from sklearn.metrics import roc_auc_score

p_val = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, p_val)
print(f"ROC-AUC: {auc:.4f}")

# Save
with open(f"{OUTPUT_DIR}/xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(f"{OUTPUT_DIR}/feature_imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)
with open(f"{OUTPUT_DIR}/feature_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{OUTPUT_DIR}/feature_names.json", "w") as f:
    json.dump(FEATURE_NAMES, f, indent=2)
with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump({"roc_auc": round(auc, 4), "features": len(FEATURE_NAMES)}, f, indent=2)

print(f"Saved to {OUTPUT_DIR}")
print("DONE")
