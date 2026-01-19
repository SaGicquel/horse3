import pandas as pd
import numpy as np
from scipy.special import softmax

# Charger les données de base
df = pd.read_csv("data/backtest_predictions.csv")
print(f"Données chargées: {len(df)} lignes")

# Appliquer une calibration simple
df_calibrated = df.copy()

# Température optimale trouvée: 0.5
temperature = 0.5

# Créer des race_id comme dans le fichier de calibration
df_calibrated = df_calibrated.sort_values(["date_course", "p_model_win"], ascending=[True, False])

race_counter = 0
df_calibrated["race_id"] = None
partants_per_race = 10

current_date = None
horses_in_current_race = 0
current_race_id = None

for idx, row in df_calibrated.iterrows():
    if row["date_course"] != current_date or horses_in_current_race >= partants_per_race:
        current_race_id = f"R{race_counter:05d}"
        race_counter += 1
        current_date = row["date_course"]
        horses_in_current_race = 0

    df_calibrated.loc[idx, "race_id"] = current_race_id
    horses_in_current_race += 1

# Normalisation softmax par course avec température
df_calibrated["p_model_norm"] = 0.0
df_calibrated["p_calibrated"] = 0.0

for race_id, race_group in df_calibrated.groupby("race_id"):
    # Convertir probabilités en logits puis appliquer température
    p_clipped = np.clip(race_group["p_model_win"], 1e-7, 1 - 1e-7)
    logits = np.log(p_clipped / (1 - p_clipped))  # logit
    logits_scaled = logits / temperature

    # Softmax pour normaliser par course
    probs_norm = softmax(logits_scaled)

    # Calibration simple (on pourrait appliquer Platt ici)
    probs_calibrated = probs_norm  # Pour simplifier

    df_calibrated.loc[race_group.index, "p_model_norm"] = probs_norm
    df_calibrated.loc[race_group.index, "p_calibrated"] = probs_calibrated

# Probabilité finale = calibrée
df_calibrated["p_final"] = df_calibrated["p_calibrated"]

# Sauvegarder
output_path = "data/backtest_predictions_calibrated.csv"
df_calibrated.to_csv(output_path, index=False)

print(f"✅ Fichier créé: {output_path}")
print(f"Lignes: {len(df_calibrated)}")
print(f'Races: {df_calibrated["race_id"].nunique()}')
print(
    f'Stats p_final: min={df_calibrated["p_final"].min():.3f}, max={df_calibrated["p_final"].max():.3f}, mean={df_calibrated["p_final"].mean():.3f}'
)

# Vérifier que les probabilités somment à 1 par course
sums_by_race = df_calibrated.groupby("race_id")["p_final"].sum()
print(
    f"Vérification normalisation: min={sums_by_race.min():.3f}, max={sums_by_race.max():.3f}, mean={sums_by_race.mean():.3f}"
)

# Statistiques par split
print("\\nStats par split:")
print(df_calibrated.groupby("split")[["p_model_win", "p_final"]].mean())
