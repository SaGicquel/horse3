import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv('data/backtest_predictions.csv')
print(f"Données chargées: {len(df)} lignes")

# Créer des race_key uniques basés sur la date et regrouper logiquement
# Simuler des courses de 8-12 partants
race_counter = 0
df['race_id'] = None
partants_per_race = 10  # Moyenne de partants par course

# Trier par date puis par probabilité pour des groupements cohérents
df = df.sort_values(['date_course', 'p_model_win'], ascending=[True, False])

current_date = None
horses_in_current_race = 0
current_race_id = None

for idx, row in df.iterrows():
    if row['date_course'] != current_date or horses_in_current_race >= partants_per_race:
        # Nouvelle course
        current_race_id = f"R{race_counter:05d}"
        race_counter += 1
        current_date = row['date_course']
        horses_in_current_race = 0
    
    df.loc[idx, 'race_id'] = current_race_id
    horses_in_current_race += 1

print(f"Races créées: {df['race_id'].nunique()}")

# Vérifier la taille des courses
race_sizes = df.groupby('race_id').size()
print(f"Taille courses - min: {race_sizes.min()}, max: {race_sizes.max()}, moyenne: {race_sizes.mean():.1f}")

# Ajouter des probabilités de marché réalistes (inverse des cotes + normalisation)
df['p_market'] = 0.0

for race_id, race_group in df.groupby('race_id'):
    # Convertir cotes normalisées en pseudo-probabilités
    # Plus la cote est faible (négative), plus la probabilité est haute
    cotes_inversees = 1.0 / (1.0 + np.exp(race_group['cote_sp']))  # sigmoide inverse
    cotes_inversees_norm = cotes_inversees / cotes_inversees.sum()  # normaliser à 1
    df.loc[race_group.index, 'p_market'] = cotes_inversees_norm

# Créer le fichier adapté pour la calibration
required_cols = ['race_id', 'p_model_win', 'is_win', 'date_course', 'split', 'cote_sp', 'cote_pm', 'p_market']
calibration_df = df[required_cols].copy()

# Sauvegarder
output_path = 'data/backtest_predictions_calibration_ready.csv'
calibration_df.to_csv(output_path, index=False)

print(f"✅ Fichier créé: {output_path}")
print(f"Colonnes: {calibration_df.columns.tolist()}")
print(f"Splits: {calibration_df['split'].value_counts()}")
print(f"Races: {calibration_df['race_id'].nunique()}")