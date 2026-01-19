# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 1: Data Preparation (Part 1)
prepare_temporal_data.py

Ce script prépare les données séquentielles pour le modèle Transformer.
Il prend les données de courses enrichies, et pour chaque cheval, il extrait
l'historique de ses 10 dernières courses.

Étapes :
1.  Charger les données enrichies (`ml_features_complete.csv`).
2.  Trier les courses par cheval et par date.
3.  Pour chaque cheval, créer une séquence de ses 10 dernières courses.
4.  Appliquer un padding (remplissage avec des zéros) pour les chevaux ayant moins de 10 courses.
5.  Créer un masque de padding pour indiquer les vraies données vs le remplissage.
6.  Normaliser les features numériques sur une échelle [-1, 1].
7.  Sauvegarder les séquences, les masques, les labels et les métadonnées dans un fichier pickle.

Output: data/phase9/temporal/sequences_v1.pkl
    - 'sequences': Tensor (N_chevaux, 10, N_features)
    - 'masks': Tensor booléen (N_chevaux, 10)
    - 'labels': Tensor (N_chevaux,) - Résultat de la dernière course
    - 'metadata': Mappings (cheval_id -> index, etc.)
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from tqdm import tqdm

# Configuration
INPUT_FILE = "data/ml_features_complete.csv"
OUTPUT_DIR = "data/phase9/temporal"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sequences_v1.pkl")
SEQ_LENGTH = 10
TARGET_COL = "victoire"  # ou 'place'

# Liste des colonnes à utiliser comme features
# Exclut les identifiants, dates, et la cible pour éviter les fuites
FEATURES_COLS = [
    "an_naissance",
    "distance",
    "nombre_partants",
    "temperature_c",
    "vent_kmh",
    "nb_courses_12m",
    "nb_victoires_12m",
    "taux_places_12m",
    "regularite",
    "jours_depuis_derniere",
    "aptitude_distance",
    "aptitude_piste",
    "aptitude_hippodrome",
    "taux_victoires_jockey",
    "taux_places_jockey",
    "taux_victoires_entraineur",
    "taux_places_entraineur",
    "synergie_jockey_cheval",
    "synergie_entraineur_cheval",
    "distance_norm",
    "niveau_moyen_concurrent",
    "rang_cote_sp",
    "rang_cote_turfbzh",
    "ecart_cote_ia",
    "gains_carriere",
    "gains_12m",
    "gains_par_course",
    "nb_premieres_places",
    "nb_deuxiemes_places",
    "nb_troisiemes_places",
    "taux_places_carriere",
    "gain_moyen_victoire",
    "evolution_gains_12m",
    "ratio_gains_courses",
    "aptitude_piste_etat",
    "ecart_temp_optimal",
    "interaction_piste_meteo",
    "handicap_meteo",
    "interaction_forme_jockey",
    "interaction_aptitude_distance",
    "interaction_elo_niveau",
    "interaction_cote_ia",
    "interaction_synergie_forme",
    "interaction_victoires_jockey",
    "popularite_hippodrome",
    "interaction_aptitude_popularite",
    "interaction_regularite_volume",
]


class TemporalDataPreparator:
    def __init__(self, input_path, seq_length=10):
        self.input_path = input_path
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        print(f"Initialisation avec une longueur de séquence de {self.seq_length}.")

    def load_and_preprocess(self):
        """Charge les données, normalise les features et effectue un pré-traitement."""
        print(f"Chargement des données depuis {self.input_path}...")
        df = pd.read_csv(self.input_path)

        # Conversion de la date et tri
        df["date_course"] = pd.to_datetime(df["date_course"])
        df = df.sort_values(by=["id_cheval", "date_course"])

        # Imputation simple des valeurs manquantes
        df[FEATURES_COLS] = df[FEATURES_COLS].fillna(0)

        # Normalisation des features AVANT la création des séquences
        print("Normalisation des features...")
        df[FEATURES_COLS] = self.scaler.fit_transform(df[FEATURES_COLS])

        print(f"Données chargées et normalisées: {df.shape[0]} courses.")
        return df

    def create_sequences(self, df):
        """Crée les séquences temporelles pour chaque cheval à partir de données pré-normalisées."""
        print("Création des séquences temporelles...")

        sequences = []
        labels = []
        masks = []
        horse_ids = []
        jockey_ids = []
        entraineur_ids = []

        # Grouper par cheval
        grouped = df.groupby("id_cheval")

        for horse_id, group in tqdm(grouped, desc="Processing horses"):
            # On a besoin d'au moins 2 courses pour avoir un historique et un label
            if len(group) < 2:
                continue

            # La dernière course est la cible, les précédentes sont l'historique
            target_race = group.iloc[-1]
            history = group.iloc[:-1]

            # Prendre les N dernières courses de l'historique
            history = history.tail(self.seq_length)

            # Créer la séquence de features
            seq_features = history[FEATURES_COLS].values

            # Padding
            padding_len = self.seq_length - len(seq_features)

            # Créer le masque AVANT le padding
            mask = np.ones(len(seq_features), dtype=bool)
            if padding_len > 0:
                mask = np.pad(mask, (padding_len, 0), "constant", constant_values=False)

            # Appliquer le padding avec une valeur constante de -1.0
            padded_features = np.pad(
                seq_features, ((padding_len, 0), (0, 0)), "constant", constant_values=-1.0
            )

            sequences.append(padded_features)
            labels.append(target_race[TARGET_COL])
            masks.append(mask)
            horse_ids.append(horse_id)
            # Ajout des IDs pour la fusion
            jockey_ids.append(target_race["id_jockey"])
            entraineur_ids.append(target_race["id_entraineur"])

        print(f"{len(sequences)} séquences créées.")
        return (
            np.array(sequences, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(masks, dtype=bool),
            horse_ids,
            jockey_ids,
            entraineur_ids,
        )

    def run(self):
        """Exécute le pipeline complet."""
        df = self.load_and_preprocess()
        sequences, labels, masks, horse_ids, jockey_ids, entraineur_ids = self.create_sequences(df)

        if len(sequences) == 0:
            print("Aucune séquence n'a pu être créée. Arrêt.")
            return

        # La normalisation est déjà faite, on convertit directement en tenseurs
        sequences_tensor = torch.from_numpy(sequences).float()
        labels_tensor = torch.from_numpy(labels).float()
        masks_tensor = torch.from_numpy(masks).bool()

        # Création des métadonnées
        metadata = {
            "horse_id_to_idx": {horse_id: i for i, horse_id in enumerate(horse_ids)},
            "idx_to_horse_id": {i: horse_id for i, horse_id in enumerate(horse_ids)},
            "idx_to_jockey_id": {i: j_id for i, j_id in enumerate(jockey_ids)},
            "idx_to_entraineur_id": {i: e_id for i, e_id in enumerate(entraineur_ids)},
            "feature_cols": FEATURES_COLS,
            "seq_length": self.seq_length,
            "scaler": self.scaler,
        }

        # Sauvegarde
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Sauvegarde des données dans {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(
                {
                    "sequences": sequences_tensor,
                    "labels": labels_tensor,
                    "masks": masks_tensor,
                    "metadata": metadata,
                },
                f,
            )

        print("---")
        print("✅ Préparation des données temporelles terminée !")
        print(f"   - Séquences: {sequences_tensor.shape}")
        print(f"   - Labels: {labels_tensor.shape}")
        print(f"   - Masques: {masks_tensor.shape}")
        print(f"   - Données sauvegardées dans {OUTPUT_FILE}")
        print("---")


if __name__ == "__main__":
    preparator = TemporalDataPreparator(input_path=INPUT_FILE, seq_length=SEQ_LENGTH)
    preparator.run()
