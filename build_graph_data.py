# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 1: Data Preparation (Part 2)
build_graph_data.py

Ce script construit un graphe hétérogène des entités (chevaux, jockeys, entraîneurs)
en utilisant PyTorch Geometric (PyG). Ce graphe capture les relations statiques
entre les entités, qui seront utilisées par le modèle GNN.

⚠️ Important: Ce script utilise PyTorch Geometric (PyG), car DGL n'est pas
supporté sur l'architecture Apple Silicon (M1/M2).

Étapes :
1.  Charger les données enrichies (`ml_features_complete.csv`).
2.  Identifier les entités uniques (chevaux, jockeys, entraîneurs) et créer des mappings d'ID.
3.  Extraire et agréger les features pour chaque type de noeud (ex: taux de victoire moyen).
4.  Créer la structure de graphe hétérogène avec `torch_geometric.data.HeteroData`.
5.  Ajouter les tenseurs de features pour chaque type de noeud.
6.  Définir les relations (edges) entre les noeuds (ex: cheval 'est_monté_par' jockey).
7.  Sauvegarder l'objet graphe PyG dans un fichier pickle.

Output: data/phase9/graphs/entity_graph_v1.pkl
    - Un objet `torch_geometric.data.HeteroData` contenant:
        - `graph['cheval'].x`: Features des chevaux
        - `graph['jockey'].x`: Features des jockeys
        - `graph['entraineur'].x`: Features des entraîneurs
        - `graph['cheval', 'est_monte_par', 'jockey'].edge_index`: Edges cheval-jockey
        - `graph['cheval', 'est_entraine_par', 'entraineur'].edge_index`: Edges cheval-entraîneur
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from tqdm import tqdm

# Configuration
INPUT_FILE = "data/ml_features_complete.csv"
OUTPUT_DIR = "data/phase9/graphs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "entity_graph_v1.pkl")

# Features pour chaque type de noeud
CHEVAL_FEATURES = [
    "an_naissance",
    "taux_places_carriere",
    "gains_carriere",
    "jours_depuis_derniere",
    "aptitude_piste",
    "aptitude_distance",
]
JOCKEY_FEATURES = ["taux_victoires_jockey", "taux_places_jockey"]
ENTRAINEUR_FEATURES = ["taux_victoires_entraineur", "taux_places_entraineur"]


class GraphBuilder:
    def __init__(self, input_path):
        self.input_path = input_path
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        print("Initialisation du GraphBuilder.")

    def load_and_aggregate(self):
        """Charge les données et agrège les features pour chaque entité."""
        print(f"Chargement des données depuis {self.input_path}...")
        df = pd.read_csv(
            self.input_path,
            usecols=[
                "id_cheval",
                "id_jockey",
                "id_entraineur",
                "date_course",
                *CHEVAL_FEATURES,
                *JOCKEY_FEATURES,
                *ENTRAINEUR_FEATURES,
            ],
        )
        df["date_course"] = pd.to_datetime(df["date_course"])

        # Imputation simple
        df.fillna(0, inplace=True)

        print("Agrégation des features par entité...")
        # Pour chaque entité, on prend les dernières valeurs connues
        df_chevaux = df.sort_values("date_course").groupby("id_cheval").last()[CHEVAL_FEATURES]
        df_jockeys = df.sort_values("date_course").groupby("id_jockey").last()[JOCKEY_FEATURES]
        df_entraineurs = (
            df.sort_values("date_course").groupby("id_entraineur").last()[ENTRAINEUR_FEATURES]
        )

        print(
            f"Entités trouvées: {len(df_chevaux)} chevaux, {len(df_jockeys)} jockeys, {len(df_entraineurs)} entraîneurs."
        )
        return df, df_chevaux, df_jockeys, df_entraineurs

    def build_hetero_graph(self, df, df_chevaux, df_jockeys, df_entraineurs):
        """Construit le graphe hétérogène avec PyTorch Geometric."""
        print("Construction du graphe HeteroData...")
        graph = HeteroData()

        # Mappings d'ID
        cheval_map = {id: i for i, id in enumerate(df_chevaux.index)}
        jockey_map = {id: i for i, id in enumerate(df_jockeys.index)}
        entraineur_map = {id: i for i, id in enumerate(df_entraineurs.index)}

        # Normalisation et création des tenseurs de features
        graph["cheval"].x = torch.tensor(
            self.scaler.fit_transform(df_chevaux.values), dtype=torch.float
        )
        graph["jockey"].x = torch.tensor(
            self.scaler.fit_transform(df_jockeys.values), dtype=torch.float
        )
        graph["entraineur"].x = torch.tensor(
            self.scaler.fit_transform(df_entraineurs.values), dtype=torch.float
        )

        print("Création des arêtes (edges)...")
        # On utilise les relations de la dernière course de chaque cheval
        last_races = df.sort_values("date_course").groupby("id_cheval").last()

        source_cheval_jockey = [cheval_map[idx] for idx in last_races.index]
        target_cheval_jockey = [jockey_map[j_id] for j_id in last_races["id_jockey"]]

        source_cheval_entraineur = [cheval_map[idx] for idx in last_races.index]
        target_cheval_entraineur = [entraineur_map[e_id] for e_id in last_races["id_entraineur"]]

        # Ajout des arêtes au graphe
        graph["cheval", "est_monte_par", "jockey"].edge_index = torch.tensor(
            [source_cheval_jockey, target_cheval_jockey], dtype=torch.long
        )
        graph["cheval", "est_entraine_par", "entraineur"].edge_index = torch.tensor(
            [source_cheval_entraineur, target_cheval_entraineur], dtype=torch.long
        )

        # On peut aussi ajouter les relations inverses pour un message passing bi-directionnel
        graph["jockey", "rev_est_monte_par", "cheval"].edge_index = torch.tensor(
            [target_cheval_jockey, source_cheval_jockey], dtype=torch.long
        )
        graph["entraineur", "rev_est_entraine_par", "cheval"].edge_index = torch.tensor(
            [target_cheval_entraineur, source_cheval_entraineur], dtype=torch.long
        )

        print("Validation du graphe...")
        try:
            graph.validate()
            print("Graphe valide !")
        except Exception as e:
            print(f"Erreur de validation du graphe: {e}")

        return graph, {
            "cheval_map": cheval_map,
            "jockey_map": jockey_map,
            "entraineur_map": entraineur_map,
        }

    def run(self):
        """Exécute le pipeline complet."""
        df, df_chevaux, df_jockeys, df_entraineurs = self.load_and_aggregate()
        graph, mappings = self.build_hetero_graph(df, df_chevaux, df_jockeys, df_entraineurs)

        # Sauvegarde
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Sauvegarde du graphe et des mappings dans {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump({"graph": graph, "mappings": mappings}, f)

        print("---")
        print("✅ Construction du graphe terminée !")
        print("Résumé du graphe :")
        print(graph)
        print(f"\n   - Graphe sauvegardé dans {OUTPUT_FILE}")
        print("---")


if __name__ == "__main__":
    builder = GraphBuilder(input_path=INPUT_FILE)
    builder.run()
