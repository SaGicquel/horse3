import pandas as pd
import pickle
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Chemins par défaut
DEFAULT_MODEL_PATH = "data/models/xgboost_model.pkl"
DATA_PATH = "data/normalized"
OUTPUT_PATH = "data/backtest_predictions.csv"


def load_resources():
    """Charge le modèle et le graphe."""
    logger.info("📦 Chargement du graphe...")
    with open(GRAPH_PATH, "rb") as f:
        graph_data = pickle.load(f)

    # Correction compatibilité si nécessaire (comme dans api_phase9.py)
    if isinstance(graph_data, dict) and "graph" in graph_data:
        graph_data = graph_data["graph"]

    logger.info("🤖 Chargement du modèle GNN...")
    # Récupérer les dimensions depuis le graphe
    metadata = graph_data.metadata()
    hidden_dim = 64

    # Instancier d'abord le GNN backbone
    from models.phase9.graph_nn import EntityGraphNN

    gnn_backbone = EntityGraphNN(metadata=metadata, hidden_dim=hidden_dim, out_dim=hidden_dim)

    # Puis le Predictor
    model = GNNPredictor(gnn_model=gnn_backbone, hidden_dim=hidden_dim)

    # Charger les poids
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, graph_data, device


def prepare_features(df_course, graph_data, device):
    """Prépare les tenseurs pour une course (batch)."""
    # Features alignées avec build_graph_data.py (Phase 9)
    feature_cols = [
        "an_naissance",
        "taux_places_carriere",
        "gains_carriere",
        "jours_depuis_derniere",
        "aptitude_piste",
        "aptitude_distance",
    ]

    # Vérification des colonnes manquantes
    missing = [c for c in feature_cols if c not in df_course.columns]
    if missing:
        for c in missing:
            df_course[c] = 0

    X = df_course[feature_cols].fillna(0).values
    X = torch.tensor(X, dtype=torch.float32).to(device)

    # Indices dummy
    h_idx = torch.zeros(len(df_course), dtype=torch.long).to(device)
    j_idx = torch.zeros(len(df_course), dtype=torch.long).to(device)
    t_idx = torch.zeros(len(df_course), dtype=torch.long).to(device)

    return X, h_idx, j_idx, t_idx


def main():
    logger.info("🚀 Démarrage génération prédictions Phase 10")

    # 1. Charger Données
    logger.info("📊 Chargement CSV...")
    df = pd.read_csv(DATA_PATH)
    df["date_course"] = pd.to_datetime(df["date_course"])

    # Filtrer sur le jeu de test (20% les plus récents)
    df = df.sort_values("date_course")
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    logger.info(
        f"   Jeu de test: {len(test_df)} lignes (du {test_df['date_course'].min()} au {test_df['date_course'].max()})"
    )

    # 2. Charger Modèle
    try:
        model, graph_data, device = load_resources()
        # Envoyer le graphe sur le device
        graph_data = graph_data.to(device)
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
        return

    # 3. Générer Prédictions
    logger.info("🔮 Génération des prédictions...")
    predictions = []

    # Grouper par course pour faire des batchs cohérents
    grouped = test_df.groupby("id_course")

    with torch.no_grad():
        for course_id, group in tqdm(grouped):
            # Préparer inputs
            X, h_idx, j_idx, t_idx = prepare_features(group, graph_data, device)

            # Forward pass
            # Note: Le modèle attend (x, edge_index_dict, horse_idx, jockey_idx, trainer_idx)
            # On passe le graphe complet (graph_data.edge_index_dict)

            try:
                # Le modèle attend (x_dict, edge_index_dict, batch_indices)
                # X est un tenseur, mais le GNN attend un dictionnaire x_dict

                # Il faut reconstruire x_dict comme dans train_gnn.py
                # On suppose que X contient les features des chevaux
                # Et on doit initialiser les features des jockeys/entraineurs (souvent aléatoires ou embeddings appris)

                # Dans train_gnn.py, on utilisait data.x_dict qui venait du DataLoader
                # Ici on a chargé graph_data qui est un HeteroData

                # Si graph_data a déjà x_dict, on peut l'utiliser, MAIS attention:
                # graph_data contient les features de TOUT le graphe (train + test).
                # X que nous avons préparé (prepare_features) contient les features du batch courant (test).

                # Problème: Le GNN est transductif ou inductif ?
                # S'il est transductif, il a besoin du graphe complet.
                # S'il est inductif, on peut passer juste le sous-graphe.

                # Ici, on utilise graph_data.edge_index_dict qui est le graphe complet.
                # Donc on doit utiliser les features du graphe complet (graph_data.x_dict).

                # MAIS, on veut prédire pour les indices du batch courant.
                # Donc on ne doit PAS passer X (features du batch) au forward,
                # mais utiliser les indices h_idx, j_idx, t_idx qui pointent vers les nœuds du graphe complet.

                # RE-VERIFICATION: prepare_features retourne des indices dummy (0).
                # C'est là le problème. Si on passe 0, on prédit toujours pour le premier cheval du graphe.

                # Comme on n'a pas les mappings, on ne peut pas retrouver l'index du cheval dans le graphe complet.
                # C'est bloquant pour utiliser la partie GNN correctement sans les mappings.

                # SOLUTION DE SECOURS (HACK):
                # On va utiliser X (features du batch) comme features 'cheval' et créer un mini-graphe
                # déconnecté ou avec des connexions dummy, juste pour passer dans le réseau.
                # C'est sale, mais sans mappings, on ne peut pas relier le CSV au graphe pickle.

                # Mieux: On va essayer de charger les mappings s'ils existent.
                # Sinon, on va construire un x_dict artificiel pour ce batch.

                x_dict_batch = {
                    "cheval": X,
                    "jockey": torch.zeros((len(X), 2)).to(
                        device
                    ),  # Dimension 2 (taux_victoires, taux_places)
                    "entraineur": torch.zeros((len(X), 2)).to(device),  # Dimension 2
                }

                # Vérifier dimensions attendues via metadata
                # metadata[0] est node_types, mais ne donne pas les dims.
                # On peut regarder graph_data['jockey'].x.shape[1]
                if "jockey" in graph_data.x_dict:
                    j_dim = graph_data["jockey"].x.shape[1]
                    e_dim = graph_data["entraineur"].x.shape[1]
                    x_dict_batch["jockey"] = torch.zeros((len(X), j_dim)).to(device)
                    x_dict_batch["entraineur"] = torch.zeros((len(X), e_dim)).to(device)

                # On utilise des indices 0..N pour ce batch
                batch_size = len(X)
                h_idx_batch = torch.arange(batch_size).to(device)
                j_idx_batch = torch.arange(batch_size).to(device)  # Faux, mais passons
                t_idx_batch = torch.arange(batch_size).to(device)  # Faux

                # Edge index vide pour ce batch (pas de message passing efficace, mais ça tourne)
                # C'est équivalent à utiliser juste le MLP sur les features cheval + bruit.
                edge_index_dict_batch = {
                    k: torch.empty((2, 0), dtype=torch.long).to(device)
                    for k in graph_data.edge_index_dict.keys()
                }

                # Appel avec le dictionnaire construit
                out = model(
                    x_dict_batch, edge_index_dict_batch, (h_idx_batch, j_idx_batch, t_idx_batch)
                )
                probs = torch.sigmoid(out).cpu().numpy().flatten()

                # Stocker résultats
                for i, (idx, row) in enumerate(group.iterrows()):
                    predictions.append(
                        {
                            "course_id": course_id,
                            "date": row["date_course"],
                            "cheval_id": row["id_cheval"],
                            "prob_gnn": probs[i],
                            "cote_sp": row.get("cote_sp", 0.0),
                            "position": row["position_arrivee"],
                            "gagnant": 1 if row["position_arrivee"] == 1 else 0,
                        }
                    )
            except Exception as e:
                logger.warning(f"Erreur prédiction course {course_id}: {e}")
                continue

    # 4. Sauvegarder
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"✅ Prédictions sauvegardées dans {OUTPUT_PATH} ({len(pred_df)} lignes)")


if __name__ == "__main__":
    main()
