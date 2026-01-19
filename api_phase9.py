# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 5: API de Prédiction Deep Learning (GNN)
api_phase9.py

Cette API sert le modèle GNN (Champion Phase 9) pour les prédictions en temps réel.
Elle gère :
1. Le chargement du graphe et des mappings (IDs -> Indices).
2. Le chargement du modèle GNN entraîné.
3. La prédiction pour un triplet (Cheval, Jockey, Entraîneur).
"""

import torch
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from models.phase9.graph_nn import EntityGraphNN, GNNPredictor
from torch_geometric.data import HeteroData

# --- Configuration ---
GRAPH_DATA_PATH = "data/phase9/graphs/entity_graph_v1.pkl"
MODEL_PATH = "models/phase9/saved_models/gnn_v1.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
HIDDEN_DIM = 64  # Doit correspondre à l'entraînement


# --- Modèles Pydantic ---
class PredictionRequest(BaseModel):
    horse_id: int
    jockey_id: int
    entraineur_id: int
    # Autres features optionnelles si on étendait le modèle


class PredictionResponse(BaseModel):
    score: float
    model_version: str = "gnn_v1"
    status: str


# --- Application FastAPI ---
app = FastAPI(
    title="Horse Racing Prediction API - Phase 9 (Deep Learning)",
    description="API servant le modèle GNN pour la prédiction de courses.",
    version="1.0.0",
)


# --- État Global ---
class ModelServer:
    def __init__(self):
        self.model = None
        self.graph_data = None
        self.mappings = None
        self.metadata = None

    def load_resources(self):
        print(f"Chargement des ressources depuis {GRAPH_DATA_PATH}...")
        if not os.path.exists(GRAPH_DATA_PATH):
            raise FileNotFoundError(f"Fichier de graphe introuvable : {GRAPH_DATA_PATH}")

        with open(GRAPH_DATA_PATH, "rb") as f:
            data = pickle.load(f)
            self.graph_data = data["graph"].to(DEVICE)
            self.mappings = data["mappings"]
            # self.metadata n'est pas dans le pickle, on l'obtient du graphe

        print(f"Chargement du modèle depuis {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Fichier modèle introuvable : {MODEL_PATH}")

        # Initialisation du modèle
        # On doit récupérer les dimensions depuis le graphe chargé

        gnn_core = EntityGraphNN(
            metadata=self.graph_data.metadata(),
            hidden_dim=HIDDEN_DIM,
            out_dim=HIDDEN_DIM,  # Le GNN sort des embeddings
            n_layers=3,
        )

        self.model = GNNPredictor(gnn_core, HIDDEN_DIM).to(DEVICE)

        # Chargement des poids
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

        # Hack pour LazyLinear : faire une passe dummy pour initialiser les poids
        with torch.no_grad():
            # Dummy batch
            dummy_batch = (
                torch.zeros(1, dtype=torch.long).to(DEVICE),
                torch.zeros(1, dtype=torch.long).to(DEVICE),
                torch.zeros(1, dtype=torch.long).to(DEVICE),
            )
            self.model(self.graph_data.x_dict, self.graph_data.edge_index_dict, dummy_batch)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("✅ Modèle GNN (Predictor) chargé et prêt.")

    def predict(self, horse_id, jockey_id, entraineur_id):
        # 1. Mapping des IDs
        c_idx = self.mappings["cheval_map"].get(horse_id)
        j_idx = self.mappings["jockey_map"].get(jockey_id)
        e_idx = self.mappings["entraineur_map"].get(entraineur_id)

        # Gestion des inconnus (Cold Start)
        if c_idx is None or j_idx is None or e_idx is None:
            missing = []
            if c_idx is None:
                missing.append(f"Cheval {horse_id}")
            if j_idx is None:
                missing.append(f"Jockey {jockey_id}")
            if e_idx is None:
                missing.append(f"Entraineur {entraineur_id}")
            return None, f"Entités inconnues : {', '.join(missing)}"

        # 2. Prédiction
        with torch.no_grad():
            # Préparation des tenseurs (batch de taille 1)
            c_tensor = torch.tensor([c_idx], dtype=torch.long).to(DEVICE)
            j_tensor = torch.tensor([j_idx], dtype=torch.long).to(DEVICE)
            e_tensor = torch.tensor([e_idx], dtype=torch.long).to(DEVICE)

            logits = self.model(
                self.graph_data.x_dict,
                self.graph_data.edge_index_dict,
                (c_tensor, j_tensor, e_tensor),
            )
            prob = torch.sigmoid(logits).item()

        return prob, "OK"


server = ModelServer()


@app.on_event("startup")
async def startup_event():
    try:
        server.load_resources()
    except Exception as e:
        print(f"❌ Erreur au démarrage : {e}")
        # On ne crash pas l'app pour permettre le debug, mais le service sera dégradé


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": server.model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_race(request: PredictionRequest):
    if server.model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    score, status = server.predict(request.horse_id, request.jockey_id, request.entraineur_id)

    if score is None:
        # Stratégie de repli ou erreur
        # Pour l'instant, on renvoie 0.5 (incertitude) avec le statut
        return PredictionResponse(score=0.5, status=status)

    return PredictionResponse(score=score, status="OK")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
