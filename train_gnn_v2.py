# -*- coding: utf-8 -*-
"""
Phase 12: Entraînement du GNN V2
train_gnn_v2.py

Entraîne le modèle EntityGraphNNV2 (GAT + BatchNorm).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle
import os
from tqdm import tqdm
import numpy as np

from models.phase12.graph_nn_v2 import EntityGraphNNV2, GNNPredictorV2

# --- Configuration ---
GRAPH_PATH = "data/phase9/graphs/entity_graph_v1.pkl"  # On réutilise le graphe V1
DATA_PATH = "data/ml_features_complete.csv"
MODEL_SAVE_DIR = "models/phase12/saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "gnn_v2.pth")

# Hyperparamètres V2
HIDDEN_DIM = 128
HEADS = 4
N_LAYERS = 3
DROPOUT = 0.2
LEARNING_RATE = 5e-4  # Un peu plus bas car modèle plus complexe
BATCH_SIZE = 512
EPOCHS = 40
WEIGHT_DECAY = 1e-4


def load_data():
    """Charge le graphe et prépare les données d'entraînement (indices)."""
    print("Chargement du graphe...")
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)

    graph = data["graph"]
    mappings = data["mappings"]

    print("Chargement des données de courses...")
    df = pd.read_csv(DATA_PATH)

    # Filtrer les colonnes nécessaires
    df = df[["id_cheval", "id_jockey", "id_entraineur", "victoire"]].copy()

    # Convertir les IDs en indices entiers grâce aux mappings
    df["c_idx"] = df["id_cheval"].map(mappings["cheval_map"])
    df["j_idx"] = df["id_jockey"].map(mappings["jockey_map"])
    df["e_idx"] = df["id_entraineur"].map(mappings["entraineur_map"])

    # Supprimer les lignes avec des NaN
    initial_len = len(df)
    df.dropna(subset=["c_idx", "j_idx", "e_idx"], inplace=True)
    if len(df) < initial_len:
        print(f"⚠️ {initial_len - len(df)} lignes ignorées car entités inconnues dans le graphe.")

    # Split Train/Val (Chronologique si possible, sinon aléatoire pour l'instant)
    # Pour l'instant on fait un split aléatoire simple comme en V1
    # Idéalement on devrait faire un split temporel

    # Conversion en tenseurs
    c_indices = torch.tensor(df["c_idx"].values, dtype=torch.long)
    j_indices = torch.tensor(df["j_idx"].values, dtype=torch.long)
    e_indices = torch.tensor(df["e_idx"].values, dtype=torch.long)
    labels = torch.tensor(df["victoire"].values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(c_indices, j_indices, e_indices, labels)

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    return graph, train_dataset, val_dataset


def train_one_epoch(model, dataloader, optimizer, criterion, graph, device):
    model.train()
    total_loss = 0.0

    # Déplacer le graphe sur le device
    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

    for c_idx, j_idx, e_idx, y in tqdm(dataloader, desc="Training", leave=False):
        c_idx, j_idx, e_idx, y = c_idx.to(device), j_idx.to(device), e_idx.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(x_dict, edge_index_dict, (c_idx, j_idx, e_idx))
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, graph, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

    with torch.no_grad():
        for c_idx, j_idx, e_idx, y in tqdm(dataloader, desc="Validation", leave=False):
            c_idx, j_idx, e_idx, y = (
                c_idx.to(device),
                j_idx.to(device),
                e_idx.to(device),
                y.to(device),
            )

            logits = model(x_dict, edge_index_dict, (c_idx, j_idx, e_idx))
            loss = criterion(logits, y)

            total_loss += loss.item()

            preds = torch.sigmoid(logits)
            all_labels.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    try:
        auc_score = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_score = 0.0

    return avg_loss, auc_score


def main():
    print("--- Début de l'entraînement du GNN V2 (GAT) ---")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # 1. Chargement
    graph, train_dataset, val_dataset = load_data()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialisation Modèle
    print("Initialisation du modèle GNN V2...")
    gnn_core = EntityGraphNNV2(
        metadata=graph.metadata(),
        hidden_dim=HIDDEN_DIM,
        out_dim=HIDDEN_DIM,  # Le GNN sort des embeddings
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        heads=HEADS,
    )

    model = GNNPredictorV2(gnn_core, hidden_dim=HIDDEN_DIM).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # 3. Boucle d'entraînement
    best_auc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, graph, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, graph, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            print(f"🔥 Nouveau record AUC! Sauvegarde dans {MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
