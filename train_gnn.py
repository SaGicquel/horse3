# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 3: Entraînement du GNN
train_gnn.py

Ce script entraîne le modèle EntityGraphNN en utilisant une approche inductive :
1. Le GNN génère des embeddings pour tous les nœuds (Chevaux, Jockeys, Entraîneurs).
2. Pour chaque course (batch), on récupère les embeddings des entités impliquées.
3. Un classifieur final prédit le résultat de la course (Victoire).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle
import os
from tqdm import tqdm
import numpy as np

from models.phase9.graph_nn import EntityGraphNN, GNNPredictor

# --- Configuration ---
GRAPH_PATH = 'data/phase9/graphs/entity_graph_v1.pkl'
DATA_PATH = 'data/ml_features_complete.csv'
MODEL_SAVE_DIR = 'models/phase9/saved_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'gnn_v1.pth')

# Hyperparamètres
HIDDEN_DIM = 64
OUT_DIM = 32 # Dimension de l'embedding de sortie pour la fusion future
N_LAYERS = 3
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
EPOCHS = 30
WEIGHT_DECAY = 5e-4

# --- Modèle Wrapper pour l'Entraînement ---
# GNNPredictor est maintenant importé depuis models.phase9.graph_nn

# --- Fonctions Utilitaires ---

def load_data():
    """Charge le graphe et prépare les données d'entraînement (indices)."""
    print("Chargement du graphe...")
    with open(GRAPH_PATH, 'rb') as f:
        data = pickle.load(f)
    
    graph = data['graph']
    mappings = data['mappings']
    
    print("Chargement des données de courses...")
    df = pd.read_csv(DATA_PATH)
    
    # Filtrer les colonnes nécessaires
    df = df[['id_cheval', 'id_jockey', 'id_entraineur', 'victoire']].copy()
    
    # Convertir les IDs en indices entiers grâce aux mappings
    # On utilise map et on drop les lignes où l'ID n'est pas trouvé (sécurité)
    df['c_idx'] = df['id_cheval'].map(mappings['cheval_map'])
    df['j_idx'] = df['id_jockey'].map(mappings['jockey_map'])
    df['e_idx'] = df['id_entraineur'].map(mappings['entraineur_map'])
    
    # Supprimer les lignes avec des NaN (entités inconnues dans le graphe)
    initial_len = len(df)
    df.dropna(subset=['c_idx', 'j_idx', 'e_idx'], inplace=True)
    if len(df) < initial_len:
        print(f"⚠️ {initial_len - len(df)} lignes ignorées car entités inconnues dans le graphe.")
    
    # Conversion en tenseurs
    c_indices = torch.tensor(df['c_idx'].values, dtype=torch.long)
    j_indices = torch.tensor(df['j_idx'].values, dtype=torch.long)
    e_indices = torch.tensor(df['e_idx'].values, dtype=torch.long)
    labels = torch.tensor(df['victoire'].values, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(c_indices, j_indices, e_indices, labels)
    
    return graph, dataset

def train_one_epoch(model, dataloader, optimizer, criterion, graph, device):
    model.train()
    total_loss = 0.0
    
    # Déplacer le graphe sur le device une seule fois
    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
    
    for c_idx, j_idx, e_idx, y in tqdm(dataloader, desc="Training"):
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
        for c_idx, j_idx, e_idx, y in tqdm(dataloader, desc="Validation"):
            c_idx, j_idx, e_idx, y = c_idx.to(device), j_idx.to(device), e_idx.to(device), y.to(device)
            
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

# --- Main ---

def main():
    print("--- Début de l'entraînement du GNN ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Chargement
    graph, full_dataset = load_data()
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # 2. Modèle
    gnn_core = EntityGraphNN(
        metadata=graph.metadata(),
        hidden_dim=HIDDEN_DIM,
        out_dim=HIDDEN_DIM, # On sort des embeddings, pas une classif directe ici
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )
    
    model = GNNPredictor(gnn_core, HIDDEN_DIM).to(device)
    
    # Initialisation des LazyLinear par une passe dummy
    print("Initialisation des couches LazyLinear...")
    dummy_x = {k: v.to(device) for k, v in graph.x_dict.items()}
    dummy_edge_index = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
    # On prend juste le premier élément pour le batch dummy
    dummy_batch = (torch.zeros(1, dtype=torch.long).to(device), 
                   torch.zeros(1, dtype=torch.long).to(device), 
                   torch.zeros(1, dtype=torch.long).to(device))
    
    with torch.no_grad():
        _ = model(dummy_x, dummy_edge_index, dummy_batch)
        
    print(f"Modèle initialisé. Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 3. Optimisation
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 4. Boucle
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, graph, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, graph, device)
        
        scheduler.step(val_auc)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH) # On sauvegarde le modèle COMPLET (GNN + Classifier)
            print(f"✨ Modèle GNN sauvegardé (AUC: {best_auc:.4f})")

    print("\n--- Entraînement terminé ---")
    print(f"Meilleur AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()
