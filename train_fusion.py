# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 4: Entraînement du Modèle de Fusion
train_fusion.py

Ce script entraîne le modèle FusionTransformerGNN en combinant les données
séquentielles et le graphe d'entités.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score
import pickle
import os
import numpy as np
from tqdm import tqdm

from models.phase9.transformer_temporal import TemporalTransformer
from models.phase9.graph_nn import EntityGraphNN
from models.phase9.fusion_model import FusionTransformerGNN

# --- Configuration ---
TEMPORAL_DATA_PATH = 'data/phase9/temporal/sequences_v1.pkl'
GRAPH_DATA_PATH = 'data/phase9/graphs/entity_graph_v1.pkl'
TRANSFORMER_WEIGHTS = 'models/phase9/saved_models/temporal_transformer_v1.pth'
GNN_WEIGHTS = 'models/phase9/saved_models/gnn_v1.pth' # Si disponible

MODEL_SAVE_DIR = 'models/phase9/saved_models'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'fusion_model_v1.pth')

# Hyperparamètres
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 5e-4
FUSION_DIM = 64
DROPOUT = 0.2

# --- Dataset ---

class FusionDataset(Dataset):
    def __init__(self, temporal_data, graph_mappings):
        self.sequences = temporal_data['sequences']
        self.labels = temporal_data['labels']
        self.masks = temporal_data['masks']
        self.metadata = temporal_data['metadata']
        self.mappings = graph_mappings
        
        self.valid_indices = []
        self.mapped_indices = [] # Stocke (c_idx, j_idx, e_idx)
        
        print("Alignement des données temporelles et du graphe...")
        
        # Pré-calcul des mappings pour éviter de le faire à chaque __getitem__
        # et pour filtrer les données manquantes
        for idx in range(len(self.sequences)):
            h_id = self.metadata['idx_to_horse_id'][idx]
            j_id = self.metadata['idx_to_jockey_id'][idx]
            e_id = self.metadata['idx_to_entraineur_id'][idx]
            
            # Vérifier si les entités existent dans le graphe
            if (h_id in self.mappings['cheval_map'] and 
                j_id in self.mappings['jockey_map'] and 
                e_id in self.mappings['entraineur_map']):
                
                c_idx = self.mappings['cheval_map'][h_id]
                j_idx = self.mappings['jockey_map'][j_id]
                e_idx = self.mappings['entraineur_map'][e_id]
                
                self.valid_indices.append(idx)
                self.mapped_indices.append((c_idx, j_idx, e_idx))
        
        print(f"Données alignées : {len(self.valid_indices)} / {len(self.sequences)} échantillons conservés.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        
        seq = self.sequences[original_idx]
        label = self.labels[original_idx]
        mask = self.masks[original_idx]
        
        c_idx, j_idx, e_idx = self.mapped_indices[idx]
        
        return (seq, mask, 
                torch.tensor(c_idx, dtype=torch.long), 
                torch.tensor(j_idx, dtype=torch.long), 
                torch.tensor(e_idx, dtype=torch.long), 
                label)

# --- Fonctions d'entraînement ---

def train_one_epoch(model, dataloader, optimizer, criterion, graph_data, device):
    model.train()
    total_loss = 0.0
    
    # Données du graphe sur GPU
    x_dict = {k: v.to(device) for k, v in graph_data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in graph_data.edge_index_dict.items()}
    
    for seq, mask, c_idx, j_idx, e_idx, label in tqdm(dataloader, desc="Training"):
        seq, mask = seq.to(device), mask.to(device)
        c_idx, j_idx, e_idx = c_idx.to(device), j_idx.to(device), e_idx.to(device)
        label = label.to(device).unsqueeze(1) # (batch, 1)
        
        optimizer.zero_grad()
        
        logits, _ = model(seq, mask, x_dict, edge_index_dict, (c_idx, j_idx, e_idx))
        loss = criterion(logits, label)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, graph_data, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    
    x_dict = {k: v.to(device) for k, v in graph_data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in graph_data.edge_index_dict.items()}
    
    with torch.no_grad():
        for seq, mask, c_idx, j_idx, e_idx, label in tqdm(dataloader, desc="Validation"):
            seq, mask = seq.to(device), mask.to(device)
            c_idx, j_idx, e_idx = c_idx.to(device), j_idx.to(device), e_idx.to(device)
            label = label.to(device).unsqueeze(1)
            
            logits, _ = model(seq, mask, x_dict, edge_index_dict, (c_idx, j_idx, e_idx))
            loss = criterion(logits, label)
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits)
            all_labels.append(label.cpu().numpy())
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
    print("--- Début de l'entraînement du Modèle de Fusion ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Chargement des données
    print("Chargement des fichiers de données...")
    with open(TEMPORAL_DATA_PATH, 'rb') as f:
        temporal_data = pickle.load(f)
    
    with open(GRAPH_DATA_PATH, 'rb') as f:
        graph_data_pkl = pickle.load(f)
        graph = graph_data_pkl['graph']
        mappings = graph_data_pkl['mappings']
        
    dataset = FusionDataset(temporal_data, mappings)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialisation des sous-modèles
    print("Initialisation des sous-modèles...")
    
    # Transformer
    transformer = TemporalTransformer(
        input_dim=47, d_model=128, n_head=8, n_layers=4, dropout=0.1, sequence_length=10
    )
    # Charger poids si existent
    if os.path.exists(TRANSFORMER_WEIGHTS):
        print(f"Chargement des poids Transformer depuis {TRANSFORMER_WEIGHTS}")
        transformer.load_state_dict(torch.load(TRANSFORMER_WEIGHTS, map_location=device))
    else:
        print("⚠️ Poids Transformer non trouvés, initialisation aléatoire.")

    # GNN
    gnn = EntityGraphNN(
        metadata=graph.metadata(), hidden_dim=64, out_dim=64, n_layers=3, dropout=0.1
    )
    # Charger poids si existent (attention, on a sauvegardé le state_dict du GNN seul dans train_gnn.py)
    if os.path.exists(GNN_WEIGHTS):
        print(f"Chargement des poids GNN depuis {GNN_WEIGHTS}")
        # Note: train_gnn.py sauvegardait model.gnn.state_dict()
        gnn.load_state_dict(torch.load(GNN_WEIGHTS, map_location=device))
    else:
        print("⚠️ Poids GNN non trouvés, initialisation aléatoire.")
        
    # Initialisation LazyLinear du GNN (important !)
    dummy_x = {k: v.to(device) for k, v in graph.x_dict.items()}
    dummy_edge_index = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
    gnn.to(device)
    with torch.no_grad():
        _ = gnn(dummy_x, dummy_edge_index)

    # 3. Modèle de Fusion
    fusion_model = FusionTransformerGNN(
        transformer_model=transformer,
        gnn_model=gnn,
        transformer_dim=128,
        gnn_dim=64 * 3, # Concaténation de 3 embeddings de dim 64
        fusion_dim=FUSION_DIM,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Modèle Fusion initialisé. Params: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):,}")
    
    # 4. Optimisation
    criterion = nn.BCEWithLogitsLoss()
    # On peut utiliser un LR plus faible pour les sous-modèles pré-entraînés (fine-tuning)
    optimizer = optim.Adam([
        {'params': fusion_model.transformer.parameters(), 'lr': 1e-5},
        {'params': fusion_model.gnn.parameters(), 'lr': 1e-5},
        {'params': fusion_model.fusion_attention.parameters(), 'lr': LEARNING_RATE},
        {'params': fusion_model.classifier.parameters(), 'lr': LEARNING_RATE},
        {'params': fusion_model.temporal_proj.parameters(), 'lr': LEARNING_RATE},
        {'params': fusion_model.graph_proj.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 5. Boucle d'entraînement
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss = train_one_epoch(fusion_model, train_loader, optimizer, criterion, graph, device)
        val_loss, val_auc = evaluate(fusion_model, val_loader, criterion, graph, device)
        
        scheduler.step(val_auc)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(fusion_model.state_dict(), MODEL_SAVE_PATH)
            print(f"✨ Modèle Fusion sauvegardé (AUC: {best_auc:.4f})")

    print("\n--- Entraînement terminé ---")
    print(f"Meilleur AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()
