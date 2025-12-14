# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 2: Entraînement du Modèle Transformer
train_transformer.py

Ce script gère le chargement des données, l'instanciation du modèle
TemporalTransformer, et l'exécution de la boucle d'entraînement et de validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np
import os
from tqdm import tqdm

from models.phase9.transformer_temporal import TemporalTransformer

# --- Configuration ---
# Chemins
DATA_PATH = 'data/phase9/temporal/sequences_v1.pkl'
MODEL_SAVE_DIR = 'models/phase9/saved_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'temporal_transformer_v1.pth')

# Hyperparamètres du modèle
INPUT_DIM = 47  # Doit correspondre à la sortie de prepare_temporal_data.py
D_MODEL = 128
N_HEAD = 8
N_LAYERS = 4
DROPOUT = 0.1
SEQUENCE_LENGTH = 10

# Hyperparamètres d'entraînement
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 50
WEIGHT_DECAY = 1e-2
VALIDATION_SPLIT = 0.2
PADDING_VALUE = -1.0

# --- Fonctions Utilitaires ---

def load_data(path: str) -> TensorDataset:
    """Charge les données séquentielles depuis un fichier pickle."""
    print(f"Chargement des données depuis {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    sequences = torch.tensor(data['sequences'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.float32).unsqueeze(1)
    
    print(f"Données chargées. Shape des séquences: {sequences.shape}, Shape des labels: {labels.shape}")
    return TensorDataset(sequences, labels)

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """Exécute une époque d'entraînement."""
    model.train()
    total_loss = 0.0
    
    for sequences, labels in tqdm(dataloader, desc="Training"):
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Créer le masque de padding
        # True pour les positions à ignorer
        src_key_padding_mask = (sequences[:, :, 0] == PADDING_VALUE).to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(sequences, src_key_padding_mask)
        loss = criterion(logits, labels)
        
        # Backward pass et optimisation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Évalue le modèle sur le jeu de données de validation."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validation"):
            sequences, labels = sequences.to(device), labels.to(device)
            
            src_key_padding_mask = (sequences[:, :, 0] == PADDING_VALUE).to(device)
            
            logits = model(sequences, src_key_padding_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    auc_score = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc_score

# --- Script Principal ---

def main():
    """Fonction principale pour l'entraînement du modèle."""
    print("--- Début de l'entraînement du TemporalTransformer ---")
    
    # Définir le device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Utilisation du device: {device}")

    # 1. Chargement et préparation des données
    full_dataset = load_data(DATA_PATH)
    
    # Division en jeux d'entraînement et de validation
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Taille du jeu d'entraînement: {len(train_dataset)}")
    print(f"Taille du jeu de validation: {len(val_dataset)}")

    # 2. Initialisation du modèle
    model = TemporalTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        sequence_length=SEQUENCE_LENGTH
    ).to(device)
    
    print(f"\nModèle initialisé. Nombre de paramètres: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 3. Définition de la fonction de perte et de l'optimiseur
    # BCEWithLogitsLoss est plus stable numériquement que Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. Boucle d'entraînement
    best_val_auc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val AUC = {val_auc:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✨ Nouveau meilleur modèle sauvegardé avec AUC = {best_val_auc:.4f} à {MODEL_SAVE_PATH}")

    print("\n--- Entraînement terminé ---")
    print(f"Meilleur AUC de validation atteint: {best_val_auc:.4f}")
    print(f"Modèle final sauvegardé à: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
