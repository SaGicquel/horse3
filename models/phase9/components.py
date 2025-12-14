# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 2: Composants du Modèle
models/phase9/components.py

Ce fichier contient des modules PyTorch réutilisables pour les différentes
architectures de modèles (Transformer, GNN, etc.).
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injecte des informations sur la position des tokens dans la séquence.
    Ceci est crucial pour les Transformers, qui ne traitent pas l'ordre
    des séquences de manière inhérente.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model (int): La dimension des embeddings (doit être paire).
            dropout (float): Le taux de dropout à appliquer.
            max_len (int): La longueur maximale des séquences.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Crée une matrice de position (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        
        # Calcule les termes de la formule de l'encodage positionnel
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        
        # Applique sin aux indices pairs et cos aux indices impairs
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # `register_buffer` enregistre le tenseur comme un état persistant du module,
        # mais pas comme un paramètre à entraîner.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Le tenseur d'input de shape (seq_len, batch_size, d_model).
        
        Returns:
            torch.Tensor: Le tenseur avec l'encodage positionnel ajouté.
        """
        # Ajoute l'encodage positionnel aux embeddings d'entrée
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

if __name__ == '__main__':
    # Test rapide du module PositionalEncoding
    d_model = 512
    seq_len = 100
    batch_size = 32

    pe = PositionalEncoding(d_model)
    
    # Crée un tenseur d'entrée factice
    x_input = torch.randn(seq_len, batch_size, d_model)
    
    # Applique l'encodage positionnel
    x_output = pe(x_input)

    print(f"Shape de l'input: {x_input.shape}")
    print(f"Shape de l'output: {x_output.shape}")
    
    assert x_input.shape == x_output.shape, "La shape ne doit pas changer."
    
    # Vérifie que quelque chose a été ajouté
    assert not torch.equal(x_input, x_output), "L'output doit être différent de l'input."

    print("\n✅ Test de PositionalEncoding réussi !")
