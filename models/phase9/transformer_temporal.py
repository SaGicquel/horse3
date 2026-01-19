# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 2: Modèle Transformer Temporel
models/phase9/transformer_temporal.py

Ce fichier définit l'architecture du modèle Transformer qui traitera les
séquences de données temporelles (historique des 10 dernières courses).
"""

import torch
import torch.nn as nn
import math
from models.phase9.components import PositionalEncoding


class TemporalTransformer(nn.Module):
    """
    Un modèle Transformer pour la classification binaire basé sur des séquences
    de caractéristiques de courses.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        dropout: float = 0.1,
        sequence_length: int = 10,
    ):
        """
        Args:
            input_dim (int): Dimension des caractéristiques d'entrée pour chaque pas de temps.
            d_model (int): Dimension interne du modèle (doit être divisible par n_head).
            n_head (int): Nombre de têtes dans l'attention multi-têtes.
            n_layers (int): Nombre de couches dans l'encodeur Transformer.
            dropout (float): Taux de dropout.
            sequence_length (int): Longueur des séquences d'entrée.
        """
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model

        # 1. Couche d'embedding d'entrée
        # Projette la dimension des caractéristiques d'entrée (input_dim) vers la
        # dimension interne du modèle (d_model).
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 2. Encodage Positionnel
        # Ajoute l'information de position dans la séquence.
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=sequence_length)

        # 3. Couches de l'encodeur Transformer
        # Le cœur du modèle, qui applique l'auto-attention.
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        # 4. Tête de classification
        # Une couche linéaire pour produire un logit unique pour la classification binaire.
        self.classifier_head = nn.Linear(d_model, 1)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise les poids du modèle."""
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier_head.bias.data.zero_()
        self.classifier_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du modèle.

        Args:
            src (torch.Tensor): Tenseur d'entrée de shape (batch_size, seq_len, input_dim).
            src_key_padding_mask (torch.Tensor): Masque pour ignorer les éléments de padding.
                                                  Shape (batch_size, seq_len).
                                                  Les positions à ignorer sont marquées par `True`.

        Returns:
            torch.Tensor: Logits de sortie de shape (batch_size, 1).
        """
        # 1. Appliquer l'embedding d'entrée
        # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, d_model)
        embedded_src = self.input_embedding(src) * math.sqrt(self.d_model)

        # 2. Ajouter l'encodage positionnel
        # Note: PositionalEncoding attend (seq_len, batch_size, d_model), donc on transpose.
        pos_encoded_src = self.pos_encoder(embedded_src.transpose(0, 1)).transpose(0, 1)

        # 3. Passer à travers l'encodeur Transformer
        # Le masque `src_key_padding_mask` empêche l'attention de se porter sur les
        # tokens de padding (ceux que nous avons ajoutés pour que toutes les séquences
        # aient la même longueur).
        transformer_output = self.transformer_encoder(
            pos_encoded_src, src_key_padding_mask=src_key_padding_mask
        )

        # 4. Pooling: Agréger les sorties du Transformer
        # On prend la sortie du premier token [CLS] (si on en avait un) ou la moyenne des sorties.
        # Ici, nous allons moyenner les sorties qui ne sont pas masquées.
        mask_expanded = ~src_key_padding_mask.unsqueeze(-1)
        pooled_output = (transformer_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

        # 5. Passer à travers la tête de classification
        # (batch_size, d_model) -> (batch_size, 1)
        logits = self.classifier_head(pooled_output)

        return logits


if __name__ == "__main__":
    # --- Paramètres de test ---
    BATCH_SIZE = 32
    SEQ_LEN = 10
    INPUT_DIM = 47  # Doit correspondre à la sortie de prepare_temporal_data.py
    D_MODEL = 128
    N_HEAD = 8
    N_LAYERS = 4
    DROPOUT = 0.1

    print("--- Test du modèle TemporalTransformer ---")

    # Création du modèle
    model = TemporalTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        sequence_length=SEQ_LEN,
    )
    print(
        f"Nombre de paramètres: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Création de données factices
    # Un batch de 32 séquences, chacune de longueur 10 avec 47 features.
    dummy_src = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)

    # Création d'un masque de padding factice
    # Simulons que les 5 derniers exemples du batch ont 3 tokens de padding à la fin.
    dummy_mask = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
    dummy_mask[-5:, -3:] = True  # Les 3 derniers tokens des 5 dernières séquences sont masqués

    print(f"\nShape de l'input (src): {dummy_src.shape}")
    print(f"Shape du masque (mask): {dummy_mask.shape}")

    # Passe avant
    model.eval()
    with torch.no_grad():
        output_logits = model(dummy_src, dummy_mask)

    print(f"Shape de la sortie (logits): {output_logits.shape}")

    # Vérifications
    assert output_logits.shape == (
        BATCH_SIZE,
        1,
    ), f"La shape de sortie est incorrecte: {output_logits.shape}"
    print("\n✅ Le test de passe avant a réussi, les dimensions sont correctes.")
