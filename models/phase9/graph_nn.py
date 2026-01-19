# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 3: Modèle Graph Neural Network (GNN)
models/phase9/graph_nn.py

Ce fichier définit l'architecture du GNN pour apprendre des embeddings
à partir des relations entre entités (Cheval, Jockey, Entraîneur).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear


class EntityGraphNN(nn.Module):
    """
    Un GNN hétérogène pour apprendre des représentations de nœuds (embeddings)
    et effectuer une classification binaire sur les nœuds 'cheval'.
    """

    def __init__(
        self,
        metadata,
        hidden_dim: int = 64,
        out_dim: int = 1,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            metadata (tuple): Métadonnées du graphe (node_types, edge_types) fournies par data.metadata().
            hidden_dim (int): Dimension des couches cachées.
            out_dim (int): Dimension de sortie (1 pour classification binaire).
            n_layers (int): Nombre de couches de convolution de graphe.
            dropout (float): Taux de dropout.
        """
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # 1. Projection initiale des features des nœuds vers hidden_dim
        # Comme chaque type de nœud peut avoir un nombre différent de features,
        # on utilise un dictionnaire de couches Linear.
        # Note: -1 permet à LazyLinear d'inférer la dimension d'entrée automatiquement lors du premier forward.
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_dim)

        # 2. Couches de convolution hétérogènes
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            # Pour chaque type de relation, on définit une convolution (ici SAGEConv)
            conv_dict = {}
            for edge_type in metadata[1]:
                # SAGEConv est efficace pour les graphes de grande taille
                # (source_dim, target_dim) -> hidden_dim
                conv_dict[edge_type] = SAGEConv((-1, -1), hidden_dim)

            # HeteroConv combine ces convolutions
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # 3. Tête de classification pour les nœuds 'cheval'
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        Passe avant du modèle.

        Args:
            x_dict (dict): Dictionnaire {node_type: x} contenant les features des nœuds.
            edge_index_dict (dict): Dictionnaire {edge_type: edge_index} contenant la structure du graphe.

        Returns:
            torch.Tensor: Logits pour les nœuds de type 'cheval'.
        """
        # 1. Projection initiale et activation
        x_dict_out = {}
        for node_type, x in x_dict.items():
            x_dict_out[node_type] = self.lin_dict[node_type](x)
            x_dict_out[node_type] = F.relu(x_dict_out[node_type])
            x_dict_out[node_type] = F.dropout(
                x_dict_out[node_type], p=self.dropout, training=self.training
            )

        # 2. Message Passing (Couches GNN)
        for conv in self.convs:
            x_dict_out = conv(x_dict_out, edge_index_dict)

            # Appliquer ReLU et Dropout après chaque couche de conv
            for node_type in x_dict_out:
                x_dict_out[node_type] = F.relu(x_dict_out[node_type])
                x_dict_out[node_type] = F.dropout(
                    x_dict_out[node_type], p=self.dropout, training=self.training
                )

        # 3. Classification (uniquement pour les chevaux)
        # On suppose que l'objectif est de prédire quelque chose sur les chevaux
        cheval_embeddings = x_dict_out["cheval"]
        logits = self.classifier(cheval_embeddings)

        return logits


class GNNPredictor(nn.Module):
    """
    Wrapper qui combine le GNN et une tête de classification pour prédire
    le résultat d'une course à partir des embeddings des entités.
    """

    def __init__(self, gnn_model, hidden_dim):
        super().__init__()
        self.gnn = gnn_model

        # Tête de classification qui prend la concaténation des embeddings
        # (Cheval + Jockey + Entraîneur)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_dict, edge_index_dict, batch_indices):
        """
        Args:
            x_dict, edge_index_dict: Entrées pour le GNN.
            batch_indices (tuple): Indices (cheval_idx, jockey_idx, entraineur_idx) pour le batch courant.
        """
        # 1. Générer les embeddings pour TOUS les nœuds
        # Le GNN renvoie un dictionnaire d'embeddings {node_type: embedding_tensor}
        embeddings_dict = self.gnn_forward_full(x_dict, edge_index_dict)

        c_idx, j_idx, e_idx = batch_indices

        # 2. Récupérer les embeddings pour les entités du batch
        c_emb = embeddings_dict["cheval"][c_idx]
        j_emb = embeddings_dict["jockey"][j_idx]
        e_emb = embeddings_dict["entraineur"][e_idx]

        # 3. Concaténer et classifier
        combined = torch.cat([c_emb, j_emb, e_emb], dim=1)
        logits = self.classifier(combined)

        return logits

    def gnn_forward_full(self, x_dict, edge_index_dict):
        """
        Version modifiée du forward de EntityGraphNN pour retourner tous les embeddings.
        """
        # Projection initiale
        x_dict_out = {}
        for node_type, x in x_dict.items():
            x_dict_out[node_type] = self.gnn.lin_dict[node_type](x)
            x_dict_out[node_type] = F.relu(x_dict_out[node_type])
            x_dict_out[node_type] = F.dropout(
                x_dict_out[node_type], p=self.gnn.dropout, training=self.gnn.training
            )

        # Message Passing
        for conv in self.gnn.convs:
            x_dict_out = conv(x_dict_out, edge_index_dict)
            for node_type in x_dict_out:
                x_dict_out[node_type] = F.relu(x_dict_out[node_type])
                x_dict_out[node_type] = F.dropout(
                    x_dict_out[node_type], p=self.gnn.dropout, training=self.gnn.training
                )

        return x_dict_out


if __name__ == "__main__":
    # --- Test rapide du modèle ---
    from torch_geometric.data import HeteroData

    print("--- Test du modèle EntityGraphNN ---")

    # Création d'un graphe hétérogène factice
    data = HeteroData()

    # Nœuds
    data["cheval"].x = torch.randn(100, 16)  # 100 chevaux, 16 features
    data["jockey"].x = torch.randn(20, 8)  # 20 jockeys, 8 features
    data["entraineur"].x = torch.randn(10, 8)  # 10 entraîneurs, 8 features

    # Arêtes (Relations)
    # Cheval -> Jockey
    data["cheval", "monte_par", "jockey"].edge_index = torch.randint(0, 20, (2, 50))
    # Jockey -> Cheval (Inverse)
    data["jockey", "monte", "cheval"].edge_index = torch.randint(0, 100, (2, 50))

    print(f"Métadonnées du graphe: {data.metadata()}")

    # Instanciation du modèle
    model = EntityGraphNN(metadata=data.metadata(), hidden_dim=32, out_dim=1, n_layers=2)

    print("Modèle créé.")

    # Passe avant
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)

    print(f"Shape des logits (cheval): {logits.shape}")

    assert logits.shape == (100, 1), "La shape de sortie doit correspondre au nombre de chevaux."
    print("✅ Test EntityGraphNN réussi !")
