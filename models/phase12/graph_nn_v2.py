# -*- coding: utf-8 -*-
"""
Phase 12: Modèle Graph Neural Network V2 (GAT + BatchNorm)
models/phase12/graph_nn_v2.py

Améliorations par rapport à la V1 :
- Utilisation de GATConv (Graph Attention Network) au lieu de SAGEConv.
- Ajout de BatchNorm pour stabiliser l'apprentissage.
- Embeddings plus profonds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear, BatchNorm


class EntityGraphNNV2(nn.Module):
    """
    GNN Hétérogène V2 avec Attention et Normalisation.
    """

    def __init__(
        self,
        metadata,
        hidden_dim: int = 128,
        out_dim: int = 1,
        n_layers: int = 3,
        dropout: float = 0.2,
        heads: int = 2,
    ):
        """
        Args:
            metadata (tuple): Métadonnées du graphe (node_types, edge_types).
            hidden_dim (int): Dimension des couches cachées (augmentée par rapport à V1).
            out_dim (int): Dimension de sortie.
            n_layers (int): Nombre de couches de convolution.
            dropout (float): Taux de dropout.
            heads (int): Nombre de têtes d'attention pour GAT.
        """
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.heads = heads

        # 1. Projection initiale
        self.lin_dict = nn.ModuleDict()
        self.bn_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_dim)
            self.bn_dict[node_type] = BatchNorm(hidden_dim)

        # 2. Couches de convolution hétérogènes (GAT)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(n_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                # GATConv: (in_channels, out_channels, heads)
                # add_self_loops=False car graphe hétérogène bipartite souvent
                conv_dict[edge_type] = GATConv(
                    (-1, -1),
                    hidden_dim // heads,
                    heads=heads,
                    add_self_loops=False,
                    dropout=dropout,
                )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.bns.append(nn.ModuleDict({nt: BatchNorm(hidden_dim) for nt in metadata[0]}))

        # 3. Tête de classification (optionnelle, si on utilise le GNN pour classifier directement les nœuds)
        # Dans notre cas, on utilise souvent les embeddings pour un classifieur externe, mais on garde ça pour la compatibilité.
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        # 1. Projection initiale
        x_dict_out = {}
        for node_type, x in x_dict.items():
            x = self.lin_dict[node_type](x)
            x = self.bn_dict[node_type](x)
            x = F.elu(x)  # ELU souvent meilleur avec GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_dict_out[node_type] = x

        # 2. Message Passing
        for i, conv in enumerate(self.convs):
            x_dict_out = conv(x_dict_out, edge_index_dict)

            for node_type in x_dict_out:
                # BatchNorm
                x_dict_out[node_type] = self.bns[i][node_type](x_dict_out[node_type])
                # Activation
                x_dict_out[node_type] = F.elu(x_dict_out[node_type])
                # Dropout
                x_dict_out[node_type] = F.dropout(
                    x_dict_out[node_type], p=self.dropout, training=self.training
                )

        return x_dict_out  # On retourne le dictionnaire d'embeddings complet


class GNNPredictorV2(nn.Module):
    """
    Wrapper pour la prédiction de course utilisant les embeddings du GNN V2.
    """

    def __init__(self, gnn_model, hidden_dim):
        super().__init__()
        self.gnn = gnn_model

        # Fusion des embeddings : Cheval + Jockey + Entraîneur
        # hidden_dim * 3 -> 1
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x_dict, edge_index_dict, indices):
        """
        Args:
            x_dict, edge_index_dict: Pour le GNN
            indices: Tuple (c_idx, j_idx, e_idx) des indices des entités pour le batch
        """
        c_idx, j_idx, e_idx = indices

        # 1. Obtenir les embeddings mis à jour par le GNN
        embeddings = self.gnn(x_dict, edge_index_dict)

        c_emb = embeddings["cheval"][c_idx]
        j_emb = embeddings["jockey"][j_idx]
        e_emb = embeddings["entraineur"][e_idx]

        # 2. Concaténation
        x = torch.cat([c_emb, j_emb, e_emb], dim=1)

        # 3. MLP Classifier
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        return self.fc_out(x)
