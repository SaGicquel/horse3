# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 4: Modèle de Fusion Multi-Tâches
models/phase9/fusion_model.py

Ce modèle combine les représentations apprises par le Transformer Temporel
(historique des courses) et le GNN (relations entités) pour une prédiction finale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionAttention(nn.Module):
    """
    Mécanisme d'attention pour pondérer l'importance relative
    des embeddings temporels et relationnels.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 2), # 2 poids : un pour temporal, un pour graph
            nn.Softmax(dim=1)
        )

    def forward(self, temporal_emb, graph_emb):
        # (batch_size, embed_dim * 2)
        combined = torch.cat([temporal_emb, graph_emb], dim=1)
        # (batch_size, 2)
        weights = self.attention(combined)
        
        # Pondération
        # weights[:, 0] -> poids pour temporal
        # weights[:, 1] -> poids pour graph
        weighted_temporal = temporal_emb * weights[:, 0].unsqueeze(1)
        weighted_graph = graph_emb * weights[:, 1].unsqueeze(1)
        
        return weighted_temporal + weighted_graph, weights

class FusionTransformerGNN(nn.Module):
    """
    Modèle hybride fusionnant Transformer et GNN.
    """
    def __init__(self, transformer_model, gnn_model, transformer_dim, gnn_dim, fusion_dim=64, dropout=0.1):
        """
        Args:
            transformer_model (nn.Module): Instance pré-entraînée (ou non) de TemporalTransformer.
            gnn_model (nn.Module): Instance pré-entraînée (ou non) de EntityGraphNN.
            transformer_dim (int): Dimension de sortie du Transformer (d_model).
            gnn_dim (int): Dimension de sortie du GNN (hidden_dim * 3 car concat C+J+E).
            fusion_dim (int): Dimension de l'espace latent commun pour la fusion.
        """
        super().__init__()
        self.transformer = transformer_model
        self.gnn = gnn_model
        
        # Projecteurs pour aligner les dimensions vers fusion_dim
        self.temporal_proj = nn.Linear(transformer_dim, fusion_dim)
        self.graph_proj = nn.Linear(gnn_dim, fusion_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Mécanisme de fusion
        self.fusion_attention = FusionAttention(fusion_dim)
        
        # Tête de classification finale
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1)
        )

    def forward(self, 
                temporal_src, temporal_mask, 
                graph_x_dict, graph_edge_index_dict, batch_indices):
        """
        Args:
            temporal_src, temporal_mask: Inputs pour le Transformer.
            graph_x_dict, graph_edge_index_dict: Inputs pour le GNN (graphe complet).
            batch_indices: Tuple (c_idx, j_idx, e_idx) pour sélectionner les nœuds du batch.
        """
        # --- 1. Branche Temporelle (Transformer) ---
        # Le Transformer retourne des logits (batch_size, 1) dans sa version actuelle.
        # Nous devons modifier ou accéder à la couche avant la classification pour avoir l'embedding.
        # Pour l'instant, supposons que nous modifions le Transformer pour retourner l'embedding
        # ou que nous utilisons un hook.
        # HACK: On va appeler les sous-modules du transformer manuellement pour récupérer l'embedding poolé.
        
        # (batch_size, seq_len, d_model)
        embedded_src = self.transformer.input_embedding(temporal_src) * torch.sqrt(torch.tensor(self.transformer.d_model))
        pos_encoded_src = self.transformer.pos_encoder(embedded_src.transpose(0, 1)).transpose(0, 1)
        transformer_output = self.transformer.transformer_encoder(pos_encoded_src, src_key_padding_mask=temporal_mask)
        
        # Pooling (Moyenne masquée)
        mask_expanded = ~temporal_mask.unsqueeze(-1)
        temporal_emb_raw = (transformer_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Projection
        temporal_emb = self.temporal_proj(temporal_emb_raw)
        temporal_emb = F.relu(temporal_emb)
        temporal_emb = self.dropout(temporal_emb)

        # --- 2. Branche Relationnelle (GNN) ---
        # Forward complet du GNN pour avoir tous les embeddings
        # Note: Idéalement, on ne recalculerait pas tout le graphe à chaque batch si le graphe est statique,
        # mais pour l'entraînement end-to-end, c'est nécessaire.
        
        # On réutilise la logique "gnn_forward_full" qu'on avait mise dans train_gnn.py
        # Il faudrait idéalement que cette méthode soit dans EntityGraphNN.
        # On va le faire "à la main" ici en itérant sur les modules du GNN.
        
        x_dict_out = {}
        for node_type, x in graph_x_dict.items():
            x_dict_out[node_type] = self.gnn.lin_dict[node_type](x)
            x_dict_out[node_type] = torch.relu(x_dict_out[node_type])
            x_dict_out[node_type] = torch.dropout(x_dict_out[node_type], p=self.gnn.dropout, train=self.gnn.training)

        for conv in self.gnn.convs:
            x_dict_out = conv(x_dict_out, graph_edge_index_dict)
            for node_type in x_dict_out:
                x_dict_out[node_type] = torch.relu(x_dict_out[node_type])
                x_dict_out[node_type] = torch.dropout(x_dict_out[node_type], p=self.gnn.dropout, train=self.gnn.training)
        
        # Sélection des embeddings du batch
        c_idx, j_idx, e_idx = batch_indices
        c_emb = x_dict_out['cheval'][c_idx]
        j_emb = x_dict_out['jockey'][j_idx]
        e_emb = x_dict_out['entraineur'][e_idx]
        
        # Concaténation (Cheval + Jockey + Entraîneur)
        graph_emb_raw = torch.cat([c_emb, j_emb, e_emb], dim=1)
        
        # Projection
        graph_emb = self.graph_proj(graph_emb_raw)
        graph_emb = F.relu(graph_emb)
        graph_emb = self.dropout(graph_emb)

        # --- 3. Fusion ---
        fused_emb, weights = self.fusion_attention(temporal_emb, graph_emb)
        
        # --- 4. Classification ---
        logits = self.classifier(fused_emb)
        
        return logits, weights

if __name__ == '__main__':
    # Test rapide
    print("--- Test FusionTransformerGNN ---")
    # TODO: Ajouter un mock test ici
    print("Classe définie.")
