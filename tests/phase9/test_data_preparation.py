# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 1: Tests Unitaires pour la Préparation des Données

Ce script contient les tests unitaires pour `prepare_temporal_data.py` et
`build_graph_data.py`. Il valide la forme, le contenu et l'intégrité
des données générées pour les modèles Transformer et GNN.

Tests:
1.  **Test Données Temporelles**:
    - `test_temporal_output_exists`: Vérifie que le fichier de séquences est créé.
    - `test_temporal_data_structure`: Valide la structure du dictionnaire (clés).
    - `test_sequence_shapes`: Contrôle les dimensions des tenseurs (séquences, labels, masques).
    - `test_padding_and_mask`: S'assure que le padding et les masques sont corrects.
    - `test_normalization_range`: Vérifie que les données normalisées sont bien dans [-1, 1].

2.  **Test Données de Graphe**:
    - `test_graph_output_exists`: Vérifie que le fichier de graphe est créé.
    - `test_graph_data_structure`: Valide la structure du dictionnaire (clés 'graph', 'mappings').
    - `test_hetero_data_validation`: S'assure que l'objet `HeteroData` est valide.
    - `test_node_features_shape`: Contrôle que les features des noeuds ont la bonne dimension.
    - `test_edge_indices`: Vérifie que les arêtes connectent bien les bons types de noeuds.
"""

import pytest
import os
import pickle
import torch
from torch_geometric.data import HeteroData

# Chemins vers les fichiers de sortie à tester
TEMPORAL_DATA_PATH = 'data/phase9/temporal/sequences_v1.pkl'
GRAPH_DATA_PATH = 'data/phase9/graphs/entity_graph_v1.pkl'

# --- Fixtures ---

@pytest.fixture(scope="module")
def temporal_data():
    """Charge les données temporelles une seule fois pour tous les tests du module."""
    if not os.path.exists(TEMPORAL_DATA_PATH):
        pytest.fail(f"Fichier de données temporelles non trouvé: {TEMPORAL_DATA_PATH}. Exécutez d'abord prepare_temporal_data.py.")
    with open(TEMPORAL_DATA_PATH, 'rb') as f:
        return pickle.load(f)

@pytest.fixture(scope="module")
def graph_data():
    """Charge les données de graphe une seule fois pour tous les tests du module."""
    if not os.path.exists(GRAPH_DATA_PATH):
        pytest.fail(f"Fichier de données de graphe non trouvé: {GRAPH_DATA_PATH}. Exécutez d'abord build_graph_data.py.")
    with open(GRAPH_DATA_PATH, 'rb') as f:
        return pickle.load(f)

# --- Tests pour prepare_temporal_data.py ---

@pytest.mark.sprint1
class TestTemporalData:
    def test_temporal_output_exists(self):
        """Vérifie que le fichier de sortie des données temporelles existe."""
        assert os.path.exists(TEMPORAL_DATA_PATH), f"Le fichier {TEMPORAL_DATA_PATH} devrait exister."

    def test_temporal_data_structure(self, temporal_data):
        """Valide que le dictionnaire de données temporelles a les bonnes clés."""
        expected_keys = {'sequences', 'labels', 'masks', 'metadata'}
        assert set(temporal_data.keys()) == expected_keys

    def test_sequence_shapes(self, temporal_data):
        """Vérifie les dimensions des tenseurs de séquences, labels et masques."""
        sequences = temporal_data['sequences']
        labels = temporal_data['labels']
        masks = temporal_data['masks']
        
        assert isinstance(sequences, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert isinstance(masks, torch.Tensor)

        num_samples = sequences.shape[0]
        seq_length = temporal_data['metadata']['seq_length']
        num_features = len(temporal_data['metadata']['feature_cols'])

        assert sequences.shape == (num_samples, seq_length, num_features)
        assert labels.shape == (num_samples,)
        assert masks.shape == (num_samples, seq_length)
        assert masks.dtype == torch.bool

    def test_padding_and_mask(self, temporal_data):
        """Vérifie un exemple de padding et de masque."""
        sequences = temporal_data['sequences']
        masks = temporal_data['masks']
        
        # Trouve une séquence avec du padding (si elle existe)
        padded_sample_idx = -1
        for i in range(masks.shape[0]):
            if not masks[i, 0]: # Le premier élément est paddé
                padded_sample_idx = i
                break
        
        if padded_sample_idx != -1:
            mask = masks[padded_sample_idx]
            sequence = sequences[padded_sample_idx]
            
            first_true = torch.where(mask)[0][0]
            
            # Les valeurs paddées doivent être proches de -1 après normalisation
            # On utilise une tolérance car la normalisation n'est pas parfaite
            padded_values = sequence[:first_true]
            assert torch.all(padded_values < -0.99), "Les valeurs de padding ne sont pas toutes proches de -1"
            # Les valeurs non paddées ne doivent pas être toutes à zéro
            assert not torch.all(sequence[first_true:] == 0)

    def test_normalization_range(self, temporal_data):
        """Vérifie que les features normalisées sont dans l'intervalle [-1, 1]."""
        sequences = temporal_data['sequences']
        # On ne teste que les valeurs non paddées
        active_sequences = sequences[temporal_data['masks']]
        
        assert torch.min(active_sequences) >= -1.001
        assert torch.max(active_sequences) <= 1.001

# --- Tests pour build_graph_data.py ---

@pytest.mark.sprint1
class TestGraphData:
    def test_graph_output_exists(self):
        """Vérifie que le fichier de sortie du graphe existe."""
        assert os.path.exists(GRAPH_DATA_PATH), f"Le fichier {GRAPH_DATA_PATH} devrait exister."

    def test_graph_data_structure(self, graph_data):
        """Valide que le dictionnaire de graphe a les bonnes clés."""
        expected_keys = {'graph', 'mappings'}
        assert set(graph_data.keys()) == expected_keys
        assert isinstance(graph_data['graph'], HeteroData)

    def test_hetero_data_validation(self, graph_data):
        """Valide l'intégrité de l'objet HeteroData."""
        graph = graph_data['graph']
        # La méthode validate() de PyG lève une exception en cas de problème
        try:
            graph.validate()
            validation_passed = True
        except Exception:
            validation_passed = False
        assert validation_passed, "graph.validate() a levé une exception"

    def test_node_features_shape(self, graph_data):
        """Vérifie que les features des noeuds ont les bonnes dimensions."""
        graph = graph_data['graph']
        
        assert 'cheval' in graph.node_types
        assert 'jockey' in graph.node_types
        assert 'entraineur' in graph.node_types
        
        num_chevaux = graph['cheval'].x.shape[0]
        num_jockeys = graph['jockey'].x.shape[0]
        num_entraineurs = graph['entraineur'].x.shape[0]
        
        # Le nombre de features est défini dans le script de build
        assert graph['cheval'].x.shape == (num_chevaux, 6)
        assert graph['jockey'].x.shape == (num_jockeys, 2)
        assert graph['entraineur'].x.shape == (num_entraineurs, 2)

    def test_edge_indices(self, graph_data):
        """Vérifie la validité des arêtes."""
        graph = graph_data['graph']
        
        # Relation cheval -> jockey
        edge_index_cj = graph['cheval', 'est_monte_par', 'jockey'].edge_index
        assert edge_index_cj.shape[0] == 2
        assert edge_index_cj[0].max() < graph['cheval'].num_nodes
        assert edge_index_cj[1].max() < graph['jockey'].num_nodes

        # Relation cheval -> entraineur
        edge_index_ce = graph['cheval', 'est_entraine_par', 'entraineur'].edge_index
        assert edge_index_ce.shape[0] == 2
        assert edge_index_ce[0].max() < graph['cheval'].num_nodes
        assert edge_index_ce[1].max() < graph['entraineur'].num_nodes
        
        # Relation inverse jockey -> cheval
        edge_index_jc_rev = graph['jockey', 'rev_est_monte_par', 'cheval'].edge_index
        assert torch.equal(edge_index_jc_rev, edge_index_cj.flip(0))
