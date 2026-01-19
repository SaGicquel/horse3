# -*- coding: utf-8 -*-
"""
Phase 9 - Sprint 2: Tests pour les Modèles
tests/phase9/test_models.py

Ce fichier contient les tests unitaires pour les composants et les modèles
définis dans le cadre du Sprint 2.
"""

import torch
import pytest
import os

from models.phase9.components import PositionalEncoding
from models.phase9.transformer_temporal import TemporalTransformer
from models.phase9.graph_nn import EntityGraphNN
from models.phase9.fusion_model import FusionTransformerGNN
from torch_geometric.data import HeteroData

# --- Fixtures ---


@pytest.fixture(scope="module")
def model_params():
    """Fournit les hyperparamètres communs pour les tests de modèles."""
    return {
        "input_dim": 47,
        "d_model": 64,
        "n_head": 4,
        "n_layers": 2,
        "dropout": 0.1,
        "sequence_length": 10,
        "batch_size": 16,
    }


# --- Tests pour `components.py` ---


class TestPositionalEncoding:
    def test_shape(self, model_params):
        """Vérifie que la shape de sortie est identique à la shape d'entrée."""
        d_model = model_params["d_model"]
        seq_len = model_params["sequence_length"]
        batch_size = model_params["batch_size"]

        pe = PositionalEncoding(d_model, max_len=seq_len)
        x_input = torch.randn(seq_len, batch_size, d_model)
        x_output = pe(x_input)

        assert x_output.shape == x_input.shape

    def test_values_changed(self, model_params):
        """Vérifie que l'encodage positionnel modifie bien les valeurs d'entrée."""
        d_model = model_params["d_model"]

        pe = PositionalEncoding(d_model)
        x_input = torch.zeros(10, 1, d_model)  # Input de zéros
        x_output = pe(x_input)

        # La sortie ne doit plus être composée uniquement de zéros
        assert not torch.all(x_output == 0)
        # L'output doit être différent de l'input
        assert not torch.equal(x_input, x_output)


# --- Tests pour `transformer_temporal.py` ---


class TestTemporalTransformer:
    @pytest.fixture
    def model(self, model_params):
        """Crée une instance du modèle TemporalTransformer."""
        return TemporalTransformer(
            input_dim=model_params["input_dim"],
            d_model=model_params["d_model"],
            n_head=model_params["n_head"],
            n_layers=model_params["n_layers"],
            dropout=model_params["dropout"],
            sequence_length=model_params["sequence_length"],
        )

    def test_forward_pass_shape(self, model, model_params):
        """Vérifie la shape de la sortie après une passe avant."""
        batch_size = model_params["batch_size"]
        seq_len = model_params["sequence_length"]
        input_dim = model_params["input_dim"]

        src = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            logits = model(src, mask)

        assert logits.shape == (batch_size, 1)

    def test_padding_mask_effect(self, model, model_params):
        """
        Vérifie que le masque de padding a un effet sur la sortie.
        Une entrée avec et sans masque doit produire des résultats différents.
        """
        batch_size = model_params["batch_size"]
        seq_len = model_params["sequence_length"]
        input_dim = model_params["input_dim"]

        src = torch.randn(batch_size, seq_len, input_dim)

        # Masque vide (pas de padding)
        no_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        # Masque avec padding (les 3 derniers tokens sont masqués)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        padding_mask[:, -3:] = True

        model.eval()
        with torch.no_grad():
            output_no_mask = model(src, no_mask)
            output_with_mask = model(src, padding_mask)

        # Les sorties devraient être différentes car le pooling est affecté par le masque
        assert not torch.allclose(
            output_no_mask, output_with_mask
        ), "Le masque de padding n'a eu aucun effet sur la sortie."

    def test_batch_independence(self, model, model_params):
        """
        Vérifie que la sortie pour un élément du batch ne dépend pas des autres.
        """
        batch_size = 2
        seq_len = model_params["sequence_length"]
        input_dim = model_params["input_dim"]

        src = torch.randn(batch_size, seq_len, input_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            # Passe avant avec un batch de 2
            output_batch = model(src, mask)

            # Passe avant pour chaque élément séparément
            output_1 = model(src[0].unsqueeze(0), mask[0].unsqueeze(0))
            output_2 = model(src[1].unsqueeze(0), mask[1].unsqueeze(0))

        # Les résultats doivent être (presque) identiques
        assert torch.allclose(output_batch[0], output_1, atol=1e-6)
        assert torch.allclose(output_batch[1], output_2, atol=1e-6)


# --- Tests pour `graph_nn.py` ---


class TestEntityGraphNN:
    @pytest.fixture
    def dummy_graph(self):
        """Crée un graphe hétérogène factice."""
        data = HeteroData()
        # Nœuds
        data["cheval"].x = torch.randn(10, 16)
        data["jockey"].x = torch.randn(5, 8)
        data["entraineur"].x = torch.randn(3, 8)

        # Arêtes
        # Attention: les indices doivent être valides par rapport au nombre de nœuds
        # Cheval (10) -> Jockey (5)
        data["cheval", "monte_par", "jockey"].edge_index = torch.stack(
            [
                torch.randint(0, 10, (20,)),  # Source: Cheval
                torch.randint(0, 5, (20,)),  # Target: Jockey
            ]
        )

        # Jockey (5) -> Cheval (10)
        data["jockey", "monte", "cheval"].edge_index = torch.stack(
            [
                torch.randint(0, 5, (20,)),  # Source: Jockey
                torch.randint(0, 10, (20,)),  # Target: Cheval
            ]
        )

        # Cheval (10) -> Entraineur (3)
        data["cheval", "entraine_par", "entraineur"].edge_index = torch.stack(
            [
                torch.randint(0, 10, (20,)),  # Source: Cheval
                torch.randint(0, 3, (20,)),  # Target: Entraineur
            ]
        )

        # Entraineur (3) -> Cheval (10)
        data["entraineur", "entraine", "cheval"].edge_index = torch.stack(
            [
                torch.randint(0, 3, (20,)),  # Source: Entraineur
                torch.randint(0, 10, (20,)),  # Target: Cheval
            ]
        )

        return data

    def test_gnn_initialization(self, dummy_graph):
        """Vérifie que le modèle s'initialise correctement."""
        model = EntityGraphNN(metadata=dummy_graph.metadata(), hidden_dim=32, out_dim=1, n_layers=2)
        assert isinstance(model, EntityGraphNN)

    def test_gnn_forward_pass(self, dummy_graph):
        """Vérifie la passe avant du GNN."""
        model = EntityGraphNN(metadata=dummy_graph.metadata(), hidden_dim=32, out_dim=1, n_layers=2)

        model.eval()
        with torch.no_grad():
            logits = model(dummy_graph.x_dict, dummy_graph.edge_index_dict)

        # Vérifie la shape de sortie (doit correspondre au nombre de chevaux)
        assert logits.shape == (10, 1)

    def test_gnn_predictor_wrapper(self, dummy_graph):
        """Vérifie le wrapper GNNPredictor utilisé pour l'entraînement."""
        from train_gnn import GNNPredictor

        gnn_core = EntityGraphNN(
            metadata=dummy_graph.metadata(), hidden_dim=32, out_dim=32, n_layers=2
        )

        model = GNNPredictor(gnn_core, hidden_dim=32)

        # Indices factices pour un batch de 2 courses
        c_idx = torch.tensor([0, 1])
        j_idx = torch.tensor([0, 1])
        e_idx = torch.tensor([0, 1])

        model.eval()
        with torch.no_grad():
            logits = model(dummy_graph.x_dict, dummy_graph.edge_index_dict, (c_idx, j_idx, e_idx))

        assert logits.shape == (2, 1)


# --- Tests pour `train_transformer.py` (simulation) ---
# Note: Ces tests ne lancent pas un entraînement complet mais vérifient
# les fonctions utilitaires si possible.


# Pour tester `load_data`, nous avons besoin d'un fichier de données factice.
@pytest.fixture(scope="session")
def dummy_data_file(tmpdir_factory):
    """Crée un fichier de données pickle factice pour les tests."""
    file_path = tmpdir_factory.mktemp("data").join("dummy_sequences.pkl")

    data = {
        "sequences": torch.randn(100, 10, 47).numpy(),
        "labels": torch.randint(0, 2, (100,)).numpy(),
    }

    import pickle

    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    return str(file_path)


def test_load_data(dummy_data_file):
    """Teste la fonction de chargement de données."""
    from train_transformer import load_data

    dataset = load_data(dummy_data_file)

    assert isinstance(dataset, torch.utils.data.TensorDataset)
    assert len(dataset) == 100

    sequences, labels = dataset.tensors
    assert sequences.shape == (100, 10, 47)
    assert labels.shape == (100, 1)
    assert sequences.dtype == torch.float32
    assert labels.dtype == torch.float32


# --- Tests pour `fusion_model.py` ---


class TestFusionModel:
    @pytest.fixture
    def fusion_setup(self):
        """Prépare les composants pour le modèle de fusion."""
        # Transformer factice
        transformer = TemporalTransformer(
            input_dim=47, d_model=32, n_head=2, n_layers=2, sequence_length=10
        )

        # GNN factice
        data = HeteroData()
        data["cheval"].x = torch.randn(10, 16)
        data["jockey"].x = torch.randn(5, 8)
        data["entraineur"].x = torch.randn(3, 8)
        data["cheval", "monte_par", "jockey"].edge_index = torch.stack(
            [torch.randint(0, 10, (20,)), torch.randint(0, 5, (20,))]
        )
        data["jockey", "monte", "cheval"].edge_index = torch.stack(
            [torch.randint(0, 5, (20,)), torch.randint(0, 10, (20,))]
        )
        data["cheval", "entraine_par", "entraineur"].edge_index = torch.stack(
            [torch.randint(0, 10, (20,)), torch.randint(0, 3, (20,))]
        )
        data["entraineur", "entraine", "cheval"].edge_index = torch.stack(
            [torch.randint(0, 3, (20,)), torch.randint(0, 10, (20,))]
        )

        gnn = EntityGraphNN(metadata=data.metadata(), hidden_dim=16, out_dim=16, n_layers=2)

        # Initialisation LazyLinear
        with torch.no_grad():
            gnn(data.x_dict, data.edge_index_dict)

        return transformer, gnn, data

    def test_fusion_forward(self, fusion_setup):
        """Vérifie la passe avant du modèle de fusion."""
        transformer, gnn, graph_data = fusion_setup

        model = FusionTransformerGNN(
            transformer_model=transformer,
            gnn_model=gnn,
            transformer_dim=32,
            gnn_dim=16 * 3,  # 16 * 3 (C+J+E)
            fusion_dim=16,
        )

        # Inputs factices
        batch_size = 4
        temporal_src = torch.randn(batch_size, 10, 47)
        temporal_mask = torch.zeros(batch_size, 10, dtype=torch.bool)

        c_idx = torch.tensor([0, 1, 2, 3])
        j_idx = torch.tensor([0, 1, 0, 1])
        e_idx = torch.tensor([0, 1, 0, 1])

        model.eval()
        with torch.no_grad():
            logits, weights = model(
                temporal_src,
                temporal_mask,
                graph_data.x_dict,
                graph_data.edge_index_dict,
                (c_idx, j_idx, e_idx),
            )

        assert logits.shape == (batch_size, 1)
        assert weights.shape == (batch_size, 2)
        # Vérifie que les poids d'attention somment à 1
        assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size))
