# Phase 9 : Deep Learning & Modèles Avancés

Cette phase a exploré l'utilisation de modèles de Deep Learning pour la prédiction des courses hippiques.

## 1. Architecture

Nous avons développé une architecture hybride composée de trois modules :

1.  **Temporal Transformer (Séquentiel) :**
    *   Traite l'historique des 10 dernières courses d'un cheval.
    *   Utilise un mécanisme d'attention pour pondérer l'importance des courses passées.
    *   *Input :* Séquence de vecteurs de features (47 features par course).
    *   *Output :* Embedding temporel.

2.  **Entity Graph Neural Network (Relationnel) :**
    *   Modélise les relations entre Chevaux, Jockeys et Entraîneurs.
    *   Utilise un GNN hétérogène (HeteroSAGE) pour apprendre des embeddings de nœuds.
    *   *Input :* Graphe hétérogène (Nœuds + Arêtes).
    *   *Output :* Embedding relationnel pour chaque triplet (Cheval, Jockey, Entraîneur).

3.  **Fusion Model (Hybride) :**
    *   Combine les embeddings du Transformer et du GNN.
    *   Utilise un mécanisme d'attention (FusionAttention) pour pondérer dynamiquement l'importance de l'historique vs les relations.
    *   *Output :* Probabilité de victoire (Top 3).

## 2. Résultats

| Modèle | AUC (Validation) | Observations |
| :--- | :--- | :--- |
| **Transformer (Seul)** | ~0.58 | Performance limitée. Suggère que l'historique seul est bruité ou insuffisant. |
| **GNN (Seul)** | **0.7614** | **Meilleure performance.** Les relations (Jockey/Entraîneur) sont très prédictives. |
| **Fusion (Transformer + GNN)** | 0.7322 | Bonne performance, mais inférieure au GNN seul. Le Transformer semble introduire du bruit. |

## 3. Analyse

*   **Dominance du Relationnel :** Le succès du GNN confirme que la qualité du couple Jockey/Entraîneur et leur historique global est plus déterminante que la séquence temporelle stricte des performances passées du cheval.
*   **Difficulté du Séquentiel :** Les courses hippiques sont très variables. Une séquence de 10 courses peut contenir beaucoup de bruit (disqualifications, mauvais parcours) qui trompe le Transformer.
*   **Piste d'amélioration :**
    *   Simplifier le Transformer (moins de couches, moins de têtes).
    *   Pré-entraîner le Transformer sur une tâche plus simple (ex: prédire la vitesse).
    *   Utiliser une fusion plus simple (concaténation simple) au lieu de l'attention.

## 4. Fichiers Clés

*   `models/phase9/transformer_temporal.py` : Code du Transformer.
*   `models/phase9/graph_nn.py` : Code du GNN.
*   `models/phase9/fusion_model.py` : Code de la Fusion.
*   `train_transformer.py`, `train_gnn.py`, `train_fusion.py` : Scripts d'entraînement.
*   `prepare_temporal_data.py` : Préparation des données.

## 5. Utilisation

Pour lancer l'entraînement du modèle de fusion (nécessite les poids pré-entraînés) :

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python3 train_fusion.py
```
