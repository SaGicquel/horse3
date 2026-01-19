#!/bin/bash

# 🚀 Quick Start - Phase 9 Deep Learning
# Ce script automatise le pipeline de Deep Learning (Transformer, GNN, Fusion)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧠 QUICK START - PHASE 9 DEEP LEARNING"
echo "======================================"
echo ""

# Configuration MPS pour Mac Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Activation venv
echo "📦 Activation environnement virtuel..."
source .venv/bin/activate
echo "   ✅ Environnement activé"
echo ""

# Vérification dépendances
echo "🔍 Vérification dépendances..."
python3 -c "import torch, torch_geometric" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ PyTorch & PyG détectés"
else
    echo "   ⚠️  Dépendances manquantes. Installation..."
    pip install torch torchvision torchaudio torch_geometric
    echo "   ✅ Installation terminée"
fi
echo ""

# Menu principal
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 QUE VEUX-TU FAIRE ?"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1) 🔄 Pipeline Complet (Data -> Train -> Eval)"
echo "      → Prépare les données"
echo "      → Entraîne Transformer, GNN, puis Fusion"
echo ""
echo "2) 💾 Préparer les Données uniquement"
echo "      → Lance prepare_temporal_data.py"
echo ""
echo "3) 🤖 Entraîner Transformer (Séquentiel)"
echo "      → Lance train_transformer.py"
echo ""
echo "4) 🕸️  Entraîner GNN (Relationnel)"
echo "      → Lance train_gnn.py"
echo ""
echo "5) 🧬 Entraîner Fusion (Hybride)"
echo "      → Lance train_fusion.py"
echo ""
echo "6) 🧪 Lancer les Tests Unitaires"
echo "      → pytest tests/phase9/"
echo ""
echo "0) ❌ Quitter"
echo ""
read -p "Choix [0-6]: " choice

case $choice in
    1)
        echo ""
        echo "🔄 LANCEMENT PIPELINE COMPLET"
        echo "============================="

        echo "1. Préparation des données..."
        python3 prepare_temporal_data.py

        echo "2. Entraînement Transformer..."
        python3 train_transformer.py

        echo "3. Entraînement GNN..."
        python3 train_gnn.py

        echo "4. Entraînement Fusion..."
        python3 train_fusion.py

        echo ""
        echo "✅ Pipeline terminé avec succès !"
        ;;

    2)
        echo ""
        echo "💾 PRÉPARATION DES DONNÉES"
        python3 prepare_temporal_data.py
        ;;

    3)
        echo ""
        echo "🤖 ENTRAÎNEMENT TRANSFORMER"
        python3 train_transformer.py
        ;;

    4)
        echo ""
        echo "🕸️  ENTRAÎNEMENT GNN"
        python3 train_gnn.py
        ;;

    5)
        echo ""
        echo "🧬 ENTRAÎNEMENT FUSION"
        python3 train_fusion.py
        ;;

    6)
        echo ""
        echo "🧪 TESTS UNITAIRES"
        pytest tests/phase9/
        ;;

    0)
        echo "Au revoir !"
        exit 0
        ;;

    *)
        echo "Choix invalide."
        exit 1
        ;;
esac
