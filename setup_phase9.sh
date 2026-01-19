#!/bin/bash

# 🚀 Setup Phase 9 - Deep Learning Environment
# Installe PyTorch, DGL, FLAML et crée structure projet

set -e

echo "🚀 SETUP PHASE 9 - DEEP LEARNING"
echo "=================================="
echo ""

# Activation venv
if [ -d ".venv" ]; then
    echo "📦 Activation environnement virtuel existant..."
    source .venv/bin/activate
else
    echo "📦 Création environnement virtuel..."
    python3 -m venv .venv
    source .venv/bin/activate
fi
echo "   ✅ Environnement activé"
echo ""

# Détection architecture (M1/M2 Mac vs x86)
ARCH=$(uname -m)
echo "🔍 Détection architecture: $ARCH"
echo ""

# Installation PyTorch
echo "🔥 Installation PyTorch..."
if [ "$ARCH" = "arm64" ]; then
    # M1/M2 Mac - PyTorch avec MPS (Metal Performance Shaders)
    echo "   → Installation PyTorch optimisé pour Apple Silicon (MPS)"
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    # x86 - PyTorch avec CUDA support
    echo "   → Installation PyTorch avec support CUDA"
    pip install torch torchvision torchaudio
fi
echo "   ✅ PyTorch installé"
echo ""

# Installation Graph Library (PyG ou DGL)
echo "🕸️  Installation Graph Neural Network Library..."
if [ "$ARCH" = "arm64" ]; then
    # M1/M2 Mac - PyTorch Geometric (meilleur support Apple Silicon)
    echo "   → Installation PyTorch Geometric pour Apple Silicon"
    echo "   ℹ️  Note: DGL non supporté sur M1/M2, utilisation de PyG"
    pip install torch-geometric
    # Extensions optionnelles (peuvent échouer, ce n'est pas bloquant)
    pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.9.0+cpu.html 2>/dev/null || \
    echo "   ℹ️  Extensions PyG optionnelles non disponibles (OK)"
    echo "   ✅ PyTorch Geometric (PyG) installé"
else
    # x86 - Essayer DGL d'abord, sinon PyG
    echo "   → Installation DGL (Deep Graph Library)"
    if pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html 2>/dev/null; then
        echo "   ✅ DGL installé"
    else
        echo "   ⚠️  DGL non disponible, installation PyTorch Geometric (PyG)"
        pip install torch-geometric
        pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.9.0+cu118.html
        echo "   ✅ PyTorch Geometric (PyG) installé"
    fi
fi
echo ""

# Installation AutoML & Optimization
echo "🤖 Installation FLAML + Optuna..."
pip install flaml[automl] optuna
echo "   ✅ FLAML & Optuna installés"
echo ""

# Installation Monitoring
echo "📊 Installation Weights & Biases (optionnel)..."
read -p "Installer wandb pour tracking ? [y/N]: " install_wandb
if [ "$install_wandb" = "y" ]; then
    pip install wandb
    echo "   ✅ wandb installé"
    echo "   🔑 Run 'wandb login' pour authentification"
else
    echo "   ⏭️  wandb non installé (peut être ajouté plus tard)"
fi
echo ""

# Installation dépendances additionnelles
echo "📦 Installation dépendances additionnelles..."
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
pip install networkx  # Pour visualisation graphes
pip install tensorboard  # Monitoring alternatif à wandb
echo "   ✅ Dépendances installées"
echo ""

# Création structure directories Phase 9
echo "📁 Création structure projet Phase 9..."
mkdir -p data/phase9/temporal
mkdir -p data/phase9/graphs
mkdir -p data/phase9/checkpoints
mkdir -p models/phase9
mkdir -p logs/phase9
mkdir -p tests/phase9
mkdir -p notebooks/phase9

echo "   ✅ Directories créés:"
echo "      • data/phase9/temporal     → Séquences temporelles"
echo "      • data/phase9/graphs       → Graphes entités"
echo "      • data/phase9/checkpoints  → Model checkpoints"
echo "      • models/phase9            → Architectures PyTorch"
echo "      • logs/phase9              → TensorBoard logs"
echo "      • tests/phase9             → Tests unitaires"
echo "      • notebooks/phase9         → Notebooks exploration"
echo ""

# Vérification installations
echo "🧪 Vérification installations..."

python << EOF
import sys

# Test PyTorch
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")

    # Test GPU/MPS disponibilité
    if torch.cuda.is_available():
        print(f"      🎮 CUDA available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"      🍎 MPS (Metal) available")
    else:
        print(f"      💻 CPU only")
except ImportError:
    print("   ❌ PyTorch non installé correctement")
    sys.exit(1)

# Test Graph Library (DGL ou PyG)
try:
    import dgl
    print(f"   ✅ DGL {dgl.__version__}")
except ImportError:
    try:
        import torch_geometric
        print(f"   ✅ PyTorch Geometric (PyG) {torch_geometric.__version__}")
        print(f"      ℹ️  Alternative à DGL pour Apple Silicon")
    except ImportError:
        print("   ❌ Aucune librairie de graphes installée")
        sys.exit(1)

# Test FLAML
try:
    import flaml
    print(f"   ✅ FLAML {flaml.__version__}")
except ImportError:
    print("   ❌ FLAML non installé correctement")
    sys.exit(1)

# Test Optuna
try:
    import optuna
    print(f"   ✅ Optuna {optuna.__version__}")
except ImportError:
    print("   ❌ Optuna non installé correctement")
    sys.exit(1)

print("")
print("🎉 Toutes les dépendances sont installées correctement !")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Erreur lors de la vérification des installations"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ SETUP PHASE 9 TERMINÉ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎯 Prochaines étapes:"
echo ""
echo "1️⃣  Préparer données temporelles:"
echo "   python prepare_temporal_data.py"
echo ""
echo "2️⃣  Construire graphe entités:"
echo "   python build_graph_data.py"
echo ""
echo "3️⃣  Entraîner Transformer:"
echo "   python train_transformer.py"
echo ""
echo "4️⃣  Consulter roadmap:"
echo "   cat ROADMAP_PHASE9_DEEP_LEARNING.md"
echo ""
echo "📖 Documentation complète dans ROADMAP_PHASE9_DEEP_LEARNING.md"
echo ""
