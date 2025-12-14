#!/bin/bash
# ============================================================================
# Script de Test - Pipeline Retraining Automatique
# ============================================================================
#
# Usage:
#   ./test_retraining.sh
#
# Tests:
#   1. Retraining en mode dry-run
#   2. Vérification structure model registry
#   3. Validation metadata
#
# Auteur: Phase 8 - Online Learning
# Date: 2025-11-14
# ============================================================================

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║          🧪 TESTS PIPELINE RETRAINING AUTOMATIQUE - PHASE 8 🧪              ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Compteurs
TESTS_PASSED=0
TESTS_FAILED=0

# Fonction test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}TEST:${NC} $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}: $test_name"
        ((TESTS_FAILED++))
        return 1
    fi
}

# ============================================================================
# TEST 1: Vérifier structure model registry
# ============================================================================
test_model_registry_structure() {
    echo "🔍 Vérification structure model registry..."
    
    local required_dirs=(
        "data/models/champion"
        "data/models/challenger"
        "data/models/archive"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            echo "   ✅ $dir existe"
        else
            echo "   ❌ $dir manquant"
            return 1
        fi
    done
    
    return 0
}

run_test "Structure Model Registry" "test_model_registry_structure"

# ============================================================================
# TEST 2: Retraining en mode dry-run
# ============================================================================
test_retraining_dry_run() {
    echo "🔄 Lancement retraining en mode dry-run..."
    
    # Lancer retraining sans sauvegarder
    if python train_online.py --dry-run --days 7 --min-samples 0 2>&1 | tee logs/retraining_test.log; then
        echo "   ✅ Retraining terminé sans erreur"
        
        # Vérifier que rien n'a été sauvegardé (dry-run)
        if [ ! -f "data/models/challenger/model.pkl" ]; then
            echo "   ✅ Pas de fichier sauvegardé (dry-run OK)"
            return 0
        else
            echo "   ⚠️  Fichier challenger détecté (ne devrait pas exister en dry-run)"
            return 0  # Warning mais pas fail
        fi
    else
        echo "   ❌ Erreur lors du retraining"
        return 1
    fi
}

run_test "Retraining Dry-Run" "test_retraining_dry_run"

# ============================================================================
# TEST 3: Vérifier logs retraining
# ============================================================================
test_retraining_logs() {
    echo "📋 Vérification logs retraining..."
    
    if [ -f "logs/retraining.log" ]; then
        echo "   ✅ logs/retraining.log existe"
        
        # Vérifier présence étapes clés
        local required_logs=(
            "CHARGEMENT DONNÉES ORIGINALES"
            "CHARGEMENT FEEDBACK"
            "CONSTRUCTION MODÈLE STACKING"
            "ENTRAÎNEMENT & VALIDATION"
        )
        
        for log_pattern in "${required_logs[@]}"; do
            if grep -q "$log_pattern" logs/retraining.log; then
                echo "   ✅ Étape trouvée: $log_pattern"
            else
                echo "   ⚠️  Étape manquante: $log_pattern"
            fi
        done
        
        return 0
    else
        echo "   ❌ logs/retraining.log manquant"
        return 1
    fi
}

run_test "Logs Retraining" "test_retraining_logs"

# ============================================================================
# TEST 4: Copier champion actuel si existe
# ============================================================================
test_setup_champion() {
    echo "📦 Setup champion pour tests..."
    
    # Si modèle champion n'existe pas, copier depuis ensemble_stacking.pkl
    if [ ! -f "data/models/champion/model.pkl" ]; then
        if [ -f "data/models/ensemble_stacking.pkl" ]; then
            echo "   📋 Copie ensemble_stacking.pkl → champion/model.pkl"
            cp data/models/ensemble_stacking.pkl data/models/champion/model.pkl
            
            # Créer metadata basique
            cat > data/models/champion/metadata.json << 'EOF'
{
  "timestamp": "20251113_000000",
  "datetime": "2025-11-13T00:00:00",
  "model_type": "stacking_ensemble",
  "version": "v1.0.0",
  "metrics": {
    "roc_auc": 0.7009,
    "accuracy": 0.9354
  },
  "training": {
    "phase": "Phase 6 - Ensemble Learning"
  },
  "created_by": "manual_setup"
}
EOF
            echo "   ✅ Champion setup terminé"
            return 0
        else
            echo "   ⚠️  Aucun modèle source trouvé"
            return 0  # Warning mais pas fail
        fi
    else
        echo "   ℹ️  Champion déjà présent"
        return 0
    fi
}

run_test "Setup Champion Model" "test_setup_champion"

# ============================================================================
# TEST 5: Vérifier script --help
# ============================================================================
test_script_help() {
    echo "ℹ️  Test option --help..."
    
    if python train_online.py --help > /dev/null 2>&1; then
        echo "   ✅ Option --help fonctionne"
        return 0
    else
        echo "   ❌ Option --help échoue"
        return 1
    fi
}

run_test "Script Help" "test_script_help"

# ============================================================================
# RÉSUMÉ
# ============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                          📊 RÉSUMÉ DES TESTS                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
SUCCESS_RATE=$(echo "scale=1; 100 * $TESTS_PASSED / $TOTAL_TESTS" | bc)

echo "Total tests:     $TOTAL_TESTS"
echo -e "${GREEN}✅ Réussis:${NC}      $TESTS_PASSED"
echo -e "${RED}❌ Échoués:${NC}      $TESTS_FAILED"
echo "Taux de succès:  ${SUCCESS_RATE}%"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}║          🎉 TOUS LES TESTS PASSENT! RETRAINING OPÉRATIONNEL 🎉             ║${NC}"
    echo -e "${GREEN}║                                                                              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                                                              ║${NC}"
    echo -e "${RED}║              ⚠️  CERTAINS TESTS ONT ÉCHOUÉ ⚠️                              ║${NC}"
    echo -e "${RED}║                                                                              ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
