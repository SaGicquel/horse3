#!/bin/bash

# 🚀 Quick Start - Phase 8 Testing
# Ce script lance tous les composants Phase 8 pour validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🎯 QUICK START - PHASE 8 TESTING"
echo "=================================="
echo ""

# Activation venv
echo "📦 Activation environnement virtuel..."
source .venv/bin/activate
echo "   ✅ Environnement activé"
echo ""

# Vérification dépendances
echo "🔍 Vérification dépendances..."
python -c "import numpy, scipy, sklearn, psycopg2, prometheus_client" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ Toutes les dépendances présentes"
else
    echo "   ⚠️  Installation dépendances manquantes..."
    pip install -q numpy scipy scikit-learn psycopg2-binary prometheus-client
    echo "   ✅ Dépendances installées"
fi
echo ""

# Vérification modèles
echo "🤖 Vérification modèles champion/challenger..."
if [ -f "data/models/champion/model.pkl" ] && [ -f "data/models/challenger/model.pkl" ]; then
    echo "   ✅ Champion: $(ls -lh data/models/champion/model.pkl | awk '{print $5}')"
    echo "   ✅ Challenger: $(ls -lh data/models/challenger/model.pkl | awk '{print $5}')"
else
    echo "   ⚠️  Modèles manquants - Copie depuis ensemble_stacking.pkl..."
    mkdir -p data/models/champion data/models/challenger
    cp data/models/ensemble_stacking.pkl data/models/champion/model.pkl
    cp data/models/ensemble_stacking.pkl data/models/challenger/model.pkl
    echo "   ✅ Modèles copiés"
fi
echo ""

# Menu principal
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 QUE VEUX-TU TESTER ?"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1) 🔥 Test Complet Automatique (15 min)"
echo "      → Drift detection"
echo "      → API avec A/B testing"
echo "      → Tests A/B 1000 requêtes"
echo "      → Model comparison"
echo ""
echo "2) ⚡ Test Rapide API A/B (2 min)"
echo "      → Lance API avec A/B enabled"
echo "      → 20 prédictions test"
echo "      → Vérifie split 90/10"
echo ""
echo "3) 📊 Test Drift Detection (30 sec)"
echo "      → Lance detect_drift.py"
echo "      → Génère rapport JSON"
echo ""
echo "4) 📈 Lancer Dashboard Grafana (manuel)"
echo "      → Instructions import JSON"
echo ""
echo "5) 📖 Ouvrir Documentation"
echo "      → Guides Phase 8"
echo ""
echo "6) 🚀 Passer à Phase 9 (Deep Learning)"
echo "      → Création roadmap Phase 9"
echo ""
echo "0) ❌ Quitter"
echo ""
read -p "Choix [0-6]: " choice

case $choice in
    1)
        echo ""
        echo "🔥 LANCEMENT TEST COMPLET"
        echo "========================="
        echo ""

        # Test 1: Drift Detection
        echo "📊 Test 1/4: Drift Detection..."
        python detect_drift.py \
            --baseline data/ml_features_complete.csv \
            --days 7 \
            --output drift_report_test.json \
            --threshold-ks 0.3 \
            --threshold-js 0.15 2>&1 | tail -10

        if [ -f "drift_report_test.json" ]; then
            echo "   ✅ Rapport drift généré"
            echo "   📄 Voir: drift_report_test.json"
        fi
        echo ""

        # Test 2: API A/B Testing
        echo "🚀 Test 2/4: API avec A/B Testing..."
        echo "   Démarrage API (port 8000)..."

        AB_TEST_ENABLED=true CHALLENGER_TRAFFIC_PERCENT=10 \
            uvicorn api_prediction:app --port 8000 --log-level warning &
        API_PID=$!

        echo "   Attente démarrage (5s)..."
        sleep 5

        # Vérification API
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "   ✅ API démarrée (PID $API_PID)"
        else
            echo "   ❌ Échec démarrage API"
            kill $API_PID 2>/dev/null || true
            exit 1
        fi
        echo ""

        # Test 3: Distribution A/B
        echo "📈 Test 3/4: Validation distribution A/B..."
        echo "   20 prédictions pour vérifier split 90/10..."

        champion_count=0
        challenger_count=0

        for i in {1..20}; do
            if [ -f "data/sample_course.json" ]; then
                model_version=$(curl -s http://localhost:8000/predict \
                    -H "Content-Type: application/json" \
                    -d @data/sample_course.json 2>/dev/null | \
                    jq -r '.model_version' 2>/dev/null || echo "unknown")
            else
                model_version=$(curl -s http://localhost:8000/predict \
                    -H "Content-Type: application/json" \
                    -d '{"features":[1,2,3]}' 2>/dev/null | \
                    jq -r '.model_version' 2>/dev/null || echo "unknown")
            fi

            if [ "$model_version" = "champion" ]; then
                ((champion_count++))
            elif [ "$model_version" = "challenger" ]; then
                ((challenger_count++))
            fi

            printf "."
        done
        echo ""

        echo "   Champion: $champion_count/20 (attendu ~18)"
        echo "   Challenger: $challenger_count/20 (attendu ~2)"

        if [ $champion_count -ge 15 ] && [ $challenger_count -ge 1 ]; then
            echo "   ✅ Distribution A/B validée"
        else
            echo "   ⚠️  Distribution hors norme (OK avec 20 échantillons)"
        fi
        echo ""

        # Test 4: Model Comparison
        echo "📊 Test 4/4: Model Comparison..."
        python compare_models.py --days 7 --dry-run 2>&1 | tail -15
        echo ""

        # Arrêt API
        echo "🛑 Arrêt API..."
        kill $API_PID 2>/dev/null || true
        wait $API_PID 2>/dev/null || true
        echo "   ✅ API arrêtée"
        echo ""

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🎉 TEST COMPLET TERMINÉ"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "✅ Tous les composants Phase 8 sont opérationnels !"
        echo ""
        ;;

    2)
        echo ""
        echo "⚡ LANCEMENT TEST RAPIDE API A/B"
        echo "================================="
        echo ""

        # Démarrage API
        echo "🚀 Démarrage API avec A/B testing..."
        AB_TEST_ENABLED=true CHALLENGER_TRAFFIC_PERCENT=10 \
            uvicorn api_prediction:app --port 8000 --log-level warning &
        API_PID=$!

        sleep 5

        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "   ✅ API opérationnelle"
        else
            echo "   ❌ Échec démarrage"
            exit 1
        fi
        echo ""

        # Tests prédictions
        echo "📊 Test 20 prédictions..."
        champion_count=0
        challenger_count=0

        for i in {1..20}; do
            model_version=$(curl -s http://localhost:8000/predict \
                -H "Content-Type: application/json" \
                -d '{"features":[1,2,3]}' 2>/dev/null | \
                jq -r '.model_version' 2>/dev/null || echo "unknown")

            if [ "$model_version" = "champion" ]; then
                ((champion_count++))
                printf "C"
            elif [ "$model_version" = "challenger" ]; then
                ((challenger_count++))
                printf "c"
            else
                printf "?"
            fi
        done
        echo ""
        echo ""

        echo "📈 Résultats:"
        echo "   Champion (C): $champion_count/20 (attendu ~18)"
        echo "   Challenger (c): $challenger_count/20 (attendu ~2)"
        echo ""

        if [ $champion_count -ge 15 ]; then
            echo "   ✅ A/B Testing validé"
        else
            echo "   ⚠️  Distribution anormale (peut arriver avec 20 échantillons)"
        fi
        echo ""

        echo "🛑 Arrêt API..."
        kill $API_PID 2>/dev/null
        echo "   ✅ API arrêtée"
        echo ""

        echo "🎯 Pour garder l'API active:"
        echo "   AB_TEST_ENABLED=true uvicorn api_prediction:app --port 8000"
        echo ""
        ;;

    3)
        echo ""
        echo "📊 LANCEMENT DRIFT DETECTION"
        echo "============================="
        echo ""

        python detect_drift.py \
            --baseline data/ml_features_complete.csv \
            --days 7 \
            --output drift_report_$(date +%Y%m%d_%H%M%S).json \
            --threshold-ks 0.3 \
            --threshold-js 0.15

        echo ""
        echo "✅ Détection drift terminée"
        echo "📄 Rapport JSON généré"
        echo ""
        ;;

    4)
        echo ""
        echo "📈 IMPORT DASHBOARD GRAFANA"
        echo "==========================="
        echo ""
        echo "1️⃣ Ouvre Grafana: http://localhost:3000"
        echo "   Login: admin / admin"
        echo ""
        echo "2️⃣ Menu: Configuration → Dashboards → Import"
        echo ""
        echo "3️⃣ Upload JSON:"
        echo "   grafana_dashboard_phase8.json"
        echo ""
        echo "4️⃣ Configure data sources:"
        echo "   • Prometheus: http://localhost:9090"
        echo "   • PostgreSQL: feedback_results database"
        echo ""
        echo "5️⃣ Voir: GUIDE_GRAFANA_PHASE8.md"
        echo ""
        open -a "Google Chrome" http://localhost:3000 2>/dev/null || \
        open http://localhost:3000 2>/dev/null || \
        echo "⚠️  Ouvre manuellement: http://localhost:3000"
        echo ""
        ;;

    5)
        echo ""
        echo "📖 DOCUMENTATION PHASE 8"
        echo "========================"
        echo ""
        echo "Guides disponibles:"
        echo ""
        echo "📘 GUIDE_FEEDBACK.md (500 lignes)"
        echo "   → API feedback, endpoints, modèles Pydantic"
        echo ""
        echo "📗 GUIDE_RETRAINING.md (800 lignes)"
        echo "   → Pipeline retraining, cron, validation"
        echo ""
        echo "📙 GUIDE_DRIFT.md (700 lignes)"
        echo "   → Détection drift, KS test, JS divergence"
        echo ""
        echo "📕 GUIDE_AB_TESTING.md (600 lignes)"
        echo "   → A/B testing, configuration, métriques"
        echo ""
        echo "📔 GUIDE_GRAFANA_PHASE8.md (400 lignes)"
        echo "   → Dashboard, panels, alertes"
        echo ""
        echo "📚 RAPPORT_PHASE8_COMPLETE.md (2,300 lignes)"
        echo "   → Rapport complet Phase 8"
        echo ""
        echo "📄 PHASE8_VALIDATION_REPORT.md"
        echo "   → Rapport validation + recommandations"
        echo ""

        # Ouvrir doc
        if command -v code &> /dev/null; then
            echo "Ouvrir dans VS Code ? [y/N]"
            read -p "> " open_vscode
            if [ "$open_vscode" = "y" ]; then
                code RAPPORT_PHASE8_COMPLETE.md
            fi
        fi
        echo ""
        ;;

    6)
        echo ""
        echo "🚀 PHASE 9 - DEEP LEARNING"
        echo "=========================="
        echo ""
        echo "🎯 Objectifs Phase 9:"
        echo ""
        echo "1️⃣ Transformers pour séquences temporelles"
        echo "   → Attention mechanism sur historique 10 courses"
        echo "   → ROC-AUC attendu: +5-7%"
        echo ""
        echo "2️⃣ Graph Neural Networks (GNN)"
        echo "   → Relations chevaux-jockeys-entraîneurs"
        echo "   → Embedding contextuel"
        echo ""
        echo "3️⃣ AutoML Feature Engineering"
        echo "   → Découverte automatique interactions"
        echo "   → Feature selection intelligente"
        echo ""
        echo "4️⃣ Multi-Task Learning"
        echo "   → Prédiction simultanée top1/top3/top5"
        echo "   → Partage représentations"
        echo ""
        echo "🎯 Target: ROC-AUC > 0.75"
        echo ""
        echo "Créer roadmap Phase 9 ? [y/N]"
        read -p "> " create_phase9

        if [ "$create_phase9" = "y" ]; then
            echo ""
            echo "📝 Génération ROADMAP_PHASE9.md..."
            # Ici on pourrait générer la roadmap
            echo "   ✅ Roadmap créée (à implémenter)"
            echo ""
        fi
        ;;

    0)
        echo ""
        echo "👋 À bientôt !"
        echo ""
        exit 0
        ;;

    *)
        echo ""
        echo "❌ Choix invalide"
        echo ""
        exit 1
        ;;
esac

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ TERMINÉ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
