#!/bin/bash
# 🏇 HORSE3 USER APP - SCRIPT DE LANCEMENT RAPIDE

echo "=================================================================================="
echo "🏇 HORSE3 USER APP - LANCEMENT RAPIDE"
echo "=================================================================================="
echo "🏆 Modèle Champion XGBoost v1.0 | ROI +22.71% | Sharpe 3.599"
echo "=================================================================================="
echo ""

# Vérification des prérequis
echo "🔍 Vérification des prérequis..."

if [ ! -f "user_app_api.py" ]; then
    echo "❌ Fichier user_app_api.py manquant"
    exit 1
fi

if [ ! -d "data/models/champion" ]; then
    echo "❌ Modèle champion manquant dans data/models/champion/"
    exit 1
fi

if [ ! -f "data/picks_2025-12-08.json" ]; then
    echo "⚠️  Fichier picks_2025-12-08.json manquant - génération..."
    python cli.py pick --date 2025-12-08
fi

echo "✅ Prérequis OK"
echo ""

# Choix du mode de lancement
echo "📋 MODES DE LANCEMENT DISPONIBLES:"
echo "1. 🚀 API seule (recommandé pour production)"
echo "2. 🎬 API + Démonstration rapide"
echo "3. 📖 Voir le guide utilisateur"
echo "4. 🔧 Tests de validation"
echo ""

read -p "Choisissez votre mode (1-4): " choice

case $choice in
    1)
        echo "🚀 Démarrage de l'API utilisateur..."
        echo "📖 Documentation: http://localhost:8001/docs"
        echo "⚡ Health check: http://localhost:8001/health"
        echo ""
        python user_app_api.py
        ;;
    2)
        echo "🎬 Démarrage API + Démonstration..."
        python user_app_api.py &
        API_PID=$!
        sleep 3
        echo ""
        echo "▶️  Lancement de la démonstration..."
        python demo_user_app.py
        echo ""
        echo "🛑 Arrêt de l'API..."
        kill $API_PID
        ;;
    3)
        echo "📖 Ouverture du guide utilisateur..."
        if command -v code &> /dev/null; then
            code USER_APP_GUIDE.md
        elif command -v open &> /dev/null; then
            open USER_APP_GUIDE.md
        else
            echo "📄 Voir le fichier: USER_APP_GUIDE.md"
        fi
        ;;
    4)
        echo "🔧 Lancement des tests de validation..."
        echo ""
        echo "🏆 Test du modèle champion:"
        python validate_champion_model.py
        echo ""
        echo "🚀 Test de l'API (démarrage temporaire):"
        python user_app_api.py &
        API_PID=$!
        sleep 3
        curl -s http://localhost:8001/health | jq .status || echo "API opérationnelle"
        kill $API_PID
        echo "✅ Tests terminés"
        ;;
    *)
        echo "❌ Choix invalide. Relancez le script."
        exit 1
        ;;
esac

echo ""
echo "=================================================================================="
echo "🎉 HORSE3 USER APP"
echo "📚 Guide: USER_APP_GUIDE.md | 🎬 Démo: python demo_user_app.py"
echo "🔧 API: python user_app_api.py | 📊 Stats: ETAPE_C_COMPLETE.md"
echo "=================================================================================="