#!/bin/bash
# ============================================================================
# 🏇 Script de Gestion API - Horse Prediction
# ============================================================================
# Gestion simplifiée de l'API (start, stop, status, logs)
# Usage: ./manage_api.sh [start|stop|restart|status|logs|test]
# ============================================================================

set -e  # Exit on error

# Configuration
API_PORT=8000
API_HOST="0.0.0.0"
LOG_FILE="logs/api.log"
PID_FILE="logs/api.pid"

# Couleurs
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonctions utilitaires
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${YELLOW}ℹ️  $1${NC}"; }

# Vérifier si l'API est en cours d'exécution
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0  # Running
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1  # Not running
}

# Démarrer l'API
start_api() {
    echo "=============================================================================="
    echo "🚀 DÉMARRAGE API PRÉDICTION"
    echo "=============================================================================="
    
    if is_running; then
        print_error "L'API est déjà en cours d'exécution (PID: $(cat $PID_FILE))"
        exit 1
    fi
    
    # Créer dossier logs si nécessaire
    mkdir -p logs
    
    # Activer virtualenv
    if [ ! -d ".venv" ]; then
        print_error "Virtualenv .venv introuvable. Exécutez: python3 -m venv .venv"
        exit 1
    fi
    
    source .venv/bin/activate
    
    # Vérifier dépendances
    print_info "Vérification des dépendances..."
    if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
        print_error "Dépendances manquantes. Installez-les avec: pip install -r requirements-prod.txt"
        exit 1
    fi
    
    # Vérifier modèle
    if [ ! -f "data/models/ensemble_stacking.pkl" ]; then
        print_error "Modèle introuvable: data/models/ensemble_stacking.pkl"
        exit 1
    fi
    
    # Démarrer API en arrière-plan
    print_info "Démarrage de l'API sur http://$API_HOST:$API_PORT..."
    nohup python api_prediction.py --host "$API_HOST" --port "$API_PORT" > "$LOG_FILE" 2>&1 &
    
    # Sauvegarder PID
    echo $! > "$PID_FILE"
    
    # Attendre démarrage
    sleep 3
    
    # Vérifier healthcheck
    if curl -s http://localhost:$API_PORT/health > /dev/null; then
        print_success "API démarrée avec succès !"
        print_info "PID: $(cat $PID_FILE)"
        print_info "URL: http://localhost:$API_PORT"
        print_info "Docs: http://localhost:$API_PORT/docs"
        print_info "Logs: tail -f $LOG_FILE"
    else
        print_error "L'API a démarré mais ne répond pas au healthcheck"
        print_info "Vérifiez les logs: cat $LOG_FILE"
        exit 1
    fi
    
    echo "=============================================================================="
}

# Arrêter l'API
stop_api() {
    echo "=============================================================================="
    echo "🛑 ARRÊT API PRÉDICTION"
    echo "=============================================================================="
    
    if ! is_running; then
        print_error "L'API n'est pas en cours d'exécution"
        exit 1
    fi
    
    PID=$(cat "$PID_FILE")
    print_info "Arrêt de l'API (PID: $PID)..."
    
    # Graceful shutdown
    kill "$PID"
    
    # Attendre max 10s
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            rm -f "$PID_FILE"
            print_success "API arrêtée proprement"
            echo "=============================================================================="
            return 0
        fi
        sleep 1
    done
    
    # Force kill si toujours actif
    print_info "Force kill..."
    kill -9 "$PID"
    rm -f "$PID_FILE"
    print_success "API arrêtée (force)"
    echo "=============================================================================="
}

# Status de l'API
status_api() {
    echo "=============================================================================="
    echo "📊 STATUS API PRÉDICTION"
    echo "=============================================================================="
    
    if is_running; then
        PID=$(cat "$PID_FILE")
        print_success "L'API est en cours d'exécution"
        print_info "PID: $PID"
        print_info "URL: http://localhost:$API_PORT"
        
        # Requête healthcheck
        if command -v curl &> /dev/null; then
            echo ""
            print_info "Healthcheck:"
            curl -s http://localhost:$API_PORT/health | python3 -m json.tool 2>/dev/null || echo "  N/A"
        fi
        
        # Stats processus
        echo ""
        print_info "Ressources:"
        ps -p "$PID" -o pid,ppid,%cpu,%mem,vsz,rss,etime,cmd
    else
        print_error "L'API n'est pas en cours d'exécution"
    fi
    
    echo "=============================================================================="
}

# Logs de l'API
logs_api() {
    if [ ! -f "$LOG_FILE" ]; then
        print_error "Fichier de logs introuvable: $LOG_FILE"
        exit 1
    fi
    
    echo "=============================================================================="
    echo "📋 LOGS API (appuyez sur Ctrl+C pour quitter)"
    echo "=============================================================================="
    tail -f "$LOG_FILE"
}

# Tester l'API
test_api() {
    echo "=============================================================================="
    echo "🧪 TEST API PRÉDICTION"
    echo "=============================================================================="
    
    if ! is_running; then
        print_error "L'API n'est pas en cours d'exécution. Démarrez-la avec: $0 start"
        exit 1
    fi
    
    source .venv/bin/activate
    
    if [ ! -f "test_api.py" ]; then
        print_error "Script de test introuvable: test_api.py"
        exit 1
    fi
    
    python test_api.py --url "http://localhost:$API_PORT" --verbose
}

# Afficher l'aide
show_help() {
    cat << EOF
============================================================================
🏇 Gestion API Prédiction Courses Hippiques
============================================================================

Usage: $0 [COMMANDE]

Commandes disponibles:

  start       Démarrer l'API en arrière-plan
  stop        Arrêter l'API
  restart     Redémarrer l'API (stop + start)
  status      Afficher le status de l'API
  logs        Afficher les logs en temps réel
  test        Exécuter les tests automatisés
  help        Afficher cette aide

Exemples:

  # Démarrer l'API
  $0 start

  # Vérifier le status
  $0 status

  # Voir les logs
  $0 logs

  # Tester l'API
  $0 test

  # Redémarrer l'API
  $0 restart

Documentation: DEPLOIEMENT_PRODUCTION.md
============================================================================
EOF
}

# Main
case "${1:-}" in
    start)
        start_api
        ;;
    stop)
        stop_api
        ;;
    restart)
        if is_running; then
            stop_api
            sleep 2
        fi
        start_api
        ;;
    status)
        status_api
        ;;
    logs)
        logs_api
        ;;
    test)
        test_api
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Commande invalide: ${1:-}"
        echo ""
        show_help
        exit 1
        ;;
esac
