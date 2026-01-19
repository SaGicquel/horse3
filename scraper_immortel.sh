#!/bin/bash

################################################################################
# SCRAPING HISTORIQUE IMMORTEL V2 - ANTI-BLOCAGE
################################################################################
#
# Version améliorée avec:
# - Détection de blocage (timeout 30 min par session)
# - Kill automatique des processus bloqués
# - Relance immédiate après blocage
# - Logs améliorés
#
################################################################################

PROJECT_DIR="/Users/gicquelsacha/horse3"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOCK_FILE="$PROJECT_DIR/.scraper_running.lock"
STOP_FILE="$PROJECT_DIR/STOP_SCRAPING"

# Configuration TURBO
PERIODS=20
MAX_SESSION_MINUTES=45  # Timeout par session (kill si dépassé)

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/immortel_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Fonction pour tuer les processus bloqués
kill_stuck_processes() {
    log "🔪 Nettoyage des processus bloqués..."
    pkill -f "scraper_historique_auto.py" 2>/dev/null
    sleep 2
    # Force kill si toujours là
    pkill -9 -f "scraper_historique_auto.py" 2>/dev/null
    rm -f "$LOCK_FILE"
}

# Vérifier si déjà en cours (avec timeout)
if [ -f "$LOCK_FILE" ]; then
    PID=$(cat "$LOCK_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        # Vérifier depuis combien de temps
        LOCK_AGE=$(( $(date +%s) - $(stat -f %m "$LOCK_FILE") ))
        LOCK_AGE_MIN=$((LOCK_AGE / 60))

        if [ $LOCK_AGE_MIN -gt $MAX_SESSION_MINUTES ]; then
            log "⚠️ Processus $PID bloqué depuis ${LOCK_AGE_MIN} min (> ${MAX_SESSION_MINUTES} min)"
            kill_stuck_processes
        else
            log "⏳ Scraper en cours (PID: $PID, ${LOCK_AGE_MIN} min), abandon"
            exit 0
        fi
    else
        log "🔄 Lock file orphelin trouvé, nettoyage..."
        rm -f "$LOCK_FILE"
    fi
fi

# Créer le lock
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

log "============================================================"
log "🚀 SCRAPER IMMORTEL V2 - MODE TURBO ANTI-BLOCAGE"
log "============================================================"
log "   Périodes par session: $PERIODS"
log "   Timeout session: ${MAX_SESSION_MINUTES} min"

# Vérifier si on doit s'arrêter
if [ -f "$STOP_FILE" ]; then
    log "🛑 Fichier STOP détecté, arrêt"
    rm -f "$STOP_FILE"
    exit 0
fi

# Vérifier le statut
STATUS=$($PYTHON "$PROJECT_DIR/scraper_historique_auto.py" --status 2>/dev/null | grep "status:" | awk '{print $2}')
PROGRESS=$($PYTHON "$PROJECT_DIR/scraper_historique_auto.py" --status 2>/dev/null | grep "progress_percent:" | awk '{print $2}')

log "📊 Statut: $STATUS | Progression: $PROGRESS%"

if [ "$STATUS" = "completed" ]; then
    log "🎉 SCRAPING TERMINÉ ! (100%)"
    exit 0
fi

# Lancer le scraping avec timeout
log "🔄 Lancement de $PERIODS périodes (timeout: ${MAX_SESSION_MINUTES}min)..."

cd "$PROJECT_DIR"

# Timeout manuel pour macOS (pas de gtimeout par défaut)
TIMEOUT_SEC=$((MAX_SESSION_MINUTES * 60))

$PYTHON "$PROJECT_DIR/scraper_historique_auto.py" --periods $PERIODS >> "$LOG_FILE" 2>&1 &
SCRAPER_PID=$!

ELAPSED=0
while kill -0 $SCRAPER_PID 2>/dev/null; do
    sleep 30
    ELAPSED=$((ELAPSED + 30))

    # Log de progression toutes les 5 minutes
    if [ $((ELAPSED % 300)) -eq 0 ]; then
        MINS=$((ELAPSED / 60))
        log "⏳ En cours depuis ${MINS} min..."
    fi

    if [ $ELAPSED -gt $TIMEOUT_SEC ]; then
        log "⏰ Timeout atteint (${MAX_SESSION_MINUTES}min), kill du processus..."
        kill -9 $SCRAPER_PID 2>/dev/null
        EXIT_CODE=124
        break
    fi
done

if [ -z "$EXIT_CODE" ]; then
    wait $SCRAPER_PID
    EXIT_CODE=$?
fi

# Afficher la nouvelle progression
NEW_PROGRESS=$($PYTHON "$PROJECT_DIR/scraper_historique_auto.py" --status 2>/dev/null | grep "progress_percent:" | awk '{print $2}')

if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 137 ]; then
    log "⚠️ Session interrompue par timeout"
else
    log "✅ Session terminée (code: $EXIT_CODE)"
fi

log "📊 Nouvelle progression: $NEW_PROGRESS%"
log "============================================================"
