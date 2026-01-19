#!/bin/bash

################################################################################
# SCRAPING HISTORIQUE EN CONTINU
################################################################################
#
# Ce script lance le scraping en boucle continue jusqu'à complétion
# Dès qu'une session de 10 périodes est terminée, une nouvelle démarre
#
# Usage:
#   ./scraper_continu.sh              # Lance en avant-plan
#   ./scraper_continu.sh &            # Lance en arrière-plan
#   nohup ./scraper_continu.sh &      # Lance et persiste après fermeture terminal
#
# Pour arrêter:
#   touch /Users/gicquelsacha/horse3/STOP_SCRAPING
#   ou kill le processus
#
################################################################################

PROJECT_DIR="/Users/gicquelsacha/horse3"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
STOP_FILE="$PROJECT_DIR/STOP_SCRAPING"

# Nombre de périodes par session (10 périodes = 30 jours)
PERIODS=10

# Pause entre sessions (secondes) - pour éviter surcharge API
PAUSE_BETWEEN=30

# Créer dossier logs
mkdir -p "$LOG_DIR"

# Supprimer fichier stop s'il existe
rm -f "$STOP_FILE"

echo "============================================================"
echo "🚀 SCRAPING CONTINU DÉMARRÉ"
echo "============================================================"
echo "   Périodes par session: $PERIODS ($(($PERIODS * 3)) jours)"
echo "   Pause entre sessions: ${PAUSE_BETWEEN}s"
echo "   Pour arrêter: touch $STOP_FILE"
echo "   Logs: $LOG_DIR/continu_*.log"
echo "============================================================"
echo ""

SESSION=1

while true; do
    # Vérifier si on doit s'arrêter
    if [ -f "$STOP_FILE" ]; then
        echo ""
        echo "🛑 Fichier STOP détecté - Arrêt du scraping continu"
        rm -f "$STOP_FILE"
        break
    fi

    # Vérifier le statut actuel
    STATUS=$($PYTHON "$PROJECT_DIR/scraper_historique_auto.py" --status 2>/dev/null | grep "status:" | awk '{print $2}')

    if [ "$STATUS" = "completed" ]; then
        echo ""
        echo "🎉 SCRAPING HISTORIQUE TERMINÉ !"
        echo "   Toutes les données sur 5 ans ont été récupérées."
        break
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/continu_${TIMESTAMP}.log"

    echo ""
    echo "============================================================"
    echo "📦 SESSION $SESSION - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    # Afficher progression avant
    echo "📊 Progression avant session:"
    $PYTHON "$PROJECT_DIR/scraper_historique_auto.py" --status 2>/dev/null | grep -E "(progress_percent|remaining_days|last_scraped)"

    echo ""
    echo "🔄 Lancement de $PERIODS périodes..."

    # Lancer le scraping
    $PYTHON "$PROJECT_DIR/scraper_historique_auto.py" --periods $PERIODS 2>&1 | tee "$LOG_FILE"

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "⚠️  Erreur détectée (code $EXIT_CODE), pause de 60s avant retry..."
        sleep 60
    else
        echo "✅ Session $SESSION terminée avec succès"
    fi

    # Afficher progression après
    echo ""
    echo "📊 Progression après session:"
    $PYTHON "$PROJECT_DIR/scraper_historique_auto.py" --status 2>/dev/null | grep -E "(progress_percent|remaining_days|last_scraped)"

    SESSION=$((SESSION + 1))

    # Petite pause pour ne pas surcharger l'API
    echo ""
    echo "⏳ Pause de ${PAUSE_BETWEEN}s avant prochaine session..."
    sleep $PAUSE_BETWEEN
done

echo ""
echo "============================================================"
echo "🏁 SCRAPING CONTINU TERMINÉ"
echo "   Sessions effectuées: $((SESSION - 1))"
echo "   Date fin: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
