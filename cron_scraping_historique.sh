#!/bin/bash

################################################################################
# SCRIPT CRON - SCRAPING HISTORIQUE AUTOMATIQUE
################################################################################
#
# Description : Script d'automatisation pour construire progressivement
#               une base historique sur 5 ans. Exécute 2 périodes par jour
#               avec vérifications automatiques.
#
# Stratégie :
#   - 2 exécutions par jour (matin et soir)
#   - 3 jours de données par période
#   - ~6 jours de données par jour
#   - ~5 ans en environ 300 jours (10 mois)
#
# Installation CRON :
#   crontab -e
#   # Scraping historique matin (4h00)
#   0 4 * * * /Users/gicquelsacha/horse3/cron_scraping_historique.sh >> /Users/gicquelsacha/horse3/logs/cron_historique.log 2>&1
#
#   # Scraping historique soir (22h00)
#   0 22 * * * /Users/gicquelsacha/horse3/cron_scraping_historique.sh >> /Users/gicquelsacha/horse3/logs/cron_historique.log 2>&1
#
################################################################################

set -e  # Arrêter en cas d'erreur critique

# Configuration
PROJECT_DIR="/Users/gicquelsacha/horse3"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Nombre de périodes à exécuter par session (10 périodes = 30 jours de données)
# Objectif: finir en 1 mois → besoin de ~60 jours/jour avec 4 sessions
PERIODS_PER_RUN=10

# Créer le dossier logs
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/historique_${TIMESTAMP}.log"

# Fonction de log
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# ============================================================================
# DÉBUT
# ============================================================================

echo "======================================================================" | tee -a "$LOG_FILE"
echo "📚 SCRAPING HISTORIQUE AUTOMATIQUE" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
log "Date       : $(date '+%Y-%m-%d %H:%M:%S')"
log "Périodes   : $PERIODS_PER_RUN"
log "Log        : $LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Vérifications
if [ ! -f "$PYTHON" ]; then
    log "❌ Python virtualenv introuvable"
    exit 1
fi

# Vérifier PostgreSQL
if ! pg_isready -h localhost -p 54624 >> "$LOG_FILE" 2>&1; then
    log "⚠️ PostgreSQL non accessible, tentative de redémarrage..."
    brew services restart postgresql@14 >> "$LOG_FILE" 2>&1 || true
    sleep 5
    if ! pg_isready -h localhost -p 54624 >> "$LOG_FILE" 2>&1; then
        log "❌ PostgreSQL inaccessible"
        exit 1
    fi
fi
log "✅ PostgreSQL OK"

# Changer de répertoire
cd "$PROJECT_DIR" || exit 1

# ============================================================================
# AFFICHER LE STATUT ACTUEL
# ============================================================================

echo "" | tee -a "$LOG_FILE"
echo "📊 STATUT ACTUEL:" | tee -a "$LOG_FILE"
"$PYTHON" scraper_historique_auto.py --status 2>&1 | tee -a "$LOG_FILE"

# ============================================================================
# EXÉCUTER LE SCRAPING
# ============================================================================

echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "🚀 EXÉCUTION DE $PERIODS_PER_RUN PÉRIODE(S)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

"$PYTHON" scraper_historique_auto.py --periods $PERIODS_PER_RUN >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"

# ============================================================================
# RAPPORT FINAL
# ============================================================================

echo "======================================================================" | tee -a "$LOG_FILE"
echo "📋 RAPPORT" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
log "Code sortie : $EXIT_CODE"
log "Log complet : $LOG_FILE"

# Afficher le nouveau statut
echo "" | tee -a "$LOG_FILE"
echo "📊 NOUVEAU STATUT:" | tee -a "$LOG_FILE"
"$PYTHON" scraper_historique_auto.py --status 2>&1 | tee -a "$LOG_FILE"

# ============================================================================
# STATISTIQUES BASE DE DONNÉES
# ============================================================================

echo "" | tee -a "$LOG_FILE"
echo "📈 STATISTIQUES BASE:" | tee -a "$LOG_FILE"
"$PYTHON" << 'EOF' 2>> "$LOG_FILE"
from db_connection import get_connection

try:
    conn = get_connection()
    cur = conn.cursor()

    # Totaux
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM courses")
    nb_courses = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM performances")
    nb_perfs = cur.fetchone()[0]

    # Plage de dates
    cur.execute("""
        SELECT MIN(SUBSTRING(id_course, 1, 8)), MAX(SUBSTRING(id_course, 1, 8))
        FROM courses
    """)
    min_date, max_date = cur.fetchone()

    print(f"   🐴 Chevaux        : {nb_chevaux:,}")
    print(f"   🏁 Courses        : {nb_courses:,}")
    print(f"   📊 Performances   : {nb_perfs:,}")
    if min_date and max_date:
        min_fmt = f"{min_date[:4]}-{min_date[4:6]}-{min_date[6:]}"
        max_fmt = f"{max_date[:4]}-{max_date[4:6]}-{max_date[6:]}"
        print(f"   📅 Période        : {min_fmt} → {max_fmt}")

    cur.close()
    conn.close()
except Exception as e:
    print(f"   ⚠️ Erreur: {e}")
EOF

echo "" | tee -a "$LOG_FILE"
log "✅ Session terminée"
echo "======================================================================" | tee -a "$LOG_FILE"

# Nettoyer vieux logs (garder 60 jours pour l'historique)
find "$LOG_DIR" -name "historique_*.log" -mtime +60 -delete 2>/dev/null || true

# Lien symbolique vers dernier log
ln -sf "$LOG_FILE" "$LOG_DIR/historique_latest.log" 2>/dev/null || true

exit $EXIT_CODE
