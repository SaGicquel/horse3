#!/bin/bash

################################################################################
# SCRIPT CRON - SCRAPING QUOTIDIEN AUTOMATIQUE
################################################################################
#
# Description : Script d'automatisation pour scraper quotidiennement les courses
#               du jour avec toutes les informations associées (participants,
#               cotes, chevaux, jockeys, etc.)
#
# Fréquence recommandée :
#   - 06h00 : Scraping matinal (données du jour)
#   - 12h00 : Mise à jour mi-journée (nouvelles cotes)
#   - 18h00 : Mise à jour soirée (résultats courses terminées)
#
# Installation CRON :
#   crontab -e
#   # Scraping du matin (6h00)
#   0 6 * * * /Users/gicquelsacha/horse3/cron_scraping_quotidien.sh >> /Users/gicquelsacha/horse3/logs/cron_scraping.log 2>&1
#
#   # Mise à jour mi-journée (12h00) - optionnel
#   0 12 * * * /Users/gicquelsacha/horse3/cron_scraping_quotidien.sh >> /Users/gicquelsacha/horse3/logs/cron_scraping.log 2>&1
#
#   # Mise à jour soirée (18h00) - optionnel
#   0 18 * * * /Users/gicquelsacha/horse3/cron_scraping_quotidien.sh >> /Users/gicquelsacha/horse3/logs/cron_scraping.log 2>&1
#
################################################################################

set -e  # Arrêter en cas d'erreur

# Configuration
PROJECT_DIR="/Users/gicquelsacha/horse3"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y%m%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Couleurs pour les logs (si terminal supporte)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Créer le dossier logs s'il n'existe pas
mkdir -p "$LOG_DIR"

# Log file pour cette exécution
LOG_FILE="$LOG_DIR/scraping_${TIMESTAMP}.log"

# Fonction de log
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1" | tee -a "$LOG_FILE"
}

# ============================================================================
# DÉBUT DU SCRIPT
# ============================================================================

echo "======================================================================" | tee -a "$LOG_FILE"
echo "🏇 SCRAPING QUOTIDIEN AUTOMATIQUE" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
log "Date       : $(date '+%Y-%m-%d %H:%M:%S')"
log "Projet     : $PROJECT_DIR"
log "Python     : $PYTHON"
log "Log        : $LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# VÉRIFICATIONS PRÉLIMINAIRES
# ============================================================================

# Vérifier que le virtualenv existe
if [ ! -f "$PYTHON" ]; then
    log_error "Python virtualenv introuvable à $PYTHON"
    exit 1
fi
log_success "Virtualenv Python trouvé"

# Vérifier PostgreSQL
log_info "Vérification PostgreSQL..."
if ! pg_isready -h localhost -p 54624 >> "$LOG_FILE" 2>&1; then
    log_warning "PostgreSQL non accessible, tentative de redémarrage..."

    # Essayer de redémarrer PostgreSQL
    if command -v brew &> /dev/null; then
        brew services restart postgresql@14 >> "$LOG_FILE" 2>&1 || true
    fi

    sleep 5

    if ! pg_isready -h localhost -p 54624 >> "$LOG_FILE" 2>&1; then
        log_error "Impossible de démarrer PostgreSQL"
        exit 1
    fi
fi
log_success "PostgreSQL opérationnel"

# Changer de répertoire
cd "$PROJECT_DIR" || exit 1
log_success "Répertoire de travail: $PROJECT_DIR"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# ÉTAPE 1 : SCRAPING DES COURSES DU JOUR
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "📡 ÉTAPE 1/5 : Scraping des courses du jour via API PMU" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

SCRAPING_EXIT=0
"$PYTHON" scraper_today.py >> "$LOG_FILE" 2>&1 || SCRAPING_EXIT=$?

if [ $SCRAPING_EXIT -eq 0 ]; then
    log_success "Scraping PMU terminé avec succès"
else
    log_warning "Scraping PMU terminé avec code $SCRAPING_EXIT"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# ÉTAPE 2 : MISE À JOUR DES RÉSULTATS (J-1)
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "📊 ÉTAPE 2/5 : Mise à jour des résultats (courses terminées)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

UPDATE_EXIT=0
if [ -f "$PROJECT_DIR/update_results.py" ]; then
    "$PYTHON" update_results.py >> "$LOG_FILE" 2>&1 || UPDATE_EXIT=$?

    if [ $UPDATE_EXIT -eq 0 ]; then
        log_success "Mise à jour des résultats terminée"
    else
        log_warning "Mise à jour des résultats: code $UPDATE_EXIT (normal si pas de paris en cours)"
    fi
else
    log_info "Script update_results.py non trouvé - étape ignorée"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# ÉTAPE 3 : ENRICHISSEMENT TURF.BZH (COTES)
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "📈 ÉTAPE 3/5 : Enrichissement Turf.bzh (cotes prédictives)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

TURF_EXIT=0
if [ -f "$PROJECT_DIR/enrichir_batch_turfbzh.py" ]; then
    "$PYTHON" enrichir_batch_turfbzh.py --days 1 >> "$LOG_FILE" 2>&1 || TURF_EXIT=$?

    if [ $TURF_EXIT -eq 0 ]; then
        log_success "Enrichissement Turf.bzh terminé"
    else
        log_warning "Enrichissement Turf.bzh: code $TURF_EXIT"
    fi
else
    log_info "Script enrichir_batch_turfbzh.py non trouvé - étape ignorée"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# ÉTAPE 4 : ENRICHISSEMENT ZONE-TURF (MUSIQUE)
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "🎵 ÉTAPE 4/5 : Enrichissement Zone-Turf (historique musique)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

ZONE_EXIT=0
if [ -f "$PROJECT_DIR/enrichir_zoneturf.py" ]; then
    "$PYTHON" enrichir_zoneturf.py --date today >> "$LOG_FILE" 2>&1 || ZONE_EXIT=$?

    if [ $ZONE_EXIT -eq 0 ]; then
        log_success "Enrichissement Zone-Turf terminé"
    else
        log_warning "Enrichissement Zone-Turf: code $ZONE_EXIT (normal si résultats pas encore publiés)"
    fi
else
    log_info "Script enrichir_zoneturf.py non trouvé - étape ignorée"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# ÉTAPE 5 : PRÉPARATION DES FEATURES ML (OPTIONNEL)
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "🤖 ÉTAPE 5/5 : Préparation des features ML" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

FEATURES_EXIT=0
TODAY_ISO=$(date +%Y-%m-%d)
if [ -f "$PROJECT_DIR/prepare_daily_features.py" ]; then
    "$PYTHON" prepare_daily_features.py --date "$TODAY_ISO" >> "$LOG_FILE" 2>&1 || FEATURES_EXIT=$?

    if [ $FEATURES_EXIT -eq 0 ]; then
        log_success "Features ML préparées"
    else
        log_warning "Préparation features: code $FEATURES_EXIT"
    fi
else
    log_info "Script prepare_daily_features.py non trouvé - étape ignorée"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# RAPPORT FINAL
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "📋 RAPPORT FINAL" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
log "Date fin        : $(date '+%Y-%m-%d %H:%M:%S')"
log "Scraping PMU    : $([ $SCRAPING_EXIT -eq 0 ] && echo '✅ OK' || echo "⚠️  Code $SCRAPING_EXIT")"
log "Résultats J-1   : $([ $UPDATE_EXIT -eq 0 ] && echo '✅ OK' || echo "⚠️  Code $UPDATE_EXIT")"
log "Turf.bzh        : $([ $TURF_EXIT -eq 0 ] && echo '✅ OK' || echo "⚠️  Code $TURF_EXIT")"
log "Zone-Turf       : $([ $ZONE_EXIT -eq 0 ] && echo '✅ OK' || echo "⚠️  Code $ZONE_EXIT")"
log "Features ML     : $([ $FEATURES_EXIT -eq 0 ] && echo '✅ OK' || echo "⚠️  Code $FEATURES_EXIT")"
log "Log complet     : $LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# STATISTIQUES BASE DE DONNÉES
# ============================================================================
echo "📊 STATISTIQUES BASE DE DONNÉES" | tee -a "$LOG_FILE"
echo "----------------------------------------------------------------------" | tee -a "$LOG_FILE"

"$PYTHON" << 'EOF' 2>> "$LOG_FILE"
from db_connection import get_connection
from datetime import date

try:
    conn = get_connection()
    cur = conn.cursor()
    today = date.today().isoformat()

    # Courses du jour
    cur.execute("""
        SELECT COUNT(*) FROM courses
        WHERE id_course LIKE %s
    """, (today.replace('-', '') + '%',))
    nb_courses = cur.fetchone()[0]

    # Performances du jour
    cur.execute("""
        SELECT COUNT(*) FROM performances p
        JOIN courses c ON p.id_course = c.id_course
        WHERE c.id_course LIKE %s
    """, (today.replace('-', '') + '%',))
    nb_perfs = cur.fetchone()[0]

    # Total chevaux
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux = cur.fetchone()[0]

    # Courses total
    cur.execute("SELECT COUNT(*) FROM courses")
    nb_courses_total = cur.fetchone()[0]

    print(f"   📅 Courses aujourd'hui ({today})  : {nb_courses}")
    print(f"   🐴 Participants aujourd'hui      : {nb_perfs}")
    print(f"   📚 Total chevaux en base         : {nb_chevaux:,}")
    print(f"   🏁 Total courses en base         : {nb_courses_total:,}")

    cur.close()
    conn.close()
except Exception as e:
    print(f"   ⚠️  Erreur stats: {e}")
EOF

echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
log_success "SCRAPING QUOTIDIEN TERMINÉ !"
echo "======================================================================" | tee -a "$LOG_FILE"

# ============================================================================
# NETTOYAGE DES VIEUX LOGS
# ============================================================================
# Garder les logs des 30 derniers jours
find "$LOG_DIR" -name "scraping_*.log" -mtime +30 -delete 2>/dev/null || true

# Créer un lien symbolique vers le dernier log
ln -sf "$LOG_FILE" "$LOG_DIR/scraping_latest.log" 2>/dev/null || true

exit 0
