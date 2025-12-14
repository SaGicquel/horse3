#!/bin/bash

################################################################################
# SCRIPT CRON - ENRICHISSEMENT QUOTIDIEN TURF.BZH
################################################################################
# 
# Description : Script d'automatisation pour enrichir quotidiennement la base
#               de données avec les cotes et prédictions de Turf.bzh
#
# Fréquence recommandée : 23h30 tous les jours (après publication des cotes PMU)
#
# Installation CRON :
#   crontab -e
#   Ajouter : 30 23 * * * /Users/gicquelsacha/horse3/cron_enrichissement_quotidien.sh >> /Users/gicquelsacha/horse3/logs/cron_quotidien.log 2>&1
#
################################################################################

# Configuration
PROJECT_DIR="/Users/gicquelsacha/horse3"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y%m%d_%H%M%S)

# Créer le dossier logs s'il n'existe pas
mkdir -p "$LOG_DIR"

# Log file pour cette exécution
LOG_FILE="$LOG_DIR/cron_quotidien_$DATE.log"

echo "======================================================================" | tee -a "$LOG_FILE"
echo "🚀 DÉMARRAGE ENRICHISSEMENT QUOTIDIEN" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "Date       : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "Projet     : $PROJECT_DIR" | tee -a "$LOG_FILE"
echo "Python     : $PYTHON" | tee -a "$LOG_FILE"
echo "Log        : $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Vérifier que le virtualenv existe
if [ ! -f "$PYTHON" ]; then
    echo "❌ ERREUR : Python virtualenv introuvable à $PYTHON" | tee -a "$LOG_FILE"
    exit 1
fi

# Vérifier PostgreSQL
echo "🔍 Vérification PostgreSQL..." | tee -a "$LOG_FILE"
pg_isready -h localhost -p 54624 >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  PostgreSQL non accessible, tentative de redémarrage..." | tee -a "$LOG_FILE"
    brew services restart postgresql@14 >> "$LOG_FILE" 2>&1
    sleep 5
    pg_isready -h localhost -p 54624 >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ ERREUR : Impossible de démarrer PostgreSQL" | tee -a "$LOG_FILE"
        exit 1
    fi
fi
echo "✅ PostgreSQL opérationnel" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Changer de répertoire
cd "$PROJECT_DIR" || exit 1

# ============================================================================
# ÉTAPE 1 : ENRICHISSEMENT TURF.BZH (Jour J + Jour J-1)
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "📊 ÉTAPE 1/2 : Enrichissement Turf.bzh (2 derniers jours)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

"$PYTHON" enrichir_batch_turfbzh.py --days 2 >> "$LOG_FILE" 2>&1
TURF_EXIT=$?

if [ $TURF_EXIT -eq 0 ]; then
    echo "✅ Enrichissement Turf.bzh terminé avec succès" | tee -a "$LOG_FILE"
else
    echo "⚠️  Enrichissement Turf.bzh terminé avec code $TURF_EXIT" | tee -a "$LOG_FILE"
    echo "   Voir détails dans $LOG_FILE" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# ÉTAPE 2 : ENRICHISSEMENT ZONE-TURF (Jour J)
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "📊 ÉTAPE 2/2 : Enrichissement Zone-Turf (aujourd'hui)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Note : Zone-Turf publie les résultats le lendemain matin
# On enrichit donc "today" qui contient les résultats de la veille au soir
"$PYTHON" enrichir_zoneturf.py --date today >> "$LOG_FILE" 2>&1
ZONE_EXIT=$?

if [ $ZONE_EXIT -eq 0 ]; then
    echo "✅ Enrichissement Zone-Turf terminé avec succès" | tee -a "$LOG_FILE"
else
    echo "⚠️  Enrichissement Zone-Turf terminé avec code $ZONE_EXIT" | tee -a "$LOG_FILE"
    echo "   (Normal si aucune course publiée encore)" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# RAPPORT FINAL
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "📋 RAPPORT FINAL" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "Date fin    : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "Turf.bzh    : $([ $TURF_EXIT -eq 0 ] && echo '✅ OK' || echo "⚠️  Code $TURF_EXIT")" | tee -a "$LOG_FILE"
echo "Zone-Turf   : $([ $ZONE_EXIT -eq 0 ] && echo '✅ OK' || echo "⚠️  Code $ZONE_EXIT")" | tee -a "$LOG_FILE"
echo "Log complet : $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Statistiques rapides de la base de données
echo "📊 STATISTIQUES BASE DE DONNÉES" | tee -a "$LOG_FILE"
echo "-" | tee -a "$LOG_FILE"
"$PYTHON" -c "
from db_connection import get_connection
conn = get_connection()
cur = conn.cursor()

# Stats performances enrichies
cur.execute(\"\"\"
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN cote_turfbzh IS NOT NULL THEN 1 END) as avec_turfbzh,
        COUNT(CASE WHEN musique IS NOT NULL THEN 1 END) as avec_musique
    FROM performances
    WHERE id_course IN (
        SELECT id_course FROM courses 
        WHERE SUBSTRING(id_course, 1, 8) >= TO_CHAR(NOW() - INTERVAL '7 days', 'YYYYMMDD')
    )
\"\"\")
total, turf, mus = cur.fetchone()

print(f'   Performances (7 derniers jours) : {total:,}')
print(f'   Avec cotes Turf.bzh             : {turf:,} ({100*turf//total if total > 0 else 0}%)')
print(f'   Avec musique Zone-Turf          : {mus:,} ({100*mus//total if total > 0 else 0}%)')

cur.close()
conn.close()
" >> "$LOG_FILE" 2>&1

echo "" | tee -a "$LOG_FILE"
echo "✅ Enrichissement quotidien terminé !" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# Nettoyer les vieux logs (garder 30 jours)
find "$LOG_DIR" -name "cron_quotidien_*.log" -mtime +30 -delete 2>/dev/null

exit 0
