#!/bin/bash

################################################################################
# SCRIPT CRON - ENRICHISSEMENT HEBDOMADAIRE COMPLET
################################################################################
#
# Description : Enrichissement complet de la semaine écoulée + maintenance DB
#
# Fréquence recommandée : Dimanche 2h00 du matin (faible charge système)
#
# Installation CRON :
#   crontab -e
#   Ajouter : 0 2 * * 0 /Users/gicquelsacha/horse3/cron_enrichissement_hebdo.sh >> /Users/gicquelsacha/horse3/logs/cron_hebdo.log 2>&1
#
################################################################################

# Configuration
PROJECT_DIR="/Users/gicquelsacha/horse3"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
DATE=$(date +%Y%m%d_%H%M%S)

# Créer le dossier logs
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/cron_hebdo_$DATE.log"

echo "======================================================================" | tee -a "$LOG_FILE"
echo "🚀 DÉMARRAGE ENRICHISSEMENT HEBDOMADAIRE" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "Date       : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "Projet     : $PROJECT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Vérifier PostgreSQL
echo "🔍 Vérification PostgreSQL..." | tee -a "$LOG_FILE"
pg_isready -h localhost -p 54624 >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ ERREUR : PostgreSQL non accessible" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ PostgreSQL opérationnel" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1

# ============================================================================
# ÉTAPE 1 : ENRICHISSEMENT TURF.BZH (7 derniers jours)
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "📊 ÉTAPE 1/3 : Enrichissement Turf.bzh (7 jours)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

"$PYTHON" enrichir_batch_turfbzh.py --days 7 >> "$LOG_FILE" 2>&1
TURF_EXIT=$?

echo "   Turf.bzh : $([ $TURF_EXIT -eq 0 ] && echo '✅ OK' || echo "⚠️  Code $TURF_EXIT")" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# ÉTAPE 2 : ENRICHISSEMENT ZONE-TURF (7 derniers jours)
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "📊 ÉTAPE 2/3 : Enrichissement Zone-Turf (7 jours)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Calculer date de début (7 jours en arrière)
DATE_START=$(date -v-7d +%Y-%m-%d)
DATE_END=$(date +%Y-%m-%d)

"$PYTHON" enrichir_zoneturf.py --date-range "$DATE_START" "$DATE_END" >> "$LOG_FILE" 2>&1
ZONE_EXIT=$?

echo "   Zone-Turf : $([ $ZONE_EXIT -eq 0 ] && echo '✅ OK' || echo "⚠️  Code $ZONE_EXIT")" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# ÉTAPE 3 : MAINTENANCE BASE DE DONNÉES
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "🔧 ÉTAPE 3/3 : Maintenance base de données" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# VACUUM ANALYZE pour optimiser les performances
echo "   🧹 VACUUM ANALYZE des tables principales..." | tee -a "$LOG_FILE"
"$PYTHON" -c "
from db_connection import get_connection
conn = get_connection()
conn.autocommit = True
cur = conn.cursor()

tables = ['courses', 'performances', 'chevaux', 'hippodromes', 'jockeys', 'entraineurs']
for table in tables:
    print(f'      Optimisation {table}...')
    cur.execute(f'VACUUM ANALYZE {table}')

cur.close()
conn.close()
print('   ✅ Optimisation terminée')
" >> "$LOG_FILE" 2>&1

echo "" | tee -a "$LOG_FILE"

# ============================================================================
# RAPPORT FINAL DÉTAILLÉ
# ============================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "📋 RAPPORT HEBDOMADAIRE COMPLET" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

"$PYTHON" -c "
from db_connection import get_connection
from datetime import datetime, timedelta

conn = get_connection()
cur = conn.cursor()

# Période analysée
date_fin = datetime.now()
date_debut = date_fin - timedelta(days=7)

print(f'📅 Période : {date_debut.strftime(\"%Y-%m-%d\")} → {date_fin.strftime(\"%Y-%m-%d\")}')
print()

# Stats par jour
cur.execute(\"\"\"
    SELECT
        SUBSTRING(c.id_course, 1, 8) as date_course,
        COUNT(DISTINCT c.id_course) as nb_courses,
        COUNT(p.id_performance) as nb_perfs,
        COUNT(CASE WHEN p.cote_turfbzh IS NOT NULL THEN 1 END) as avec_turf,
        COUNT(CASE WHEN p.musique IS NOT NULL THEN 1 END) as avec_musique
    FROM courses c
    JOIN performances p ON c.id_course = p.id_course
    WHERE SUBSTRING(c.id_course, 1, 8) >= %s
      AND SUBSTRING(c.id_course, 1, 8) <= %s
    GROUP BY SUBSTRING(c.id_course, 1, 8)
    ORDER BY date_course DESC
\"\"\", (date_debut.strftime('%Y%m%d'), date_fin.strftime('%Y%m%d')))

rows = cur.fetchall()

print('📊 DÉTAIL PAR JOUR')
print('-' * 80)
print(f'   {\"Date\":>10} {\"Courses\":>8} {\"Perfs\":>8} {\"Turf.bzh\":>10} {\"Zone-Turf\":>10}')
print('   ' + '-' * 70)

total_courses = 0
total_perfs = 0
total_turf = 0
total_mus = 0

for date, courses, perfs, turf, mus in rows:
    turf_pct = (turf / perfs * 100) if perfs > 0 else 0
    mus_pct = (mus / perfs * 100) if perfs > 0 else 0
    print(f'   {date:>10} {courses:8,} {perfs:8,} {turf_pct:9.1f}% {mus_pct:9.1f}%')
    total_courses += courses
    total_perfs += perfs
    total_turf += turf
    total_mus += mus

if total_perfs > 0:
    turf_pct_total = (total_turf / total_perfs * 100)
    mus_pct_total = (total_mus / total_perfs * 100)
    print('   ' + '-' * 70)
    print(f'   {\"TOTAL\":>10} {total_courses:8,} {total_perfs:8,} {turf_pct_total:9.1f}% {mus_pct_total:9.1f}%')

print()
print('📈 ÉVOLUTION GLOBALE BASE DE DONNÉES')
print('-' * 80)

# Stats totales
cur.execute(\"\"\"
    SELECT
        COUNT(*) as total,
        COUNT(CASE WHEN cote_turfbzh IS NOT NULL THEN 1 END) as avec_turf,
        COUNT(CASE WHEN musique IS NOT NULL THEN 1 END) as avec_musique,
        COUNT(CASE WHEN temps_total IS NOT NULL THEN 1 END) as avec_temps
    FROM performances
\"\"\")
total, turf, mus, temps = cur.fetchone()

print(f'   Performances totales             : {total:,}')
print(f'   Avec cotes Turf.bzh              : {turf:,} ({100*turf//total if total > 0 else 0}%)')
print(f'   Avec musique Zone-Turf           : {mus:,} ({100*mus//total if total > 0 else 0}%)')
print(f'   Avec temps de course             : {temps:,} ({100*temps//total if total > 0 else 0}%)')

cur.close()
conn.close()
" >> "$LOG_FILE" 2>&1

echo "" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "Date fin    : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "✅ Enrichissement hebdomadaire terminé !" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# Nettoyer les vieux logs (garder 90 jours pour les hebdos)
find "$LOG_DIR" -name "cron_hebdo_*.log" -mtime +90 -delete 2>/dev/null

exit 0
