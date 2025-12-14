#!/bin/bash

# Script de surveillance de la progression du scraping Phase 2A

while true; do
    clear
    echo "========================================================================"
    echo "🔄 MONITORING SCRAPING PHASE 2A - $(date '+%H:%M:%S')"
    echo "========================================================================"
    echo ""
    
    # Vérifier si le processus tourne
    if ps aux | grep -E "python.*scraper_pmu_adapter.*2025-10-13" | grep -v grep > /dev/null; then
        echo "✅ Processus de scraping actif (PID: $(pgrep -f 'scraper_pmu_adapter.*2025-10-13'))"
    else
        echo "❌ Processus de scraping arrêté !"
    fi
    echo ""
    
    # Stats globales
    python3 << 'PYEOF'
from db_connection import get_connection

conn = get_connection()
cur = conn.cursor()

# Stats 2025
cur.execute("""
    SELECT 
        COUNT(p.id_performance) as total,
        COUNT(CASE WHEN p.musique IS NOT NULL THEN 1 END) as musique,
        COUNT(CASE WHEN p.temps_total IS NOT NULL THEN 1 END) as temps
    FROM performances p
    JOIN courses c ON p.id_course = c.id_course
    WHERE SUBSTRING(c.id_course, 1, 4) = '2025'
""")
total, mus, tps = cur.fetchone()

mus_pct = (mus / total * 100) if total > 0 else 0
tps_pct = (tps / total * 100) if total > 0 else 0

print(f"📊 GLOBAL 2025")
print(f"   Total      : {total:6,} performances")
print(f"   Musique    : {mus:6,} ({mus_pct:5.1f}%)")
print(f"   Temps      : {tps:6,} ({tps_pct:5.1f}%)")
print()

# Dates enrichies
cur.execute("""
    SELECT 
        COUNT(DISTINCT CASE 
            WHEN COUNT(CASE WHEN p.musique IS NOT NULL THEN 1 END) * 100.0 / COUNT(*) > 50 
            THEN SUBSTRING(c.id_course, 1, 8) 
        END) as dates_ok,
        COUNT(DISTINCT SUBSTRING(c.id_course, 1, 8)) as dates_total
    FROM courses c
    JOIN performances p ON c.id_course = p.id_course
    WHERE SUBSTRING(c.id_course, 1, 4) = '2025'
    GROUP BY SUBSTRING(c.id_course, 1, 8)
""")
result = cur.fetchone()
dates_ok = 0
dates_total = 0

# Compter manuellement
cur.execute("""
    SELECT 
        SUBSTRING(c.id_course, 1, 8) as date_course,
        COUNT(*) as total,
        COUNT(CASE WHEN p.musique IS NOT NULL THEN 1 END) as avec_mus
    FROM courses c
    JOIN performances p ON c.id_course = p.id_course
    WHERE SUBSTRING(c.id_course, 1, 4) = '2025'
    GROUP BY SUBSTRING(c.id_course, 1, 8)
""")
for date, tot, mus_cnt in cur.fetchall():
    dates_total += 1
    if tot > 0 and (mus_cnt * 100.0 / tot) > 50:
        dates_ok += 1

print(f"📅 PROGRESSION")
print(f"   Dates enrichies : {dates_ok:2}/{dates_total:2} ({dates_ok*100//dates_total if dates_total > 0 else 0}%)")

if dates_ok < dates_total:
    dates_restantes = dates_total - dates_ok
    eta_min = dates_restantes * 2
    print(f"   ⏱️  ETA : ~{eta_min} minutes")

cur.close()
conn.close()
PYEOF
    
    echo ""
    echo "========================================================================"
    echo "⏳ Rafraîchissement dans 30s... (Ctrl+C pour arrêter)"
    echo ""
    
    sleep 30
done
