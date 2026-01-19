#!/usr/bin/env python3
"""
Script pour lancer le scraping sur les 30 derniers jours
"""

from datetime import datetime, timedelta
from scraper_pmu_simple import run
import time

# Dates - 30 derniers jours depuis aujourd'hui
end_date = datetime.now()
start_date = end_date - timedelta(days=29)  # 30 jours au total (aujourd'hui inclus)

dates = []
current = start_date
while current <= end_date:
    dates.append(current.strftime("%Y-%m-%d"))
    current += timedelta(days=1)

print(f"\n{'=' * 80}")
print(f"🏇 SCRAPING PMU - {len(dates)} JOURS")
print(f"{'=' * 80}")
print(f"Du {dates[0]} au {dates[-1]}")
print(f"Mode : Multi-threadé ({8} workers)")
print(f"{'=' * 80}\n")

start_time = time.time()
success_count = 0
error_count = 0

for i, date_iso in enumerate(dates, 1):
    print(f"\n[{i}/{len(dates)}] 📅 {date_iso}")
    print("-" * 80)
    try:
        run(date_iso, recalc_after=False, use_threading=True)
        success_count += 1
        print(f"✅ {date_iso} terminé")
    except Exception as e:
        error_count += 1
        print(f"❌ {date_iso} échoué : {e}")

elapsed = time.time() - start_time

print(f"\n{'=' * 80}")
print("📊 RÉSUMÉ FINAL")
print(f"{'=' * 80}")
print(f"Jours scrapés : {success_count}/{len(dates)}")
print(f"Erreurs : {error_count}")
print(f"Temps total : {elapsed/60:.1f} minutes")
print(f"{'=' * 80}\n")
