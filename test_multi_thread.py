#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test rapide du scraper multi-threadé
Scrape seulement quelques courses pour vérifier que tout fonctionne
"""

import sys
from datetime import date
from scraper_pmu_simple import discover_reunions, discover_courses, run


def main():
    today = date.today().isoformat()

    print("🔍 Analyse du programme du jour...")
    reunions = discover_reunions(today)

    if not reunions:
        print(f"❌ Aucune course aujourd'hui ({today})")
        return

    total_courses = 0
    for r in reunions[:3]:  # Max 3 réunions pour le test
        courses = discover_courses(today, r)
        total_courses += len(courses)
        print(f"   • R{r}: {len(courses)} courses")

    print(f"\n📊 Total: {len(reunions)} réunions, ~{total_courses} courses")
    print("\n" + "=" * 60)
    print("⚠️  Ce test va scraper seulement 3 réunions pour vérifier")
    print("    que le multi-threading fonctionne correctement.")
    print("=" * 60)

    response = input("\n▶️  Continuer le test ? (o/N): ")

    if response.lower() != "o":
        print("❌ Test annulé")
        return

    print("\n🚀 Lancement du test multi-threadé...\n")

    import time

    start = time.time()

    # Le scraper ne prendra que les courses découvertes
    run(today, recalc_after=False, use_threading=True)

    elapsed = time.time() - start

    print(f"\n✅ Test terminé en {elapsed:.1f}s")
    print("\n💡 Pour un scraping complet:")
    print("   python scraper_pmu_simple.py")


if __name__ == "__main__":
    main()
