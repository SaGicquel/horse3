#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de nettoyage et re-scraping
Nettoie la table cheval_courses_seen et re-scrape les courses du jour
"""

import sqlite3
from scraper_pmu_simple import *
from datetime import date

def clean_and_rescrape():
    print("=" * 80)
    print("🧹 NETTOYAGE ET RE-SCRAPING")
    print("=" * 80)
    print()
    
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    # Compter avant nettoyage
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_avant = cur.fetchone()[0]
    
    print(f"📊 Avant nettoyage:")
    print(f"   • Entrées dans cheval_courses_seen: {nb_avant}")
    print()
    
    # Nettoyer
    print("🧹 Nettoyage de la table cheval_courses_seen...")
    cur.execute("DELETE FROM cheval_courses_seen")
    con.commit()
    print("✅ Table nettoyée")
    print()
    
    # Re-scraper
    today = date.today().isoformat()
    print(f"🔄 Re-scraping des courses du {today}...")
    print()
    
    db_setup(con)
    
    reunions = discover_reunions(today)
    if not reunions:
        print(f"❌ Aucune course trouvée pour {today}")
        con.close()
        return
    
    print(f"✅ {len(reunions)} réunion(s) trouvée(s): R{', R'.join(map(str, reunions))}")
    print()
    
    total_courses = 0
    total_chevaux = 0
    
    for r in reunions:
        courses = discover_courses(today, r)
        if not courses:
            continue
        
        print(f"📍 RÉUNION R{r}")
        
        for c in courses:
            try:
                print(f"   🏁 R{r}C{c}...", end=" ", flush=True)
                enrich_from_course(cur, today, r, c, sleep_s=0.3)
                con.commit()
                
                # Compter les participants
                plist = fetch_participants(today, r, c)
                nb = len(plist) if plist else 0
                total_chevaux += nb
                total_courses += 1
                
                print(f"✓ {nb} chevaux")
                
            except Exception as e:
                print(f"❌ Erreur: {e}")
        
        print()
    
    # Statistiques finales
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_apres = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT nom_norm) FROM cheval_courses_seen")
    nb_chevaux_distincts = cur.fetchone()[0]
    
    # Vérifier les doublons
    cur.execute("""
        SELECT nom_norm, COUNT(*) as nb
        FROM cheval_courses_seen
        GROUP BY nom_norm
        HAVING COUNT(*) > 1
    """)
    doublons = cur.fetchall()
    
    con.close()
    
    print()
    print("=" * 80)
    print("✅ RE-SCRAPING TERMINÉ")
    print("=" * 80)
    print()
    print(f"📊 Résultats:")
    print(f"   • Courses scrapées: {total_courses}")
    print(f"   • Chevaux traités: {total_chevaux}")
    print(f"   • Entrées avant nettoyage: {nb_avant}")
    print(f"   • Entrées après re-scraping: {nb_apres}")
    print(f"   • Chevaux distincts: {nb_chevaux_distincts}")
    print(f"   • Doublons trouvés: {len(doublons)}")
    
    if doublons:
        print()
        print("⚠️  Chevaux avec plusieurs entrées:")
        for nom, nb in doublons[:10]:
            print(f"   • {nom}: {nb} entrées")
    else:
        print()
        print("✅ Aucun doublon ! Chaque cheval n'apparaît qu'une fois par course.")
    
    print()
    print("💡 Les anciennes performances des chevaux sont maintenant stockées")
    print("   dans la colonne 'dernieres_performances' de la table 'chevaux'")
    print()

if __name__ == "__main__":
    try:
        clean_and_rescrape()
    except KeyboardInterrupt:
        print("\n\n⚠️  Processus interrompu")
    except Exception as e:
        print(f"\n\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
