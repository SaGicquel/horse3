#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scraper PMU - Courses du jour
Récupère toutes les informations disponibles pour les courses du jour
Gère les doublons par nom + date de naissance
"""

from scraper_pmu_simple import *
from datetime import date
import sys

def main():
    # Date du jour
    today = date.today().isoformat()
    
    print("=" * 80)
    print(f"🏇 SCRAPER PMU - COURSES DU {today}")
    print("=" * 80)
    print()
    
    # Connexion à la base de données
    con = sqlite3.connect(DB_PATH)
    db_setup(con)
    
    # Statistiques avant scraping
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux_avant = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_courses_avant = cur.fetchone()[0]
    
    print(f"📊 État initial de la base de données:")
    print(f"   • Chevaux enregistrés: {nb_chevaux_avant}")
    print(f"   • Courses distinctes: {nb_courses_avant}")
    print()
    
    # Découverte et scraping des réunions
    print(f"🔍 Recherche des réunions du {today}...")
    reunions = discover_reunions(today)
    
    if not reunions:
        print(f"❌ Aucune course trouvée pour {today}")
        print("   Possible raisons:")
        print("   - Pas de courses programmées aujourd'hui")
        print("   - API PMU indisponible")
        print("   - Date incorrecte")
        con.close()
        return
    
    print(f"✅ {len(reunions)} réunion(s) trouvée(s): R{', R'.join(map(str, reunions))}")
    print()
    
    # Scraping de chaque réunion
    total_courses = 0
    total_chevaux_scraped = 0
    
    for r in reunions:
        courses = discover_courses(today, r)
        if not courses:
            continue
        
        print(f"📍 RÉUNION R{r} - {len(courses)} course(s)")
        
        for c in courses:
            try:
                print(f"   🏁 Course C{c}...", end=" ", flush=True)
                
                # Scraping de la course
                enrich_from_course(cur, today, r, c, sleep_s=0.5)
                con.commit()
                
                # Compter les chevaux de cette course
                plist = fetch_participants(today, r, c)
                nb_participants = len(plist) if plist else 0
                total_chevaux_scraped += nb_participants
                total_courses += 1
                
                print(f"✓ {nb_participants} chevaux")
                
            except requests.HTTPError as e:
                print(f"❌ Erreur HTTP: {e}")
            except Exception as e:
                print(f"❌ Erreur: {e}")
        
        print()
    
    # Recalcul des totaux
    print("🔄 Recalcul des statistiques...")
    recalc_totals_from_seen(con)
    
    # Statistiques après scraping
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux_apres = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_courses_apres = cur.fetchone()[0]
    
    con.close()
    
    # Récapitulatif final
    print()
    print("=" * 80)
    print("✅ SCRAPING TERMINÉ")
    print("=" * 80)
    print()
    print(f"📊 Statistiques:")
    print(f"   • Courses scrapées: {total_courses}")
    print(f"   • Chevaux traités: {total_chevaux_scraped}")
    print()
    print(f"📈 Base de données:")
    print(f"   • Chevaux avant: {nb_chevaux_avant}")
    print(f"   • Chevaux après: {nb_chevaux_apres}")
    print(f"   • Nouveaux chevaux: {nb_chevaux_apres - nb_chevaux_avant}")
    print(f"   • Courses distinctes avant: {nb_courses_avant}")
    print(f"   • Courses distinctes après: {nb_courses_apres}")
    print(f"   • Nouvelles courses: {nb_courses_apres - nb_courses_avant}")
    print()
    print("💾 Base de données: data/database.db")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
