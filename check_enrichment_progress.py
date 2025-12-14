#!/usr/bin/env python3
"""
Script de monitoring de la progression de l'enrichissement Phase 2A.
Affiche les statistiques de musique et temps enrichis en temps réel.
"""

from db_connection import get_connection
import time

def check_enrichment():
    """Affiche les statistiques d'enrichissement."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Total performances
    cur.execute("SELECT COUNT(*) FROM performances")
    total = cur.fetchone()[0]
    
    # Musique enrichie
    cur.execute("SELECT COUNT(*) FROM performances WHERE musique IS NOT NULL")
    musique_count = cur.fetchone()[0]
    
    # Temps enrichi
    cur.execute("SELECT COUNT(*) FROM performances WHERE temps_total IS NOT NULL")
    temps_count = cur.fetchone()[0]
    
    # Vitesse enrichie
    cur.execute("SELECT COUNT(*) FROM performances WHERE vitesse_moyenne IS NOT NULL")
    vitesse_count = cur.fetchone()[0]
    
    # Courses uniques
    cur.execute("SELECT COUNT(DISTINCT id_course) FROM performances")
    courses_count = cur.fetchone()[0]
    
    cur.close()
    conn.close()
    
    musique_pct = (musique_count / total * 100) if total > 0 else 0
    temps_pct = (temps_count / total * 100) if total > 0 else 0
    vitesse_pct = (vitesse_count / total * 100) if total > 0 else 0
    
    print("=" * 90)
    print("📊 PROGRESSION ENRICHISSEMENT PHASE 2A")
    print("=" * 90)
    print()
    print(f"Courses scraped    : {courses_count:6d}")
    print(f"Performances total : {total:6d}")
    print()
    print(f"Musique enrichie   : {musique_count:6d} / {total:6d} ({musique_pct:6.2f}%)")
    print(f"Temps enrichi      : {temps_count:6d} / {total:6d} ({temps_pct:6.2f}%)")
    print(f"Vitesse enrichie   : {vitesse_count:6d} / {total:6d} ({vitesse_pct:6.2f}%)")
    print()
    print("=" * 90)

if __name__ == '__main__':
    import sys
    
    if '--watch' in sys.argv:
        # Mode watch : rafraîchir toutes les 5 secondes
        try:
            while True:
                print("\033[2J\033[H")  # Clear screen
                check_enrichment()
                print("\n⏳ Rafraîchissement dans 5s... (Ctrl+C pour arrêter)")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring arrêté.")
    else:
        # Mode simple : affichage unique
        check_enrichment()
