#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse finale après scraping d'octobre : identifie les colonnes non enrichies
"""

import psycopg2
import psycopg2.extras
from datetime import datetime

def connect_db():
    """Connexion à la base"""
    return psycopg2.connect(
        host="localhost",
        port=54624,
        database="pmubdd",
        user="postgres",
        password="okokok",
        cursor_factory=psycopg2.extras.RealDictCursor
    )

def analyser_colonnes_vides():
    """Analyse toutes les colonnes pour détecter celles qui sont vides"""
    
    conn = connect_db()
    cur = conn.cursor()
    
    print("\n" + "="*80)
    print("📊 ANALYSE FINALE - COLONNES NON ENRICHIES (OCTOBRE 2024)")
    print("="*80)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Total participations octobre
    cur.execute("""
        SELECT COUNT(*) as total
        FROM cheval_courses_seen
        WHERE race_key LIKE '2024-10-%'
    """)
    total = cur.fetchone()['total']
    print(f"📋 Total participations octobre 2024: {total:,}\n")
    
    # Récupérer toutes les colonnes de la table
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'cheval_courses_seen'
        AND column_name NOT IN ('nom_norm', 'race_key', 'annee', 'is_win')
        ORDER BY column_name
    """)
    colonnes = cur.fetchall()
    
    print(f"🔍 ANALYSE DE {len(colonnes)} COLONNES:\n")
    
    colonnes_vides = []
    colonnes_partielles = []
    colonnes_pleines = []
    
    for col_info in colonnes:
        col_name = col_info['column_name']
        
        # Compter les valeurs NULL
        cur.execute(f"""
            SELECT COUNT(*) as total_null
            FROM cheval_courses_seen
            WHERE race_key LIKE '2024-10-%'
            AND {col_name} IS NULL
        """)
        nb_null = cur.fetchone()['total_null']
        
        # Compter les valeurs non-NULL
        nb_rempli = total - nb_null
        taux_remplissage = (nb_rempli * 100) / total if total > 0 else 0
        
        # Catégoriser
        if taux_remplissage == 0:
            colonnes_vides.append((col_name, nb_null, nb_rempli, taux_remplissage))
        elif taux_remplissage < 80:
            colonnes_partielles.append((col_name, nb_null, nb_rempli, taux_remplissage))
        else:
            colonnes_pleines.append((col_name, nb_null, nb_rempli, taux_remplissage))
    
    # === RAPPORT ===
    
    print(f"✅ COLONNES BIEN REMPLIES (≥80%) : {len(colonnes_pleines)}")
    print(f"⚠️  COLONNES PARTIELLES (<80%)   : {len(colonnes_partielles)}")
    print(f"❌ COLONNES VIDES (0%)           : {len(colonnes_vides)}\n")
    
    # Afficher colonnes vides
    if colonnes_vides:
        print(f"\n{'='*80}")
        print(f"❌ COLONNES COMPLÈTEMENT VIDES ({len(colonnes_vides)}):")
        print(f"{'='*80}\n")
        for col_name, nb_null, nb_rempli, taux in colonnes_vides:
            print(f"   • {col_name}")
    
    # Afficher colonnes partielles (nécessitent enrichissement)
    if colonnes_partielles:
        print(f"\n{'='*80}")
        print(f"⚠️  COLONNES PARTIELLEMENT REMPLIES ({len(colonnes_partielles)}):")
        print(f"{'='*80}\n")
        colonnes_partielles.sort(key=lambda x: x[3])  # Trier par taux croissant
        for col_name, nb_null, nb_rempli, taux in colonnes_partielles:
            print(f"   {taux:5.1f}% │ {col_name:40s} │ {nb_rempli:,}/{total:,}")
    
    # Afficher statistiques globales
    print(f"\n{'='*80}")
    print(f"📈 STATISTIQUES GLOBALES:")
    print(f"{'='*80}\n")
    
    taux_moyen = sum(t[3] for t in colonnes_pleines + colonnes_partielles + colonnes_vides) / len(colonnes)
    print(f"   Taux remplissage moyen : {taux_moyen:.1f}%")
    print(f"   Colonnes pleines       : {len(colonnes_pleines)}/{len(colonnes)} ({len(colonnes_pleines)*100//len(colonnes)}%)")
    print(f"   Colonnes à enrichir    : {len(colonnes_partielles) + len(colonnes_vides)}/{len(colonnes)}")
    
    # Recommandations
    print(f"\n{'='*80}")
    print(f"💡 RECOMMANDATIONS:")
    print(f"{'='*80}\n")
    
    if len(colonnes_vides) > 50:
        print(f"   ⚠️  {len(colonnes_vides)} colonnes sont complètement vides")
        print(f"   → Vérifier si ces colonnes sont utiles ou peuvent être supprimées\n")
    
    if len(colonnes_partielles) > 0:
        # Identifier les colonnes des scrapers
        scrapers_cols = {
            'SCRAPER 1': ['course_id', 'meeting_id', 'heure_locale', 'classe_course', 'meteo_code', 'vent_kmh', 'temperature_c'],
            'SCRAPER 2': ['handicap_distance', 'oeilleres', 'deferrage'],
            'SCRAPER 3': ['nb_places_top3_12m', 'taux_places_12m', 'consistance_12m'],
            'SCRAPER 4': ['biais_stalle', 'pace_debut', 'pace_milieu', 'pace_fin'],
            'SCRAPER 5': ['cote_finale', 'probabilite_implicite', 'tendance_marche'],
            'SCRAPER 7': ['score_composite', 'probabilite_victoire', 'kelly_criterion'],
            'SCRAPER 8': ['entraineur_winrate_90j', 'jockey_winrate_90j']
        }
        
        for scraper, cols in scrapers_cols.items():
            cols_manquantes = [c for c in cols if any(c == pc[0] for pc in colonnes_partielles + colonnes_vides)]
            if cols_manquantes:
                print(f"   🔧 {scraper}: {len(cols_manquantes)} colonnes à enrichir")
                for col in cols_manquantes:
                    print(f"      • {col}")
                print()
    
    print(f"{'='*80}\n")
    
    cur.close()
    conn.close()
    
    return {
        'total': total,
        'colonnes_vides': colonnes_vides,
        'colonnes_partielles': colonnes_partielles,
        'colonnes_pleines': colonnes_pleines,
        'taux_moyen': taux_moyen
    }

if __name__ == "__main__":
    try:
        resultat = analyser_colonnes_vides()
        
        # Sauvegarder dans un fichier
        with open('ANALYSE_ENRICHISSEMENT_OCTOBRE_2024.txt', 'w', encoding='utf-8') as f:
            f.write(f"ANALYSE ENRICHISSEMENT OCTOBRE 2024\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"="*80 + "\n\n")
            
            f.write(f"Total participations : {resultat['total']:,}\n")
            f.write(f"Colonnes vides       : {len(resultat['colonnes_vides'])}\n")
            f.write(f"Colonnes partielles  : {len(resultat['colonnes_partielles'])}\n")
            f.write(f"Colonnes pleines     : {len(resultat['colonnes_pleines'])}\n")
            f.write(f"Taux moyen           : {resultat['taux_moyen']:.1f}%\n\n")
            
            f.write(f"COLONNES À ENRICHIR:\n")
            f.write(f"-"*80 + "\n")
            for col_name, nb_null, nb_rempli, taux in resultat['colonnes_partielles']:
                f.write(f"{taux:5.1f}% │ {col_name}\n")
        
        print(f"✅ Rapport sauvegardé dans: ANALYSE_ENRICHISSEMENT_OCTOBRE_2024.txt\n")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
