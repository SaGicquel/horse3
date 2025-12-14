#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENRICHISSEMENT BATCH HIPPODROMES - Phase 2C
============================================
Enrichit tous les hippodromes français avec données géographiques

Filtre les hippodromes étrangers et priorise les principaux hippodromes
"""

import sys
import time
import logging
from db_connection import get_connection
from scraper_hippodromes import ScraperHippodromes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Codes des principaux hippodromes français (prioritaires)
HIPPODROMES_PRINCIPAUX = [
    'VINC', 'LGC', 'AUT', 'CHT', 'MAI', 'SAI', 'DEA', 'CAG',
    'CLM', 'FCH', 'COM', 'LAF', 'PAU', 'VIC', 'MNT', 'BOR',
    'LYN', 'MRS', 'TOU', 'NCE', 'NAN', 'REN', 'ANG', 'LAV',
    'CAE', 'STR', 'CHA', 'DOU', 'AMI', 'LES', 'ARR', 'CRA',
    'EVR', 'NMR', 'CFO', 'SNS', 'MSE', 'VIT', 'CHC', 'LPL'
]

# Mots-clés pour identifier les hippodromes étrangers
MOTS_CLES_ETRANGERS = [
    'USA', 'KINGDOM', 'ALLEMAGNE', 'ALL', 'ITALIE', 'ITA',
    'ESPAGNE', 'ESP', 'BELGIQUE', 'BEL', 'SUISSE', 'SUI',
    'JAPON', 'JAP', 'AUSTRALIE', 'AUS', 'ARGENTINE', 'ARG',
    'BRESIL', 'BRE', 'CHILI', 'CHI', 'MEXIQUE', 'MEX',
    'CANADA', 'CAN', 'IRLANDE', 'IRE', 'GRANDE-BRETAGNE', 'GB',
    'PAYS-BAS', 'PAY', 'DANEMARK', 'DAN', 'SUEDE', 'SUE',
    'NORVEGE', 'NOR', 'POLOGNE', 'POL', 'HONGRIE', 'HON',
    'TCHEQUE', 'TCH', 'SLOVAQUIE', 'SLO', 'AUTRICHE', 'AUT-ETR',
    'GOTEBORG', 'BERLIN', 'MUNICH', 'ROME', 'MILAN', 'MADRID',
    'BARCELONE', 'LONDRES', 'NEWMARKET', 'ASCOT', 'EPSOM',
    'FLEMINGTON', 'RANDWICK', 'TOKYO', 'SHA', 'HONG KONG',
    'DUBAI', 'MEYDAN', 'PIMLICO', 'BELMONT', 'CHURCHILL',
    'KEENELAND', 'GULFSTREAM', 'SANTA ANITA', 'DEL MAR',
    'AQUEDUCT', 'SARATOGA', 'BADEN-BADEN', 'GAVEA', 'RIO'
]


def est_hippodrome_francais(nom: str, code: str) -> bool:
    """Détermine si un hippodrome est français"""
    # Si dans la liste prioritaire
    if code in HIPPODROMES_PRINCIPAUX:
        return True
    
    # Vérifier mots-clés étrangers
    nom_upper = nom.upper()
    for mot_cle in MOTS_CLES_ETRANGERS:
        if mot_cle in nom_upper:
            return False
    
    # Si pas de mots-clés étrangers, probablement français
    return True


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrichissement batch hippodromes français")
    parser.add_argument('--skip', type=int, default=0, help='Sauter N hippodromes')
    parser.add_argument('--limit', type=int, help='Limiter à N hippodromes')
    parser.add_argument('--prioritaires-only', action='store_true', 
                        help='Enrichir uniquement les hippodromes principaux')
    
    args = parser.parse_args()
    
    logger.info("=" * 90)
    logger.info("🚀 ENRICHISSEMENT BATCH HIPPODROMES - PHASE 2C")
    logger.info("=" * 90)
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Récupérer tous les hippodromes
    cur.execute("""
        SELECT id_hippodrome, code_pmu, nom_hippodrome, ville
        FROM hippodromes
        ORDER BY nom_hippodrome
    """)
    
    tous_hippodromes = cur.fetchall()
    cur.close()
    conn.close()
    
    # Filtrer hippodromes français
    hippodromes_francais = []
    hippodromes_etrangers = []
    
    for row in tous_hippodromes:
        id_hipp, code, nom, ville = row
        
        # Si déjà ville renseignée, passer
        if ville:
            continue
        
        if est_hippodrome_francais(nom, code):
            # Prioriser les principaux
            if code in HIPPODROMES_PRINCIPAUX:
                hippodromes_francais.insert(0, row)
            else:
                hippodromes_francais.append(row)
        else:
            hippodromes_etrangers.append(row)
    
    logger.info(f"\n📊 STATISTIQUES:")
    logger.info(f"   Hippodromes français      : {len(hippodromes_francais)}")
    logger.info(f"   Hippodromes étrangers     : {len(hippodromes_etrangers)}")
    logger.info(f"   Déjà enrichis (ville OK)  : {len(tous_hippodromes) - len(hippodromes_francais) - len(hippodromes_etrangers)}")
    
    # Filtrer si --prioritaires-only
    if args.prioritaires_only:
        hippodromes_francais = [
            row for row in hippodromes_francais 
            if row[1] in HIPPODROMES_PRINCIPAUX
        ]
        logger.info(f"   Mode: Principaux uniquement ({len(hippodromes_francais)})")
    
    # Appliquer skip/limit
    if args.skip > 0:
        hippodromes_francais = hippodromes_francais[args.skip:]
        logger.info(f"   Saut de {args.skip} hippodromes")
    
    if args.limit:
        hippodromes_francais = hippodromes_francais[:args.limit]
        logger.info(f"   Limitation à {args.limit} hippodromes")
    
    total = len(hippodromes_francais)
    
    if total == 0:
        logger.info("\n✅ Tous les hippodromes français sont déjà enrichis !")
        return
    
    logger.info(f"\n🎯 {total} hippodromes à enrichir")
    logger.info("")
    
    # Enrichir
    scraper = ScraperHippodromes()
    enrichis = 0
    echecs = 0
    
    for i, (id_hipp, code, nom, _) in enumerate(hippodromes_francais, 1):
        logger.info(f"[{i + args.skip}/{total + args.skip}] {nom} ({code})")
        
        try:
            success = scraper.enrich_hippodrome(id_hipp, code, nom)
            
            if success:
                enrichis += 1
            else:
                echecs += 1
        
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Interruption utilisateur (Ctrl+C)")
            logger.info(f"💡 Pour reprendre : --skip {i + args.skip}")
            break
        
        except Exception as e:
            logger.error(f"❌ Erreur inattendue : {e}")
            echecs += 1
        
        # Pause entre hippodromes
        if i < total:
            time.sleep(2)
        
        print()
    
    # Rapport final
    logger.info("=" * 90)
    logger.info("📊 RAPPORT FINAL")
    logger.info("=" * 90)
    logger.info(f"   Hippodromes traités    : {enrichis + echecs}")
    logger.info(f"   Hippodromes enrichis   : {enrichis} ({100*enrichis//(enrichis+echecs) if (enrichis+echecs) > 0 else 0}%)")
    logger.info(f"   Échecs                 : {echecs}")
    logger.info("=" * 90)


if __name__ == '__main__':
    main()
