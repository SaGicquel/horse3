#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scraper PMU - RÉSULTATS & RAPPORTS
Scrape les informations post-course :
- Arrivées définitives (ordre, temps, écarts)
- Rapports PMU (dividendes Gagnant, Placé, Couplé, Trio, Quarté, Quinté)
- Incidents (réclamations, disqualifications, observations)

Usage:
    python scraper_results.py 2025-11-01          # Tous les résultats du jour
    python scraper_results.py 2025-11-01 R1      # Tous les résultats de R1
    python scraper_results.py 2025-11-01 R1 C3   # Résultats de R1C3 uniquement
"""

import sqlite3
import sys
import time
import requests
from datetime import date

# Réutiliser la config du scraper principal
DB_PATH = "data/database.db"

UA = "horse2-enricher/1.4 (+contact: youremail@example.com)"
COMMON_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.pmu.fr/turf/",
    "Connection": "keep-alive",
}

BASE = "https://online.turfinfo.api.pmu.fr/rest/client/7"
FALLBACK_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/7"

def to_pmu_date(dt: str) -> str:
    """'YYYY-MM-DD' -> 'DDMMYYYY'"""
    ds = dt.replace("-", "")
    if len(ds) != 8 or not ds.isdigit():
        raise ValueError(f"Bad date: {dt}")
    yyyy, mm, dd = ds[:4], ds[4:6], ds[6:8]
    return f"{dd}{mm}{yyyy}"

def get_json(url: str, timeout=15):
    """GET JSON robuste"""
    try:
        r = requests.get(url, headers=COMMON_HEADERS, timeout=timeout)
        if r.status_code in (204, 404):
            return None
        r.raise_for_status()
        return r.json()
    except:
        return None

def get_race_id(cur, date_iso, race_code):
    """Récupère le race_id depuis la table races"""
    cur.execute("""
        SELECT race_id FROM races 
        WHERE race_date = ? AND race_code = ?
    """, (date_iso, race_code))
    row = cur.fetchone()
    return row[0] if row else None

def scrape_results(cur, date_iso, reunion, course):
    """
    Scrape les résultats officiels d'une course et met à jour race_participants
    """
    d = to_pmu_date(date_iso)
    race_code = f"R{reunion}C{course}"
    
    # Récupérer le race_id
    race_id = get_race_id(cur, date_iso, race_code)
    if not race_id:
        print(f"  ⚠️  Course {race_code} non trouvée dans la BDD")
        return False
    
    # ========================================
    # 1. RÉCUPÉRER LES PARTICIPANTS (contient les résultats)
    # ========================================
    arrivee_data = None
    for base in (BASE, FALLBACK_BASE):
        # Les résultats sont dans l'endpoint /participants avec ordreArrivee
        url = f"{base}/programme/{d}/R{reunion}/C{course}/participants"
        arrivee_data = get_json(url)
        if arrivee_data and arrivee_data.get("participants"):
            # Vérifier si au moins un participant a un ordreArrivee (= résultats dispo)
            has_results = any(p.get("ordreArrivee") is not None 
                            for p in arrivee_data.get("participants", []))
            if has_results:
                break
            else:
                arrivee_data = None
    
    if not arrivee_data or not arrivee_data.get("participants"):
        print(f"  ⚠️  Pas de résultats disponibles pour {race_code}")
        return False
    
    print(f"  🏁 Traitement arrivée {race_code}...")
    
    # Mettre à jour le statut de la course
    cur.execute("""
        UPDATE races SET status = 'completed' 
        WHERE race_id = ?
    """, (race_id,))
    
    # ========================================
    # 2. METTRE À JOUR LES PARTICIPANTS
    # ========================================
    participants = arrivee_data.get("participants", [])
    updated_count = 0
    
    for p in participants:
        num_pmu = p.get("numPmu")
        if not num_pmu:
            continue
        
        # Extraire toutes les infos de résultat
        ordre_arrivee = p.get("ordreArrivee")  # 1, 2, 3...
        place = p.get("place", {}) or {}
        statut = place.get("statut")  # ARRIVEE, NON_PARTANT, DISQUALIFIE, etc.
        
        # Temps
        temps_dict = p.get("temps") or {}
        temps_str = temps_dict.get("valeur")
        temps_sec = temps_dict.get("secondes")
        
        # Réduction kilométrique
        reduction = p.get("reductionKilometrique")
        if reduction and isinstance(reduction, dict):
            reduction_sec = reduction.get("secondes")
        else:
            reduction_sec = None
        
        # Écarts
        ecarts = p.get("ecarts")
        
        # Gains de la course
        gains_course = p.get("gain")
        
        # Non-partant / Disqualifié
        is_non_runner = statut == "NON_PARTANT"
        is_disqualified = statut == "DISQUALIFIE"
        
        # Observations
        observations = p.get("observations")
        
        # Mettre à jour le participant
        cur.execute("""
            UPDATE race_participants 
            SET finish_position = ?,
                finish_status = ?,
                finish_time_str = ?,
                finish_time_sec = ?,
                reduction_km_sec = ?,
                gaps = ?,
                earnings_race = ?,
                is_non_runner = ?,
                is_disqualified = ?,
                post_race_notes = ?
            WHERE race_id = ? AND horse_num_pmu = ?
        """, (ordre_arrivee, statut, temps_str, temps_sec, reduction_sec,
              ecarts, gains_course, is_non_runner, is_disqualified,
              observations, race_id, num_pmu))
        
        if cur.rowcount > 0:
            updated_count += 1
    
    print(f"    ✓ {updated_count} participants mis à jour")
    
    # ========================================
    # 3. RAPPORTS PMU
    # ========================================
    rapports_data = None
    for base in (BASE, FALLBACK_BASE):
        url = f"{base}/programme/{d}/R{reunion}/C{course}/rapports-definitifs"
        rapports_data = get_json(url)
        if rapports_data:
            break
    
    if not rapports_data:
        print(f"    ℹ️  Pas de rapports pour {race_code}")
        return True
    
    print(f"  💰 Traitement rapports {race_code}...")
    
    # Extraire les différents types de paris
    # Vérifier si rapports_data est une liste ou un dict
    if isinstance(rapports_data, list):
        rapports = rapports_data
    else:
        rapports = rapports_data.get("rapports", []) or []
    
    rapport_count = 0
    
    for rapport in rapports:
        bet_type = rapport.get("typePari")  # SIMPLE_GAGNANT, SIMPLE_PLACE, COUPLE_ORDRE, etc.
        if not bet_type:
            continue
        
        # Combinaisons gagnantes
        combinaisons = rapport.get("combinaisons", []) or []
        
        for combo in combinaisons:
            # Numéros gagnants
            numeros = combo.get("numeros", [])
            winning_combination = "-".join(map(str, numeros)) if numeros else None
            
            # Rapport
            dividend = combo.get("rapport")
            
            # Montant total des enjeux
            pool_total = rapport.get("montantPari")
            
            # Mise de base
            base_stake = rapport.get("miseBase")
            
            # Nombre de gagnants
            num_winners = combo.get("nombreGagnants")
            
            # Insérer dans race_betting
            cur.execute("""
                INSERT INTO race_betting (race_id, bet_type, winning_combination,
                                         dividend, pool_total, base_stake, num_winners)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (race_id, bet_type, winning_combination, dividend,
                  pool_total, base_stake, num_winners))
            
            rapport_count += 1
    
    print(f"    ✓ {rapport_count} rapports enregistrés")
    
    # ========================================
    # 4. INCIDENTS (si disponibles)
    # ========================================
    incidents_data = arrivee_data.get("incidents") or []
    incident_count = 0
    
    for incident in incidents_data:
        incident_type = incident.get("type")
        description = incident.get("description")
        chevaux_concernes = incident.get("chevauxConcernes")
        decision = incident.get("decision")
        
        cur.execute("""
            INSERT INTO race_incidents (race_id, incident_type, description,
                                       affected_horses, stewards_decision)
            VALUES (?, ?, ?, ?, ?)
        """, (race_id, incident_type, description, 
              str(chevaux_concernes) if chevaux_concernes else None, decision))
        
        incident_count += 1
    
    if incident_count > 0:
        print(f"    ⚠️  {incident_count} incidents enregistrés")
    
    return True

def get_races_to_scrape(cur, date_iso, reunion=None):
    """Récupère les courses à scraper depuis la BDD"""
    if reunion:
        cur.execute("""
            SELECT r.race_date, rm.meeting_number, r.race_number
            FROM races r
            JOIN race_meetings rm ON r.meeting_id = rm.meeting_id
            WHERE r.race_date = ? AND rm.meeting_number = ?
            ORDER BY rm.meeting_number, r.race_number
        """, (date_iso, reunion))
    else:
        cur.execute("""
            SELECT r.race_date, rm.meeting_number, r.race_number
            FROM races r
            JOIN race_meetings rm ON r.meeting_id = rm.meeting_id
            WHERE r.race_date = ?
            ORDER BY rm.meeting_number, r.race_number
        """, (date_iso,))
    
    return cur.fetchall()

def main():
    if len(sys.argv) < 2:
        print("Usage: python scraper_results.py <date> [reunion] [course]")
        print("Exemples:")
        print("  python scraper_results.py 2025-11-01")
        print("  python scraper_results.py 2025-11-01 1")
        print("  python scraper_results.py 2025-11-01 1 3")
        return 1
    
    date_iso = sys.argv[1]
    reunion = int(sys.argv[2]) if len(sys.argv) > 2 else None
    course = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    print(f"\n🏁 Scraping des résultats pour {date_iso}")
    if reunion:
        print(f"   Réunion: R{reunion}")
    if course:
        print(f"   Course: C{course}")
    print()
    
    success_count = 0
    error_count = 0
    skip_count = 0
    
    if course and reunion:
        # Une seule course
        try:
            if scrape_results(cur, date_iso, reunion, course):
                success_count += 1
            else:
                skip_count += 1
            con.commit()
        except Exception as e:
            print(f"  ❌ Erreur R{reunion}C{course}: {e}")
            error_count += 1
    else:
        # Récupérer toutes les courses depuis la BDD
        races = get_races_to_scrape(cur, date_iso, reunion)
        
        if not races:
            print(f"❌ Aucune course trouvée pour {date_iso}")
            if reunion:
                print(f"   (Réunion R{reunion} demandée)")
            return 1
        
        print(f"📋 {len(races)} courses à traiter\n")
        
        for race_date, meeting_num, race_num in races:
            try:
                if scrape_results(cur, race_date, meeting_num, race_num):
                    success_count += 1
                else:
                    skip_count += 1
                con.commit()
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"  ❌ Erreur R{meeting_num}C{race_num}: {e}")
                error_count += 1
    
    con.close()
    
    print(f"\n✅ Terminé!")
    print(f"   • {success_count} courses avec résultats")
    print(f"   • {skip_count} courses sans résultats (pas encore courues)")
    print(f"   • {error_count} erreurs")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
