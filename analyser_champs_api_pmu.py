#!/usr/bin/env python3
"""
Script pour analyser les champs disponibles dans l'API PMU
et identifier ceux qui ne sont pas récupérés dans le scraper
"""

import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict

UA = "horse-analyzer/1.0"
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json",
    "Accept-Language": "fr-FR,fr;q=0.9",
}

BASE = "https://online.turfinfo.api.pmu.fr/rest/client/7"

def analyze_api_fields():
    """Analyse les champs disponibles dans l'API PMU"""
    
    print("="*80)
    print("ANALYSE DES CHAMPS DISPONIBLES DANS L'API PMU")
    print("="*80)
    
    # Utiliser une date récente
    date_test = (datetime.now() - timedelta(days=1)).strftime('%d%m%Y')
    
    print(f"\nDate de test: {date_test}")
    print("\n1. Récupération du programme...")
    
    # Récupérer le programme
    url_prog = f"{BASE}/programme/{date_test}"
    resp = requests.get(url_prog, headers=HEADERS, timeout=10)
    
    if resp.status_code != 200:
        print(f"❌ Erreur {resp.status_code}")
        return
    
    prog = resp.json()
    reunions = prog.get("programme", {}).get("reunions", [])
    
    if not reunions:
        print("❌ Pas de réunions trouvées")
        return
    
    print(f"✅ {len(reunions)} réunions trouvées")
    
    # Prendre la première réunion et première course
    reunion = reunions[0]
    reunion_num = reunion.get("numOfficiel")
    courses = reunion.get("courses", [])
    
    if not courses:
        print("❌ Pas de courses")
        return
    
    course = courses[0]
    course_num = course.get("numOrdre")
    
    print(f"\n2. Analyse R{reunion_num}C{course_num}...")
    
    # Récupérer les détails de la course
    url_course = f"{BASE}/programme/{date_test}/R{reunion_num}/C{course_num}"
    resp_course = requests.get(url_course, headers=HEADERS, timeout=10)
    
    if resp_course.status_code != 200:
        print(f"❌ Erreur {resp_course.status_code}")
        return
    
    course_data = resp_course.json()
    
    print("\n" + "="*80)
    print("STRUCTURE DE LA RÉPONSE API")
    print("="*80)
    
    def analyze_structure(data, prefix="", level=0):
        """Analyse récursive de la structure"""
        results = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if value is None:
                    results.append((full_key, "NULL", level))
                elif isinstance(value, (dict, list)):
                    type_str = "dict" if isinstance(value, dict) else f"list[{len(value)}]"
                    results.append((full_key, type_str, level))
                    results.extend(analyze_structure(value, full_key, level + 1))
                else:
                    type_str = type(value).__name__
                    sample = str(value)[:50] if len(str(value)) > 50 else str(value)
                    results.append((full_key, f"{type_str}: {sample}", level))
        
        elif isinstance(data, list) and data:
            # Analyser le premier élément de la liste
            results.extend(analyze_structure(data[0], prefix, level))
        
        return results
    
    # Analyser la structure
    structure = analyze_structure(course_data)
    
    # Grouper par catégorie
    categories = defaultdict(list)
    for field, type_info, level in structure:
        parts = field.split('.')
        category = parts[0] if parts else "root"
        categories[category].append((field, type_info, level))
    
    # Afficher par catégorie
    for category in sorted(categories.keys()):
        print(f"\n📦 {category.upper()}")
        print("-" * 80)
        
        for field, type_info, level in categories[category][:20]:  # Limite à 20 par catégorie
            indent = "  " * level
            print(f"{indent}{field}: {type_info}")
        
        if len(categories[category]) > 20:
            print(f"  ... et {len(categories[category]) - 20} autres champs")
    
    # Analyser les participants
    print("\n" + "="*80)
    print("ANALYSE DES PARTICIPANTS")
    print("="*80)
    
    participants = course_data.get("participants", [])
    if participants:
        print(f"\n{len(participants)} participants trouvés")
        print("\nChamps disponibles pour un participant:")
        print("-" * 80)
        
        # Analyser le premier participant
        part_structure = analyze_structure(participants[0], "participant")
        
        for field, type_info, level in part_structure[:50]:
            indent = "  " * level
            print(f"{indent}{field}: {type_info}")
    
    # Analyser les rapports (si disponibles)
    print("\n" + "="*80)
    print("ANALYSE DES RAPPORTS")
    print("="*80)
    
    url_rapports = f"{BASE}/programme/{date_test}/R{reunion_num}/C{course_num}/rapports-definitifs"
    resp_rapports = requests.get(url_rapports, headers=HEADERS, timeout=10)
    
    if resp_rapports.status_code == 200:
        rapports = resp_rapports.json()
        print("\n✅ Rapports disponibles")
        
        rapport_structure = analyze_structure(rapports, "rapports")
        
        for field, type_info, level in rapport_structure[:30]:
            indent = "  " * level
            print(f"{indent}{field}: {type_info}")
    else:
        print("\n⚠️  Rapports non disponibles (course probablement non terminée)")
    
    # Analyser les performances historiques
    print("\n" + "="*80)
    print("ANALYSE DES PERFORMANCES HISTORIQUES")
    print("="*80)
    
    if participants:
        id_cheval = participants[0].get("idCheval")
        if id_cheval:
            url_perf = f"{BASE}/cheval/{id_cheval}?specialisation=TOUT&nombreHistorique=5"
            resp_perf = requests.get(url_perf, headers=HEADERS, timeout=10)
            
            if resp_perf.status_code == 200:
                perf_data = resp_perf.json()
                print(f"\n✅ Performances disponibles pour cheval {id_cheval}")
                
                perf_structure = analyze_structure(perf_data, "performances")
                
                for field, type_info, level in perf_structure[:40]:
                    indent = "  " * level
                    print(f"{indent}{field}: {type_info}")
    
    print("\n" + "="*80)
    print("ANALYSE TERMINÉE")
    print("="*80)
    
    # Sauvegarder un exemple complet
    output = {
        "date_analyse": datetime.now().isoformat(),
        "course_data": course_data,
        "rapports": rapports if resp_rapports.status_code == 200 else None
    }
    
    with open("exemple_api_pmu.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n💾 Exemple complet sauvegardé dans: exemple_api_pmu.json")

if __name__ == "__main__":
    try:
        analyze_api_fields()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
