#!/usr/bin/env python3
"""
Script de génération d'un fichier CSV exemple conforme au format Zone-Turf.
Utile pour tester le scraper.
"""

import csv
from pathlib import Path
from datetime import date

def generate_example_csv(output_path: Path):
    """Génère un CSV d'exemple avec données fictives."""
    
    headers = [
        # Course
        'date_course', 'heure_course', 'reunion', 'course', 'hippodrome', 'code_hippodrome',
        'discipline', 'distance', 'allocation', 'nombre_partants', 'etat_piste', 'corde',
        'categorie_age', 'sexe_condition', 'poids_condition',
        # Cheval
        'cheval', 'sexe', 'age', 'robe', 'origine', 'pere', 'mere', 'eleveur', 'proprietaire',
        # Équipe
        'jockey', 'code_jockey', 'entraineur', 'code_entraineur',
        # Performance
        'numero', 'numero_dossard', 'poids', 'musique', 'deferre', 'oeilleres',
        # Résultat
        'position', 'ecart', 'non_partant', 'disqualifie',
        # Cotes
        'cote_pm', 'cote_sp', 'rapport_gagnant', 'rapport_place',
        # Temps
        'temps', 'vitesse', 'gain'
    ]
    
    # Données exemple pour une course fictive
    rows = [
        # Course 1 - Vincennes R1C1
        [
            '2025-11-11', '14:30', '1', '1', 'Vincennes', 'VINC',
            'TROT', '2700', '50000', '16', 'Bon', 'G',
            '4ans et +', 'Ouvert à tous', 'Poids par âge',
            # Cheval 1 - Gagnant
            'BOLD EAGLE', 'M', '5', 'Bai', 'Trotteur Français', 'Ready Cash', 'Uta du Lupin',
            'Elevage Normandie', 'Ecurie Elite',
            'E. Raffin', 'RAF', 'J.M. Bazire', 'BAZ',
            '1', '1', '65', '1p2p1p1p', '4', '',
            '1', '', 'False', 'False',
            '3.5', '3.2', '3.20', '1.50',
            '74.2', '70.5', '5000'
        ],
        [
            '2025-11-11', '14:30', '1', '1', 'Vincennes', 'VINC',
            'TROT', '2700', '50000', '16', 'Bon', 'G',
            '4ans et +', 'Ouvert à tous', 'Poids par âge',
            # Cheval 2
            'DAVIDSON DU RIB', 'M', '6', 'Alezan', 'Trotteur Français', 'Love You', 'Quille du Rib',
            'Elevage Bretagne', 'Ecurie Champion',
            'M. Abrivard', 'ABR', 'P. Levesque', 'LEV',
            '2', '2', '66', '2p1p3p1p', '4', 'A',
            '2', '2L', 'False', 'False',
            '5.2', '5.8', '', '2.30',
            '74.5', '70.2', '2000'
        ],
        [
            '2025-11-11', '14:30', '1', '1', 'Vincennes', 'VINC',
            'TROT', '2700', '50000', '16', 'Bon', 'G',
            '4ans et +', 'Ouvert à tous', 'Poids par âge',
            # Cheval 3
            'FACE TIME BOURBON', 'M', '7', 'Gris', 'Trotteur Français', 'Ready Cash', 'Royal Dream',
            'Elevage Haras', 'Ecurie Pro',
            'G. Gelormini', 'GEL', 'S. Guarato', 'GUA',
            '3', '3', '67', '1p1p2p3p', '4', '',
            '3', 'tête', 'False', 'False',
            '4.1', '4.5', '', '1.80',
            '74.5', '70.2', '1500'
        ],
        [
            '2025-11-11', '14:30', '1', '1', 'Vincennes', 'VINC',
            'TROT', '2700', '50000', '16', 'Bon', 'G',
            '4ans et +', 'Ouvert à tous', 'Poids par âge',
            # Cheval 4 - Non partant
            'DJANGO RIFF', 'M', '5', 'Bai', 'Trotteur Français', 'Offshore Dream', 'Quenelle de Rêve',
            'Elevage Sud', 'Ecurie Turf',
            'F. Nivard', 'NIV', 'D. Bonne', 'BON',
            '4', '4', '66', '3p4p2p1p', '4', 'O',
            '', '', 'True', 'False',
            '8.5', '', '', '',
            '', '', ''
        ],
        # Course 2 - Vincennes R1C2 (Plat)
        [
            '2025-11-11', '15:10', '1', '2', 'Vincennes', 'VINC',
            'PLAT', '1600', '30000', '12', 'Souple', 'G',
            '3ans', 'Ouvert à tous', 'Référence',
            # Cheval 1
            'ALMANZOR', 'M', '3', 'Bai', 'Pur Sang', 'Wootton Bassett', 'Darkova',
            'Elevage Aga Khan', 'Ecurie Wildenstein',
            'C. Demuro', 'DEM', 'J.C. Rouget', 'ROU',
            '5', '5', '58', '1p1p2p', '', 'A',
            '1', '', 'False', 'False',
            '2.1', '2.3', '2.30', '1.20',
            '95.8', '60.1', '15000'
        ],
    ]
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"✅ Fichier CSV exemple généré : {output_path}")
    print(f"   {len(rows)} lignes de données")
    print(f"   {len(headers)} colonnes")

if __name__ == '__main__':
    output = Path(__file__).parent / 'data' / 'exemple_zoneturf.csv'
    output.parent.mkdir(exist_ok=True)
    generate_example_csv(output)
    
    print(f"\n📝 Pour tester le scraper :")
    print(f"   python scraper_zoneturf.py --csv {output}")
