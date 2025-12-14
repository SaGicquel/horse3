#!/usr/bin/env python3
"""
ENRICHISSEMENT ZONE-TURF - Phase 2A

Ce script enrichit les données PMU déjà en base avec les données Zone-Turf :
- Temps précis des courses
- Écarts entre chevaux (2L, 1/2L, etc.)
- Musique complète (20 dernières courses)
- Cotes PMU au départ (cote_sp)

Le scraper PMU importe la structure de base, Zone-Turf complète les détails.

Usage:
    python enrichir_zoneturf.py --csv fichier.csv
    python enrichir_zoneturf.py --date 2025-11-11
    python enrichir_zoneturf.py --date-range 2025-11-01 2025-11-11
"""

import argparse
import csv
import re
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sys

from db_connection import get_connection

# Configuration
ZONE_TURF_BASE_URL = "https://www.zone-turf.fr/telechargements/"
UA = "horse3-enrichisseur/1.0 (+contact@example.com)"
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/csv,text/plain,*/*",
}

class EnrichisseurZoneTurf:
    """Enrichit les données PMU avec Zone-Turf."""
    
    def __init__(self):
        self.conn = None
        self.cur = None
        self.stats = {
            'courses_enrichies': 0,
            'performances_enrichies': 0,
            'temps_ajoutes': 0,
            'ecarts_ajoutes': 0,
            'musiques_ajoutees': 0,
            'cotes_ajoutees': 0,
            'courses_introuvables': 0,
            'chevaux_introuvables': 0,
        }
        self.courses_cache = {}  # Cache des courses par clé (date, hippodrome, reunion, course)
    
    def connect_db(self):
        """Connexion à la base de données."""
        self.conn = get_connection()
        self.cur = self.conn.cursor()
        print("✅ Connecté à la base de données")
    
    def close_db(self):
        """Fermeture de la connexion."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
    
    def enrich_from_csv(self, csv_path: Path):
        """
        Enrichit les données depuis un fichier CSV Zone-Turf.
        
        Args:
            csv_path: Chemin vers le fichier CSV
        """
        print(f"\n📄 Lecture du fichier CSV : {csv_path}")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            
            rows_by_course = {}  # Grouper par course
            
            for row in reader:
                # Clé unique de la course
                course_key = self._get_course_key(row)
                if not course_key:
                    continue
                
                if course_key not in rows_by_course:
                    rows_by_course[course_key] = []
                
                rows_by_course[course_key].append(row)
            
            print(f"📊 {len(rows_by_course)} courses trouvées dans le CSV")
            
            # Enrichir chaque course
            for course_key, rows in rows_by_course.items():
                self._enrich_course(course_key, rows)
        
        self.conn.commit()
    
    def _get_course_key(self, row: Dict) -> Optional[str]:
        """
        Génère une clé unique pour identifier une course.
        
        Format: YYYYMMDD_HIPPODROME_R1_C1
        
        Returns:
            Clé de la course ou None si données manquantes
        """
        date = row.get('date', row.get('date_course', ''))
        hippodrome = row.get('hippodrome', row.get('nom_hippodrome', ''))
        reunion = row.get('reunion', row.get('num_reunion', ''))
        course = row.get('course', row.get('num_course', ''))
        
        if not all([date, hippodrome, reunion, course]):
            return None
        
        # Normaliser la date (DD/MM/YYYY -> YYYYMMDD)
        if '/' in date:
            parts = date.split('/')
            if len(parts) == 3:
                date = f"{parts[2]}{parts[1]}{parts[0]}"
        else:
            date = date.replace('-', '')
        
        # Normaliser l'hippodrome (extraire le code)
        code_hippo = self._extract_code_hippodrome(hippodrome)
        
        return f"{date}_{code_hippo}_R{reunion}_C{course}"
    
    def _extract_code_hippodrome(self, hippodrome: str) -> str:
        """
        Extrait un code hippodrome normalisé.
        
        Exemples:
        - "VINCENNES" -> "VINC"
        - "DEAUVILLE" -> "DEAU"
        - "CHANTILLY" -> "CHAN"
        """
        # Codes connus PMU (vérifier en base d'abord)
        codes_connus = {
            'VINCENNES': 'VINC',
            'PARIS-VINCENNES': 'VINC',
            'DEAUVILLE': 'DEAU',
            'CHANTILLY': 'CHAN',
            'LONGCHAMP': 'LONG',
            'AUTEUIL': 'AUTE',
            'SAINT-CLOUD': 'SC',
            'MAISONS-LAFFITTE': 'ML',
            'LYON': 'LYO',
            'LYON-PARILLY': 'LYO',
            'MARSEILLE': 'BOR',
            'MARSEILLE-BORELY': 'BOR',
            'BORDEAUX': 'BOU',
            'BORDEAUX-BOUSCAT': 'BOU',
        }
        
        hippo_upper = hippodrome.upper().strip()
        
        # Chercher code connu
        for nom, code in codes_connus.items():
            if nom in hippo_upper:
                return code
        
        # Chercher en base de données
        if self.cur:
            self.cur.execute("""
                SELECT code_pmu FROM hippodromes 
                WHERE UPPER(nom_hippodrome) LIKE %s
                LIMIT 1
            """, (f'%{hippo_upper}%',))
            result = self.cur.fetchone()
            if result:
                return result[0]
        
        # Sinon, prendre les 3-4 premières lettres
        clean = re.sub(r'[^A-Z]', '', hippo_upper)
        return clean[:4] if len(clean) >= 4 else clean[:3]
    
    def _enrich_course(self, course_key: str, rows: List[Dict]):
        """
        Enrichit une course spécifique avec ses performances.
        
        Args:
            course_key: Clé unique de la course
            rows: Liste des lignes CSV pour cette course
        """
        # Chercher la course en base
        id_course = self._find_course_in_db(course_key)
        
        if not id_course:
            self.stats['courses_introuvables'] += 1
            print(f"  ⚠️  Course introuvable : {course_key}")
            return
        
        print(f"  📝 Enrichissement course : {course_key} ({len(rows)} chevaux)")
        
        # Enrichir les métadonnées de la course
        course_data = rows[0]  # Prendre la première ligne pour les infos course
        self._update_course_metadata(id_course, course_data)
        
        # Enrichir chaque performance (cheval)
        for row in rows:
            self._enrich_performance(id_course, row)
        
        self.stats['courses_enrichies'] += 1
    
    def _find_course_in_db(self, course_key: str) -> Optional[str]:
        """
        Cherche une course en base par sa clé.
        
        Args:
            course_key: Format YYYYMMDD_CODE_R1_C1
        
        Returns:
            id_course ou None
        """
        # Cache
        if course_key in self.courses_cache:
            return self.courses_cache[course_key]
        
        # Chercher par id_course exact
        self.cur.execute("""
            SELECT id_course FROM courses WHERE id_course = %s
        """, (course_key,))
        
        result = self.cur.fetchone()
        if result:
            self.courses_cache[course_key] = result[0]
            return result[0]
        
        # Chercher par pattern (au cas où le code hippodrome diffère)
        parts = course_key.split('_')
        if len(parts) >= 3:
            date = parts[0]
            reunion = parts[2]
            course = parts[3] if len(parts) > 3 else parts[2]
            
            # Format date YYYYMMDD -> YYYY-MM-DD
            date_iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            
            self.cur.execute("""
                SELECT id_course FROM courses 
                WHERE date_course = %s 
                AND id_course LIKE %s
                LIMIT 1
            """, (date_iso, f"%_{reunion}_{course}"))
            
            result = self.cur.fetchone()
            if result:
                self.courses_cache[course_key] = result[0]
                return result[0]
        
        return None
    
    def _update_course_metadata(self, id_course: str, data: Dict):
        """
        Met à jour les métadonnées de la course.
        
        Champs enrichis :
        - corde (main/droite)
        - etat_piste
        - meteo
        - temperature_c
        - vent_kmh
        """
        updates = []
        params = []
        
        # Corde
        if corde := data.get('corde'):
            updates.append("corde = %s")
            params.append(corde)
        
        # État piste
        if etat_piste := data.get('terrain', data.get('etat_piste')):
            updates.append("etat_piste = %s")
            params.append(etat_piste)
        
        # Météo
        if meteo := data.get('meteo'):
            updates.append("meteo = %s")
            params.append(meteo)
        
        # Température
        if temp := self._parse_float(data.get('temperature')):
            updates.append("temperature_c = %s")
            params.append(temp)
        
        # Vent
        if vent := self._parse_float(data.get('vent')):
            updates.append("vent_kmh = %s")
            params.append(vent)
        
        if not updates:
            return
        
        # UPDATE
        params.append(id_course)
        query = f"""
            UPDATE courses 
            SET {', '.join(updates)}
            WHERE id_course = %s
        """
        
        self.cur.execute(query, params)
    
    def _enrich_performance(self, id_course: str, data: Dict):
        """
        Enrichit une performance (cheval dans la course).
        
        Champs enrichis :
        - musique (20 dernières courses)
        - ecart (2L, 1/2L, etc.)
        - temps_total (secondes)
        - vitesse_moyenne (km/h)
        - cote_sp (cote au départ)
        - cote_pm (cote PMU)
        """
        # Identifier le cheval dans cette course
        numero_corde = self._parse_int(data.get('numero', data.get('numero_corde', data.get('corde'))))
        nom_cheval = data.get('cheval', data.get('nom_cheval', ''))
        
        if not numero_corde:
            return
        
        # Chercher la performance
        id_performance = self._find_performance(id_course, numero_corde, nom_cheval)
        
        if not id_performance:
            self.stats['chevaux_introuvables'] += 1
            return
        
        updates = []
        params = []
        
        # Musique
        if musique := data.get('musique'):
            updates.append("musique = %s")
            params.append(musique)
            self.stats['musiques_ajoutees'] += 1
        
        # Écart
        if ecart := data.get('ecart'):
            updates.append("ecart = %s")
            params.append(ecart)
            self.stats['ecarts_ajoutes'] += 1
        
        # Temps
        if temps := self._parse_float(data.get('temps', data.get('temps_total'))):
            updates.append("temps_total = %s")
            params.append(temps)
            self.stats['temps_ajoutes'] += 1
        
        # Vitesse
        if vitesse := self._parse_float(data.get('vitesse', data.get('vitesse_moyenne'))):
            updates.append("vitesse_moyenne = %s")
            params.append(vitesse)
        
        # Cote départ
        if cote_sp := self._parse_float(data.get('cote', data.get('cote_sp', data.get('cote_depart')))):
            updates.append("cote_sp = %s")
            params.append(cote_sp)
            self.stats['cotes_ajoutees'] += 1
        
        # Cote PMU
        if cote_pm := self._parse_float(data.get('cote_pm', data.get('cote_ouverture'))):
            updates.append("cote_pm = %s")
            params.append(cote_pm)
        
        if not updates:
            return
        
        # UPDATE
        params.append(id_performance)
        query = f"""
            UPDATE performances 
            SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
            WHERE id_performance = %s
        """
        
        self.cur.execute(query, params)
        self.stats['performances_enrichies'] += 1
    
    def _find_performance(self, id_course: str, numero_corde: int, nom_cheval: str) -> Optional[int]:
        """
        Cherche une performance en base.
        
        Args:
            id_course: ID de la course
            numero_corde: Numéro de corde
            nom_cheval: Nom du cheval (fallback)
        
        Returns:
            id_performance ou None
        """
        # Chercher par course + numéro corde
        self.cur.execute("""
            SELECT id_performance FROM performances 
            WHERE id_course = %s AND numero_corde = %s
        """, (id_course, numero_corde))
        
        result = self.cur.fetchone()
        if result:
            return result[0]
        
        # Fallback : chercher par nom cheval
        if nom_cheval:
            self.cur.execute("""
                SELECT p.id_performance 
                FROM performances p
                JOIN chevaux c ON p.id_cheval = c.id_cheval
                WHERE p.id_course = %s AND c.nom_cheval = %s
            """, (id_course, nom_cheval))
            
            result = self.cur.fetchone()
            if result:
                return result[0]
        
        return None
    
    def _parse_int(self, value) -> Optional[int]:
        """Parse un entier."""
        if not value or value == '':
            return None
        try:
            return int(str(value).replace(' ', '').replace(',', ''))
        except (ValueError, AttributeError):
            return None
    
    def _parse_float(self, value) -> Optional[float]:
        """Parse un float."""
        if not value or value == '':
            return None
        try:
            return float(str(value).replace(' ', '').replace(',', '.'))
        except (ValueError, AttributeError):
            return None
    
    def download_zoneturf_csv(self, date: str) -> Optional[Path]:
        """
        Télécharge le CSV Zone-Turf pour une date donnée.
        
        Args:
            date: Format YYYY-MM-DD
        
        Returns:
            Path vers le fichier téléchargé ou None si erreur
        """
        # Format URL Zone-Turf : https://www.zone-turf.fr/telechargements/YYYYMMDD.csv
        date_str = date.replace('-', '')
        url = f"{ZONE_TURF_BASE_URL}{date_str}.csv"
        
        print(f"📥 Téléchargement : {url}")
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            
            # Sauvegarder dans un fichier temporaire
            output_path = Path(f"/tmp/zoneturf_{date_str}.csv")
            output_path.write_bytes(response.content)
            
            print(f"✅ Téléchargé : {output_path}")
            return output_path
            
        except requests.RequestException as e:
            print(f"❌ Erreur téléchargement : {e}")
            return None
    
    def enrich_date(self, date: str):
        """
        Enrichit les données pour une date donnée.
        
        Args:
            date: Format YYYY-MM-DD ou 'today'
        """
        if date.lower() == 'today':
            date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n📅 Enrichissement de la date : {date}")
        
        # Télécharger le CSV
        csv_path = self.download_zoneturf_csv(date)
        
        if not csv_path:
            print("❌ Impossible de télécharger le CSV")
            return
        
        # Enrichir depuis le CSV
        self.enrich_from_csv(csv_path)
        
        # Nettoyer le fichier temporaire
        csv_path.unlink()
    
    def enrich_date_range(self, start_date: str, end_date: str):
        """
        Enrichit les données sur une plage de dates.
        
        Args:
            start_date: Format YYYY-MM-DD
            end_date: Format YYYY-MM-DD
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            self.enrich_date(date_str)
            current += timedelta(days=1)
    
    def show_stats(self):
        """Affiche les statistiques d'enrichissement."""
        print("\n" + "=" * 70)
        print("📊 STATISTIQUES D'ENRICHISSEMENT")
        print("=" * 70)
        print(f"   courses_enrichies         : {self.stats['courses_enrichies']:6d}")
        print(f"   performances_enrichies    : {self.stats['performances_enrichies']:6d}")
        print(f"   temps_ajoutes             : {self.stats['temps_ajoutes']:6d}")
        print(f"   ecarts_ajoutes            : {self.stats['ecarts_ajoutes']:6d}")
        print(f"   musiques_ajoutees         : {self.stats['musiques_ajoutees']:6d}")
        print(f"   cotes_ajoutees            : {self.stats['cotes_ajoutees']:6d}")
        print(f"   courses_introuvables      : {self.stats['courses_introuvables']:6d}")
        print(f"   chevaux_introuvables      : {self.stats['chevaux_introuvables']:6d}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Enrichir données PMU avec Zone-Turf')
    parser.add_argument('--csv', type=str, help='Chemin vers fichier CSV Zone-Turf')
    parser.add_argument('--date', type=str, help='Date ISO (YYYY-MM-DD) ou "today"')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                       help='Plage de dates (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    enrichisseur = EnrichisseurZoneTurf()
    enrichisseur.connect_db()
    
    try:
        if args.csv:
            # Mode fichier CSV direct
            csv_path = Path(args.csv)
            if not csv_path.exists():
                print(f"❌ Fichier introuvable : {csv_path}")
                sys.exit(1)
            
            enrichisseur.enrich_from_csv(csv_path)
        
        elif args.date:
            # Mode date unique
            enrichisseur.enrich_date(args.date)
        
        elif args.date_range:
            # Mode plage de dates
            enrichisseur.enrich_date_range(args.date_range[0], args.date_range[1])
        
        else:
            print("❌ Vous devez spécifier --csv, --date ou --date-range")
            parser.print_help()
            sys.exit(1)
        
        enrichisseur.show_stats()
        
    finally:
        enrichisseur.close_db()
    
    print("\n✅ Enrichissement terminé avec succès !")


if __name__ == '__main__':
    main()
