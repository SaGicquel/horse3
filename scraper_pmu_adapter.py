#!/usr/bin/env python3
"""
ADAPTATEUR SCRAPER PMU -> NOUVEAU SCHÉMA
Transforme les données de scraper_pmu_simple.py vers le nouveau schéma normalisé.

Usage:
    python scraper_pmu_adapter.py --date 2025-11-11
    python scraper_pmu_adapter.py --date today
    python scraper_pmu_adapter.py --date-range 2025-11-01 2025-11-10
"""

import sys
import argparse
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import re

# Import du scraper PMU existant
from scraper_pmu_simple import (
    discover_reunions,
    discover_courses,
    get_json,
    fetch_participants,
    fetch_performances,
    to_pmu_date,
    map_sexe,
    BASE,
    FALLBACK_BASE
)

from db_connection import get_connection

class PMUToSchemaAdapter:
    """Adapte les données PMU vers le nouveau schéma."""
    
    def __init__(self):
        self.conn = None
        self.cur = None
        self.stats = {
            'hippodromes': 0,
            'courses': 0,
            'chevaux': 0,
            'personnes': 0,
            'performances': 0,
        }
        # Cache pour éviter les doublons
        self.cache_hippodromes = {}
        self.cache_chevaux = {}
        self.cache_personnes = {}
    
    def connect_db(self):
        """Connexion à la base."""
        self.conn = get_connection()
        self.cur = self.conn.cursor()
    
    def close_db(self):
        """Fermeture connexion."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
    
    def resolve_hippodrome_identifiers(self, hippo_data):
        """Extrait code et nom depuis les données hippodrome."""
        if not hippo_data:
            return None, None
        
        if isinstance(hippo_data, dict):
            code = (
                hippo_data.get("code") or
                hippo_data.get("codeHippodrome") or
                hippo_data.get("codeLieu") or
                hippo_data.get("libelleCourt")
            )
            name = (
                hippo_data.get("libelleLong") or
                hippo_data.get("libelleCourt") or
                hippo_data.get("nom") or
                hippo_data.get("libelle")
            )
        else:
            code = str(hippo_data).strip()
            name = code
        
        code_norm = code.strip().upper() if code else None
        name_norm = name.strip() if name else None
        return code_norm, name_norm
    
    def get_or_create_hippodrome(self, hippo_data: dict, date_course: str) -> int:
        """
        Récupère ou crée un hippodrome depuis les données PMU.
        
        Args:
            hippo_data: Données hippodrome PMU
            date_course: Date pour générer l'ID
        
        Returns:
            id_hippodrome
        """
        code, nom = self.resolve_hippodrome_identifiers(hippo_data)
        
        if not code and not nom:
            return None
        
        # Utiliser le code comme clé de cache
        cache_key = code or nom
        if cache_key in self.cache_hippodromes:
            return self.cache_hippodromes[cache_key]
        
        # Normaliser
        if not code:
            code = ''.join(c for c in nom.upper() if c.isalpha())[:4]
        if not nom:
            nom = code
        
        # Extraire pays depuis hippo_data
        pays = 'FR'
        if isinstance(hippo_data, dict):
            pays = hippo_data.get('pays') or hippo_data.get('codePays') or 'FR'
        
        # Chercher existant
        self.cur.execute("""
            SELECT id_hippodrome FROM hippodromes 
            WHERE code_pmu = %s OR nom_hippodrome = %s
        """, (code, nom))
        
        result = self.cur.fetchone()
        if result:
            self.cache_hippodromes[cache_key] = result[0]
            return result[0]
        
        # Créer nouveau
        self.cur.execute("""
            INSERT INTO hippodromes (nom_hippodrome, code_pmu, pays)
            VALUES (%s, %s, %s)
            ON CONFLICT (code_pmu) DO UPDATE 
            SET nom_hippodrome = EXCLUDED.nom_hippodrome
            RETURNING id_hippodrome
        """, (nom, code, pays))
        
        id_hippodrome = self.cur.fetchone()[0]
        self.conn.commit()
        self.stats['hippodromes'] += 1
        self.cache_hippodromes[cache_key] = id_hippodrome
        
        return id_hippodrome
    
    def get_or_create_cheval(self, cheval_data: dict) -> int:
        """
        Récupère ou crée un cheval depuis les données PMU.
        
        Args:
            cheval_data: Données cheval PMU
        
        Returns:
            id_cheval
        """
        nom = cheval_data.get('nom')
        if not nom:
            return None
        
        # Extraire infos
        sexe = map_sexe(cheval_data.get('sexe')) or 'M'
        an_naissance = None
        
        # Chercher année de naissance
        if cheval_data.get('age'):
            current_year = datetime.now().year
            an_naissance = current_year - int(cheval_data['age'])
        elif cheval_data.get('dateNaissance'):
            try:
                dn = datetime.strptime(str(cheval_data['dateNaissance']), '%Y-%m-%d')
                an_naissance = dn.year
            except:
                pass
        
        if not an_naissance:
            # Par défaut, estimer l'année
            an_naissance = datetime.now().year - 4
        
        # Cache key
        cache_key = f"{nom}_{an_naissance}_{sexe}"
        if cache_key in self.cache_chevaux:
            return self.cache_chevaux[cache_key]
        
        # Chercher existant
        self.cur.execute("""
            SELECT id_cheval FROM chevaux 
            WHERE nom_cheval = %s AND an_naissance = %s AND sexe_cheval = %s
        """, (nom, an_naissance, sexe))
        
        result = self.cur.fetchone()
        if result:
            self.cache_chevaux[cache_key] = result[0]
            return result[0]
        
        # Extraire infos complémentaires
        robe_data = cheval_data.get('robe')
        if isinstance(robe_data, dict):
            robe = robe_data.get('libelleLong') or robe_data.get('libelleCourt')
        else:
            robe = robe_data
        
        origine = cheval_data.get('race')
        nom_pere = cheval_data.get('pere') or cheval_data.get('nomPere')
        nom_mere = cheval_data.get('mere') or cheval_data.get('nomMere')
        proprietaire = cheval_data.get('proprietaire')
        eleveur = cheval_data.get('eleveur')
        
        # Créer nouveau
        self.cur.execute("""
            INSERT INTO chevaux (
                nom_cheval, sexe_cheval, an_naissance,
                robe, origine, nom_pere, nom_mere, 
                proprietaire, eleveur
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id_cheval
        """, (nom, sexe, an_naissance, robe, origine, nom_pere, nom_mere, 
              proprietaire, eleveur))
        
        id_cheval = self.cur.fetchone()[0]
        self.conn.commit()
        self.stats['chevaux'] += 1
        self.cache_chevaux[cache_key] = id_cheval
        
        return id_cheval
    
    def get_or_create_personne(self, nom: str, type_personne: str, 
                              personne_data: dict = None) -> int:
        """
        Récupère ou crée une personne (jockey/entraîneur).
        
        Args:
            nom: Nom complet
            type_personne: 'JOCKEY' ou 'ENTRAINEUR'
            personne_data: Données optionnelles PMU
        
        Returns:
            id_personne
        """
        if not nom:
            return None
        
        # Cache key
        cache_key = f"{type_personne}_{nom}"
        if cache_key in self.cache_personnes:
            return self.cache_personnes[cache_key]
        
        # Chercher existant
        self.cur.execute("""
            SELECT id_personne FROM personnes 
            WHERE nom_complet = %s AND type = %s
        """, (nom, type_personne))
        
        result = self.cur.fetchone()
        if result:
            self.cache_personnes[cache_key] = result[0]
            return result[0]
        
        # Extraire code PMU si disponible
        code_pmu = None
        if personne_data:
            code_pmu = personne_data.get('id') or personne_data.get('numPmu')
        
        # Créer nouveau
        self.cur.execute("""
            INSERT INTO personnes (nom_complet, type, code_pmu)
            VALUES (%s, %s, %s)
            RETURNING id_personne
        """, (nom, type_personne, code_pmu))
        
        id_personne = self.cur.fetchone()[0]
        self.conn.commit()
        self.stats['personnes'] += 1
        self.cache_personnes[cache_key] = id_personne
        
        return id_personne
    
    def create_course(self, course_data: dict, date_iso: str, reunion: int, course: int) -> str:
        """
        Crée une course depuis les données PMU.
        
        Returns:
            id_course
        """
        # Récupérer l'hippodrome
        hippo_data = course_data.get('hippodrome') or course_data.get('reunion', {}).get('hippodrome')
        id_hippodrome = self.get_or_create_hippodrome(hippo_data, date_iso)
        
        # Générer ID unique
        code_hippo = 'XXXX'
        if id_hippodrome:
            self.cur.execute("SELECT code_pmu FROM hippodromes WHERE id_hippodrome = %s", (id_hippodrome,))
            result = self.cur.fetchone()
            if result:
                code_hippo = result[0]
        
        id_course = f"{date_iso.replace('-', '')}_{code_hippo}_R{reunion}_C{course}"
        
        # Extraire infos course
        discipline_raw = course_data.get('discipline') or course_data.get('typeCourse') or 'PLAT'
        discipline_map = {
            'PLAT': 'Plat',
            'TROT': 'Trot',
            'ATTELE': 'Trot',
            'MONTE': 'Trot',
            'HAIES': 'Obstacle',
            'STEEPLE': 'Obstacle',
            'CROSS': 'Obstacle',
            'OBSTACLE': 'Obstacle',
        }
        discipline = discipline_map.get(discipline_raw.upper(), 'Plat')
        
        # Extraire autres infos
        distance = course_data.get('distance')
        allocation = course_data.get('montantPrix') or course_data.get('allocation')
        nombre_partants = course_data.get('nombreDeclaresPartants') or len(course_data.get('participants', []))
        
        # Heure
        heure_course = None
        if course_data.get('heureDepart'):
            try:
                heure_course = datetime.strptime(str(course_data['heureDepart']), '%H:%M:%S').time()
            except:
                pass
        
        # État piste
        etat_piste_raw = course_data.get('penetrometre') or course_data.get('etatPiste')
        if isinstance(etat_piste_raw, dict):
            etat_piste = etat_piste_raw.get('intitule') or etat_piste_raw.get('commentaire')
        else:
            etat_piste = etat_piste_raw
        
        # Statut
        statut = 'PREVUE'
        if course_data.get('estTerminee') or course_data.get('arriveeDefinitive'):
            statut = 'TERMINEE'
        elif course_data.get('participants'):
            # Vérifier si au moins un participant a un ordre d'arrivée
            for p in course_data.get('participants', []):
                if p.get('ordreArrivee'):
                    statut = 'TERMINEE'
                    break
        
        # Insérer
        self.cur.execute("""
            INSERT INTO courses (
                id_course, date_course, heure_course,
                num_reunion, num_course, id_hippodrome,
                discipline, distance, allocation, nombre_partants,
                etat_piste, statut
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id_course) DO UPDATE SET
                nombre_partants = EXCLUDED.nombre_partants,
                statut = EXCLUDED.statut,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id_course
        """, (
            id_course, date_iso, heure_course,
            reunion, course, id_hippodrome,
            discipline, distance, allocation, nombre_partants,
            etat_piste, statut
        ))
        
        self.conn.commit()
        self.stats['courses'] += 1
        
        return id_course
    
    def create_performance(self, id_course: str, participant_data: dict) -> int:
        """
        Crée une performance depuis les données PMU.
        
        Returns:
            id_performance
        """
        # Récupérer/créer les entités
        cheval_data = {
            'nom': participant_data.get('nom'),
            'sexe': participant_data.get('sexe'),
            'age': participant_data.get('age'),
            'robe': participant_data.get('robe'),
            'race': participant_data.get('race'),
            'pere': participant_data.get('nomPere'),
            'mere': participant_data.get('nomMere'),
            'proprietaire': participant_data.get('proprietaire'),
            'eleveur': participant_data.get('eleveur'),
        }
        id_cheval = self.get_or_create_cheval(cheval_data)
        
        # Jockey
        nom_jockey = participant_data.get('driver') or participant_data.get('jockey')
        id_jockey = self.get_or_create_personne(nom_jockey, 'JOCKEY') if nom_jockey else None
        
        # Entraîneur
        nom_entraineur = participant_data.get('entraineur')
        id_entraineur = self.get_or_create_personne(nom_entraineur, 'ENTRAINEUR') if nom_entraineur else None
        
        # Extraire infos performance
        numero_corde = participant_data.get('numero') or participant_data.get('numPmu')
        
        # Poids (handicapPoids est en dixièmes de kg)
        poids_porte_raw = participant_data.get('handicapPoids')
        poids_porte = (poids_porte_raw / 10.0) if poids_porte_raw else None
        
        # Cotes - NOUVEAUTÉ : Extraire cote_pm depuis dernierRapportDirect
        cote_pm = None
        dernier_rapport_direct = participant_data.get('dernierRapportDirect')
        if dernier_rapport_direct and isinstance(dernier_rapport_direct, dict):
            cote_pm = dernier_rapport_direct.get('rapport')
        
        # Cote SP depuis dernierRapportReference (ou fallback rapportProbable)
        cote_sp = None
        dernier_rapport_ref = participant_data.get('dernierRapportReference')
        if dernier_rapport_ref and isinstance(dernier_rapport_ref, dict):
            cote_sp = dernier_rapport_ref.get('rapport')
        if not cote_sp:
            cote_sp = participant_data.get('rapportProbable') or participant_data.get('cote')
        
        # Résultat
        position_arrivee = participant_data.get('ordreArrivee') or participant_data.get('place')
        non_partant = participant_data.get('nonPartant', False)
        disqualifie = participant_data.get('disqualifie', False)
        
        # NOUVEAUTÉ : Statut participant
        statut = participant_data.get('statut')  # PARTANT / NON_PARTANT
        if statut == 'NON_PARTANT':
            non_partant = True
        
        # Musique (historique) - Priorité : musique_enrichie > musique > historique
        musique = participant_data.get('musique_enrichie') or participant_data.get('musique')
        if not musique and participant_data.get('historiqueCourses'):
            # Fallback : Construire musique depuis historique si pas fournie
            hist = participant_data['historiqueCourses'][:5]  # 5 dernières
            musique_parts = []
            for h in hist:
                place = h.get('ordreArrivee') or h.get('place')
                if place:
                    musique_parts.append(str(place))
            if musique_parts:
                musique = 'p'.join(musique_parts)
        
        # Équipement (tronquer ou mapper pour respecter les contraintes BDD)
        deferre_raw = participant_data.get('deferre')
        if deferre_raw:
            # Mapper les valeurs PMU aux codes courts
            deferre_map = {
                'DEFERRE_TOTAL': '4',
                'DEFERRE_POSTERIEURS': '2AR',
                'DEFERRE_ANTERIEURS': '2AV',
                'FERRE_TOTAL': '',
                'FERRE': ''
            }
            deferre = deferre_map.get(str(deferre_raw), str(deferre_raw)[:5])
        else:
            deferre = None
        
        oeilleres_raw = participant_data.get('oeilleres')
        if oeilleres_raw:
            # Mapper les valeurs PMU
            oeilleres_map = {
                'SANS_OEILLERES': 'S',
                'AVEC_OEILLERES': 'A',
                'OEILLERES_RETIREES': 'O',
                'OEILLERES_AUSTRALIENNES': 'AO'
            }
            oeilleres = oeilleres_map.get(str(oeilleres_raw), str(oeilleres_raw)[:20])
        else:
            oeilleres = None
        
        # Gains - NOUVEAUTÉ : Extraire gains_carriere complets
        gains_carriere = None
        gains_data = participant_data.get('gainsParticipant') or participant_data.get('gains')
        if isinstance(gains_data, dict):
            gains_carriere = gains_data.get('gainsCarriere')
            gain_course = gains_data.get('gainsAnneeEnCours')
        else:
            gains_carriere = participant_data.get('gainsCarriere')
            gain_course = participant_data.get('gain')
        
        # NOUVEAUTÉ : Statistiques complètes
        nombre_courses = participant_data.get('nombreCourses')
        nombre_victoires = participant_data.get('nombreVictoires')
        nombre_places = participant_data.get('nombrePlaces')
        nombre_places_second = participant_data.get('nombrePlacesSecond')
        nombre_places_troisieme = participant_data.get('nombrePlacesTroisieme')
        
        # Rapport
        rapport_gagnant = None
        rapport_place = None
        if position_arrivee == 1:
            rapport_gagnant = participant_data.get('rapportDirect')
        if position_arrivee and position_arrivee <= 3:
            rapport_place = participant_data.get('rapportPlace')
        
        # Temps et vitesse (si disponibles)
        temps_total = None
        vitesse_moyenne = None
        
        # Extraire tempsObtenu de l'API PMU (en millisecondes)
        temps_obtenu_ms = participant_data.get('tempsObtenu')
        if temps_obtenu_ms:
            try:
                temps_total = float(temps_obtenu_ms) / 1000.0  # Convertir ms en secondes
            except:
                pass
        
        # Fallback : chercher dans tempsCourse
        if not temps_total and participant_data.get('tempsCourse'):
            try:
                temps_total = float(participant_data['tempsCourse'])
            except:
                pass
        
        # Calculer vitesse si on a temps + distance
        if temps_total and temps_total > 0:
            # Récupérer la distance de la course
            self.cur.execute("SELECT distance FROM courses WHERE id_course = %s", (id_course,))
            result = self.cur.fetchone()
            if result and result[0]:
                distance_m = result[0]
                # Vitesse = distance (m) / temps (s) * 3.6 pour km/h
                vitesse_moyenne = (distance_m / temps_total) * 3.6
        
        # Insérer - AVEC TOUS LES NOUVEAUX CHAMPS
        self.cur.execute("""
            INSERT INTO performances (
                id_course, id_cheval, id_jockey, id_entraineur,
                numero_corde, poids_porte,
                cote_pm, cote_sp,
                position_arrivee, disqualifie, non_partant,
                musique, deferre, oeilleres,
                gain_course, gains_carriere,
                nombre_courses, nombre_victoires, nombre_places,
                nombre_places_second, nombre_places_troisieme,
                rapport_gagnant, rapport_place,
                temps_total, vitesse_moyenne
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id_course, numero_corde) DO UPDATE SET
                position_arrivee = EXCLUDED.position_arrivee,
                cote_pm = EXCLUDED.cote_pm,
                cote_sp = EXCLUDED.cote_sp,
                musique = EXCLUDED.musique,
                deferre = EXCLUDED.deferre,
                oeilleres = EXCLUDED.oeilleres,
                gains_carriere = EXCLUDED.gains_carriere,
                nombre_courses = EXCLUDED.nombre_courses,
                nombre_victoires = EXCLUDED.nombre_victoires,
                nombre_places = EXCLUDED.nombre_places,
                nombre_places_second = EXCLUDED.nombre_places_second,
                nombre_places_troisieme = EXCLUDED.nombre_places_troisieme,
                rapport_gagnant = EXCLUDED.rapport_gagnant,
                rapport_place = EXCLUDED.rapport_place,
                temps_total = EXCLUDED.temps_total,
                vitesse_moyenne = EXCLUDED.vitesse_moyenne,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id_performance
        """, (
            id_course, id_cheval, id_jockey, id_entraineur,
            numero_corde, poids_porte,
            cote_pm, cote_sp,
            position_arrivee, disqualifie, non_partant,
            musique, deferre, oeilleres,
            gain_course, gains_carriere,
            nombre_courses, nombre_victoires, nombre_places,
            nombre_places_second, nombre_places_troisieme,
            rapport_gagnant, rapport_place,
            temps_total, vitesse_moyenne
        ))
        
        id_performance = self.cur.fetchone()[0]
        self.conn.commit()
        self.stats['performances'] += 1
        
        return id_performance
    
    def scrape_date(self, date_iso: str):
        """Scrape toutes les courses d'une date."""
        print(f"\n📅 Scraping du {date_iso}...")
        
        # Découvrir les réunions
        reunions = discover_reunions(date_iso)
        if not reunions:
            print(f"   ⚠️  Aucune réunion trouvée pour {date_iso}")
            return
        
        print(f"   ✅ {len(reunions)} réunions trouvées : R{', R'.join(map(str, reunions))}")
        
        # Pour chaque réunion
        for reunion in reunions:
            courses = discover_courses(date_iso, reunion)
            if not courses:
                continue
            
            print(f"\n   🏇 Réunion R{reunion} : {len(courses)} courses")
            
            # Pour chaque course
            for course_num in courses:
                try:
                    self.scrape_course(date_iso, reunion, course_num)
                except Exception as e:
                    print(f"      ❌ Erreur R{reunion}C{course_num} : {e}")
                    self.conn.rollback()  # Annuler la transaction en erreur
    
    def scrape_course(self, date_iso: str, reunion: int, course: int):
        """Scrape une course spécifique."""
        pmu_date = to_pmu_date(date_iso)
        
        # Récupérer les données de la course (métadonnées)
        for base in (BASE, FALLBACK_BASE):
            url = f"{base}/programme/{pmu_date}/R{reunion}/C{course}"
            data = get_json(url)
            
            if data:
                # Créer la course
                id_course = self.create_course(data, date_iso, reunion, course)
                
                # Récupérer les participants via l'endpoint spécialisé
                participants = fetch_participants(date_iso, reunion, course)
                
                if not participants:
                    print(f"      ⚠️  R{reunion}C{course} : pas de participants")
                    return
                
                # Récupérer les données de performance (musique, stats)
                perfs_data = fetch_performances(date_iso, reunion, course, self.cur, None, max_hist=20)
                
                # Créer les performances
                nb_perfs = 0
                for participant in participants:
                    try:
                        # Enrichir participant avec musique si disponible (utiliser nom normalisé)
                        from scraper_pmu_simple import norm
                        nom_cheval = participant.get('nom', '')
                        if nom_cheval and perfs_data:
                            nom_norm = norm(nom_cheval)
                            if nom_norm and nom_norm in perfs_data:
                                musique = perfs_data[nom_norm].get('musique')
                                participant['musique_enrichie'] = musique
                        
                        self.create_performance(id_course, participant)
                        nb_perfs += 1
                    except Exception as e:
                        print(f"         ⚠️  Erreur participant {participant.get('nom')}: {e}")
                        self.conn.rollback()  # Annuler la transaction en erreur
                
                print(f"      ✅ R{reunion}C{course} : {nb_perfs} participants")
                return
        
        print(f"      ⚠️  R{reunion}C{course} : aucune donnée disponible")
    
    def show_stats(self):
        """Affiche les statistiques."""
        print("\n" + "=" * 70)
        print("📊 STATISTIQUES D'IMPORT")
        print("=" * 70)
        for key, value in self.stats.items():
            print(f"   {key:20s} : {value:6d} créés")
        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Adaptateur Scraper PMU vers nouveau schéma')
    parser.add_argument('--date', type=str, help='Date ISO (YYYY-MM-DD) ou "today"')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                       help='Plage de dates (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if not any([args.date, args.date_range]):
        parser.print_help()
        sys.exit(1)
    
    adapter = PMUToSchemaAdapter()
    adapter.connect_db()
    
    try:
        if args.date:
            if args.date.lower() == 'today':
                date_iso = datetime.now().strftime('%Y-%m-%d')
            else:
                date_iso = args.date
            
            adapter.scrape_date(date_iso)
        
        elif args.date_range:
            start_date = datetime.strptime(args.date_range[0], '%Y-%m-%d')
            end_date = datetime.strptime(args.date_range[1], '%Y-%m-%d')
            
            current_date = start_date
            while current_date <= end_date:
                adapter.scrape_date(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=1)
        
        adapter.show_stats()
        print("\n✅ Scraping terminé avec succès !")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
    finally:
        adapter.close_db()

if __name__ == '__main__':
    main()
