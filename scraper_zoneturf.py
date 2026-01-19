#!/usr/bin/env python3
"""
SCRAPER ZONE-TURF - Conforme au plan complet
Source principale : CSV Zone-Turf (58+ colonnes)

Phase 1 : Données obligatoires
- Courses (métadonnées)
- Hippodromes
- Chevaux (profil)
- Personnes (jockey/entraîneur)
- Performances (résultats)

Usage:
    python scraper_zoneturf.py --date 2025-11-11
    python scraper_zoneturf.py --date today
    python scraper_zoneturf.py --date-range 2025-11-01 2025-11-10
"""

import argparse
import csv
import re
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

from db_connection import get_connection

# Configuration
UA = "horse3-zoneturf-scraper/2.0 (+contact@example.com)"
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/csv,text/plain,*/*",
    "Accept-Language": "fr-FR,fr;q=0.9",
}

# Mapping disciplines
DISCIPLINE_MAP = {
    "PLAT": "Plat",
    "TROT": "Trot",
    "HAIES": "Obstacle",
    "STEEPLE": "Obstacle",
    "CROSS": "Obstacle",
    "OBSTACLE": "Obstacle",
}

# Mapping sexes
SEXE_MAP = {
    "M": "M",  # Mâle
    "H": "H",  # Hongre
    "F": "F",  # Femelle
    "C": "M",  # Colt -> Mâle
    "G": "H",  # Gelding -> Hongre
}


class ZoneTurfScraper:
    """Scraper pour Zone-Turf avec nouveau schéma."""

    def __init__(self):
        self.conn = None
        self.cur = None
        self.stats = {
            "hippodromes": 0,
            "courses": 0,
            "chevaux": 0,
            "personnes": 0,
            "performances": 0,
        }

    def connect_db(self):
        """Connexion à la base de données."""
        self.conn = get_connection()
        self.cur = self.conn.cursor()

    def close_db(self):
        """Fermeture de la connexion."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def get_or_create_hippodrome(self, nom: str, code_pmu: str = None, pays: str = "FR") -> int:
        """
        Récupère ou crée un hippodrome.

        Returns:
            id_hippodrome
        """
        if not nom:
            return None

        # Normaliser le code PMU
        if not code_pmu:
            # Générer un code depuis le nom (ex: "Vincennes" -> "VINC")
            code_pmu = re.sub(r"[^A-Z]", "", nom.upper())[:4]

        # Chercher existant
        self.cur.execute(
            """
            SELECT id_hippodrome FROM hippodromes
            WHERE nom_hippodrome = %s OR code_pmu = %s
        """,
            (nom, code_pmu),
        )

        result = self.cur.fetchone()
        if result:
            return result[0]

        # Créer nouveau
        self.cur.execute(
            """
            INSERT INTO hippodromes (nom_hippodrome, code_pmu, pays)
            VALUES (%s, %s, %s)
            ON CONFLICT (code_pmu) DO UPDATE
            SET nom_hippodrome = EXCLUDED.nom_hippodrome
            RETURNING id_hippodrome
        """,
            (nom, code_pmu, pays),
        )

        id_hippodrome = self.cur.fetchone()[0]
        self.conn.commit()
        self.stats["hippodromes"] += 1

        return id_hippodrome

    def get_or_create_cheval(self, data: Dict) -> int:
        """
        Récupère ou crée un cheval.

        Args:
            data: Dict avec clés : nom_cheval, sexe_cheval, an_naissance, etc.

        Returns:
            id_cheval
        """
        nom = data.get("nom_cheval")
        sexe = data.get("sexe_cheval", "M")
        an_naissance = data.get("an_naissance")

        if not nom or not an_naissance:
            return None

        # Normaliser sexe
        sexe = SEXE_MAP.get(sexe.upper(), "M")

        # Chercher existant
        self.cur.execute(
            """
            SELECT id_cheval FROM chevaux
            WHERE nom_cheval = %s AND an_naissance = %s AND sexe_cheval = %s
        """,
            (nom, an_naissance, sexe),
        )

        result = self.cur.fetchone()
        if result:
            return result[0]

        # Créer nouveau
        self.cur.execute(
            """
            INSERT INTO chevaux (
                nom_cheval, sexe_cheval, an_naissance,
                robe, origine, entier_hongre,
                nom_pere, nom_mere, eleveur, proprietaire, code_proprio
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id_cheval
        """,
            (
                nom,
                sexe,
                an_naissance,
                data.get("robe"),
                data.get("origine"),
                data.get("entier_hongre"),
                data.get("nom_pere"),
                data.get("nom_mere"),
                data.get("eleveur"),
                data.get("proprietaire"),
                data.get("code_proprio"),
            ),
        )

        id_cheval = self.cur.fetchone()[0]
        self.conn.commit()
        self.stats["chevaux"] += 1

        return id_cheval

    def get_or_create_personne(self, nom: str, type_personne: str, code_pmu: str = None) -> int:
        """
        Récupère ou crée une personne (jockey/entraîneur).

        Args:
            nom: Nom complet
            type_personne: 'JOCKEY' ou 'ENTRAINEUR'
            code_pmu: Code PMU optionnel

        Returns:
            id_personne
        """
        if not nom:
            return None

        # Chercher existant
        self.cur.execute(
            """
            SELECT id_personne FROM personnes
            WHERE nom_complet = %s AND type = %s
        """,
            (nom, type_personne),
        )

        result = self.cur.fetchone()
        if result:
            return result[0]

        # Créer nouveau
        self.cur.execute(
            """
            INSERT INTO personnes (nom_complet, type, code_pmu)
            VALUES (%s, %s, %s)
            RETURNING id_personne
        """,
            (nom, type_personne, code_pmu),
        )

        id_personne = self.cur.fetchone()[0]
        self.conn.commit()
        self.stats["personnes"] += 1

        return id_personne

    def create_course(self, data: Dict) -> str:
        """
        Crée ou met à jour une course.

        Args:
            data: Dict avec toutes les infos course

        Returns:
            id_course (format: YYYYMMDD_CODE_R1_C1)
        """
        date_course = data["date_course"]
        hippodrome = data["nom_hippodrome"]
        num_reunion = data["num_reunion"]
        num_course = data["num_course"]

        # Générer ID unique
        id_hippodrome = self.get_or_create_hippodrome(hippodrome, data.get("code_hippodrome"))

        self.cur.execute(
            "SELECT code_pmu FROM hippodromes WHERE id_hippodrome = %s", (id_hippodrome,)
        )
        code_hippo = self.cur.fetchone()[0]

        id_course = f"{date_course.replace('-', '')}_{code_hippo}_R{num_reunion}_C{num_course}"

        # Normaliser discipline
        discipline_raw = data.get("discipline", "PLAT").upper()
        discipline = DISCIPLINE_MAP.get(discipline_raw, "Plat")

        # Insérer ou mettre à jour
        self.cur.execute(
            """
            INSERT INTO courses (
                id_course, date_course, heure_course,
                num_reunion, num_course, id_hippodrome,
                discipline, distance, allocation, nombre_partants,
                corde, etat_piste, categorie_age, sexe_condition, poids_condition,
                meteo, temperature_c, vent_kmh, statut
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id_course) DO UPDATE SET
                heure_course = EXCLUDED.heure_course,
                nombre_partants = EXCLUDED.nombre_partants,
                etat_piste = EXCLUDED.etat_piste,
                statut = EXCLUDED.statut,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id_course
        """,
            (
                id_course,
                data["date_course"],
                data.get("heure_course"),
                num_reunion,
                num_course,
                id_hippodrome,
                discipline,
                data.get("distance"),
                data.get("allocation"),
                data.get("nombre_partants"),
                data.get("corde"),
                data.get("etat_piste"),
                data.get("categorie_age"),
                data.get("sexe_condition"),
                data.get("poids_condition"),
                data.get("meteo"),
                data.get("temperature_c"),
                data.get("vent_kmh"),
                data.get("statut", "TERMINEE"),
            ),
        )

        self.conn.commit()
        self.stats["courses"] += 1

        return id_course

    def create_performance(self, id_course: str, data: Dict) -> int:
        """
        Crée une performance (résultat de course).

        Args:
            id_course: ID de la course
            data: Dict avec toutes les infos performance

        Returns:
            id_performance
        """
        # Récupérer/créer les entités liées
        id_cheval = self.get_or_create_cheval(data["cheval"])
        id_jockey = self.get_or_create_personne(
            data.get("nom_jockey"), "JOCKEY", data.get("code_jockey")
        )
        id_entraineur = self.get_or_create_personne(
            data.get("nom_entraineur"), "ENTRAINEUR", data.get("code_entraineur")
        )

        # Insérer performance
        self.cur.execute(
            """
            INSERT INTO performances (
                id_course, id_cheval, id_jockey, id_entraineur,
                numero_corde, numero_dossard, poids_porte,
                cote_pm, cote_sp,
                position_arrivee, ecart, disqualifie, non_partant,
                musique, deferre, oeilleres, champ_volonte,
                gain_course, rapport_gagnant, rapport_place,
                temps_total, vitesse_moyenne
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id_course, numero_corde) DO UPDATE SET
                position_arrivee = EXCLUDED.position_arrivee,
                cote_sp = EXCLUDED.cote_sp,
                rapport_gagnant = EXCLUDED.rapport_gagnant,
                rapport_place = EXCLUDED.rapport_place,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id_performance
        """,
            (
                id_course,
                id_cheval,
                id_jockey,
                id_entraineur,
                data.get("numero_corde"),
                data.get("numero_dossard"),
                data.get("poids_porte"),
                data.get("cote_pm"),
                data.get("cote_sp"),
                data.get("position_arrivee"),
                data.get("ecart"),
                data.get("disqualifie", False),
                data.get("non_partant", False),
                data.get("musique"),
                data.get("deferre"),
                data.get("oeilleres"),
                data.get("champ_volonte"),
                data.get("gain_course"),
                data.get("rapport_gagnant"),
                data.get("rapport_place"),
                data.get("temps_total"),
                data.get("vitesse_moyenne"),
            ),
        )

        id_performance = self.cur.fetchone()[0]
        self.conn.commit()
        self.stats["performances"] += 1

        return id_performance

    def scrape_csv_file(self, csv_path: Path):
        """
        Parse un fichier CSV Zone-Turf et insère dans la BDD.

        Format attendu : CSV avec colonnes nommées (header)
        """
        print(f"\n📄 Lecture du fichier : {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")

            current_course = None
            course_data = {}

            for row in reader:
                # Extraire infos course
                if not current_course or current_course != (row.get("id_course")):
                    course_data = self._parse_course_from_row(row)
                    current_course = self.create_course(course_data)

                # Extraire infos performance
                perf_data = self._parse_performance_from_row(row)
                if perf_data:
                    self.create_performance(current_course, perf_data)

        print(f"✅ Fichier traité : {csv_path.name}")

    def _parse_course_from_row(self, row: Dict) -> Dict:
        """Parse les données course depuis une ligne CSV."""
        return {
            "date_course": row.get("date_course", row.get("date")),
            "heure_course": row.get("heure_course", row.get("heure")),
            "num_reunion": int(row.get("num_reunion", row.get("reunion", 1))),
            "num_course": int(row.get("num_course", row.get("course", 1))),
            "nom_hippodrome": row.get("hippodrome", row.get("lieu")),
            "code_hippodrome": row.get("code_hippodrome"),
            "discipline": row.get("discipline", row.get("type_course", "PLAT")),
            "distance": self._parse_int(row.get("distance")),
            "allocation": self._parse_int(row.get("allocation", row.get("prix"))),
            "nombre_partants": self._parse_int(row.get("nombre_partants", row.get("partants"))),
            "corde": row.get("corde"),
            "etat_piste": row.get("etat_piste", row.get("terrain")),
            "categorie_age": row.get("categorie_age", row.get("age")),
            "sexe_condition": row.get("sexe_condition"),
            "poids_condition": row.get("poids_condition"),
            "meteo": row.get("meteo"),
            "temperature_c": self._parse_float(row.get("temperature")),
            "vent_kmh": self._parse_float(row.get("vent")),
            "statut": "TERMINEE" if row.get("position_arrivee") else "PREVUE",
        }

    def _parse_performance_from_row(self, row: Dict) -> Dict:
        """Parse les données performance depuis une ligne CSV."""
        # Données cheval
        cheval_data = {
            "nom_cheval": row.get("cheval", row.get("nom_cheval")),
            "sexe_cheval": row.get("sexe", row.get("sexe_cheval", "M")),
            "an_naissance": self._parse_int(row.get("an_naissance", row.get("age"))),
            "robe": row.get("robe"),
            "origine": row.get("origine", row.get("race")),
            "entier_hongre": row.get("entier_hongre"),
            "nom_pere": row.get("pere"),
            "nom_mere": row.get("mere"),
            "eleveur": row.get("eleveur"),
            "proprietaire": row.get("proprietaire"),
            "code_proprio": row.get("code_proprio"),
        }

        return {
            "cheval": cheval_data,
            "nom_jockey": row.get("jockey"),
            "code_jockey": row.get("code_jockey"),
            "nom_entraineur": row.get("entraineur"),
            "code_entraineur": row.get("code_entraineur"),
            "numero_corde": self._parse_int(
                row.get("numero_corde", row.get("numero", row.get("corde")))
            ),
            "numero_dossard": self._parse_int(row.get("numero_dossard")),
            "poids_porte": self._parse_int(row.get("poids", row.get("poids_porte"))),
            "cote_pm": self._parse_float(row.get("cote_pm", row.get("cote_ouverture"))),
            "cote_sp": self._parse_float(
                row.get("cote_sp", row.get("cote_depart", row.get("cote")))
            ),
            "position_arrivee": self._parse_int(
                row.get("position_arrivee", row.get("position", row.get("place")))
            ),
            "ecart": row.get("ecart"),
            "disqualifie": row.get("disqualifie", "False").lower() == "true",
            "non_partant": row.get("non_partant", "False").lower() == "true",
            "musique": row.get("musique"),
            "deferre": row.get("deferre"),
            "oeilleres": row.get("oeilleres"),
            "champ_volonte": row.get("champ_volonte"),
            "gain_course": self._parse_int(row.get("gain", row.get("gain_course"))),
            "rapport_gagnant": self._parse_float(row.get("rapport_gagnant", row.get("rapport_sg"))),
            "rapport_place": self._parse_float(row.get("rapport_place", row.get("rapport_sp"))),
            "temps_total": self._parse_float(row.get("temps", row.get("temps_total"))),
            "vitesse_moyenne": self._parse_float(row.get("vitesse", row.get("vitesse_moyenne"))),
        }

    def _parse_int(self, value) -> Optional[int]:
        """Parse un entier depuis une string."""
        if not value or value == "":
            return None
        try:
            return int(str(value).replace(" ", "").replace(",", ""))
        except (ValueError, AttributeError):
            return None

    def _parse_float(self, value) -> Optional[float]:
        """Parse un float depuis une string."""
        if not value or value == "":
            return None
        try:
            return float(str(value).replace(" ", "").replace(",", "."))
        except (ValueError, AttributeError):
            return None

    def show_stats(self):
        """Affiche les statistiques d'import."""
        print("\n" + "=" * 70)
        print("📊 STATISTIQUES D'IMPORT")
        print("=" * 70)
        for key, value in self.stats.items():
            print(f"   {key:20s} : {value:6d} créés")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Scraper Zone-Turf vers nouveau schéma")
    parser.add_argument("--csv", type=str, help="Chemin vers fichier CSV Zone-Turf")
    parser.add_argument("--date", type=str, help='Date ISO (YYYY-MM-DD) ou "today"')
    parser.add_argument(
        "--date-range", nargs=2, metavar=("START", "END"), help="Plage de dates (YYYY-MM-DD)"
    )

    args = parser.parse_args()

    scraper = ZoneTurfScraper()
    scraper.connect_db()

    try:
        if args.csv:
            # Mode fichier CSV direct
            csv_path = Path(args.csv)
            if not csv_path.exists():
                print(f"❌ Fichier introuvable : {csv_path}")
                sys.exit(1)

            scraper.scrape_csv_file(csv_path)

        elif args.date:
            # Mode date unique
            if args.date.lower() == "today":
                date = datetime.now().strftime("%Y-%m-%d")
            else:
                date = args.date

            print(f"📅 Scraping de la date : {date}")
            # TODO: Implémenter téléchargement CSV depuis Zone-Turf
            print("⚠️  Mode non implémenté : utilisez --csv pour l'instant")

        elif args.date_range:
            # Mode plage de dates
            start_date = datetime.strptime(args.date_range[0], "%Y-%m-%d")
            end_date = datetime.strptime(args.date_range[1], "%Y-%m-%d")

            print(f"📅 Scraping de {start_date.date()} à {end_date.date()}")
            # TODO: Implémenter boucle sur dates
            print("⚠️  Mode non implémenté : utilisez --csv pour l'instant")

        else:
            print("❌ Vous devez spécifier --csv, --date ou --date-range")
            parser.print_help()
            sys.exit(1)

        scraper.show_stats()

    finally:
        scraper.close_db()

    print("\n✅ Scraping terminé avec succès !")


if __name__ == "__main__":
    main()
