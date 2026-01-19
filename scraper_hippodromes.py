#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRAPER HIPPODROMES - Phase 2C
===============================
Enrichit la table hippodromes avec données géographiques et techniques
depuis Boturfers.fr et autres sources

Données scrapées :
- Ville, région, pays
- Coordonnées GPS (latitude, longitude)
- Altitude
- Type de piste (herbe, sable, synthétique)
- Configuration (main gauche/droite, ligne droite)
- Périmètre de la piste
- Dénivelé

Sources :
- Wikipédia (principal - données géographiques)
- France-Galop.com (caractéristiques techniques)
- OpenStreetMap Nominatim (géolocalisation GPS)
"""

import requests
from bs4 import BeautifulSoup
import time
import re
import logging
from typing import Dict, Optional, Tuple
from db_connection import get_connection
import json
import unicodedata

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ScraperHippodromes:
    """Scraper pour enrichir les données des hippodromes"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
        self.default_timeout = (10, 30)
        self.max_retries = 2

        # Cache pour éviter requêtes répétées
        self.cache_gps = {}

        # Mapping codes hippodrome vers noms Wikipedia
        self.mapping_hippodromes = self._init_mapping()

        # Mapping codes vers villes (données connues)
        self.code_to_ville = self._init_ville_mapping()

    def _init_mapping(self) -> Dict[str, str]:
        """Mapping codes PMU vers noms Wikipedia"""
        return {
            "VINC": "Hippodrome de Vincennes",
            "LGC": "Hippodrome de Longchamp",
            "CAG": "Hippodrome de la Côte d'Azur",
            "DEA": "Hippodrome de Deauville-La Touques",
            "CHT": "Hippodrome de Chantilly",
            "CHA": "Hippodrome de Chantilly",
            "EVR": "Hippodrome d'Évreux",
            "AUT": "Hippodrome d'Auteuil",
            "MAI": "Hippodrome de Maisons-Laffitte",
            "CLM": "Hippodrome de Clairefontaine",
            "SAI": "Hippodrome de Saint-Cloud",
            "COM": "Hippodrome de Compiègne",
            "FCH": "Hippodrome de la Solle",
            "PAU": "Hippodrome du Pont-Long",
            "VIC": "Hippodrome de Vichy",
            "MNT": "Hippodrome de Mont-de-Marsan",
            "STR": "Hippodrome de Strasbourg-Hoerdt",
            "LAV": "Hippodrome de Laval",
            "CAE": "Hippodrome de Caen",
            "ANG": "Hippodrome d'Angers",
            "AMI": "Hippodrome d'Amiens",
            "BOR": "Hippodrome du Bouscat",
            "LYN": "Hippodrome de Lyon-Parilly",
            "MRS": "Hippodrome de Marseille-Borély",
            "TOU": "Hippodrome de La Cépière",
            "NCE": "Hippodrome de la Côte d'Azur",
        }

    def _init_ville_mapping(self) -> Dict[str, str]:
        """Mapping codes PMU vers villes (données connues)"""
        return {
            "VINC": "Paris",
            "LGC": "Paris",
            "AUT": "Paris",
            "CAG": "Cagnes-sur-Mer",
            "DEA": "Deauville",
            "CHT": "Chantilly",
            "EVR": "Évreux",
            "MAI": "Maisons-Laffitte",
            "CLM": "Clairefontaine",
            "SAI": "Saint-Cloud",
            "COM": "Compiègne",
            "FCH": "Fontainebleau",
            "PAU": "Pau",
            "VIC": "Vichy",
            "MNT": "Mont-de-Marsan",
            "BOR": "Bordeaux",
            "LYN": "Lyon",
            "MRS": "Marseille",
            "TOU": "Toulouse",
            "NCE": "Nice",
        }

    def normalize_name(self, name: str) -> str:
        """Normalise un nom d'hippodrome pour URL Boturfers"""
        # Enlever accents
        replacements = {
            "é": "e",
            "è": "e",
            "ê": "e",
            "ë": "e",
            "à": "a",
            "â": "a",
            "ä": "a",
            "î": "i",
            "ï": "i",
            "ô": "o",
            "ö": "o",
            "ù": "u",
            "û": "u",
            "ü": "u",
            "ç": "c",
            "œ": "oe",
        }

        name_lower = name.lower()
        for old, new in replacements.items():
            name_lower = name_lower.replace(old, new)

        # Remplacer espaces et caractères spéciaux par tirets
        name_lower = re.sub(r"[^a-z0-9]+", "-", name_lower)
        name_lower = name_lower.strip("-")

        return name_lower

    def get_hippodrome_url(self, code_pmu: str, nom: str) -> str:
        """Construit l'URL Wikipedia pour un hippodrome"""
        # Essayer mapping d'abord
        if code_pmu in self.mapping_hippodromes:
            wiki_name = self.mapping_hippodromes[code_pmu]
        else:
            # Vérifier si le nom commence déjà par "HIPPODROME"
            nom_upper = nom.upper()
            if nom_upper.startswith("HIPPODROME"):
                # Ne pas ajouter de préfixe si déjà présent
                wiki_name = nom
            else:
                # Ajouter le préfixe
                wiki_name = f"Hippodrome de {nom}"

        # URL-encode pour Wikipedia
        wiki_name_encoded = wiki_name.replace(" ", "_")
        return f"https://fr.wikipedia.org/wiki/{wiki_name_encoded}"

    def scrape_wikipedia(self, code_pmu: str, nom: str) -> Optional[Dict]:
        """Scrape les infos d'un hippodrome depuis Wikipedia"""
        url = self.get_hippodrome_url(code_pmu, nom)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"🔍 Scraping Wikipedia : {url}")
                response = self.session.get(url, timeout=self.default_timeout)

                if response.status_code == 404:
                    logger.warning(f"⚠️  Page non trouvée : {url}")
                    return None

                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")

                data = {
                    "ville": None,
                    "region": None,
                    "latitude": None,
                    "longitude": None,
                    "altitude_m": None,
                    "type_piste": None,
                    "piste_config": None,
                    "perimetre_piste": None,
                    "denivele_piste": None,
                }

                # Chercher l'infobox (encadré de droite avec les infos)
                infobox = soup.find("table", class_="infobox")
                if infobox:
                    # Extraire les lignes de l'infobox
                    rows = infobox.find_all("tr")

                    for row in rows:
                        th = row.find("th")
                        td = row.find("td")

                        if not th or not td:
                            continue

                        label = th.get_text().strip().lower()
                        value = td.get_text().strip()

                        # Ville
                        if "localisation" in label or "commune" in label or "ville" in label:
                            data["ville"] = value.split(",")[0].strip()

                        # Région (département)
                        if "département" in label or "région" in label:
                            data["region"] = value.split("\n")[0].strip()

                        # GPS (coordonnées)
                        if "coordonnées" in label:
                            # Chercher les coordonnées dans le HTML
                            geo = td.find("span", class_="geo")
                            if geo:
                                coords_text = geo.get_text().strip()
                                # Format: "48.842556; 2.467889"
                                coords = coords_text.split(";")
                                if len(coords) == 2:
                                    try:
                                        data["latitude"] = float(coords[0].strip())
                                        data["longitude"] = float(coords[1].strip())
                                    except:
                                        pass

                        # Altitude
                        if "altitude" in label:
                            alt_match = re.search(r"(\d+)\s*m", value)
                            if alt_match:
                                data["altitude_m"] = int(alt_match.group(1))

                        # Type de piste
                        if "piste" in label or "surface" in label:
                            if (
                                "gazon" in value.lower()
                                or "herbe" in value.lower()
                                or "pelouse" in value.lower()
                            ):
                                data["type_piste"] = "Herbe"
                            elif "sable" in value.lower() or "fibresable" in value.lower():
                                data["type_piste"] = "Sable"
                            elif "synthétique" in value.lower() or "pst" in value.lower():
                                data["type_piste"] = "Synthétique"

                # Chercher dans le texte principal si pas trouvé
                content = soup.find("div", id="mw-content-text")
                if content:
                    text = content.get_text()

                    # Configuration main gauche/droite
                    if "main gauche" in text.lower():
                        data["piste_config"] = "Main gauche"
                    elif "main droite" in text.lower():
                        data["piste_config"] = "Main droite"
                    elif "ligne droite" in text.lower():
                        data["piste_config"] = "Ligne droite"

                    # Périmètre
                    perim_match = re.search(
                        r"(?:périmètre|longueur).*?(\d{3,4})\s*(?:m|mètres)", text, re.IGNORECASE
                    )
                    if perim_match:
                        data["perimetre_piste"] = int(perim_match.group(1))

                # Si ville connue dans mapping mais pas trouvée
                if not data["ville"] and code_pmu in self.code_to_ville:
                    data["ville"] = self.code_to_ville[code_pmu]
                    logger.info(f"   📍 Ville depuis mapping: {data['ville']}")

                # GPS avec OpenStreetMap Nominatim si pas trouvé dans Wikipedia
                if not data["latitude"] and data["ville"]:
                    gps = self.get_gps_from_nominatim(data["ville"], nom)
                    if gps:
                        data["latitude"] = gps[0]
                        data["longitude"] = gps[1]

                logger.info(
                    f"✅ Données extraites : ville={data['ville']}, GPS={data['latitude']}/{data['longitude']}"
                )
                return data

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    logger.warning(
                        f"⏱️  Timeout (tentative {attempt}/{self.max_retries}), retry dans 2s..."
                    )
                    time.sleep(2)
                else:
                    logger.error(f"❌ Timeout après {self.max_retries} tentatives : {url}")
                    return None

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Erreur HTTP : {e}")
                return None

            except Exception as e:
                logger.error(f"❌ Erreur lors du parsing : {e}")
                return None

        return None

    def get_gps_from_nominatim(
        self, ville: str, nom_hippodrome: str
    ) -> Optional[Tuple[float, float]]:
        """Récupère coordonnées GPS via OpenStreetMap Nominatim"""
        # Vérifier cache
        cache_key = f"{ville}_{nom_hippodrome}"
        if cache_key in self.cache_gps:
            return self.cache_gps[cache_key]

        try:
            # Essayer avec nom complet d'abord
            query = f"Hippodrome {nom_hippodrome}, {ville}, France"
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": query, "format": "json", "limit": 1}

            logger.info(f"🌍 Recherche GPS : {query}")
            response = self.session.get(url, params=params, timeout=(5, 10))
            response.raise_for_status()

            data = response.json()
            if data and len(data) > 0:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                logger.info(f"✅ GPS trouvé : {lat}, {lon}")

                # Mettre en cache
                self.cache_gps[cache_key] = (lat, lon)

                # Respecter rate limit Nominatim (1 req/sec)
                time.sleep(1)
                return (lat, lon)

            # Si pas trouvé, essayer juste la ville
            logger.warning("⚠️  GPS non trouvé avec nom hippodrome, essai avec ville seule")
            params["q"] = f"{ville}, France"
            response = self.session.get(url, params=params, timeout=(5, 10))
            data = response.json()

            if data and len(data) > 0:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                logger.info(f"✅ GPS ville trouvé : {lat}, {lon}")

                self.cache_gps[cache_key] = (lat, lon)
                time.sleep(1)
                return (lat, lon)

            logger.warning(f"⚠️  Aucune coordonnée GPS trouvée pour {ville}")
            return None

        except Exception as e:
            logger.error(f"❌ Erreur Nominatim : {e}")
            return None

    def enrich_hippodrome(self, id_hippodrome: int, code_pmu: str, nom: str) -> bool:
        """Enrichit un hippodrome avec les données scrapées"""
        logger.info(f"\n{'='*90}")
        logger.info(f"🏇 Enrichissement : {nom} ({code_pmu})")
        logger.info(f"{'='*90}")

        # Scraper Wikipedia
        data = self.scrape_wikipedia(code_pmu, nom)

        if not data:
            logger.warning(f"⚠️  Aucune donnée trouvée pour {nom}")
            return False

        # Mettre à jour la base de données
        try:
            conn = get_connection()
            cur = conn.cursor()

            update_fields = []
            values = []

            for field, value in data.items():
                if value is not None:
                    update_fields.append(f"{field} = %s")
                    values.append(value)

            if update_fields:
                values.append(id_hippodrome)
                query = f"""
                    UPDATE hippodromes
                    SET {', '.join(update_fields)},
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id_hippodrome = %s
                """

                cur.execute(query, values)
                conn.commit()

                logger.info(f"✅ Base de données mise à jour ({len(update_fields)} champs)")

                # Afficher résumé
                summary = []
                if data["ville"]:
                    summary.append(f"Ville: {data['ville']}")
                if data["region"]:
                    summary.append(f"Région: {data['region']}")
                if data["latitude"] and data["longitude"]:
                    summary.append(f"GPS: {data['latitude']:.4f}, {data['longitude']:.4f}")
                if data["altitude_m"]:
                    summary.append(f"Altitude: {data['altitude_m']}m")
                if data["type_piste"]:
                    summary.append(f"Piste: {data['type_piste']}")

                if summary:
                    logger.info(f"   📍 {' | '.join(summary)}")

                cur.close()
                conn.close()
                return True
            else:
                logger.warning("⚠️  Aucune donnée à mettre à jour")
                cur.close()
                conn.close()
                return False

        except Exception as e:
            logger.error(f"❌ Erreur lors de la mise à jour BDD : {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False

    def enrich_all_hippodromes(self, limit: Optional[int] = None):
        """Enrichit tous les hippodromes de la base"""
        logger.info("=" * 90)
        logger.info("🚀 ENRICHISSEMENT HIPPODROMES - PHASE 2C")
        logger.info("=" * 90)

        conn = get_connection()
        cur = conn.cursor()

        # Récupérer tous les hippodromes
        query = """
            SELECT id_hippodrome, code_pmu, nom_hippodrome
            FROM hippodromes
            ORDER BY nom_hippodrome
        """

        if limit:
            query += f" LIMIT {limit}"

        cur.execute(query)
        hippodromes = cur.fetchall()
        total = len(hippodromes)

        cur.close()
        conn.close()

        logger.info(f"\n📊 {total} hippodromes à enrichir")
        logger.info("")

        enrichis = 0
        echecs = 0

        for i, (id_hipp, code, nom) in enumerate(hippodromes, 1):
            logger.info(f"[{i}/{total}] {nom} ({code})")

            success = self.enrich_hippodrome(id_hipp, code, nom)

            if success:
                enrichis += 1
            else:
                echecs += 1

            # Pause entre requêtes
            if i < total:
                time.sleep(2)

            print()

        # Rapport final
        logger.info("=" * 90)
        logger.info("📊 RAPPORT FINAL")
        logger.info("=" * 90)
        logger.info(f"   Hippodromes traités    : {total}")
        logger.info(
            f"   Hippodromes enrichis   : {enrichis} ({100*enrichis//total if total > 0 else 0}%)"
        )
        logger.info(f"   Échecs                 : {echecs}")
        logger.info("=" * 90)


def main():
    """Fonction principale"""
    import argparse

    parser = argparse.ArgumentParser(description="Enrichissement des hippodromes (Phase 2C)")
    parser.add_argument("--code", type=str, help="Code PMU d'un hippodrome spécifique")
    parser.add_argument("--limit", type=int, help="Limiter le nombre d'hippodromes (pour tests)")
    parser.add_argument("--all", action="store_true", help="Enrichir tous les hippodromes")

    args = parser.parse_args()

    scraper = ScraperHippodromes()

    if args.code:
        # Enrichir un hippodrome spécifique
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id_hippodrome, code_pmu, nom_hippodrome
            FROM hippodromes
            WHERE code_pmu = %s
        """,
            (args.code.upper(),),
        )

        result = cur.fetchone()
        cur.close()
        conn.close()

        if result:
            id_hipp, code, nom = result
            scraper.enrich_hippodrome(id_hipp, code, nom)
        else:
            print(f"❌ Hippodrome {args.code} non trouvé")

    elif args.all or args.limit:
        # Enrichir tous (ou limité)
        scraper.enrich_all_hippodromes(limit=args.limit)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
