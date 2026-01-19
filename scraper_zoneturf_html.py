#!/usr/bin/env python3
"""
SCRAPER HTML ZONE-TURF

Scrape les résultats des courses depuis les pages HTML de Zone-Turf.
Génère un CSV compatible avec enrichir_zoneturf.py

Usage:
    python scraper_zoneturf_html.py --date 2024-10-20
    python scraper_zoneturf_html.py --date-range 2024-10-20 2024-10-26
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import csv
import argparse
from pathlib import Path
import time


class ScraperZoneTurfHTML:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
        self.base_url = "https://www.zone-turf.fr"

    def get_reunions_du_jour(self, date):
        """
        Récupère la liste des réunions pour une date donnée

        Args:
            date (str): Date au format YYYY-MM-DD

        Returns:
            list: Liste des URLs de réunions
        """
        # Formater la date pour Zone-Turf (ex: "mardi-15-octobre-2024")
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        mois = [
            "",
            "janvier",
            "février",
            "mars",
            "avril",
            "mai",
            "juin",
            "juillet",
            "août",
            "septembre",
            "octobre",
            "novembre",
            "décembre",
        ]

        jour_nom = jours[date_obj.weekday()]
        mois_nom = mois[date_obj.month]

        url_date = f"{jour_nom}-{date_obj.day}-{mois_nom}-{date_obj.year}"
        url = f"{self.base_url}/resultats/resultats-pmu-du-{url_date}.html"

        print(f"📅 Recherche réunions pour {date} : {url}")

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Chercher les liens vers les réunions
            reunions = []

            # Zone-Turf liste les réunions sous forme de liens
            # Ex: /programmes/r1-vincennes-123456.html
            links = soup.find_all("a", href=re.compile(r"/programmes/r\d+-"))

            for link in links:
                href = link.get("href")
                if href and href not in reunions:
                    reunions.append(href)

            print(f"   ✅ {len(reunions)} réunions trouvées")
            return reunions

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print("   ⚠️  Aucune réunion pour cette date (404)")
                return []
            else:
                print(f"   ❌ Erreur HTTP {e.response.status_code}")
                return []
        except Exception as e:
            print(f"   ❌ Erreur : {e}")
            return []

    def scrape_reunion(self, reunion_url):
        """
        Scrape une réunion complète avec toutes ses courses

        Args:
            reunion_url (str): URL de la réunion

        Returns:
            list: Liste de dictionnaires (une ligne par cheval)
        """
        print(f"   🏇 Scraping réunion : {reunion_url}")

        url = f"{self.base_url}{reunion_url}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extraire les informations de la réunion
            # Ex: R1-VINCENNES -> reunion=1, hippodrome=VINCENNES
            match = re.search(r"r(\d+)-([a-z\-]+)", reunion_url, re.IGNORECASE)
            if not match:
                print("      ❌ Impossible de parser l'URL de la réunion")
                return []

            numero_reunion = match.group(1)
            hippodrome_code = match.group(2).upper().replace("-", " ")

            # Extraire la date depuis le contenu
            # TODO: Implémenter extraction date réelle

            # Chercher toutes les courses de la réunion
            courses_data = []

            # Zone-Turf structure: chaque course a un ID unique
            # Les résultats sont dans des tableaux HTML

            # Chercher les tableaux de résultats
            tables = soup.find_all("table", class_=re.compile(r"result|tableau"))

            print(f"      📊 {len(tables)} tableaux trouvés")

            # Pour l'instant, retourner vide (structure à compléter)
            # Cette implémentation nécessite d'analyser plus en détail la structure HTML

            return courses_data

        except Exception as e:
            print(f"      ❌ Erreur : {e}")
            return []

    def scrape_date(self, date):
        """
        Scrape toutes les courses d'une date

        Args:
            date (str): Date au format YYYY-MM-DD

        Returns:
            list: Liste de toutes les performances
        """
        reunions = self.get_reunions_du_jour(date)

        all_data = []
        for reunion_url in reunions:
            data = self.scrape_reunion(reunion_url)
            all_data.extend(data)
            time.sleep(0.5)  # Pause pour ne pas surcharger le serveur

        return all_data

    def save_to_csv(self, data, output_file):
        """
        Sauvegarde les données au format CSV compatible avec enrichir_zoneturf.py

        Args:
            data (list): Liste de dictionnaires
            output_file (str): Chemin du fichier CSV de sortie
        """
        if not data:
            print("   ⚠️  Aucune donnée à sauvegarder")
            return

        # Format attendu par enrichir_zoneturf.py :
        # Date;Hippodrome;Reunion;Course;Num;Cheval;Musique;Corde;Ecart;Temps;CoteDirect;CoteRef

        fieldnames = [
            "Date",
            "Hippodrome",
            "Reunion",
            "Course",
            "Num",
            "Cheval",
            "Musique",
            "Corde",
            "Ecart",
            "Temps",
            "CoteDirect",
            "CoteRef",
        ]

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(data)

        print(f"   ✅ {len(data)} lignes sauvegardées dans {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Scraper HTML Zone-Turf")
    parser.add_argument("--date", type=str, help="Date à scraper (YYYY-MM-DD)")
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        help="Plage de dates (YYYY-MM-DD YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/zoneturf", help="Répertoire de sortie pour les CSV"
    )

    args = parser.parse_args()

    # Créer le répertoire de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scraper = ScraperZoneTurfHTML()

    # Déterminer les dates à scraper
    if args.date:
        dates = [args.date]
    elif args.date_range:
        start = datetime.strptime(args.date_range[0], "%Y-%m-%d")
        end = datetime.strptime(args.date_range[1], "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
    else:
        # Par défaut : hier
        hier = datetime.now() - timedelta(days=1)
        dates = [hier.strftime("%Y-%m-%d")]

    print("\n" + "=" * 70)
    print("🏇 SCRAPER HTML ZONE-TURF")
    print("=" * 70)
    print(f"📅 Dates : {dates[0]}" + (f" → {dates[-1]}" if len(dates) > 1 else ""))
    print()

    # Scraper chaque date
    for date in dates:
        print("=" * 70)
        print(f"📆 {date}")
        print("=" * 70)

        data = scraper.scrape_date(date)

        if data:
            output_file = output_dir / f"zoneturf_{date.replace('-', '')}.csv"
            scraper.save_to_csv(data, str(output_file))
        else:
            print("   ⚠️  Aucune donnée récupérée pour cette date")

        print()

    print("=" * 70)
    print("✅ Scraping terminé !")
    print("=" * 70)


if __name__ == "__main__":
    main()
