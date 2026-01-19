#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour mettre à jour les vainqueurs dans la table cheval_courses_seen
à partir des résultats disponibles dans l'API PMU.
"""

import sqlite3
import requests
import time
from datetime import datetime, timedelta

DB_PATH = "data/database.db"
BASE = "https://online.turfinfo.api.pmu.fr/rest/client/7"
FALLBACK_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/7"

UA = "horse-winner-updater/1.0"
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json",
    "Accept-Language": "fr-FR,fr;q=0.9",
}


def to_pmu_date(date_iso: str) -> str:
    """Convertit YYYY-MM-DD en DDMMYYYY"""
    yyyy, mm, dd = date_iso.split("-")
    return f"{dd}{mm}{yyyy}"


def norm(s: str) -> str:
    """Normalise un nom de cheval (minuscules, sans accents)"""
    import unicodedata
    import re

    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def get_race_results(date_iso: str, reunion: int, course: int):
    """
    Récupère les résultats d'une course via l'API PMU.
    Retourne un dict: nom_norm -> place (int ou None)
    """
    date_pmu = to_pmu_date(date_iso)

    for base in (BASE, FALLBACK_BASE):
        url = f"{base}/programme/{date_pmu}/R{reunion}/C{course}/participants"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                participants = data.get("participants", [])

                results = {}
                for p in participants:
                    nom = p.get("nom")
                    if not nom:
                        continue

                    # Chercher la place dans les différents champs possibles
                    place = (
                        p.get("ordreArrivee")
                        or p.get("place")
                        or p.get("rang")
                        or p.get("classement")
                    )

                    # Convertir en int si possible
                    if place is not None:
                        try:
                            place = int(place)
                        except (ValueError, TypeError):
                            place = None

                    results[norm(nom)] = place

                return results

        except Exception as e:
            print(f"⚠️  Erreur pour {base}: {e}")
            continue

    return None


def update_winners():
    """
    Met à jour le champ is_win dans cheval_courses_seen
    en interrogeant l'API PMU pour chaque course.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("=" * 70)
    print("   MISE À JOUR DES VAINQUEURS")
    print("=" * 70)

    # Récupérer toutes les courses distinctes
    cur.execute("""
        SELECT DISTINCT race_key, annee
        FROM cheval_courses_seen
        ORDER BY race_key DESC
    """)

    races = cur.fetchall()
    print(f"\n📊 {len(races)} courses à traiter\n")

    updated_races = 0
    updated_winners = 0
    errors = 0

    for race_key, annee in races:
        # Parser race_key: "YYYY-MM-DD|R#|C#|HIPPO"
        try:
            parts = race_key.split("|")
            date_iso = parts[0]
            reunion = int(parts[1].replace("R", ""))
            course = int(parts[2].replace("C", ""))
            hippo = parts[3]
        except (IndexError, ValueError) as e:
            print(f"⚠️  Impossible de parser: {race_key} ({e})")
            errors += 1
            continue

        # Ne traiter que les courses passées (pas aujourd'hui ni futur)
        course_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
        today = datetime.now().date()

        if course_date >= today:
            continue  # Course future, pas de résultats

        # Récupérer les résultats
        results = get_race_results(date_iso, reunion, course)

        if results is None:
            print(f"❌ R{reunion}C{course} {date_iso}: API inaccessible")
            errors += 1
            continue

        if not results:
            print(f"⚠️  R{reunion}C{course} {date_iso}: Pas de résultats")
            continue

        # Trouver le(s) gagnant(s)
        winners = [nom for nom, place in results.items() if place == 1]

        if not winners:
            print(f"ℹ️  R{reunion}C{course} {date_iso}: Pas de gagnant identifié")
            continue

        # Mettre à jour la base de données
        course_updated = False
        for winner in winners:
            cur.execute(
                """
                UPDATE cheval_courses_seen
                SET is_win = 1
                WHERE race_key = ? AND LOWER(nom_norm) = ?
            """,
                (race_key, winner),
            )

            if cur.rowcount > 0:
                updated_winners += 1
                course_updated = True

        if course_updated:
            updated_races += 1
            winner_names = ", ".join(winners)
            print(f"✅ R{reunion}C{course} {date_iso} à {hippo}: {winner_names}")

        # Pause pour ne pas surcharger l'API
        time.sleep(0.3)

    conn.commit()

    print("\n" + "=" * 70)
    print("   RÉSUMÉ")
    print("=" * 70)
    print(f"\n✅ Courses mises à jour: {updated_races}")
    print(f"✅ Vainqueurs identifiés: {updated_winners}")
    print(f"❌ Erreurs: {errors}")

    # Vérification finale
    cur.execute("""
        SELECT COUNT(*)
        FROM cheval_courses_seen
        WHERE is_win = 1
    """)
    total_winners = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    total_participations = cur.fetchone()[0]

    print(f"\n📊 Total des victoires enregistrées: {total_winners}/{total_participations}")
    print(f"📊 Taux de victoires: {total_winners/total_participations*100:.2f}%")

    conn.close()
    print("\n✅ Mise à jour terminée!")


if __name__ == "__main__":
    update_winners()
