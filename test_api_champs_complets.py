#!/usr/bin/env python3
"""
Test pour voir TOUS les champs disponibles dans l'API PMU
pour 1 participant complet
"""

import requests
import psycopg2
import json
import os


def main():
    # Connexion BDD (vraies credentials Docker)
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=54624,
        database="pmu_database",
        user="pmu_user",
        password="pmu_secure_password_2025",
    )
    cur = conn.cursor()

    # Prendre une course de 2024
    cur.execute("""
        SELECT id_course, date_course
        FROM courses
        WHERE date_course < '2025-01-01'
        ORDER BY date_course DESC
        LIMIT 1
    """)
    result = cur.fetchone()

    if not result:
        print("❌ Aucune course de 2024 trouvée")
        cur.close()
        conn.close()
        return

    id_course, date_course = result
    print(f"✅ Course trouvée : {id_course} ({date_course})")

    # Parser l'ID
    parts = id_course.split("_")
    date_str = parts[0]  # Format YYYYMMDD
    reunion = int(parts[2].replace("R", ""))
    course = int(parts[3].replace("C", ""))

    # Formatter pour l'API
    date_iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

    print(f"\n📅 Date ISO : {date_iso}")
    print(f"🏇 Réunion {reunion} - Course {course}")

    # Test 1 : /participants
    url = f"https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{date_iso}/R{reunion}/C{course}/participants"
    print("\n🔍 Test /participants :")
    print(f"URL : {url}\n")

    resp = requests.get(url, timeout=10)
    print(f"Status : {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        if "programme" in data and "participants" in data["programme"]:
            participants = data["programme"]["participants"]
            print(f"✅ {len(participants)} participants trouvés")

            # Afficher le 1er participant COMPLET
            if participants:
                print("\n" + "=" * 80)
                print("📊 STRUCTURE COMPLÈTE du 1er participant :")
                print("=" * 80)
                print(json.dumps(participants[0], indent=2, ensure_ascii=False))

                # Liste des clés de niveau 1
                print("\n" + "=" * 80)
                print("🔑 CLÉS DISPONIBLES (niveau 1) :")
                print("=" * 80)
                for key in sorted(participants[0].keys()):
                    value = participants[0][key]
                    value_type = type(value).__name__
                    if isinstance(value, dict):
                        subkeys = list(value.keys())[:5]
                        print(f"  • {key} ({value_type}) → {subkeys}")
                    elif isinstance(value, list):
                        print(f"  • {key} ({value_type}) → {len(value)} éléments")
                    else:
                        print(f"  • {key} ({value_type}) = {value}")

        else:
            print(f"❌ Structure inattendue : {list(data.keys())}")
    else:
        print(f"❌ Erreur {resp.status_code} : {resp.text[:200]}")

    # Test 2 : /programme (métadonnées course)
    url2 = f"https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{date_iso}/R{reunion}/C{course}"
    print("\n\n🔍 Test /programme (métadonnées course) :")
    print(f"URL : {url2}\n")

    resp2 = requests.get(url2, timeout=10)
    print(f"Status : {resp2.status_code}")

    if resp2.status_code == 200:
        data2 = resp2.json()
        print("\n" + "=" * 80)
        print("📊 STRUCTURE des métadonnées course :")
        print("=" * 80)
        print(json.dumps(data2, indent=2, ensure_ascii=False))

        # Liste des clés de niveau 1
        if "programme" in data2:
            print("\n" + "=" * 80)
            print("🔑 CLÉS DISPONIBLES dans 'programme' :")
            print("=" * 80)
            for key in sorted(data2["programme"].keys()):
                value = data2["programme"][key]
                value_type = type(value).__name__
                if isinstance(value, dict):
                    subkeys = list(value.keys())[:5]
                    print(f"  • {key} ({value_type}) → {subkeys}")
                elif isinstance(value, list):
                    print(f"  • {key} ({value_type}) → {len(value)} éléments")
                else:
                    print(f"  • {key} ({value_type}) = {value}")
    else:
        print(f"❌ Erreur {resp2.status_code} : {resp2.text[:200]}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
