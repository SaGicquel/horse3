import requests

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "fr-FR,fr;q=0.9",
    }
)

date = "2025-11-26"
date_nodash = date.replace("-", "")

# D'abord récupérer le programme pour voir les courses disponibles
url = f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_nodash}"
resp = session.get(url, timeout=10)
print(f"Status: {resp.status_code}")
data = resp.json()
print(f"Reunions: {len(data.get('programme', {}).get('reunions', []))}")

if data.get("programme", {}).get("reunions"):
    r = data["programme"]["reunions"][0]
    print(f"R1: {r.get('hippodrome', {}).get('libelleLong')}")
    courses = r.get("courses", [])
    print(f"Courses: {len(courses)}")

    if courses:
        c = courses[0]
        reunion_num = r.get("numOfficiel", 1)
        course_num = c.get("numOrdre", 1)

        # Récupérer les participants
        url2 = f"https://online.turfinfo.api.pmu.fr/rest/client/7/programme/{date_nodash}/R{reunion_num}/C{course_num}/participants"
        resp2 = session.get(url2, timeout=10)
        data2 = resp2.json()
        participants = data2.get("participants", [])
        print(f"Participants: {len(participants)}")

        # Afficher les sexes et id trouvés
        for p in participants[:5]:
            print(f"Nom: {p.get('nom')}")
            print(f"  Sexe: '{p.get('sexe')}' (type={type(p.get('sexe')).__name__})")
            id_ch = p.get("idCheval")
            print(f"  idCheval: '{id_ch}' (len={len(str(id_ch)) if id_ch else 0})")
            print()
