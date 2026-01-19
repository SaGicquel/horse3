import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection
import json
from datetime import datetime

conn = get_connection()
cur = conn.cursor()

print("=" * 100)
print("EXTRACTION DES VRAIS PARIS POUR COMPARAISON AVEC BACKTESTER")
print("=" * 100)

# Extraire tous les paris des 21 derniers jours
cur.execute("""
SELECT
    race_key,
    event_date,
    hippodrome,
    selection,
    bet_type,
    stake,
    odds,
    status,
    created_at
FROM user_bets
WHERE is_simulation = false
  AND created_at >= NOW() - INTERVAL '21 days'
ORDER BY created_at ASC
""")

real_bets = []
for row in cur.fetchall():
    race_key, event_date, hippodrome, selection, bet_type, stake, odds, status, created = row

    real_bets.append(
        {
            "race_key": race_key,
            "date": str(event_date),
            "hippodrome": hippodrome,
            "cheval": selection,
            "bet_type": bet_type,
            "stake": float(stake),
            "odds": float(odds),
            "status": status,
            "created_at": str(created),
            "expected_return": float(stake * odds) if status == "WIN" else 0.0,
        }
    )

print(f"\nExtrait {len(real_bets)} paris sur 21 jours")

# Sauvegarder en JSON
output_file = "/Users/gicquelsacha/horse3/real_bets_21days.json"
with open(output_file, "w") as f:
    json.dump(real_bets, f, indent=2, ensure_ascii=False)

print(f"Sauvegardé dans: {output_file}")

# Stats résumées
total_stake = sum(b["stake"] for b in real_bets)
total_return = sum(b["expected_return"] for b in real_bets)
nb_win = sum(1 for b in real_bets if b["status"] == "WIN")
nb_lose = sum(1 for b in real_bets if b["status"] == "LOSE")
nb_void = sum(1 for b in real_bets if b["status"] == "VOID")

print("\nRésumé:")
print(f"  Total paris: {len(real_bets)}")
print(f"    WIN: {nb_win}")
print(f"    LOSE: {nb_lose}")
print(f"    VOID: {nb_void}")
print(f"  Misé: {total_stake:.2f}€")
print(f"  Retour: {total_return:.2f}€")
print(f"  Profit: {total_return - total_stake:.2f}€")
print(f"  ROI: {(total_return - total_stake)/total_stake * 100:.2f}%")

# Maintenant, vérifier si ces courses existent dans cheval_courses_seen
print(f"\n{'='*100}")
print("VERIFICATION DISPONIBILITE DES COURSES DANS LA BDD")
print(f"{'='*100}")

# Extraire les race_keys uniques
race_keys = list(set(b["race_key"] for b in real_bets))
print(f"\n{len(race_keys)} courses différentes")

# Vérifier chaque course
courses_trouvees = 0
courses_manquantes = []

for race_key in race_keys:
    # Vérifier si la course existe
    cur.execute(
        """
        SELECT COUNT(DISTINCT nom_norm) as nb_chevaux
        FROM cheval_courses_seen
        WHERE race_key = %s
    """,
        (race_key,),
    )

    result = cur.fetchone()
    nb_chevaux = result[0] if result else 0

    if nb_chevaux > 0:
        courses_trouvees += 1
    else:
        courses_manquantes.append(race_key)

print(f"  Trouvées: {courses_trouvees}/{len(race_keys)}")
print(f"  Manquantes: {len(courses_manquantes)}/{len(race_keys)}")

if courses_manquantes:
    print("\nCourses manquantes (échantillon):")
    for race in courses_manquantes[:10]:
        print(f"  - {race}")

# Pour les paris où la course existe, vérifier si le cheval existe
print(f"\n{'='*100}")
print("VERIFICATION CHEVAUX DANS LA BDD")
print(f"{'='*100}")

chevaux_trouves = 0
chevaux_manquants = []

for bet in real_bets:
    race_key = bet["race_key"]
    cheval_nom = bet["cheval"]

    # Chercher le cheval (case insensitive, approximatif)
    cur.execute(
        """
        SELECT nom_norm
        FROM cheval_courses_seen
        WHERE race_key = %s
          AND LOWER(nom_norm) = LOWER(%s)
        LIMIT 1
    """,
        (race_key, cheval_nom),
    )

    result = cur.fetchone()
    if result:
        chevaux_trouves += 1
        bet["nom_found"] = result[0]
    else:
        chevaux_manquants.append({"race": race_key, "cheval": cheval_nom})

print(f"  Chevaux trouvés: {chevaux_trouves}/{len(real_bets)}")
print(f"  Chevaux manquants: {len(chevaux_manquants)}/{len(real_bets)}")

if chevaux_manquants:
    print("\nExemples chevaux manquants:")
    for item in chevaux_manquants[:10]:
        print(f"  - {item['race']} | {item['cheval']}")

conn.close()

print(f"\n{'='*100}")
print("Données exportées. Prochaine étape: comparer avec le backtester")
print(f"{'='*100}")
