#!/usr/bin/env python3
"""
Compare les VRAIS paris vs ce qu'aurait fait le BACKTESTER sur les MÊMES courses
"""

import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection
import json

# Charger les vrais paris
with open("/Users/gicquelsacha/horse3/real_bets_21days.json", "r") as f:
    real_bets = json.load(f)

print("=" * 120)
print("COMPARAISON: VRAIS PARIS vs STRATEGIE BACKTEST sur les MÊMES courses")
print("=" * 120)

print(
    f"\nVrais paris: {len(real_bets)} paris sur {len(set(b['race_key'] for b in real_bets))} courses"
)

# Stats des vrais paris
real_stake = sum(b["stake"] for b in real_bets)
real_return = sum(b["expected_return"] for b in real_bets)
real_profit = real_return - real_stake
real_roi = (real_profit / real_stake) * 100

print("Performance réelle:")
print(f"  Misé: {real_stake:.2f}€")
print(f"  Retour: {real_return:.2f}€")
print(f"  Profit: {real_profit:.2f}€")
print(f"  ROI: {real_roi:.2f}%")

# Maintenant, pour chaque vrai pari, vérifier ce qu'aurait calculé le modèle
conn = get_connection()
cur = conn.cursor()

print(f"\n{'='*120}")
print("ANALYSE PAR PARI:")
print(f"{'='*120}")

compatible_bets = 0
incompatible_bets = []

for i, bet in enumerate(real_bets[:10]):  # Première 10 pour analyse
    race_key = bet["race_key"]
    cheval_nom = bet["cheval"]
    real_stake_val = bet["stake"]
    real_odds = bet["odds"]
    status = bet["status"]

    # Récupérer les données de ce cheval à cette course
    cur.execute(
        """
        SELECT
            nom_norm,
            cote_reference,
            cote_finale,
            is_win,
            place_finale
        FROM cheval_courses_seen
        WHERE race_key = %s
          AND LOWER(nom_norm) = LOWER(%s)
        LIMIT 1
    """,
        (race_key, cheval_nom),
    )

    result = cur.fetchone()

    if result:
        nom, cote_ref, cote_fin, is_win, place_finale = result

        # Déterminer si placé (gagnant ou place_finale <= 3)
        is_placed = is_win or (place_finale and place_finale <= 3)

        print(f"\nPari #{i+1}: {race_key} | {nom}")
        print(f"  Vrai pari: {real_stake_val}€ @ {real_odds} => {status}")
        print(
            f"  BDD: cote_ref={cote_ref}, cote_fin={cote_fin}, is_win={is_win}, place={place_finale}, placé={is_placed}"
        )

        # Vérifier cohérence
        if is_placed and status == "WIN":
            print("  ✅ Cohérent: WIN et placé")
            compatible_bets += 1
        elif not is_placed and status == "LOSE":
            print("  ✅ Cohérent: LOSE et non placé")
            compatible_bets += 1
        elif status == "VOID":
            print("  ⚠️  VOID (course annulée?)")
            compatible_bets += 1
        else:
            print(f"  ❌ INCOHÉRENCE: status={status} mais placé={is_placed}")
            incompatible_bets.append(
                {"race": race_key, "cheval": nom, "status": status, "placed": is_placed}
            )

print(f"\n{'='*120}")
print(f"Compatible: {compatible_bets}/{min(10, len(real_bets))}")
print(f"Incohérent: {len(incompatible_bets)}/{min(10, len(real_bets))}")

# Maintenant, vérifions la LOGIQUE de sélection
# Pour chaque course où vous avez parié, regarder TOUS les chevaux disponibles
print(f"\n{'='*120}")
print("ANALYSE DES CRITÈRES DE SÉLECTION (5 premières courses)")
print(f"{'='*120}")

races_analyzed = list(set(b["race_key"] for b in real_bets))[:5]

for race_key in races_analyzed:
    print(f"\n{race_key}")
    print("-" * 120)

    # Vos paris sur cette course
    your_bets_this_race = [b for b in real_bets if b["race_key"] == race_key]

    # Tous les chevaux de cette course
    cur.execute(
        """
        SELECT
            nom_norm,
            cote_reference,
            cote_finale,
            is_win,
            place_finale
        FROM cheval_courses_seen
        WHERE race_key = %s
        ORDER BY cote_reference ASC NULLS LAST
    """,
        (race_key,),
    )

    all_horses = cur.fetchall()

    print(f"  {len(all_horses)} chevaux partants")
    print(f"  Vous avez parié sur {len(your_bets_this_race)} chevaux:")

    for yb in your_bets_this_race:
        print(
            f"    - {yb['cheval']:30} | Mise: {yb['stake']:5.2f}€ @ {yb['odds']:6.2f} => {yb['status']}"
        )

    print("\n  Tous les chevaux (Top 10 + vos paris):")
    print(
        f"  {'Nom':30} | {'Cote Ref':10} | {'Cote Fin':10} | {'Placé':7} | {'Vous avez parié?':15}"
    )

    your_horses = set(b["cheval"].lower() for b in your_bets_this_race)

    for i, (nom, cote_ref, cote_fin, is_win, place_finale) in enumerate(all_horses):
        placed = "OUI" if (is_win or (place_finale and place_finale <= 3)) else "NON"
        you_bet = "✓ OUI" if nom.lower() in your_horses else ""

        if i < 10 or nom.lower() in your_horses:
            print(
                f"  {nom:30} | {cote_ref or 'N/A':>10} | {cote_fin or 'N/A':>10} | {placed:>7} | {you_bet:>15}"
            )

conn.close()

print(f"\n{'='*120}")
print("CONCLUSION:")
print(f"{'='*120}")
print("Prochaine étape: identifier les CRITÈRES qui différencient vos paris gagnants")
print("des autres chevaux disponibles dans ces courses")
