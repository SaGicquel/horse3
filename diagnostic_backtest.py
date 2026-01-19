"""
Diagnostic: Pourquoi le backtest échoue alors que la production fonctionne ?
Compare un échantillon de picks réels VS simulation
"""

import sys
import json

sys.path.append("/Users/gicquelsacha/horse3")
sys.path.append("/Users/gicquelsacha/horse3/web/backend")

from db_connection import get_connection
from web.backend.main import run_benter_head_for_date, calculate_prediction_score

# Charger les picks réels du 08/12/2025
with open("/Users/gicquelsacha/horse3/data/picks/picks_2025-12-08.json") as f:
    picks_data = json.load(f)

print("=" * 80)
print(" COMPARAISON PRODUCTION VS SIMULATION")
print("=" * 80)
print()

# Prendre les 5 premiers picks
sample_picks = picks_data["picks"][:5]

conn = get_connection()
cur = conn.cursor()

print("PRODUCTION (Picks réels générés):")
print("-" * 80)
for i, pick in enumerate(sample_picks, 1):
    print(
        f"{i}. {pick['nom']:20} | "
        f"Odds: {pick.get('cote', 0):5.1f} | "
        f"P_Win: {pick.get('p_win', 0)*100:5.1f}% | "
        f"Value: {pick.get('value_pct', 0):+6.1f}% | "
        f"Kelly: {pick.get('kelly_pct', 0):5.1f}%"
    )

print()
print("SIMULATION (Ce que le backtest calcule):")
print("-" * 80)

# Lancer Benter pour la même date
try:
    benter_result = run_benter_head_for_date("2025-12-08", cur=cur)
    b_map = benter_result.get("by_runner", {})
    print(f"✅ Benter executé : {len(b_map)} runners")
except Exception as e:
    print(f"❌ Benter failed: {e}")
    b_map = {}

# Pour chaque pick, voir ce que le simulateur aurait calculé
for i, pick in enumerate(sample_picks, 1):
    race_key = pick["race_key"]
    numero = pick["numero"]
    nom = pick["nom"]

    # Chercher dans la DB
    cur.execute(
        """
        SELECT cote_reference, cote_finale, tendance_cote, amplitude_tendance,
               est_favori, avis_entraineur
        FROM cheval_courses_seen
        WHERE race_key = %s AND numero_dossard = %s
    """,
        (race_key, numero),
    )

    row = cur.fetchone()
    if not row:
        print(f"{i}. {nom:20} | ❌ NOT FOUND IN DB")
        continue

    c_ref, c_fin, tend, amp, fav, avis = row

    # Quelle cote le simulateur utiliserait ?
    odds_sim = c_ref if (c_ref and c_ref > 1) else c_fin

    # Benter prob
    b_key = (race_key, numero)
    b_info = b_map.get(b_key)

    if b_info:
        p_benter = b_info.get("p_calibrated", 0) * 100
        source = "Benter"
    else:
        # Fallback
        score = calculate_prediction_score(odds_sim, c_ref, tend, amp, fav, avis)
        prob_imp = 1.0 / odds_sim if odds_sim > 0 else 0.01
        adjustment = (score - 50) / 100
        p_benter = max(0.01, min(0.95, prob_imp * (1 + adjustment))) * 100
        source = "Fallback"

    # Value calculation
    expected_odds = 100 / p_benter if p_benter > 0 else 999
    value_pct = ((odds_sim / expected_odds) - 1) * 100

    # Display
    print(
        f"{i}. {nom:20} | "
        f"Ref: {c_ref if c_ref else 'N/A':>5} "
        f"Fin: {c_fin:>5.1f} | "
        f"P: {p_benter:5.1f}% ({source:8}) | "
        f"Value: {value_pct:+6.1f}% | "
        f"{'⚠️ NO REF!' if not c_ref else ''}"
    )

print()
print("ANALYSE:")
print("-" * 80)
print("Si Value Simulation << Value Production:")
print("  → Le fallback Benter est trop pessimiste")
print("  → OU les cotes de référence manquent")
print()
print("Si P_Win Simulation << P_Win Production:")
print("  → Le modèle Champion n'est pas utilisé")
print("  → OU Benter n'est pas calibré correctement")

conn.close()
