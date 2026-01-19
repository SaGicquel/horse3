#!/usr/bin/env python3
"""
Simulation de l'explosion des mises avec Kelly sans cap absolu.
"""

print("💥 SIMULATION DE L'EXPLOSION DES MISES")
print("=" * 50)

# Configuration du problème
kelly_fraction = 0.25
max_stake_pct = 0.05
initial_bankroll = 1000
total_bets = 1627
total_staked = 688000

print("📊 CONFIG:")
print(f"   Kelly fraction: {kelly_fraction}")
print(f"   Max stake %: {max_stake_pct}%")
print(f"   Bankroll initiale: {initial_bankroll}€")

print("\n🔥 SIMULATION RAPIDE:")

bankroll = initial_bankroll
cumul_staked = 0
bets_simulated = 0

# Simule une séquence de paris avec 60% de win rate et EV moyen de 20%
import random

random.seed(42)

for i in range(50):  # Simule 50 premiers paris
    # Stakes max = 5% de la bankroll
    max_stake = bankroll * max_stake_pct

    # Suppose Kelly optimise entre 1% et 5%
    kelly_stake = bankroll * random.uniform(0.01, max_stake_pct)
    stake = min(kelly_stake, max_stake)

    # Pari
    cumul_staked += stake
    bets_simulated += 1

    # 60% de chance de gagner avec cote moyenne ~2.5
    if random.random() < 0.6:
        # Gagne : profit = stake * (2.5 - 1) = stake * 1.5
        profit = stake * 1.5
        bankroll += profit
    else:
        # Perd
        bankroll -= stake

    if i < 10:
        print(f"   Pari {i+1:2d}: Mise={stake:6.0f}€  Bankroll={bankroll:8.0f}€")
    elif i == 10:
        print("   ...")

print("\n📈 RÉSULTATS SIMULATION:")
print(f"   Paris simulés: {bets_simulated}")
print(f"   Mise moyenne: {cumul_staked/bets_simulated:.0f}€")
print(f"   Bankroll finale: {bankroll:.0f}€")

print(f"\n🎯 EXTRAPOLATION SUR {total_bets} PARIS:")
avg_simulated = cumul_staked / bets_simulated
avg_real = total_staked / total_bets

print(f"   Moyenne simulée: {avg_simulated:.0f}€")
print(f"   Moyenne réelle: {avg_real:.0f}€")
print(f"   Ratio: {avg_real/avg_simulated:.1f}x")

print("\n⚠️ PROBLÈME IDENTIFIÉ:")
print("   → Pas de CAP ABSOLU sur les mises")
print("   → Avec une bankroll de 500k€, 5% = 25k€ par pari !!")
print("   → Kelly + croissance exponentielle = explosion des mises")

print("\n✅ SOLUTIONS POSSIBLES:")
print("   1. CAP ABSOLU: max 100€ par pari")
print("   2. KELLY plus conservateur: 10% au lieu de 25%")
print("   3. MAX_STAKE plus petit: 2% au lieu de 5%")
print("   4. Réinitialiser bankroll périodiquement")
