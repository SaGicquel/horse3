#!/usr/bin/env python3
"""
Test pour analyser les mises du backtester.
"""

# Lire le rapport du dernier backtest
print("🔍 ANALYSE DES MISES ANORMALES")
print("=" * 50)

# Simulons les données du rapport
n_bets = 1627
total_staked = 688000
roi_pct = 96.30
final_bankroll = 663568.53
initial_bankroll = 1000
total_profit = total_staked * (roi_pct / 100)

print("📊 DONNÉES DU RAPPORT:")
print(f"   Nombre de paris: {n_bets:,}")
print(f"   Total misé: {total_staked:,.0f}€")
print(f"   ROI: {roi_pct}%")
print(f"   Bankroll finale: {final_bankroll:,.0f}€")

print("\n🧮 CALCULS:")
avg_stake = total_staked / n_bets
print(f"   Mise moyenne: {avg_stake:.0f}€ par pari")
print(f"   Profit total: {total_profit:,.0f}€")
print(
    f"   Bankroll calculée: {initial_bankroll} + {total_profit:,.0f} = {initial_bankroll + total_profit:,.0f}€"
)

print("\n⚠️ PROBLÈMES DÉTECTÉS:")
print(f"   1. Mise moyenne ÉNORME: {avg_stake:.0f}€")
print("      → Dans l'ancien backtest: ~15€")
print(f"      → Ici: x{avg_stake/15:.0f} fois plus !!")

print("\n   2. Croissance exponentielle:")
if n_bets > 0:
    # Simulation d'une croissance avec Kelly
    bankroll_sim = initial_bankroll
    stakes = []
    for i in range(min(10, n_bets)):
        stake_pct = 0.02  # 2% max selon config
        stake = bankroll_sim * stake_pct
        stakes.append(stake)
        # Supposons 25% de taux de gain avec EV moyen de 48%
        if i % 4 == 0:  # 1 sur 4 gagne (25%)
            bankroll_sim += stake * 0.48  # EV moyen
        else:
            bankroll_sim -= stake

    print("      Mises sur premiers paris (simulation):")
    for i, stake in enumerate(stakes):
        print(f"        Pari {i+1}: {stake:.0f}€")

print("\n🎯 HYPOTHÈSES SUR LE BUG:")
print("   A. Kelly mal calibré: fraction trop élevée")
print("   B. Caps journaliers ignorés ou mal appliqués")
print("   C. Bankroll qui explose → stakes qui explosent")
print("   D. Bug dans le calcul des stakes (récursion)")

print("\n✅ VÉRIFICATION RAPIDE:")
print("   Si on misait 15€ par pari fixe:")
fixed_stake_total = n_bets * 15
fixed_profit = fixed_stake_total * (roi_pct / 100)
fixed_bankroll = initial_bankroll + fixed_profit
print(f"   Total misé: {fixed_stake_total:,}€")
print(f"   Profit: {fixed_profit:,.0f}€")
print(f"   Bankroll finale: {fixed_bankroll:,.0f}€")
print("   → Beaucoup plus réaliste !")
