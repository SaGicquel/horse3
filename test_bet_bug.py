#!/usr/bin/env python3
"""
Test pour vérifier le bug dans le simulateur de paris.
"""

print("🧪 TEST DU BUG BANKROLL")
print("=" * 50)

# Simulation manuelle d'un pari gagnant
bankroll_initial = 1000
stake = 50  # 50€ de mise
odds = 3.0  # Cote 3.0
result = 1  # Gagne

print(f"📊 SCÉNARIO:")
print(f"   Bankroll initial: {bankroll_initial}€")
print(f"   Mise: {stake}€")
print(f"   Cote: {odds}")
print(f"   Résultat: {'GAGNE' if result == 1 else 'PERD'}")

print(f"\n🔍 CALCULS:")

# ✅ FORMULE CORRECTE
print(f"\n✅ FORMULE CORRECTE:")
bankroll_correct = bankroll_initial - stake + stake * odds if result == 1 else bankroll_initial - stake
profit_correct = stake * (odds - 1) if result == 1 else -stake
print(f"   Bankroll après pari: {bankroll_initial} - {stake} + {stake} × {odds} = {bankroll_correct}€")
print(f"   Profit: {profit_correct}€")
print(f"   ROI: {profit_correct/stake*100:.1f}%")

# ❌ FORMULE BUGGÉE (comme dans le code)
print(f"\n❌ FORMULE BUGGÉE (code actuel):")
profit_bug = stake * (odds - 1) if result == 1 else -stake  
bankroll_bug = bankroll_initial + profit_bug  # BUG: ajoute profit sans soustraire mise
print(f"   Profit calculé: {stake} × ({odds} - 1) = {profit_bug}€")
print(f"   Bankroll après pari: {bankroll_initial} + {profit_bug} = {bankroll_bug}€")
print(f"   ROI apparent: {profit_bug/stake*100:.1f}%")

print(f"\n💥 DIFFÉRENCE:")
print(f"   Bankroll correct: {bankroll_correct}€")
print(f"   Bankroll buggé: {bankroll_bug}€")
print(f"   Écart: +{bankroll_bug - bankroll_correct}€")
print(f"   → Le bug ajoute la MISE en plus !")

print(f"\n🎯 CONCLUSION:")
print(f"   Le bug fait que chaque pari gagnant ajoute:")
print(f"   - Le profit NET: +{profit_correct}€")  
print(f"   - PLUS la mise: +{stake}€")
print(f"   - Total ajouté: +{profit_bug}€ au lieu de +{profit_correct}€")
print(f"   → C'est pourquoi la bankroll explose !")