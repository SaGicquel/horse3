#!/usr/bin/env python3
"""
Analyse des vrais paris stockés dans user_bets
"""

import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection

conn = get_connection()
cur = conn.cursor()

print("=" * 100)
print("STRUCTURE TABLE user_bets")
print("=" * 100)

# Structure de la table
cur.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'user_bets'
    ORDER BY ordinal_position
""")

for col, dtype, nullable in cur.fetchall():
    print(f"{col:30} {dtype:20} NULL: {nullable}")

print("\n" + "=" * 100)
print("STATISTIQUES GLOBALES")
print("=" * 100)

# Stats globales
cur.execute("""
    SELECT
        COUNT(*) as total_bets,
        COUNT(CASE WHEN status = 'won' THEN 1 END) as won,
        COUNT(CASE WHEN status = 'lost' THEN 1 END) as lost,
        COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
        COALESCE(SUM(stake), 0) as total_staked,
        COALESCE(SUM(CASE WHEN status = 'won' THEN payout ELSE 0 END), 0) as total_returned
    FROM user_bets
""")

row = cur.fetchone()
total, won, lost, pending, staked, returned = row

print(f"Total paris: {total}")
print(f"  Gagnés: {won}")
print(f"  Perdus: {lost}")
print(f"  En cours: {pending}")

if staked and staked > 0:
    profit = returned - staked
    roi = (profit / staked) * 100
    print(f"\nTotal misé: {staked:.2f}€")
    print(f"Total retourné: {returned:.2f}€")
    print(f"Profit: {profit:.2f}€")
    print(f"ROI: {roi:.2f}%")
else:
    print("\nAucun paris trouvé dans la base")

# Regarder les 10 derniers paris
print("\n" + "=" * 100)
print("10 DERNIERS PARIS")
print("=" * 100)

cur.execute("""
    SELECT
        id,
        user_id,
        created_at,
        stake,
        odds,
        status,
        payout,
        race_key,
        cheval_id
    FROM user_bets
    ORDER BY created_at DESC
    LIMIT 10
""")

for row in cur.fetchall():
    bet_id, user_id, created, stake, odds, status, payout, race_key, cheval_id = row
    print(f"\nBet #{bet_id} - {created}")
    print(f"  User: {user_id}")
    print(f"  Race: {race_key}, Cheval: {cheval_id}")
    print(f"  Mise: {stake}€ @ {odds} - Status: {status}")
    if payout:
        print(f"  Retour: {payout}€ (Profit: {payout - stake:.2f}€)")

# Vérifier s'il y a des paris pour sachagicquel152@gmail.com
print("\n" + "=" * 100)
print("PARIS POUR sachagicquel152@gmail.com")
print("=" * 100)

cur.execute("""
    SELECT u.email, COUNT(ub.*) as num_bets
    FROM users u
    LEFT JOIN user_bets ub ON u.id = ub.user_id
    WHERE u.email = 'sachagicquel152@gmail.com'
    GROUP BY u.email
""")

row = cur.fetchone()
if row:
    email, num_bets = row
    print(f"Email: {email}")
    print(f"Nombre de paris: {num_bets}")
else:
    print("Aucun utilisateur trouvé avec cet email")

conn.close()
