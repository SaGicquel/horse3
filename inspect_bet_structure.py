import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection

conn = get_connection()
cur = conn.cursor()

# Voir les données brutes
cur.execute("""
SELECT id, race_key, selection, stake, odds, status, notes, created_at
FROM user_bets
WHERE is_simulation = false AND status = 'WIN'
ORDER BY created_at DESC
LIMIT 10
""")

print("PARIS GAGNANTS - DONNEES BRUTES:")
print("=" * 120)
for row in cur.fetchall():
    bet_id, race, sel, stake, odds, status, notes, created = row
    print(f"\nID {bet_id} | {created} | {race}")
    print(f"  Selection: {sel}")
    print(f"  Stake: {stake} EUR")
    print(f"  Odds: {odds}")
    print(f"  Status: {status}")
    print(f"  Notes: {notes}")

# Vérifier toutes les colonnes
print("\n\n" + "=" * 120)
print("TOUTES LES COLONNES:")
print("=" * 120)
cur.execute("SELECT * FROM user_bets WHERE is_simulation = false LIMIT 1")
cols = [desc[0] for desc in cur.description]
row = cur.fetchone()

for col, val in zip(cols, row):
    print(f"  {col:20} = {val}")

conn.close()
