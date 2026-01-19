import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection

conn = get_connection()
cur = conn.cursor()

print("ANALYSE DES VRAIS PARIS")
print("=" * 100)

# Stats par status
cur.execute("""
SELECT
    status,
    COUNT(*) as nb,
    SUM(stake) as total_mise,
    SUM(CASE WHEN status = 'won' THEN stake * odds ELSE 0 END) as total_retour
FROM user_bets
WHERE is_simulation = false
GROUP BY status
ORDER BY status
""")

print("\nPar status:")
total_mise_all = 0
total_retour_all = 0
for row in cur.fetchall():
    status, nb, mise, retour = row
    mise = mise or 0
    retour = retour or 0
    total_mise_all += mise
    total_retour_all += retour
    print(f"  {status:15} | {nb:4} paris | Mise: {mise:8.2f} EUR | Retour: {retour:9.2f} EUR")

if total_mise_all > 0:
    profit = total_retour_all - total_mise_all
    roi = (profit / total_mise_all) * 100
    print("\nGLOBAL:")
    print(f"  Mise totale: {total_mise_all:.2f} EUR")
    print(f"  Retour total: {total_retour_all:.2f} EUR")
    print(f"  Profit: {profit:.2f} EUR")
    print(f"  ROI: {roi:.2f}%")

# 21 derniers jours
print("\n" + "=" * 100)
print("DERNIERS 21 JOURS")
print("=" * 100)

cur.execute("""
SELECT
    status,
    COUNT(*) as nb,
    SUM(stake) as total_mise,
    SUM(CASE WHEN status = 'won' THEN stake * odds ELSE 0 END) as total_retour
FROM user_bets
WHERE is_simulation = false
  AND created_at >= NOW() - INTERVAL '21 days'
GROUP BY status
ORDER BY status
""")

total_mise_21d = 0
total_retour_21d = 0
for row in cur.fetchall():
    status, nb, mise, retour = row
    mise = mise or 0
    retour = retour or 0
    total_mise_21d += mise
    total_retour_21d += retour
    print(f"  {status:15} | {nb:4} paris | Mise: {mise:8.2f} EUR | Retour: {retour:9.2f} EUR")

if total_mise_21d > 0:
    profit_21d = total_retour_21d - total_mise_21d
    roi_21d = (profit_21d / total_mise_21d) * 100
    print("\n21 JOURS:")
    print(f"  Mise totale: {total_mise_21d:.2f} EUR")
    print(f"  Retour total: {total_retour_21d:.2f} EUR")
    print(f"  Profit: {profit_21d:.2f} EUR")
    print(f"  ROI: {roi_21d:.2f}%")

# Exemples de paris gagnants
print("\n" + "=" * 100)
print("EXEMPLES PARIS GAGNANTS (derniers 10)")
print("=" * 100)

cur.execute("""
SELECT race_key, event_date, selection, stake, odds, created_at
FROM user_bets
WHERE is_simulation = false AND status = 'won'
ORDER BY created_at DESC
LIMIT 10
""")

for row in cur.fetchall():
    race_key, event_date, selection, stake, odds, created = row
    gain = stake * odds
    print(f"\n  {created} | {race_key}")
    print(f"    Cheval: {selection} | Mise: {stake} EUR @ {odds} => Gain: {gain:.2f} EUR")

conn.close()
