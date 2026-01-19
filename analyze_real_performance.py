import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection

conn = get_connection()
cur = conn.cursor()

print("=" * 100)
print("ANALYSE PERFORMANCE REELLE DES PARIS")
print("=" * 100)

# Stats globales
cur.execute("""
SELECT
    status,
    COUNT(*) as nb,
    SUM(stake) as total_mise,
    SUM(CASE WHEN status = 'WIN' THEN stake * odds ELSE 0 END) as total_retour
FROM user_bets
WHERE is_simulation = false
GROUP BY status
ORDER BY status
""")

total_mise = 0
total_retour = 0

print("\nPar status:")
for row in cur.fetchall():
    status, nb, mise, retour = row
    mise = mise or 0
    retour = retour or 0
    total_mise += mise
    total_retour += retour
    print(f"  {status:10} | {nb:4} paris | Mise: {mise:8.2f} EUR | Retour: {retour:10.2f} EUR")

if total_mise > 0:
    profit = total_retour - total_mise
    roi = (profit / total_mise) * 100
    taux_reussite = cur.execute("""
        SELECT COUNT(*) * 1.0 / (SELECT COUNT(*) FROM user_bets WHERE is_simulation = false)
        FROM user_bets WHERE is_simulation = false AND status = 'WIN'
    """)
    taux = cur.fetchone()[0] * 100

    print(f"\n{'='*100}")
    print("RESULTAT GLOBAL:")
    print(f"  Taux réussite: {taux:.1f}%")
    print(f"  Mise totale: {total_mise:.2f} EUR")
    print(f"  Retour total: {total_retour:.2f} EUR")
    print(f"  Profit: {profit:.2f} EUR")
    print(f"  ROI: {roi:.2f}%")

# 3 dernières semaines
print(f"\n{'='*100}")
print("PERFORMANCE 21 DERNIERS JOURS:")
print(f"{'='*100}")

cur.execute("""
SELECT
    status,
    COUNT(*) as nb,
    SUM(stake) as total_mise,
    SUM(CASE WHEN status = 'WIN' THEN stake * odds ELSE 0 END) as total_retour
FROM user_bets
WHERE is_simulation = false
  AND created_at >= NOW() - INTERVAL '21 days'
GROUP BY status
ORDER BY status
""")

total_mise_21d = 0
total_retour_21d = 0
nb_paris_21d = 0

for row in cur.fetchall():
    status, nb, mise, retour = row
    mise = mise or 0
    retour = retour or 0
    total_mise_21d += mise
    total_retour_21d += retour
    nb_paris_21d += nb
    print(f"  {status:10} | {nb:4} paris | Mise: {mise:8.2f} EUR | Retour: {retour:10.2f} EUR")

if total_mise_21d > 0:
    profit_21d = total_retour_21d - total_mise_21d
    roi_21d = (profit_21d / total_mise_21d) * 100

    cur.execute("""
        SELECT COUNT(*)
        FROM user_bets
        WHERE is_simulation = false
          AND status = 'WIN'
          AND created_at >= NOW() - INTERVAL '21 days'
    """)
    nb_win_21d = cur.fetchone()[0]
    taux_21d = (nb_win_21d / nb_paris_21d) * 100

    print("\nRÉSULTAT 21 JOURS:")
    print(f"  Nombre de paris: {nb_paris_21d}")
    print(f"  Taux réussite: {taux_21d:.1f}%")
    print(f"  Mise totale: {total_mise_21d:.2f} EUR")
    print(f"  Retour total: {total_retour_21d:.2f} EUR")
    print(f"  Profit: {profit_21d:.2f} EUR")
    print(f"  ROI: {roi_21d:.2f}%")

# Statistiques par cote
print(f"\n{'='*100}")
print("PERFORMANCE PAR TRANCHE DE COTE (21 jours):")
print(f"{'='*100}")

cur.execute("""
SELECT
    CASE
        WHEN odds < 2.0 THEN '< 2.0 (favoris)'
        WHEN odds < 5.0 THEN '2.0-5.0 (moyens)'
        WHEN odds < 10.0 THEN '5.0-10.0 (outsiders)'
        ELSE '>= 10.0 (longshots)'
    END as tranche,
    COUNT(*) as nb_paris,
    SUM(CASE WHEN status = 'WIN' THEN 1 ELSE 0 END) as nb_win,
    SUM(stake) as mise,
    SUM(CASE WHEN status = 'WIN' THEN stake * odds ELSE 0 END) as retour
FROM user_bets
WHERE is_simulation = false
  AND created_at >= NOW() - INTERVAL '21 days'
GROUP BY tranche
ORDER BY tranche
""")

for row in cur.fetchall():
    tranche, nb, nb_win, mise, retour = row
    mise = mise or 0
    retour = retour or 0
    taux = (nb_win / nb * 100) if nb > 0 else 0
    roi_tranche = ((retour - mise) / mise * 100) if mise > 0 else 0
    print(f"\n  {tranche:20} | {nb:3} paris | {nb_win:3} WIN ({taux:.1f}%)")
    print(f"    Mise: {mise:8.2f} EUR | Retour: {retour:10.2f} EUR | ROI: {roi_tranche:+7.2f}%")

conn.close()
