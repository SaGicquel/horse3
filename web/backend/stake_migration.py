import psycopg2
from decimal import Decimal


def get_kelly_stake(odds, bankroll=1000):
    odds = float(odds)
    if odds <= 1.0:
        return 5

    # Courbe simplifiée pour matcher la simulation précédente:
    # 1.5 -> 30, 2.0 -> 25, 3.0 -> 20, 5.0 -> 15, 10.0 -> 10, 15.0 -> 5
    if odds < 1.4:
        return 35
    if odds < 1.7:
        return 30
    if odds < 2.3:
        return 25
    if odds < 3.5:
        return 20
    if odds < 6.0:
        return 15
    if odds < 12.0:
        return 10
    return 5


# Données récupérées (id | odds)
data = """
 226 |  2.20
 214 |  2.10
 219 |  1.20
 218 |  2.60
 228 |  1.70
 217 |  1.50
 216 |  3.00
 238 |  1.90
 224 |  8.60
 229 |  7.90
 215 |  1.50
 221 |  1.50
 220 |  6.80
 225 | 15.00
 222 |  5.00
 213 |  6.70
 230 | 10.00
 223 |  1.90
 227 | 10.00
 247 |  2.20
 243 |  1.50
 239 |  1.60
 237 |  6.20
 245 |  1.50
 242 |  5.30
 236 |  9.00
 235 |  3.00
 234 | 16.00
 240 |  8.80
 244 |  7.80
 210 |  1.60
 241 |  8.20
 252 |  1.30
 246 |  1.60
 211 |  1.90
 254 |  1.60
 212 |  2.70
 250 |  5.70
 249 |  6.70
 255 |  1.60
 253 |  2.80
 251 |  1.50
 248 |  2.30
 256 |  5.50
 261 |  4.50
 259 |  2.00
 258 |  1.40
 257 | 12.00
 260 |  1.80
"""

sql_updates = []
for line in data.strip().split("\n"):
    parts = line.split("|")
    if len(parts) == 2:
        bet_id = parts[0].strip()
        odds = parts[1].strip()
        new_stake = get_kelly_stake(odds)
        sql_updates.append(f"UPDATE user_bets SET stake = {new_stake} WHERE id = {bet_id};")

print("\n".join(sql_updates))
