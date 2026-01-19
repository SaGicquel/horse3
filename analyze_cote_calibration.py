import sys

sys.path.append("/Users/gicquelsacha/horse3")
from db_connection import get_connection
import pandas as pd

conn = get_connection()

query = """
SELECT
    CASE
        WHEN cote_reference <= 3 THEN '0-3 (favoris)'
        WHEN cote_reference <= 6 THEN '3-6'
        WHEN cote_reference <= 10 THEN '6-10'
        WHEN cote_reference <= 20 THEN '10-20'
        ELSE '>20'
    END as tranche_cote,
    COUNT(*) as nb_courses,
    SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) as nb_places,
    AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) * 100 as pct_place,
    AVG(cote_reference) as cote_moy
FROM cheval_courses_seen
WHERE cote_reference IS NOT NULL
  AND place_finale IS NOT NULL
  AND annee >= 2025
GROUP BY tranche_cote
ORDER BY cote_moy
"""

df = pd.read_sql(query, conn)

print("RELATION COTE vs PLACEMENT (2025):")
print("=" * 100)
print(df.to_string(index=False))
print()
print("VÉRIFICATION CALIBRATION:")
print("  - Si cote 2.0 => prob réelle devrait être ~50% (1/2.0)")
print("  - Si cote 5.0 => prob réelle devrait être ~20% (1/5.0)")
print("  - Si cote 10.0 => prob réelle devrait être ~10% (1/10.0)")
print()

# Calculer le ROI théorique attendu
print("ESPÉRANCE DE GAIN PAR TRANCHE:")
print("=" * 100)

for _, row in df.iterrows():
    tranche = row["tranche_cote"]
    pct = row["pct_place"] / 100
    cote_moy = row["cote_moy"]

    # ROI théorique si on mise 1€ sur tous les chevaux de cette tranche
    esperance_retour = pct * cote_moy  # prob_gagner * cote
    roi = (esperance_retour - 1) * 100

    print(
        f"{tranche:15} | Cote moy: {cote_moy:6.2f} | Place: {pct*100:5.1f}% | ROI attendu: {roi:+7.2f}%"
    )

print()
print("EXPLICATION ROI +226%:")
print("=" * 100)

# Chercher si certaines tranches sont vraiment profitables
print("\nSi le modèle arrive à sélectionner UNIQUEMENT les meilleurs chevaux")
print("dans chaque tranche, il pourrait battre le marché.")
print()
print("Mais +226% suggère un problème de données ou de calcul.")
print()

# Vérifier les données test spécifiquement
print("ANALYSE DONNÉES TEST (>= 2025-12-15):")
print("=" * 100)

query_test = """
SELECT
    CASE
        WHEN cote_reference <= 5 THEN 'favoris <= 5'
        WHEN cote_reference <= 10 THEN 'moyens 5-10'
        ELSE 'outsiders > 10'
    END as type,
    COUNT(*) as nb,
    AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) * 100 as pct_place,
    AVG(cote_reference) as cote_moy,
    SUM(CASE WHEN place_finale <= 3 THEN cote_reference ELSE 0 END) / COUNT(*) as esperance
FROM cheval_courses_seen
WHERE cote_reference IS NOT NULL
  AND place_finale IS NOT NULL
  AND race_key >= '2025-12-15'
GROUP BY type
ORDER BY cote_moy
"""

df_test = pd.read_sql(query_test, conn)
print(df_test.to_string(index=False))

conn.close()

print()
print("CONCLUSION:")
print("  Si ROI reste > 100% même sans cote_finale, il y a 3 possibilités:")
print("  1. Les données test ont un biais (période trop courte, échantillon non représentatif)")
print("  2. Le modèle sur-fit malgré la validation")
print("  3. Il reste une fuite de données subtile")
