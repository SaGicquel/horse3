#!/usr/bin/env python3
"""
Détection du look-ahead bias dans le modèle
"""

import sys

sys.path.append("/Users/gicquelsacha/horse3")
import pickle
import numpy as np

print("=" * 100)
print("ANALYSE DU LOOK-AHEAD BIAS")
print("=" * 100)

# Charger le modèle
with open("/Users/gicquelsacha/horse3/models_optimized/cat_best.pkl", "rb") as f:
    model = pickle.load(f)

# Voir l'importance des features
try:
    feature_importance = model.get_feature_importance()
    feature_names = [
        "cote_reference",
        "cote_finale",
        "distance_m",
        "age",
        "cote_log",
        "cote_drift",
        "discipline_enc",
        "sexe_enc",
        "hippodrome_code_enc",
        "categorie_cote_enc",
    ]

    print("\nIMPORTANCE DES FEATURES:")
    print("-" * 100)

    importance_pairs = list(zip(feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    for name, importance in importance_pairs:
        marker = " ⚠️  SUSPECT!" if "finale" in name or "drift" in name else ""
        print(f"  {name:30} {importance:>10.2f}{marker}")

    print("\n" + "=" * 100)
    print("PROBLÈMES DÉTECTÉS:")
    print("=" * 100)

    print("\n1. ❌ cote_finale - Cette cote n'est connue qu'APRÈS le départ de la course!")
    print("   La cote finale intègre les derniers mouvements de paris.")
    print("   On ne peut PAS l'utiliser pour prédire AVANT la course.")

    print("\n2. ❌ cote_drift (finale - reference) - Utilise cote_finale donc LEAK!")
    print("   Le drift de cote se produit pendant la course/juste avant.")

    print("\n3. ✅ cote_reference - OK, disponible avant la course")
    print("4. ✅ age, distance, discipline, etc. - OK, connus avant")

    print("\n" + "=" * 100)
    print("SOLUTION:")
    print("=" * 100)
    print("\n➡️  Réentraîner le modèle EN UTILISANT UNIQUEMENT:")
    print("   - cote_reference (pas finale)")
    print("   - Features structurelles (age, distance, hippodrome, discipline)")
    print("   - Stats historiques du cheval/jockey/entraîneur")
    print("\n➡️  EXCLURE:")
    print("   - cote_finale")
    print("   - cote_drift")
    print("   - Toute info post-départ")

except Exception as e:
    print(f"Erreur: {e}")
    print("\nLe modèle CatBoost ne permet pas d'extraire l'importance facilement.")
    print("Mais le problème est clair:")

print("\n" + "=" * 100)
print("FEATURES UTILISÉES ACTUELLEMENT:")
print("=" * 100)
print("""
1. cote_reference      ✅ OK - Disponible avant course
2. cote_finale         ❌ LEAK - Connue après départ
3. distance_m          ✅ OK
4. age                 ✅ OK
5. cote_log            ✅ OK - Dérivée de cote_reference
6. cote_drift          ❌ LEAK - Utilise cote_finale
7. discipline_enc      ✅ OK
8. sexe_enc            ✅ OK
9. hippodrome_code_enc ✅ OK
10. categorie_cote_enc ✅ OK - Dérivée de cote_reference
""")

print("\n" + "=" * 100)
print("EXPLICATION DU ROI IRRÉALISTE:")
print("=" * 100)
print("""
Le modèle voit la cote_finale et peut donc "prédire le futur":

Exemple:
- Avant course: cote_reference = 5.0
- Pendant paris: mouvement vers favori
- Finale: cote_finale = 3.2
- Résultat: Cheval gagne

Le modèle apprend: "Si cote_finale < cote_reference ET cote basse => WIN"
=> Il détecte les chevaux qui sont devenus favoris juste avant le départ
=> C'est de la triche involontaire !

ROI réel attendu avec seulement cote_reference: 5-20% (pas 222%)
""")
