#!/usr/bin/env python3
"""
Test de la normalisation des races
"""

import sys
sys.path.insert(0, '/Users/gicquelsacha/horse3')

from scraper_pmu_simple import normalize_race

# Tests
test_cases = [
    ("PUR SANG", "PUR SANG"),
    ("PUR-SANG", "PUR SANG"),
    ("*ANGLO-ARABE*", "ANGLO ARABE"),
    ("ANGLO-ARABE", "ANGLO ARABE"),
    ("anglo-arabe", "ANGLO ARABE"),
    ("  *PUR-SANG*  ", "PUR SANG"),
    ("TROTTEUR FRANCAIS", "TROTTEUR FRANCAIS"),
    ("TROTTEUR-FRANCAIS", "TROTTEUR FRANCAIS"),
]

print("🧪 Tests de normalisation des races\n")

all_passed = True
for input_val, expected in test_cases:
    result = normalize_race(input_val)
    status = "✅" if result == expected else "❌"
    if result != expected:
        all_passed = False
    print(f"{status} '{input_val}' -> '{result}' (attendu: '{expected}')")

if all_passed:
    print("\n🎉 Tous les tests sont passés!")
else:
    print("\n❌ Certains tests ont échoué")
    sys.exit(1)
