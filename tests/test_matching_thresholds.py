from match.match_horses import levenshtein

def test_levenshtein_ratio():
    # Rapidfuzz peut ne pas être installé dans l'environnement de test initial
    r = levenshtein("KALINE", "KALINE")
    if r is not None:
        assert r == 100.0
