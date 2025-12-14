def select_preoff_market_odds(cote_finale, cote_reference, place_finale=None):
    """
    Garantit l'usage d'un snapshot pré-off pour le marché.
    - Si place_finale (ou résultat) est présent, on refuse d'utiliser la cote post-off.
    - Préférence pour cote_reference (pré-off), sinon cote_finale uniquement si aucune issue n'est connue.
    """
    if place_finale is not None:
        raise ValueError("post-off odds detected (place_finale non nul) – anti-fuite activé")
    return cote_reference or cote_finale
