# Import IFCE - Résumé

**Date**: 7 novembre 2025

## Données IFCE importées

### Filtres appliqués
- **Vivants**: Chevaux sans DATE_DE_DECES
- **Non-consommation**: CHE_COCONSO ≠ 'O' (exclusion chevaux destinés à la consommation)
- **Âge course**: Entre 2 et 15 ans (calculé depuis DATE_DE_NAISSANCE)

### Statistiques d'import

| Métrique | Valeur |
|----------|--------|
| **Chevaux importés** | 204 835 |
| **Chevaux liés (horse_ifce)** | 197 380 |
| **Décédés (ignorés)** | 736 928 |
| **Consommation (ignorés)** | 2 739 805 |
| **Âge hors limites (ignorés)** | 145 605 |

### Structure des données

**Colonnes CSV source**:
- RACE, SEXE, ROBE, DATE_DE_NAISSANCE, PAYS_DE_NAISSANCE, NOM, CHE_COCONSO, DATE_DE_DECES

**Tables remplies**:
- `core.horse` (source='IFCE'): chevaux de base normalisés
- `core.horse_ifce`: liaison avec identifiant IFCE synthétique

### Normalisation appliquée
- Noms: Unicode NFKD, uppercase, [^A-Z0-9] → espace, collapse whitespace
- Dates: DD/MM/YYYY → YYYY-MM-DD (PostgreSQL DATE)
- Pays: 3 premiers caractères uppercase

## Matching IFCE ↔ PMU

**En cours**: Matching de 197 380 chevaux IFCE contre 694 chevaux PMU collectés le 2025-11-05.

**Méthodes**:
1. Exact match (nom normalisé + année naissance ±1)
2. Trigram similarity (seuil confirmé ≥ 0.90)
3. Levenshtein fallback (ratio ≥ 95%)

**Optimisations**:
- Index par première lettre du nom
- Limite max 50 comparaisons par cheval IFCE
- Commit par batch de 1000

---

**Fichier source**: `/Users/gicquelsacha/horse3/fichier-des-equides.csv`  
**Script**: `etl/load_ifce_csv.py`  
**Matching**: `match/match_horses_fast.py`
