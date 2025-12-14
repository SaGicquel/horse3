# Intégration Estimateur p(place) - Documentation

## Vue d'ensemble

Le système d'estimation p(place) permet de calculer les probabilités qu'un cheval termine dans les X premières places, et de générer des combinaisons optimales pour les paris exotiques (Trio, Quarté, Quinté).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│            /conseils  /exotics  /picks                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│                     Backend FastAPI                              │
│  /exotics/build      POST - Génère des tickets selon budget      │
│  /exotics/advanced   POST - Analyse avancée avec détails         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│           PlaceEstimatorService (web/backend/services/)          │
│  • analyze_race()    - Analyse complète d'une course            │
│  • generate_packs()  - Génère packs SÛR/ÉQUILIBRÉ/RISQUÉ        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│           PlaceProbabilityEstimator (racine/)                    │
│  Estimateurs:                                                    │
│  • Harville (1973)   - Classique, peut surestimer favoris       │
│  • Henery (γ=0.81)   - Corrige biais favori                     │
│  • Stern (λ=0.15)    - Lissage                                  │
│  • Lo-Bacon-Shone    - Réallocation itérative                   │
│                                                                  │
│  Simulateur: Plackett-Luce avec température par discipline       │
│  Monte Carlo: N≥20000 simulations                               │
└─────────────────────────────────────────────────────────────────┘
```

## Estimateurs par discipline

| Discipline | Estimateur | Température | Justification |
|------------|-----------|-------------|---------------|
| Plat       | Henery    | 0.95        | Favoris souvent surestimés |
| Trot       | Lo-Bacon-Shone | 1.05   | Plus d'incertitude tactique |
| Obstacle   | Stern     | 1.10        | Forte variance (chutes, etc.) |

## Endpoints API

### POST /exotics/build

Génère des tickets de paris exotiques optimisés.

**Request:**
```json
{
  "budget": 50.0,
  "pack": "EQUILIBRE",  // SUR, EQUILIBRE, RISQUE
  "race_key": "2025-01-15_PARIS_R1C3",  // optionnel
  "use_advanced_estimator": true,
  "n_simulations": 10000
}
```

**Response:**
```json
{
  "tickets": [
    {
      "type": "Trio Ordre",
      "bet_type": "tierce",
      "combo": [{"nom": "GOLDEN STAR", "numero": 1}, ...],
      "selections": ["GOLDEN STAR", "SILVER BOLT", "BRONZE FLASH"],
      "numeros": [1, 2, 3],
      "mise": 15.0,
      "prob_pct": 8.14,
      "ev_pct": -16.0,
      "description": "Combo EV+8.1% - Prob 8.14%"
    }
  ],
  "race_key": "2025-01-15_PARIS_R1C3",
  "discipline": "plat",
  "budget": 50.0,
  "pack": "EQUILIBRE",
  "budget_utilise": 50.0,
  "ev_totale": -45.2,
  "advanced_estimator": true,
  "estimator_used": "HeneryEstimator",
  "n_simulations": 10000
}
```

### POST /exotics/advanced

Analyse avancée avec toutes les probabilités détaillées.

**Request:**
```json
{
  "race_key": "2025-01-15_PARIS_R1C3",
  "estimator": "auto",  // harville, henery, stern, lbs, auto
  "n_simulations": 20000,
  "min_ev_percent": 5.0
}
```

**Response:**
```json
{
  "race_key": "...",
  "hippodrome": "PARIS",
  "discipline": "plat",
  "nb_partants": 12,
  "estimator_used": "HeneryEstimator",
  "n_simulations": 20000,
  "temperature": 0.95,
  "calibration": {"brier_score": 0.15, "ece": 0.03},
  "partants": [
    {
      "numero": 1,
      "nom": "GOLDEN STAR",
      "cote": 2.5,
      "p_win": 0.35,
      "p_place_2": 0.65,
      "p_place_3": 0.82,
      "score_model": 75
    }
  ],
  "top_combos_ev_positive": [...],
  "trio_top5": [...],
  "quarte_top5": [...],
  "quinte_top5": [...],
  "recommendations": [
    "⚠️ Favori dominant (GOLDEN STAR, p=35%) - Trio ordre recommandé",
    "💰 3 combinaisons avec EV > 10% détectées",
    "✅ Modèle bien calibré (Brier < 0.15)"
  ]
}
```

## Configuration

### config/pro_betting.yaml

```yaml
place_estimators:
  # Températures Plackett-Luce par discipline
  temperature_plat: 0.95
  temperature_trot: 1.05
  temperature_obstacle: 1.10
  
  # Paramètres estimateurs
  henery_gamma: 0.81
  stern_lambda: 0.15
  
  # Sélection auto par discipline
  estimator_plat: henery
  estimator_trot: lbs
  estimator_obstacle: stern
  
  # Simulations
  n_simulations_default: 20000
  takeout_rate: 0.16
```

## Packs de Paris

| Pack | Description | Seuil EV | Seuil Prob | Tickets |
|------|-------------|----------|------------|---------|
| SÛR | Combinaisons à forte probabilité | ≥0% | ≥0.3% | 3 |
| ÉQUILIBRÉ | Équilibre prob/rendement | ≥5% | ≥0.1% | 5 |
| RISQUÉ | Gros rapports potentiels | ≥10% | ≥0.05% | 6 |

## Algorithmes Clés

### 1. Conversion cotes → probabilités

```python
p_win = 1 / cote
p_normalized = p_win / sum(p_win)  # Supprime overround
```

### 2. Blend avec scores modèle

```python
factor = 1.0 + (score - 50) / 50
p_adjusted = p_market * factor
p_final = p_adjusted / sum(p_adjusted)
```

### 3. Calcul EV

```python
payout_expected = (1 - takeout) / prob
EV = prob * payout_expected - 1
```

## Tests

```bash
# Tests unitaires
cd /Users/gicquelsacha/horse3
.venv/bin/python -m pytest tests/test_place_estimator.py -v

# Test service
.venv/bin/python web/backend/services/place_estimator_service.py
```

## Fichiers Créés

| Fichier | Description |
|---------|-------------|
| `place_probability_estimator.py` | Module principal (~900 lignes) |
| `integrate_place_estimator.py` | Script d'intégration |
| `learn_pl_temperature.py` | Apprentissage températures |
| `web/backend/services/place_estimator_service.py` | Service API |
| `tests/test_place_estimator.py` | Tests unitaires |
| `config/pro_betting.yaml` | Configuration (mis à jour) |
| `config/loader.py` | Chargeur config (mis à jour) |

## Références

- Harville, D.A. (1973) "Assigning Probabilities to the Outcomes of Multi-Entry Competitions"
- Henery, R.J. (1981) "Permutation Probabilities as Models for Horse Races"
- Lo, V.S.Y., Bacon-Shone, J. (1994) "Approximating the Ordering Probabilities"
- Stern, H. (1990) "Models for Distributions on Permutations"
