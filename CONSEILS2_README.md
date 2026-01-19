# Page Conseils V2 - Algo Brut Optimisé

## 🎯 Vue d'ensemble

La page **Conseils2** est une copie de la page Conseils originale, mais utilise exclusivement l'**algo brut optimisé** validé à **+71% ROI** sur 5 mois de données historiques.

## 📊 Différences avec Conseils V1

| Caractéristique | Conseils V1 (Original) | Conseils V2 (Algo Brut) |
|----------------|------------------------|-------------------------|
| **Backend API** | Port 8000 - `user_app_api.py` | Port 8001 - `user_app_api_v2.py` |
| **Stratégie** | Agent IA + Value filters + Kelly | Algo brut seul |
| **Mises** | Kelly fractional (variable) | Uniforme 10€ |
| **Filtres cotes** | Dynamique selon value | Fixe 7-15 (semi-outsiders) |
| **Seuil proba** | Variable selon Agent IA | Fixe ≥50% |
| **ROI validé** | En cours de test (20 jours) | +71.47% sur 5 mois |
| **Complexité** | Haute (LLM, règles, Kelly) | Basse (algo seul) |

## 🔧 Modifications techniques

### 1. Fichiers créés/modifiés

```
✅ web/frontend/src/pages/Conseils2.jsx          (copie modifiée)
✅ web/frontend/src/App.jsx                       (route ajoutée)
✅ web/frontend/src/components/Navigation.jsx    (lien menu ajouté)
```

### 2. Appels API modifiés

**Avant (Conseils V1):**
```javascript
// Ligne 188
`${API_BASE}/portfolio/today?bankroll=${bankroll}&kelly_profile=${kellyProfile}`

// Ligne 199
`${API_BASE}/picks/today?zone=${userZone}&bankroll=${bankroll}`

// Lignes 247-299
`${API_BASE}/agent/today`
`${API_BASE}/agent/run`
```

**Après (Conseils V2):**
```javascript
// Ligne 183
const today = new Date().toISOString().split('T')[0];
const response = await fetch(`http://localhost:8001/daily-advice-v2?date_str=${today}`);

// Agent IA supprimé
// Pas de Kelly, pas de value filters
```

### 3. Transformation des données

L'API V2 retourne un format simplifié :
```json
[
  {
    "course_id": 123456,
    "race_key": "20251101-VINCENNES-R1-C3",
    "hippodrome": "VINCENNES",
    "heure": "14:30",
    "numero": 4,
    "nom": "GIADA GRIF",
    "cote": 8.2,
    "cote_place": 2.8,
    "proba": 58.2,
    "mise": 10.0,
    "gain_potentiel": 30.57
  }
]
```

Le code frontend transforme ce format pour compatibilité :
```javascript
const formattedBets = picks.map(pick => ({
  race_key: pick.race_key,
  hippodrome: pick.hippodrome,
  heure: pick.heure,
  numero: pick.numero,
  nom: pick.nom,
  cote: pick.cote,
  cote_place: pick.cote_place,
  proba: pick.proba,
  mise_recommandee: pick.mise || 10,
  gain_potentiel: pick.gain_potentiel,
  edge: ((pick.proba / 100) * pick.cote - 1) * 100,
  value_pct: ((pick.proba / 100) * pick.cote - 1) * 100,
  rationale: `Semi-outsider (cote ${pick.cote.toFixed(1)}) avec probabilité ${pick.proba.toFixed(1)}% (algo brut optimisé)`,
}));
```

## 🚀 Utilisation

### Démarrer l'API V2

Si l'API V2 n'est pas démarrée :
```bash
cd /Users/gicquelsacha/horse3
python3 user_app_api_v2.py &
```

Vérifier que l'API fonctionne :
```bash
curl http://localhost:8001/health
# Réponse attendue : {"status":"ok","version":"2.0.0","algo":"Brut optimisé (+71% ROI)"}
```

### Accéder à la page

1. Démarrer le frontend (si pas déjà fait)
2. Ouvrir http://localhost:5173/conseils2
3. La page affichera les paris du jour selon l'algo brut

### Navigation

Un nouveau lien **"Conseils V2 🎯"** apparaît dans la navigation principale entre "Conseils" et "Mes Paris".

## 📈 Configuration de l'algo brut

**Features utilisées :**
- `cote_reference` (cote PMU)
- `cote_log` (log de la cote)
- `distance_m` (distance de la course)
- `age` (âge du cheval)
- `poids_kg` (poids porté)
- `hippodrome_place_rate` (taux de place de l'hippodrome)
- `hippodrome_avg_cote` (cote moyenne de l'hippodrome)

**Hyperparamètres XGBoost :**
```python
{
    'max_depth': 7,
    'learning_rate': 0.04,
    'n_estimators': 350,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}
```

**Filtres appliqués :**
- Cotes : 7.0 ≤ cote ≤ 15.0 (semi-outsiders)
- Probabilité : proba ≥ 50%
- Mise : Uniforme 10€ par pari

## 🎯 Performance validée

**Test sur 5 mois (mai-septembre 2025) :**
```
Période 1 (mai)    : +93.06% ROI, 51 paris ✅ (p=0.0050)
Période 2 (juin)   : +71.23% ROI, 46 paris ✅ (p=0.0029)
Période 3 (juillet): +75.81% ROI, 44 paris ✅ (p=0.0038)
Période 4 (août)   : +50.29% ROI, 43 paris ✅ (p=0.0211)
Période 5 (sept)   : +62.11% ROI, 38 paris ⚠️  (p=0.0572)

GLOBAL : +71.47% ROI, 222 paris, 56.8% win rate
```

**Validation statistique :** 4/5 périodes significatives (p < 0.05)

## ⚠️ Notes importantes

1. **API V2 doit tourner** : Le backend sur port 8001 doit être actif
2. **Pas de mode simulation** : Contrairement à V1, pas de toggle simulation
3. **Pas d'Agent IA** : Pas d'analyse LLM, uniquement l'algo
4. **Mises fixes** : Toujours 10€, pas de Kelly
5. **Test en parallèle** : Permet de comparer V1 vs V2 sur 20 jours réels

## 🔍 Comparaison des systèmes

Pour comparer les deux versions :
```bash
python3 compare_conseils.py --date 2025-11-01
```

Cela affichera côte à côte :
- V1 : Paris avec Agent IA + Kelly
- V2 : Paris avec algo brut seul

## 📝 Fichiers de référence

- **Backend API V2** : `user_app_api_v2.py` (428 lignes)
- **Audit validation** : `audit_ultimate_config.py` (12 checks passés)
- **Comparaison** : `audit_full_system.py` (algo brut vs value system)
- **Frontend** : `web/frontend/src/pages/Conseils2.jsx` (2232 lignes)

## 🎯 Objectif du test

Comparer sur 20 jours réels :
- **Stratégie A (V1)** : Système complet avec IA → ROI inconnu
- **Stratégie B (V2)** : Algo brut seul → +71% ROI validé historique

Le meilleur système sera déployé en production définitive.
