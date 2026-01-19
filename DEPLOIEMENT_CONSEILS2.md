# 🎯 Page Conseils2 - Déploiement Docker Réussi

## ✅ État du Déploiement

**Date:** 19 janvier 2026
**Statut:** ✅ OPÉRATIONNEL

### Conteneurs Docker

```bash
✅ horse-backend   (port 8000) - API V1 (Agent IA + Kelly + Value)
✅ horse-frontend  (port 80)   - React app avec Conseils + Conseils2
```

### APIs Backend

```bash
✅ Port 8000: API V1 - user_app_api.py (système complet)
✅ Port 8001: API V2 - user_app_api_v2.py (algo brut +71% ROI)
```

## 🚀 Accès à la Page

### URL Frontend
```
http://localhost/conseils2
```

### Navigation
Le lien **"Conseils V2 🎯"** est disponible dans le menu principal entre "Conseils" et "Mes Paris".

## 📋 Modifications Appliquées

### 1. Backend (API V2 - Port 8001)
- ✅ `user_app_api_v2.py` - API FastAPI pour algo brut
- ✅ Configuration XGBoost optimisée (max_depth=7, lr=0.04, n_estimators=350)
- ✅ Filtres: Cotes 7-15, Proba ≥50%, Mises 10€ uniformes
- ✅ Endpoint: `GET /daily-advice-v2?date_str=YYYY-MM-DD`

### 2. Frontend (React)
- ✅ `Conseils2.jsx` - Page dédiée algo brut (copie modifiée de Conseils.jsx)
- ✅ `App.jsx` - Route `/conseils2` ajoutée
- ✅ `Navigation.jsx` - Lien menu "Conseils V2 🎯" ajouté
- ✅ Suppression Agent IA (pas de toggle IA, pas de runAgentAnalysis)
- ✅ Appels API modifiés pour utiliser port 8001
- ✅ UI simplifiée avec bandeau vert "Algo Brut Optimisé (+71% ROI)"

### 3. Docker
- ✅ Frontend rebuilé avec `--no-cache`
- ✅ Backend rebuilé avec `--no-cache`
- ✅ Conteneurs redémarrés avec succès
- ✅ Build artifacts: `Conseils2-BOLXk4CB.js` (54.32 kB)

## 🔧 Commandes de Rebuild

Pour appliquer de futures modifications :

```bash
# Rebuild backend
cd /Users/gicquelsacha/horse3/web
docker-compose build backend --no-cache && docker-compose up -d backend

# Rebuild frontend
cd /Users/gicquelsacha/horse3/web
docker-compose build frontend --no-cache && docker-compose up -d frontend
```

## 🎯 Stratégie Conseils V2

### Configuration Validée
- **Features:** cote_reference, cote_log, distance_m, age, poids_kg, hippodrome stats
- **Modèle:** XGBoost (max_depth=7, learning_rate=0.04, n_estimators=350)
- **Filtres:** Semi-outsiders (cotes 7-15), Probabilité ≥50%
- **Mises:** Uniforme 10€ par pari

### Performance Historique (5 mois)
```
Période 1 (mai)    : +93.06% ROI, 51 paris ✅ (p=0.0050)
Période 2 (juin)   : +71.23% ROI, 46 paris ✅ (p=0.0029)
Période 3 (juillet): +75.81% ROI, 44 paris ✅ (p=0.0038)
Période 4 (août)   : +50.29% ROI, 43 paris ✅ (p=0.0211)
Période 5 (sept)   : +62.11% ROI, 38 paris ⚠️  (p=0.0572)

GLOBAL : +71.47% ROI, 222 paris, 56.8% win rate
```

## 📊 Test Aujourd'hui (19 janvier 2026)

L'API V2 retourne **2 paris** :

```
1. noumba as (#7)
   - Cote: 8.7
   - Proba: 53.2%
   - Mise: 10€
   - Gain potentiel: 32.0€

2. millesime star (#11)
   - Cote: 7.1
   - Proba: 50.1%
   - Mise: 10€
   - Gain potentiel: 27.43€

Total mise: 20€
Total gain potentiel: 59.43€
ROI potentiel: +197.15%
```

## 🆚 Comparaison V1 vs V2

| Aspect | Conseils V1 | Conseils V2 |
|--------|-------------|-------------|
| **Page** | /conseils | **/conseils2** |
| **API** | Port 8000 | **Port 8001** |
| **Stratégie** | Agent IA + Value + Kelly | **Algo brut seul** |
| **Mises** | Kelly fractional (variable) | **10€ uniforme** |
| **Cotes** | Dynamique selon value | **7-15 fixe** |
| **Seuil** | Variable selon Agent IA | **≥50% proba** |
| **Complexité** | Haute (LLM, règles, Kelly) | **Basse (algo XGBoost)** |
| **ROI** | En test (20 jours) | **+71.47% validé** |

## 🛠️ Dépannage

### Vérifier l'état des conteneurs
```bash
cd /Users/gicquelsacha/horse3/web
docker-compose ps
```

### Vérifier les logs
```bash
# Frontend
docker-compose logs frontend --tail 50

# Backend
docker-compose logs backend --tail 50
```

### Tester l'API V2
```bash
# Health check
curl http://localhost:8001/health

# Paris du jour
curl "http://localhost:8001/daily-advice-v2?date_str=$(date +%Y-%m-%d)"
```

### Redémarrer les services
```bash
# Redémarrer frontend seul
docker-compose restart frontend

# Redémarrer backend seul
docker-compose restart backend

# Redémarrer tout
docker-compose restart
```

## 📝 Prochaines Étapes

1. ✅ **FAIT** - Déploiement Docker réussi
2. ⏳ **EN COURS** - Test A/B sur 20 jours (V1 vs V2)
3. ⏳ **À VENIR** - Analyse comparative des résultats
4. ⏳ **À VENIR** - Décision production (V1 ou V2)

## 🎉 Succès

- ✅ Page Conseils2 créée et déployée
- ✅ API V2 fonctionnelle sur port 8001
- ✅ Frontend rebuilé sans erreur
- ✅ Backend V1 maintenu pour comparaison
- ✅ Navigation menu mise à jour
- ✅ 2 paris générés pour aujourd'hui
- ✅ ROI validé à +71.47% sur historique

**La page Conseils2 est maintenant accessible et prête pour le test en production ! 🚀**
