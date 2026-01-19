# Réponses aux Questions ChatGPT - Configuration Agent IA

Ce document répond à toutes les questions posées par ChatGPT pour configurer un agent IA en complément de l'algorithme d'analyse existant du projet **horse3**.

---

## 1) Objectif réel et définition de "meilleur bet"

### Q: Ton objectif principal, c'est : ROI, régularité, hit-rate, faible drawdown, ou "trouver de la value" ?

**Réponse basée sur le code:** Le système vise principalement le **ROI** et la **value betting**. La configuration dans `config/pro_betting.yaml` définit clairement:
- `value_cutoff: 0.10` (seuil value minimum ≥ 10% pour profil ULTRA_SUR)
- Calcul EV = proba × payout - coût
- Métriques de backtest: ROI, Sharpe ratio, max drawdown

Les backtests récents montrent: ROI +30-67%, Sharpe ~5.03, drawdown max ~18%.

### Q: Tu veux optimiser quoi exactement ?

**Réponse:** Le système optimise actuellement:
- **Profit net / variance** via Kelly fractionnel (15-50% selon profil)
- **Max drawdown** avec stop-loss (20-30% selon zone bankroll)
- **EV lift** vs marché (mesuré à +53% dans les calibration reports)

### Q: Tu paries pour toi uniquement, ou tu veux en faire un produit pour d'autres ?

**Réponse:** Le système est conçu comme un **produit multi-utilisateurs** avec:
- Frontend React accessible
- Système de comptes utilisateurs (`users`, `user_settings`, `user_bets`)
- Profils de risque personnalisables (PRUDENT/STANDARD/AGRESSIF)
- API publique pour servir les prédictions

### Q: Tu veux que l'agent propose : sélections, tickets, mises ?

**Réponse:** Le système propose **les trois**:
1. **Sélections** (`betting_advisor.py` - `generer_recommandations()`)
2. **Tickets** (`exotic_ticket_generator.py` - Trio/Quinté via Monte Carlo)
3. **Mises** (`betting_policy.py` - `select_portfolio_from_picks()` avec Kelly)

### Q: "Meilleur bet" = plus forte probabilité de gagner ou meilleure value ?

**Réponse:** **Meilleure value**. Le système calcule:
```python
value = p_model * odds - 1  # EV positive si value > 0
```
Les seuils de filtrage sont basés sur `value_cutoff` (2-15% selon zone/profil).

### Q: Quel horizon : quotidien, anticipation (la veille) ?

**Réponse:** **Quotidien** principalement. Le scraper récupère les courses du jour (`scraper_pmu_simple.py` - `discover_reunions(date_iso)`). Les cotes utilisées sont les cotes finales pré-départ.

---

## 2) Périmètre PMU / types de paris / contraintes

### Q: Quels types de paris PMU ?

**Réponse:** Configurés dans `pro_betting.yaml`:
- **Simple Gagnant**
- **Simple Placé** (préféré pour zones micro/small)
- **E/P (GAGNANT-PLACÉ)** (préféré pour zone full)
- **Couplé/Trio/Quinté** via `exotic_ticket_generator.py`

### Q: Tu veux couvrir plat / trot / obstacle ?

**Réponse:** **Tous les trois** sont supportés avec calibration spécifique:
```yaml
blend_alpha_plat: 0.0      # α pour courses plat
blend_alpha_trot: 0.4      # α pour courses trot
blend_alpha_obstacle: 0.4  # α pour courses obstacle
```
Les estimateurs Plackett-Luce ont aussi des températures par discipline.

### Q: Tu veux filtrer par : hippodromes, distances, pistes, etc. ?

**Réponse:** La base de données contient toutes ces informations:
- `hippodrome_code`, `hippodrome_nom`
- `distance_m`, `type_piste`, `profil_piste`, `corde`
- `etat_piste`, `meteo`, `penetrometre`
- `classe_course`, `conditions_course`

Le filtrage est possible mais pas systématiquement implémenté dans la politique de mise actuelle.

### Q: Tu as des contraintes "métier" ?

**Réponse:** Oui, configurées dans `pro_betting.yaml`:
- `max_bets_per_day: 2-6` selon zone bankroll
- `max_bets_per_race: 2`
- `value_cutoff_win: 2-15%` selon zone
- `min_proba_model: 8-20%` selon zone
- `max_odds_win: 5-15` selon zone

### Q: Tu acceptes des bets "longshots" ou tu préfères "safe" ?

**Réponse:** **Dépend du profil et de la zone bankroll**:
- **Micro (<50€):** Cotes max 5, proba min 20% → Safe
- **Small (50-500€):** Cotes max 6, proba min 18% → Safe
- **Full (>500€):** Cotes max 15, proba min 8% → Longshots acceptés

### Q: Tu joues avant la course, live, ou uniquement pré-course ?

**Réponse:** **Pré-course uniquement**. Les cotes utilisées sont `cote_finale` (dernière cote avant départ). Pas de live betting implémenté.

---

## 3) Données : ce que tu as vraiment

### Q: Ton historique contient quoi exactement ?

**Réponse:** La table `cheval_courses_seen` contient **~140 colonnes** incluant:

**Résultats complets:**
- `place_finale`, `temps_str`, `temps_sec`, `reduction_km_sec`
- `ecart_premier`, `ecart_precedent`, `vitesse_moyenne`
- `statut_arrivee` (DA, AR, NP, etc.)

**Partants / Non partants:**
- `non_partant`, `disqualifie`

**Musique, entraîneur, jockey, etc.:**
- `musique`, `entraineur`, `driver_jockey`, `proprietaire`
- `poids_kg`, `deferrage`, `equipement`
- `handicap_distance`, `rend_m`, `autostart_ligne`

**Conditions:**
- `meteo`, `etat_piste`, `temperature_c`, `vent_kmh`, `penetrometre`

### Q: Tu as les cotes : ouverture, avant départ, évolution ?

**Réponse:**
- `cote_matin` ✅
- `cote_finale` ✅ (avant départ)
- **Évolution temporelle:** ❌ Pas de courbe de cote historique

### Q: Tu as les volumes/rapport PMU ?

**Réponse:** Oui:
- `rapport_gagnant`, `rapport_place`
- `rapport_couple`, `rapport_trio`, `rapport_quarte`, `rapport_quinte`
- `montant_enjeux_total`
- `rapports_json` (détails complets)

### Q: Les données sont propres à PMU ou tu mixes ?

**Réponse:** Principalement **PMU** via l'API PMU officielle (`scraper_pmu_simple.py`). Des données IFCE (fichier équidés) sont aussi intégrées (`fichier-des-equides.csv` - 276MB).

### Q: Sous quel format ? Volume total ?

**Réponse:**
- **Format:** PostgreSQL (principal) + SQLite (backup)
- **Volume base:** ~350 MB (`data/database.db`)
- **ML Features:** ~330 MB CSV
- **Backtest predictions:** ~87 MB CSV

### Q: Qualité des données ?

**Réponse:** Des scripts de nettoyage existent:
- `nettoyer_doublons.py`, `fix_doublons.py`
- `clean_orphans.py`
- Normalisation des noms (`norm()` function)
- Identifiants stables: `id_cheval`, `id_cheval_pmu`

### Q: À quelle fréquence tu peux récupérer les nouvelles données ?

**Réponse:** **Daily** via cron jobs:
- `cron_scraping_quotidien.sh`
- `cron_enrichissement_quotidien.sh`

### Q: Tu as un pipeline existant (ETL) ?

**Réponse:** Oui, scripts automatisés:
- `scraper_pmu_simple.py` → Scraping
- `prepare_ml_features.py` → Feature engineering
- `calibration_pipeline.py` → Calibration
- Orchestration via `cli.py`

### Q: Tes données incluent-elles les conditions détaillées ?

**Réponse:** Oui: `meteo`, `etat_piste`, `temperature_c`, `vent_kmh`, `penetrometre`, `incident`.

---

## 4) Ton algo actuel : comment il marche

### Q: Ton algo actuel sort des bets basés sur : règles heuristiques, modèle ML, scoring maison ?

**Réponse:** **Modèle ML (XGBoost)** + calibration:

```
XGBoost → Softmax Température → Calibration Platt/Isotonic → Blend Marché → Value
```

Le modèle principal: `models/xgb_proba_v9.joblib`

### Q: Quelles features/signaux principaux ?

**Réponse:** Features dans `prepare_ml_features.py`:
- Historique cheval (nb_courses, nb_victoires, taux_victoire)
- Forme récente (30j, 60j, 90j)
- Stats jockey/entraîneur
- Distance, hippodrome, discipline
- Cote (implicite)
- ELO courant

### Q: Est-ce qu'il produit une probabilité par cheval ?

**Réponse:** Oui, `p_model_win` par cheval, normalisé par course via softmax:
```python
p_normalized = softmax(logits / temperature)
```

### Q: Sais-tu mesurer son historique : ROI, yield, drawdown, hit-rate, nb de bets ?

**Réponse:** Oui, métriques disponibles:
- **ROI:** ~30-67% (selon période)
- **ROC-AUC Test:** 0.6189
- **Brier Score:** ~0.081
- **Max Drawdown:** ~18%
- **Sharpe:** ~5.03
- **EV Lift vs Market:** +53%

### Q: Quel est son point faible actuel ?

**Réponse basée sur le code:**
- Pas de cotes en temps réel (évolution)
- Pas de modèle direct pour combinaisons (approximation)
- Généralisation sur nouvelles disciplines/hippodromes
- Calibration peut dériver (drift detection implémenté)

### Q: Tu veux que l'IA remplace, améliore, ou surcouche ton algo ?

**❓ À confirmer par l'utilisateur**

### Q: Tu veux garder une logique "explicable" ou boîte noire ?

**Réponse:** Le système actuel génère des explications:
- `signaux_positifs`, `signaux_negatifs` par cheval
- Feature importance XGBoost
- Scores par composante (historique, forme, jockey, etc.)

---

## 5) "Agent IA" : comportement attendu

### Q: Tu imagines un agent qui ingère données → calcule features → appelle modèle → applique règles → génère rapport → publie ?

**Réponse:** C'est **exactement le pipeline actuel**:
1. `scraper_pmu_simple.py` → Ingestion
2. `prepare_ml_features.py` → Features
3. `ModelManager.predict()` → Modèle
4. `betting_policy.py` → Règles de risque
5. API `/picks/today` → Rapport
6. Frontend React → Publication

### Q: Il doit pouvoir "raisonner" et expliquer pourquoi un bet est proposé ?

**Réponse:** Partiellement implémenté:
- `betting_advisor.py` génère `raison_principale` et `raisons_secondaires`
- Signaux positifs/négatifs listés
- **LLM non intégré actuellement** pour explications naturelles

### Q: Tu veux du conversationnel (chat) ?

**Réponse:** L'API a un endpoint `/chat`:
```python
class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
```
Mais l'intégration LLM n'est pas complète.

### Q: L'agent doit-il apprendre en continu (online learning) ?

**Réponse:** Non actuellement. Le système utilise:
- `detect_drift.py` pour détecter la dérive
- Retrain périodique (`crontab_retraining.txt`)
- Pas d'online learning

### Q: Tu veux une séparation module proba / value / tickets / staking ?

**Réponse:** **Déjà séparé**:
- `calibration_pipeline.py` → Probabilités
- `market_debiaser.py` → Value (blend marché)
- `exotic_ticket_generator.py` → Tickets
- `betting_policy.py` → Staking

### Q: Tu veux un mode "simulation" ?

**Réponse:** **Backtest disponible**:
- `strategy_backtester.py`
- API `/backtest/run`
- Frontend `/backtest`

---

## 6) Modélisation : probabilités, value, et construction de tickets

### Q: Tu veux prédire : probabilité de gagner, top 2/3, distribution de rangs ?

**Réponse:**
- **P(victoire):** ✅ `p_model_win`
- **P(place):** ✅ via `place_probability_estimator.py`
- **Distribution de rangs:** ✅ via Harville/Henery/Stern/LBS

### Q: Si tu joues des combinés (trio/quinté), tu veux un moteur qui approxime ou modèle direct ?

**Réponse:** **Approximation via Monte Carlo Plackett-Luce**:
```python
class PlackettLuceSimulator:
    # 20000 simulations pour estimer P(trio), P(quinté)
```

### Q: Comment tu définis la "value" ?

**Réponse:**
```python
value = p_model * odds - 1
# Ou: value = EV_pct = (p_model * odds - 1) * 100
```

### Q: Tu veux que l'agent sorte 1 ticket optimal ou plusieurs options ?

**Réponse:** Plusieurs options via profils:
- ULTRA_SUR, SUR, STANDARD, AMBITIEUX
- Safe / Balanced / Agressif

---

## 7) Backtest : comment on prouve que ça marche

### Q: Tu as déjà un backtest fiable ?

**Réponse:** Oui, `strategy_backtester.py` avec:
- Split temporel strict (anti-fuite)
- Leak detection
- Validation cotes pré-off

### Q: Comment tu gères le biais temporel ?

**Réponse:**
```python
strict_temporal_split: bool = True
leak_detection_enabled: bool = True
```
Split par date: train 2023-01-09, val 2023-10-11, test 2024+

### Q: Tu veux mesurer la robustesse par segments ?

**Réponse:** Possible mais pas systématiquement implémenté. Les données permettent de segmenter par:
- Discipline (plat/trot/obstacle)
- Hippodrome
- Taille peloton
- Favoris vs outsiders (via cote)

### Q: Tu as une notion de bankroll et limites de mises ?

**Réponse:** Oui, dans `BacktestConfig`:
```python
initial_bankroll: float = 1000.0
max_stake_pct: float = 0.05  # 5% max
daily_budget_pct: float = 0.12  # 12% jour
```

### Q: Tu veux optimiser sous contrainte (ex: drawdown max) ?

**Réponse:** Partiellement:
- `max_drawdown_stop: 0.20-0.30` dans les zones
- Analyse drawdown dans `BetSimulator._analyze_drawdown_series()`

---

## 8) Temps réel : cotes qui bougent, scratchs, dernières infos

### Q: L'agent doit-il se mettre à jour quand une cote bouge, un cheval est NP, etc. ?

**Réponse:** **Non implémenté actuellement**. Le système utilise les cotes finales statiques. Pas de flux temps réel.

### Q: À quel moment tu figes les propositions ?

**Réponse:** Les propositions sont générées à la demande via API. La cote utilisée est `cote_finale` (pré-départ).

### Q: Tu veux des alertes (Telegram/Discord/mail) ?

**Réponse:** **Non implémenté actuellement**. ❓ À confirmer si souhaité.

---

## 9) Produit : UX, site, sortie attendue

### Q: Ton site est fait avec quoi ?

**Réponse:**
- **Frontend:** React + Vite + Tailwind CSS
- **Backend:** FastAPI (Python)
- **Hébergement:** Docker

### Q: Comment tu affiches les bets aujourd'hui ?

**Réponse:** Pages frontend:
- `/conseil` (Recommandations)
- `/mes-paris` (Paris enregistrés)
- `/backtest` (Simulation historique)
- `/analytics` (Analyses)

### Q: Tu veux une fiche course avec ranking, proba, cote, value, explication ?

**Réponse:** **Partiellement implémenté**. L'API retourne ces données, le frontend les affiche.

### Q: Tu veux garder un historique public des recommandations + résultats ?

**Réponse:** Oui, tables:
- `api_predictions` (prédictions historiques)
- `feedback_results` (résultats réels)
- `bets_log` (log des paris)

### Q: Tu veux des filtres utilisateur (style de risque, budget, type de pari) ?

**Réponse:** **Implémenté**:
- Profil risque: PRUDENT/STANDARD/AGRESSIF
- Bankroll personnalisé
- Types de paris selon zone

---

## 10) Tech : infra, coûts, latence, sécurité

### Q: Tu héberges où ?

**Réponse:** Docker local actuellement. `docker-compose.yml` prêt pour déploiement cloud.

### Q: Quel langage pour le pipeline ?

**Réponse:** **Python** (backend, ML) + **JavaScript/React** (frontend)

### Q: Ta DB ?

**Réponse:** **PostgreSQL** (principal) + SQLite (backup)

### Q: Volume de requêtes/jour attendu ?

**Réponse:** ❓ Non spécifié. Usage personnel/petit groupe actuellement.

### Q: Latence acceptable pour recalculer une journée de courses ?

**Réponse:** L'API actuelle a une latence ~50-200ms par prédiction. Le backtest complet peut prendre plusieurs minutes.

### Q: Budget mensuel infra + APIs ?

**Réponse:** ❓ Non spécifié. L'API PMU est gratuite.

### Q: Tu veux que l'agent soit automatique ou avec validation humaine ?

**Réponse:** Actuellement **validation humaine** (le user décide de parier ou non).

### Q: Besoin de logs/monitoring, versioning des modèles ?

**Réponse:** **Implémenté**:
- Prometheus + Grafana dans docker-compose
- Versioning modèles: `model_version` dans les métriques
- Logs backend

---

## 11) LLM (OpenAI/Gemini) : rôle exact

### Q: Tu veux utiliser un LLM pour : explications, rapports, feature engineering, anomalies, orchestration ?

**Réponse:** Quelques fichiers existent:
- `ai_assistant.py`
- `ai_error_analyzer.py`
- `ai_model_optimizer.py`
- `ai_results_analyzer.py`

Mais l'intégration LLM complète n'est **pas finalisée**.

### Q: Tu veux du "tool calling" ?

**Réponse:** ❓ À confirmer. L'infrastructure API existe pour ça.

### Q: Quels risques tu veux éviter ?

**Réponse:** ❓ Hallucinations, surconfiance à éviter.

### Q: Tu veux des réponses strictement basées sur données (RAG) ?

**Réponse:** ❓ À confirmer.

---

## 12) Garde-fous : risque, conformité, "jeu responsable"

### Q: Tu veux imposer des limites ?

**Réponse:** **Oui, implémenté**:
- `max_bets_per_day: 2-6`
- `max_stake_eur: 2-40€`
- `max_drawdown_stop: 20-30%`
- `daily_budget_rate: 12%`

### Q: Tu veux un mode "simulation uniquement" ?

**Réponse:** **Oui** - Paper trading + backtest disponibles.

### Q: Si c'est pour des clients : disclaimers, CGU ?

**Réponse:** ❓ Non implémenté actuellement.

### Q: Tu veux bloquer certains comportements (ex: "double la mise pour te refaire") ?

**Réponse:** Le système Kelly empêche intrinsèquement le Martingale. Pas de blocage explicite.

---

## 13) Priorités : MVP

### Q: Ton MVP en 2–4 semaines, ce serait quoi ?

**État actuel du projet:**
- ✅ (A) Ranking + value sur Simple G/P → **Implémenté**
- ✅ (B) Génération tickets Trio → **Implémenté** (`exotic_ticket_generator.py`)
- ⚠️ (C) Quinté+ complet → **Partiellement** (MC simulation existe)
- ⚠️ (D) Module "explications" sur algo actuel → **Partiel** (signaux mais pas LLM)

### Q: Quelle est la 1ère amélioration qui te ferait dire "ça vaut le coup" ?

**❓ À confirmer par l'utilisateur**

---

## ✅ Réponses utilisateur aux questions

### 1. Tu veux que l'IA remplace, améliore, ou surcouche ton algo ?

**Réponse: SURCOUCHER avec raisonnement multi-étapes**

Architecture souhaitée:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ÉTAPE 1: ALGO EXISTANT                            │
│  Mon algorithme fait son analyse complète:                                  │
│  - Génère un RAPPORT COMPLET de toute son analyse                          │
│  - Paris gardés + POURQUOI                                                  │
│  - Paris non gardés + POURQUOI                                              │
│  - Mise proposée + POURQUOI (calculs, critères)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ÉTAPE 2: AGENT IA                                 │
│  L'agent IA reçoit le rapport et:                                          │
│  - Analyse le rapport en profondeur                                         │
│  - RAISONNE sur les choix proposés                                          │
│  - VÉRIFIE sur sources fiables (internet, sa propre BDD, l'historique)     │
│  - Propose SON ANALYSE dans un rapport complet                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ÉTAPE 3: AUTO-CRITIQUE                               │
│  L'agent se questionne sur son propre rapport:                              │
│  - Points forts de son analyse                                              │
│  - Points faibles / risques                                                 │
│  - Questions restantes                                                      │
│  - Niveau de confiance                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ÉTAPE 4: PROPOSITION FINALE                            │
│  Bets finaux proposés avec:                                                 │
│  - Justification complète                                                   │
│  - Score de confiance IA                                                    │
│  - Éléments vérifiés                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Tu veux des alertes (Telegram/Discord/mail) ?

**Réponse: NON**

Tout le raisonnement et toutes les étapes doivent être visibles sur une **interface web admin backend** qui permet de:
- Voir TOUT ce qui se passe en dessous
- Tracer chaque étape du raisonnement
- Comprendre les décisions à chaque niveau

### 3. Volume de requêtes/jour attendu ?

**Réponse:** Inconnu - à déterminer en fonction de l'usage.

### 4. Budget mensuel infra + APIs ?

**Réponse:** Pas de budget défini - recherche du **meilleur rapport qualité/prix**.

### 5. Quelle utilisation exacte du LLM ?

**Réponse:** Voir question 1 - Pipeline de raisonnement multi-étapes avec:
- Analyse du rapport algo
- Vérification sources externes
- Auto-critique
- Proposition finale argumentée

### 6. Besoin de disclaimers/CGU ?

**Réponse: NON** - Pas de suite pour l'instant.

### 7. Quelle est ta priorité MVP ?

**Réponse:** Le pipeline IA décrit en question 1:
- Rapport algo complet → Agent IA → Auto-critique → Bets finaux
- Interface admin pour tout visualiser

---

## Résumé Architecture Actuelle

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                        │
│  - Conseil, Mes Paris, Backtest, Analytics, Settings            │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND API (FastAPI)                        │
│  - /picks/today, /portfolio/today, /backtest/run                │
│  - Users, Settings, Bets                                        │
└─────────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   ML Engine     │  │   Betting       │  │   Exotic        │
│   (XGBoost)     │  │   Policy        │  │   Tickets       │
│   + Calibration │  │   (Kelly)       │  │   (Monte Carlo) │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PostgreSQL Database                         │
│  - chevaux, cheval_courses_seen, entraineurs, drivers            │
│  - users, user_bets, predictions, feedback                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Pipeline (Daily)                       │
│  - Scraper PMU → Features → Calibration → Predictions            │
└─────────────────────────────────────────────────────────────────┘
```
