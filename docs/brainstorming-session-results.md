**Session Date:** 2026-01-23
**Facilitator:** Business Analyst Mary
**Participant:** User

## Executive Summary

**Topic:** Optimisation d'un algorithme de paris hippiques (Horse Racing) et gestion de bankroll stricte (100€) avec une BDD de 5 ans.

**Session Goals:**
- Identifier des pistes d'amélioration pour l'algo existant (SCAMPER).
- Détecter les failles critiques et risques de ruine (Inversion).
- Définir une stratégie de bankroll équilibrée et réaliste (Six Chapeaux).

**Techniques Used:**
1.  **SCAMPER:** Pour l'innovation incrémentale et radicale.
2.  **Inversion:** Pour le stress-test et la gestion des risques.
3.  **Six Chapeaux de Bono:** Pour une vision holistique et stratégique.

**Key Themes Identified:**
- **Gestion du Risque (Priorité #1):** Le système actuel (mise fixe 10€) est mathématiquement insoutenable pour 100€ de capital (risque de ruine ~50%).
- **Qualité des Données & Overfitting:** Le ROI backtesté de +71% est suspect et probablement overfitté. Nécessité absolue de validation Walk-Forward.
- **Obsolescence des Données:** La BDD de 5 ans contient des biais critiques (jockeys retraités, rénovations d'hippodromes) qui faussent les prédictions futures.
- **Approche Progressive:** Passer d'une stratégie "All-in" à une stratégie par étapes (Validation -> Scale -> Monetize).

---

## Technique Sessions

### SCAMPER - Innovation Algorithmique
**Description:** Technique de créativité par modification de l'existant (Substituer, Combiner, Adapter...).

**Ideas Generated:**
1.  **Substituer:** Remplacer la mise fixe par une mise dynamique (Kelly Criterion).
2.  **Substituer:** Remplacer le seuil de probabilité fixe par un seuil dynamique dépendant de la cote (ex: 55% pour cote < 8).
3.  **Combiner:** Enrichir le modèle ML avec le sentiment du marché (cotes bookmakers) et la météo.
4.  **Combiner:** Créer un "Ensemble Model" (XGBoost + Random Forest + Logistic Regression) pour plus de stabilité.
5.  **Adapter:** Adapter l'algo à une bankroll de 100€ via le Kelly Fraction (0.25) et un Stop-Loss journalier.

**Insights Discovered:**
- La mise fixe est l'ennemi juré d'une petite bankroll.
- L'algo doit s'adapter au contexte (hippodrome, jour) plutôt que d'appliquer des règles rigides.

### Inversion - Stress Test & Gestion des Risques
**Description:** Chercher comment échouer pour identifier les protections nécessaires.

**Ideas Generated:**
1.  **Comment se ruiner en 3 jours ?** Utiliser une Martingale ou chasser les pertes (Chase Losses). -> **Solution:** Stop-Loss journalier (-10%) et Kelly Betting.
2.  **Comment avoir -50% de ROI ?** Overfitter le modèle sur le passé et ignorer les changements de régime (rénovations de pistes). -> **Solution:** Walk-Forward Validation et Audit BDD.
3.  **Biais Cognitifs:** Overconfidence après une série de gains, Sunk Cost Fallacy après des pertes. -> **Solution:** Tracking rigoureux de TOUS les paris.
4.  **Données Trompeuses:** Jockeys retraités toujours dans la BDD, changements de règles ou de surfaces. -> **Solution:** Filtrage strict des données obsolètes.

**Insights Discovered:**
- Le backtesting (+71% ROI) ne garantit RIEN en live si les conditions ont changé.
- Les biais psychologiques sont aussi dangereux que les erreurs de code.

### Six Chapeaux - Stratégie Bankroll
**Description:** Analyse multi-angles (Faits, Émotions, Risques, Optimisme, Créativité, Processus).

**Ideas Generated:**
1.  **Blanc (Faits):** Win rate 56.8% validé, mais ROI live inconnu. Sample size de 222 paris insuffisant pour long terme.
2.  **Rouge (Émotions):** Peur justifiée de l'overfitting. Confiance dans l'expertise technique. Besoin de rassurer l'émotionnel par du "Paper Trading".
3.  **Noir (Risques):** Risque de ruine ~50% avec mise 10€. Risque d'overfitting brutal. -> Réduire mise à 2-3€ (Kelly 0.25).
4.  **Jaune (Opportunités):** Potentiel de x5 (100€ -> 500€) en 6 mois si exécution parfaite. Possibilité de monétiser l'API plus tard.
5.  **Vert (Alternatives):** Spécialisation VINCENNES-only pour réduire la variance. Approche Hybride (Algo + Tipster).

**Insights Discovered:**
- L'objectif n'est pas de faire fortune en 1 mois, mais de valider le système pour scaler ensuite.
- La stratégie "Vincennes-Only" est une excellente alternative pour démarrer avec des données plus propres.

---

## Idea Categorization

### Immediate Opportunities (Ready to implement)
1.  **Intégration Kelly Criterion (Fraction 0.25)**
    - *Description:* Remplacer la mise uniforme de 10€ par une mise calculée dynamiquement (2-3€ pour 100€ de bankroll).
    - *Why immediate:* Vital pour éviter la ruine mathématique immédiate.
    - *Resources needed:* Code Python (formule fournie).

2.  **Stop-Loss Journalier (10%)**
    - *Description:* L'API refuse de donner des conseils si la perte du jour atteint 10% de la bankroll initiale.
    - *Why immediate:* Protection critique contre le "Tilt" et les mauvaises séries.
    - *Resources needed:* Ajout logique dans l'API `/daily-advice`.

3.  **Walk-Forward Validation**
    - *Description:* Refaire le backtest en mode "simulé réel" (entraîne sur T, teste sur T+1) pour vérifier le ROI de 71%.
    - *Why immediate:* Pour savoir si l'algo est overfitté avant de risquer de l'argent.
    - *Resources needed:* Script Python de backtest modifié.

### Future Innovations (Requires development)
1.  **Enrichissement Données (Météo & Bookies)**
    - *Description:* Ajouter les données météo et les cotes/mouvements des bookmakers aux features du modèle.
    - *Development:* Besoin d'API externes ou de scraping + ré-entraînement modèle.
    - *Timeline:* 1-2 mois.

2.  **Modèle Ensemble (Voting Classifier)**
    - *Description:* Combiner XGBoost avec Random Forest et Regression Logistique.
    - *Development:* Entraînement et tuning de nouveaux modèles.
    - *Timeline:* 1 mois.

3.  **Spécialisation par Hippodrome**
    - *Description:* Créer des modèles spécifiques pour chaque hippodrome majeur (Vincennes, Auteuil, etc.).
    - *Development:* Segmentation BDD et gestion de plusieurs modèles en prod.
    - *Timeline:* 2-3 mois.

### Insights & Learnings
- **Le Backtest n'est pas la réalité:** Un ROI de 71% est suspect. Il faut le challenger avec des données "Out-of-Sample".
- **La Bankroll dicte la mise:** On ne peut pas parier 10% de son capital par course (10€ sur 100€). C'est du suicide financier.
- **L'Inversion sauve le compte:** Penser à "comment tout perdre" a permis de mettre en place les garde-fous (Stop-loss, filtres données obsolètes).

---

## Action Planning

### Top 3 Priority Ideas

#### #1 Priority: Implémenter la Gestion de Bankroll (Kelly + Stop-Loss)
- **Rationale:** Sans ça, le risque de ruine est de ~50% en quelques semaines. C'est la base de la survie.
- **Next steps:** Modifier l'API pour accepter `current_bankroll` et retourner une mise optimisée (Kelly 0.25). Ajouter la logique de Stop-Loss.
- **Resources needed:** Code Python existant.
- **Timeline:** Immédiat (Aujourd'hui/Demain).

#### #2 Priority: Audit de Données & Walk-Forward Validation
- **Rationale:** Vérifier la réalité du ROI de 71% et nettoyer la BDD des biais (jockeys retraités, etc.).
- **Next steps:** Écrire le script de validation Walk-Forward. Identifier les dates de rupture (rénovations pistes) et filtrer les données.
- **Resources needed:** Temps de calcul, analyse BDD.
- **Timeline:** Semaine 1.

#### #3 Priority: Lancement Phase 1 "Validation" (Vincennes-Only / Paper Trading)
- **Rationale:** Tester le système en conditions réelles mais à risque réduit (ou nul).
- **Next steps:** Configurer l'algo pour ne parier que sur Vincennes (données plus propres). Commencer le tracking live (Excel/BDD) de chaque pari.
- **Resources needed:** Spreadsheet de tracking, discipline.
- **Timeline:** Semaine 2 à Mois 3.

---

## Reflection & Follow-up

**What Worked Well:**
- L'utilisation combinée de SCAMPER (créativité), Inversion (sécurité) et Six Chapeaux (stratégie) a donné une feuille de route très complète.
- L'apport de références académiques et de simulations chiffrées a crédibilisé l'analyse.

**Areas for Further Exploration:**
- **Sentiment Analysis:** Explorer comment scraper et utiliser les avis des réseaux sociaux/forums spécialisés.
- **API Monetization:** Garder en tête l'architecture pour une future commercialisation (multi-users, gestion de droits).

**Recommended Follow-up Techniques:**
- **Pre-mortem:** Avant de lancer la "Phase 2 (Scale)", refaire une session pour imaginer pourquoi cela pourrait échouer.

---

*Session facilitated using the BMAD-METHOD™ brainstorming framework*
