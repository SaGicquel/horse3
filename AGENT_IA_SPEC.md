# AGENT IA - Spécification Technique

> Surcouche IA pour le système de paris hippiques horse3

---

## 📋 Invariants Projet (Ce qui ne doit JAMAIS changer)

### Invariant 1: Rétrocompatibilité API
> Les endpoints actuels restent identiques et continuent de renvoyer les mêmes résultats.

- ✅ `/picks/today` → Inchangé
- ✅ `/portfolio/today` → Inchangé
- ✅ `/backtest/run` → Inchangé
- ✅ Tous les endpoints existants dans `web/backend/main.py`

**Test de non-régression:** Avant/après déploiement, les réponses API doivent être identiques (hash JSON).

---

### Invariant 2: Pipeline ML = Source de Vérité
> Le pipeline "scraper → features → modèle → betting_policy → API" reste la source de vérité.

```
scraper_pmu_simple.py → prepare_ml_features.py → XGBoost → calibration_pipeline.py → betting_policy.py
```

L'agent IA **consomme** les outputs de ce pipeline, il ne les **remplace pas**.

---

### Invariant 3: IA = Critique, pas Calcul
> La surcouche IA ne modifie pas les probabilités ni les cotes.

| Interdit ❌ | Autorisé ✅ |
|------------|------------|
| Recalculer `p_model_win` | Critiquer si `p_model_win` semble incohérent |
| Modifier `value_win` | Signaler si `value_win < 0` mais bet gardé |
| Changer `cote_finale` | Vérifier que la cote en DB = cote PMU réelle |
| Inventer des données | Proposer retrait/ajout avec justification explicite |

---

### Invariant 4: Traçabilité Totale
> Toute décision finale doit être traçable étape par étape dans l'admin.

Chaque exécution de l'agent génère:
- `run_id` unique
- Logs de chaque étape (A, B, C, D)
- Inputs/outputs JSON stockés
- Timestamps et durées
- Evidence pour chaque claim

---

### Invariant 5: Anti-Hallucination
> L'IA doit éviter hallucinations et surconfiance.

Règles strictes:
1. **Pas de fait sans source** → Tout claim = preuve attachée (DB, API, Web)
2. **Incertitude explicite** → "Non vérifié" si pas de preuve
3. **Outputs validés** → JSON validé par Pydantic, sinon rejet + retry
4. **Pas de martingale** → Le système Kelly reste la règle de mise

---

## 🔄 Pipeline Agent IA - 4 Étapes

### Étape A: Génération Rapport Algo (Sans LLM)

**Input:** Date, profil utilisateur, bankroll

**Process:**
1. Appeler le pipeline existant (`betting_policy.select_portfolio_from_picks`)
2. Pour chaque pick: extraire décision + raisons + règles appliquées
3. Structurer en JSON selon le schéma défini

**Output:** `AlgoReportJSON` (voir schéma ci-dessous)

**Definition of Done:**
- [ ] JSON valide selon schéma Pydantic
- [ ] Tous les champs obligatoires remplis
- [ ] Chaque decision a `status`, `why[]`, `failed_rules[]`
- [ ] Stocké en DB table `agent_runs`
- [ ] Aucun appel LLM

---

### Étape B: Analyse IA (LLM + Outils Internes)

**Input:** `AlgoReportJSON` de l'étape A

**Process:**
1. LLM reçoit le rapport JSON complet
2. LLM peut appeler des outils définis:
   - `get_runner_history(runner_id, n=10)` → Historique cheval
   - `get_race_context(race_id)` → Conditions course
   - `recompute_constraints(report)` → Vérifier règles
   - `explain_features(runner_id)` → Feature importance
3. LLM analyse cohérence, anomalies, propositions

**Output:** `AnalysisReportJSON`
```json
{
  "anomalies": [{"pick_id": "...", "issue": "...", "severity": "HIGH|MEDIUM|LOW"}],
  "modifications": [{"pick_id": "...", "action": "REMOVE|REDUCE_STAKE|ADD", "reason": "..."}],
  "questions": ["..."]
}
```

**Definition of Done:**
- [ ] LLM a uniquement accès aux outils whitelist
- [ ] Output validé par Pydantic
- [ ] Chaque anomalie a severity + justification
- [ ] Chaque modification a raison + evidence
- [ ] Stocké en `agent_steps` (run_id, step="B")
- [ ] Durée et tokens loggés

---

### Étape C: Vérification (LLM + Preuves)

**Input:** `AnalysisReportJSON` de l'étape B

**Process:**
1. Pour chaque claim de l'étape B → chercher preuve
2. Sources autorisées:
   - **Interne:** DB horse3 (historique, courses, partants)
   - **Externe (optionnel):** API PMU, France Galop, LeTrot (lecture seule)
3. Marquer chaque claim comme `VERIFIED` ou `UNVERIFIED`

**Output:** `VerificationReportJSON`
```json
{
  "verified_claims": [{"claim": "...", "source": "DB|API|WEB", "evidence": {...}}],
  "unverified_claims": [{"claim": "...", "potential_impact": "HIGH|MEDIUM|LOW"}],
  "verification_rate": 0.85
}
```

**Definition of Done:**
- [ ] Chaque claim a un statut vérifié/non vérifié
- [ ] Chaque vérification a une source traçable
- [ ] Pas de claim sans preuve marqué comme "vérifié"
- [ ] Evidence stockée dans `agent_evidence`
- [ ] `verification_rate` calculé

---

### Étape D: Auto-Critique + Proposition Finale (LLM)

**Input:** Rapports des étapes A, B, C

**Process:**
1. LLM analyse sa propre analyse (méta-réflexion)
2. Identifie forces / faiblesses / risques
3. Produit recommandation finale avec score de confiance
4. Compare avec picks algo original (diff)

**Output:** `FinalReportJSON`
```json
{
  "self_critique": {
    "strengths": ["..."],
    "weaknesses": ["..."],
    "remaining_questions": ["..."],
    "risk_notes": ["..."]
  },
  "final_bets": [
    {
      "race_id": "R3C5",
      "runner_id": "...",
      "bet_type": "PLACE",
      "stake_eur": 10,
      "expected_value": 0.15,
      "justification": "...",
      "verified_elements": ["..."]
    }
  ],
  "confidence_score": 72,
  "diff_vs_algo": {
    "kept": [...],
    "removed": [...],
    "modified": [...]
  }
}
```

**Definition of Done:**
- [ ] Self-critique présente avec les 4 catégories
- [ ] Chaque bet final a justification complète
- [ ] `confidence_score` calculé (pas "ressenti" LLM)
- [ ] Diff explicite vs picks algo original
- [ ] Stocké en `agent_steps` + `agent_diffs`

---

## 📊 Score de Confiance (Calculé, pas LLM)

Le score est déterministe, basé sur:

| Facteur | Poids | Mesure |
|---------|-------|--------|
| Qualité modèle | 20% | drift_status: OK=100, WARN=50, ALERT=0 |
| Marge value | 25% | (value - cutoff) / cutoff × 100, cap 100 |
| Consensus modèle/marché | 20% | 1 - \|p_model - p_implied\| × 5 |
| Risque | 20% | 100 - (odds/max_odds × 50 + field_size/20 × 50) |
| Taux vérification | 15% | verification_rate × 100 |

**Formule finale:**
```python
confidence = (drift * 0.20 + value_margin * 0.25 + consensus * 0.20
              + risk_score * 0.20 + verif_rate * 0.15)
```

---

## 🗄️ Schéma Base de Données

### Table `agent_runs`
```sql
CREATE TABLE agent_runs (
    run_id UUID PRIMARY KEY,
    date_run DATE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    profile VARCHAR(20),  -- PRUDENT|STANDARD|AGRESSIF
    bankroll DECIMAL(10,2),
    status VARCHAR(20),   -- PENDING|RUNNING|SUCCESS|FAILED
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    algo_report JSONB,    -- Étape A
    final_report JSONB,   -- Étape D
    confidence_score INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Table `agent_steps`
```sql
CREATE TABLE agent_steps (
    step_id UUID PRIMARY KEY,
    run_id UUID REFERENCES agent_runs(run_id),
    step_name VARCHAR(1),  -- A|B|C|D
    input_json JSONB,
    output_json JSONB,
    llm_model VARCHAR(50),
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd DECIMAL(10,6),
    duration_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Table `agent_evidence`
```sql
CREATE TABLE agent_evidence (
    evidence_id UUID PRIMARY KEY,
    run_id UUID REFERENCES agent_runs(run_id),
    claim_id VARCHAR(100),
    claim_text TEXT,
    source_type VARCHAR(10),  -- DB|API|WEB
    source_url TEXT,
    payload JSONB,
    verified BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Table `agent_diffs`
```sql
CREATE TABLE agent_diffs (
    diff_id UUID PRIMARY KEY,
    run_id UUID REFERENCES agent_runs(run_id),
    pick_id VARCHAR(100),
    action VARCHAR(20),  -- KEPT|REMOVED|MODIFIED|ADDED
    algo_decision JSONB,
    agent_decision JSONB,
    reason TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🔌 Endpoints API (Nouveaux)

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/agent/run` | Lance pipeline complet |
| GET | `/agent/runs` | Liste tous les runs |
| GET | `/agent/runs/{run_id}` | Détails d'un run |
| GET | `/agent/runs/{run_id}/steps` | Toutes les étapes |
| GET | `/agent/runs/{run_id}/diff` | Diff algo vs agent |
| GET | `/agent/runs/{run_id}/evidence` | Preuves collectées |
| POST | `/agent/runs/{run_id}/replay` | Rejouer un run |

---

## 🖥️ Interface Admin

### Vue 1: Liste des Runs
- Tableau: date, status, #bets algo, #bets final, confiance, durée
- Filtres: date, status, profil
- Actions: voir détails, rejouer

### Vue 2: Détail Run (Timeline)
- Accordéon par étape (A → B → C → D)
- Pour chaque étape: input, output, durée, tokens
- Onglet "Evidence": claims vérifiés + sources
- Onglet "Logs": trace complète

### Vue 3: Diff Picks
- Colonne gauche: picks algo
- Colonne droite: picks agent
- Highlight: ajoutés (vert), retirés (rouge), modifiés (orange)
- Pour chaque diff: raison + evidence

---

## ✅ Checklist MVP

### Phase 1: Infrastructure
- [ ] Créer tables DB (`agent_runs`, `agent_steps`, `agent_evidence`, `agent_diffs`)
- [ ] Définir modèles Pydantic pour tous les JSON
- [ ] Créer endpoints API basiques

### Phase 2: Étape A (Rapport Algo)
- [ ] Modifier `betting_policy.py` pour exporter décisions structurées
- [ ] Générer JSON complet avec `why[]` et `failed_rules[]`
- [ ] Stocker en DB

### Phase 3: Étape B + D (Analyse + Final)
- [ ] Intégrer LLM (OpenAI/Gemini)
- [ ] Définir tools/functions disponibles
- [ ] Implémenter analyse + auto-critique
- [ ] Calculer score de confiance

### Phase 4: Admin UI
- [ ] Vue liste runs
- [ ] Vue détail timeline
- [ ] Vue diff picks

### Phase 5 (Optionnel): Étape C (Vérification externe)
- [ ] Connecter sources externes
- [ ] Implémenter collecte evidence
- [ ] Marquer claims vérifiés/non vérifiés

---

## 📝 Notes d'Implémentation

### Choix LLM recommandé (qualité/prix)
- **GPT-4o-mini** pour analyse (bon rapport qualité/prix)
- **GPT-4o** ou **Claude 3.5 Sonnet** pour auto-critique (meilleur raisonnement)
- Alternative économique: **Gemini 1.5 Flash**

### Estimation coûts (ordre de grandeur)
- ~5-10 courses/jour × 4 étapes × ~2-5K tokens = ~40-200K tokens/jour
- GPT-4o-mini: ~$0.15-0.60 / mille tokens → ~$6-30/mois
- GPT-4o: ~$2.50-10 / mille tokens → ~$100-500/mois

### Latence estimée
- Étape A: <1s (pas de LLM)
- Étape B: 5-15s (LLM + tools)
- Étape C: 2-10s (selon sources externes)
- Étape D: 5-10s (LLM)
- **Total: 15-40s par run**
