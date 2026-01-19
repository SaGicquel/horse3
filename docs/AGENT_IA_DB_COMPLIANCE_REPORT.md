# 📊 Rapport de Conformité DB - Agent IA

**Date:** 2024-12-21
**Base de données:** `pmu_database` @ Docker `pmuBDD`
**Migrations appliquées:** `agent_ia_migration.sql` + `agent_ia_migration_v2.sql`

---

## ✅ Checklist de Conformité

| # | Exigence | Statut | Détails |
|---|----------|--------|---------|
| 1 | CHECK constraint `step_name` | ✅ OK | `('A','B','C','D')` |
| 2 | CHECK constraint `status` | ✅ OK | `('PENDING','RUNNING','SUCCESS','FAILED','SKIPPED')` |
| 3 | Unicité `(run_id, step_name, attempt)` | ✅ OK | Index UNIQUE créé |
| 4 | FK CASCADE sur toutes les tables | ✅ OK | Toutes les FK ont `ON DELETE CASCADE` |
| 5 | Colonnes JSONB | ✅ OK | 8 colonnes JSONB |
| 6 | Index GIN pour recherche admin | ✅ OK | 5 index GIN créés |

---

## 1️⃣ CHECK Constraints

```sql
-- 5 contraintes CHECK actives
agent_runs_status_chk     → ('PENDING','RUNNING','STEP_A'...'SUCCESS','FAILED','CANCELLED')
agent_runs_profile_chk    → ('PRUDENT','STANDARD','AGRESSIF','SUR','ULTRA_SUR','AMBITIEUX')
agent_steps_step_name_chk → ('A','B','C','D')
agent_steps_status_chk    → ('PENDING','RUNNING','SUCCESS','FAILED','SKIPPED')
agent_diffs_action_chk    → ('KEPT','REMOVED','MODIFIED','ADDED')
```

---

## 2️⃣ Unicité avec Retries

```sql
-- Colonne attempt ajoutée
agent_steps.attempt INTEGER NOT NULL DEFAULT 1

-- Index UNIQUE pour éviter les doublons
CREATE UNIQUE INDEX agent_steps_run_step_attempt_uq
    ON agent_steps(run_id, step_name, attempt);
```

**Comportement:**
- Première exécution: `attempt = 1`
- Retry: `attempt = 2, 3, ...`
- Garantit qu'on ne peut pas avoir 2 steps identiques pour le même run/attempt

---

## 3️⃣ Foreign Keys avec CASCADE

| FK Constraint | Table | Référence | ON DELETE |
|--------------|-------|-----------|-----------|
| `agent_steps_run_id_fkey` | agent_steps | agent_runs(run_id) | **CASCADE** |
| `agent_evidence_run_id_fkey` | agent_evidence | agent_runs(run_id) | **CASCADE** |
| `agent_evidence_step_id_fkey` | agent_evidence | agent_steps(step_id) | **CASCADE** |
| `agent_diffs_run_id_fkey` | agent_diffs | agent_runs(run_id) | **CASCADE** |
| `agent_runs_user_id_fkey` | agent_runs | users(id) | SET NULL |

**Nettoyage automatique:** Supprimer un `agent_run` supprime automatiquement tous les `steps`, `evidence`, et `diffs` associés.

---

## 4️⃣ Colonnes JSONB

| Table | Colonne | Usage |
|-------|---------|-------|
| agent_runs | `algo_report` | Rapport Algo complet (Étape A) |
| agent_runs | `final_report` | Rapport Final (Étape D) |
| agent_runs | `replay_inputs` | Inputs pour rejouer le run |
| agent_steps | `input_json` | Entrées de l'étape |
| agent_steps | `output_json` | Sorties de l'étape |
| agent_evidence | `payload` | Données brutes de preuve |
| agent_diffs | `algo_decision` | Décision algo originale |
| agent_diffs | `agent_decision` | Décision modifiée par agent |

---

## 5️⃣ Index GIN pour Recherche Admin

```sql
-- 5 index GIN créés (jsonb_path_ops pour performance)
idx_agent_runs_algo_report_gin    ON agent_runs(algo_report)
idx_agent_runs_final_report_gin   ON agent_runs(final_report)
idx_agent_steps_input_gin         ON agent_steps(input_json)
idx_agent_steps_output_gin        ON agent_steps(output_json)
idx_agent_evidence_payload_gin    ON agent_evidence(payload)
```

**Exemples de requêtes optimisées:**
```sql
-- Chercher les runs avec un cheval spécifique
SELECT * FROM agent_runs
WHERE algo_report @> '{"races": [{"kept_runners": ["horse_123"]}]}';

-- Chercher les steps avec erreur dans l'output
SELECT * FROM agent_steps
WHERE output_json @> '{"error": true}';
```

---

## 📋 Résumé des Tables

| Table | PK | FK | CHECK | UNIQUE | GIN | Colonnes |
|-------|----|----|-------|--------|-----|----------|
| agent_runs | ✅ | 1 | 2 | 1 | 2 | 22 |
| agent_steps | ✅ | 1 | 2 | 1 | 2 | 17 |
| agent_evidence | ✅ | 2 | 0 | 1 | 1 | 14 |
| agent_diffs | ✅ | 1 | 1 | 1 | 0 | 12 |

---

## 🚀 Prochaines Étapes

La base de données est maintenant **production-ready** pour le pipeline Agent IA:

1. ✅ Intégrité référentielle garantie (FK CASCADE)
2. ✅ Données valides uniquement (CHECK constraints)
3. ✅ Pas de doublons (UNIQUE avec attempt)
4. ✅ Recherche performante dans l'admin (GIN indexes)
5. ✅ Replay possible (replay_inputs JSONB)

**→ Prêt pour Phase 3: Intégration LLM**
