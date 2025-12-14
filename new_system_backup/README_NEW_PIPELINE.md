# Nouveau Pipeline Data PMU + IFCE (Rebuild Total)

## Objectif
Reconstruction complète: ingestion brute PMU (programme, participants, performances, paris) + import IFCE CSV. Normalisation forte et matching chevaux IFCE ↔ PMU.

## Étapes d’exécution (ordre)
1. psql -f sql/drop_all.sql
2. psql -f sql/bootstrap.sql
3. psql -f sql/schema_v1.sql
4. python etl/load_ifce_csv.py fichier-des-equides.csv
5. python collect/pmu_program.py YYYY-MM-DD
6. python collect/pmu_participants.py YYYY-MM-DD
7. python collect/pmu_performances.py YYYY-MM-DD
8. python collect/pmu_pools.py YYYY-MM-DD
9. python transform/pmu_to_core_fact.py YYYY-MM-DD
10. python match/match_horses.py

## Principes
| Couche | Schéma | Description |
|--------|--------|-------------|
| Staging | stg | JSON brut idempotent |
| Core | core | Référentiels hippodromes, chevaux, personnel |
| Fact | fact | Meetings, races, participants, pools, payouts, snapshots marché |
| Aux | aux | Matching, logs techniques |

## Normalisation
- Nom cheval/personnel: majuscule, accents retirés, caractères non alphanum → espace, collapse espaces.
- Pays: 3 chars max, uppercase.
- Matching: trigram pg_trgm (≥0.90 confirmé, 0.85–0.90 ambigu).

## Fichiers clés
- sql/drop_all.sql : reset complet
- sql/bootstrap.sql : extensions + schémas
- sql/schema_v1.sql : DDL complet V1
- etl/load_ifce_csv.py : import CSV IFCE filtrage âge
- collect/* : collecte PMU vers staging
- transform/pmu_to_core_fact.py : mapping staging → core/fact
- match/match_horses.py : liaison IFCE ↔ PMU

## Améliorations futures
- Parse fin équipement (œillères, déferrage) → colonnes dédiées
- Sectionnels / chronos avancés (web scraping) → fact.sectionnels
- Market live snapshots job scheduler
- Qualité: vues d’audit (doublons, incidents, ordres invalides)

## Sécurité & Résilience
- Idempotence: clés (source, endpoint, key) en staging
- Conflits: ON CONFLICT pour upserts
- Re-run journée: re-collecte écrase payload JSON sans doublons

## Tests (à créer)
- test_normalize.py : vérifier normalisation
- test_match.py : faux jeux de noms proches
- test_transform.py : payload minimal → ligne fact.race

## TODO Prioritaires
1. Ajouter parsing équipements
2. Intégrer performances ‘red_km’, temps_officiel
3. Créer vues analytiques (winrate par hippodrome / draw bias)
4. Intégrer incidents en table atomique distincte

---
✅ Pipeline initial prêt. Voir scripts dans collect/, etl/, transform/, match/
