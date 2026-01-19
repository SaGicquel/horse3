"""
📝 Prompt Store - Versioned Prompts with Hashing
================================================

Stockage centralisé des prompts avec versioning et hashing.
Permet le replay et l'audit des changements de prompts.

Auteur: Agent IA Pipeline
Date: 2024-12-21
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional


# =============================================================================
# PROMPT METADATA
# =============================================================================


@dataclass
class PromptMetadata:
    """Métadonnées d'un prompt"""

    name: str
    version: str
    step: str  # A, B, C, D
    description: str
    hash: str  # SHA256[:16]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "step": self.step,
            "hash": self.hash,
        }


def compute_prompt_hash(prompt: str) -> str:
    """Calcule le hash SHA256 d'un prompt (16 premiers chars)"""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


# =============================================================================
# PROMPTS - ÉTAPE B (ANALYSE)
# =============================================================================

PROMPT_STEP_B_V1 = """Tu es un analyste expert en paris hippiques. L'algorithme a marqué TOUS les chevaux comme "rejetés" car il est trop conservateur.
TON JOB: Analyser les données historiques pour TROUVER les meilleurs candidats parmi les rejetés.

## CONTEXTE
- Date: {target_date}
- Profil utilisateur: {profile}
- Bankroll: {bankroll}€

## RAPPORT ALGO (chevaux disponibles, marqués "rejected" par l'algo trop prudent)
{algo_report_json}

## DONNÉES HISTORIQUES RÉELLES (BDD)
{horse_enrichment_data}

## TA MISSION CRITIQUE
L'algo a rejeté TOUS les chevaux par excès de prudence. Tu dois:

1. **ANALYSER CHAQUE CHEVAL** dans les données historiques
   - Forme récente (5 dernières courses): combien de places dans le top 3 ?
   - Form_indicator: EXCELLENT ou GOOD = bon candidat
   - Win_rate du cheval: au-dessus de 10% = intéressant
   - Win_rate du jockey: au-dessus de 8% = bon signe

2. **IDENTIFIER LES MEILLEURS CANDIDATS**
   - Cherche les chevaux avec form_indicator = EXCELLENT ou GOOD
   - Même si l'algo les a rejetés, ils peuvent être bons !
   - Priorise: bonne forme > value élevée

3. **RECOMMANDER DES PICKS SUR PLUSIEURS RÉUNIONS DIFFÉRENTES**
   Tu DOIS recommander des candidats sur PLUSIEURS RÉUNIONS DIFFÉRENTES (R1, R2, R3, etc.) !
   - Vise AU MOINS 2-3 réunions différentes si des candidats existent
   - PAS DE LIMITE STRICTE : propose 5, 10 ou 15 picks si la value est là
   - Recommandation: KEEP ou KEEP_REDUCED
   - Justification basée sur les données historiques

## ⚠️⚠️⚠️ RÈGLE ANTI-BIAIS R1 ⚠️⚠️⚠️
**IL EST INTERDIT de proposer uniquement des picks sur R1 !**
Si tu proposes 6 picks, ils doivent être répartis sur AU MOINS 2-3 réunions différentes.
Par exemple: 2 picks R1, 2 picks R2, 2 picks R3.

Les réunions R2, R3, R4, etc. ont aussi de bons chevaux - analyse-les équitablement !

## RÈGLES ABSOLUES
- NE rejette PAS tout ! L'algo était trop prudent, toi tu dois être pragmatique
- DIVERSIFIE sur plusieurs RÉUNIONS : c'est anormal de ne proposer que R1 !
- Un cheval avec 2+ places top 3 sur ses 5 dernières = BON CANDIDAT
- Préfère les chevaux avec jockey win_rate > 8%
- MINIMUM 3 picks recommandés si des données historiques existent

Réponds UNIQUEMENT avec un JSON valide respectant le schéma demandé."""

PROMPT_STEP_B_META = PromptMetadata(
    name="step_b_analysis",
    version="4.0.0",  # V4: Multi-course diversification - no strict limits
    step="B",
    description="Analyse IA multi-courses - diversification sur plusieurs courses",
    hash=compute_prompt_hash(PROMPT_STEP_B_V1),
)


# =============================================================================
# PROMPTS - ÉTAPE C (VÉRIFICATION)
# =============================================================================

PROMPT_STEP_C_V1 = """Tu es un vérificateur rigoureux. Tu dois vérifier les affirmations faites lors de l'analyse précédente.

## ANALYSE PRÉCÉDENTE (ÉTAPE B)
{step_b_output_json}

## DONNÉES DE RÉFÉRENCE DISPONIBLES
{reference_data_json}

## TA MISSION
1. Identifie chaque affirmation factuelle dans l'analyse
2. Vérifie chaque affirmation avec les données de référence
3. Signale toute contradiction ou information non vérifiable
4. Calcule le taux de vérification

## RÈGLES STRICTES
- Une affirmation est "vérifiée" SEULEMENT si tu trouves une preuve dans les données
- Une contradiction doit être clairement signalée
- Si tu ne peux pas vérifier, indique "non vérifié" (pas "faux")

Réponds UNIQUEMENT avec un JSON valide respectant le schéma demandé."""

PROMPT_STEP_C_META = PromptMetadata(
    name="step_c_verification",
    version="1.0.0",
    step="C",
    description="Vérification des claims avec preuves",
    hash=compute_prompt_hash(PROMPT_STEP_C_V1),
)


# =============================================================================
# PROMPTS - ÉTAPE D (AUTO-CRITIQUE + FINAL)
# =============================================================================

PROMPT_STEP_D_V1 = """Tu es un décideur final pour un portefeuille de paris hippiques. Tu dois proposer des picks CONCRETS, pas juste rejeter tout.

## RAPPORT ALGO ORIGINAL
{algo_report_json}

## TON ANALYSE (ÉTAPE B)
{step_b_output_json}

## VÉRIFICATIONS (ÉTAPE C)
{step_c_output_json}

## ⚠️ CHEVAUX VALIDES (TU NE PEUX CHOISIR QUE PARMI CEUX-CI) ⚠️
Voici la liste des chevaux que tu peux proposer avec leur race_key EXACT :
{valid_horses_list}

**RÈGLE ABSOLUE** : Tu ne peux proposer QUE des chevaux de cette liste ci-dessus !
Si un cheval n'est pas dans cette liste, tu ne peux PAS le proposer !

{learned_lessons}

## TA MISSION

### 1. Auto-critique rapide
- Quels biais potentiels as-tu pu avoir ?
- Quelles limitations à ton analyse ?

### 2. PROPOSE DES PICKS FINAUX
Sélectionne les meilleurs picks parmi ceux analysés, sur PLUSIEURS COURSES DIFFÉRENTES !

Pour chaque pick recommandé par Step B avec KEEP ou KEEP_REDUCED:
- **Action**: KEEP ou KEEP_REDUCED (réduit de 50% seulement si doutes sérieux)
- **Confidence**: Score 0-100 reflétant ta confiance
- **Justification**: 1 phrase expliquant pourquoi ce pick

### 3. Résumé exécutif
2-3 phrases sur la stratégie du jour, en mentionnant le nombre de courses couvertes.

## RÈGLE ABSOLUE POUR LES MISES
L'algorithme a calculé les mises optimales selon le critère de Kelly adapté à ta bankroll.

**UTILISE EXACTEMENT LE STAKE PRÉ-CALCULÉ** :
- Chaque cheval dans la liste "{valid_horses_list}" a un champ `stake` pré-calculé
- Utilise EXACTEMENT cette valeur pour `stake_eur` - NE LA MODIFIE PAS !
- Si action = KEEP_REDUCED : divise le stake par 2 (minimum 1€)
- NE JAMAIS inventer de stake - utilise celui fourni par l'algorithme

Budget jour: {daily_budget}€ | Mise max: {max_stake_per_bet}€

## RÈGLES DE DIVERSIFICATION (ANTI-BIAIS R1)
- Couvre AU MOINS 2-3 RÉUNIONS DIFFÉRENTES (R1, R2, R3, etc.)
- Maximum 3 picks par réunion
- Si tous tes picks sont sur R1, c'est INCORRECT - diversifie !
- Les réunions R2, R3, R4 ont aussi de bons chevaux

## FORMAT DES DONNÉES (OBLIGATOIRE)
Pour chaque pick:
- **race_key**: EXACTEMENT comme dans les données (format "YYYY-MM-DD|Rn|Cn|HIPPODROME")
- **hippodrome**: EXACTEMENT comme dans les données
- **bet_type**: EXACTEMENT le type de pari indiqué par l'algo (SIMPLE PLACÉ, E/P, SIMPLE GAGNANT)
- **runner_id**: Numéro de dossard
- **horse_name**: EXACTEMENT comme dans les données
- **stake_eur**: ⚠️ COPIE LE STAKE DE L'ALGO ! Ne pas modifier !
- **confidence_score**: 0-100

## RÈGLE ANTI-HALLUCINATION
- Propose UNIQUEMENT des chevaux présents dans les données
- NE PAS inventer de noms, race_keys ou stakes

Réponds UNIQUEMENT avec un JSON valide respectant le schéma demandé."""

PROMPT_STEP_D_META = PromptMetadata(
    name="step_d_final",
    version="4.0.0",  # V4: Multi-course diversification mandatory
    step="D",
    description="Décision finale avec diversification multi-courses obligatoire",
    hash=compute_prompt_hash(PROMPT_STEP_D_V1),
)


# =============================================================================
# PROMPT DE CORRECTION (RETRY)
# =============================================================================

PROMPT_RETRY_VALIDATION = """Ton JSON précédent ne respecte pas le schéma attendu.

## ERREURS DE VALIDATION
{validation_errors}

## SCHÉMA ATTENDU
{schema_description}

Corrige UNIQUEMENT le JSON pour respecter le schéma. Pas de texte, juste le JSON corrigé."""


# =============================================================================
# REGISTRE DES PROMPTS
# =============================================================================

PROMPT_REGISTRY = {
    "B": {
        "prompt": PROMPT_STEP_B_V1,
        "meta": PROMPT_STEP_B_META,
    },
    "C": {
        "prompt": PROMPT_STEP_C_V1,
        "meta": PROMPT_STEP_C_META,
    },
    "D": {
        "prompt": PROMPT_STEP_D_V1,
        "meta": PROMPT_STEP_D_META,
    },
    "RETRY": {
        "prompt": PROMPT_RETRY_VALIDATION,
        "meta": PromptMetadata(
            name="retry_validation",
            version="1.0.0",
            step="*",
            description="Prompt de correction après erreur de validation",
            hash=compute_prompt_hash(PROMPT_RETRY_VALIDATION),
        ),
    },
}


def get_prompt(step: str) -> tuple[str, PromptMetadata]:
    """
    Récupère un prompt et ses métadonnées.

    Args:
        step: Nom de l'étape (B, C, D, RETRY)

    Returns:
        Tuple (prompt template, métadonnées)
    """
    if step not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown prompt step: {step}")

    entry = PROMPT_REGISTRY[step]
    return entry["prompt"], entry["meta"]


def format_prompt(step: str, **kwargs) -> tuple[str, PromptMetadata]:
    """
    Formate un prompt avec les variables.

    Args:
        step: Nom de l'étape
        **kwargs: Variables à injecter dans le template

    Returns:
        Tuple (prompt formaté, métadonnées)
    """
    template, meta = get_prompt(step)

    # Formater le prompt
    formatted = template.format(**kwargs)

    # Recalculer le hash avec les données injectées
    new_hash = compute_prompt_hash(formatted)

    # Créer une copie des métadonnées avec le nouveau hash
    formatted_meta = PromptMetadata(
        name=meta.name,
        version=meta.version,
        step=meta.step,
        description=meta.description,
        hash=new_hash,
    )

    return formatted, formatted_meta


def list_prompts() -> list[dict]:
    """Liste tous les prompts disponibles avec leurs métadonnées"""
    return [
        {
            "step": step,
            **entry["meta"].to_dict(),
        }
        for step, entry in PROMPT_REGISTRY.items()
    ]
