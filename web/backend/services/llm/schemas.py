"""
📋 LLM Output Schemas - SIMPLIFIED for Gemini Compatibility
============================================================

Schémas Pydantic SIMPLIFIÉS pour une meilleure compatibilité avec Gemini.
Gemini sans response_schema a du mal avec les structures imbriquées complexes.

STRATÉGIE:
- Schémas plats (éviter les nested objects required)
- Valeurs par défaut sur tous les champs possibles
- Strings simples plutôt qu'Enums quand possible
- Listes de dicts simples plutôt que nested Pydantic models

Auteur: Agent IA Pipeline
Date: 2024-12-21
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# =============================================================================
# ENUMS SIMPLES (convertis en strings pour Gemini)
# =============================================================================


class ConfidenceLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class RecommendationAction(str, Enum):
    KEEP = "KEEP"
    KEEP_REDUCED = "KEEP_REDUCED"
    REMOVE = "REMOVE"
    FLAG = "FLAG"


class RiskAssessment(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    UNACCEPTABLE = "UNACCEPTABLE"


# =============================================================================
# STEP B - ANALYSE IA (SIMPLIFIÉ)
# =============================================================================


class StepBOutput(BaseModel):
    """
    Sortie de l'étape B - Analyse IA du rapport algo.
    VERSION SIMPLIFIÉE pour compatibilité Gemini.
    """

    # Résumé global (REQUIRED)
    global_assessment: str = Field(
        ..., description="Évaluation globale du portefeuille en 2-3 phrases"
    )
    market_conditions: str = Field(
        default="Conditions normales", description="Analyse courte des conditions de marché"
    )

    # Confiance (string simple plutôt qu'enum)
    overall_confidence: str = Field(
        default="MEDIUM", description="Niveau de confiance: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH"
    )

    # Compteurs (avec defaults)
    total_reviewed: int = Field(default=0, description="Nombre de picks analysés")
    recommended_keep: int = Field(default=0, description="Nombre à garder")
    recommended_remove: int = Field(default=0, description="Nombre à supprimer")

    # Liste des recommandations (flat list of dicts, pas de nested models)
    picks_analysis: list[dict] = Field(
        default_factory=list,
        description="Liste des analyses: [{runner_id, horse_name, action, reason}]",
    )

    # Observations clés (simple list)
    key_observations: list[str] = Field(
        default_factory=list, description="3-5 observations clés sur le portefeuille"
    )

    # Risques identifiés (simple list)
    risk_factors: list[str] = Field(
        default_factory=list, description="Facteurs de risque principaux"
    )


# =============================================================================
# STEP C - VÉRIFICATION (SIMPLIFIÉ)
# =============================================================================


class StepCOutput(BaseModel):
    """
    Sortie de l'étape C - Vérification des claims.
    VERSION SIMPLIFIÉE pour compatibilité Gemini.
    """

    # Stats de vérification
    total_claims: int = Field(default=0, description="Nombre de claims vérifiés")
    verified_count: int = Field(default=0, description="Claims confirmés")
    unverified_count: int = Field(default=0, description="Claims non vérifiables")
    contradictions_count: int = Field(default=0, description="Contradictions trouvées")

    # Taux (0-100%)
    verification_rate_pct: int = Field(
        default=0, ge=0, le=100, description="Taux de vérification en pourcentage (0-100)"
    )

    # Claims vérifiés (flat list)
    verified_claims: list[str] = Field(
        default_factory=list, description="Liste des claims vérifiés"
    )

    # Contradictions (flat list)
    contradictions: list[str] = Field(
        default_factory=list, description="Contradictions trouvées avec corrections"
    )

    # Ajustement confiance
    confidence_adjustment: str = Field(
        default="Aucun ajustement nécessaire", description="Impact sur le niveau de confiance"
    )


# =============================================================================
# STEP D - PROPOSITION FINALE (SIMPLIFIÉ)
# =============================================================================


class StepDOutput(BaseModel):
    """
    Sortie de l'étape D - Proposition finale avec auto-critique.
    VERSION SIMPLIFIÉE pour compatibilité Gemini.
    """

    # Auto-critique (simple strings)
    analysis_quality: str = Field(
        default="Analyse standard", description="Évaluation de la qualité de l'analyse"
    )
    potential_biases: list[str] = Field(
        default_factory=list, description="Biais potentiels identifiés"
    )
    limitations: list[str] = Field(default_factory=list, description="Limitations de l'analyse")

    # Picks finaux (flat list of dicts)
    final_picks: list[dict] = Field(
        default_factory=list,
        description=(
            "Picks finaux DIVERSIFIÉS sur PLUSIEURS COURSES DIFFÉRENTES (au moins 3-4 courses, max 3 par course). "
            "Format obligatoire pour chaque pick: "
            "{runner_id: int, horse_name: str, race_key: 'YYYY-MM-DD|Rn|Cn|HIPPODROME', "
            "hippodrome: 'HIPPODROME DE [NOM]', bet_type: 'SIMPLE GAGNANT' ou 'SIMPLE PLACÉ', "
            "action: 'KEEP' ou 'KEEP_REDUCED', stake_eur: float, confidence_score: int 0-100, "
            "justification: str}"
        ),
    )

    # Totaux
    total_picks: int = Field(default=0, ge=0, description="Nombre de picks")
    total_stake_eur: float = Field(default=0.0, ge=0, description="Mise totale €")
    expected_ev_eur: float = Field(default=0.0, description="EV estimée €")

    # Confiance (int simple)
    portfolio_confidence: int = Field(
        default=50, ge=0, le=100, description="Confiance globale 0-100"
    )

    # Risque (string)
    risk_assessment: str = Field(
        default="ELEVATED", description="Niveau de risque: ACCEPTABLE, ELEVATED, HIGH, UNACCEPTABLE"
    )

    # Résumé
    executive_summary: str = Field(
        ..., description="Résumé exécutif en 2-3 phrases pour l'utilisateur"
    )


# =============================================================================
# REGISTRE DES SCHÉMAS
# =============================================================================

SCHEMA_REGISTRY = {
    "B": StepBOutput,
    "C": StepCOutput,
    "D": StepDOutput,
}


def get_schema_for_step(step_name: str) -> type[BaseModel]:
    """Retourne le schéma Pydantic pour une étape"""
    if step_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown step: {step_name}")
    return SCHEMA_REGISTRY[step_name]
