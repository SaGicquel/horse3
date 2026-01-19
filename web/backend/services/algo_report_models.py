"""
🎯 Modèles Pydantic pour le Rapport Algo (Agent IA - Étape A)
=============================================================

Schéma JSON structuré pour l'export complet des décisions de l'algorithme.
Ce rapport est ensuite analysé par l'Agent IA (Étapes B, C, D).

## Sémantique des champs (IMPORTANT)

- odds_morning / odds_final: Cotes DÉCIMALES européennes (ex: 5.0 = "4/1")
- p_model_*: Probabilités CALIBRÉES [0,1], pas des scores bruts
- value_*_pct: Expected Value approx = (p_model × odds - 1) × 100
- kelly_*_pct: Fraction Kelly = (p × b - q) / b × 100, où b = odds - 1

Auteur: Agent IA Pipeline
Date: 2024-12-21
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, date
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# =============================================================================
# VERSIONING
# =============================================================================

# Schéma version: MAJOR.MINOR.PATCH
# - MAJOR: Changement breaking (structure incompatible)
# - MINOR: Ajout de champs (rétrocompatible)
# - PATCH: Corrections/clarifications
SCHEMA_VERSION = "1.1.0"

# Policy version: identifie les règles métier appliquées
POLICY_VERSION = "2024.12.21"

# Default model version (overridden by actual model)
DEFAULT_MODEL_VERSION = "xgb_proba_v9"


# =============================================================================
# ENUMS
# =============================================================================


class DecisionStatus(str, Enum):
    """Statut de la décision algo pour un partant"""

    KEPT = "KEPT"  # Paris gardé
    REJECTED = "REJECTED"  # Paris rejeté


class BetType(str, Enum):
    """Types de paris supportés"""

    SIMPLE_GAGNANT = "SIMPLE GAGNANT"
    SIMPLE_PLACE = "SIMPLE PLACÉ"
    EP_GAGNANT_PLACE = "E/P (GAGNANT-PLACÉ)"
    COUPLE = "COUPLE"
    TRIO = "TRIO"
    QUARTE = "QUARTÉ+"
    QUINTE = "QUINTÉ+"


class RiskLevel(str, Enum):
    """Niveaux de risque"""

    FAIBLE = "Faible"
    MODERE = "Modéré"
    ELEVE = "Élevé"
    TRES_ELEVE = "Très élevé"


class DriftStatus(str, Enum):
    """Statut de dérive du modèle"""

    OK = "OK"
    WARN = "WARN"
    ALERT = "ALERT"


# =============================================================================
# MODÈLES DE DÉTAIL
# =============================================================================


class AlgoDecision(BaseModel):
    """Décision de l'algorithme pour un partant"""

    status: DecisionStatus = Field(..., description="KEPT si le pari est gardé, REJECTED sinon")
    bet_type: Optional[str] = Field(
        None, description="Type de pari proposé (SIMPLE GAGNANT, SIMPLE PLACÉ, E/P, etc.)"
    )
    stake_eur: Optional[float] = Field(
        None, ge=0, description="Mise proposée en euros (null si rejeté)"
    )
    kelly_raw_pct: Optional[float] = Field(None, description="Kelly brut calculé (%)")
    kelly_adjusted_pct: Optional[float] = Field(
        None, description="Kelly après ajustements (fraction, caps)"
    )
    ev_eur: Optional[float] = Field(None, description="Expected Value en euros")

    # Justifications
    why_kept: list[str] = Field(
        default_factory=list, description="Raisons pour lesquelles le pari est gardé"
    )
    failed_rules: list[str] = Field(default_factory=list, description="Règles violées (si rejeté)")
    exclude_reason: Optional[str] = Field(
        None, description="Raison principale d'exclusion (si rejeté)"
    )


class RunnerAnalysis(BaseModel):
    """
    Analyse complète d'un partant dans une course.

    SÉMANTIQUE DES CHAMPS:
    - p_model_*: Probabilités CALIBRÉES [0,1] issues du modèle XGBoost + Platt
    - odds_*: Cotes DÉCIMALES européennes (ex: 5.0 signifie "parie 1€, gagne 5€")
    - value_*_pct: EV approx = (p_model × odds - 1) × 100 (en pourcentage)
    - kelly_*_pct: Fraction Kelly optimale (en pourcentage de bankroll)
    """

    # Identifiants
    runner_id: str = Field(..., description="ID unique du partant (ex: pmu:123456)")
    horse_name: str = Field(..., description="Nom du cheval")
    numero: Optional[int] = Field(None, description="Numéro de dossard")

    # Probabilités modèle - CALIBRÉES [0,1]
    p_model_win: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probabilité CALIBRÉE victoire [0,1], issue XGBoost + Platt/Isotonic",
    )
    p_model_place: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Probabilité CALIBRÉE placé [0,1], issue XGBoost + Platt/Isotonic",
    )

    # Cotes - DÉCIMALES européennes (1.0 = mise récupérée, >1 = gain)
    odds_morning: Optional[float] = Field(
        None, ge=1, description="Cote DÉCIMALE du matin (ex: 5.0 = rapport 5:1)"
    )
    odds_final: Optional[float] = Field(
        None, ge=1, description="Cote DÉCIMALE finale pré-départ (ex: 4.5 = rapport 4.5:1)"
    )
    odds_implied_prob: Optional[float] = Field(
        None, ge=0, le=1, description="Probabilité implicite marché = 1/odds_final"
    )

    # Value - Expected Value approx en %
    value_win_pct: Optional[float] = Field(
        None, description="Value victoire (%) = (p_model_win × odds - 1) × 100. Positif = +EV"
    )
    value_place_pct: Optional[float] = Field(
        None, description="Value placé (%) = (p_model_place × odds_place - 1) × 100"
    )

    # Kelly - fraction optimale en % de bankroll
    kelly_win_pct: Optional[float] = Field(
        None, description="Kelly victoire (%) = ((p×(odds-1) - (1-p)) / (odds-1)) × 100"
    )
    kelly_place_pct: Optional[float] = Field(
        None, description="Kelly placé (%) = fraction optimale pour pari placé"
    )

    # Signaux qualitatifs
    signals_positive: list[str] = Field(
        default_factory=list, description="Signaux positifs détectés (forme, jockey, etc.)"
    )
    signals_negative: list[str] = Field(
        default_factory=list, description="Signaux négatifs détectés (rentrée, distance?, etc.)"
    )

    # Risque évalué
    bet_risk: Optional[str] = Field(None, description="Niveau de risque (Faible/Modéré/Élevé)")

    # Décision algo
    algo_decision: AlgoDecision = Field(..., description="Décision de l'algorithme")

    # Données contextuelles pour vérification IA
    jockey: Optional[str] = Field(None, description="Nom du jockey/driver")
    trainer: Optional[str] = Field(None, description="Nom de l'entraîneur")
    musique: Optional[str] = Field(None, description="Musique récente (ex: 1p2a3s)")
    nb_courses_total: Optional[int] = Field(None, description="Nombre total de courses")
    nb_victoires_total: Optional[int] = Field(None, description="Nombre total de victoires")


class RaceAnalysis(BaseModel):
    """Analyse complète d'une course"""

    # Identifiants
    race_id: str = Field(..., description="Identifiant unique de la course (ex: R3C5)")
    race_key: str = Field(..., description="Clé complète (date|reunion|course|hippodrome)")

    # Contexte
    hippodrome: str = Field(..., description="Nom de l'hippodrome")
    discipline: str = Field(..., description="Discipline (trot/plat/obstacle)")
    distance_m: Optional[int] = Field(None, description="Distance en mètres")
    start_time: Optional[str] = Field(None, description="Heure de départ (HH:MM)")
    race_name: Optional[str] = Field(None, description="Nom de la course")
    allocation_eur: Optional[int] = Field(None, description="Allocation totale en euros")

    # Conditions
    terrain_state: Optional[str] = Field(None, description="État du terrain")
    weather: Optional[str] = Field(None, description="Conditions météo")
    nb_partants: Optional[int] = Field(None, description="Nombre de partants")

    # Partants avec analyses
    runners: list[RunnerAnalysis] = Field(
        default_factory=list, description="Liste des partants avec leurs analyses"
    )

    # Résumé décisions
    kept_runners: list[str] = Field(default_factory=list, description="Liste des runner_id gardés")
    rejected_runners: list[str] = Field(
        default_factory=list, description="Liste des runner_id rejetés"
    )

    # Stats de la course
    total_stake_eur: Optional[float] = Field(None, description="Mise totale sur cette course")
    total_ev_eur: Optional[float] = Field(None, description="EV totale sur cette course")


class PolicyConstraints(BaseModel):
    """Contraintes et seuils de la politique de mise"""

    # Zone bankroll
    zone: str = Field(..., description="Zone bankroll (micro/small/full)")
    profile: str = Field(..., description="Profil utilisateur (PRUDENT/STANDARD/AGRESSIF)")

    # Quotas
    max_bets_per_day: int = Field(..., description="Max paris par jour")
    max_bets_per_race: int = Field(..., description="Max paris par course")

    # Seuils value/proba
    min_proba_model: Optional[float] = Field(None, description="Proba minimum requise")
    value_cutoff_win_pct: Optional[float] = Field(None, description="Seuil value victoire (%)")
    value_cutoff_place_pct: Optional[float] = Field(None, description="Seuil value placé (%)")
    max_odds_win: Optional[float] = Field(None, description="Cote max autorisée")

    # Mises
    kelly_fraction: float = Field(..., description="Fraction Kelly utilisée")
    cap_per_bet: float = Field(..., description="Cap par pari (% bankroll)")
    daily_budget_rate: float = Field(..., description="Budget jour (% bankroll)")

    # Risques autorisés
    allowed_risks: list[str] = Field(
        default_factory=list, description="Niveaux de risque autorisés"
    )
    allowed_bet_types: Optional[list[str]] = Field(None, description="Types de paris autorisés")

    # Garde-fous
    max_drawdown_stop: Optional[float] = Field(None, description="Stop-loss drawdown (%)")


class AlgoMetrics(BaseModel):
    """Métriques du modèle et état du système"""

    model_version: str = Field(..., description="Version du modèle XGBoost")
    calibrator_type: Optional[str] = Field(None, description="Type de calibration (platt/isotonic)")
    temperature: Optional[float] = Field(None, description="Température softmax")
    blend_alpha: Optional[float] = Field(None, description="Alpha de blend modèle/marché")
    drift_status: DriftStatus = Field(
        default=DriftStatus.OK, description="Statut de dérive du modèle"
    )
    last_retrain_date: Optional[date] = Field(None, description="Date dernier entraînement")


# =============================================================================
# RAPPORT ALGO COMPLET
# =============================================================================


class ReplayInputs(BaseModel):
    """
    Inputs pour replay déterministe.
    Stocke tous les paramètres nécessaires pour reproduire exactement le même rapport.
    """

    bankroll: float
    profile: str
    target_date: str  # ISO format
    policy_version: str
    model_version: str
    seed: Optional[int] = Field(None, description="Seed pour stochastique (Monte Carlo exotics)")

    def compute_hash(self) -> str:
        """Calcule un hash SHA256 des inputs pour vérifier la reproductibilité"""
        data = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class RapportAlgo(BaseModel):
    """
    Rapport Algo Complet - Format JSON standardisé v1.1

    Ce rapport contient toutes les décisions de l'algorithme de sélection
    pour une date donnée, avec justifications complètes pour chaque décision.

    VERSIONING:
    - schema_version: Version du format JSON (breaking changes = major bump)
    - policy_version: Version des règles métier appliquées
    - model_version: Version du modèle ML utilisé

    REPRODUCTIBILITÉ:
    - replay_inputs: Tous les paramètres pour rejouer le run
    - inputs_hash: Hash des inputs pour vérifier le replay

    Utilisé par l'Agent IA pour:
    1. Analyser les décisions (Étape B)
    2. Vérifier la cohérence (Étape C)
    3. Produire une auto-critique et proposition finale (Étape D)
    """

    # =========================================================================
    # VERSIONING (CRITIQUE pour stabilité)
    # =========================================================================
    schema_version: str = Field(
        default=SCHEMA_VERSION, description="Version du schéma JSON (ex: 1.1.0)"
    )
    policy_version: str = Field(
        default=POLICY_VERSION, description="Version des règles métier (ex: 2024.12.21)"
    )

    # =========================================================================
    # MÉTADONNÉES
    # =========================================================================
    run_id: UUID = Field(default_factory=uuid4, description="Identifiant unique du run")
    generated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp de génération UTC"
    )

    # =========================================================================
    # REPRODUCTIBILITÉ / REPLAY
    # =========================================================================
    replay_inputs: Optional[ReplayInputs] = Field(
        None, description="Inputs pour rejouer ce run de manière déterministe"
    )
    inputs_hash: Optional[str] = Field(
        None, description="SHA256 hash des inputs (16 chars) pour vérifier replay"
    )

    # =========================================================================
    # CONTEXTE UTILISATEUR
    # =========================================================================
    target_date: date = Field(..., description="Date des courses analysées")
    user_id: Optional[int] = Field(None, description="ID utilisateur (si authentifié)")
    bankroll_eur: float = Field(..., ge=0, description="Bankroll en euros")
    profile: str = Field(default="STANDARD", description="Profil de risque")

    # =========================================================================
    # POLITIQUE ET MODÈLE
    # =========================================================================
    policy_constraints: PolicyConstraints = Field(
        ..., description="Contraintes et seuils de la politique de mise"
    )
    algo_metrics: AlgoMetrics = Field(..., description="Métriques du modèle et état du système")

    # =========================================================================
    # DONNÉES DE COURSES
    # =========================================================================
    races: list[RaceAnalysis] = Field(
        default_factory=list, description="Liste des courses avec leurs analyses"
    )

    # =========================================================================
    # RÉSUMÉ ET CAPS
    # =========================================================================
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Résumé agrégé (total picks, excluded, stake, etc.)"
    )
    caps: dict[str, float] = Field(
        default_factory=dict, description="Caps calculés (daily_budget, max_stake_per_bet, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "generated_at": "2024-12-21T15:30:00Z",
                "target_date": "2024-12-21",
                "bankroll_eur": 500.0,
                "profile": "STANDARD",
                "races": [],
                "summary": {
                    "total_races_analyzed": 12,
                    "total_picks_kept": 4,
                    "total_picks_rejected": 28,
                    "total_stake_eur": 45.0,
                    "total_ev_eur": 6.75,
                },
            }
        }


# =============================================================================
# MODÈLES DE REQUÊTE/RÉPONSE API
# =============================================================================


class GenerateReportRequest(BaseModel):
    """Requête pour générer un rapport algo"""

    target_date: Optional[date] = Field(None, description="Date cible (par défaut: aujourd'hui)")
    bankroll: Optional[float] = Field(
        None, ge=0, description="Bankroll en euros (par défaut: settings utilisateur)"
    )
    profile: Optional[str] = Field(
        None, description="Profil de risque (par défaut: settings utilisateur)"
    )


class GenerateReportResponse(BaseModel):
    """Réponse de génération de rapport"""

    success: bool
    run_id: UUID
    message: str
    report: Optional[RapportAlgo] = None
    generation_time_ms: Optional[float] = None
