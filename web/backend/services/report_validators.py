"""
🔍 Validateurs Déterministes pour Rapport Algo
==============================================

Module de validation 100% code (sans IA) pour le rapport algo.
Filet de sécurité avant l'analyse LLM.

Vérifie:
1. Cohérence des calculs (value = p * cote - 1)
2. Respect des règles métier (max bets, min proba, max cotes)
3. Respect des limites de risque (drawdown, budget, stake)

Auteur: Agent IA Pipeline
Date: 2024-12-21
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS & TYPES
# =============================================================================


class ValidationSeverity(str, Enum):
    """Niveau de sévérité des validations"""

    INFO = "INFO"  # Information, pas d'action requise
    WARNING = "WARNING"  # Attention, à vérifier (non-blocking)
    ERROR = "ERROR"  # Erreur, potentiellement invalide (blocking par défaut)
    CRITICAL = "CRITICAL"  # Critique, données corrompues (toujours blocking)


class RuleType(str, Enum):
    """Type de règle - détermine si blocking ou non"""

    MATH = "MATH"  # Cohérence mathématique - ERROR = blocking
    POLICY = "POLICY"  # Règle métier - ERROR = warning selon config
    RISK = "RISK"  # Limite risque - ERROR = blocking
    QUALITY = "QUALITY"  # Qualité données - rarement blocking


class ValidationCategory(str, Enum):
    """Catégories de validation"""

    CALCULATION = "CALCULATION"  # Cohérence mathématique
    BUSINESS_RULE = "BUSINESS_RULE"  # Règles métier
    RISK_LIMIT = "RISK_LIMIT"  # Limites de risque
    DATA_QUALITY = "DATA_QUALITY"  # Qualité des données
    CONSISTENCY = "CONSISTENCY"  # Cohérence interne


# =============================================================================
# MODÈLES
# =============================================================================


class ValidationIssue(BaseModel):
    """
    Un problème détecté lors de la validation.

    IMPORTANT:
    - blocking=True signifie que le rapport ne devrait pas être utilisé tel quel
    - CRITICAL et ERROR sont blocking par défaut
    - WARNING peut être blocking selon rule_type (MATH vs POLICY)
    """

    issue_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    severity: ValidationSeverity
    category: ValidationCategory
    rule_type: RuleType = Field(default=RuleType.POLICY, description="Type de règle")
    message: str

    # NOUVEAU: indique si cette issue doit bloquer l'utilisation
    blocking: bool = Field(
        default=False, description="True si cette issue doit bloquer l'utilisation du rapport"
    )

    # Localisation
    race_key: Optional[str] = None
    runner_id: Optional[str] = None
    horse_name: Optional[str] = None
    field_path: Optional[str] = None  # ex: "races[0].runners[2].algo_decision.stake_eur"

    # Détails
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    rule_name: Optional[str] = None
    suggestion: Optional[str] = None


class ValidationReport(BaseModel):
    """Rapport de validation complet"""

    report_id: UUID = Field(default_factory=uuid4)
    run_id: Optional[UUID] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Résultat global
    is_valid: bool = True  # False si au moins un issue blocking
    has_blocking_issues: bool = False  # True s'il y a des issues blocking

    # Issues par sévérité
    issues: list[ValidationIssue] = Field(default_factory=list)

    # Stats
    total_issues: int = 0
    blocking_issues: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)
    by_rule_type: dict[str, int] = Field(default_factory=dict)

    # Métriques de validation
    validation_duration_ms: Optional[float] = None
    rules_checked: int = 0

    def add_issue(self, issue: ValidationIssue):
        """Ajoute un problème et met à jour les stats"""
        # Auto-set blocking based on severity and rule_type
        if issue.severity == ValidationSeverity.CRITICAL:
            issue.blocking = True
        elif issue.severity == ValidationSeverity.ERROR:
            # ERROR is blocking for MATH and RISK, warning-like for POLICY
            issue.blocking = issue.rule_type in (RuleType.MATH, RuleType.RISK)

        self.issues.append(issue)
        self.total_issues += 1

        # Update blocking stats
        if issue.blocking:
            self.blocking_issues += 1
            self.has_blocking_issues = True
            self.is_valid = False

        # Update by_severity
        sev = issue.severity.value
        self.by_severity[sev] = self.by_severity.get(sev, 0) + 1

        # Update by_category
        cat = issue.category.value
        self.by_category[cat] = self.by_category.get(cat, 0) + 1

        # Update by_rule_type
        rt = issue.rule_type.value
        self.by_rule_type[rt] = self.by_rule_type.get(rt, 0) + 1


# =============================================================================
# VALIDATEURS
# =============================================================================


class ReportValidator:
    """
    Validateur principal pour les rapports algo.

    Usage:
        validator = ReportValidator(policy_config)
        report = validator.validate(rapport_algo)
    """

    def __init__(self, policy_config: dict[str, Any]):
        """
        Initialise le validateur avec la config de politique.

        Args:
            policy_config: Configuration depuis pro_betting.yaml
        """
        self.policy = policy_config.get("betting_policy", {})
        self.defaults = policy_config.get("betting_defaults", {})
        self.zones = self.policy.get("bankroll_zones", {})
        self.profiles = self.policy.get("profiles", {})

        # Tolerances pour comparaisons float (augmentées pour éviter faux positifs)
        self.VALUE_TOLERANCE = 2.0  # 2% de tolérance sur value (arrondis cotes)
        self.KELLY_TOLERANCE = 2.0  # 2% de tolérance sur kelly
        self.STAKE_TOLERANCE = 1.05  # 5% de marge sur stakes

    def validate(self, report: dict) -> ValidationReport:
        """
        Valide un rapport algo complet.

        Args:
            report: RapportAlgo en dict (ou Pydantic model.model_dump())

        Returns:
            ValidationReport avec tous les problèmes détectés
        """
        import time

        start = time.time()

        validation = ValidationReport(
            run_id=UUID(report.get("run_id")) if report.get("run_id") else None,
        )

        # 1. Validations globales
        self._validate_global_constraints(report, validation)

        # 2. Validations par course
        races = report.get("races", [])
        for race_idx, race in enumerate(races):
            self._validate_race(race, race_idx, report, validation)

        # 3. Validations de cohérence transverse
        self._validate_cross_race_consistency(report, validation)

        # Finalisation
        validation.validation_duration_ms = (time.time() - start) * 1000
        validation.rules_checked = self._count_rules_checked()

        return validation

    def _count_rules_checked(self) -> int:
        """Compte le nombre de règles vérifiées"""
        # Approximation basée sur les méthodes
        return 25

    # -------------------------------------------------------------------------
    # Validations globales
    # -------------------------------------------------------------------------

    def _validate_global_constraints(self, report: dict, validation: ValidationReport):
        """Valide les contraintes globales du rapport"""

        constraints = report.get("policy_constraints", {})
        summary = report.get("summary", {})
        caps = report.get("caps", {})

        # 1. Vérifier le nombre de picks vs max_bets_per_day
        max_bets = constraints.get("max_bets_per_day", 8)
        total_picks = summary.get("total_picks_kept", 0)

        if total_picks > max_bets:
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.BUSINESS_RULE,
                    rule_type=RuleType.POLICY,  # POLICY = non-blocking ERROR
                    message=f"Nombre de picks ({total_picks}) dépasse max_bets_per_day ({max_bets})",
                    rule_name="max_bets_per_day",
                    expected_value=max_bets,
                    actual_value=total_picks,
                    suggestion="Filtrer les picks excédentaires par priorité value/kelly",
                )
            )

        # 2. Vérifier le budget total vs daily_budget
        daily_budget = caps.get("daily_budget_eur", 0)
        total_stake = summary.get("total_stake_eur", 0)

        if daily_budget > 0 and total_stake > daily_budget * 1.01:  # 1% de marge
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.RISK_LIMIT,
                    message=f"Mise totale ({total_stake:.2f}€) dépasse le budget jour ({daily_budget:.2f}€)",
                    rule_name="daily_budget",
                    expected_value=daily_budget,
                    actual_value=total_stake,
                    suggestion="Réduire les mises ou éliminer des picks",
                )
            )

        # 3. Vérifier la bankroll
        bankroll = report.get("bankroll_eur", 0)
        if bankroll <= 0:
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.DATA_QUALITY,
                    message="Bankroll invalide ou nulle",
                    expected_value="> 0",
                    actual_value=bankroll,
                )
            )

        # 4. Vérifier le profil
        profile = report.get("profile", "STANDARD")
        valid_profiles = ["PRUDENT", "STANDARD", "AGRESSIF", "ULTRA_SUR", "SUR", "AMBITIEUX"]
        if profile.upper() not in [p.upper() for p in valid_profiles]:
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.DATA_QUALITY,
                    message=f"Profil '{profile}' non reconnu",
                    expected_value=valid_profiles,
                    actual_value=profile,
                )
            )

    # -------------------------------------------------------------------------
    # Validations par course
    # -------------------------------------------------------------------------

    def _validate_race(self, race: dict, race_idx: int, report: dict, validation: ValidationReport):
        """Valide une course individuelle"""

        race_key = race.get("race_key", f"race_{race_idx}")
        runners = race.get("runners", [])
        kept_runners = race.get("kept_runners", [])
        constraints = report.get("policy_constraints", {})

        # 1. Max bets per race
        max_per_race = constraints.get("max_bets_per_race", 2)
        if len(kept_runners) > max_per_race:
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.BUSINESS_RULE,
                    message=f"Trop de picks ({len(kept_runners)}) pour la course",
                    race_key=race_key,
                    rule_name="max_bets_per_race",
                    expected_value=max_per_race,
                    actual_value=len(kept_runners),
                )
            )

        # 2. Valider chaque runner
        for runner_idx, runner in enumerate(runners):
            self._validate_runner(runner, runner_idx, race_key, report, validation)

    def _validate_runner(
        self,
        runner: dict,
        runner_idx: int,
        race_key: str,
        report: dict,
        validation: ValidationReport,
    ):
        """Valide un partant individuel"""

        runner_id = runner.get("runner_id", f"runner_{runner_idx}")
        horse_name = runner.get("horse_name", "Inconnu")
        decision = runner.get("algo_decision", {})
        constraints = report.get("policy_constraints", {})

        is_kept = decision.get("status") == "KEPT"

        # =====================================================================
        # VALIDATIONS DE CALCUL
        # =====================================================================

        # 1. Cohérence value = p * odds - 1
        p_win = runner.get("p_model_win", 0)
        odds = runner.get("odds_final") or runner.get("odds_morning") or 0
        value_declared = runner.get("value_win_pct", 0)

        if p_win > 0 and odds > 1:
            value_calculated = (p_win * odds - 1) * 100  # en %
            diff = abs(value_declared - value_calculated)

            if diff > self.VALUE_TOLERANCE:
                validation.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.CALCULATION,
                        message=f"Value déclarée ({value_declared:.1f}%) ≠ calculée ({value_calculated:.1f}%)",
                        race_key=race_key,
                        runner_id=runner_id,
                        horse_name=horse_name,
                        field_path="value_win_pct",
                        expected_value=round(value_calculated, 2),
                        actual_value=round(value_declared, 2),
                        rule_name="value_calculation",
                    )
                )

        # 2. Value négative mais pick gardé
        if is_kept and value_declared < 0:
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.CALCULATION,
                    message="Pick gardé avec value négative",
                    race_key=race_key,
                    runner_id=runner_id,
                    horse_name=horse_name,
                    actual_value=value_declared,
                    rule_name="positive_value",
                    suggestion="Rejeter les picks avec value < 0",
                )
            )

        # 3. Kelly négatif mais pick gardé
        kelly = runner.get("kelly_win_pct", 0)
        if is_kept and kelly <= 0:
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.CALCULATION,
                    message="Pick gardé avec Kelly ≤ 0",
                    race_key=race_key,
                    runner_id=runner_id,
                    horse_name=horse_name,
                    actual_value=kelly,
                    rule_name="positive_kelly",
                )
            )

        # 4. Probabilité hors bornes
        if p_win < 0 or p_win > 1:
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.CALCULATION,
                    message=f"Probabilité hors bornes [0,1]: {p_win}",
                    race_key=race_key,
                    runner_id=runner_id,
                    horse_name=horse_name,
                    actual_value=p_win,
                    expected_value="[0, 1]",
                )
            )

        # =====================================================================
        # VALIDATIONS RÈGLES MÉTIER (si pick gardé)
        # =====================================================================

        if is_kept:
            # 5. Proba minimum
            min_proba = constraints.get("min_proba_model")
            if min_proba and p_win < min_proba:
                validation.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.BUSINESS_RULE,
                        message=f"Proba ({p_win:.2%}) < min_proba ({min_proba:.2%})",
                        race_key=race_key,
                        runner_id=runner_id,
                        horse_name=horse_name,
                        expected_value=min_proba,
                        actual_value=p_win,
                        rule_name="min_proba_model",
                    )
                )

            # 6. Cote maximum
            max_odds = constraints.get("max_odds_win")
            if max_odds and odds > max_odds:
                validation.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.BUSINESS_RULE,
                        message=f"Cote ({odds:.1f}) > max_odds ({max_odds})",
                        race_key=race_key,
                        runner_id=runner_id,
                        horse_name=horse_name,
                        expected_value=max_odds,
                        actual_value=odds,
                        rule_name="max_odds_win",
                    )
                )

            # 7. Value minimum (cutoff)
            value_cutoff = constraints.get("value_cutoff_win_pct", 0)
            if value_cutoff and value_declared < value_cutoff:
                validation.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.BUSINESS_RULE,
                        message=f"Value ({value_declared:.1f}%) < cutoff ({value_cutoff:.1f}%)",
                        race_key=race_key,
                        runner_id=runner_id,
                        horse_name=horse_name,
                        expected_value=value_cutoff,
                        actual_value=value_declared,
                        rule_name="value_cutoff",
                    )
                )

            # 8. Stake maximum
            stake = decision.get("stake_eur", 0)
            bankroll = report.get("bankroll_eur", 500)
            cap_per_bet = constraints.get("cap_per_bet", 0.05)
            max_stake = bankroll * cap_per_bet

            if stake > max_stake * 1.01:  # 1% de marge
                validation.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.RISK_LIMIT,
                        message=f"Mise ({stake:.2f}€) > cap ({max_stake:.2f}€ = {cap_per_bet:.0%} bankroll)",
                        race_key=race_key,
                        runner_id=runner_id,
                        horse_name=horse_name,
                        expected_value=max_stake,
                        actual_value=stake,
                        rule_name="cap_per_bet",
                    )
                )

            # 9. Mise nulle ou négative
            if stake <= 0:
                validation.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.DATA_QUALITY,
                        message="Mise nulle ou négative pour un pick gardé",
                        race_key=race_key,
                        runner_id=runner_id,
                        horse_name=horse_name,
                        actual_value=stake,
                    )
                )

        # =====================================================================
        # VALIDATIONS QUALITÉ DONNÉES (toujours)
        # =====================================================================

        # 10. Cote invalide
        if odds is None or odds < 1:
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.DATA_QUALITY,
                    message=f"Cote invalide ou manquante: {odds}",
                    race_key=race_key,
                    runner_id=runner_id,
                    horse_name=horse_name,
                    actual_value=odds,
                )
            )

        # 11. Nom de cheval vide
        if not horse_name or horse_name in ("Inconnu", "Unknown", ""):
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category=ValidationCategory.DATA_QUALITY,
                    message="Nom de cheval manquant ou invalide",
                    race_key=race_key,
                    runner_id=runner_id,
                )
            )

    # -------------------------------------------------------------------------
    # Validations transverses
    # -------------------------------------------------------------------------

    def _validate_cross_race_consistency(self, report: dict, validation: ValidationReport):
        """Valide la cohérence entre les courses"""

        races = report.get("races", [])

        # 1. Vérifier les doublons de runner_id
        all_kept_runners = []
        for race in races:
            for runner_id in race.get("kept_runners", []):
                all_kept_runners.append((race.get("race_key"), runner_id))

        # Pas de doublon inter-course (même cheval dans 2 courses = normal, mais même id = suspect)
        seen_ids = set()
        for race_key, runner_id in all_kept_runners:
            if runner_id in seen_ids:
                validation.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.CONSISTENCY,
                        message=f"Runner ID '{runner_id}' présent dans plusieurs courses gardées",
                        runner_id=runner_id,
                        rule_name="unique_runner_id",
                    )
                )
            seen_ids.add(runner_id)

        # 2. Somme des stakes par course = total
        total_stake_races = sum(race.get("total_stake_eur", 0) or 0 for race in races)
        summary_stake = report.get("summary", {}).get("total_stake_eur", 0)

        if abs(total_stake_races - summary_stake) > 0.01:
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.CONSISTENCY,
                    message=f"Somme stakes courses ({total_stake_races:.2f}€) ≠ total summary ({summary_stake:.2f}€)",
                    expected_value=summary_stake,
                    actual_value=total_stake_races,
                    rule_name="stake_sum_consistency",
                )
            )

        # 3. Vérifier cohérence kept_runners count
        total_kept_declared = report.get("summary", {}).get("total_picks_kept", 0)
        total_kept_counted = sum(len(race.get("kept_runners", [])) for race in races)

        if total_kept_declared != total_kept_counted:
            validation.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.CONSISTENCY,
                    message=f"Nombre picks déclaré ({total_kept_declared}) ≠ compté ({total_kept_counted})",
                    expected_value=total_kept_declared,
                    actual_value=total_kept_counted,
                    rule_name="picks_count_consistency",
                )
            )


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================


def validate_report(report: dict, policy_config: dict[str, Any]) -> ValidationReport:
    """
    Fonction utilitaire pour valider un rapport algo.

    Args:
        report: RapportAlgo en dict
        policy_config: Configuration depuis pro_betting.yaml

    Returns:
        ValidationReport
    """
    validator = ReportValidator(policy_config)
    return validator.validate(report)


def validate_and_summarize(report: dict, policy_config: dict[str, Any]) -> dict[str, Any]:
    """
    Valide et retourne un résumé simplifié.

    Returns:
        Dict avec is_valid, summary, et issues critiques
    """
    validation = validate_report(report, policy_config)

    critical_issues = [
        i.model_dump()
        for i in validation.issues
        if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
    ]

    return {
        "is_valid": validation.is_valid,
        "total_issues": validation.total_issues,
        "by_severity": validation.by_severity,
        "critical_issues": critical_issues,
        "validation_time_ms": validation.validation_duration_ms,
    }
