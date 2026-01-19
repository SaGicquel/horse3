"""
🎯 Générateur de Rapport Algo (Agent IA - Étape A)
===================================================

Service qui génère un RapportAlgo JSON complet à partir du pipeline existant.
Ce rapport est ensuite analysé par l'Agent IA.

OBSERVABILITÉ (loggé):
- Durée génération
- Taille JSON
- Nombre races/runners/bets

Auteur: Agent IA Pipeline
Date: 2024-12-21
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from typing import Any, Optional
from uuid import uuid4

# Import des modèles
from services.algo_report_models import (
    RapportAlgo,
    RaceAnalysis,
    RunnerAnalysis,
    AlgoDecision,
    PolicyConstraints,
    AlgoMetrics,
    DecisionStatus,
    DriftStatus,
    GenerateReportRequest,
    GenerateReportResponse,
    ReplayInputs,
    SCHEMA_VERSION,
    POLICY_VERSION,
    DEFAULT_MODEL_VERSION,
)

# Import du module betting_policy existant
from services.betting_policy import select_portfolio_from_picks

# Logger pour observabilité
logger = logging.getLogger("agent_ia.report_generator")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_model_metrics() -> dict[str, Any]:
    """
    Récupère les métriques du modèle depuis les fichiers de calibration.

    Returns:
        Dict avec version, calibrateur, température, etc.
    """
    import json
    from pathlib import Path

    # Chemins possibles pour les métriques
    possible_paths = [
        Path(__file__).parent.parent.parent.parent
        / "calibration"
        / "champion"
        / "calibration_report.json",
        Path(__file__).parent.parent.parent.parent / "calibration" / "calibration_report.json",
        Path("/app/calibration/champion/calibration_report.json"),
    ]

    metrics = {
        "model_version": "xgb_proba_v9",
        "calibrator_type": "platt",
        "temperature": 1.254,
        "blend_alpha": 0.2,
        "drift_status": "OK",
        "last_retrain_date": None,
    }

    for path in possible_paths:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    stats = data.get("stats", {})
                    metrics["temperature"] = stats.get("temperature", metrics["temperature"])
                    metrics["calibrator_type"] = stats.get(
                        "calibrator_type", metrics["calibrator_type"]
                    )
                    metrics["blend_alpha"] = stats.get("alpha_blend", metrics["blend_alpha"])
                break
            except Exception:
                pass

    return metrics


def build_runner_analysis(
    pick: dict[str, Any], is_kept: bool, exclude_reason: Optional[str] = None
) -> RunnerAnalysis:
    """
    Construit l'analyse d'un partant à partir d'un pick.

    Args:
        pick: Données du pick
        is_kept: True si le pari est gardé
        exclude_reason: Raison d'exclusion si rejeté

    Returns:
        RunnerAnalysis structurée
    """
    # Extraire les données de base
    runner_id = pick.get("cheval_id") or pick.get("pmu_id") or str(pick.get("numero", "unknown"))
    horse_name = pick.get("cheval") or pick.get("horse") or pick.get("nom") or "Inconnu"

    # Probabilités
    p_win = pick.get("p_win") or pick.get("p_model_win") or 0.0
    p_place = pick.get("p_place") or pick.get("p_model_place")

    # Cotes
    odds_final = pick.get("cote") or pick.get("odds") or pick.get("cote_finale")
    odds_morning = pick.get("cote_matin")

    # Value et Kelly
    value_win = pick.get("value") or pick.get("value_win") or 0.0
    value_place = pick.get("value_place") or 0.0
    kelly_win = pick.get("kelly") or pick.get("kelly_win") or 0.0
    kelly_place = pick.get("kelly_place") or 0.0

    # Signaux
    signals_pos = pick.get("signaux_positifs") or pick.get("signals_pos") or []
    signals_neg = pick.get("signaux_negatifs") or pick.get("signals_neg") or []
    if isinstance(signals_pos, str):
        signals_pos = [s.strip() for s in signals_pos.split(",") if s.strip()]
    if isinstance(signals_neg, str):
        signals_neg = [s.strip() for s in signals_neg.split(",") if s.strip()]

    # Décision
    if is_kept:
        why_kept = []
        stake = pick.get("stake") or pick.get("stake_user") or 0.0
        ev = pick.get("ev") or 0.0
        kelly_adj = pick.get("kelly_adjusted") or kelly_win

        # Construire les raisons de garder
        if value_win > 0:
            why_kept.append(f"value={value_win:.1f}% > seuil")
        if kelly_win > 0:
            why_kept.append(f"kelly={kelly_win:.1f}% > 0")
        if odds_final and odds_final < 20:
            why_kept.append(f"cote={odds_final:.1f} < max")

        decision = AlgoDecision(
            status=DecisionStatus.KEPT,
            bet_type=pick.get("bet_type"),
            stake_eur=stake,
            kelly_raw_pct=kelly_win,
            kelly_adjusted_pct=kelly_adj,
            ev_eur=ev,
            why_kept=why_kept,
            failed_rules=[],
            exclude_reason=None,
        )
    else:
        decision = AlgoDecision(
            status=DecisionStatus.REJECTED,
            bet_type=pick.get("bet_type"),
            stake_eur=None,
            kelly_raw_pct=kelly_win,
            kelly_adjusted_pct=None,
            ev_eur=None,
            why_kept=[],
            failed_rules=[exclude_reason] if exclude_reason else [],
            exclude_reason=exclude_reason,
        )

    return RunnerAnalysis(
        runner_id=str(runner_id),
        horse_name=str(horse_name),
        numero=pick.get("numero"),
        p_model_win=float(p_win) if p_win else 0.0,
        p_model_place=float(p_place) if p_place else None,
        odds_morning=float(odds_morning) if odds_morning else None,
        odds_final=float(odds_final) if odds_final else None,
        odds_implied_prob=1.0 / float(odds_final) if odds_final and odds_final > 1 else None,
        value_win_pct=float(value_win) if value_win else None,
        value_place_pct=float(value_place) if value_place else None,
        kelly_win_pct=float(kelly_win) if kelly_win else None,
        kelly_place_pct=float(kelly_place) if kelly_place else None,
        signals_positive=signals_pos,
        signals_negative=signals_neg,
        bet_risk=pick.get("bet_risk") or pick.get("risk_level"),
        algo_decision=decision,
        jockey=pick.get("driver") or pick.get("jockey"),
        trainer=pick.get("entraineur") or pick.get("trainer"),
        musique=pick.get("musique"),
        nb_courses_total=pick.get("nb_courses"),
        nb_victoires_total=pick.get("nb_victoires"),
    )


def build_race_analysis(
    race_key: str,
    kept_picks: list[dict],
    rejected_picks: list[dict],
    race_info: Optional[dict] = None,
) -> RaceAnalysis:
    """
    Construit l'analyse d'une course.

    Args:
        race_key: Clé de la course
        kept_picks: Liste des picks gardés pour cette course
        rejected_picks: Liste des picks rejetés pour cette course
        race_info: Infos supplémentaires sur la course (optionnel)

    Returns:
        RaceAnalysis structurée
    """
    race_info = race_info or {}

    # Parser le race_key (format: DATE|Rn|Cn|HIPPODROME)
    parts = race_key.split("|")
    race_id = f"{parts[1] if len(parts) > 1 else 'R?'}{parts[2] if len(parts) > 2 else 'C?'}"
    hippodrome = parts[3] if len(parts) > 3 else "Inconnu"

    # Construire les analyses des partants
    runners = []
    kept_ids = []
    rejected_ids = []

    for pick in kept_picks:
        runner = build_runner_analysis(pick, is_kept=True)
        runners.append(runner)
        kept_ids.append(runner.runner_id)

    for pick in rejected_picks:
        exclude_reason = pick.get("excludeReason") or pick.get("exclude_reason")
        runner = build_runner_analysis(pick, is_kept=False, exclude_reason=exclude_reason)
        runners.append(runner)
        rejected_ids.append(runner.runner_id)

    # Calculs agrégés
    total_stake = sum(r.algo_decision.stake_eur or 0 for r in runners)
    total_ev = sum(r.algo_decision.ev_eur or 0 for r in runners)

    return RaceAnalysis(
        race_id=race_id,
        race_key=race_key,
        hippodrome=hippodrome,
        discipline=race_info.get("discipline") or "inconnu",
        distance_m=race_info.get("distance_m"),
        start_time=race_info.get("heure_depart"),
        race_name=race_info.get("course_nom"),
        allocation_eur=race_info.get("allocation_totale"),
        terrain_state=race_info.get("etat_piste"),
        weather=race_info.get("meteo"),
        nb_partants=race_info.get("nombre_partants") or len(runners),
        runners=runners,
        kept_runners=kept_ids,
        rejected_runners=rejected_ids,
        total_stake_eur=total_stake,
        total_ev_eur=total_ev,
    )


def generate_rapport_algo(
    picks: list[dict[str, Any]],
    bankroll: float,
    profile: str,
    target_date: date,
    policy_config: dict[str, Any],
    user_id: Optional[int] = None,
) -> RapportAlgo:
    """
    Génère un rapport algo complet à partir des picks et de la politique.

    Args:
        picks: Liste des picks bruts (tous partants avec p/value/kelly)
        bankroll: Bankroll utilisateur en euros
        profile: Profil de risque (PRUDENT/STANDARD/AGRESSIF)
        target_date: Date des courses
        policy_config: Configuration de la politique depuis pro_betting.yaml
        user_id: ID utilisateur optionnel

    Returns:
        RapportAlgo complet
    """
    # Extraire la configuration centrale
    betting_defaults = policy_config.get("betting_defaults", {})
    betting_policy = policy_config.get("betting_policy", {})

    kelly_fraction_map = betting_defaults.get("kelly_fraction_map", {})
    kelly_fraction = kelly_fraction_map.get(profile, kelly_fraction_map.get("STANDARD", 0.33))

    # Appeler le pipeline de sélection existant
    portfolio = select_portfolio_from_picks(
        picks=picks,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        cap_per_bet=betting_defaults.get("cap_per_bet", 0.05),
        daily_budget_rate=betting_defaults.get("daily_budget_rate", 0.12),
        rounding=betting_defaults.get("rounding_increment_eur", 1.0),
        policy=betting_policy,
        profile=profile,
    )

    kept_picks = portfolio.get("positions", [])
    excluded_picks = portfolio.get("excluded", [])
    policy_info = portfolio.get("policy", {})
    caps = portfolio.get("caps", {})

    # Grouper par course
    races_kept: dict[str, list] = {}
    races_rejected: dict[str, list] = {}

    for pick in kept_picks:
        rk = pick.get("race_key") or "unknown"
        races_kept.setdefault(rk, []).append(pick)

    for pick in excluded_picks:
        rk = pick.get("race_key") or pick.get("_race_key") or "unknown"
        races_rejected.setdefault(rk, []).append(pick)

    # Construire les analyses de courses
    all_race_keys = set(races_kept.keys()) | set(races_rejected.keys())
    races = []

    for rk in sorted(all_race_keys):
        race = build_race_analysis(
            race_key=rk,
            kept_picks=races_kept.get(rk, []),
            rejected_picks=races_rejected.get(rk, []),
        )
        races.append(race)

    # Métriques modèle
    model_metrics = get_model_metrics()

    # Construire les contraintes
    zone_configs = betting_policy.get("bankroll_zones", {})
    zone = policy_info.get("zone", "full")
    zone_conf = zone_configs.get(zone, {})

    constraints = PolicyConstraints(
        zone=zone,
        profile=profile,
        max_bets_per_day=policy_info.get("max_bets_per_day", 6),
        max_bets_per_race=policy_info.get("max_bets_per_race", 2),
        min_proba_model=zone_conf.get("min_proba_model"),
        value_cutoff_win_pct=zone_conf.get("value_cutoff_win"),
        value_cutoff_place_pct=zone_conf.get("value_cutoff_place"),
        max_odds_win=policy_info.get("max_odds_win"),
        kelly_fraction=kelly_fraction,
        cap_per_bet=betting_defaults.get("cap_per_bet", 0.05),
        daily_budget_rate=betting_defaults.get("daily_budget_rate", 0.12),
        allowed_risks=policy_info.get("allowed_risks", []),
        allowed_bet_types=policy_info.get("allowed_bet_types"),
        max_drawdown_stop=zone_conf.get("max_drawdown_stop"),
    )

    # Résumé global
    total_runners = sum(len(r.runners) for r in races)
    summary = {
        "total_races_analyzed": len(races),
        "total_runners_analyzed": total_runners,
        "total_picks_kept": sum(len(r.kept_runners) for r in races),
        "total_picks_rejected": sum(len(r.rejected_runners) for r in races),
        "total_stake_eur": portfolio.get("total_stake", 0),
        "total_ev_eur": portfolio.get("total_ev", 0),
        "budget_left_eur": portfolio.get("budget_left", 0),
    }

    # Créer les inputs pour replay
    replay_inputs = ReplayInputs(
        bankroll=bankroll,
        profile=profile,
        target_date=str(target_date),
        policy_version=POLICY_VERSION,
        model_version=model_metrics.get("model_version", DEFAULT_MODEL_VERSION),
        seed=None,  # Pas de stochastique pour l'instant
    )
    inputs_hash = replay_inputs.compute_hash()

    report = RapportAlgo(
        run_id=uuid4(),
        schema_version=SCHEMA_VERSION,
        policy_version=POLICY_VERSION,
        generated_at=datetime.utcnow(),
        replay_inputs=replay_inputs,
        inputs_hash=inputs_hash,
        target_date=target_date,
        user_id=user_id,
        bankroll_eur=bankroll,
        profile=profile,
        policy_constraints=constraints,
        algo_metrics=AlgoMetrics(
            model_version=model_metrics.get("model_version", DEFAULT_MODEL_VERSION),
            calibrator_type=model_metrics.get("calibrator_type"),
            temperature=model_metrics.get("temperature"),
            blend_alpha=model_metrics.get("blend_alpha"),
            drift_status=DriftStatus(model_metrics.get("drift_status", "OK")),
            last_retrain_date=model_metrics.get("last_retrain_date"),
        ),
        races=races,
        summary=summary,
        caps=caps,
    )

    # Observabilité: calculer taille JSON et logger
    try:
        json_size = len(json.dumps(report.model_dump(), default=str))
        logger.info(
            f"📊 RapportAlgo généré | "
            f"races={len(races)} runners={total_runners} "
            f"kept={summary['total_picks_kept']} rejected={summary['total_picks_rejected']} "
            f"stake={summary['total_stake_eur']:.2f}€ "
            f"json_size={json_size/1024:.1f}KB "
            f"hash={inputs_hash}"
        )
    except Exception as e:
        logger.warning(f"Erreur logging observabilité: {e}")

    return report


async def generate_algo_report_endpoint(
    request: GenerateReportRequest,
    picks: list[dict[str, Any]],
    policy_config: dict[str, Any],
    user_id: Optional[int] = None,
    user_bankroll: float = 500.0,
    user_profile: str = "STANDARD",
) -> GenerateReportResponse:
    """
    Point d'entrée API pour générer un rapport algo.

    Args:
        request: Paramètres de la requête
        picks: Picks du jour (depuis /picks/today ou équivalent)
        policy_config: Configuration depuis pro_betting.yaml
        user_id: ID utilisateur si authentifié
        user_bankroll: Bankroll par défaut ou depuis settings
        user_profile: Profil par défaut ou depuis settings

    Returns:
        GenerateReportResponse avec le rapport ou erreur
    """
    start_time = time.time()

    try:
        # Résoudre les paramètres
        target_date = request.target_date or date.today()
        bankroll = request.bankroll or user_bankroll
        profile = request.profile or user_profile

        # Générer le rapport
        report = generate_rapport_algo(
            picks=picks,
            bankroll=bankroll,
            profile=profile,
            target_date=target_date,
            policy_config=policy_config,
            user_id=user_id,
        )

        generation_time = (time.time() - start_time) * 1000

        return GenerateReportResponse(
            success=True,
            run_id=report.run_id,
            message=f"Rapport généré avec succès: {report.summary.get('total_picks_kept', 0)} picks gardés",
            report=report,
            generation_time_ms=round(generation_time, 2),
        )

    except Exception as e:
        return GenerateReportResponse(
            success=False,
            run_id=uuid4(),
            message=f"Erreur lors de la génération: {str(e)}",
            report=None,
            generation_time_ms=(time.time() - start_time) * 1000,
        )
