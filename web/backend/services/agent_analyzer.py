"""
🤖 Agent Analyzer Service - Orchestration Pipeline B/C/D
========================================================

Service principal qui orchestre le pipeline complet de l'Agent IA:
- Step A: Génération RapportAlgo (déjà fait par algo_report_generator)
- Step B: Analyse IA (LLM)
- Step C: Vérification des claims (LLM + DB queries)
- Step D: Auto-critique + proposition finale (LLM)

Auteur: Agent IA Pipeline
Date: 2024-12-21
"""

from __future__ import annotations

import json
import logging
import time
import os
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

# Imports LLM
from services.llm import (
    LLMProvider,
    LLMProviderType,
    LLMConfig,
    LLMBudgetConfig,
    LLMTrace,
    get_provider,
    format_prompt,
    StepBOutput,
    StepCOutput,
    StepDOutput,
)

# Imports persistence
from services.agent_persistence import (
    AgentPersistenceService,
    AgentRunCreate,
    AgentRunUpdate,
    AgentStepCreate,
    AgentStepUpdate,
    RunStatus,
    StepName,
    StepStatus,
    get_persistence_service,
)

# Imports report
from services.algo_report_models import RapportAlgo


# Logger
logger = logging.getLogger("agent_ia.analyzer")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# CONFIGURATION
# =============================================================================


class AgentConfig(BaseModel):
    """Configuration de l'agent"""

    # Provider
    provider: LLMProviderType = Field(
        default=LLMProviderType.GEMINI, description="Provider LLM à utiliser"
    )
    model: str = Field(default="gemini-2.0-flash-exp", description="Modèle à utiliser")

    # LLM settings
    temperature: float = Field(default=0.0, description="Température (0 = déterministe)")
    max_tokens: int = Field(
        default=16384, description="Max tokens par appel - augmenté pour réponses JSON complètes"
    )

    # Budget
    max_calls_per_run: int = Field(default=10, description="Max appels LLM par run")
    max_tokens_per_run: int = Field(default=100000, description="Max tokens par run")
    max_cost_per_run_usd: float = Field(default=1.0, description="Max coût par run en USD")

    # Behavior
    skip_verification: bool = Field(default=False, description="Skip Step C (verification)")

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Charge la config depuis les variables d'environnement"""
        provider_str = os.getenv("AGENT_DEFAULT_PROVIDER", "gemini")
        provider = LLMProviderType.GEMINI if provider_str == "gemini" else LLMProviderType.OPENAI

        return cls(
            provider=provider,
            model=os.getenv("AGENT_DEFAULT_MODEL", "gemini-2.0-flash-exp"),
            max_calls_per_run=int(os.getenv("AGENT_MAX_LLM_CALLS_PER_RUN", "10")),
            max_tokens_per_run=int(os.getenv("AGENT_MAX_TOKENS_PER_RUN", "100000")),
            max_cost_per_run_usd=float(os.getenv("AGENT_MAX_COST_PER_RUN_USD", "1.0")),
        )


# =============================================================================
# DB ENRICHMENT - Données historiques pour l'analyse
# =============================================================================


def _enrich_picks_with_db(picks: list[dict]) -> dict:
    """
    Enrichit les picks avec des données historiques de la BDD.

    Pour chaque cheval, récupère:
    - Historique récent (5 dernières courses)
    - Stats jockey
    - Stats entraîneur
    - Performance sur l'hippodrome

    Returns:
        Dict avec les données d'enrichissement par cheval
    """
    try:
        from main import get_db_connection, adapt_query
    except ImportError:
        logger.warning("DB non disponible pour enrichissement")
        return {}

    enrichment = {}

    try:
        con = get_db_connection()
        cur = con.cursor()

        for pick in picks:
            horse_name = pick.get("nom") or pick.get("selection") or pick.get("horse_name", "")
            if not horse_name:
                continue

            horse_key = horse_name.lower().replace("'", "_")

            # 1. Historique récent du cheval (5 dernières courses)
            cur.execute(
                adapt_query("""
                SELECT
                    race_key, hippodrome_nom, distance_m,
                    place_finale, is_win, cote_finale
                FROM cheval_courses_seen
                WHERE LOWER(nom_norm) LIKE %s
                ORDER BY race_key DESC
                LIMIT 5
            """),
                (f"%{horse_key}%",),
            )

            recent_races = []
            for row in cur.fetchall():
                recent_races.append(
                    {
                        "date": row[0].split("|")[0] if row[0] else None,
                        "hippodrome": row[1],
                        "distance": row[2],
                        "place": row[3],
                        "win": row[4],
                        "odds": row[5],
                    }
                )

            # 2. Stats globales du cheval
            cur.execute(
                adapt_query("""
                SELECT
                    COUNT(*) as nb_courses,
                    SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                    SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) as places,
                    AVG(place_finale) as place_moyenne
                FROM cheval_courses_seen
                WHERE LOWER(nom_norm) LIKE %s
            """),
                (f"%{horse_key}%",),
            )

            stats_row = cur.fetchone()
            horse_stats = {
                "total_courses": stats_row[0] or 0,
                "wins": stats_row[1] or 0,
                "places_top3": stats_row[2] or 0,
                "avg_place": round(stats_row[3], 1) if stats_row[3] else None,
                "win_rate": round((stats_row[1] or 0) / stats_row[0] * 100, 1)
                if stats_row[0]
                else 0,
            }

            # 3. Stats jockey (si disponible dans le pick)
            jockey_name = pick.get("jockey") or pick.get("driver_jockey", "")
            jockey_stats = None
            if jockey_name:
                cur.execute(
                    adapt_query("""
                    SELECT
                        COUNT(*) as courses,
                        SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
                    FROM cheval_courses_seen
                    WHERE driver_jockey ILIKE %s
                """),
                    (f"%{jockey_name}%",),
                )
                jrow = cur.fetchone()
                if jrow and jrow[0] > 0:
                    jockey_stats = {
                        "courses": jrow[0],
                        "wins": jrow[1] or 0,
                        "win_rate": round((jrow[1] or 0) / jrow[0] * 100, 1),
                    }

            # 4. Stats entraîneur (si disponible)
            trainer_name = pick.get("entraineur") or pick.get("trainer", "")
            trainer_stats = None
            if trainer_name:
                cur.execute(
                    adapt_query("""
                    SELECT
                        COUNT(*) as courses,
                        SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
                    FROM cheval_courses_seen
                    WHERE entraineur ILIKE %s
                """),
                    (f"%{trainer_name}%",),
                )
                trow = cur.fetchone()
                if trow and trow[0] > 0:
                    trainer_stats = {
                        "courses": trow[0],
                        "wins": trow[1] or 0,
                        "win_rate": round((trow[1] or 0) / trow[0] * 100, 1),
                    }

            # Compiler l'enrichissement pour ce cheval
            enrichment[horse_name] = {
                "recent_form": recent_races,
                "stats": horse_stats,
                "jockey": jockey_stats,
                "trainer": trainer_stats,
                "form_indicator": _calculate_form(recent_races),
            }

        con.close()

    except Exception as e:
        logger.warning(f"Erreur enrichissement DB: {e}")

    return enrichment


def _calculate_form(recent_races: list[dict]) -> str:
    """Calcule un indicateur de forme basé sur les 5 dernières courses"""
    if not recent_races:
        return "UNKNOWN"

    wins = sum(1 for r in recent_races if r.get("win"))
    top3 = sum(1 for r in recent_races if r.get("place") and r["place"] <= 3)

    if wins >= 2:
        return "EXCELLENT"
    elif wins >= 1 or top3 >= 3:
        return "GOOD"
    elif top3 >= 1:
        return "AVERAGE"
    else:
        return "POOR"


# =============================================================================
# AGENT ANALYZER SERVICE
# =============================================================================


class AgentAnalyzerService:
    """
    Service d'analyse IA pour les rapports algo.

    Orchestration du pipeline:
    1. Reçoit un RapportAlgo (Step A déjà fait)
    2. Step B: Analyse IA du rapport
    3. Step C: Vérification des affirmations
    4. Step D: Auto-critique et proposition finale

    Usage:
        analyzer = AgentAnalyzerService()
        result = await analyzer.run_analysis(rapport_algo)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        persistence: Optional[AgentPersistenceService] = None,
    ):
        self.config = config or AgentConfig.from_env()
        self.persistence = persistence
        self._provider: Optional[LLMProvider] = None
        self._budget: Optional[LLMBudgetConfig] = None

    @property
    def provider(self) -> LLMProvider:
        """Lazy loading du provider LLM"""
        if self._provider is None:
            api_key = None
            if self.config.provider == LLMProviderType.OPENAI:
                api_key = os.getenv("OPENAI_API_KEY")
            else:
                api_key = os.getenv("GEMINI_API_KEY")

            self._provider = get_provider(self.config.provider, api_key)

            # Configurer le budget
            self._budget = LLMBudgetConfig(
                max_llm_calls_per_run=self.config.max_calls_per_run,
                max_tokens_per_run=self.config.max_tokens_per_run,
                max_cost_per_run_usd=self.config.max_cost_per_run_usd,
            )
            self._provider.set_budget(self._budget)

        return self._provider

    async def run_analysis(
        self,
        rapport_algo: RapportAlgo,
        user_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Exécute l'analyse complète du rapport algo.

        Args:
            rapport_algo: Rapport algo généré à l'étape A
            user_id: ID utilisateur optionnel

        Returns:
            Dict avec les résultats de chaque étape et le rapport final
        """
        start_time = time.time()
        run_id = rapport_algo.run_id

        logger.info(f"🚀 Démarrage analyse Agent IA | run_id={run_id}")

        # Créer le run en DB si persistence disponible
        if self.persistence:
            try:
                self.persistence.create_run(
                    AgentRunCreate(
                        run_id=run_id,  # Utilise le run_id du rapport
                        target_date=rapport_algo.target_date,
                        user_id=user_id,
                        bankroll=rapport_algo.bankroll_eur,
                        profile=rapport_algo.profile,
                        model_version=rapport_algo.algo_metrics.model_version,
                    )
                )
            except Exception as e:
                logger.warning(f"Erreur création run DB: {e}")

        result = {
            "run_id": str(run_id),
            "started_at": datetime.utcnow().isoformat(),
            "config": self.config.model_dump(),
            "steps": {},
            "traces": [],
            "success": False,
            "error": None,
        }

        try:
            # =========================================================
            # STEP B: ANALYSE IA
            # =========================================================
            step_b_result, step_b_trace = await self._run_step_b(rapport_algo, run_id)
            result["steps"]["B"] = step_b_result.model_dump() if step_b_result else None
            result["traces"].append(step_b_trace.model_dump() if step_b_trace else None)

            if not step_b_result:
                raise RuntimeError("Step B failed")

            # =========================================================
            # STEP C: VÉRIFICATION (optionnel)
            # =========================================================
            step_c_result = None
            if not self.config.skip_verification:
                step_c_result, step_c_trace = await self._run_step_c(
                    rapport_algo, step_b_result, run_id
                )
                result["steps"]["C"] = step_c_result.model_dump() if step_c_result else None
                result["traces"].append(step_c_trace.model_dump() if step_c_trace else None)
            else:
                logger.info("⏭️ Step C skipped (config.skip_verification=True)")
                result["steps"]["C"] = {"skipped": True}

            # =========================================================
            # STEP D: AUTO-CRITIQUE + FINAL
            # =========================================================
            step_d_result, step_d_trace = await self._run_step_d(
                rapport_algo, step_b_result, step_c_result, run_id
            )
            result["steps"]["D"] = step_d_result.model_dump() if step_d_result else None
            result["traces"].append(step_d_trace.model_dump() if step_d_trace else None)

            if not step_d_result:
                raise RuntimeError("Step D failed")

            # FINALISATION
            # =========================================================
            result["success"] = True

            # VALIDATION: Créer un mapping horse_name -> (race_key, hippodrome) pour corriger les erreurs
            valid_picks = []
            horse_to_race = {}  # {horse_name_lower: (race_key, hippodrome)}

            # Construire le mapping depuis le rapport
            for race in rapport_algo.races:
                for runner in race.runners:
                    h_name = runner.horse_name.lower().strip()
                    horse_to_race[h_name] = (race.race_key, race.hippodrome)

            logger.info(
                f"🔍 Mapping construit: {len(horse_to_race)} chevaux. Exemples: {list(horse_to_race.keys())[:5]}"
            )

            for pick in step_d_result.final_picks:
                pick_dict = pick.model_dump() if hasattr(pick, "model_dump") else pick
                horse_name = str(pick_dict.get("horse_name", "")).lower().strip()
                pick_race_key = pick_dict.get("race_key", "")

                logger.info(f"🔍 Traitement pick: '{horse_name}' | race_key={pick_race_key}")

                # Chercher le cheval dans le mapping (exact ou partiel)
                correct_race_key = None
                correct_hippodrome = None

                # Match exact
                if horse_name in horse_to_race:
                    correct_race_key, correct_hippodrome = horse_to_race[horse_name]
                else:
                    # Match partiel (contient)
                    for h_name, (rk, hippo) in horse_to_race.items():
                        if horse_name in h_name or h_name in horse_name:
                            correct_race_key, correct_hippodrome = rk, hippo
                            break

                if correct_race_key:
                    # Corriger la race_key si nécessaire
                    if pick_race_key != correct_race_key:
                        logger.warning(
                            f"⚠️ Race corrigée: {horse_name} de {pick_race_key} → {correct_race_key}"
                        )
                        pick_dict["race_key"] = correct_race_key
                        pick_dict["hippodrome"] = correct_hippodrome
                    valid_picks.append(pick_dict)
                else:
                    logger.warning(f"⚠️ Pick halluciné rejeté: {horse_name} (pas dans le rapport)")

            result["final_picks"] = valid_picks
            result["portfolio_confidence"] = step_d_result.portfolio_confidence
            result["executive_summary"] = step_d_result.executive_summary

            # Update run en DB
            if self.persistence:
                try:
                    self.persistence.update_run(
                        run_id,
                        AgentRunUpdate(
                            status=RunStatus.SUCCESS,
                            final_report=result["steps"]["D"],
                            confidence_score=step_d_result.portfolio_confidence,
                            total_picks_final=step_d_result.total_picks,
                            total_stake_final=step_d_result.total_stake_eur,
                            finished_at=datetime.utcnow(),
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Erreur update run DB: {e}")

        except Exception as e:
            logger.error(f"❌ Erreur analyse: {e}")
            result["error"] = str(e)

            # Update run en DB avec erreur
            if self.persistence:
                try:
                    self.persistence.update_run(
                        run_id,
                        AgentRunUpdate(
                            status=RunStatus.FAILED,
                            error_message=str(e),
                            finished_at=datetime.utcnow(),
                        ),
                    )
                except Exception as db_e:
                    logger.warning(f"Erreur update run DB: {db_e}")

        # Stats finales
        result["finished_at"] = datetime.utcnow().isoformat()
        result["duration_ms"] = int((time.time() - start_time) * 1000)

        if self._budget:
            result["budget_used"] = {
                "calls": self._budget.current_calls,
                "tokens": self._budget.current_tokens,
                "cost_usd": round(self._budget.current_cost_usd, 6),
            }

        logger.info(
            f"{'✅' if result['success'] else '❌'} Analyse terminée | "
            f"run_id={run_id} | duration={result['duration_ms']}ms | "
            f"success={result['success']}"
        )

        return result

    # -------------------------------------------------------------------------
    # STEP B: ANALYSE IA
    # -------------------------------------------------------------------------

    async def _run_step_b(
        self,
        rapport: RapportAlgo,
        run_id: UUID,
    ) -> tuple[Optional[StepBOutput], Optional[LLMTrace]]:
        """Exécute l'étape B - Analyse IA du rapport"""
        logger.info("📊 Step B: Analyse IA du rapport...")

        # Créer le step en DB
        step_id = None
        if self.persistence:
            try:
                step_id = self.persistence.create_step(
                    AgentStepCreate(
                        run_id=run_id,
                        step_name=StepName.B,
                        input_json={"rapport_summary": rapport.summary},
                    )
                )
            except Exception as e:
                logger.warning(f"Erreur création step DB: {e}")

        try:
            # Enrichir les picks avec les données historiques de la BDD
            all_picks = []
            for race in rapport.races:
                for runner in race.runners:
                    all_picks.append(
                        {
                            "nom": runner.horse_name,
                            "selection": runner.horse_name,
                            "jockey": getattr(runner, "jockey", None),
                            "entraineur": getattr(runner, "trainer", None),
                        }
                    )

            enrichment_data = _enrich_picks_with_db(all_picks)
            logger.info(
                f"📊 Enrichissement: {len(enrichment_data)} chevaux avec données historiques"
            )

            # Préparer le prompt avec les données enrichies
            prompt, meta = format_prompt(
                "B",
                target_date=str(rapport.target_date),
                profile=rapport.profile,
                bankroll=rapport.bankroll_eur,
                algo_report_json=json.dumps(rapport.model_dump(), default=str, indent=2)[
                    :18000
                ],  # Équilibré pour Gemini
                horse_enrichment_data=json.dumps(enrichment_data, default=str, indent=2)[:10000]
                if enrichment_data
                else "Aucune donnée historique disponible",
            )

            # Appel LLM
            cfg = LLMConfig(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                step_name="B",
                run_id=str(run_id),
            )

            result, trace = self.provider.generate_structured(
                model=self.config.model,
                prompt=prompt,
                schema=StepBOutput,
                cfg=cfg,
            )

            # Update step en DB
            if self.persistence and step_id:
                try:
                    self.persistence.update_step(
                        step_id,
                        AgentStepUpdate(
                            status=StepStatus.SUCCESS,
                            output_json=result.model_dump(),
                            llm_model=trace.model,
                            llm_prompt_tokens=trace.tokens_in,
                            llm_completion_tokens=trace.tokens_out,
                            llm_cost_usd=trace.cost_estimate_usd,
                            duration_ms=trace.latency_ms,
                            finished_at=datetime.utcnow(),
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Erreur update step DB: {e}")

            logger.info(f"✅ Step B terminé | picks analysés={result.total_reviewed}")
            return result, trace

        except Exception as e:
            logger.error(f"❌ Step B échoué: {e}")

            if self.persistence and step_id:
                try:
                    self.persistence.update_step(
                        step_id,
                        AgentStepUpdate(
                            status=StepStatus.FAILED,
                            error_message=str(e),
                            finished_at=datetime.utcnow(),
                        ),
                    )
                except Exception as db_e:
                    logger.warning(f"Erreur update step DB: {db_e}")

            return None, None

    # -------------------------------------------------------------------------
    # STEP C: VÉRIFICATION
    # -------------------------------------------------------------------------

    async def _run_step_c(
        self,
        rapport: RapportAlgo,
        step_b: StepBOutput,
        run_id: UUID,
    ) -> tuple[Optional[StepCOutput], Optional[LLMTrace]]:
        """Exécute l'étape C - Vérification des claims"""
        logger.info("🔍 Step C: Vérification des affirmations...")

        # Créer le step en DB
        step_id = None
        if self.persistence:
            try:
                step_id = self.persistence.create_step(
                    AgentStepCreate(
                        run_id=run_id,
                        step_name=StepName.C,
                        input_json={
                            "step_b_summary": {
                                "total_reviewed": step_b.total_reviewed,
                                "recommended_keep": step_b.recommended_keep,
                            }
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"Erreur création step DB: {e}")

        try:
            # Préparer les données de référence (simplifié pour l'instant)
            reference_data = {
                "rapport_summary": rapport.summary,
                "policy_constraints": rapport.policy_constraints.model_dump(),
                "algo_metrics": rapport.algo_metrics.model_dump(),
            }

            # Préparer le prompt
            prompt, meta = format_prompt(
                "C",
                step_b_output_json=json.dumps(step_b.model_dump(), default=str, indent=2)[:10000],
                reference_data_json=json.dumps(reference_data, default=str, indent=2),
            )

            # Appel LLM
            cfg = LLMConfig(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                step_name="C",
                run_id=str(run_id),
            )

            result, trace = self.provider.generate_structured(
                model=self.config.model,
                prompt=prompt,
                schema=StepCOutput,
                cfg=cfg,
            )

            # Update step en DB
            if self.persistence and step_id:
                try:
                    self.persistence.update_step(
                        step_id,
                        AgentStepUpdate(
                            status=StepStatus.SUCCESS,
                            output_json=result.model_dump(),
                            llm_model=trace.model,
                            llm_prompt_tokens=trace.tokens_in,
                            llm_completion_tokens=trace.tokens_out,
                            llm_cost_usd=trace.cost_estimate_usd,
                            duration_ms=trace.latency_ms,
                            finished_at=datetime.utcnow(),
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Erreur update step DB: {e}")

            logger.info(
                f"✅ Step C terminé | claims={result.total_claims} | "
                f"verified={result.verified_count} | rate={result.verification_rate_pct}%"
            )
            return result, trace

        except Exception as e:
            logger.error(f"❌ Step C échoué: {e}")

            if self.persistence and step_id:
                try:
                    self.persistence.update_step(
                        step_id,
                        AgentStepUpdate(
                            status=StepStatus.FAILED,
                            error_message=str(e),
                            finished_at=datetime.utcnow(),
                        ),
                    )
                except Exception as db_e:
                    logger.warning(f"Erreur update step DB: {db_e}")

            return None, None

    # -------------------------------------------------------------------------
    # STEP D: AUTO-CRITIQUE + FINAL
    # -------------------------------------------------------------------------

    async def _run_step_d(
        self,
        rapport: RapportAlgo,
        step_b: StepBOutput,
        step_c: Optional[StepCOutput],
        run_id: UUID,
    ) -> tuple[Optional[StepDOutput], Optional[LLMTrace]]:
        """Exécute l'étape D - Auto-critique et proposition finale"""
        logger.info("🎯 Step D: Auto-critique et proposition finale...")

        # Créer le step en DB
        step_id = None
        if self.persistence:
            try:
                step_id = self.persistence.create_step(
                    AgentStepCreate(
                        run_id=run_id,
                        step_name=StepName.D,
                        input_json={
                            "step_b_confidence": step_b.overall_confidence,
                            "step_c_verification_rate": step_c.verification_rate_pct
                            if step_c
                            else None,
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"Erreur création step DB: {e}")

        try:
            # Calculer les caps
            daily_budget = rapport.caps.get("daily_budget_eur", rapport.bankroll_eur * 0.12)
            max_stake = rapport.caps.get("max_stake_per_bet", rapport.bankroll_eur * 0.05)

            # Récupérer les leçons apprises (self-learning)
            learned_lessons = ""
            try:
                from services.agent_memory import get_memory_service

                memory = get_memory_service()
                learned_lessons = memory.get_lessons_for_prompt(max_lessons=10)
                if learned_lessons:
                    logger.info(
                        f"📚 {len(learned_lessons)} caractères de leçons injectées dans le prompt"
                    )
            except Exception as e:
                logger.warning(f"Impossible de charger les leçons: {e}")

            # Construire la liste des chevaux valides avec leurs STAKES et BET_TYPE déjà calculés par l'algo
            valid_horses = []
            for race in rapport.races:
                for runner in race.runners:
                    # Récupérer les données depuis algo_decision et les champs directs
                    stake_val = (
                        runner.algo_decision.stake_eur
                        if runner.algo_decision and runner.algo_decision.stake_eur
                        else None
                    )
                    bet_type_val = (
                        runner.algo_decision.bet_type
                        if runner.algo_decision and runner.algo_decision.bet_type
                        else None
                    )
                    kelly_val = runner.kelly_win_pct if runner.kelly_win_pct else None
                    odds_val = runner.odds_final if runner.odds_final else runner.odds_morning
                    value_val = runner.value_win_pct if runner.value_win_pct else None
                    is_kept = (
                        runner.algo_decision.status.value == "KEPT"
                        if runner.algo_decision
                        else False
                    )

                    # Construire la chaîne d'info
                    info_parts = []
                    if bet_type_val:
                        info_parts.append(f"pari={bet_type_val}")
                    if odds_val:
                        info_parts.append(f"cote={odds_val:.2f}")
                    if value_val:
                        info_parts.append(f"value={value_val:.1f}%")
                    if stake_val:
                        info_parts.append(f"stake={stake_val:.0f}€")
                    if is_kept:
                        info_parts.append("✓KEPT")

                    info_str = f" ({', '.join(info_parts)})" if info_parts else ""
                    valid_horses.append(f"{runner.horse_name} | {race.race_key}{info_str}")
            valid_horses_list = "\n".join(valid_horses)

            # Préparer le prompt
            prompt, meta = format_prompt(
                "D",
                algo_report_json=json.dumps(
                    {
                        "summary": rapport.summary,
                        "target_date": str(rapport.target_date),
                        "bankroll": rapport.bankroll_eur,
                        "profile": rapport.profile,
                        "total_races": len(rapport.races),  # Ajouter le nombre de courses
                        "races_list": [r.race_key for r in rapport.races],  # Liste des courses
                    },
                    default=str,
                    indent=2,
                ),
                step_b_output_json=json.dumps(step_b.model_dump(), default=str, indent=2)[
                    :15000
                ],  # Augmenté
                step_c_output_json=json.dumps(
                    step_c.model_dump() if step_c else {"skipped": True}, default=str, indent=2
                )[:8000],
                learned_lessons=learned_lessons,
                daily_budget=daily_budget,
                max_stake_per_bet=max_stake,
                valid_horses_list=valid_horses_list,
            )

            # Appel LLM
            cfg = LLMConfig(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                step_name="D",
                run_id=str(run_id),
            )

            result, trace = self.provider.generate_structured(
                model=self.config.model,
                prompt=prompt,
                schema=StepDOutput,
                cfg=cfg,
            )

            # Update step en DB
            if self.persistence and step_id:
                try:
                    self.persistence.update_step(
                        step_id,
                        AgentStepUpdate(
                            status=StepStatus.SUCCESS,
                            output_json=result.model_dump(),
                            llm_model=trace.model,
                            llm_prompt_tokens=trace.tokens_in,
                            llm_completion_tokens=trace.tokens_out,
                            llm_cost_usd=trace.cost_estimate_usd,
                            duration_ms=trace.latency_ms,
                            finished_at=datetime.utcnow(),
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Erreur update step DB: {e}")

            logger.info(
                f"✅ Step D terminé | final_picks={result.total_picks} | "
                f"confidence={result.portfolio_confidence}% | "
                f"risk={result.risk_assessment}"
            )
            return result, trace

        except Exception as e:
            logger.error(f"❌ Step D échoué: {e}")

            if self.persistence and step_id:
                try:
                    self.persistence.update_step(
                        step_id,
                        AgentStepUpdate(
                            status=StepStatus.FAILED,
                            error_message=str(e),
                            finished_at=datetime.utcnow(),
                        ),
                    )
                except Exception as db_e:
                    logger.warning(f"Erreur update step DB: {db_e}")

            return None, None


# =============================================================================
# SINGLETON
# =============================================================================

_analyzer_service: Optional[AgentAnalyzerService] = None


def get_analyzer_service(
    config: Optional[AgentConfig] = None,
) -> AgentAnalyzerService:
    """Retourne le service d'analyse (singleton)"""
    global _analyzer_service

    if _analyzer_service is None or config is not None:
        persistence = None
        try:
            persistence = get_persistence_service()
        except Exception as e:
            logger.warning(f"Persistence non disponible: {e}")

        _analyzer_service = AgentAnalyzerService(
            config=config,
            persistence=persistence,
        )

    return _analyzer_service
