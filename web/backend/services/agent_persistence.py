"""
🗄️ Agent IA Persistence Service
================================

Service de persistance pour le pipeline Agent IA.
Gère les tables: agent_runs, agent_steps, agent_evidence, agent_diffs

Auteur: Agent IA Pipeline
Date: 2024-12-21
"""

from __future__ import annotations

import json
from datetime import datetime, date
from typing import Any, Optional
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================


class RunStatus(str, Enum):
    """Statuts possibles d'un run"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    STEP_A = "STEP_A"
    STEP_B = "STEP_B"
    STEP_C = "STEP_C"
    STEP_D = "STEP_D"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class StepName(str, Enum):
    """Noms des étapes"""

    A = "A"  # Génération Rapport Algo
    B = "B"  # Analyse IA
    C = "C"  # Vérification
    D = "D"  # Auto-critique + Final


class StepStatus(str, Enum):
    """Statuts d'une étape"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class SourceType(str, Enum):
    """Types de sources pour les preuves"""

    DB = "DB"
    API = "API"
    WEB = "WEB"


class DiffAction(str, Enum):
    """Actions de diff"""

    KEPT = "KEPT"
    REMOVED = "REMOVED"
    MODIFIED = "MODIFIED"
    ADDED = "ADDED"


# =============================================================================
# MODÈLES PYDANTIC
# =============================================================================


class AgentRunCreate(BaseModel):
    """Données pour créer un nouveau run"""

    run_id: Optional[UUID] = None  # Si fourni, utilise ce run_id
    target_date: date
    user_id: Optional[int] = None
    bankroll: float
    profile: str = "STANDARD"
    model_version: Optional[str] = None


class AgentRunUpdate(BaseModel):
    """Données pour mettre à jour un run"""

    status: Optional[RunStatus] = None
    algo_report: Optional[dict] = None
    final_report: Optional[dict] = None
    confidence_score: Optional[int] = None
    total_picks_algo: Optional[int] = None
    total_picks_final: Optional[int] = None
    total_stake_algo: Optional[float] = None
    total_stake_final: Optional[float] = None
    drift_status: Optional[str] = None
    error_message: Optional[str] = None
    finished_at: Optional[datetime] = None


class AgentRun(BaseModel):
    """Modèle complet d'un run"""

    run_id: UUID
    target_date: date
    user_id: Optional[int]
    bankroll: float
    profile: str
    status: RunStatus
    algo_report: Optional[dict] = None
    final_report: Optional[dict] = None
    confidence_score: Optional[int] = None
    total_picks_algo: Optional[int] = None
    total_picks_final: Optional[int] = None
    total_stake_algo: Optional[float] = None
    total_stake_final: Optional[float] = None
    model_version: Optional[str] = None
    drift_status: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime
    error_message: Optional[str] = None


class AgentStepCreate(BaseModel):
    """Données pour créer une étape"""

    run_id: UUID
    step_name: StepName
    input_json: Optional[dict] = None


class AgentStepUpdate(BaseModel):
    """Données pour mettre à jour une étape"""

    status: Optional[StepStatus] = None
    output_json: Optional[dict] = None
    llm_model: Optional[str] = None
    llm_prompt_tokens: Optional[int] = None
    llm_completion_tokens: Optional[int] = None
    llm_cost_usd: Optional[float] = None
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    finished_at: Optional[datetime] = None


class AgentStep(BaseModel):
    """Modèle complet d'une étape"""

    step_id: UUID
    run_id: UUID
    step_name: StepName
    step_order: int
    status: StepStatus
    input_json: Optional[dict] = None
    output_json: Optional[dict] = None
    llm_model: Optional[str] = None
    llm_prompt_tokens: Optional[int] = None
    llm_completion_tokens: Optional[int] = None
    llm_cost_usd: Optional[float] = None
    duration_ms: Optional[int] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime
    error_message: Optional[str] = None


class AgentEvidenceCreate(BaseModel):
    """Données pour créer une preuve"""

    run_id: UUID
    step_id: Optional[UUID] = None
    claim_id: Optional[str] = None
    claim_text: str
    claim_category: Optional[str] = None
    source_type: SourceType
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    query_used: Optional[str] = None
    payload: Optional[dict] = None
    verified: bool = False
    verification_note: Optional[str] = None


class AgentDiffCreate(BaseModel):
    """Données pour créer un diff"""

    run_id: UUID
    race_key: Optional[str] = None
    runner_id: Optional[str] = None
    horse_name: Optional[str] = None
    action: DiffAction
    algo_decision: Optional[dict] = None
    agent_decision: Optional[dict] = None
    reason: Optional[str] = None
    stake_change: Optional[float] = None
    ev_change: Optional[float] = None


# =============================================================================
# SERVICE DE PERSISTANCE
# =============================================================================


class AgentPersistenceService:
    """
    Service de persistance pour le pipeline Agent IA.

    Gère les opérations CRUD sur les tables agent_*.
    """

    def __init__(self, get_connection_func):
        """
        Initialise le service.

        Args:
            get_connection_func: Fonction qui retourne une connexion DB
        """
        self.get_connection = get_connection_func
        self._use_postgresql = True  # Supposé PostgreSQL

    def _adapt_query(self, sql: str) -> str:
        """Adapte les placeholders si nécessaire"""
        return sql  # PostgreSQL utilise %s

    def _to_json(self, data: Any) -> Optional[str]:
        """Convertit en JSON string pour PostgreSQL"""
        if data is None:
            return None
        if isinstance(data, str):
            return data
        return json.dumps(data, default=str)

    def _from_json(self, data: Any) -> Optional[dict]:
        """Parse JSON depuis PostgreSQL"""
        if data is None:
            return None
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            return json.loads(data)
        return dict(data)

    # -------------------------------------------------------------------------
    # RUNS
    # -------------------------------------------------------------------------

    def create_run(self, data: AgentRunCreate) -> UUID:
        """
        Crée un nouveau run.

        Args:
            data: Données du run

        Returns:
            run_id du nouveau run
        """
        run_id = data.run_id or uuid4()  # Utilise run_id fourni ou génère un nouveau
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                INSERT INTO agent_runs
                (run_id, target_date, user_id, bankroll, profile, status,
                 model_version, started_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    str(run_id),
                    data.target_date,
                    data.user_id,
                    data.bankroll,
                    data.profile,
                    RunStatus.RUNNING.value,
                    data.model_version,
                    datetime.utcnow(),
                ),
            )
            con.commit()
            return run_id
        finally:
            con.close()

    def get_run(self, run_id: UUID) -> Optional[AgentRun]:
        """Récupère un run par son ID"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                SELECT run_id, target_date, user_id, bankroll, profile, status,
                       algo_report, final_report, confidence_score,
                       total_picks_algo, total_picks_final,
                       total_stake_algo, total_stake_final,
                       model_version, drift_status,
                       started_at, finished_at, created_at, error_message
                FROM agent_runs
                WHERE run_id = %s
            """,
                (str(run_id),),
            )

            row = cur.fetchone()
            if not row:
                return None

            return AgentRun(
                run_id=UUID(row[0]) if isinstance(row[0], str) else row[0],
                target_date=row[1],
                user_id=row[2],
                bankroll=float(row[3]),
                profile=row[4],
                status=RunStatus(row[5]),
                algo_report=self._from_json(row[6]),
                final_report=self._from_json(row[7]),
                confidence_score=row[8],
                total_picks_algo=row[9],
                total_picks_final=row[10],
                total_stake_algo=float(row[11]) if row[11] else None,
                total_stake_final=float(row[12]) if row[12] else None,
                model_version=row[13],
                drift_status=row[14],
                started_at=row[15],
                finished_at=row[16],
                created_at=row[17],
                error_message=row[18],
            )
        finally:
            con.close()

    def update_run(self, run_id: UUID, data: AgentRunUpdate) -> bool:
        """Met à jour un run"""
        updates = []
        values = []

        if data.status is not None:
            updates.append("status = %s")
            values.append(data.status.value)
        if data.algo_report is not None:
            updates.append("algo_report = %s")
            values.append(self._to_json(data.algo_report))
        if data.final_report is not None:
            updates.append("final_report = %s")
            values.append(self._to_json(data.final_report))
        if data.confidence_score is not None:
            updates.append("confidence_score = %s")
            values.append(data.confidence_score)
        if data.total_picks_algo is not None:
            updates.append("total_picks_algo = %s")
            values.append(data.total_picks_algo)
        if data.total_picks_final is not None:
            updates.append("total_picks_final = %s")
            values.append(data.total_picks_final)
        if data.total_stake_algo is not None:
            updates.append("total_stake_algo = %s")
            values.append(data.total_stake_algo)
        if data.total_stake_final is not None:
            updates.append("total_stake_final = %s")
            values.append(data.total_stake_final)
        if data.drift_status is not None:
            updates.append("drift_status = %s")
            values.append(data.drift_status)
        if data.error_message is not None:
            updates.append("error_message = %s")
            values.append(data.error_message)
        if data.finished_at is not None:
            updates.append("finished_at = %s")
            values.append(data.finished_at)

        if not updates:
            return False

        values.append(str(run_id))

        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                f"""
                UPDATE agent_runs
                SET {', '.join(updates)}
                WHERE run_id = %s
            """,
                tuple(values),
            )
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()

    def list_runs(
        self,
        user_id: Optional[int] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[AgentRun]:
        """Liste les runs avec filtres optionnels"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            query = """
                SELECT run_id, target_date, user_id, bankroll, profile, status,
                       algo_report, final_report, confidence_score,
                       total_picks_algo, total_picks_final,
                       total_stake_algo, total_stake_final,
                       model_version, drift_status,
                       started_at, finished_at, created_at, error_message
                FROM agent_runs
            """
            params = []

            if user_id is not None:
                query += " WHERE user_id = %s"
                params.append(user_id)

            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cur.execute(query, tuple(params))

            runs = []
            for row in cur.fetchall():
                runs.append(
                    AgentRun(
                        run_id=UUID(row[0]) if isinstance(row[0], str) else row[0],
                        target_date=row[1],
                        user_id=row[2],
                        bankroll=float(row[3]),
                        profile=row[4],
                        status=RunStatus(row[5]),
                        algo_report=self._from_json(row[6]),
                        final_report=self._from_json(row[7]),
                        confidence_score=row[8],
                        total_picks_algo=row[9],
                        total_picks_final=row[10],
                        total_stake_algo=float(row[11]) if row[11] else None,
                        total_stake_final=float(row[12]) if row[12] else None,
                        model_version=row[13],
                        drift_status=row[14],
                        started_at=row[15],
                        finished_at=row[16],
                        created_at=row[17],
                        error_message=row[18],
                    )
                )

            return runs
        finally:
            con.close()

    # -------------------------------------------------------------------------
    # STEPS
    # -------------------------------------------------------------------------

    def create_step(self, data: AgentStepCreate) -> UUID:
        """Crée une nouvelle étape"""
        step_id = uuid4()
        step_order = {"A": 1, "B": 2, "C": 3, "D": 4}.get(data.step_name.value, 0)

        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                INSERT INTO agent_steps
                (step_id, run_id, step_name, step_order, status, input_json, started_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    str(step_id),
                    str(data.run_id),
                    data.step_name.value,
                    step_order,
                    StepStatus.RUNNING.value,
                    self._to_json(data.input_json),
                    datetime.utcnow(),
                ),
            )
            con.commit()
            return step_id
        finally:
            con.close()

    def update_step(self, step_id: UUID, data: AgentStepUpdate) -> bool:
        """Met à jour une étape"""
        updates = []
        values = []

        if data.status is not None:
            updates.append("status = %s")
            values.append(data.status.value)
        if data.output_json is not None:
            updates.append("output_json = %s")
            values.append(self._to_json(data.output_json))
        if data.llm_model is not None:
            updates.append("llm_model = %s")
            values.append(data.llm_model)
        if data.llm_prompt_tokens is not None:
            updates.append("llm_prompt_tokens = %s")
            values.append(data.llm_prompt_tokens)
        if data.llm_completion_tokens is not None:
            updates.append("llm_completion_tokens = %s")
            values.append(data.llm_completion_tokens)
        if data.llm_cost_usd is not None:
            updates.append("llm_cost_usd = %s")
            values.append(data.llm_cost_usd)
        if data.duration_ms is not None:
            updates.append("duration_ms = %s")
            values.append(data.duration_ms)
        if data.error_message is not None:
            updates.append("error_message = %s")
            values.append(data.error_message)
        if data.finished_at is not None:
            updates.append("finished_at = %s")
            values.append(data.finished_at)

        if not updates:
            return False

        values.append(str(step_id))

        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                f"""
                UPDATE agent_steps
                SET {', '.join(updates)}
                WHERE step_id = %s
            """,
                tuple(values),
            )
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()

    def get_steps_for_run(self, run_id: UUID) -> list[AgentStep]:
        """Récupère toutes les étapes d'un run"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                SELECT step_id, run_id, step_name, step_order, status,
                       input_json, output_json,
                       llm_model, llm_prompt_tokens, llm_completion_tokens, llm_cost_usd,
                       duration_ms, started_at, finished_at, created_at, error_message
                FROM agent_steps
                WHERE run_id = %s
                ORDER BY step_order
            """,
                (str(run_id),),
            )

            steps = []
            for row in cur.fetchall():
                steps.append(
                    AgentStep(
                        step_id=UUID(row[0]) if isinstance(row[0], str) else row[0],
                        run_id=UUID(row[1]) if isinstance(row[1], str) else row[1],
                        step_name=StepName(row[2]),
                        step_order=row[3],
                        status=StepStatus(row[4]),
                        input_json=self._from_json(row[5]),
                        output_json=self._from_json(row[6]),
                        llm_model=row[7],
                        llm_prompt_tokens=row[8],
                        llm_completion_tokens=row[9],
                        llm_cost_usd=float(row[10]) if row[10] else None,
                        duration_ms=row[11],
                        started_at=row[12],
                        finished_at=row[13],
                        created_at=row[14],
                        error_message=row[15],
                    )
                )

            return steps
        finally:
            con.close()

    # -------------------------------------------------------------------------
    # EVIDENCE
    # -------------------------------------------------------------------------

    def add_evidence(self, data: AgentEvidenceCreate) -> UUID:
        """Ajoute une preuve"""
        evidence_id = uuid4()

        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                INSERT INTO agent_evidence
                (evidence_id, run_id, step_id, claim_id, claim_text, claim_category,
                 source_type, source_name, source_url, query_used, payload,
                 verified, verification_note)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    str(evidence_id),
                    str(data.run_id),
                    str(data.step_id) if data.step_id else None,
                    data.claim_id,
                    data.claim_text,
                    data.claim_category,
                    data.source_type.value,
                    data.source_name,
                    data.source_url,
                    data.query_used,
                    self._to_json(data.payload),
                    data.verified,
                    data.verification_note,
                ),
            )
            con.commit()
            return evidence_id
        finally:
            con.close()

    def get_evidence_for_run(self, run_id: UUID) -> list[dict]:
        """Récupère toutes les preuves d'un run"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                SELECT evidence_id, run_id, step_id, claim_id, claim_text, claim_category,
                       source_type, source_name, source_url, query_used, payload,
                       verified, verification_note, created_at
                FROM agent_evidence
                WHERE run_id = %s
                ORDER BY created_at
            """,
                (str(run_id),),
            )

            evidence = []
            for row in cur.fetchall():
                evidence.append(
                    {
                        "evidence_id": str(row[0]),
                        "run_id": str(row[1]),
                        "step_id": str(row[2]) if row[2] else None,
                        "claim_id": row[3],
                        "claim_text": row[4],
                        "claim_category": row[5],
                        "source_type": row[6],
                        "source_name": row[7],
                        "source_url": row[8],
                        "query_used": row[9],
                        "payload": self._from_json(row[10]),
                        "verified": row[11],
                        "verification_note": row[12],
                        "created_at": row[13].isoformat() if row[13] else None,
                    }
                )

            return evidence
        finally:
            con.close()

    # -------------------------------------------------------------------------
    # DIFFS
    # -------------------------------------------------------------------------

    def add_diff(self, data: AgentDiffCreate) -> UUID:
        """Ajoute un diff"""
        diff_id = uuid4()

        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                INSERT INTO agent_diffs
                (diff_id, run_id, race_key, runner_id, horse_name,
                 action, algo_decision, agent_decision, reason,
                 stake_change, ev_change)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    str(diff_id),
                    str(data.run_id),
                    data.race_key,
                    data.runner_id,
                    data.horse_name,
                    data.action.value,
                    self._to_json(data.algo_decision),
                    self._to_json(data.agent_decision),
                    data.reason,
                    data.stake_change,
                    data.ev_change,
                ),
            )
            con.commit()
            return diff_id
        finally:
            con.close()

    def get_diffs_for_run(self, run_id: UUID) -> list[dict]:
        """Récupère tous les diffs d'un run"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                SELECT diff_id, run_id, race_key, runner_id, horse_name,
                       action, algo_decision, agent_decision, reason,
                       stake_change, ev_change, created_at
                FROM agent_diffs
                WHERE run_id = %s
                ORDER BY race_key, runner_id
            """,
                (str(run_id),),
            )

            diffs = []
            for row in cur.fetchall():
                diffs.append(
                    {
                        "diff_id": str(row[0]),
                        "run_id": str(row[1]),
                        "race_key": row[2],
                        "runner_id": row[3],
                        "horse_name": row[4],
                        "action": row[5],
                        "algo_decision": self._from_json(row[6]),
                        "agent_decision": self._from_json(row[7]),
                        "reason": row[8],
                        "stake_change": float(row[9]) if row[9] else None,
                        "ev_change": float(row[10]) if row[10] else None,
                        "created_at": row[11].isoformat() if row[11] else None,
                    }
                )

            return diffs
        finally:
            con.close()

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def init_tables(self) -> bool:
        """
        Initialise les tables si elles n'existent pas.
        Lit et exécute les fichiers de migration SQL (v1 + v2).
        """
        from pathlib import Path

        # Liste des migrations à exécuter dans l'ordre
        migration_files = [
            "agent_ia_migration.sql",  # Tables de base
            "agent_ia_migration_v2.sql",  # Contraintes CHECK, attempt, GIN
        ]

        base_paths = [
            Path(__file__).parent.parent.parent.parent / "sql",
            Path("/app/sql"),
        ]

        success_count = 0

        for migration_file in migration_files:
            migration_sql = None

            for base_path in base_paths:
                path = base_path / migration_file
                if path.exists():
                    with open(path) as f:
                        migration_sql = f.read()
                    break

            if not migration_sql:
                print(f"[WARN] Migration {migration_file} not found")
                continue

            con = self.get_connection()
            cur = con.cursor()

            try:
                cur.execute(migration_sql)
                con.commit()
                print(f"[INFO] Migration {migration_file} executed successfully")
                success_count += 1
            except Exception as e:
                print(
                    f"[WARN] Migration {migration_file} error (may be OK if already applied): {e}"
                )
                con.rollback()
            finally:
                con.close()

        return success_count > 0

    def get_run_summary(self, run_id: UUID) -> dict:
        """Récupère un résumé complet d'un run avec ses étapes et diffs"""
        run = self.get_run(run_id)
        if not run:
            return {}

        steps = self.get_steps_for_run(run_id)
        diffs = self.get_diffs_for_run(run_id)
        evidence = self.get_evidence_for_run(run_id)

        return {
            "run": run.model_dump(),
            "steps": [s.model_dump() for s in steps],
            "diffs": diffs,
            "evidence": evidence,
            "stats": {
                "total_steps": len(steps),
                "completed_steps": sum(1 for s in steps if s.status == StepStatus.SUCCESS),
                "total_diffs": len(diffs),
                "total_evidence": len(evidence),
                "verified_evidence": sum(1 for e in evidence if e.get("verified")),
            },
        }


# =============================================================================
# INSTANCE GLOBALE (initialisée à l'import)
# =============================================================================

_persistence_service: Optional[AgentPersistenceService] = None


def get_persistence_service() -> AgentPersistenceService:
    """Retourne le service de persistance (singleton)"""
    global _persistence_service

    if _persistence_service is None:
        # Import dynamique pour éviter les dépendances circulaires
        try:
            import sys
            import os

            parent_dir = os.path.join(os.path.dirname(__file__), "..")
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from main import get_db_connection

            _persistence_service = AgentPersistenceService(get_db_connection)
        except ImportError:
            # Fallback pour tests
            def mock_connection():
                raise RuntimeError("Database not configured")

            _persistence_service = AgentPersistenceService(mock_connection)

    return _persistence_service
