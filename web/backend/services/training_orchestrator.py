"""
🎓 Training Orchestrator - Automated Training Pipeline
======================================================

Orchestrateur d'entraînement automatisé pour l'Agent IA.
Lance des backtests progressifs période par période, génère des leçons,
et suit l'évolution de l'accuracy au fil du temps.

Auteur: Agent IA Pipeline
Date: 2024-12-22
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Optional, Callable
from uuid import UUID
import uuid

logger = logging.getLogger("agent_ia.training")


# =============================================================================
# TRAINING ORCHESTRATOR
# =============================================================================


class TrainingOrchestrator:
    """
    Orchestrateur d'entraînement automatisé.

    Gère:
    - Lancement de sessions d'entraînement (backtest par période)
    - Pause et reprise
    - Suivi de la courbe d'apprentissage
    """

    def __init__(self, db_connection_fn: Callable):
        self.get_connection = db_connection_fn
        self._memory_service = None

    @property
    def memory(self):
        """Lazy load du service mémoire"""
        if self._memory_service is None:
            from services.agent_memory import get_memory_service

            self._memory_service = get_memory_service()
        return self._memory_service

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def start_training(self, start_date: date, end_date: date, period_type: str = "WEEK") -> dict:
        """
        Démarre une nouvelle session d'entraînement.

        Args:
            start_date: Date de début (ex: 2020-01-01)
            end_date: Date de fin (ex: 2024-12-31)
            period_type: WEEK ou MONTH

        Returns:
            Session info avec session_id
        """
        con = self.get_connection()
        cur = con.cursor()

        try:
            # Calculer le nombre de périodes
            total_periods = self._calculate_periods(start_date, end_date, period_type)

            # Créer la session
            session_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO agent_training_sessions (
                    session_id, start_date, end_date, period_type,
                    status, current_period_start, current_period_end,
                    total_periods, learning_curve
                ) VALUES (%s, %s, %s, %s, 'RUNNING', %s, %s, %s, '[]')
                RETURNING session_id
            """,
                (
                    session_id,
                    start_date,
                    end_date,
                    period_type,
                    start_date,
                    self._get_period_end(start_date, period_type),
                    total_periods,
                ),
            )

            con.commit()

            logger.info(
                f"🎓 Started training session {session_id}: {start_date} → {end_date} ({total_periods} periods)"
            )

            return {
                "success": True,
                "session_id": session_id,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "total_periods": total_periods,
                "status": "RUNNING",
            }

        except Exception as e:
            con.rollback()
            logger.error(f"Failed to start training: {e}")
            return {"success": False, "error": str(e)}
        finally:
            con.close()

    def pause_training(self, session_id: str) -> dict:
        """Met en pause une session d'entraînement"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                UPDATE agent_training_sessions
                SET status = 'PAUSED', paused_at = NOW(), updated_at = NOW()
                WHERE session_id = %s AND status = 'RUNNING'
                RETURNING session_id
            """,
                (session_id,),
            )

            if cur.rowcount == 0:
                return {"success": False, "error": "Session not found or not running"}

            con.commit()
            logger.info(f"⏸️ Paused training session {session_id}")

            return {"success": True, "session_id": session_id, "status": "PAUSED"}

        except Exception as e:
            con.rollback()
            return {"success": False, "error": str(e)}
        finally:
            con.close()

    def resume_training(self, session_id: str) -> dict:
        """Reprend une session d'entraînement en pause"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                UPDATE agent_training_sessions
                SET status = 'RUNNING', paused_at = NULL, updated_at = NOW()
                WHERE session_id = %s AND status = 'PAUSED'
                RETURNING session_id, current_period_start, current_period_end
            """,
                (session_id,),
            )

            row = cur.fetchone()
            if not row:
                return {"success": False, "error": "Session not found or not paused"}

            con.commit()
            logger.info(f"▶️ Resumed training session {session_id}")

            return {
                "success": True,
                "session_id": session_id,
                "status": "RUNNING",
                "current_period": f"{row[1]} → {row[2]}",
            }

        except Exception as e:
            con.rollback()
            return {"success": False, "error": str(e)}
        finally:
            con.close()

    def get_session(self, session_id: str) -> Optional[dict]:
        """Récupère les détails d'une session"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                SELECT session_id, start_date, end_date, period_type, status,
                       current_period_start, current_period_end,
                       periods_completed, total_periods,
                       total_predictions, total_correct, cumulative_pnl,
                       lessons_generated, learning_curve,
                       created_at, updated_at, paused_at, completed_at
                FROM agent_training_sessions
                WHERE session_id = %s
            """,
                (session_id,),
            )

            row = cur.fetchone()
            if not row:
                return None

            return {
                "session_id": str(row[0]),
                "start_date": str(row[1]),
                "end_date": str(row[2]),
                "period_type": row[3],
                "status": row[4],
                "current_period_start": str(row[5]) if row[5] else None,
                "current_period_end": str(row[6]) if row[6] else None,
                "periods_completed": row[7],
                "total_periods": row[8],
                "progress_pct": round(100 * row[7] / row[8], 1) if row[8] else 0,
                "total_predictions": row[9],
                "total_correct": row[10],
                "accuracy_pct": round(100 * row[10] / row[9], 2) if row[9] else 0,
                "cumulative_pnl": float(row[11]) if row[11] else 0,
                "lessons_generated": row[12],
                "learning_curve": row[13] or [],
                "created_at": row[14].isoformat() if row[14] else None,
                "updated_at": row[15].isoformat() if row[15] else None,
            }

        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return None
        finally:
            con.close()

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """Liste les sessions d'entraînement"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                SELECT session_id, start_date, end_date, status,
                       periods_completed, total_periods,
                       total_predictions, total_correct, cumulative_pnl, lessons_generated,
                       created_at
                FROM agent_training_sessions
                ORDER BY created_at DESC
                LIMIT %s
            """,
                (limit,),
            )

            sessions = []
            for row in cur.fetchall():
                sessions.append(
                    {
                        "session_id": str(row[0]),
                        "start_date": str(row[1]),
                        "end_date": str(row[2]),
                        "status": row[3],
                        "periods_completed": row[4],
                        "total_periods": row[5],
                        "progress_pct": round(100 * row[4] / row[5], 1) if row[5] else 0,
                        "total_predictions": row[6],
                        "accuracy_pct": round(100 * row[7] / row[6], 2) if row[6] else 0,
                        "cumulative_pnl": float(row[8]) if row[8] else 0,
                        "lessons_generated": row[9],
                        "created_at": row[10].isoformat() if row[10] else None,
                    }
                )

            return sessions

        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []
        finally:
            con.close()

    # =========================================================================
    # TRAINING EXECUTION
    # =========================================================================

    def run_training_loop(self, session_id: str) -> dict:
        """
        Exécute la boucle d'entraînement pour une session.

        Pour chaque période:
        1. Lance un backtest sur la période
        2. Génère les leçons
        3. Enregistre les résultats dans la courbe d'apprentissage
        4. Passe à la période suivante

        S'arrête si la session est mise en pause.
        """
        logger.info(f"🏃 Starting training loop for {session_id}")

        while True:
            # Vérifier le statut de la session
            session = self.get_session(session_id)
            if not session:
                return {"success": False, "error": "Session not found"}

            if session["status"] == "PAUSED":
                logger.info(
                    f"⏸️ Training paused at period {session['periods_completed']}/{session['total_periods']}"
                )
                return {
                    "success": True,
                    "status": "PAUSED",
                    "periods_completed": session["periods_completed"],
                }

            if session["status"] == "COMPLETED":
                logger.info("✅ Training already completed")
                return {
                    "success": True,
                    "status": "COMPLETED",
                    "periods_completed": session["periods_completed"],
                }

            if session["periods_completed"] >= session["total_periods"]:
                # Marquer comme terminé
                self._complete_session(session_id)
                return {
                    "success": True,
                    "status": "COMPLETED",
                    "periods_completed": session["periods_completed"],
                    "total_predictions": session["total_predictions"],
                    "accuracy_pct": session["accuracy_pct"],
                    "lessons_generated": session["lessons_generated"],
                }

            # Traiter la période courante
            period_start = date.fromisoformat(session["current_period_start"])
            period_end = date.fromisoformat(session["current_period_end"])

            logger.info(
                f"📅 Processing period {session['periods_completed'] + 1}/{session['total_periods']}: {period_start} → {period_end}"
            )

            try:
                # 1. Lancer le backtest pour cette période
                backtest_result = self.memory.create_backtest(
                    start_date=period_start, end_date=period_end, profile="STANDARD", bankroll=500.0
                )

                if not backtest_result.get("success"):
                    logger.warning(f"Backtest failed: {backtest_result.get('error')}")
                    # Continuer quand même
                    predictions, correct, pnl, lessons = 0, 0, 0, 0
                else:
                    backtest_id = backtest_result.get("backtest_id")

                    # Exécuter le backtest
                    run_result = self.memory.run_backtest(backtest_id)

                    predictions = run_result.get("total_predictions", 0)
                    correct = run_result.get("correct_predictions", 0)
                    pnl = run_result.get("total_pnl", 0)
                    lessons = run_result.get("lessons_created", 0)

                # 2. Mettre à jour la session
                self._update_session_progress(
                    session_id=session_id,
                    period_start=period_start,
                    period_end=period_end,
                    period_type=session["period_type"],
                    predictions=predictions,
                    correct=correct,
                    pnl=pnl,
                    lessons=lessons,
                    end_date=date.fromisoformat(session["end_date"]),
                )

            except Exception as e:
                logger.error(f"Error processing period: {e}")
                # Continuer avec la période suivante
                self._advance_to_next_period(
                    session_id, session["period_type"], date.fromisoformat(session["end_date"])
                )

    def _update_session_progress(
        self,
        session_id: str,
        period_start: date,
        period_end: date,
        period_type: str,
        predictions: int,
        correct: int,
        pnl: float,
        lessons: int,
        end_date: date,
    ):
        """Met à jour la progression de la session après une période"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            # Calculer la prochaine période
            next_start = period_end + timedelta(days=1)
            next_end = self._get_period_end(next_start, period_type)

            # Limiter à la date de fin
            if next_end > end_date:
                next_end = end_date

            # Ajouter à la courbe d'apprentissage
            accuracy = round(100 * correct / predictions, 2) if predictions > 0 else 0
            curve_point = {
                "period": f"{period_start} → {period_end}",
                "start": str(period_start),
                "end": str(period_end),
                "predictions": predictions,
                "correct": correct,
                "accuracy": accuracy,
                "pnl": round(pnl, 2),
                "lessons": lessons,
            }

            cur.execute(
                """
                UPDATE agent_training_sessions
                SET
                    periods_completed = periods_completed + 1,
                    current_period_start = %s,
                    current_period_end = %s,
                    total_predictions = total_predictions + %s,
                    total_correct = total_correct + %s,
                    cumulative_pnl = cumulative_pnl + %s,
                    lessons_generated = lessons_generated + %s,
                    learning_curve = learning_curve || %s::jsonb,
                    updated_at = NOW()
                WHERE session_id = %s
            """,
                (
                    next_start,
                    next_end,
                    predictions,
                    correct,
                    pnl,
                    lessons,
                    json.dumps([curve_point]),
                    session_id,
                ),
            )

            con.commit()
            logger.info(
                f"📊 Period done: {predictions} predictions, {accuracy}% accuracy, {pnl}€ PnL, {lessons} lessons"
            )

        except Exception as e:
            con.rollback()
            logger.error(f"Error updating progress: {e}")
        finally:
            con.close()

    def _advance_to_next_period(self, session_id: str, period_type: str, end_date: date):
        """Avance à la période suivante sans enregistrer de résultats"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                SELECT current_period_end FROM agent_training_sessions
                WHERE session_id = %s
            """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return

            current_end = row[0]
            next_start = current_end + timedelta(days=1)
            next_end = self._get_period_end(next_start, period_type)

            if next_end > end_date:
                next_end = end_date

            cur.execute(
                """
                UPDATE agent_training_sessions
                SET
                    periods_completed = periods_completed + 1,
                    current_period_start = %s,
                    current_period_end = %s,
                    updated_at = NOW()
                WHERE session_id = %s
            """,
                (next_start, next_end, session_id),
            )

            con.commit()

        except Exception as e:
            con.rollback()
            logger.error(f"Error advancing period: {e}")
        finally:
            con.close()

    def _complete_session(self, session_id: str):
        """Marque une session comme terminée"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                UPDATE agent_training_sessions
                SET status = 'COMPLETED', completed_at = NOW(), updated_at = NOW()
                WHERE session_id = %s
            """,
                (session_id,),
            )
            con.commit()
            logger.info(f"✅ Training session {session_id} completed!")
        except Exception as e:
            con.rollback()
            logger.error(f"Error completing session: {e}")
        finally:
            con.close()

    async def run_llm_training_loop(self, session_id: str) -> dict:
        """
        Boucle d'entraînement avec vrai Agent IA Gemini.

        Pour chaque JOUR (pas semaine):
        1. Appelle _process_backtest_day_with_llm() avec Gemini
        2. Génère les leçons
        3. Les leçons sont automatiquement injectées dans les prochains jours

        Plus lent mais vraie boucle d'apprentissage!
        """
        import asyncio

        logger.info(f"🎓 Starting LLM training loop for {session_id}")

        while True:
            # Vérifier le statut
            session = self.get_session(session_id)
            if not session:
                return {"success": False, "error": "Session not found"}

            if session["status"] == "PAUSED":
                logger.info(f"⏸️ LLM Training paused at day {session['periods_completed']}")
                return {
                    "success": True,
                    "status": "PAUSED",
                    "periods_completed": session["periods_completed"],
                }

            if session["status"] == "COMPLETED":
                return {"success": True, "status": "COMPLETED"}

            if session["periods_completed"] >= session["total_periods"]:
                self._complete_session(session_id)
                return {"success": True, "status": "COMPLETED", **session}

            # Calculer le jour actuel à traiter
            start_date = date.fromisoformat(session["start_date"])
            current_day = start_date + timedelta(days=session["periods_completed"])
            end_date = date.fromisoformat(session["end_date"])

            if current_day > end_date:
                self._complete_session(session_id)
                return {"success": True, "status": "COMPLETED"}

            logger.info(
                f"📅 LLM Processing day {session['periods_completed'] + 1}/{session['total_periods']}: {current_day}"
            )

            try:
                # 1. Appeler le vrai Agent IA pour ce jour
                day_result = await self.memory._process_backtest_day_with_llm(
                    target_date=current_day, profile="STANDARD", bankroll=500.0
                )

                predictions = day_result.get("predictions", 0)
                correct = day_result.get("correct", 0)
                pnl = day_result.get("pnl", 0)

                # 2. Générer les leçons après ce jour (utilise tous les outcomes accumulés)
                lessons = 0
                if predictions > 0:
                    try:
                        lessons_result = self.memory.generate_lessons_all()
                        lessons = lessons_result.get("lessons_created", 0)
                    except Exception as e:
                        logger.warning(f"Lesson generation failed: {e}")

                # 3. Mettre à jour la session
                self._update_llm_session_progress(
                    session_id=session_id,
                    day_processed=current_day,
                    predictions=predictions,
                    correct=correct,
                    pnl=pnl,
                    lessons=lessons,
                )

                # Petit délai pour rate limiting Gemini
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing day {current_day}: {e}")
                # Avancer quand même
                self._advance_llm_day(session_id)

    def _update_llm_session_progress(
        self,
        session_id: str,
        day_processed: date,
        predictions: int,
        correct: int,
        pnl: float,
        lessons: int,
    ):
        """Met à jour la session après un jour LLM"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            accuracy = round(100 * correct / predictions, 2) if predictions > 0 else 0

            # Ajouter à la courbe d'apprentissage (grouper par semaine pour lisibilité)
            curve_point = {
                "day": str(day_processed),
                "predictions": predictions,
                "correct": correct,
                "accuracy": accuracy,
                "pnl": round(pnl, 2),
                "lessons": lessons,
            }

            cur.execute(
                """
                UPDATE agent_training_sessions
                SET
                    periods_completed = periods_completed + 1,
                    total_predictions = total_predictions + %s,
                    total_correct = total_correct + %s,
                    cumulative_pnl = cumulative_pnl + %s,
                    lessons_generated = lessons_generated + %s,
                    learning_curve = learning_curve || %s::jsonb,
                    updated_at = NOW()
                WHERE session_id = %s
            """,
                (predictions, correct, pnl, lessons, json.dumps([curve_point]), session_id),
            )

            con.commit()
            logger.info(
                f"📊 Day {day_processed}: {predictions} picks, {accuracy}% acc, {pnl}€, {lessons} lessons"
            )

        except Exception as e:
            con.rollback()
            logger.error(f"Error updating LLM progress: {e}")
        finally:
            con.close()

    def _advance_llm_day(self, session_id: str):
        """Avance d'un jour sans résultats"""
        con = self.get_connection()
        cur = con.cursor()

        try:
            cur.execute(
                """
                UPDATE agent_training_sessions
                SET periods_completed = periods_completed + 1, updated_at = NOW()
                WHERE session_id = %s
            """,
                (session_id,),
            )
            con.commit()
        except Exception as e:
            con.rollback()
        finally:
            con.close()

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _calculate_periods(self, start: date, end: date, period_type: str) -> int:
        """Calcule le nombre de périodes entre deux dates"""
        if period_type == "WEEK":
            return ((end - start).days // 7) + 1
        elif period_type == "MONTH":
            months = (end.year - start.year) * 12 + (end.month - start.month) + 1
            return months
        else:
            return ((end - start).days // 7) + 1

    def _get_period_end(self, start: date, period_type: str) -> date:
        """Calcule la date de fin d'une période"""
        if period_type == "WEEK":
            return start + timedelta(days=6)
        elif period_type == "MONTH":
            # Dernier jour du mois
            if start.month == 12:
                next_month = date(start.year + 1, 1, 1)
            else:
                next_month = date(start.year, start.month + 1, 1)
            return next_month - timedelta(days=1)
        else:
            return start + timedelta(days=6)


# =============================================================================
# SINGLETON
# =============================================================================

_training_service: Optional[TrainingOrchestrator] = None


def get_training_service() -> TrainingOrchestrator:
    """Retourne le service d'entraînement (singleton)"""
    global _training_service

    if _training_service is None:
        try:
            import sys
            import os

            parent_dir = os.path.join(os.path.dirname(__file__), "..")
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from main import get_db_connection

            _training_service = TrainingOrchestrator(get_db_connection)
        except Exception as e:
            logger.error(f"Error init training service: {e}")
            raise

    return _training_service
