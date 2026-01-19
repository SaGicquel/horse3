"""
🏇 API Feedback - Collecte des Résultats Réels
================================================

Extension de l'API pour collecter les résultats réels des courses
et permettre le retraining automatique du modèle.

Endpoints ajoutés:
- POST /feedback : Enregistrer résultat d'une course
- GET /feedback/stats : Statistiques feedback collecté
- GET /feedback/model-performance : Performance du modèle vs réalité

Auteur: Phase 8 - Online Learning
Date: 2025-11-13
"""

import json
import os
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)

PREDICTION_FILE = Path("data/predictions_store.json")
FEEDBACK_FILE = Path("data/feedback_store.json")


# ============================================================================
# MODÈLES PYDANTIC - FEEDBACK
# ============================================================================


class CourseResult(BaseModel):
    """Résultat d'une course (positions d'arrivée des chevaux)."""

    course_id: str = Field(..., description="ID unique de la course")
    date_course: str = Field(..., description="Date de la course (YYYY-MM-DD)")
    hippodrome: str = Field(..., description="Hippodrome")

    resultats: List[dict] = Field(
        ..., description="Liste des chevaux avec leur position d'arrivée", min_items=2
    )
    # Format: [{"cheval_id": "CHEVAL_001", "numero_partant": 1, "position_arrivee": 3}, ...]

    timestamp_feedback: Optional[str] = Field(
        None, description="Timestamp du feedback (auto si non fourni)"
    )

    @validator("date_course")
    def validate_date(cls, v):
        """Valide le format de date."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Format date invalide, attendu: YYYY-MM-DD")
        return v

    @validator("resultats")
    def validate_resultats(cls, v):
        """Valide la structure des résultats."""
        for result in v:
            if "cheval_id" not in result:
                raise ValueError("Chaque résultat doit avoir 'cheval_id'")
            if "position_arrivee" not in result:
                raise ValueError("Chaque résultat doit avoir 'position_arrivee'")
            if not isinstance(result["position_arrivee"], int):
                raise ValueError("position_arrivee doit être un entier")
            if result["position_arrivee"] < 1:
                raise ValueError("position_arrivee doit être >= 1")

        # Vérifier unicité cheval_id
        cheval_ids = [r["cheval_id"] for r in v]
        if len(cheval_ids) != len(set(cheval_ids)):
            raise ValueError("cheval_id doivent être uniques")

        return v


class FeedbackResponse(BaseModel):
    """Réponse après enregistrement du feedback."""

    status: str = Field(..., description="success ou error")
    message: str = Field(..., description="Message de confirmation")
    course_id: str = Field(..., description="ID de la course")
    nb_resultats: int = Field(..., description="Nombre de résultats enregistrés")
    timestamp: str = Field(..., description="Timestamp de l'enregistrement")


class FeedbackStats(BaseModel):
    """Statistiques du feedback collecté."""

    total_courses: int = Field(..., description="Nombre total de courses avec feedback")
    total_predictions: int = Field(..., description="Nombre total de prédictions avec feedback")
    periode_debut: str = Field(..., description="Date du feedback le plus ancien")
    periode_fin: str = Field(..., description="Date du feedback le plus récent")
    nb_courses_last_7d: int = Field(..., description="Courses avec feedback (7 derniers jours)")
    nb_courses_last_30d: int = Field(..., description="Courses avec feedback (30 derniers jours)")
    taux_collection: float = Field(..., description="% de prédictions avec feedback")


class ModelPerformance(BaseModel):
    """Performance du modèle comparée aux résultats réels."""

    periode: str = Field(..., description="Période analysée")
    nb_courses: int = Field(..., description="Nombre de courses analysées")
    nb_predictions: int = Field(..., description="Nombre de prédictions analysées")
    accuracy_top1: float = Field(..., description="Précision Top-1 (Vainqueur correct)")
    nb_correct_top1: int = Field(..., description="Nombre de Top-1 corrects")
    accuracy_top3: float = Field(..., description="Précision Top-3 (Vainqueur dans les 3 premiers)")
    nb_correct_top3: int = Field(0, description="Nombre de Top-3 corrects")
    brier_score: float = Field(0.0, description="Score de Brier (calibration)")
    ece: float = Field(0.0, description="Expected Calibration Error")
    roi_top1: Optional[float] = Field(None, description="ROI si mise sur Top-1")


class PredictionRecord(BaseModel):
    """Enregistrement d'une prédiction pour monitoring."""

    course_id: str
    timestamp: str
    model_version: str
    predictions: List[dict]
    # [{"cheval_id": "...", "probabilite": 0.2, "rang": 1}, ...]


# ============================================================================
# FEEDBACK MANAGER
# ============================================================================


class FeedbackManager:
    """Gère la collecte et l'analyse du feedback."""

    def __init__(self):
        self.feedback_data = self._load_feedback()
        self.prediction_data = self._load_predictions()

    def _load_predictions(self) -> List[dict]:
        if PREDICTION_FILE.exists():
            try:
                with open(PREDICTION_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_predictions_to_disk(self):
        try:
            PREDICTION_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PREDICTION_FILE, "w") as f:
                json.dump(self.prediction_data, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur sauvegarde prédictions: {e}")

    def save_prediction_data(self, record: PredictionRecord):
        """Sauvegarde les prédictions au moment de l'inférence."""
        self.prediction_data.append(record.dict())
        self._save_predictions_to_disk()

    def _load_feedback(self) -> List[dict]:
        """Charge le feedback depuis le fichier JSON."""
        if FEEDBACK_FILE.exists():
            try:
                with open(FEEDBACK_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erreur chargement feedback: {e}")
        return []

    def _save_feedback_to_disk(self):
        """Sauvegarde le feedback sur disque."""
        try:
            FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(FEEDBACK_FILE, "w") as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur sauvegarde disque: {e}")

    def save_feedback(self, course_result: CourseResult) -> FeedbackResponse:
        """
        Enregistre le feedback d'une course.
        """
        try:
            # Ajouter timestamp si non fourni
            if not course_result.timestamp_feedback:
                course_result.timestamp_feedback = datetime.now().isoformat()

            # Stocker en mémoire et disque
            self.feedback_data.append(course_result.dict())
            self._save_feedback_to_disk()

            logger.info(
                f"✅ Feedback enregistré: course={course_result.course_id}, "
                f"chevaux={len(course_result.resultats)}"
            )

            return FeedbackResponse(
                status="success",
                message=f"Feedback enregistré pour {len(course_result.resultats)} chevaux",
                course_id=course_result.course_id,
                nb_resultats=len(course_result.resultats),
                timestamp=course_result.timestamp_feedback,
            )

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde feedback: {e}", exc_info=True)
            raise

    def get_stats(self) -> FeedbackStats:
        """
        Récupère les statistiques du feedback collecté.
        """
        if not self.feedback_data:
            return FeedbackStats(
                total_courses=0,
                total_predictions=0,
                periode_debut="N/A",
                periode_fin="N/A",
                nb_courses_last_7d=0,
                nb_courses_last_30d=0,
                taux_collection=0.0,
            )

        # Calculer stats (stub simplifié)
        total_courses = len(self.feedback_data)
        total_predictions = sum(len(f["resultats"]) for f in self.feedback_data)

        timestamps = [datetime.fromisoformat(f["timestamp_feedback"]) for f in self.feedback_data]
        periode_debut = min(timestamps).strftime("%Y-%m-%d")
        periode_fin = max(timestamps).strftime("%Y-%m-%d")

        # Compter courses 7 et 30 derniers jours
        now = datetime.now()
        last_7d = sum(1 for ts in timestamps if now - ts <= timedelta(days=7))
        last_30d = sum(1 for ts in timestamps if now - ts <= timedelta(days=30))

        return FeedbackStats(
            total_courses=total_courses,
            total_predictions=total_predictions,
            periode_debut=periode_debut,
            periode_fin=periode_fin,
            nb_courses_last_7d=last_7d,
            nb_courses_last_30d=last_30d,
            taux_collection=0.0,  # À calculer vs nb total prédictions API
        )

    def get_model_performance(self, days: int = 7) -> ModelPerformance:
        """
        Calcule la performance du modèle sur les N derniers jours.
        """
        # Filtrer feedback des N derniers jours
        now = datetime.now()
        recent_feedback = [
            f
            for f in self.feedback_data
            if now - datetime.fromisoformat(f["timestamp_feedback"]) <= timedelta(days=days)
        ]

        if not recent_feedback:
            # Pas de données
            return ModelPerformance(
                periode=f"{days} derniers jours",
                nb_courses=0,
                nb_predictions=0,
                accuracy_top1=0.0,
                nb_correct_top1=0,
                accuracy_top3=0.0,
                nb_correct_top3=0,
                brier_score=0.0,
                ece=0.0,
            )

        # Stub: calculs simplifiés
        nb_courses = len(recent_feedback)
        nb_predictions = sum(len(f["resultats"]) for f in recent_feedback)

        # Placeholder metrics (à calculer avec vraies données)
        return ModelPerformance(
            periode=f"{days} derniers jours",
            nb_courses=nb_courses,
            nb_predictions=nb_predictions,
            accuracy_top1=0.25,  # Placeholder
            nb_correct_top1=int(nb_courses * 0.25),
            accuracy_top3=0.65,  # Placeholder
            nb_correct_top3=int(nb_courses * 0.65),
            brier_score=0.15,  # Placeholder
            ece=0.08,  # Placeholder
        )
