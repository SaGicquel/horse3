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


class PredictionRecord(BaseModel):
    """Enregistrement d'une prédiction pour monitoring."""

    course_id: str
    timestamp: str
    model_version: str
    predictions: List[dict]
    # [{"cheval_id": "...", "probabilite": 0.2, "rang": 1}, ...]


# ... (CourseResult and others)


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

    # ... (rest of methods)

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

        En production: Requêtes SQL sur table `feedback_results`.
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

        En production: JOIN entre `predictions` et `feedback_results`.
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
        # En production: requêtes SQL complexes pour joindre prédictions et résultats
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


# ============================================================================
# ENDPOINTS FASTAPI (À AJOUTER À api_prediction.py)
# ============================================================================

"""
# À ajouter dans api_prediction.py:

from api_feedback import (
    FeedbackManager, CourseResult, FeedbackResponse,
    FeedbackStats, ModelPerformance
)

# Initialiser gestionnaire feedback
feedback_manager = FeedbackManager()


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(course_result: CourseResult):
    '''
    Enregistrer le résultat réel d'une course pour améliorer le modèle.

    Args:
        course_result: CourseResult avec course_id et positions d'arrivée

    Returns:
        FeedbackResponse avec statut de l'enregistrement

    Raises:
        HTTPException 400: Données invalides
        HTTPException 500: Erreur interne
    '''
    try:
        response = feedback_manager.save_feedback(course_result)
        return response
    except Exception as e:
        logger.error(f"❌ Erreur feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/stats", response_model=FeedbackStats, tags=["Feedback"])
async def get_feedback_stats():
    '''
    Récupérer les statistiques du feedback collecté.

    Returns:
        FeedbackStats avec métriques de collection
    '''
    try:
        stats = feedback_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"❌ Erreur stats feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/model-performance", response_model=ModelPerformance, tags=["Feedback"])
async def get_model_performance(days: int = 7):
    '''
    Analyser la performance du modèle sur les N derniers jours.

    Args:
        days: Nombre de jours à analyser (défaut: 7)

    Returns:
        ModelPerformance avec métriques de performance réelles
    '''
    try:
        if days < 1 or days > 90:
            raise HTTPException(status_code=400, detail="days doit être entre 1 et 90")

        performance = feedback_manager.get_model_performance(days=days)
        return performance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur performance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
"""


# ============================================================================
# SCRIPT D'EXEMPLE - SOUMETTRE FEEDBACK
# ============================================================================

if __name__ == "__main__":
    import requests
    import json

    # Exemple: Soumettre feedback pour une course
    feedback = {
        "course_id": "VINCENNES_2025-11-13_R1C3",
        "date_course": "2025-11-13",
        "hippodrome": "VINCENNES",
        "resultats": [
            {"cheval_id": "CHEVAL_001", "numero_partant": 1, "position_arrivee": 3},
            {"cheval_id": "CHEVAL_002", "numero_partant": 2, "position_arrivee": 1},  # Gagnant réel
            {"cheval_id": "CHEVAL_003", "numero_partant": 3, "position_arrivee": 2},
            {"cheval_id": "CHEVAL_004", "numero_partant": 4, "position_arrivee": 5},
            {"cheval_id": "CHEVAL_005", "numero_partant": 5, "position_arrivee": 4},
        ],
    }

    # Envoyer feedback (nécessite API en cours d'exécution)
    try:
        response = requests.post(
            "http://localhost:8000/feedback",
            json=feedback,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        print("✅ Feedback envoyé:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Erreur: {e}")
