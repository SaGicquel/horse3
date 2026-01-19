from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from services.ai_supervisor import AiSupervisor, RaceContext, HorseAnalysis
from services.betting_manager import BettingManager
from services.prediction_service import ModelManager
from services.api_feedback import FeedbackManager, PredictionRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyze", tags=["Superviseur"])

# Global services
supervisor: Optional[AiSupervisor] = None
betting_manager: Optional[BettingManager] = None
model_manager: Optional[ModelManager] = None
feedback_manager: Optional[FeedbackManager] = None


def init_analysis_services():
    """Initialise les services (appelé au démarrage)."""
    global supervisor, betting_manager, model_manager, feedback_manager

    try:
        model_manager = ModelManager()
        if model_manager.load_model():
            logger.info("✅ ModelManager initialized")
        else:
            logger.warning("⚠️ ModelManager failed to load model")

        supervisor = AiSupervisor()
        logger.info("✅ AiSupervisor initialized")

        betting_manager = BettingManager(strategy="balanced")
        logger.info("✅ BettingManager initialized")

        feedback_manager = FeedbackManager()
        logger.info("✅ FeedbackManager initialized")

    except Exception as e:
        logger.error(f"❌ Failed to init analysis services: {e}")


# Models
class PartantRequest(BaseModel):
    cheval_id: str
    numero_partant: int
    nom: str
    cote_sp: Optional[float] = 0.0
    forme_5c: Optional[float] = 0.5
    forme_10c: Optional[float] = 0.5
    nb_courses_12m: Optional[int] = 0
    nb_victoires_12m: Optional[int] = 0
    taux_victoires_jockey: Optional[float] = 0.0

    # Allow extra fields
    class Config:
        extra = "ignore"


class AnalysisRequest(BaseModel):
    course_id: str
    date_course: str
    hippodrome: str
    distance: int
    type_piste: str
    partants: List[PartantRequest]


class AnalysisResponse(BaseModel):
    course_id: str
    timestamp: str
    analysis: str
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_score: float
    provider: str
    betting_suggestions: List[Dict[str, Any]]


@router.post("", response_model=AnalysisResponse)
async def analyze_race(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyse complète : ML + Superviseur + Betting.
    """
    if not model_manager or not model_manager.model:
        # Try lazy init
        init_analysis_services()
        if not model_manager or not model_manager.model:
            raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        # 1. Prediction ML
        predictions, _, model_version = model_manager.predict(request.partants)

        # Save prediction feedback
        try:
            background_tasks.add_task(
                feedback_manager.save_prediction_data,
                PredictionRecord(
                    course_id=request.course_id,
                    timestamp=datetime.now().isoformat(),
                    model_version=model_version,
                    predictions=predictions,
                ),
            )
        except Exception:
            pass

        # 2. Supervisor Context
        race_context = RaceContext(
            course_id=request.course_id,
            date=request.date_course,
            hippodrome=request.hippodrome,
            distance=request.distance,
            discipline=request.type_piste,
            nombre_partants=len(request.partants),
        )

        # Map predictions to analysis objects
        pred_map = {p["numero_partant"]: p for p in predictions}
        horses_analysis = []

        for p in request.partants:
            pred = pred_map.get(p.numero_partant)
            if pred:
                horses_analysis.append(
                    HorseAnalysis(
                        cheval_id=p.cheval_id,
                        nom=p.nom,
                        numero=p.numero_partant,
                        cote_sp=p.cote_sp or 0.0,
                        prob_model=pred["probabilite_victoire"],
                        rang_model=pred["rang_prediction"],
                        forme_5c=p.forme_5c or 0.5,
                        nb_courses_12m=p.nb_courses_12m,
                        nb_victoires_12m=p.nb_victoires_12m,
                    )
                )

        # 3. Run Supervisor
        result = supervisor.analyze(race_context, horses_analysis)

        # 4. Betting Strategy
        # Use supervisor confidence to modulate stakes
        bets = betting_manager.calculate_stakes(
            predictions, confidence_score=result.confidence_score
        )

        # Convert bets to dict for response
        bet_dicts = [
            {
                "cheval_id": b.cheval_id,
                "numero": b.numero,
                "nom": b.nom,
                "mise_conseillee": b.mise_conseillee,
                "pourcentage_bankroll": b.pourcentage_bankroll,
                "value_edge": b.value_edge,
                "kelly_fraction": b.kelly_fraction,
            }
            for b in bets
        ]

        return AnalysisResponse(
            course_id=result.course_id,
            timestamp=result.timestamp,
            analysis=result.analysis,
            anomalies=result.anomalies,
            recommendations=result.recommendations,
            confidence_score=result.confidence_score,
            provider=result.provider or "unknown",
            betting_suggestions=bet_dicts,
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
