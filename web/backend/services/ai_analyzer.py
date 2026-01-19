"""
🧠 AI Analyzer Service - Service Backend pour l'Analyse IA
==========================================================
Endpoints FastAPI pour l'analyse IA des résultats de paris.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

# Ajouter les répertoires pour importer ai_results_analyzer
# En Docker: /project/ai_results_analyzer.py
# En local: ../../ai_results_analyzer.py
sys.path.insert(0, "/project")  # Docker mount point
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============================================================================
# MODÈLES PYDANTIC
# =============================================================================


class BetData(BaseModel):
    """Données d'un pari pour l'analyse."""

    name: str = Field(..., description="Nom du cheval")
    stake: float = Field(..., description="Mise en euros")
    odds: float = Field(..., description="Cote")
    result: str = Field(..., description="Résultat: WIN ou LOSE")
    returns: float = Field(0, description="Gains en euros")
    date: Optional[str] = None
    hippodrome: Optional[str] = None
    discipline: Optional[str] = None
    bet_type: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """Requête d'analyse de paris."""

    bets: List[BetData] = Field(..., description="Liste des paris à analyser")


class ModelMetrics(BaseModel):
    """Métriques du modèle pour les suggestions."""

    auc: float = Field(..., description="Area Under Curve")
    roi: float = Field(..., description="Return on Investment en %")
    win_rate: float = Field(..., description="Taux de réussite en %")
    brier_score: Optional[float] = None
    avg_kelly: Optional[float] = None


class SuggestRequest(BaseModel):
    """Requête de suggestions d'amélioration."""

    model_metrics: ModelMetrics
    recent_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    feature_importances: Optional[Dict[str, float]] = None


class AIAnalysisResponse(BaseModel):
    """Réponse d'analyse IA."""

    success: bool
    provider: Optional[str] = None
    analysis: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    timestamp: str
    error: Optional[str] = None


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(prefix="/ai", tags=["AI Analysis"])


def get_analyzer():
    """Dependency pour obtenir l'analyseur IA."""
    try:
        from ai_results_analyzer import AIResultsAnalyzer, AIConfig

        config = AIConfig.from_env()
        return AIResultsAnalyzer(config)
    except ImportError as e:
        logger.error(f"Failed to import AI analyzer: {e}")
        return None


@router.get("/status")
async def get_ai_status():
    """Vérifie le statut des APIs IA."""
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")

    analyzer = get_analyzer()

    return {
        "openai_configured": bool(openai_key),
        "gemini_configured": bool(gemini_key),
        "provider_available": analyzer.is_available() if analyzer else False,
        "active_provider": type(analyzer.provider).__name__
        if analyzer and analyzer.provider
        else None,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/analyze-results", response_model=AIAnalysisResponse)
async def analyze_results(request: AnalyzeRequest):
    """
    Analyse les résultats des paris avec l'IA.

    Retourne une analyse détaillée des patterns de succès/échec,
    des conditions favorables et des suggestions d'amélioration.
    """
    analyzer = get_analyzer()

    if not analyzer or not analyzer.is_available():
        return AIAnalysisResponse(
            success=False,
            error="No AI provider available. Configure OPENAI_API_KEY or GOOGLE_API_KEY.",
            timestamp=datetime.now().isoformat(),
        )

    # Convertir les BetData en dicts
    bets = [bet.dict() for bet in request.bets]

    try:
        result = analyzer.analyze_bets_performance(bets)

        return AIAnalysisResponse(
            success=result.get("success", False),
            provider=result.get("provider"),
            analysis=result.get("analysis"),
            stats=result.get("stats"),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
            error=result.get("error"),
        )
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return AIAnalysisResponse(success=False, error=str(e), timestamp=datetime.now().isoformat())


@router.post("/suggest-improvements")
async def suggest_improvements(request: SuggestRequest):
    """
    Suggère des améliorations pour le modèle de prédiction.
    """
    analyzer = get_analyzer()

    if not analyzer or not analyzer.is_available():
        raise HTTPException(
            status_code=503,
            detail="No AI provider available. Configure OPENAI_API_KEY or GOOGLE_API_KEY.",
        )

    try:
        result = analyzer.suggest_model_improvements(
            model_metrics=request.model_metrics.dict(),
            recent_predictions=request.recent_predictions,
            feature_importances=request.feature_importances,
        )

        return result
    except Exception as e:
        logger.error(f"AI suggestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-losing-streak")
async def analyze_losing_streak(request: AnalyzeRequest):
    """
    Analyse spécifique d'une série de paris perdants.
    """
    analyzer = get_analyzer()

    if not analyzer or not analyzer.is_available():
        raise HTTPException(status_code=503, detail="No AI provider available")

    # Filtrer uniquement les paris perdants
    losing_bets = [bet.dict() for bet in request.bets if bet.result == "LOSE"]

    if not losing_bets:
        return {"message": "No losing bets to analyze", "success": True}

    try:
        result = analyzer.analyze_losing_streak(losing_bets)
        return result
    except Exception as e:
        logger.error(f"AI losing streak analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick-insights")
async def get_quick_insights():
    """
    Retourne des insights rapides basés sur les paris récents en base.
    """
    analyzer = get_analyzer()

    if not analyzer or not analyzer.is_available():
        return {
            "available": False,
            "message": "Configure OPENAI_API_KEY or GOOGLE_API_KEY pour activer l'analyse IA",
        }

    # Dans une vraie implémentation, on récupérerait les paris de la BDD
    return {
        "available": True,
        "provider": type(analyzer.provider).__name__,
        "message": "AI analysis ready. Send bets data to /ai/analyze-results for insights.",
    }
