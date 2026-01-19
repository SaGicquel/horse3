"""
🏇 API de Prédiction de Courses Hippiques
==========================================

API REST FastAPI pour servir le modèle XGBoost Champion (ROC-AUC Test: 0.6189, Backtest ROI: 22.71%).

Endpoints:
- POST /predict : Prédire les probabilités de victoire pour une course
- GET /health : Vérifier l'état de l'API
- GET /metrics : Métriques Prometheus pour monitoring
- POST /feedback : Soumettre résultats réels (Phase 8 - Online Learning)
- GET /feedback/stats : Statistiques feedback collecté
- GET /feedback/model-performance : Performance modèle vs réalité

Auteur: Phase 7-8 - Production & Online Learning
Date: 2025-11-13
"""

import os
import sys
import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import json
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Intégration du router Races (Task 2)
try:
    from backend.routers import races

    RACES_ROUTER_AVAILABLE = True
except ImportError as e:
    # Fallback si le module backend n'est pas trouvé
    print(f"Warning: Could not import backend.routers.races: {e}")
    RACES_ROUTER_AVAILABLE = False

try:
    from config.loader import CalibrationConfig
except ImportError:
    CalibrationConfig = None

try:
    from api_phase9 import ModelServer as GNNServer

    GNN_AVAILABLE = True
except ImportError:
    GNNServer = None
    GNN_AVAILABLE = False

try:
    from ai_assistant import HorseRacingAssistant

    CHAT_AVAILABLE = True
except ImportError:
    HorseRacingAssistant = None
    CHAT_AVAILABLE = False

# Supervisor AI
try:
    from ai_supervisor import AiSupervisor, RaceContext, HorseAnalysis, SupervisorResult

    SUPERVISOR_AVAILABLE = True
except ImportError:
    AiSupervisor = None
    SUPERVISOR_AVAILABLE = False

# Configuration logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODÈLES PYDANTIC (Validation entrées/sorties)
# ============================================================================


class ChevalFeatures(BaseModel):
    """Features d'un cheval participant à une course."""

    # Identifiants
    cheval_id: str = Field(..., description="ID unique du cheval")
    jockey_id: Optional[int] = Field(None, description="ID du jockey (pour GNN)")
    entraineur_id: Optional[int] = Field(None, description="ID de l'entraîneur (pour GNN)")
    numero_partant: int = Field(..., ge=1, le=30, description="Numéro de départ (1-30)")

    # Features de forme (calculées sur historique)
    forme_5c: float = Field(
        default=0.0, ge=0, le=1, description="Forme sur 5 dernières courses (0-1)"
    )
    forme_10c: float = Field(
        default=0.0, ge=0, le=1, description="Forme sur 10 dernières courses (0-1)"
    )
    nb_courses_12m: int = Field(default=0, ge=0, description="Nombre de courses 12 derniers mois")
    nb_victoires_12m: int = Field(
        default=0, ge=0, description="Nombre de victoires 12 derniers mois"
    )
    nb_places_12m: int = Field(
        default=0, ge=0, description="Nombre de places (top 3) 12 derniers mois"
    )
    recence: float = Field(default=90.0, ge=0, description="Jours depuis dernière course")
    regularite: float = Field(default=0.0, ge=0, le=1, description="Régularité performances (0-1)")

    # Features d'aptitude
    aptitude_distance: float = Field(
        default=0.0, ge=0, le=1, description="Performance sur cette distance"
    )
    aptitude_piste: float = Field(
        default=0.0, ge=0, le=1, description="Performance sur ce type de piste"
    )
    aptitude_hippodrome: float = Field(
        default=0.0, ge=0, le=1, description="Performance sur cet hippodrome"
    )

    # Features jockey/entraîneur
    taux_victoires_jockey: float = Field(
        default=0.0, ge=0, le=1, description="Taux de victoire du jockey"
    )
    taux_places_jockey: float = Field(
        default=0.0, ge=0, le=1, description="Taux de places du jockey"
    )
    taux_victoires_entraineur: float = Field(
        default=0.0, ge=0, le=1, description="Taux de victoire entraîneur"
    )
    taux_places_entraineur: float = Field(
        default=0.0, ge=0, le=1, description="Taux de places entraîneur"
    )
    synergie_jockey_cheval: float = Field(
        default=0.0, ge=0, le=1, description="Synergie jockey-cheval"
    )
    synergie_entraineur_cheval: float = Field(
        default=0.0, ge=0, le=1, description="Synergie entraîneur-cheval"
    )

    # Features de course
    distance_norm: float = Field(default=0.0, ge=0, le=1, description="Distance normalisée (0-1)")
    niveau_moyen_concurrent: float = Field(
        default=0.0, ge=0, description="Niveau moyen adversaires"
    )
    nb_partants: int = Field(
        default=10, ge=2, le=30, description="Nombre de partants dans la course"
    )

    # Features marché (optionnelles - peuvent être absentes avant course)
    cote_turfbzh: Optional[float] = Field(None, ge=1.0, description="Cote TurfBZH (si disponible)")
    rang_cote_turfbzh: Optional[float] = Field(None, ge=1, description="Rang selon cote TurfBZH")
    cote_sp: Optional[float] = Field(None, ge=1.0, description="Cote StartingPrice PMU")
    rang_cote_sp: Optional[float] = Field(None, ge=1, description="Rang selon cote SP")
    prediction_ia_gagnant: Optional[float] = Field(
        None, ge=0, le=1, description="Prédiction IA du bookmaker"
    )
    elo_cheval: Optional[float] = Field(None, ge=0, description="Score ELO du cheval")
    ecart_cote_ia: Optional[float] = Field(None, description="Écart entre cote et prédiction IA")

    @validator(
        "forme_5c",
        "forme_10c",
        "aptitude_distance",
        "aptitude_piste",
        "aptitude_hippodrome",
        "taux_victoires_jockey",
        "taux_places_jockey",
        "synergie_jockey_cheval",
        pre=True,
    )
    def validate_probabilities(cls, v):
        """Valide que les probabilités sont entre 0 et 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError(f"Valeur doit être entre 0 et 1, reçu: {v}")
        return v


class CourseRequest(BaseModel):
    """Requête de prédiction pour une course complète."""

    course_id: str = Field(..., description="ID unique de la course")
    date_course: str = Field(..., description="Date de la course (YYYY-MM-DD)")
    hippodrome: str = Field(..., description="Nom de l'hippodrome")
    distance: int = Field(..., ge=800, le=8000, description="Distance en mètres")
    type_piste: str = Field(..., description="Type de piste (Plat, Obstacle, etc.)")
    partants: List[ChevalFeatures] = Field(
        ..., min_items=2, max_items=30, description="Liste des chevaux"
    )

    @validator("date_course")
    def validate_date(cls, v):
        """Valide le format de date."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Format date invalide, attendu: YYYY-MM-DD")
        return v

    @validator("partants")
    def validate_unique_numbers(cls, v):
        """Vérifie que les numéros de partants sont uniques."""
        numbers = [p.numero_partant for p in v]
        if len(numbers) != len(set(numbers)):
            raise ValueError("Les numéros de partants doivent être uniques")
        return v


class PredictionCheval(BaseModel):
    """Prédiction pour un cheval."""

    cheval_id: str
    numero_partant: int
    probabilite_victoire: float = Field(
        ..., ge=0, le=1, description="P(victoire) prédite par le modèle"
    )
    rang_prediction: int = Field(..., ge=1, description="Rang selon prédiction (1=favori)")
    confiance: str = Field(..., description="Niveau de confiance (Haute/Moyenne/Faible)")


class PredictionResponse(BaseModel):
    """Réponse de prédiction pour une course."""

    course_id: str
    timestamp: str
    model_version: str
    top_3: List[PredictionCheval] = Field(..., description="Top 3 des favoris selon le modèle")
    toutes_predictions: List[PredictionCheval] = Field(
        ..., description="Prédictions pour tous les partants"
    )
    latence_ms: float = Field(..., description="Temps de calcul en millisecondes")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    timestamp: str


class HealthResponse(BaseModel):
    """Réponse du healthcheck."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    total_predictions: int


# ============================================================================
# CLASSE GESTIONNAIRE DE MODÈLE
# ============================================================================


class ModelManager:
    """Gère le chargement et l'utilisation du modèle ML."""

    def __init__(self, model_path: str, challenger_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.model_dir = self.model_path.parent
        self.model = None
        self.feature_names = None
        self.feature_scaler = None
        self.feature_imputer = None
        self.model_version = "xgboost_champion_v1.0"
        self.loaded_at = None

        self.calibrator_platt = None
        self.temperature = 1.0
        self.calibration_enabled = True

        self.challenger_path = Path(challenger_path) if challenger_path else None
        self.challenger_model = None
        self.challenger_version = None
        self.ab_test_enabled = os.getenv("AB_TEST_ENABLED", "false").lower() == "true"
        self.challenger_traffic_percent = float(os.getenv("CHALLENGER_TRAFFIC_PERCENT", "10"))

        # Compteurs pour monitoring
        self.total_predictions = 0
        self.total_latency_ms = 0.0
        self.champion_predictions = 0
        self.challenger_predictions = 0

        # GNN Server (Phase 9)
        self.gnn_server = None

    def load_model(self) -> bool:
        """Charge le modèle champion (et challenger si A/B testing activé)."""
        try:
            # Chargement champion
            if not self.model_path.exists():
                logger.error(f"❌ Modèle champion introuvable: {self.model_path}")
                return False

            logger.info(f"📦 Chargement modèle champion depuis {self.model_path}...")
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            # Charger les noms de features depuis feature_names.json
            feature_names_path = self.model_dir / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, "r") as f:
                    self.feature_names = json.load(f)
                logger.info(f"   Features chargées: {len(self.feature_names)}")
            else:
                logger.warning(f"⚠️ feature_names.json introuvable: {feature_names_path}")

            scaler_path = self.model_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                try:
                    with open(scaler_path, "rb") as f:
                        self.feature_scaler = pickle.load(f)
                    logger.info("   Feature scaler chargé")
                except Exception as e:
                    logger.warning(f"⚠️ Échec chargement scaler: {e}")

            imputer_path = self.model_dir / "feature_imputer.pkl"
            if imputer_path.exists():
                try:
                    with open(imputer_path, "rb") as f:
                        self.feature_imputer = pickle.load(f)
                    logger.info("   Feature imputer chargé")
                except Exception as e:
                    logger.warning(f"⚠️ Échec chargement imputer: {e}")

            calibration_dir = Path("calibration/champion")
            platt_path = calibration_dir / "calibrator_platt.pkl"
            if platt_path.exists():
                try:
                    with open(platt_path, "rb") as f:
                        self.calibrator_platt = pickle.load(f)
                    logger.info("   Platt calibrator chargé")
                except Exception as e:
                    logger.warning(f"⚠️ Échec chargement Platt calibrator: {e}")

            temp_path = calibration_dir / "scaler_temperature.pkl"
            if temp_path.exists() and CalibrationConfig is not None:
                try:
                    with open(temp_path, "rb") as f:
                        temp_data = pickle.load(f)
                    if isinstance(temp_data, dict) and "temperature" in temp_data:
                        self.temperature = float(temp_data["temperature"])
                    logger.info(f"   Temperature scaler chargé: T={self.temperature:.4f}")
                except Exception as e:
                    logger.warning(f"⚠️ Échec chargement temperature scaler: {e}")
                    report_path = calibration_dir / "calibration_report.json"
                    if report_path.exists():
                        try:
                            with open(report_path, "r") as f:
                                report = json.load(f)
                            if "temperature" in report:
                                self.temperature = float(report["temperature"])
                                logger.info(f"   Temperature from report: T={self.temperature:.4f}")
                        except Exception:
                            pass

            self.loaded_at = datetime.now()
            logger.info(f"✅ Modèle champion chargé: {self.model_version}")
            logger.info(f"   Type: {type(self.model).__name__}")

            # Chargement challenger si A/B testing activé
            if self.ab_test_enabled and GNN_AVAILABLE:
                logger.info("🔬 Tentative chargement GNN (Challenger Phase 9)...")
                try:
                    self.gnn_server = GNNServer()
                    self.gnn_server.load_resources()
                    self.challenger_model = self.gnn_server
                    self.challenger_version = "gnn_v1"
                    logger.info(f"✅ Modèle GNN chargé: {self.challenger_version}")
                except Exception as e:
                    logger.warning(f"⚠️  Échec chargement GNN: {e}")
                    if self.challenger_path and self.challenger_path.exists():
                        logger.info(
                            f"🔬 Chargement modèle challenger legacy depuis {self.challenger_path}..."
                        )
                        try:
                            with open(self.challenger_path, "rb") as f:
                                self.challenger_model = pickle.load(f)
                            self.challenger_version = "challenger_legacy"
                            logger.info("✅ Modèle challenger legacy chargé")
                        except Exception as ex:
                            logger.warning(f"⚠️  Échec chargement challenger legacy: {ex}")
                            self.challenger_model = None
                    else:
                        self.challenger_model = None

            return True

        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}", exc_info=True)
            return False

    def prepare_features(self, partants: List[ChevalFeatures]) -> pd.DataFrame:
        """Convertit les features Pydantic en DataFrame pour le modèle."""

        data = []
        for cheval in partants:
            # Convertir Pydantic model en dict
            features = cheval.dict()
            data.append(features)

        df = pd.DataFrame(data)

        # Si on a les feature names, construire le DataFrame avec les bonnes colonnes
        if self.feature_names:
            # Créer un DataFrame vide avec les colonnes attendues
            X = pd.DataFrame(0.0, index=range(len(partants)), columns=self.feature_names)

            # Remplir les colonnes disponibles
            for col in self.feature_names:
                if col in df.columns:
                    X[col] = df[col].values

            return X
        else:
            return df

    def select_model_ab_test(self) -> tuple:
        """
        Sélectionne le modèle à utiliser selon stratégie A/B testing.

        Returns:
            (model, model_version) tuple
        """
        # Si A/B testing désactivé ou pas de challenger, utiliser champion
        if not self.ab_test_enabled or self.challenger_model is None:
            return self.model, self.model_version

        # Tirer au sort selon pourcentage configuré
        random_value = np.random.random() * 100

        if random_value < self.challenger_traffic_percent:
            # Utiliser challenger
            self.challenger_predictions += 1
            logger.info(f"🔬 A/B Test: Utilisation challenger ({self.challenger_version})")
            return self.challenger_model, self.challenger_version
        else:
            # Utiliser champion
            self.champion_predictions += 1
            return self.model, self.model_version

    def predict(self, partants: List[ChevalFeatures]) -> tuple:
        """
        Prédit les probabilités de victoire pour une liste de partants.
        Support A/B testing champion vs challenger (Phase 8).

        Returns:
            (predictions, latency_ms, model_version) tuple
        """
        start_time = time.time()

        try:
            # Préparer features
            X = self.prepare_features(partants)

            # Sélectionner modèle (A/B testing)
            selected_model, selected_version = self.select_model_ab_test()

            # Logique spécifique pour le GNN (Phase 9)
            if selected_version == "gnn_v1":
                predictions = []
                for cheval in partants:
                    # Vérifier présence des IDs
                    if cheval.jockey_id is None or cheval.entraineur_id is None:
                        # Fallback silencieux sur le champion si IDs manquants
                        logger.warning(
                            f"⚠️ IDs manquants pour GNN (cheval {cheval.cheval_id}), fallback Champion"
                        )
                        selected_model = self.model
                        selected_version = self.model_version
                        break  # Sortir de la boucle et utiliser logique standard

                    # Conversion str -> int si nécessaire (le modèle attend des int)
                    try:
                        h_id = int(cheval.cheval_id)
                        j_id = int(cheval.jockey_id)
                        e_id = int(cheval.entraineur_id)

                        score, status = selected_model.predict(h_id, j_id, e_id)

                        if score is None:
                            score = 0.0  # Ou fallback

                        predictions.append(
                            {
                                "cheval_id": cheval.cheval_id,
                                "numero_partant": cheval.numero_partant,
                                "probabilite_victoire": float(score),
                            }
                        )
                    except ValueError:
                        logger.error(f"❌ IDs invalides pour GNN: {cheval.cheval_id}")
                        selected_model = self.model
                        selected_version = self.model_version
                        break

                # Si on a réussi à prédire avec le GNN pour tous les partants
                if selected_version == "gnn_v1":
                    # Normalisation Softmax (optionnel mais recommandé pour avoir des probas qui somment à 1)
                    # Le GNN sort des logits ou des sigmoids indépendants.
                    # Ici on a des sigmoids (0-1). On peut les normaliser.
                    scores = np.array([p["probabilite_victoire"] for p in predictions])
                    if scores.sum() > 0:
                        scores = scores / scores.sum()
                    for i, p in enumerate(predictions):
                        p["probabilite_victoire"] = float(scores[i])

            # Logique Standard (Champion XGBoost Booster)
            if selected_version != "gnn_v1":
                dmatrix = xgb.DMatrix(X.values.astype(np.float32), feature_names=self.feature_names)
                raw_probas = selected_model.predict(dmatrix)

                if self.calibration_enabled and self.calibrator_platt is not None:
                    probas = self.calibrator_platt.predict_proba(raw_probas.reshape(-1, 1))[:, 1]
                else:
                    probas = raw_probas

                predictions = []
                for i, (cheval, proba) in enumerate(zip(partants, probas)):
                    predictions.append(
                        {
                            "cheval_id": cheval.cheval_id,
                            "numero_partant": cheval.numero_partant,
                            "probabilite_victoire": float(proba),
                        }
                    )

            # Trier par probabilité décroissante
            predictions.sort(key=lambda x: x["probabilite_victoire"], reverse=True)

            # Ajouter rangs
            for rank, pred in enumerate(predictions, 1):
                pred["rang_prediction"] = rank

                # Niveau de confiance basé sur probabilité
                if pred["probabilite_victoire"] >= 0.3:
                    pred["confiance"] = "Haute"
                elif pred["probabilite_victoire"] >= 0.15:
                    pred["confiance"] = "Moyenne"
                else:
                    pred["confiance"] = "Faible"

            # Mise à jour métriques
            latency_ms = (time.time() - start_time) * 1000
            self.total_predictions += len(partants)
            self.total_latency_ms += latency_ms

            logger.info(
                f"✅ Prédiction réussie: {len(partants)} chevaux en {latency_ms:.1f}ms (modèle: {selected_version})"
            )

            return predictions, latency_ms, selected_version

        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

    def get_uptime(self) -> float:
        """Retourne le temps d'activité en secondes."""
        if self.loaded_at:
            return (datetime.now() - self.loaded_at).total_seconds()
        return 0.0

    def get_avg_latency(self) -> float:
        """Retourne la latence moyenne en millisecondes."""
        if self.total_predictions > 0:
            return self.total_latency_ms / self.total_predictions
        return 0.0

    def get_avg_latency(self) -> float:
        """Retourne la latence moyenne en millisecondes."""
        if self.total_predictions > 0:
            return self.total_latency_ms / self.total_predictions
        return 0.0


# ============================================================================
# INITIALISATION FASTAPI
# ============================================================================

# Créer l'application
app = FastAPI(
    title="🏇 API Prédiction Courses Hippiques",
    description="API pour prédire les probabilités de victoire avec Stacking Ensemble (ROC-AUC Test: 0.7009)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (permettre appels depuis frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Races router if available
if RACES_ROUTER_AVAILABLE:
    app.include_router(races.router)
    logger.info("✅ Router /races intégré")

# Gestionnaire de modèle (global)
model_manager: Optional[ModelManager] = None

supervisor: Optional["AiSupervisor"] = None

# Gestionnaire de feedback (Phase 8 - Online Learning)
from api_feedback import (
    FeedbackManager,
    CourseResult,
    FeedbackResponse,
    FeedbackStats,
    ModelPerformance,
    PredictionRecord,
)

feedback_manager = FeedbackManager()

# Assistant IA (Chat)
chat_assistant: Optional[HorseRacingAssistant] = None


@app.on_event("startup")
async def startup_event():
    """Événement de démarrage: charge le modèle champion (et challenger si A/B testing)."""
    global model_manager, chat_assistant, supervisor

    logger.info("=" * 80)
    logger.info("🚀 DÉMARRAGE API PRÉDICTION")
    logger.info("=" * 80)

    if CHAT_AVAILABLE:
        chat_assistant = HorseRacingAssistant()
        logger.info("🤖 Assistant IA initialisé")
    else:
        logger.warning("⚠️ Assistant IA non disponible (dépendances manquantes)")

    if SUPERVISOR_AVAILABLE:
        try:
            supervisor = AiSupervisor()
            logger.info("🧠 Supervisor IA initialisé")
        except Exception as e:
            logger.error(f"❌ Échec initialisation Supervisor: {e}")
            supervisor = None
    else:
        logger.warning("⚠️ Supervisor IA non disponible (dépendances manquantes)")

    champion_path = os.getenv("MODEL_PATH", "data/models/champion/xgboost_model.pkl")
    challenger_path = os.getenv("CHALLENGER_MODEL_PATH", "data/models/challenger/model.pkl")

    model_manager = ModelManager(champion_path, challenger_path)

    if not model_manager.load_model():
        logger.error("❌ ÉCHEC CHARGEMENT MODÈLE - API en mode dégradé")
    else:
        logger.info("✅ API prête à recevoir des requêtes")

    logger.info("=" * 80)


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/", tags=["Info"])
async def root():
    """Page d'accueil de l'API."""
    return {
        "message": "🏇 API Prédiction Courses Hippiques",
        "version": "1.0.0",
        "model": "Stacking Ensemble (RF + XGBoost + LightGBM)",
        "performance": "ROC-AUC Test: 0.7009",
        "endpoints": {
            "predict": "POST /predict - Prédire une course",
            "health": "GET /health - État de l'API",
            "metrics": "GET /metrics - Métriques monitoring",
            "docs": "GET /docs - Documentation interactive",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Healthcheck pour vérifier l'état de l'API.
    Utilisé par Kubernetes/Docker pour liveness/readiness probes.
    """
    if model_manager is None or model_manager.model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "model_version": "N/A",
                "uptime_seconds": 0.0,
                "total_predictions": 0,
            },
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=model_manager.model_version,
        uptime_seconds=model_manager.get_uptime(),
        total_predictions=model_manager.total_predictions,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Métriques Prometheus pour monitoring.
    Format: nom_metrique{label="value"} valeur
    """
    if model_manager is None:
        return ""

    metrics_text = f"""# HELP predictions_total Nombre total de prédictions effectuées
# TYPE predictions_total counter
predictions_total {model_manager.total_predictions}

# HELP predictions_by_model Nombre de prédictions par modèle (A/B testing)
# TYPE predictions_by_model counter
predictions_by_model{{model_version="champion"}} {model_manager.champion_predictions}
predictions_by_model{{model_version="challenger"}} {model_manager.challenger_predictions}

# HELP prediction_latency_ms Latence moyenne de prédiction en millisecondes
# TYPE prediction_latency_ms gauge
prediction_latency_ms {model_manager.get_avg_latency():.2f}

# HELP api_uptime_seconds Temps d'activité de l'API en secondes
# TYPE api_uptime_seconds gauge
api_uptime_seconds {model_manager.get_uptime():.1f}

# HELP model_loaded Modèle chargé (1) ou non (0)
# TYPE model_loaded gauge
model_loaded {1 if model_manager.model else 0}

# HELP ab_test_enabled A/B testing activé (1) ou non (0)
# TYPE ab_test_enabled gauge
ab_test_enabled {1 if model_manager.ab_test_enabled and model_manager.challenger_model else 0}

# HELP challenger_traffic_percent Pourcentage de traffic vers challenger
# TYPE challenger_traffic_percent gauge
challenger_traffic_percent {model_manager.challenger_traffic_percent if model_manager.ab_test_enabled else 0}
"""

    return metrics_text


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_course(request: CourseRequest):
    """
    Prédit les probabilités de victoire pour tous les partants d'une course.

    Args:
        request: CourseRequest avec course_id, date, hippodrome et liste des partants

    Returns:
        PredictionResponse avec top 3 + toutes prédictions + métadonnées

    Raises:
        HTTPException 503: Modèle non chargé
        HTTPException 400: Requête invalide
        HTTPException 500: Erreur interne
    """
    # Vérifier que le modèle est chargé
    if model_manager is None or model_manager.model is None:
        raise HTTPException(
            status_code=503, detail="Modèle non chargé. Veuillez contacter l'administrateur."
        )

    # Log de la requête
    logger.info(
        f"📥 Nouvelle prédiction: course={request.course_id}, "
        f"hippodrome={request.hippodrome}, partants={len(request.partants)}"
    )

    try:
        # Prédire (avec A/B testing si activé)
        predictions, latency_ms, model_version = model_manager.predict(request.partants)

        # Convertir en Pydantic models
        all_predictions = [PredictionCheval(**pred) for pred in predictions]
        top_3 = all_predictions[:3]

        # Créer réponse
        response = PredictionResponse(
            course_id=request.course_id,
            timestamp=datetime.now().isoformat(),
            model_version=model_version,  # Version du modèle utilisé (champion ou challenger)
            top_3=top_3,
            toutes_predictions=all_predictions,
            latence_ms=latency_ms,
        )

        logger.info(
            f"✅ Prédiction envoyée: top_3=[{', '.join(str(p.numero_partant) for p in top_3)}], "
            f"latence={latency_ms:.1f}ms"
        )

        # Sauvegarder pour monitoring
        try:
            feedback_manager.save_prediction_data(
                PredictionRecord(
                    course_id=request.course_id,
                    timestamp=response.timestamp,
                    model_version=model_version,
                    predictions=[
                        {
                            "cheval_id": p.cheval_id,
                            "numero_partant": p.numero_partant,
                            "probabilite": p.probabilite_victoire,
                            "rang": p.rang_prediction,
                        }
                        for p in all_predictions
                    ],
                )
            )
        except Exception as e:
            logger.warning(f"⚠️ Échec sauvegarde monitoring: {e}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur prédiction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")


class AnalysisResponse(BaseModel):
    """Réponse de l'analyse superviseur."""

    course_id: str
    timestamp: str
    analysis: str
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_score: float
    provider: str


@app.post("/analyze", response_model=AnalysisResponse, tags=["Superviseur"])
async def analyze_course(request: CourseRequest):
    """
    🔍 Analyse une course avec le Superviseur IA.

    Combine les prédictions ML avec une analyse qualitative (LLM) pour:
    1. Valider les favoris
    2. Détecter des anomalies (ex: cheval bien classé mais forme douteuse)
    3. Générer des recommandations textuelles

    Utilise OpenAI ou Gemini si configurés, sinon fallback sur des règles métier.
    """
    if supervisor is None:
        raise HTTPException(
            status_code=503,
            detail="Superviseur non initialisé (dépendances manquantes ou erreur init)",
        )

    if model_manager is None or model_manager.model is None:
        raise HTTPException(status_code=503, detail="Modèle ML non chargé")

    try:
        # 1. Obtenir les prédictions ML
        predictions, _, _ = model_manager.predict(request.partants)

        # 2. Préparer le contexte pour le superviseur
        race_context = RaceContext(
            course_id=request.course_id,
            date=request.date_course,
            hippodrome=request.hippodrome,
            distance=request.distance,
            discipline=request.type_piste,
            nombre_partants=len(request.partants),
        )

        # Mapper les partants et prédictions vers HorseAnalysis
        horses_analysis = []
        pred_map = {p["numero_partant"]: p for p in predictions}

        for p in request.partants:
            pred = pred_map.get(p.numero_partant)
            if pred:
                horses_analysis.append(
                    HorseAnalysis(
                        cheval_id=p.cheval_id,
                        nom=f"Cheval {p.numero_partant}",  # Nom non dispo dans CourseRequest actuel, on met un placeholder
                        numero=p.numero_partant,
                        cote_sp=p.cote_sp or 0.0,
                        prob_model=pred["probabilite_victoire"],
                        rang_model=pred["rang_prediction"],
                        forme_5c=p.forme_5c,
                        forme_10c=p.forme_10c,
                        nb_courses_12m=p.nb_courses_12m,
                        nb_victoires_12m=p.nb_victoires_12m,
                        taux_victoires_jockey=p.taux_victoires_jockey,
                    )
                )

        # 3. Lancer l'analyse
        result = supervisor.analyze(race_context, horses_analysis)

        return AnalysisResponse(
            course_id=result.course_id,
            timestamp=result.timestamp,
            analysis=result.analysis,
            anomalies=result.anomalies,
            recommendations=result.recommendations,
            confidence_score=result.confidence_score,
            provider=result.provider or "unknown",
        )

    except Exception as e:
        logger.error(f"❌ Erreur analyse: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur analyse: {str(e)}")


# ============================================================================
# ENDPOINTS - PHASE 8 : ONLINE LEARNING & FEEDBACK
# ============================================================================


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(course_result: CourseResult):
    """
    📝 Enregistrer le résultat réel d'une course pour améliorer le modèle.

    Permet de collecter les positions d'arrivée réelles et comparer avec les prédictions
    pour retrainer automatiquement le modèle (online learning).

    Args:
        course_result: CourseResult avec course_id et positions d'arrivée

    Returns:
        FeedbackResponse avec statut de l'enregistrement

    Raises:
        HTTPException 400: Données invalides
        HTTPException 500: Erreur interne

    Exemple:
    ```json
    {
        "course_id": "VINCENNES_2025-11-13_R1C3",
        "date_course": "2025-11-13",
        "hippodrome": "VINCENNES",
        "resultats": [
            {"cheval_id": "CHEVAL_001", "numero_partant": 1, "position_arrivee": 3},
            {"cheval_id": "CHEVAL_002", "numero_partant": 2, "position_arrivee": 1}
        ]
    }
    ```
    """
    try:
        response = feedback_manager.save_feedback(course_result)

        logger.info(
            f"📝 Feedback enregistré: course={course_result.course_id}, "
            f"nb_chevaux={len(course_result.resultats)}"
        )

        return response

    except ValueError as e:
        # Erreur de validation
        logger.warning(f"⚠️ Feedback invalide: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Erreur feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement: {str(e)}")


@app.get("/feedback/stats", response_model=FeedbackStats, tags=["Feedback"])
async def get_feedback_stats():
    """
    📊 Récupérer les statistiques du feedback collecté.

    Affiche le nombre de courses et prédictions avec feedback, la période couverte,
    et le taux de collection.

    Returns:
        FeedbackStats avec métriques de collection

    Exemple de réponse:
    ```json
    {
        "total_courses": 150,
        "total_predictions": 1234,
        "periode_debut": "2025-10-01",
        "periode_fin": "2025-11-13",
        "nb_courses_last_7d": 25,
        "nb_courses_last_30d": 89,
        "taux_collection": 0.65
    }
    ```
    """
    try:
        stats = feedback_manager.get_stats()

        logger.info(
            f"📊 Stats feedback récupérées: courses={stats.total_courses}, "
            f"predictions={stats.total_predictions}"
        )

        return stats

    except Exception as e:
        logger.error(f"❌ Erreur stats feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération: {str(e)}")


@app.get("/feedback/model-performance", response_model=ModelPerformance, tags=["Feedback"])
async def get_model_performance(days: int = 7):
    """
    🎯 Analyser la performance du modèle sur les N derniers jours.

    Compare les prédictions du modèle avec les résultats réels pour calculer:
    - Accuracy top 1 (% gagnants prédits correctement)
    - Accuracy top 3 (% podiums prédits correctement)
    - Brier Score (calibration des probabilités)
    - Expected Calibration Error (ECE)

    Args:
        days: Nombre de jours à analyser (défaut: 7, min: 1, max: 90)

    Returns:
        ModelPerformance avec métriques de performance réelles

    Raises:
        HTTPException 400: Paramètre days invalide
        HTTPException 500: Erreur interne

    Exemple:
    ```
    GET /feedback/model-performance?days=30
    ```

    Réponse:
    ```json
    {
        "periode": "30 derniers jours",
        "nb_courses": 120,
        "nb_predictions": 987,
        "accuracy_top1": 0.28,
        "nb_correct_top1": 34,
        "accuracy_top3": 0.67,
        "nb_correct_top3": 80,
        "brier_score": 0.142,
        "ece": 0.065
    }
    ```
    """
    try:
        # Validation
        if days < 1 or days > 90:
            raise HTTPException(
                status_code=400, detail="Le paramètre 'days' doit être entre 1 et 90"
            )

        performance = feedback_manager.get_model_performance(days=days)

        logger.info(
            f"🎯 Performance calculée: période={days}j, courses={performance.nb_courses}, "
            f"accuracy_top1={performance.accuracy_top1:.2%}"
        )

        return performance

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur performance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse, tags=["Assistant IA"])
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint pour l'assistant IA.
    Traite le message de l'utilisateur et retourne une réponse contextuelle.
    """
    try:
        if not chat_assistant:
            raise HTTPException(status_code=503, detail="Assistant IA non initialisé")

        # Convertir les messages Pydantic en dict pour l'assistant
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history]

        response_text = chat_assistant.process_message(request.message, history_dicts)

        return ChatResponse(response=response_text, timestamp=datetime.now().isoformat())
    except Exception as e:
        logger.error(f"❌ Erreur chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur assistant: {str(e)}")


# ============================================================================
# MAIN
# ============================================================================


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API de prédiction courses hippiques")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (défaut: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (défaut: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload en dev")
    parser.add_argument("--model-path", type=str, help="Chemin vers le modèle .pkl")

    args = parser.parse_args()

    # Définir MODEL_PATH si fourni
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path

    # Lancer serveur
    uvicorn.run(
        "api_prediction:app", host=args.host, port=args.port, reload=args.reload, log_level="info"
    )
