import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from config.loader import CalibrationConfig
except ImportError:
    CalibrationConfig = None


class ModelManager:
    """Gère le chargement et l'utilisation du modèle ML (Champion XGBoost)."""

    def __init__(self, model_path: str = None, challenger_path: Optional[str] = None):
        if not model_path:
            model_path = os.getenv("MODEL_PATH", "/project/data/models/champion/xgboost_model.pkl")

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

        # A/B Testing support
        self.challenger_path = Path(challenger_path) if challenger_path else None
        self.challenger_model = None
        self.challenger_version = None
        self.ab_test_enabled = os.getenv("AB_TEST_ENABLED", "false").lower() == "true"
        self.challenger_traffic_percent = float(os.getenv("CHALLENGER_TRAFFIC_PERCENT", "10"))

        self.total_predictions = 0
        self.total_latency_ms = 0.0
        self.champion_predictions = 0
        self.challenger_predictions = 0

    def load_model(self) -> bool:
        """Charge le modèle champion."""
        try:
            if not self.model_path.exists():
                logger.error(f"❌ Modèle champion introuvable: {self.model_path}")
                return False

            logger.info(f"📦 Chargement modèle champion depuis {self.model_path}...")
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            # Charger les noms de features
            feature_names_path = self.model_dir / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, "r") as f:
                    self.feature_names = json.load(f)
                logger.info(f"   Features chargées: {len(self.feature_names)}")

            # Charger calibration
            calibration_dir = self.model_dir.parent.parent.parent / "calibration" / "champion"
            # Fallback path if model is mounted elsewhere
            if not calibration_dir.exists():
                calibration_dir = Path("/project/calibration/champion")

            platt_path = calibration_dir / "calibrator_platt.pkl"
            if platt_path.exists():
                try:
                    with open(platt_path, "rb") as f:
                        self.calibrator_platt = pickle.load(f)
                    logger.info("   Platt calibrator chargé")
                except Exception as e:
                    logger.warning(f"⚠️ Échec chargement Platt calibrator: {e}")

            temp_path = calibration_dir / "scaler_temperature.pkl"
            if temp_path.exists():
                try:
                    # Dirty hack for pickling created with config.loader
                    import sys

                    if "/project" not in sys.path:
                        sys.path.append("/project")

                    with open(temp_path, "rb") as f:
                        temp_data = pickle.load(f)
                    if isinstance(temp_data, dict) and "temperature" in temp_data:
                        self.temperature = float(temp_data["temperature"])
                    logger.info(f"   Temperature scaler chargé: T={self.temperature:.4f}")
                except Exception as e:
                    logger.warning(f"⚠️ Échec chargement temperature scaler: {e}")

            self.loaded_at = datetime.now()
            logger.info("✅ Modèle chargé avec succès")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}", exc_info=True)
            return False

    def prepare_features(self, partants: List[Any]) -> pd.DataFrame:
        """Convertit les features en DataFrame."""
        data = []
        for cheval in partants:
            # Support Pydantic or dict
            features = cheval.dict() if hasattr(cheval, "dict") else cheval
            data.append(features)

        df = pd.DataFrame(data)

        if self.feature_names:
            X = pd.DataFrame(0.0, index=range(len(partants)), columns=self.feature_names)
            for col in self.feature_names:
                if col in df.columns:
                    X[col] = df[col].values
            return X
        return df

    def predict(self, partants: List[Any]) -> tuple:
        start_time = datetime.now()

        try:
            X = self.prepare_features(partants)

            # XGBoost Booster uses DMatrix
            dmatrix = xgb.DMatrix(X.values.astype(np.float32), feature_names=self.feature_names)
            raw_probas = self.model.predict(dmatrix)

            # Calibration
            if self.calibration_enabled and self.calibrator_platt is not None:
                probas = self.calibrator_platt.predict_proba(raw_probas.reshape(-1, 1))[:, 1]
            else:
                probas = raw_probas

            predictions = []
            for i, proba in enumerate(probas):
                cheval = partants[i]
                # Support object or dict
                c_id = getattr(cheval, "cheval_id", cheval.get("cheval_id"))
                num = getattr(cheval, "numero_partant", cheval.get("numero_partant"))

                predictions.append(
                    {"cheval_id": c_id, "numero_partant": num, "probabilite_victoire": float(proba)}
                )

            # Rank
            predictions.sort(key=lambda x: x["probabilite_victoire"], reverse=True)
            for rank, pred in enumerate(predictions, 1):
                pred["rang_prediction"] = rank

            latency = (datetime.now() - start_time).total_seconds() * 1000
            return predictions, latency, self.model_version

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
