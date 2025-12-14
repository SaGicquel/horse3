import json
import os
import pickle
from dataclasses import dataclass
from typing import Any


@dataclass
class ChampionArtifacts:
    model_path: str
    scaler_path: str
    imputer_path: str
    metadata_path: str | None = None
    calibration_report_path: str | None = None
    platt_calibrator_path: str | None = None
    feature_names_path: str | None = None


class ChampionPredictor:
    """
    Charge et exécute le modèle XGBoost champion (avec imputer + scaler).

    Le backend web ne possède pas toutes les features "Phase ML" dans cheval_courses_seen,
    on reconstruit donc un sous-ensemble depuis les tables stats_* et l'historique, et on
    complète le reste avec des valeurs neutres (l'imputer gère les manquants).
    """

    def __init__(self, artifacts: ChampionArtifacts):
        self.artifacts = artifacts
        self._loaded = False
        self._model = None
        self._scaler = None
        self._imputer = None
        self._platt = None
        self._temperature: float = 1.0
        self._expected_n_features: int | None = None
        self._fallback_used = {"imputer": False, "scaler": False}
        self._feature_names: list[str] | None = None

    def load(self) -> None:
        if self._loaded:
            return

        # Imports lazy (dans Docker on aura sklearn/xgboost).
        import numpy as np  # noqa: F401

        for path in (
            self.artifacts.model_path,
            self.artifacts.scaler_path,
            self.artifacts.imputer_path,
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Champion artifact missing: {path}")

        with open(self.artifacts.imputer_path, "rb") as f:
            self._imputer = pickle.load(f)
        with open(self.artifacts.scaler_path, "rb") as f:
            self._scaler = pickle.load(f)
        with open(self.artifacts.model_path, "rb") as f:
            self._model = pickle.load(f)

        # Charger méta/calibration (optionnel mais recommandé)
        self._load_calibration()

        # Déduire le nombre de features attendu.
        self._expected_n_features = getattr(self._model, "n_features_in_", None) or getattr(
            self._scaler, "n_features_in_", None
        )

        # Certains artefacts (imputer) peuvent être incompatibles ou entraînés sur un sous-ensemble.
        if self._expected_n_features and getattr(self._imputer, "n_features_in_", None) not in (
            None,
            self._expected_n_features,
        ):
            self._imputer = None
            self._fallback_used["imputer"] = True

        # Déduire les features attendues.
        if hasattr(self._scaler, "feature_names_in_"):
            self._feature_names = list(self._scaler.feature_names_in_)
        elif hasattr(self._imputer, "feature_names_in_"):
            self._feature_names = list(self._imputer.feature_names_in_)
        elif hasattr(self._model, "feature_names_in_"):
            self._feature_names = list(self._model.feature_names_in_)
        else:
            self._feature_names = None

        # Fallback: feature_names.json (export du dataset d'entraînement) si dispo.
        if (
            not self._feature_names
            and self.artifacts.feature_names_path
            and os.path.exists(self.artifacts.feature_names_path)
        ):
            try:
                with open(self.artifacts.feature_names_path, encoding="utf-8") as f:
                    names = json.load(f)
                if isinstance(names, list) and all(isinstance(x, str) for x in names):
                    self._feature_names = names
            except Exception:
                pass

        self._loaded = True

    def _load_calibration(self) -> None:
        metadata_path = self.artifacts.metadata_path
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    meta = json.load(f) or {}
                calib = (meta.get("calibration_files") or {}) if isinstance(meta, dict) else {}
                base_dir = os.path.dirname(metadata_path)
                report_rel = calib.get("calibration_report")
                platt_rel = calib.get("platt_calibrator")
                if report_rel and not self.artifacts.calibration_report_path:
                    self.artifacts.calibration_report_path = os.path.normpath(
                        os.path.join(base_dir, report_rel)
                    )
                if platt_rel and not self.artifacts.platt_calibrator_path:
                    self.artifacts.platt_calibrator_path = os.path.normpath(
                        os.path.join(base_dir, platt_rel)
                    )
            except Exception:
                pass

        report_path = self.artifacts.calibration_report_path
        if report_path and os.path.exists(report_path):
            try:
                with open(report_path, encoding="utf-8") as f:
                    report = json.load(f) or {}
                t = report.get("temperature")
                if isinstance(t, (int, float)) and t > 0:
                    self._temperature = float(t)
            except Exception:
                pass

        platt_path = self.artifacts.platt_calibrator_path
        if platt_path and os.path.exists(platt_path):
            try:
                with open(platt_path, "rb") as f:
                    self._platt = pickle.load(f)
            except Exception:
                self._platt = None

    @property
    def feature_names(self) -> list[str] | None:
        return self._feature_names

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def platt_loaded(self) -> bool:
        return self._platt is not None

    @property
    def fallback_used(self) -> dict[str, bool]:
        return dict(self._fallback_used)

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[float]:
        """
        rows: list of feature dicts.
        returns: list of probabilities for class 1.
        """
        try:
            self.load()
        except Exception:
            # Si load échoue, retourner des probas uniformes
            return [1.0 / len(rows)] * len(rows) if rows else []

        import numpy as np
        import pandas as pd

        n_rows = len(rows) if rows else 1

        try:
            df = pd.DataFrame(rows)

            # Completer les colonnes manquantes si on connait le schéma.
            if self._feature_names:
                for col in self._feature_names:
                    if col not in df.columns:
                        df[col] = np.nan
                df = df[self._feature_names]

            # Convertir tout en numérique, les non-numériques deviennent NaN -> imputer
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            X = df.values.astype(np.float32)

            # Nettoyer les valeurs infinies et trop grandes AVANT imputer/scaler
            X = np.where(np.isinf(X), 0.0, X)
            X = np.where(np.isnan(X), 0.0, X)
            X = np.clip(X, -1e6, 1e6)
            X = X.astype(np.float32)

            if self._imputer is not None:
                try:
                    X = self._imputer.transform(X)
                except Exception:
                    self._fallback_used["imputer"] = True
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            if self._scaler is not None:
                try:
                    X = self._scaler.transform(X)
                except Exception:
                    self._fallback_used["scaler"] = True
                    try:
                        mean_ = getattr(self._scaler, "mean_", None)
                        scale_ = getattr(self._scaler, "scale_", None)
                        if mean_ is not None and scale_ is not None:
                            mean_ = np.asarray(mean_, dtype=float).reshape(1, -1)
                            scale_ = np.asarray(scale_, dtype=float).reshape(1, -1)
                            scale_ = np.where(scale_ == 0, 1.0, scale_)
                            X = (X - mean_) / scale_
                    except Exception:
                        pass

            # Nettoyer après scaler
            X = np.where(np.isinf(X), 0.0, X)
            X = np.where(np.isnan(X), 0.0, X)
            X = np.clip(X, -100, 100).astype(np.float32)

            # Vérification finale
            if not np.isfinite(X).all():
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            X = np.ascontiguousarray(X, dtype=np.float32)

            # Prédiction
            margins = None
            try:
                margins = self._model.predict(X, output_margin=True)
            except ValueError:
                try:
                    raw = self._model.predict_proba(X)[:, 1]
                    raw = np.clip(raw, 1e-12, 1 - 1e-12)
                    margins = np.log(raw / (1 - raw))
                except Exception:
                    margins = np.zeros(X.shape[0], dtype=np.float32)
            except Exception:
                margins = None

            if margins is None:
                raw = self._model.predict_proba(X)[:, 1]
                raw = np.clip(raw, 1e-12, 1 - 1e-12)
                margins = np.log(raw / (1 - raw))
            else:
                margins = np.asarray(margins).reshape(-1)

            # Temperature softmax
            t = max(1e-6, float(self._temperature or 1.0))
            scaled = margins / t
            scaled = scaled - np.max(scaled)
            exp_scaled = np.exp(scaled)
            p_norm = exp_scaled / (np.sum(exp_scaled) or 1.0)

            # Calibration Platt (optionnelle)
            p_cal = p_norm
            if self._platt is not None and hasattr(self._platt, "predict_proba"):
                p_clip = np.clip(p_norm, 1e-12, 1 - 1e-12)
                logits = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
                p_cal = self._platt.predict_proba(logits)[:, 1]
                s = float(np.sum(p_cal) or 1.0)
                p_cal = p_cal / s

            return [float(x) for x in p_cal]

        except Exception:
            # En cas d'erreur, retourner des probabilités uniformes
            return [1.0 / n_rows] * n_rows


def default_champion_artifacts() -> ChampionArtifacts:
    base = os.getenv("CHAMPION_DIR", "/project/data/models/champion")
    return ChampionArtifacts(
        model_path=os.path.join(base, "xgboost_model.pkl"),
        scaler_path=os.path.join(base, "feature_scaler.pkl"),
        imputer_path=os.path.join(base, "feature_imputer.pkl"),
        metadata_path=os.path.join(base, "metadata.json"),
        # Fallback direct (si metadata non dispo)
        calibration_report_path=os.getenv(
            "CHAMPION_CALIBRATION_REPORT", "/project/calibration/champion/calibration_report.json"
        ),
        platt_calibrator_path=os.getenv(
            "CHAMPION_PLATT_PATH", "/project/calibration/champion/calibrator_platt.pkl"
        ),
        feature_names_path=os.path.join(base, "feature_names.json"),
    )
