from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _safe_prob(x: float) -> float:
    return float(max(1e-12, min(1 - 1e-12, x)))


def _logit(p: float) -> float:
    p = _safe_prob(p)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    # stabilité pour |z| grand
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _normalize_discipline(discipline: str) -> str:
    d = (discipline or "").lower().strip()
    if "trot" in d or "attel" in d or "mont" in d:
        return "trot"
    if "obst" in d or "haie" in d or "steeple" in d or "cross" in d:
        return "obstacle"
    return "plat"


@dataclass
class PlaceModel:
    features: List[str]
    intercept: float
    coef: Dict[str, float]
    metadata: Dict[str, Any]

    def predict(self, runners: List[Dict[str, Any]]) -> List[float]:
        out: List[float] = []
        n = max(1, len(runners))
        denom = max(1, n - 1)
        for idx, r in enumerate(runners):
            p_place_h = float(r.get("p_place_harville") or 0.0)
            field_size = int(r.get("field_size") or n)
            rank_odds = int(r.get("rank_odds") or (idx + 1))
            rank_pct = float((rank_odds - 1) / denom) if denom > 0 else 0.0
            discipline = _normalize_discipline(str(r.get("discipline") or ""))
            feats = {
                "logit_p_place_harville": _logit(p_place_h),
                "log_field": math.log(max(4.0, float(field_size))),
                "rank_pct": rank_pct,
                "is_trot": 1.0 if discipline == "trot" else 0.0,
                "is_obstacle": 1.0 if discipline == "obstacle" else 0.0,
            }
            z = float(self.intercept)
            for f in self.features:
                z += float(self.coef.get(f, 0.0)) * float(feats.get(f, 0.0))
            out.append(float(max(0.0, min(0.999, _sigmoid(z)))))
        return out


def load_place_model() -> Optional[PlaceModel]:
    """
    Charge un modèle calibré p_place depuis un JSON.
    Priorité:
      1) PLACE_MODEL_PATH
      2) /project/config/place_model.json (docker)
      3) ./config/place_model.json (local)
    """
    path = os.getenv("PLACE_MODEL_PATH")
    candidates = [
        path,
        "/project/config/place_model.json",
        "/app/config/place_model.json",
        os.path.join(os.getcwd(), "config", "place_model.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "place_model.json"),
    ]
    chosen = None
    for c in candidates:
        if c and os.path.exists(c):
            chosen = c
            break
    if not chosen:
        return None

    with open(chosen, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    features = payload.get("features") or []
    intercept = float(payload.get("intercept") or 0.0)
    coef = payload.get("coef") or {}
    if not isinstance(features, list) or not isinstance(coef, dict):
        return None
    return PlaceModel(
        features=[str(x) for x in features],
        intercept=intercept,
        coef={str(k): float(v) for k, v in coef.items()},
        metadata={"path": chosen, "metrics": payload.get("metrics"), "model": payload.get("model")},
    )
