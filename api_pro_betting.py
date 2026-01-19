#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏇 API PRO BETTING - Endpoint pour analyse calibrée
===================================================

Expose l'analyseur pro via API FastAPI.
Toutes les sorties sont en JSON strict.

Endpoints:
- GET /healthz                  -> Health check simple
- GET /calibration/health       -> Métriques de calibration
- POST /portfolio               -> Optimisation de portefeuille
- GET /analyze/{race_key}       -> Analyse d'une course
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging

from pro_betting_analyzer import ProBettingAnalyzer
from db_connection import get_connection

# Logger JSON pour les paris
bet_logger = logging.getLogger("bets")
bet_logger.setLevel(logging.INFO)

# Import du loader d'artefacts (source de vérité)
try:
    from calibration.artifacts_loader import (
        load_calibration_state,
        CalibrationState,
        warn_if_mismatch,
        log_calibration_init,
    )

    ARTIFACTS_LOADER_AVAILABLE = True
except ImportError:
    ARTIFACTS_LOADER_AVAILABLE = False

# =============================================================================
# MODÈLES PYDANTIC
# =============================================================================


class RunnerOutput(BaseModel):
    """Sortie par partant"""

    numero: int
    nom: str
    p_win: Optional[float] = Field(None, description="Probabilité victoire (somme=1)")
    p_place: Optional[float] = Field(None, description="Probabilité placé (top 3)")
    fair_odds: Optional[float] = Field(None, description="Cote juste = 1/p_win")
    market_odds: Optional[float] = Field(None, description="Cote marché")
    value_pct: Optional[float] = Field(None, description="Value % (+ = bon)")
    kelly_fraction: Optional[float] = Field(None, description="Kelly fractionnaire")
    rationale: List[str] = Field(default_factory=list, description="2-3 puces max")


class RaceOutput(BaseModel):
    """Sortie par course"""

    race_id: str
    timestamp: str
    hippodrome: str
    distance_m: int
    discipline: str
    nb_partants: int
    model_version: str
    runners: List[RunnerOutput]
    run_notes: List[str] = Field(default_factory=list, description="Alertes/warnings")


class BatchRequest(BaseModel):
    """Requête pour plusieurs courses"""

    race_keys: List[str]


class BetCandidate(BaseModel):
    """Candidat pour le portefeuille"""

    horse_id: str
    name: Optional[str] = None
    race_id: Optional[str] = None
    market: str = "WIN"
    p: float = Field(..., gt=0, lt=1, description="Probabilité estimée")
    odds: float = Field(..., gt=1, description="Cote")
    ev: Optional[float] = Field(None, description="EV calculée (optionnel)")
    jockey: Optional[str] = None
    trainer: Optional[str] = None


class PortfolioRequest(BaseModel):
    """Requête d'optimisation de portefeuille"""

    candidates: List[BetCandidate]
    bankroll: float = Field(..., gt=0, description="Bankroll total")
    budget_today: Optional[float] = Field(None, description="Budget du jour (défaut: 10% bankroll)")


class PortfolioOutput(BaseModel):
    """Sortie du portefeuille optimisé"""

    budget_today: float
    kelly_fraction: float
    selection: List[Dict[str, Any]]
    excluded: List[Dict[str, Any]]
    summary: Dict[str, Any]
    run_notes: List[str]
    config_hash: str


class CalibrationHealthOutput(BaseModel):
    """Métriques de santé de calibration"""

    temperature: float
    calibrator: str
    alpha_by_disc: Dict[str, float]
    ece_7d: Optional[float] = Field(None, description="Expected Calibration Error 7j")
    brier_7d: Optional[float] = Field(None, description="Brier Score 7j")
    last_artifacts: Optional[str] = Field(None, description="Date derniers artefacts")
    config_hash: str


class BetLogEntry(BaseModel):
    """Entrée de log de pari"""

    race_id: str
    horse_id: str
    horse_name: Optional[str] = None
    bet_type: str = "WIN"
    p: float
    odds: float
    value_pct: Optional[float] = None
    kelly_raw: Optional[float] = None
    kelly_adjusted: Optional[float] = None
    stake: float
    result: Optional[int] = None  # 1=win, 0=loss, None=pending
    payout: Optional[float] = None


# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="🏇 Pro Betting API",
    description="""
    API d'analyse de paris hippiques niveau professionnel.

    **Caractéristiques:**
    - Probabilités calibrées et cohérentes (somme p_win = 1)
    - Normalisation softmax à température
    - Fusion bayésienne modèle/marché
    - Kelly criterion fractionnaire
    - Aucune fuite temporelle
    """,
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instance globale
_analyzer = None
_portfolio_optimizer = None
_calibration_state = None  # État de calibration (source de vérité)

# Charger config centralisée - AVEC priorité aux artefacts
if ARTIFACTS_LOADER_AVAILABLE:
    # Charger l'état de calibration depuis les artefacts (source de vérité)
    _calibration_state = load_calibration_state(prefer_artifacts=True)
    _TEMPERATURE = _calibration_state.temperature
    _BLEND_ALPHA = _calibration_state.alpha
    _ALPHA_BY_DISC = _calibration_state.alpha_by_disc
    _CALIBRATOR = _calibration_state.calibrator

    # Logger l'initialisation
    logging.info(log_calibration_init(_calibration_state))

    # Vérifier si YAML ≠ artefacts et logger un warning
    warn_if_mismatch()

    # Charger le reste depuis le loader config si disponible
    try:
        from config.loader import get_config

        _cfg = get_config()
        _KELLY_FRACTION = _cfg.kelly.fraction
        _VERSION_HASH = _cfg.version_hash
        CONFIG_AVAILABLE = True
    except ImportError:
        _KELLY_FRACTION = 0.25
        _VERSION_HASH = "unknown"
        CONFIG_AVAILABLE = False

    get_latest_calibration_report = lambda: None

elif CONFIG_AVAILABLE := False:
    pass  # Fallback ci-dessous
else:
    # Fallback si artifacts_loader non disponible
    try:
        from config.loader import get_config, get_latest_calibration_report

        _cfg = get_config()
        _TEMPERATURE = _cfg.calibration.temperature
        _BLEND_ALPHA = _cfg.calibration.blend_alpha_global
        _KELLY_FRACTION = _cfg.kelly.fraction
        _VERSION_HASH = _cfg.version_hash
        _ALPHA_BY_DISC = {
            "plat": _cfg.calibration.blend_alpha_plat,
            "trot": _cfg.calibration.blend_alpha_trot,
            "obstacle": _cfg.calibration.blend_alpha_obstacle,
            "global": _cfg.calibration.blend_alpha_global,
        }
        _CALIBRATOR = _cfg.calibration.calibrator
        CONFIG_AVAILABLE = True
    except ImportError:
        _TEMPERATURE = 1.254
        _BLEND_ALPHA = 0.2
        _KELLY_FRACTION = 0.25
        _VERSION_HASH = "unknown"
        _ALPHA_BY_DISC = {"plat": 0.0, "trot": 0.4, "obstacle": 0.4, "global": 0.2}
        _CALIBRATOR = "platt"
        CONFIG_AVAILABLE = False
        get_latest_calibration_report = lambda: None

# Import portfolio optimizer
try:
    from betting_portfolio_optimizer import BettingPortfolioOptimizer

    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False


def get_analyzer() -> ProBettingAnalyzer:
    """Singleton pour l'analyseur - config depuis pro_betting.yaml"""
    global _analyzer
    if _analyzer is None:
        conn = get_connection()
        _analyzer = ProBettingAnalyzer(
            conn,
            softmax_temperature=_TEMPERATURE,
            market_weight=1.0 - _BLEND_ALPHA,  # market_weight = 1 - blend_alpha
            kelly_fraction=_KELLY_FRACTION,
        )
    return _analyzer


def get_portfolio_optimizer() -> "BettingPortfolioOptimizer":
    """Singleton pour l'optimiseur de portefeuille"""
    global _portfolio_optimizer
    if _portfolio_optimizer is None and PORTFOLIO_AVAILABLE:
        _portfolio_optimizer = BettingPortfolioOptimizer()
    return _portfolio_optimizer


def log_bet_to_db(entry: dict):
    """Log un pari dans la table bets_log"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO bets_log (
                race_id, horse_id, horse_name, bet_type,
                p, odds, value_pct, kelly_raw, kelly_adjusted, stake,
                result, payout, pnl, config_hash, model_version, discipline, hippodrome
            ) VALUES (
                %(race_id)s, %(horse_id)s, %(horse_name)s, %(bet_type)s,
                %(p)s, %(odds)s, %(value_pct)s, %(kelly_raw)s, %(kelly_adjusted)s, %(stake)s,
                %(result)s, %(payout)s, %(pnl)s, %(config_hash)s, %(model_version)s,
                %(discipline)s, %(hippodrome)s
            )
        """,
            {
                "race_id": entry.get("race_id"),
                "horse_id": entry.get("horse_id"),
                "horse_name": entry.get("horse_name"),
                "bet_type": entry.get("bet_type", "WIN"),
                "p": entry.get("p"),
                "odds": entry.get("odds"),
                "value_pct": entry.get("value_pct"),
                "kelly_raw": entry.get("kelly_raw"),
                "kelly_adjusted": entry.get("kelly_adjusted"),
                "stake": entry.get("stake"),
                "result": entry.get("result"),
                "payout": entry.get("payout"),
                "pnl": entry.get("pnl"),
                "config_hash": _VERSION_HASH,
                "model_version": "2.0.0",
                "discipline": entry.get("discipline"),
                "hippodrome": entry.get("hippodrome"),
            },
        )
        conn.commit()
    except Exception as e:
        logging.warning(f"Erreur log pari: {e}")


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/", tags=["Info"])
async def root():
    """Page d'accueil"""
    return {
        "api": "Pro Betting Analyzer",
        "version": "2.0.0",
        "config_hash": _VERSION_HASH,
        "endpoints": [
            "/healthz",
            "/health",
            "/calibration/health",
            "/portfolio",
            "/analyze/{race_key}",
            "/analyze/batch",
            "/today",
            "/value-bets",
        ],
    }


@app.get("/healthz", tags=["Monitoring"])
async def healthz():
    """Health check simple (pour k8s/Docker)"""
    return {"status": "ok"}


@app.get("/health", tags=["Monitoring"])
async def health():
    """Health check détaillé"""
    return {
        "status": "healthy",
        "config_hash": _VERSION_HASH,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
    }


@app.get("/calibration/health", response_model=CalibrationHealthOutput, tags=["Monitoring"])
async def calibration_health():
    """
    Métriques de santé de la calibration.

    Retourne:
    - temperature: T* utilisée
    - calibrator: méthode (platt, isotonic)
    - alpha_by_disc: α par discipline
    - ece_7d: Expected Calibration Error sur 7 jours
    - brier_7d: Brier Score sur 7 jours
    - last_artifacts: date des derniers artefacts
    """
    # Calculer ECE et Brier depuis la BDD
    ece_7d = None
    brier_7d = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        # ECE via fonction SQL
        cur.execute("SELECT calculate_ece_7d()")
        result = cur.fetchone()
        if result and result[0]:
            ece_7d = round(result[0], 4)

        # Brier via fonction SQL
        cur.execute("SELECT calculate_brier_7d()")
        result = cur.fetchone()
        if result and result[0]:
            brier_7d = round(result[0], 4)

    except Exception as e:
        logging.warning(f"Erreur calcul métriques calibration: {e}")

    # Derniers artefacts
    last_artifacts = None
    if CONFIG_AVAILABLE:
        report = get_latest_calibration_report()
        if report:
            last_artifacts = report.get("timestamp", report.get("date"))

    return CalibrationHealthOutput(
        temperature=_TEMPERATURE,
        calibrator=_CALIBRATOR,
        alpha_by_disc=_ALPHA_BY_DISC,
        ece_7d=ece_7d,
        brier_7d=brier_7d,
        last_artifacts=last_artifacts,
        config_hash=_VERSION_HASH,
    )


@app.post("/portfolio", response_model=PortfolioOutput, tags=["Portfolio"])
async def optimize_portfolio(request: PortfolioRequest):
    """
    Optimise un portefeuille de paris.

    Applique:
    - Kelly fractionnaire (config.kelly.fraction)
    - Caps YAML (max_stake_pct, max_same_race, etc.)
    - Filtrage par value_cutoff
    - Pénalisation des corrélations

    **Input:**
    - candidates: liste de paris candidats (horse_id, p, odds, ...)
    - bankroll: capital total
    - budget_today: budget du jour (optionnel, défaut 10% bankroll)

    **Output:**
    - selection: paris sélectionnés avec stakes
    - excluded: paris exclus avec raison
    - summary: métriques du portefeuille
    """
    if not PORTFOLIO_AVAILABLE:
        raise HTTPException(status_code=503, detail="Portfolio optimizer non disponible")

    optimizer = get_portfolio_optimizer()

    # Convertir en format attendu
    bets = []
    for c in request.candidates:
        bets.append(
            {
                "horse_id": c.horse_id,
                "name": c.name or c.horse_id,
                "race_id": c.race_id,
                "market": c.market,
                "p": c.p,
                "odds": c.odds,
                "ev": c.ev if c.ev is not None else (c.p * c.odds - 1),
                "jockey": c.jockey,
                "trainer": c.trainer,
            }
        )

    # Optimiser
    result = optimizer.optimize(
        bets=bets, bankroll=request.bankroll, budget_today=request.budget_today
    )

    # Log chaque pari sélectionné
    for bet in result.selection:
        log_bet_to_db(
            {
                "race_id": bet.get("race_id"),
                "horse_id": bet.get("horse_id"),
                "horse_name": bet.get("name"),
                "bet_type": bet.get("market", "WIN"),
                "p": bet.get("p"),
                "odds": bet.get("odds"),
                "value_pct": bet.get("ev", 0) * 100,
                "kelly_raw": bet.get("kelly_raw"),
                "kelly_adjusted": bet.get("kelly_adjusted"),
                "stake": bet.get("stake"),
                "result": None,  # pending
            }
        )

    return PortfolioOutput(
        budget_today=result.budget_today,
        kelly_fraction=result.kelly_fraction,
        selection=result.selection,
        excluded=result.excluded,
        summary=result.summary,
        run_notes=result.run_notes,
        config_hash=_VERSION_HASH,
    )


@app.get("/analyze/{race_key:path}", response_model=RaceOutput, tags=["Analyse"])
async def analyze_race(race_key: str):
    """
    Analyse une course et retourne les probabilités calibrées.

    **Format race_key:** YYYY-MM-DD_RX_CY (ex: 2025-12-02_R1_C1)

    **Sortie:**
    - p_win: Probabilité de victoire (somme = 1 pour la course)
    - p_place: Probabilité d'être placé (top 3)
    - fair_odds: Cote juste calculée (1/p_win)
    - value_pct: Pourcentage de value (+ = favorable)
    - kelly_fraction: Fraction Kelly recommandée
    - rationale: 2-3 points clés maximum
    """
    analyzer = get_analyzer()

    try:
        result_json = analyzer.analyze_race(race_key)
        result = json.loads(result_json)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch", tags=["Analyse"])
async def analyze_batch(request: BatchRequest):
    """
    Analyse plusieurs courses en batch.

    **Limite:** 10 courses maximum par requête.
    """
    if len(request.race_keys) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 courses par requête")

    analyzer = get_analyzer()
    results = []

    for race_key in request.race_keys:
        try:
            result_json = analyzer.analyze_race(race_key)
            results.append(json.loads(result_json))
        except Exception as e:
            results.append({"race_id": race_key, "error": str(e)})

    return {"races": results}


@app.get("/today", tags=["Analyse"])
async def analyze_today(limit: int = Query(default=5, ge=1, le=20)):
    """
    Analyse les courses du jour.

    **Params:**
    - limit: Nombre max de courses à analyser (défaut: 5, max: 20)
    """
    analyzer = get_analyzer()
    conn = get_connection()
    cur = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")

    cur.execute(
        """
        SELECT DISTINCT race_key
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        ORDER BY race_key
        LIMIT %s
    """,
        (today + "%", limit),
    )

    race_keys = [row[0] for row in cur.fetchall()]

    if not race_keys:
        return {
            "date": today,
            "nb_courses": 0,
            "races": [],
            "run_notes": ["Aucune course trouvée pour aujourd'hui"],
        }

    results = []
    for race_key in race_keys:
        try:
            result_json = analyzer.analyze_race(race_key)
            results.append(json.loads(result_json))
        except Exception as e:
            results.append({"race_id": race_key, "error": str(e)})

    return {"date": today, "nb_courses": len(results), "races": results}


@app.get("/value-bets", tags=["Analyse"])
async def get_value_bets(
    min_value: float = Query(default=5.0, description="Value % minimum"),
    max_kelly: float = Query(default=0.05, description="Kelly max"),
    limit: int = Query(default=10, ge=1, le=50),
):
    """
    Récupère les value bets du jour.

    Filtre les partants avec:
    - value_pct >= min_value
    - kelly_fraction > 0 et <= max_kelly
    """
    analyzer = get_analyzer()
    conn = get_connection()
    cur = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")

    cur.execute(
        """
        SELECT DISTINCT race_key
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        ORDER BY race_key
    """,
        (today + "%",),
    )

    race_keys = [row[0] for row in cur.fetchall()]

    value_bets = []

    for race_key in race_keys:
        try:
            result = analyzer.analyze_race_dict(race_key)

            if "error" in result:
                continue

            for runner in result.get("runners", []):
                value = runner.get("value_pct")
                kelly = runner.get("kelly_fraction")

                if (
                    value is not None
                    and value >= min_value
                    and kelly is not None
                    and kelly > 0
                    and kelly <= max_kelly
                ):
                    value_bets.append(
                        {
                            "race_id": race_key,
                            "hippodrome": result.get("hippodrome"),
                            "numero": runner.get("numero"),
                            "nom": runner.get("nom"),
                            "p_win": runner.get("p_win"),
                            "market_odds": runner.get("market_odds"),
                            "fair_odds": runner.get("fair_odds"),
                            "value_pct": value,
                            "kelly_fraction": kelly,
                            "rationale": runner.get("rationale", []),
                        }
                    )
        except:
            continue

    # Trier par value décroissante
    value_bets.sort(key=lambda x: x.get("value_pct", 0), reverse=True)

    return {
        "date": today,
        "nb_value_bets": len(value_bets[:limit]),
        "filters": {"min_value_pct": min_value, "max_kelly": max_kelly},
        "bets": value_bets[:limit],
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
