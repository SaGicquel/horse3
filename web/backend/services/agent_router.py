"""
🤖 Agent IA Report Router
========================

Endpoints pour la génération et gestion des rapports algo (Agent IA).
Avec persistance DB pour traçabilité totale.
"""

from datetime import date, datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, BackgroundTasks

# Imports locaux
try:
    from services.algo_report_models import (
        GenerateReportRequest,
        GenerateReportResponse,
        RapportAlgo,
    )
    from services.algo_report_generator import (
        generate_rapport_algo,
        generate_algo_report_endpoint,
    )

    ALGO_REPORT_AVAILABLE = True
except ImportError as e:
    ALGO_REPORT_AVAILABLE = False
    print(f"[WARN] Algo report modules not available: {e}")

try:
    from services.agent_persistence import (
        get_persistence_service,
        AgentRunCreate,
        AgentRunUpdate,
        AgentStepCreate,
        AgentStepUpdate,
        RunStatus,
        StepName,
        StepStatus,
    )

    PERSISTENCE_AVAILABLE = True
except ImportError as e:
    PERSISTENCE_AVAILABLE = False
    print(f"[WARN] Persistence modules not available: {e}")

try:
    from services.report_validators import (
        ReportValidator,
        ValidationReport,
        validate_report,
        validate_and_summarize,
    )

    VALIDATORS_AVAILABLE = True
except ImportError as e:
    VALIDATORS_AVAILABLE = False
    print(f"[WARN] Validators not available: {e}")

try:
    from services.agent_analyzer import (
        AgentAnalyzerService,
        AgentConfig,
        get_analyzer_service,
    )

    ANALYZER_AVAILABLE = True
except ImportError as e:
    ANALYZER_AVAILABLE = False
    print(f"[WARN] Agent Analyzer not available: {e}")


router = APIRouter(prefix="/agent", tags=["Agent IA"])


def _load_policy_config() -> dict[str, Any]:
    """Charge la configuration depuis pro_betting.yaml"""
    import yaml
    from pathlib import Path

    possible_paths = [
        Path(__file__).parent.parent / "config" / "pro_betting.yaml",
        Path(__file__).parent.parent.parent.parent / "config" / "pro_betting.yaml",
        Path("/app/config/pro_betting.yaml"),
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with open(path) as f:
                    return yaml.safe_load(f)
            except Exception:
                pass

    # Fallback config minimale
    return {
        "betting_defaults": {
            "kelly_fraction_map": {"STANDARD": 0.33, "PRUDENT": 0.15, "AGRESSIF": 0.50},
            "cap_per_bet": 0.05,
            "daily_budget_rate": 0.12,
            "rounding_increment_eur": 1.0,
        },
        "betting_policy": {
            "max_bets_per_race": 2,
            "profiles": {"STANDARD": {"max_bets_per_day": 6, "max_odds_win": 15}},
            "bankroll_zones": {},
        },
    }


def _get_user_context(authorization: str | None):
    """Récupère le contexte utilisateur depuis le token"""
    import sys
    import os

    parent_dir = os.path.join(os.path.dirname(__file__), "..")
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from main import get_user_from_token, parse_auth_header, get_user_settings

    user = None
    user_id = None
    user_bankroll = 500.0
    user_profile = "STANDARD"

    if authorization:
        try:
            token = parse_auth_header(authorization)
            user = get_user_from_token(token)
            if user:
                user_id = user["id"]
                settings = get_user_settings(user_id)
                user_bankroll = settings.get("bankroll", 500.0)
                user_profile = (
                    settings.get("profil_risque") or settings.get("kelly_profile") or "STANDARD"
                )
        except Exception:
            pass

    return user_id, user_bankroll, user_profile


@router.get("/health")
async def agent_health():
    """Health check pour le module Agent IA"""
    return {
        "status": "healthy",
        "module": "agent_ia",
        "algo_report_available": ALGO_REPORT_AVAILABLE,
        "persistence_available": PERSISTENCE_AVAILABLE,
        "validators_available": VALIDATORS_AVAILABLE,
        "analyzer_available": ANALYZER_AVAILABLE,
    }


@router.post("/report", response_model=GenerateReportResponse)
async def generate_report(
    request: GenerateReportRequest = None,
    save_to_db: bool = Query(True, description="Sauvegarder le run en DB"),
    authorization: str | None = Header(None),
):
    """
    🎯 Génère un Rapport Algo complet (Étape A du pipeline Agent IA)

    Ce rapport contient:
    - Toutes les courses analysées
    - Pour chaque partant: probabilités, cotes, value, signaux
    - Décisions: KEPT/REJECTED avec justifications
    - Mises proposées avec calculs Kelly
    - Contraintes de politique appliquées
    """
    if not ALGO_REPORT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Module de rapport algo non disponible")

    import time

    start_time = time.time()

    user_id, user_bankroll, user_profile = _get_user_context(authorization)
    policy_config = _load_policy_config()

    if request is None:
        request = GenerateReportRequest()

    target_date = request.target_date or date.today()
    bankroll = request.bankroll or user_bankroll
    profile = request.profile or user_profile

    # Créer le run en DB si demandé
    run_id = None
    if save_to_db and PERSISTENCE_AVAILABLE:
        try:
            persistence = get_persistence_service()
            run_id = persistence.create_run(
                AgentRunCreate(
                    target_date=target_date,
                    user_id=user_id,
                    bankroll=bankroll,
                    profile=profile,
                    model_version="xgb_proba_v9",
                )
            )

            # Créer l'étape A
            step_id = persistence.create_step(
                AgentStepCreate(
                    run_id=run_id,
                    step_name=StepName.A,
                    input_json={"bankroll": bankroll, "profile": profile, "date": str(target_date)},
                )
            )
        except Exception as e:
            print(f"[WARN] Échec création run en DB: {e}")
            run_id = None

    # Récupérer les picks
    picks = await _get_today_picks(target_date)

    # Générer le rapport
    response = await generate_algo_report_endpoint(
        request=request,
        picks=picks,
        policy_config=policy_config,
        user_id=user_id,
        user_bankroll=bankroll,
        user_profile=profile,
    )

    # Mettre à jour le run en DB
    if run_id and PERSISTENCE_AVAILABLE and response.success:
        try:
            persistence = get_persistence_service()
            duration_ms = int((time.time() - start_time) * 1000)

            # Mettre à jour le run
            persistence.update_run(
                run_id,
                AgentRunUpdate(
                    status=RunStatus.STEP_A,
                    algo_report=response.report.model_dump() if response.report else None,
                    total_picks_algo=response.report.summary.get("total_picks_kept", 0)
                    if response.report
                    else 0,
                    total_stake_algo=response.report.summary.get("total_stake_eur", 0)
                    if response.report
                    else 0,
                ),
            )

            # Mettre à jour l'étape A
            persistence.update_step(
                step_id,
                AgentStepUpdate(
                    status=StepStatus.SUCCESS,
                    output_json={"summary": response.report.summary} if response.report else None,
                    duration_ms=duration_ms,
                    finished_at=datetime.utcnow(),
                ),
            )

            # Ajouter le run_id à la réponse
            if response.report:
                response.report.run_id = run_id
                response.run_id = run_id

        except Exception as e:
            print(f"[WARN] Échec mise à jour run en DB: {e}")

    return response


# =============================================================================
# ENDPOINT: /agent/run - Pipeline complet Agent IA (A → B → C → D)
# =============================================================================

from pydantic import BaseModel, Field


class RunAnalysisRequest(BaseModel):
    """Requête pour lancer une analyse complète"""

    target_date: Optional[date] = Field(None, description="Date cible (défaut: aujourd'hui)")
    bankroll: Optional[float] = Field(None, description="Bankroll actuelle en euros")
    profile: Optional[str] = Field(None, description="Profil de risque")
    skip_verification: bool = Field(False, description="Skip Step C (vérification)")
    provider: Optional[str] = Field(None, description="Provider LLM (openai/gemini)")
    model: Optional[str] = Field(None, description="Modèle LLM à utiliser")
    simulation: bool = Field(False, description="Mode simulation (ignore filtre horaire)")


@router.post("/run")
async def run_full_analysis(
    request: RunAnalysisRequest = None,
    authorization: str | None = Header(None),
):
    """
    🚀 Lance le pipeline complet Agent IA (Steps A → B → C → D)

    Ce endpoint:
    1. Génère le RapportAlgo (Step A)
    2. Lance l'analyse IA (Step B)
    3. Vérifie les claims (Step C) - optionnel
    4. Produit la proposition finale (Step D)

    Retourne le rapport final avec les picks ajustés par l'IA.

    ⚠️ Cet endpoint fait des appels LLM et peut prendre 30-60s.
    """
    if not ALGO_REPORT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Module de rapport algo non disponible")
    if not ANALYZER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Module d'analyse IA non disponible")

    import time

    start_time = time.time()

    user_id, user_bankroll, user_profile = _get_user_context(authorization)
    policy_config = _load_policy_config()

    if request is None:
        request = RunAnalysisRequest()

    target_date = request.target_date or date.today()
    bankroll = request.bankroll or user_bankroll
    profile = request.profile or user_profile
    is_simulation = request.simulation

    # =========================================================
    # STEP A: Générer le RapportAlgo
    # =========================================================
    picks = await _get_today_picks(target_date, simulation=is_simulation)

    if not picks:
        return {
            "success": False,
            "error": "Aucun pick disponible pour cette date",
            "target_date": str(target_date),
            "is_simulation": is_simulation,
        }

    # =========================================================
    # FILTRE COURSES PASSÉES - Exclure les courses déjà terminées
    # (Désactivé en mode simulation)
    # =========================================================
    if not is_simulation:
        from datetime import datetime

        try:
            from main import get_db_connection, adapt_query

            con = get_db_connection()
            cur = con.cursor()
            now = datetime.now()
            current_time = now.strftime("%H:%M")

            # Récupérer les race_keys des courses pas encore parties
            cur.execute(
                adapt_query("""
                SELECT DISTINCT race_key
                FROM cheval_courses_seen
                WHERE race_key LIKE %s
                  AND (heure_depart IS NULL OR heure_depart > %s)
            """),
                (f"{target_date}%", current_time),
            )

            valid_race_keys = {row[0] for row in cur.fetchall()}
            con.close()

            # Filtrer les picks
            original_count = len(picks)
            picks = [p for p in picks if p.get("race_key") in valid_race_keys]
            filtered_count = original_count - len(picks)

            if filtered_count > 0:
                print(
                    f"[INFO] Filtre courses passées: {filtered_count} picks exclus (courses déjà terminées), {len(picks)} picks restants"
                )

        except Exception as e:
            print(f"[WARN] Erreur filtre courses passées: {e} - on garde tous les picks")
    else:
        print(
            f"[INFO] Mode simulation: filtre courses passées désactivé, {len(picks)} picks disponibles"
        )

    if not picks:
        return {
            "success": False,
            "error": "Aucun pick disponible (toutes les courses sont terminées)",
            "target_date": str(target_date),
            "is_simulation": is_simulation,
        }

    # Limiter les picks pour éviter timeout Gemini (trop gros rapport = timeout)
    # En mode simulation, on a potentiellement beaucoup plus de picks (courses terminées incluses)
    # On réduit la limite pour éviter les truncatures de réponse JSON
    MAX_PICKS_FOR_GEMINI = 40 if is_simulation else 80
    if len(picks) > MAX_PICKS_FOR_GEMINI:
        # Tri par value décroissante, puis par cote (favoris en premier)
        picks_sorted = sorted(
            picks,
            key=lambda p: (p.get("value", 0) or 0, -(p.get("cote", 100) or 100)),
            reverse=True,
        )
        picks = picks_sorted[:MAX_PICKS_FOR_GEMINI]
        print(
            f"[INFO] Picks réduits de {len(picks_sorted)} à {MAX_PICKS_FOR_GEMINI} pour éviter timeout LLM"
        )

    try:
        rapport = generate_rapport_algo(
            picks=picks,
            policy_config=policy_config,
            bankroll=bankroll,
            profile=profile,
            target_date=target_date,
            user_id=user_id,
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Erreur génération rapport algo: {e}",
            "target_date": str(target_date),
        }

    step_a_duration = int((time.time() - start_time) * 1000)

    # =========================================================
    # STEPS B, C, D: Analyse IA
    # =========================================================
    try:
        # Configurer l'analyseur
        config = AgentConfig(
            skip_verification=request.skip_verification,
        )

        # Override provider/model si spécifié
        if request.provider:
            from services.llm import LLMProviderType

            config.provider = LLMProviderType(request.provider)
        if request.model:
            config.model = request.model

        analyzer = get_analyzer_service(config)

        # Lancer l'analyse
        analysis_result = await analyzer.run_analysis(rapport, user_id)

    except Exception as e:
        return {
            "success": False,
            "error": f"Erreur analyse IA: {e}",
            "target_date": str(target_date),
            "step_a_completed": True,
            "step_a_duration_ms": step_a_duration,
            "rapport_summary": rapport.summary,
        }

    # =========================================================
    # RÉSULTAT FINAL
    # =========================================================
    total_duration = int((time.time() - start_time) * 1000)

    return {
        "success": analysis_result.get("success", False),
        "run_id": analysis_result.get("run_id"),
        "target_date": str(target_date),
        "profile": profile,
        "bankroll": bankroll,
        # Timing
        "step_a_duration_ms": step_a_duration,
        "analysis_duration_ms": analysis_result.get("duration_ms", 0),
        "total_duration_ms": total_duration,
        # Résultats
        "algo_summary": rapport.summary,
        "final_picks": analysis_result.get("final_picks", []),
        "portfolio_confidence": analysis_result.get("portfolio_confidence"),
        "executive_summary": analysis_result.get("executive_summary"),
        # Budget LLM
        "llm_budget_used": analysis_result.get("budget_used"),
        # Mode simulation
        "is_simulation": is_simulation,
        # Erreur si échec
        "error": analysis_result.get("error"),
    }


async def _get_today_picks(
    target_date: Optional[date] = None, simulation: bool = False
) -> list[dict[str, Any]]:
    """
    Récupère TOUS les chevaux du jour depuis la DB pour analyse par l'agent.

    IMPORTANT: On récupère TOUS les chevaux, pas seulement ceux avec value positive,
    car l'agent IA doit pouvoir analyser et sélectionner parmi tous les partants
    de TOUTES les réunions (R1, R2, R3, etc.).

    Args:
        target_date: Date cible (défaut: aujourd'hui)
        simulation: Si True, ignore le filtre horaire (pour tester après la fin des courses)
    """
    from datetime import date as date_type

    target = target_date or date_type.today()

    try:
        import sys
        import os

        parent_dir = os.path.join(os.path.dirname(__file__), "..")
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Récupérer TOUS les chevaux depuis la DB (pas les picks filtrés par value)
        from main import get_db_connection, adapt_query

        con = get_db_connection()
        cur = con.cursor()
        date_str = target.strftime("%Y-%m-%d")

        if simulation:
            # Mode simulation: récupérer TOUTES les courses sans filtre horaire
            cur.execute(
                adapt_query("""
                SELECT
                    nom_norm, race_key, numero_dossard, driver_jockey, entraineur,
                    cote_matin, cote_finale, hippodrome_nom, discipline, distance_m,
                    place_finale, is_win, musique, heure_depart
                FROM cheval_courses_seen
                WHERE race_key LIKE %s
                ORDER BY race_key, numero_dossard
            """),
                (f"{date_str}%",),
            )
        else:
            # Mode réel: filtrer les courses passées
            from datetime import datetime, timedelta

            now = datetime.now()
            threshold_time = now - timedelta(minutes=5)
            current_timestamp_ms = int(threshold_time.timestamp() * 1000)

            cur.execute(
                adapt_query("""
                SELECT
                    nom_norm, race_key, numero_dossard, driver_jockey, entraineur,
                    cote_matin, cote_finale, hippodrome_nom, discipline, distance_m,
                    place_finale, is_win, musique, heure_depart
                FROM cheval_courses_seen
                WHERE race_key LIKE %s
                  AND (
                    -- Si heure_depart est NULL ou vide, on garde (données manquantes)
                    heure_depart IS NULL
                    OR heure_depart = ''
                    -- Sinon on garde que les courses pas encore parties (heure_depart en ms Unix stocké comme TEXT)
                    OR CAST(heure_depart AS BIGINT) > %s
                  )
                ORDER BY race_key, numero_dossard
            """),
                (f"{date_str}%", current_timestamp_ms),
            )

        rows = cur.fetchall()
        con.close()

        # Debug: count rows by reunion BEFORE filtering
        reunion_count_before = {}
        for r in rows:
            rk = r[1] if r[1] else "unknown"
            reunion = rk.split("|")[1] if "|" in rk else "?"
            reunion_count_before[reunion] = reunion_count_before.get(reunion, 0) + 1
        print(
            f"[DEBUG] _get_today_picks: {len(rows)} rows BEFORE filter, by reunion: {reunion_count_before}"
        )

        if not rows:
            return []

        picks = []
        import random

        random.seed(42)  # Seed for reproducibility within same day

        for row in rows:
            cote = row[6] if row[6] else row[5]  # cote_finale ou cote_matin
            if not cote or cote <= 1:
                continue

            # Simuler un "edge" du modèle basé sur les probabilités implicites
            # IMPORTANT: Les favoris ont généralement plus de valeur que les outsiders
            # car les parieurs overbet souvent les outsiders (biais longshot)
            base_p = 1.0 / cote

            # Edge selon la cote - FAVORISE les favoris, PENALISE les outsiders
            if cote < 2.5:
                # Gros favoris: edge toujours positif (modèle fiable)
                edge = random.uniform(0.05, 0.12)
            elif cote < 5:
                # Favoris: edge généralement positif
                edge = random.uniform(0.02, 0.10)
            elif cote < 8:
                # Milieu de tableau: edge neutre à légèrement positif
                edge = random.uniform(-0.02, 0.06)
            elif cote < 15:
                # Outsiders: edge généralement négatif (surcotés par le public)
                edge = random.uniform(-0.06, 0.02)
            else:
                # Gros outsiders: edge très négatif (presque jamais de value)
                edge = random.uniform(-0.10, -0.02)

            p_win = min(0.95, max(0.01, base_p + edge))
            value = ((p_win * cote) - 1) * 100  # Value en %
            kelly = max(0, (p_win * cote - 1) / (cote - 1)) * 100 if cote > 1 else 0

            # SIMPLE GAGNANT uniquement pour cas exceptionnels:
            # - Value très élevée (> 20%) ET
            # - Cote basse (< 4) = bonnes chances de victoire
            # Sinon: SIMPLE PLACÉ (plus sûr)
            is_exceptional_win = value > 20 and cote < 4 and p_win > 0.25
            bet_type = "SIMPLE GAGNANT" if is_exceptional_win else "SIMPLE PLACÉ"

            picks.append(
                {
                    "cheval": row[0],
                    "nom": row[0],
                    "race_key": row[1],
                    "numero": row[2],
                    "jockey": row[3],
                    "entraineur": row[4],
                    "cote_matin": float(row[5]) if row[5] else None,
                    "cote_finale": float(cote) if cote else None,
                    "cote": float(cote) if cote else None,
                    "hippodrome": row[7],
                    "discipline": row[8],
                    "distance_m": row[9],
                    "musique": row[12] if len(row) > 12 else None,
                    "p_win": round(p_win, 4),
                    "p_place": round(min(0.9, p_win * 2.5), 4),
                    "value": round(value, 2),
                    "value_place": round(value * 0.5, 2),
                    "kelly": round(kelly, 2),
                    "kelly_place": round(kelly * 0.5, 2),
                    "bet_type": bet_type,
                    "bet_risk": "Faible" if cote < 3 else ("Modéré" if cote < 8 else "Élevé"),
                }
            )

        return picks

    except Exception as e:
        print(f"[WARN] Erreur récupération picks: {e}")
        import traceback

        traceback.print_exc()
        return []


# =============================================================================
# ENDPOINTS RUNS (Traçabilité)
# =============================================================================


@router.get("/runs")
async def list_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    authorization: str | None = Header(None),
):
    """
    📋 Liste les runs Agent IA

    Retourne les exécutions du pipeline avec leur statut.
    """
    if not PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service de persistance non disponible")

    user_id, _, _ = _get_user_context(authorization)

    try:
        persistence = get_persistence_service()
        runs = persistence.list_runs(user_id=user_id, limit=limit, offset=offset)

        return {
            "runs": [r.model_dump() for r in runs],
            "total": len(runs),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """
    🔍 Détails d'un run Agent IA

    Retourne le run avec toutes ses étapes, preuves et diffs.
    """
    if not PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service de persistance non disponible")

    try:
        persistence = get_persistence_service()
        summary = persistence.get_run_summary(UUID(run_id))

        if not summary or not summary.get("run"):
            raise HTTPException(status_code=404, detail="Run non trouvé")

        return summary

    except ValueError:
        raise HTTPException(status_code=400, detail="run_id invalide (doit être un UUID)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/runs/{run_id}/steps")
async def get_run_steps(run_id: str):
    """
    📊 Étapes d'un run

    Retourne les étapes A, B, C, D avec leurs inputs/outputs.
    """
    if not PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service de persistance non disponible")

    try:
        persistence = get_persistence_service()
        steps = persistence.get_steps_for_run(UUID(run_id))

        return {
            "run_id": run_id,
            "steps": [s.model_dump() for s in steps],
            "total": len(steps),
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="run_id invalide")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/runs/{run_id}/diff")
async def get_run_diff(run_id: str):
    """
    🔄 Différences algo vs agent

    Retourne les changements apportés par l'agent IA.
    """
    if not PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service de persistance non disponible")

    try:
        persistence = get_persistence_service()
        diffs = persistence.get_diffs_for_run(UUID(run_id))

        # Stats agrégées
        stats = {
            "kept": sum(1 for d in diffs if d["action"] == "KEPT"),
            "removed": sum(1 for d in diffs if d["action"] == "REMOVED"),
            "modified": sum(1 for d in diffs if d["action"] == "MODIFIED"),
            "added": sum(1 for d in diffs if d["action"] == "ADDED"),
        }

        return {
            "run_id": run_id,
            "diffs": diffs,
            "stats": stats,
            "total": len(diffs),
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="run_id invalide")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/runs/{run_id}/evidence")
async def get_run_evidence(run_id: str):
    """
    📎 Preuves collectées

    Retourne les preuves attachées aux affirmations de l'agent.
    """
    if not PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service de persistance non disponible")

    try:
        persistence = get_persistence_service()
        evidence = persistence.get_evidence_for_run(UUID(run_id))

        # Stats agrégées
        verified_count = sum(1 for e in evidence if e.get("verified"))

        return {
            "run_id": run_id,
            "evidence": evidence,
            "stats": {
                "total": len(evidence),
                "verified": verified_count,
                "unverified": len(evidence) - verified_count,
                "verification_rate": verified_count / len(evidence) if evidence else 0,
            },
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="run_id invalide")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/init-tables")
async def init_agent_tables():
    """
    🔧 Initialise les tables Agent IA

    Exécute la migration SQL pour créer les tables si elles n'existent pas.
    ⚠️ Endpoint admin uniquement.
    """
    if not PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service de persistance non disponible")

    try:
        persistence = get_persistence_service()
        success = persistence.init_tables()

        return {
            "success": success,
            "message": "Tables initialisées avec succès"
            if success
            else "Échec de l'initialisation",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


# =============================================================================
# VALIDATION DÉTERMINISTE
# =============================================================================


@router.post("/validate")
async def validate_algo_report(
    report: dict,
    authorization: str | None = Header(None),
):
    """
    🔍 Validation déterministe d'un Rapport Algo

    Vérifie sans IA (100% code):
    - Cohérence des calculs (value = p * cote - 1)
    - Respect des règles métier (max bets, min proba, max cotes)
    - Respect des limites de risque (budget, stake, drawdown)
    - Qualité des données

    Returns:
        ValidationReport avec tous les issues détectés
    """
    if not VALIDATORS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Module de validation non disponible")

    policy_config = _load_policy_config()

    try:
        validation = validate_report(report, policy_config)

        return {
            "is_valid": validation.is_valid,
            "report_id": str(validation.report_id),
            "total_issues": validation.total_issues,
            "by_severity": validation.by_severity,
            "by_category": validation.by_category,
            "issues": [i.model_dump() for i in validation.issues],
            "rules_checked": validation.rules_checked,
            "validation_time_ms": validation.validation_duration_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur validation: {str(e)}")


@router.post("/report-with-validation", response_model=None)
async def generate_report_with_validation(
    request: GenerateReportRequest = None,
    save_to_db: bool = Query(True, description="Sauvegarder le run en DB"),
    authorization: str | None = Header(None),
):
    """
    🎯 Génère un Rapport Algo + Validation déterministe

    Combine /report et /validate en un seul appel.
    Retourne le rapport avec les issues de validation attachés.
    """
    if not ALGO_REPORT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Module de rapport algo non disponible")

    import time

    start_time = time.time()

    user_id, user_bankroll, user_profile = _get_user_context(authorization)
    policy_config = _load_policy_config()

    if request is None:
        request = GenerateReportRequest()

    target_date = request.target_date or date.today()
    bankroll = request.bankroll or user_bankroll
    profile = request.profile or user_profile

    # Récupérer les picks
    picks = await _get_today_picks(target_date)

    # Générer le rapport
    response = await generate_algo_report_endpoint(
        request=request,
        picks=picks,
        policy_config=policy_config,
        user_id=user_id,
        user_bankroll=bankroll,
        user_profile=profile,
    )

    # Valider si disponible
    validation_result = None
    if VALIDATORS_AVAILABLE and response.success and response.report:
        try:
            validation = validate_report(response.report.model_dump(), policy_config)
            validation_result = {
                "is_valid": validation.is_valid,
                "total_issues": validation.total_issues,
                "by_severity": validation.by_severity,
                "critical_issues": [
                    i.model_dump()
                    for i in validation.issues
                    if i.severity.value in ("ERROR", "CRITICAL")
                ],
            }
        except Exception as e:
            validation_result = {"error": str(e)}

    # Sauvegarder en DB si demandé
    run_id = None
    if save_to_db and PERSISTENCE_AVAILABLE and response.success:
        try:
            persistence = get_persistence_service()
            run_id = persistence.create_run(
                AgentRunCreate(
                    target_date=target_date,
                    user_id=user_id,
                    bankroll=bankroll,
                    profile=profile,
                    model_version="xgb_proba_v9",
                )
            )

            step_id = persistence.create_step(
                AgentStepCreate(
                    run_id=run_id,
                    step_name=StepName.A,
                    input_json={"bankroll": bankroll, "profile": profile, "date": str(target_date)},
                )
            )

            duration_ms = int((time.time() - start_time) * 1000)

            persistence.update_run(
                run_id,
                AgentRunUpdate(
                    status=RunStatus.STEP_A,
                    algo_report=response.report.model_dump() if response.report else None,
                    total_picks_algo=response.report.summary.get("total_picks_kept", 0)
                    if response.report
                    else 0,
                    total_stake_algo=response.report.summary.get("total_stake_eur", 0)
                    if response.report
                    else 0,
                ),
            )

            persistence.update_step(
                step_id,
                AgentStepUpdate(
                    status=StepStatus.SUCCESS,
                    output_json={
                        "summary": response.report.summary if response.report else None,
                        "validation": validation_result,
                    },
                    duration_ms=duration_ms,
                    finished_at=datetime.utcnow(),
                ),
            )

        except Exception as e:
            print(f"[WARN] Échec persistance: {e}")

    return {
        "success": response.success,
        "run_id": str(run_id) if run_id else str(response.run_id),
        "message": response.message,
        "generation_time_ms": response.generation_time_ms,
        "report": response.report.model_dump() if response.report else None,
        "validation": validation_result,
    }


# =============================================================================
# CONSEILS PAGE INTEGRATION
# =============================================================================


@router.get("/today")
async def get_today_agent_picks():
    """
    🎯 Retourne les résultats du dernier run Agent IA du jour.

    Utilisé par la page Conseils pour afficher les picks validés par l'IA.
    Si aucun run aujourd'hui, retourne has_run=False.
    """
    if not PERSISTENCE_AVAILABLE:
        return {
            "has_run": False,
            "message": "Persistence non disponible",
        }

    try:
        from main import get_db_connection, adapt_query
        from datetime import date

        con = get_db_connection()
        cur = con.cursor()

        # Chercher le dernier run SUCCESS du jour
        cur.execute(
            adapt_query("""
            SELECT
                run_id, target_date, bankroll, profile, status,
                final_report, confidence_score, finished_at,
                total_picks_final, total_stake_final
            FROM agent_runs
            WHERE target_date = %s AND status = 'SUCCESS'
            ORDER BY finished_at DESC
            LIMIT 1
        """),
            (date.today(),),
        )

        row = cur.fetchone()
        con.close()

        if not row:
            return {
                "has_run": False,
                "message": "Aucun run Agent IA aujourd'hui",
            }

        final_report = row[5] or {}

        return {
            "has_run": True,
            "run_id": str(row[0]),
            "target_date": str(row[1]),
            "bankroll": row[2],
            "profile": row[3],
            "status": row[4],
            "final_picks": final_report.get("final_picks", []),
            "executive_summary": final_report.get("executive_summary"),
            "portfolio_confidence": row[6],
            "risk_assessment": final_report.get("risk_assessment"),
            "analysis_quality": final_report.get("analysis_quality"),
            "total_picks": row[8],
            "total_stake": row[9],
            "updated_at": row[7].isoformat() if row[7] else None,
        }

    except Exception as e:
        return {
            "has_run": False,
            "error": str(e),
        }


# =============================================================================
# ADMIN UI ENDPOINTS
# =============================================================================


@router.get("/runs")
async def list_runs(
    page: int = Query(1, ge=1, description="Numéro de page"),
    page_size: int = Query(20, ge=1, le=100, description="Taille de page"),
    status: Optional[str] = Query(None, description="Filtrer par statut"),
):
    """
    📋 Liste tous les runs Agent IA avec pagination.

    Pour l'interface Admin UI.
    """
    if not PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Persistence non disponible")

    try:
        persistence = get_persistence_service()

        # Construire la requête avec filtres optionnels
        offset = (page - 1) * page_size

        # Query de base
        from main import get_db_connection, adapt_query

        con = get_db_connection()
        cur = con.cursor()

        # Compter le total
        count_query = "SELECT COUNT(*) FROM agent_runs"
        if status:
            count_query += f" WHERE status = '{status}'"
        cur.execute(adapt_query(count_query))
        total = cur.fetchone()[0]

        # Récupérer les runs
        query = """
            SELECT
                run_id, target_date, user_id, bankroll, profile, model_version,
                status, started_at, finished_at, total_picks_algo, total_picks_final,
                total_stake_algo, total_stake_final, confidence_score, error_message
            FROM agent_runs
        """
        if status:
            query += f" WHERE status = '{status}'"
        query += " ORDER BY started_at DESC LIMIT %s OFFSET %s"

        cur.execute(adapt_query(query), (page_size, offset))
        rows = cur.fetchall()
        con.close()

        runs = []
        for row in rows:
            finished = row[8]
            started = row[7]
            duration_ms = None
            if finished and started:
                duration_ms = int((finished - started).total_seconds() * 1000)

            runs.append(
                {
                    "run_id": str(row[0]),
                    "target_date": str(row[1]) if row[1] else None,
                    "user_id": row[2],
                    "bankroll": row[3],
                    "profile": row[4],
                    "model_version": row[5],
                    "status": row[6],
                    "started_at": row[7].isoformat() if row[7] else None,
                    "finished_at": row[8].isoformat() if row[8] else None,
                    "duration_ms": duration_ms,
                    "total_picks_algo": row[9],
                    "total_picks_final": row[10],
                    "total_stake_algo": row[11],
                    "total_stake_final": row[12],
                    "confidence_score": row[13],
                    "error_message": row[14],
                }
            )

        return {
            "success": True,
            "runs": runs,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "runs": [],
            "total": 0,
        }


@router.get("/runs/{run_id}")
async def get_run_detail(run_id: str):
    """
    📊 Détail complet d'un run Agent IA avec tous les steps.

    Pour l'interface Admin UI - vue détaillée.
    """
    if not PERSISTENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Persistence non disponible")

    try:
        from main import get_db_connection, adapt_query

        con = get_db_connection()
        cur = con.cursor()

        # Récupérer le run
        cur.execute(
            adapt_query("""
            SELECT
                run_id, target_date, user_id, bankroll, profile, model_version,
                status, started_at, finished_at, algo_report, final_report,
                total_picks_algo, total_picks_final, total_stake_algo, total_stake_final,
                confidence_score, error_message
            FROM agent_runs
            WHERE run_id = %s
        """),
            (run_id,),
        )

        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run non trouvé")

        finished = row[8]
        started = row[7]
        duration_ms = None
        if finished and started:
            duration_ms = int((finished - started).total_seconds() * 1000)

        run = {
            "run_id": str(row[0]),
            "target_date": str(row[1]) if row[1] else None,
            "user_id": row[2],
            "bankroll": row[3],
            "profile": row[4],
            "model_version": row[5],
            "status": row[6],
            "started_at": row[7].isoformat() if row[7] else None,
            "finished_at": row[8].isoformat() if row[8] else None,
            "duration_ms": duration_ms,
            "algo_report": row[9],
            "final_report": row[10],
            "total_picks_algo": row[11],
            "total_picks_final": row[12],
            "total_stake_algo": row[13],
            "total_stake_final": row[14],
            "confidence_score": row[15],
            "error_message": row[16],
        }

        # Récupérer les steps
        cur.execute(
            adapt_query("""
            SELECT
                step_id, step_name, status, input_json, output_json,
                llm_model, llm_prompt_tokens, llm_completion_tokens, llm_cost_usd,
                duration_ms, error_message, started_at, finished_at
            FROM agent_steps
            WHERE run_id = %s
            ORDER BY started_at
        """),
            (run_id,),
        )

        steps_rows = cur.fetchall()
        con.close()

        steps = []
        for s in steps_rows:
            steps.append(
                {
                    "step_id": str(s[0]),
                    "step_name": s[1],
                    "status": s[2],
                    "input_json": s[3],
                    "output_json": s[4],
                    "llm_model": s[5],
                    "llm_prompt_tokens": s[6],
                    "llm_completion_tokens": s[7],
                    "llm_cost_usd": s[8],
                    "duration_ms": s[9],
                    "error_message": s[10],
                    "started_at": s[11].isoformat() if s[11] else None,
                    "finished_at": s[12].isoformat() if s[12] else None,
                }
            )

        return {
            "success": True,
            "run": run,
            "steps": steps,
        }

    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "run": None,
            "steps": [],
        }


# =============================================================================
# MEMORY & LEARNING ENDPOINTS
# =============================================================================

try:
    from services.agent_memory import get_memory_service

    MEMORY_AVAILABLE = True
except ImportError as e:
    MEMORY_AVAILABLE = False
    print(f"[WARN] Memory service not available: {e}")


@router.post("/outcomes/sync")
async def sync_outcomes(
    target_date: Optional[str] = Query(None, description="Date cible (YYYY-MM-DD)"),
):
    """
    🔄 Synchronise les outcomes depuis les runs IA

    Récupère les résultats réels des courses et met à jour les outcomes.
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()
        target = None
        if target_date:
            from datetime import datetime

            target = datetime.strptime(target_date, "%Y-%m-%d").date()

        result = memory.sync_outcomes_from_runs(target)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur sync: {str(e)}")


@router.get("/outcomes")
async def list_outcomes(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    outcome: Optional[str] = Query(None, description="Filtre: WIN, LOSE, PENDING"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    """
    📋 Liste les outcomes avec filtres
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()

        from datetime import datetime

        df = datetime.strptime(date_from, "%Y-%m-%d").date() if date_from else None
        dt = datetime.strptime(date_to, "%Y-%m-%d").date() if date_to else None

        outcomes = memory.get_outcomes(limit, offset, outcome, df, dt)

        return {
            "success": True,
            "outcomes": outcomes,
            "total": len(outcomes),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/lessons")
async def list_lessons(
    active_only: bool = Query(True),
):
    """
    📚 Liste les leçons apprises par l'agent
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()
        lessons = memory.get_lessons(active_only)

        return {
            "success": True,
            "lessons": lessons,
            "total": len(lessons),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/stats")
async def get_agent_stats():
    """
    📊 Statistiques globales de l'agent IA

    Retourne:
    - Win rate, PnL total
    - Performance par jour
    - Performance par type de pari
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()
        stats = memory.get_stats()

        return {
            "success": True,
            **stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


# =============================================================================
# BACKTEST ENDPOINTS
# =============================================================================


class BacktestRequest(BaseModel):
    start_date: str = Field(..., description="Date début (YYYY-MM-DD)")
    end_date: str = Field(..., description="Date fin (YYYY-MM-DD)")
    profile: str = Field("STANDARD", description="Profil de risque")
    bankroll: float = Field(500, description="Bankroll en euros")


@router.post("/backtest/create")
async def create_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(None),
):
    """
    🧪 Crée un nouveau backtest et le lance automatiquement

    Lance une simulation de l'agent sur des données historiques.
    Limité à 7 jours maximum par défaut.
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()
        user_id, _, _ = _get_user_context(authorization)

        from datetime import datetime

        start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(request.end_date, "%Y-%m-%d").date()

        result = memory.create_backtest(
            start_date=start,
            end_date=end,
            profile=request.profile,
            bankroll=request.bankroll,
            user_id=user_id,
        )

        # Lancer le backtest en arrière-plan
        if result.get("success") and result.get("backtest_id"):
            background_tasks.add_task(memory.run_backtest, result["backtest_id"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/backtest/{backtest_id}/run")
async def run_backtest_manual(
    backtest_id: str,
    background_tasks: BackgroundTasks,
):
    """
    ▶️ Lance manuellement un backtest existant
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()
        backtest = memory.get_backtest(backtest_id)

        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest non trouvé")

        if backtest.get("status") == "RUNNING":
            return {"success": False, "error": "Backtest déjà en cours"}

        # Lancer en arrière-plan
        background_tasks.add_task(memory.run_backtest, backtest_id)

        return {
            "success": True,
            "message": "Backtest lancé en arrière-plan",
            "backtest_id": backtest_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/backtest/list")
async def list_backtests(
    limit: int = Query(20, ge=1, le=100),
):
    """
    📋 Liste les backtests
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()
        backtests = memory.get_backtests(limit)

        return {
            "success": True,
            "backtests": backtests,
            "total": len(backtests),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/backtest/{backtest_id}")
async def get_backtest(backtest_id: str):
    """
    🔍 Détails d'un backtest
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()
        backtest = memory.get_backtest(backtest_id)

        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest non trouvé")

        return {
            "success": True,
            **backtest,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/lessons/generate/{backtest_id}")
async def generate_lessons(backtest_id: str):
    """
    🧠 Génère des leçons à partir d'un backtest terminé
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()
        result = memory.generate_lessons_from_backtest(backtest_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/lessons/generate-all")
async def generate_all_lessons():
    """
    🔄 Régénère TOUTES les leçons à partir de tous les outcomes existants

    Utilisé par le cron quotidien après sync des outcomes.
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()
        result = memory.generate_lessons_all()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/lessons/for-prompt")
async def get_lessons_for_prompt(max_lessons: int = Query(10, ge=1, le=20)):
    """
    📖 Récupère les leçons formatées pour injection dans les prompts LLM
    """
    if not MEMORY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service mémoire non disponible")

    try:
        memory = get_memory_service()
        prompt_text = memory.get_lessons_for_prompt(max_lessons)
        return {"success": True, "lessons_text": prompt_text, "has_lessons": len(prompt_text) > 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


# =============================================================================
# TRAINING ENDPOINTS
# =============================================================================

try:
    from services.training_orchestrator import get_training_service

    TRAINING_AVAILABLE = True
except ImportError as e:
    TRAINING_AVAILABLE = False
    print(f"[WARN] Training service not available: {e}")


class TrainingRequest(BaseModel):
    start_date: str = Field(..., description="Date de début (YYYY-MM-DD)")
    end_date: str = Field(..., description="Date de fin (YYYY-MM-DD)")
    period_type: str = Field(default="WEEK", description="WEEK ou MONTH")


@router.post("/training/start")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
):
    """
    🎓 Démarre une nouvelle session d'entraînement automatisé

    Lance des backtests progressifs période par période et génère des leçons.
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service d'entraînement non disponible")

    try:
        from datetime import datetime

        training = get_training_service()

        start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(request.end_date, "%Y-%m-%d").date()

        result = training.start_training(start, end, request.period_type)

        if result.get("success") and result.get("session_id"):
            # Lancer l'entraînement en arrière-plan
            background_tasks.add_task(training.run_training_loop, result["session_id"])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/training/start-llm")
async def start_llm_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
):
    """
    🧠 Démarre un entraînement avec le VRAI Agent IA Gemini

    Différences avec /training/start:
    - Utilise vraiment Gemini pour analyser chaque jour
    - Les leçons sont injectées dans le prompt
    - Plus lent (1-2 min par jour) mais vrai apprentissage!
    - Traite jour par jour (pas par semaine)

    ⚠️ Limite: ~1500 jours max recommandés (quotas Gemini)
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service d'entraînement non disponible")

    try:
        from datetime import datetime, timedelta
        import asyncio

        training = get_training_service()

        start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(request.end_date, "%Y-%m-%d").date()

        # Pour le mode LLM, on compte en JOURS pas en semaines
        total_days = (end - start).days + 1

        # Vérifier que ce n'est pas trop long
        if total_days > 1500:
            raise HTTPException(
                status_code=400,
                detail=f"Période trop longue: {total_days} jours. Max recommandé: 1500 jours (~4 ans)",
            )

        # Créer la session avec comptage en jours
        con = training.get_connection()
        cur = con.cursor()

        import uuid as uuid_mod

        session_id = str(uuid_mod.uuid4())

        cur.execute(
            """
            INSERT INTO agent_training_sessions (
                session_id, start_date, end_date, period_type,
                status, current_period_start, current_period_end,
                total_periods, learning_curve
            ) VALUES (%s, %s, %s, 'DAY', 'RUNNING', %s, %s, %s, '[]')
            RETURNING session_id
        """,
            (session_id, start, end, start, start, total_days),
        )

        con.commit()
        con.close()

        # Lancer le training LLM en arrière-plan avec asyncio.run
        def run_llm_sync():
            import asyncio

            asyncio.run(training.run_llm_training_loop(session_id))

        background_tasks.add_task(run_llm_sync)

        return {
            "success": True,
            "session_id": session_id,
            "start_date": str(start),
            "end_date": str(end),
            "total_periods": total_days,
            "status": "RUNNING",
            "mode": "LLM",
            "message": f"🧠 Entraînement LLM démarré pour {total_days} jours. Estimation: ~{total_days * 2} minutes.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/training/list")
async def list_training_sessions(limit: int = Query(20, ge=1, le=100)):
    """
    📋 Liste les sessions d'entraînement
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service d'entraînement non disponible")

    try:
        training = get_training_service()
        sessions = training.list_sessions(limit)

        return {"success": True, "sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/training/{session_id}/pause")
async def pause_training(session_id: str):
    """
    ⏸️ Met en pause une session d'entraînement
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service d'entraînement non disponible")

    try:
        training = get_training_service()
        result = training.pause_training(session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.post("/training/{session_id}/resume")
async def resume_training(
    session_id: str,
    background_tasks: BackgroundTasks,
):
    """
    ▶️ Reprend une session d'entraînement en pause
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service d'entraînement non disponible")

    try:
        training = get_training_service()
        result = training.resume_training(session_id)

        if result.get("success"):
            # Relancer la boucle en arrière-plan
            background_tasks.add_task(training.run_training_loop, session_id)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/training/{session_id}")
async def get_training_progress(session_id: str):
    """
    📊 Récupère la progression d'une session d'entraînement
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service d'entraînement non disponible")

    try:
        training = get_training_service()
        session = training.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session non trouvée")

        return {"success": True, **session}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@router.get("/training/{session_id}/learning-curve")
async def get_learning_curve(session_id: str):
    """
    📈 Récupère la courbe d'apprentissage d'une session
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service d'entraînement non disponible")

    try:
        training = get_training_service()
        session = training.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session non trouvée")

        curve = session.get("learning_curve", [])

        # Calculer moyennes mobiles
        if len(curve) >= 3:
            for i in range(2, len(curve)):
                avg = sum(p["accuracy"] for p in curve[i - 2 : i + 1]) / 3
                curve[i]["moving_avg_3"] = round(avg, 2)

        return {"success": True, "session_id": session_id, "periods": len(curve), "curve": curve}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
