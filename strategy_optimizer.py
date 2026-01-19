import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import itertools
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))
sys.path.append(str(project_root / "web/backend"))

from db_connection import get_connection
from user_app_api import calculate_kelly_stake
from web.backend.main import run_benter_head_for_date, calculate_prediction_score

# Import ChampionPredictor
try:
    from services.champion_predictor import ChampionPredictor, default_champion_artifacts

    USE_CHAMPION = True
except ImportError:
    USE_CHAMPION = False

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("StrategyOptimizer")

START_DATE = "2025-11-18"
END_DATE = "2026-01-18"
BANKROLL_START = 1000.0

# =============================================================================
# DATA HELPERS
# =============================================================================


def get_dates_in_range(start_str, end_str):
    d1 = datetime.strptime(start_str, "%Y-%m-%d")
    d2 = datetime.strptime(end_str, "%Y-%m-%d")
    delta = d2 - d1
    return [(d1 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]


def get_safe_stats(runner_ids, race_date, cur):
    """
    Computes stats_chevaux equivalent fields on the fly using historical data only.
    """
    if not runner_ids:
        return {}

    query = """
        SELECT id_cheval, place_finale
        FROM cheval_courses_seen
        WHERE id_cheval = ANY(%s)
        AND SUBSTRING(race_key FROM 1 FOR 10) < %s
        AND place_finale IS NOT NULL
        ORDER BY race_key DESC
    """
    cur.execute(query, (runner_ids, race_date))
    rows = cur.fetchall()

    history_map = {}
    for r in rows:
        history_map.setdefault(r[0], []).append(r[1])

    stats = {}
    for cid in runner_ids:
        places = history_map.get(cid, [])
        last_5 = places[:5]
        forme_5c = (sum(last_5) / len(last_5)) if last_5 else 50.0

        # Simple proxy for other stats just to feed the model shape
        place_rate = (sum(1 for p in places if p <= 3) / len(places) * 100) if places else 0.0
        regularite = np.std(places) if len(places) > 1 else 0.0

        stats[cid] = (forme_5c, place_rate, place_rate, place_rate, regularite)
    return stats


def _build_features_safe(runners, race_date, cur, stats_map):
    """
    Minimal feature builder for speed
    """
    ids = [r.get("id_cheval") for r in runners if r.get("id_cheval")]
    if not ids:
        return []

    sc = stats_map

    # Minimal History Stats
    cur.execute(
        """
        SELECT id_cheval, COUNT(*), SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END)
        FROM cheval_courses_seen
        WHERE id_cheval = ANY(%s)
        AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date
        AND SUBSTRING(race_key FROM 1 FOR 10)::date >= (%s::date - INTERVAL '365 days')
        GROUP BY id_cheval
        """,
        (ids, race_date, race_date),
    )
    hist_raw = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    # Minimal Jockey/Trainer Stats
    jockeys = list(set(r.get("jockey", "") for r in runners))
    trainers = list(set(r.get("entraineur", "") for r in runners))

    j_stats = {}
    if jockeys:
        cur.execute(
            """
            SELECT driver_jockey, COUNT(*), SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END)
            FROM cheval_courses_seen WHERE driver_jockey = ANY(%s)
            AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date AND SUBSTRING(race_key FROM 1 FOR 10)::date >= (%s::date - INTERVAL '365 days')
            GROUP BY driver_jockey
        """,
            (jockeys, race_date, race_date),
        )
        for row in cur.fetchall():
            j_stats[row[0]] = (row[2] / row[1]) if row[1] else 0.0

    t_stats = {}
    if trainers:
        cur.execute(
            """
            SELECT entraineur, COUNT(*), SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END)
            FROM cheval_courses_seen WHERE entraineur = ANY(%s)
            AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date AND SUBSTRING(race_key FROM 1 FOR 10)::date >= (%s::date - INTERVAL '365 days')
            GROUP BY entraineur
        """,
            (trainers, race_date, race_date),
        )
        for row in cur.fetchall():
            t_stats[row[0]] = (row[2] / row[1]) if row[1] else 0.0

    # Race Context
    try:
        rk = runners[0]["race_key"]
        cur.execute(
            """
            SELECT id_cheval, draw_stalle, cote_matin, meteo_code, temperature_c, vent_kmh, allocation_totale, etat_piste, sexe, age
            FROM cheval_courses_seen WHERE race_key = %s AND id_cheval = ANY(%s)
        """,
            (rk, ids),
        )
        extra = {r[0]: r[1:] for r in cur.fetchall()}
    except:
        extra = {}

    feature_rows = []
    nb_partants = len(runners)
    mean_forme = np.mean([sc.get(r["id_cheval"], (50,))[0] for r in runners])

    cotes = [r.get("cote_preoff") or 999 for r in runners]
    sorted_idx = np.argsort(cotes)
    rank_map = {original_idx: rank + 1 for rank, original_idx in enumerate(sorted_idx)}

    for idx, r in enumerate(runners):
        cid = r["id_cheval"]
        f5c, apt_dist, apt_piste, apt_hippo, reg = sc.get(cid, (50.0, 0.0, 0.0, 0.0, 0.0))
        nb_c, nb_p = hist_raw.get(cid, (0, 0))

        j_rate = j_stats.get(r.get("jockey"), 0.0)
        t_rate = t_stats.get(r.get("entraineur"), 0.0)

        ex = extra.get(cid, [0] * 9)
        # ex: draw, cm, met, temp, vent, alloc, etat, sexe, age

        row_dict = {
            "numero_corde": float(ex[0] or 0),
            "cote_sp": float(r.get("cote_preoff") or 0),
            "cote_pm": float(ex[1] or r.get("cote_preoff") or 0),
            "distance": float(r.get("distance") or 2000),
            "nombre_partants": float(nb_partants),
            "allocation": float(ex[5] or 0),
            "forme_5c": float(f5c),
            "forme_10c": float(f5c),  # proxy
            "nb_courses_12m": float(nb_c),
            "nb_victoires_12m": 0.0,  # omitted for speed/proxy
            "nb_places_12m": float(nb_p),
            "regularite": float(reg),
            "jours_depuis_derniere": 30.0,  # default
            "aptitude_distance": float(apt_dist),
            "aptitude_piste": float(apt_piste),
            "aptitude_hippodrome": float(apt_hippo),
            "synergie_jockey_cheval": float(j_rate * f5c),
            "synergie_entraineur_cheval": float(t_rate * f5c),
            "jockey_win_rate": float(j_rate),
            "jockey_place_rate": float(j_rate),  # proxy
            "entraineur_win_rate": float(t_rate),
            "entraineur_place_rate": float(t_rate),  # proxy
            "distance_norm": float(r.get("distance", 2000) / 1000),
            "niveau_moyen_concurrent": float(mean_forme),
            "rang_cote_sp": float(rank_map.get(idx, 1)),
            "an_naissance": 2020.0,  # default
            "age": float(ex[8] or 0),
            "temperature_c": float(ex[3] or 15),
            "vent_kmh": float(ex[4] or 10),
            "meteo_code": 0.0,
            "interaction_forme_jockey": float(f5c * j_rate),
            "interaction_aptitude_distance": float(apt_dist * (r.get("distance", 2000) / 1000)),
            "interaction_synergie_forme": float(2 * j_rate * f5c),
            "interaction_aptitude_popularite": 0.0,
            "interaction_regularite_volume": float(reg * nb_c),
            "discipline_Plat": 1.0 if "plat" in r.get("discipline", "").lower() else 0.0,
            "discipline_Trot": 1.0 if "trot" in r.get("discipline", "").lower() else 0.0,
            "discipline_Obstacle": 0.0,
            "sexe_H": 1.0 if str(ex[7]).startswith("H") else 0.0,
            "sexe_M": 1.0 if str(ex[7]).startswith("M") else 0.0,
            "sexe_F": 1.0 if str(ex[7]).startswith("F") else 0.0,
            "etat_Bon": 1.0 if "bon" in str(ex[6]).lower() else 0.0,
            "etat_Souple": 1.0 if "souple" in str(ex[6]).lower() else 0.0,
            "etat_Lourd": 0.0,
            "etat_PSF": 0.0,
            "etat_Leger": 0.0,
        }
        feature_rows.append(row_dict)
    return feature_rows


# =============================================================================
# OPTIMIZATION ENGINE
# =============================================================================


def load_period_data():
    """
    Loads all race data for the period into memory.
    """
    print(f"📥 Loading data from {START_DATE} to {END_DATE}...")

    conn = get_connection()
    cur = conn.cursor()
    dates = get_dates_in_range(START_DATE, END_DATE)

    predictor = None
    if USE_CHAMPION:
        try:
            predictor = ChampionPredictor(default_champion_artifacts())
        except Exception as e:
            print(f"⚠️ Champion Error: {e}")

    all_races = []

    for d in dates:
        print(".", end="", flush=True)
        # Benter (re-run logic to get probs)
        try:
            conn.rollback()
            benter_res = run_benter_head_for_date(d, cur=cur)
            b_map = benter_res.get("by_runner", {})
        except:
            b_map = {}

        cur.execute(
            """
            SELECT race_key, id_cheval, numero_dossard, cote_reference, cote_finale, is_win, place_finale, nom_norm, driver_jockey, entraineur, discipline, distance_m, tendance_cote, amplitude_tendance, est_favori, avis_entraineur, etat_piste
            FROM cheval_courses_seen
            WHERE race_key LIKE %s
            AND cote_finale IS NOT NULL AND cote_finale > 0 AND cote_finale < 200
        """,
            (d + "%",),
        )

        rows = cur.fetchall()

        # Group by race
        races_dict = {}
        for r in rows:
            rk = r[0]
            races_dict.setdefault(rk, []).append(
                {
                    "race_key": rk,
                    "id_cheval": r[1],
                    "numero": r[2],
                    "cote_ref": r[3],
                    "cote_fin": r[4],
                    "is_win": (r[5] == 1 or r[6] == 1),
                    "nom": r[7],
                    "jockey": r[8],
                    "entraineur": r[9],
                    "discipline": r[10],
                    "distance": r[11],
                    "tendance": r[12],
                    "amplitude": r[13],
                    "est_favori": r[14],
                    "avis": r[15],
                    "etat": r[16],
                }
            )

        # Process races
        for rk, runners in races_dict.items():
            if not runners:
                continue

            # Calculate Benter Probs
            for run in runners:
                # Resolve Odds
                c_ref = run["cote_ref"]
                c_fin = run["cote_fin"]
                odds_input = c_ref if (c_ref and c_ref > 1) else c_fin
                run["cote_preoff"] = odds_input  # store for features

                # Benter Prob from Map or Fallback
                b_key = (rk, run["numero"])
                b_info = b_map.get(b_key)
                if not b_info:  # Try by name matches sometimes fail?
                    # check (rk, nom)
                    pass

                if b_info:
                    p_benter = b_info.get("p_calibrated") or b_info.get("p_model_norm") or 0
                else:
                    # Fallback amélioré : utilise la cote de référence comme en production
                    score = calculate_prediction_score(
                        odds_input,
                        c_ref,
                        run["tendance"],
                        run["amplitude"],
                        run["est_favori"],
                        run["avis"],
                    )
                    prob_imp = 1.0 / odds_input if odds_input > 0 else 0.01
                    # Ajustement basé sur le score (comme dans user_app_api)
                    adjustment = (score - 50) / 100  # Plus conservateur
                    p_benter = max(0.01, min(0.95, prob_imp * (1 + adjustment)))

                run["p_benter"] = p_benter * 100  # Store as percentage 0-100
                run["p_champion"] = 0.0

            # Calculate Champion Probs batch
            if USE_CHAMPION and predictor:
                try:
                    stats = get_safe_stats([x["id_cheval"] for x in runners], d, cur)
                    feats = _build_features_safe(runners, d, cur, stats)
                    if feats:
                        preds = predictor.predict_proba(feats)
                        for i, run in enumerate(runners):
                            run["p_champion"] = preds[i] * 100  # Store as percentage
                except Exception as e:
                    print(f" (Ch err: {e})", end="")
                    # Important: Rollback if query failed to prevent 'current transaction is aborted'
                    try:
                        cur.connection.rollback()
                    except:
                        pass

            # Clean up runner object for memory
            clean_runners = []
            for run in runners:
                # STRATEGIE REELLE :
                # - Decision sur COTE REFERENCE (opening odds, disponibles à T-5min)
                # - Paiement sur COTE FINALE (SP, résultat officiel)
                # C'est là que se trouve la VALUE !
                cote_decision = (
                    run["cote_ref"]
                    if (run["cote_ref"] and run["cote_ref"] > 1)
                    else run["cote_fin"]
                )
                cote_payout = run["cote_fin"]

                clean_runners.append(
                    {
                        "odds": cote_decision,  # ✅ COTE REFERENCE pour décision
                        "odds_result": cote_payout,  # ✅ COTE FINALE pour payout
                        "is_win": run["is_win"],
                        "p_benter": run["p_benter"],
                        "p_champion": run["p_champion"],
                    }
                )

            all_races.append({"date": d, "key": rk, "runners": clean_runners})

    conn.close()
    print(f"\n✅ Loaded {len(all_races)} races.")
    return all_races


def run_simulation(races, params):
    """
    Fast simulation on pre-loaded data
    """
    bankroll = BANKROLL_START
    bets = 0
    wins = 0
    staked = 0.0
    returned = 0.0

    kf = params["kelly_fraction"]
    vt = params["value_threshold"]
    mp = params["min_prob"]
    mo = params["max_odds"]
    cw = params["champion_weight"]
    bw = 1.0 - cw

    for race in races:
        for r in race["runners"]:
            p_b = r["p_benter"]
            p_c = r["p_champion"]

            if p_c == 0 and cw > 0:
                p_final = p_b  # Fallback
            else:
                p_final = (p_b * bw) + (p_c * cw)

            odds = r["odds"]
            if odds <= 1.0:
                continue

            # Filters
            if p_final < mp:
                continue
            if odds > mo:
                continue

            expected_odds = 100.0 / p_final if p_final > 0 else 999
            value_pct = ((odds / expected_odds) - 1.0) * 100.0

            if value_pct < vt:
                continue

            # Stake (Kelly avec seuils de production)
            stake_pct = calculate_kelly_stake(p_final, odds, kelly_fraction=kf)
            if stake_pct <= 0:
                continue

            # Standard strategy: Base Stake * Percentage
            mise = (stake_pct / 100.0) * BANKROLL_START

            # Mise minimum de 5€ (comme en production)
            if mise > 0 and mise < 5.0:
                mise = 5.0

            # Cap à 5% de la bankroll (comme user_app_api.py)
            max_mise = BANKROLL_START * 0.05
            if mise > max_mise:
                mise = max_mise

            # Skip si mise finale < 2€ (filtre qualité)
            if mise < 2.0:
                continue

            bets += 1
            staked += mise

            if r["is_win"]:
                winnings = mise * r["odds_result"]
                returned += winnings
                wins += 1

    profit = returned - staked
    roi = (profit / staked * 100.0) if staked > 0 else -999.0

    return {
        "params": params,
        "bets": bets,
        "staked": staked,
        "profit": profit,
        "roi": roi,
        "wins": wins,
    }


def print_result(res, rank):
    p = res["params"]
    print(
        f"#{rank:<2} | ROI: {res['roi']:>6.2f}% | P/L: {res['profit']:>8.2f}€ | Bets: {res['bets']:>4} | "
        f"Kelly: {p['kelly_fraction']:.2f}, Val: {p['value_threshold']:>3}%, "
        f"MinP: {p['min_prob']:>2}%, W_Champ: {p['champion_weight']:.1f}"
    )


if __name__ == "__main__":
    # 1. Load Data
    data = load_period_data()

    if not data:
        print("❌ No data loaded.")
        sys.exit(1)

    # 2. Define Grid
    # ⚠️ GRILLE REALISTE basée sur la stratégie de production à +26% ROI
    grid = {
        "kelly_fraction": [0.20, 0.25, 0.30],  # Fraction Kelly standard
        "value_threshold": [-10, -5, 0],  # Value négative acceptée (opportunities)
        "min_prob": [5, 10, 15],  # Probabilités plus basses pour coverage
        "max_odds": [20, 30, 50],  # Cotes moyennes à élevées
        "champion_weight": [0.3, 0.5, 0.7],  # Mix Benter/Champion (production ~0.5)
    }

    keys, values = zip(*grid.items())
    param_sets = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\n🔬 Testing {len(param_sets)} strategies...")

    results = []
    # No multithreading here to avoid complexity with results, strict sequential is fast enough for 6k items
    for params in param_sets:
        res = run_simulation(data, params)
        results.append(res)

    # 3. Sort & Report
    valid_results = [r for r in results if r["bets"] > 20]
    if not valid_results:
        valid_results = results

    sorted_results = sorted(valid_results, key=lambda x: x["profit"], reverse=True)

    print("\n🏆 TOP 10 STRATEGIES (by Profit):")
    print("-" * 80)
    for i, r in enumerate(sorted_results[:10]):
        print_result(r, i + 1)

    print("-" * 80)
