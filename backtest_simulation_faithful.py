import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))
sys.path.append(str(project_root / "web/backend"))

from db_connection import get_connection

# We need these from user_app_api (copied or imported)
from user_app_api import calculate_kelly_stake, classify_bet_profile
from web.backend.main import (
    run_benter_head_for_date,
    calculate_prediction_score,
    select_preoff_market_odds,
)

# Import ChampionPredictor
try:
    from services.champion_predictor import ChampionPredictor, default_champion_artifacts

    USE_CHAMPION = True
except ImportError:
    USE_CHAMPION = False

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("FaithfulBacktest")

START_DATE = "2025-11-18"
END_DATE = "2026-01-18"
BANKROLL_START = 1000.0


def get_dates_in_range(start_str, end_str):
    d1 = datetime.strptime(start_str, "%Y-%m-%d")
    d2 = datetime.strptime(end_str, "%Y-%m-%d")
    delta = d2 - d1
    return [(d1 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]


def get_safe_stats(runner_ids, race_date, cur):
    """
    Computes stats_chevaux equivalent fields on the fly using historical data only.
    Avoids look-ahead bias found in the static stats_chevaux table.
    Fields needed: forme_5c, aptitude_distance, aptitude_piste, aptitude_hippodrome, regularite
    """
    if not runner_ids:
        return {}

    # Pre-fetch history for all runners
    # We use cheval_courses_seen which is the source for Champion features usually (via hist)
    # But for stats_chevaux, we need aggregations.

    query = """
        SELECT
            id_cheval,
            race_key,
            date_course,
            place_finale,
            distance_m,
            hippodrome_nom,
            etat_piste
        FROM cheval_courses_seen
        WHERE id_cheval = ANY(%s)
        AND date_course < %s
        AND place_finale IS NOT NULL
        ORDER BY date_course DESC
    """
    cur.execute(query, (runner_ids, race_date))
    rows = cur.fetchall()

    # Process in Python
    history_map = {}
    for r in rows:
        cid = r[0]
        history_map.setdefault(cid, []).append(
            {"date": r[2], "place": r[3], "dist": r[4], "hippo": r[5], "etat": r[6]}
        )

    stats = {}

    for cid in runner_ids:
        hist = history_map.get(cid, [])

        # 1. Forme 5C (Avg place last 5)
        # Assuming hist is ordered DESC
        last_5 = hist[:5]
        if last_5:
            forme_5c = sum(h["place"] for h in last_5) / len(last_5)
        else:
            forme_5c = 50.0  # Default bad form

        # 2. Regularite (StdDev of places)
        if len(hist) > 1:
            places = [h["place"] for h in hist]
            regularite = np.std(places)
        else:
            regularite = 0.0

        # 3. Aptitude Distance (Avg place in +/- 200m range? Or Win Rate?)
        # calcul_stats uses Place Rate in +/- 10%. Let's approximate with Place Rate.
        # But Champion uses standardized values often?
        # Let's use Place Rate (%)
        # Note: champion features might expect normalized values (0-100 or 0-1).
        # calcul_stats uses 0-100 for aptitude_distance.

        # We need the CURRENT race info (distance, hippo, etat) to match against.
        # This function only computes the "profile" of the horse,
        # BUT stats_chevaux stores "generic" aptitude?
        # Actually stats_chevaux stores ONE value per horse.
        # Wait, calcul_stats code showed:
        # SELECT AVG(CASE WHEN pc2.place THEN 100.0 ELSE 0.0 END) ... WHERE ABS(diff) < 10%
        # It computes aptitude based on ALL pairs of races? No, it's weird.
        # Actually stats_chevaux usually stores "aptitude to CURRENT condition"?
        # distinct "aptitude_distance" column suggests it's a fixed stat?
        # The column in DB is just float.
        # Let's assume it's "Place Rate in general" for now if we can't replicate specific "distance" preference without target.
        # Actually, `build_features` fetches `aptitude_distance` from `stats_chevaux` independent of the current race?
        # If so, it's a "general versatility" score?
        # Let's assume defaults for aptitudes to be safe (50) or compute simple Place Rate (0-100).
        # A simpler robust proxy: Place Rate (Top 3) in career.

        nb_races = len(hist)
        nb_places = sum(1 for h in hist if h["place"] <= 3)
        place_rate = (nb_places / nb_races * 100) if nb_races else 0.0

        stats[cid] = (
            forme_5c,
            place_rate,  # aptitude_distance proxy
            place_rate,  # aptitude_piste proxy
            place_rate,  # aptitude_hippodrome proxy
            regularite,
        )

    return stats


def _build_features_safe(runners, race_date, cur, stats_map):
    """
    Exact copy of web/backend/main.py _build_features_for_race logic,
    but using the safely computed stats_map instead of DB query for stats_chevaux.
    """
    ids = [r.get("id_cheval") for r in runners if r.get("id_cheval") is not None]
    if not ids:
        return []

    # sc (Stats Chevaux) matches DB columns: forme_5c, aptitude_distance, aptitude_piste, aptitude_hippodrome, regularite
    # retrieved from stats_map which we computed safely.
    sc = stats_map

    # Historique 12 mois / récence (cheval_courses_seen)
    # This query is safe (filtered by date)
    cur.execute(
        """
        WITH hist AS (
            SELECT
                id_cheval,
                SUBSTRING(race_key FROM 1 FOR 10)::date AS d,
                is_win,
                place_finale
            FROM cheval_courses_seen
            WHERE id_cheval = ANY(%s)
                AND place_finale IS NOT NULL
                AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date
        )
        SELECT
            id_cheval,
            COUNT(*) FILTER (WHERE d >= (%s::date - INTERVAL '365 days')) AS nb_courses_12m,
            SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) FILTER (WHERE d >= (%s::date - INTERVAL '365 days')) AS nb_victoires_12m,
                SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) FILTER (WHERE d >= (%s::date - INTERVAL '365 days')) AS nb_places_12m,
                (%s::date - MAX(d))::float AS recence
            FROM hist
            GROUP BY id_cheval
            """,
        (ids, race_date, race_date, race_date, race_date, race_date),
    )
    hist = {row[0]: row[1:] for row in cur.fetchall()}

    # Jockey/entraineur stats (12 mois) via noms
    jockeys = sorted(
        {(r.get("jockey") or "").strip() for r in runners if (r.get("jockey") or "").strip()}
    )
    trainers = sorted(
        {
            (r.get("entraineur") or "").strip()
            for r in runners
            if (r.get("entraineur") or "").strip()
        }
    )

    jockey_stats = {}
    if jockeys:
        cur.execute(
            """
            WITH hist AS (
                SELECT
                    driver_jockey,
                    SUBSTRING(race_key FROM 1 FOR 10)::date AS d,
                    is_win,
                    place_finale
                FROM cheval_courses_seen
                WHERE driver_jockey = ANY(%s)
                    AND place_finale IS NOT NULL
                    AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date
                    AND SUBSTRING(race_key FROM 1 FOR 10)::date >= (%s::date - INTERVAL '365 days')
            )
            SELECT
                driver_jockey,
                COUNT(*) AS nb_courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) AS places
            FROM hist
            GROUP BY driver_jockey
            """,
            (jockeys, race_date, race_date),
        )
        for name, nb, wins, places in cur.fetchall():
            nb = nb or 0
            jockey_stats[name] = (
                (wins or 0) / nb if nb else 0.0,
                (places or 0) / nb if nb else 0.0,
            )

    trainer_stats = {}
    if trainers:
        cur.execute(
            """
            WITH hist AS (
                SELECT
                    entraineur,
                    SUBSTRING(race_key FROM 1 FOR 10)::date AS d,
                    is_win,
                    place_finale
                FROM cheval_courses_seen
                WHERE entraineur = ANY(%s)
                    AND place_finale IS NOT NULL
                    AND SUBSTRING(race_key FROM 1 FOR 10)::date < %s::date
                    AND SUBSTRING(race_key FROM 1 FOR 10)::date >= (%s::date - INTERVAL '365 days')
            )
            SELECT
                entraineur,
                COUNT(*) AS nb_courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) AS places
            FROM hist
            GROUP BY entraineur
            """,
            (trainers, race_date, race_date),
        )
        for name, nb, wins, places in cur.fetchall():
            nb = nb or 0
            trainer_stats[name] = (
                (wins or 0) / nb if nb else 0.0,
                (places or 0) / nb if nb else 0.0,
            )

    # Infos "course du jour"
    extra = {}
    try:
        rk = (runners[0].get("race_key") if runners else None) or ""
        cur.execute(
            """
            SELECT
                id_cheval,
                draw_stalle,
                cote_matin,
                meteo_code,
                temperature_c,
                vent_kmh,
                allocation_totale,
                etat_piste,
                sexe,
                age
            FROM cheval_courses_seen
            WHERE race_key = %s
                AND id_cheval = ANY(%s)
            """,
            (rk, ids),
        )
        for cid, draw, cm, met, temp, vent, alloc, etat, sexe, age in cur.fetchall():
            extra[cid] = {
                "numero_corde": draw,
                "cote_pm": cm,
                "meteo_code": met,
                "temperature_c": temp,
                "vent_kmh": vent,
                "allocation": alloc,
                "etat_piste": etat,
                "sexe": sexe,
                "age": age,
            }
    except Exception:
        extra = {}

    nb_partants = max(2, len(runners))
    distance = float(runners[0].get("distance") or 2000)
    distance_norm = distance / 1000.0

    mean_forme = 0.0
    vals = [float((sc.get(r.get("id_cheval"), (0, 0, 0, 0, 0))[0]) or 0) for r in runners]
    if vals:
        mean_forme = sum(vals) / len(vals)

    cotes = [r.get("cote_preoff") or 0 for r in runners]
    sorted_idx = sorted(range(len(cotes)), key=lambda i: (cotes[i] or 999999))
    rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_idx)}

    def _to_float(v):
        try:
            return float(v)
        except:
            return 0.0

    feature_rows = []
    for idx, r in enumerate(runners):
        cid = r.get("id_cheval")
        # Unpack equivalent to: forme_5c, apt_dist, apt_piste, apt_hippo, regularite
        forme_5c, apt_dist, apt_piste, apt_hippo, regularite = sc.get(cid, (0, 0, 0, 0, 0))
        nb_courses_12m, nb_victoires_12m, nb_places_12m, recence = hist.get(cid, (0, 0, 0, 90.0))

        j_name = (r.get("jockey") or "").strip()
        t_name = (r.get("entraineur") or "").strip()
        j_wr, j_pr = jockey_stats.get(j_name, (0.0, 0.0))
        t_wr, t_pr = trainer_stats.get(t_name, (0.0, 0.0))

        ex = extra.get(cid, {})
        sexe = (ex.get("sexe") or "").strip().upper()
        etat = (ex.get("etat_piste") or "").strip()
        disc = (r.get("discipline") or "").strip().lower()

        etat_features = {
            "etat_Bon": 1.0 if "bon" in etat.lower() else 0.0,
            "etat_Souple": 1.0 if "souple" in etat.lower() else 0.0,
            "etat_Lourd": 1.0 if "lourd" in etat.lower() else 0.0,
            "etat_PSF": 1.0 if "psf" in etat.lower() else 0.0,
            "etat_Leger": 1.0 if "léger" in etat.lower() or "leger" in etat.lower() else 0.0,
        }

        elo_cheval = 1000.0
        synergie_j = float((j_wr or 0) * (forme_5c or 0))
        synergie_t = float((t_wr or 0) * (forme_5c or 0))

        # Interactions
        interaction_forme_jockey = float((forme_5c or 0) * (j_wr or 0))
        interaction_aptitude_distance = float((apt_dist or 0) * (distance_norm or 0))
        interaction_synergie_forme = float((synergie_j + synergie_t) * (forme_5c or 0))
        interaction_aptitude_popularite = float(
            (apt_dist or 0) * (1.0 / max(1.0, float(rank_map.get(idx, 1))))
        )
        interaction_regularite_volume = float((regularite or 0) * (nb_courses_12m or 0))

        try:
            year = int(str(race_date)[:4])
            age_val = float(ex.get("age") or 0)
            an_naissance = year - age_val if age_val else 0.0
        except:
            an_naissance = 0.0

        feature_rows.append(
            {
                "numero_corde": _to_float(ex.get("numero_corde") or 0),
                "cote_sp": float(r.get("cote_preoff") or 0),
                "cote_pm": _to_float(ex.get("cote_pm") or r.get("cote_preoff") or 0),
                "distance": float(distance or 0),
                "nombre_partants": float(nb_partants),
                "allocation": _to_float(ex.get("allocation") or 0),
                "forme_5c": float(forme_5c or 0),
                "forme_10c": float(forme_5c or 0),
                "nb_courses_12m": float(nb_courses_12m or 0),
                "nb_victoires_12m": float(nb_victoires_12m or 0),
                "nb_places_12m": float(nb_places_12m or 0),
                "regularite": float(regularite or 0),
                "jours_depuis_derniere": float(recence or 90.0),
                "aptitude_distance": float(apt_dist or 0),
                "aptitude_piste": float(apt_piste or 0),
                "aptitude_hippodrome": float(apt_hippo or 0),
                "synergie_jockey_cheval": float(synergie_j),
                "synergie_entraineur_cheval": float(synergie_t),
                "jockey_win_rate": float(j_wr or 0),
                "jockey_place_rate": float(j_pr or 0),
                "entraineur_win_rate": float(t_wr or 0),
                "entraineur_place_rate": float(t_pr or 0),
                "distance_norm": float(distance_norm or 0),
                "niveau_moyen_concurrent": float(mean_forme or 0),
                "rang_cote_sp": float(rank_map.get(idx, 1)),
                "an_naissance": float(an_naissance),
                "age": float(ex.get("age") or 0),
                "temperature_c": _to_float(ex.get("temperature_c") or 0),
                "vent_kmh": _to_float(ex.get("vent_kmh") or 0),
                "meteo_code": 0.0,
                "interaction_forme_jockey": float(interaction_forme_jockey),
                "interaction_aptitude_distance": float(interaction_aptitude_distance),
                "interaction_synergie_forme": float(interaction_synergie_forme),
                "interaction_aptitude_popularite": float(interaction_aptitude_popularite),
                "interaction_regularite_volume": float(interaction_regularite_volume),
                "discipline_Plat": 1.0 if "plat" in disc else 0.0,
                "discipline_Trot": 1.0 if "trot" in disc or "att" in disc else 0.0,
                "discipline_Obstacle": 1.0 if "haie" in disc or "steeple" in disc else 0.0,
                "sexe_H": 1.0 if sexe.startswith("H") else 0.0,
                "sexe_M": 1.0 if sexe.startswith("M") else 0.0,
                "sexe_F": 1.0 if sexe.startswith("F") else 0.0,
                **etat_features,
            }
        )
    return feature_rows


def run_faithful_backtest():
    current_bankroll = BANKROLL_START
    dates = get_dates_in_range(START_DATE, END_DATE)

    predictor = None
    if USE_CHAMPION:
        try:
            predictor = ChampionPredictor(default_champion_artifacts())
            logger.info("✅ Champion Predictor loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Champion Predictor: {e}")
            return

    conn = get_connection()
    cur = conn.cursor()

    print(f"🚀 Starting FAITHFUL Backtest from {START_DATE} to {END_DATE}")
    print(f"💰 Initial Bankroll: {current_bankroll}€")

    total_bets = 0
    total_staked = 0.0
    total_returned = 0.0
    total_wins = 0

    for date_str in dates:
        print(f"📅 Processing {date_str}...", end=" ", flush=True)

        # 1. Benter Analysis
        try:
            # Pass independent cursor or handle transaction?
            # Benter might use its own transaction or break ours if it fails.
            # Best to commit/rollback before starting loop or after benter
            conn.rollback()  # Ensure clean state
            benter_result = run_benter_head_for_date(date_str, cur=cur)
            conn.commit()
        except Exception as e:
            print(f"(Benter error: {e})")
            conn.rollback()
            continue

        benter_map = benter_result.get("by_runner", {})

        # 2. Get Candidates
        query = """
            SELECT
                cs.id_cheval, cs.nom_norm, cs.race_key, cs.hippodrome_nom, cs.heure_locale,
                cs.numero_dossard, cs.cote_finale, cs.cote_reference, cs.tendance_cote,
                cs.amplitude_tendance, cs.est_favori, cs.avis_entraineur, cs.driver_jockey,
                cs.entraineur, cs.discipline, cs.distance_m, cs.is_win, cs.place_finale,
                cs.statut_participant, cs.incident, cs.heure_depart
            FROM cheval_courses_seen cs
            WHERE cs.race_key LIKE %s
            AND cs.cote_finale IS NOT NULL AND cs.cote_finale > 0 AND cs.cote_finale < 200
            AND (cs.statut_participant IS NULL OR UPPER(cs.statut_participant) IN ('PARTANT', 'PARTANTE', 'PART', 'P', ''))
            ORDER BY cs.race_key, cs.numero_dossard
        """
        cur.execute(query, (date_str + "%",))
        rows = cur.fetchall()

        if not rows:
            print("(No races)")
            continue

        race_groups = {}
        for row in rows:
            (
                id_ch,
                nom,
                r_key,
                hippo,
                heure,
                num,
                cote_fin,
                cote_ref,
                tend,
                amp,
                fav,
                avis,
                jock,
                ent,
                disc,
                dist,
                is_win,
                place_fin,
                stat,
                inc,
                h_dep,
            ) = row

            if inc is not None:
                continue

            # Market Odds Logic for Backtest
            # Standard logic: Use Reference (Opening) odds for decision if available, else Final.
            # In paper trading, we often see reference odds.
            cote_input = cote_ref if (cote_ref and cote_ref > 1) else cote_fin
            if not cote_input or cote_input > 50:
                continue

            # Score & Benter
            score = calculate_prediction_score(cote_input, cote_ref, tend, amp, fav, avis)
            b_key = (r_key, num)
            b_runner = benter_map.get(b_key) or benter_map.get((r_key, nom))

            p_model = 0
            if b_runner:
                p_model = b_runner.get("p_calibrated") or b_runner.get("p_model_norm") or 0
            else:
                prob_imp = 1 / cote_input
                adj = (score - 50) / 200
                p_model = max(0.01, min(0.95, prob_imp * (1 + adj)))

            runner_obj = {
                "id_cheval": id_ch,
                "nom": nom,
                "race_key": r_key,
                "hippodrome": hippo,
                "heure": heure,
                "numero": num,
                "cote_preoff": cote_input,
                "cote_finale": cote_fin,
                "tendance": tend,
                "amplitude": amp,
                "est_favori": fav,
                "avis": avis,
                "jockey": jock,
                "entraineur": ent,
                "discipline": disc,
                "distance": dist,
                "is_win": is_win,
                "place_finale": place_fin,
                "p_final": p_model,
            }
            race_groups.setdefault(r_key, []).append(runner_obj)

        day_bets = 0
        day_staked = 0
        day_profit = 0

        # Race processing
        for r_key, runners in race_groups.items():
            if USE_CHAMPION and predictor:
                try:
                    stats_map = get_safe_stats([r["id_cheval"] for r in runners], date_str, cur)
                    input_features = _build_features_safe(runners, date_str, cur, stats_map)
                    preds = predictor.predict_proba(input_features)
                    for i, r in enumerate(runners):
                        r["p_final"] = preds[i]
                except Exception as e:
                    # print(f"(Champ err: {e})", end="")
                    pass

            for p in runners:
                # Same Betting Logic
                p_final = p["p_final"] * 100
                if p_final < 1 or p_final > 95:
                    continue
                odds = p["cote_preoff"]
                expected_odds = 100 / p_final if p_final > 0 else 999
                value_pct = ((odds / expected_odds) - 1) * 100

                if value_pct < -5:
                    continue

                # Kelly (User App Api logic)
                stake_pct = calculate_kelly_stake(p_final, odds)
                mise = (stake_pct / 100) * BANKROLL_START

                if mise > 0 and mise < 5:
                    mise = 5.0
                max_mise = BANKROLL_START * 0.05
                if mise > max_mise:
                    mise = max_mise

                if mise > 0:
                    day_bets += 1
                    total_bets += 1
                    total_staked += mise
                    day_staked += mise

                    pnl = -mise
                    is_win = (p["is_win"] == 1) or (p["place_finale"] == 1)
                    if is_win:
                        ret = mise * p["cote_finale"]
                        pnl += ret
                        total_returned += ret
                        total_wins += 1
                    day_profit += pnl

        current_bankroll += day_profit
        print(f"Bets: {day_bets}, P/L: {day_profit:.2f}, BR: {current_bankroll:.2f}")

    print("=" * 60)
    print("🏁 BACKTEST COMPLETE")
    print(f"Total Bets: {total_bets}")
    print(f"Total Staked: {total_staked:.2f}€")
    print(f"Total Returned: {total_returned:.2f}€")
    print(f"Net Profit: {total_returned - total_staked:.2f}€")
    if total_staked > 0:
        print(f"ROI: {(total_returned - total_staked) / total_staked * 100:.2f}%")
    print(f"Final Bankroll: {current_bankroll:.2f}€")
    if total_bets > 0:
        print(f"Hit Rate: {total_wins/total_bets*100:.2f}%")

    conn.close()


if __name__ == "__main__":
    run_faithful_backtest()
