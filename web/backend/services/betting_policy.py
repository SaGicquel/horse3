from __future__ import annotations

import math
from typing import Any


def _round_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        return round(value, 2)
    return round(value / increment) * increment


def _risk_rank(risk: str) -> int:
    order = {"Faible": 0, "Modéré": 1, "Élevé": 2, "Très élevé": 3}
    return order.get((risk or "").strip(), 99)


def _detect_zone(bankroll: float) -> str:
    if bankroll < 50:
        return "micro"
    if bankroll < 250:
        return "small"
    return "full"


def _bucket_from_bet_risk(bet_risk: str) -> str:
    r = (bet_risk or "").strip()
    if r == "Faible":
        return "SÛR"
    if r == "Modéré":
        return "ÉQUILIBRÉ"
    return "RISQUÉ"


def select_portfolio_from_picks(
    *,
    picks: list[dict[str, Any]],
    bankroll: float,
    kelly_fraction: float,
    cap_per_bet: float,
    daily_budget_rate: float,
    rounding: float,
    policy: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    """
    Transforme un ensemble de picks (moteur proba/value) en positions (politique de mise).
    Ne modifie jamais p/value: uniquement filtrage, quotas et stake.
    """
    profile = (profile or "STANDARD").upper()
    zone = _detect_zone(bankroll)

    max_daily_share = float(policy.get("max_daily_budget_share_per_bet", 0.10) or 0.10)
    max_bets_per_race = int(policy.get("max_bets_per_race", 2) or 2)
    profiles = policy.get("profiles", {}) if isinstance(policy.get("profiles", {}), dict) else {}
    pconf = (
        profiles.get(profile, profiles.get("STANDARD", {})) if isinstance(profiles, dict) else {}
    )

    zones = (
        policy.get("bankroll_zones", {})
        if isinstance(policy.get("bankroll_zones", {}), dict)
        else {}
    )
    zconf = zones.get(zone, {}) if isinstance(zones, dict) else {}

    max_bets_per_day = int(zconf.get("max_bets_per_day") or pconf.get("max_bets_per_day") or 8)
    allowed_risks = set(pconf.get("allowed_risks") or ["Faible", "Modéré", "Élevé"])
    allowed_bet_types = zconf.get("allowed_bet_types") or None
    if allowed_bet_types is not None:
        allowed_bet_types = set(allowed_bet_types)

    max_odds_win = float(pconf.get("max_odds_win", 18) or 18)

    # Value cutoffs par zone/bucket, ajustés par profil
    value_map = (
        policy.get("value_min_pct_by_zone", {})
        if isinstance(policy.get("value_min_pct_by_zone", {}), dict)
        else {}
    )
    zone_map = value_map.get(zone, {}) if isinstance(value_map.get(zone, {}), dict) else {}
    profile_mult_map = (
        policy.get("profile_value_multiplier", {})
        if isinstance(policy.get("profile_value_multiplier", {}), dict)
        else {}
    )
    profile_mult = float(
        profile_mult_map.get(profile, profile_mult_map.get("STANDARD", 1.0)) or 1.0
    )

    stake_scale_bucket = (
        policy.get("stake_scale_by_bucket", {})
        if isinstance(policy.get("stake_scale_by_bucket", {}), dict)
        else {}
    )
    stake_scale_zone = (
        policy.get("stake_scale_by_zone", {})
        if isinstance(policy.get("stake_scale_by_zone", {}), dict)
        else {}
    )
    zone_scale = float(stake_scale_zone.get(zone, 1.0) or 1.0)

    daily_budget = bankroll * float(daily_budget_rate or 0.0)
    cap_bankroll = bankroll * float(cap_per_bet or 0.0)
    cap_budget = daily_budget * float(max_daily_share or 0.0)
    max_stake_per_bet = max(0.0, min(cap_bankroll, cap_budget if cap_budget > 0 else cap_bankroll))

    excluded: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for pick in picks:
        bet_type = pick.get("bet_type") or ""
        bet_risk = pick.get("bet_risk") or ""
        race_key = pick.get("race_key")
        cote = float(pick.get("cote") or 0) if pick.get("cote") is not None else 0.0

        if allowed_bet_types is not None and bet_type not in allowed_bet_types:
            excluded.append({**pick, "excludeReason": f"type {bet_type} exclu (zone {zone})"})
            continue
        if bet_risk not in allowed_risks:
            excluded.append(
                {**pick, "excludeReason": f"risque {bet_risk} exclu (profil {profile})"}
            )
            continue

        is_place = "PLACÉ" in bet_type
        value_pct = float(pick.get("value_place") if is_place else pick.get("value") or 0)
        kelly_pct = float(pick.get("kelly_place") if is_place else pick.get("kelly") or 0)
        bucket = _bucket_from_bet_risk(bet_risk)
        kind = "place" if is_place else "win"

        # cutoff selon zone/bucket, puis ajustement profil
        cutoff = None
        bmap = zone_map.get(bucket) if isinstance(zone_map.get(bucket), dict) else None
        if bmap and kind in bmap:
            cutoff = float(bmap.get(kind) or 0.0) * profile_mult
        else:
            # fallback: valeur permissive (1% = toute value positive)
            cutoff = (1.0 if kind == "win" else 1.0) * profile_mult

        if is_place:
            if value_pct < cutoff:
                excluded.append(
                    {**pick, "excludeReason": f"value_place {value_pct:.1f}% < seuil {cutoff:.1f}%"}
                )
                continue
        else:
            if cote and cote > max_odds_win:
                excluded.append(
                    {**pick, "excludeReason": f"cote {cote:.1f} > max {max_odds_win:.0f} (WIN)"}
                )
                continue
            if value_pct < cutoff:
                excluded.append(
                    {**pick, "excludeReason": f"value {value_pct:.1f}% < seuil {cutoff:.1f}%"}
                )
                continue

        if kelly_pct <= 0:
            excluded.append({**pick, "excludeReason": "kelly ≤ 0"})
            continue

        candidates.append(
            {
                **pick,
                "_value_pct": value_pct,
                "_kelly_pct": kelly_pct,
                "_race_key": race_key,
                "_bucket": bucket,
                "_cutoff": cutoff,
            }
        )

    candidates.sort(
        key=lambda p: (
            _risk_rank(p.get("bet_risk") or ""),
            -(p.get("_value_pct") or 0.0),
            -(p.get("_kelly_pct") or 0.0),
        )
    )

    selected: list[dict[str, Any]] = []
    race_counts: dict[str, int] = {}
    total_stake = 0.0
    total_ev = 0.0

    for cand in candidates:
        if len(selected) >= max_bets_per_day:
            excluded.append({**cand, "excludeReason": f"max {max_bets_per_day} paris/jour"})
            continue

        rk = cand.get("_race_key") or "unknown"
        race_counts[rk] = race_counts.get(rk, 0)
        if race_counts[rk] >= max_bets_per_race:
            excluded.append({**cand, "excludeReason": f"> {max_bets_per_race} paris/course"})
            continue

        kelly_raw = (cand.get("_kelly_pct") or 0.0) / 100.0
        stake_rate = min(
            max(0.0, kelly_raw * float(kelly_fraction or 0.0)), float(cap_per_bet or 0.0)
        )
        stake = bankroll * stake_rate
        bucket = cand.get("_bucket") or "RISQUÉ"
        bscale = float(stake_scale_bucket.get(bucket, 1.0) or 1.0)
        stake *= bscale * zone_scale
        stake = _round_to_increment(stake, rounding)
        stake = min(stake, max_stake_per_bet)

        if rounding > 0 and stake > 0:
            stake = max(rounding, stake)

        if daily_budget > 0 and total_stake + stake > daily_budget:
            remaining = daily_budget - total_stake
            remaining = max(0.0, remaining)
            remaining = (
                math.floor(remaining / rounding) * rounding if rounding > 0 else round(remaining, 2)
            )
            if remaining <= 0:
                excluded.append({**cand, "excludeReason": "budget jour épuisé"})
                continue
            stake = remaining

        ev_decimal = (cand.get("_value_pct") or 0.0) / 100.0
        updated = {k: v for k, v in cand.items() if not str(k).startswith("_")}
        updated["stake"] = float(stake)
        updated["stake_user"] = float(stake)
        updated["ev_decimal"] = float(ev_decimal)
        updated["ev"] = round(stake * ev_decimal, 2)
        updated["policy_zone"] = zone
        updated["profile"] = profile
        updated["policy_bucket"] = bucket
        updated["policy_value_cutoff_pct"] = float(cand.get("_cutoff") or 0.0)

        selected.append(updated)
        race_counts[rk] += 1
        total_stake += stake
        total_ev += updated["ev"]

    budget_left = daily_budget - total_stake if daily_budget > 0 else 0.0

    return {
        "positions": selected,
        "excluded": excluded,
        "total_stake": round(total_stake, 2),
        "total_ev": round(total_ev, 2),
        "budget_left": round(budget_left, 2),
        "caps": {
            "daily_budget_eur": round(daily_budget, 2),
            "max_stake_per_bet_eur": round(max_stake_per_bet, 2),
            "cap_per_bet": float(cap_per_bet),
            "daily_budget_rate": float(daily_budget_rate),
            "max_daily_budget_share_per_bet": float(max_daily_share),
        },
        "policy": {
            "zone": zone,
            "profile": profile,
            "max_bets_per_day": max_bets_per_day,
            "max_bets_per_race": max_bets_per_race,
            "max_odds_win": max_odds_win,
            "allowed_risks": sorted(list(allowed_risks), key=_risk_rank),
            "allowed_bet_types": sorted(list(allowed_bet_types))
            if allowed_bet_types is not None
            else None,
            "profile_value_multiplier": profile_mult,
            "value_min_pct_by_zone": value_map.get(zone)
            if isinstance(value_map.get(zone), dict)
            else None,
            "stake_scale_by_bucket": stake_scale_bucket,
            "stake_scale_by_zone": stake_scale_zone,
        },
    }
