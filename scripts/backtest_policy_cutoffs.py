#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest rapide: cutoffs value par risque + zone BK
===================================================

Objectif: mesurer l'impact des cutoffs (SÛR/ÉQUILIBRÉ/RISQUÉ) sur:
- ROI global et par bucket
- nombre de paris / jour
- max drawdown

Hypothèses:
- Dataset: data/backtest_predictions_calibrated.csv (p_final + cote_sp + is_win)
- Backtest "WIN" uniquement (pas de rapport_place dans ce fichier)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


def classify_bucket(value_pct: float, odds: float) -> str:
    # Même esprit que web/backend/main.py:_classify_risk_profile
    if value_pct < 5 or odds <= 3:
        return "SÛR"
    if value_pct < 15 or odds < 6:
        return "ÉQUILIBRÉ"
    return "RISQUÉ"


def detect_zone(bankroll: float) -> str:
    if bankroll < 50:
        return "micro"
    if bankroll < 250:
        return "small"
    return "full"


def kelly_full(p: float, odds: float) -> float:
    if odds <= 1 or p <= 0:
        return 0.0
    num = p * (odds - 1) - (1 - p)
    den = odds - 1
    return max(0.0, num / den) if den > 0 else 0.0


def round_inc(x: float, inc: float) -> float:
    if inc <= 0:
        return round(x, 2)
    return round(x / inc) * inc


@dataclass
class Policy:
    daily_budget_rate: float
    cap_per_bet: float
    rounding: float
    max_daily_share_per_bet: float
    max_bets_per_day: int
    max_bets_per_race: int
    max_odds_win: float
    kelly_fraction: float
    value_min_pct_by_zone: Dict
    profile_value_multiplier: float
    stake_scale_by_bucket: Dict[str, float]
    stake_scale_by_zone: Dict[str, float]


def load_policy(config_path: Path, profile: str, bankroll: float) -> Policy:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    betting_defaults = cfg.get("betting_defaults", {}) or {}
    policy = cfg.get("betting_policy", {}) or {}

    kmap = betting_defaults.get("kelly_fraction_map", {}) or {
        "SUR": 0.25,
        "STANDARD": 0.33,
        "AMBITIEUX": 0.5,
    }
    daily_budget_rate = float(betting_defaults.get("daily_budget_rate", 0.12) or 0.12)
    cap_per_bet = float(betting_defaults.get("cap_per_bet", 0.02) or 0.02)
    rounding = float(betting_defaults.get("rounding_increment_eur", 0.5) or 0.5)

    max_daily_share = float(policy.get("max_daily_budget_share_per_bet", 0.10) or 0.10)
    max_bets_per_race = int(policy.get("max_bets_per_race", 2) or 2)

    profiles = policy.get("profiles", {}) or {}
    pconf = profiles.get(profile, profiles.get("STANDARD", {})) or {}
    max_bets_per_day = int(pconf.get("max_bets_per_day", 8) or 8)
    max_odds_win = float(pconf.get("max_odds_win", 18) or 18)

    mult_map = policy.get("profile_value_multiplier", {}) or {}
    profile_mult = float(mult_map.get(profile, mult_map.get("STANDARD", 1.0)) or 1.0)

    stake_scale_by_bucket = policy.get("stake_scale_by_bucket", {}) or {
        "SÛR": 1.0,
        "ÉQUILIBRÉ": 0.7,
        "RISQUÉ": 0.25,
    }
    stake_scale_by_zone = policy.get("stake_scale_by_zone", {}) or {
        "micro": 0.5,
        "small": 0.8,
        "full": 1.0,
    }

    return Policy(
        daily_budget_rate=daily_budget_rate,
        cap_per_bet=cap_per_bet,
        rounding=rounding,
        max_daily_share_per_bet=max_daily_share,
        max_bets_per_day=max_bets_per_day,
        max_bets_per_race=max_bets_per_race,
        max_odds_win=max_odds_win,
        kelly_fraction=float(kmap.get(profile, kmap.get("STANDARD", 0.33)) or 0.33),
        value_min_pct_by_zone=policy.get("value_min_pct_by_zone", {}) or {},
        profile_value_multiplier=profile_mult,
        stake_scale_by_bucket=stake_scale_by_bucket,
        stake_scale_by_zone=stake_scale_by_zone,
    )


def cutoff_for(policy: Policy, zone: str, bucket: str) -> float:
    zmap = (
        policy.value_min_pct_by_zone.get(zone, {})
        if isinstance(policy.value_min_pct_by_zone, dict)
        else {}
    )
    bmap = zmap.get(bucket, {}) if isinstance(zmap.get(bucket, {}), dict) else {}
    base = float(bmap.get("win", 8) or 8)
    return base * policy.profile_value_multiplier


def run_backtest(df: pd.DataFrame, bankroll: float, profile: str, policy: Policy) -> Dict:
    zone = detect_zone(bankroll)
    zone_scale = float(policy.stake_scale_by_zone.get(zone, 1.0) or 1.0)

    daily_budget = bankroll * policy.daily_budget_rate
    cap_bankroll = bankroll * policy.cap_per_bet
    cap_budget = daily_budget * policy.max_daily_share_per_bet
    max_stake_per_bet = min(cap_bankroll, cap_budget)

    equity = bankroll
    peak = bankroll
    max_dd = 0.0

    results_by_bucket = {
        b: {"stake": 0.0, "profit": 0.0, "bets": 0} for b in ["SÛR", "ÉQUILIBRÉ", "RISQUÉ"]
    }
    bets_per_day = []

    for day, day_df in df.groupby("date_course", sort=True):
        day_budget_left = daily_budget
        day_bets = 0
        race_counts: Dict[str, int] = {}

        # Pré-calc value + bucket
        tmp = day_df.copy()
        tmp["value_pct"] = (tmp["p_final"] * tmp["cote_sp"] - 1.0) * 100.0
        tmp["bucket"] = [classify_bucket(v, o) for v, o in zip(tmp["value_pct"], tmp["cote_sp"])]
        tmp = tmp[(tmp["cote_sp"] > 1.0) & (tmp["cote_sp"] <= policy.max_odds_win)]

        # Filtrer cutoffs
        def ok_row(r):
            return r["value_pct"] >= cutoff_for(policy, zone, r["bucket"])

        tmp = tmp[tmp.apply(ok_row, axis=1)]

        # Trier: bucket (SÛR first), puis value desc
        bucket_rank = {"SÛR": 0, "ÉQUILIBRÉ": 1, "RISQUÉ": 2}
        tmp = tmp.sort_values(
            by=["bucket", "value_pct"],
            ascending=[True, False],
            key=lambda s: s.map(bucket_rank) if s.name == "bucket" else s,
        )

        for _, row in tmp.iterrows():
            if day_bets >= policy.max_bets_per_day:
                break

            race_key = row["race_key"]
            race_counts[race_key] = race_counts.get(race_key, 0)
            if race_counts[race_key] >= policy.max_bets_per_race:
                continue

            p = float(row["p_final"])
            odds = float(row["cote_sp"])
            if odds <= 1:
                continue

            k = kelly_full(p, odds)
            stake_rate = min(k * policy.kelly_fraction, policy.cap_per_bet)
            stake = bankroll * stake_rate
            stake *= float(policy.stake_scale_by_bucket.get(row["bucket"], 1.0) or 1.0) * zone_scale
            stake = round_inc(stake, policy.rounding)
            stake = min(stake, max_stake_per_bet)
            if policy.rounding > 0 and stake > 0:
                stake = max(policy.rounding, stake)

            if stake <= 0:
                continue
            if stake > day_budget_left:
                stake = round_inc(day_budget_left, policy.rounding)
                if stake <= 0:
                    break

            win = int(row["is_win"]) == 1
            profit = stake * (odds - 1.0) if win else -stake

            equity += profit
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            b = row["bucket"]
            results_by_bucket[b]["stake"] += stake
            results_by_bucket[b]["profit"] += profit
            results_by_bucket[b]["bets"] += 1

            day_budget_left -= stake
            day_bets += 1
            race_counts[race_key] += 1

        bets_per_day.append(day_bets)

    total_stake = sum(v["stake"] for v in results_by_bucket.values())
    total_profit = sum(v["profit"] for v in results_by_bucket.values())

    out = {
        "profile": profile,
        "bankroll": bankroll,
        "zone": zone,
        "total_bets": int(sum(v["bets"] for v in results_by_bucket.values())),
        "avg_bets_per_day": round(sum(bets_per_day) / len(bets_per_day), 2)
        if bets_per_day
        else 0.0,
        "total_stake": round(total_stake, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round((total_profit / total_stake) * 100.0, 2) if total_stake > 0 else 0.0,
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "by_bucket": {},
    }
    for b, v in results_by_bucket.items():
        stake = v["stake"]
        profit = v["profit"]
        out["by_bucket"][b] = {
            "bets": int(v["bets"]),
            "roi_pct": round((profit / stake) * 100.0, 2) if stake > 0 else 0.0,
            "stake": round(stake, 2),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/backtest_predictions_calibrated.csv")
    ap.add_argument("--config", default="config/pro_betting.yaml")
    ap.add_argument("--profile", default="STANDARD", choices=["SUR", "STANDARD", "AMBITIEUX"])
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--limit_days", type=int, default=365)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.split != "all":
        df = df[df["split"] == args.split]

    df["date_course"] = pd.to_datetime(df["date_course"])
    max_day = df["date_course"].max()
    if pd.notna(max_day) and args.limit_days:
        min_day = max_day - pd.Timedelta(days=int(args.limit_days))
        df = df[df["date_course"] >= min_day]

    # stabiliser le tri
    df = df.sort_values(["date_course", "race_key"])

    cfg_path = Path(args.config)
    policy = load_policy(cfg_path, args.profile, args.bankroll)
    out = run_backtest(df, args.bankroll, args.profile, policy)

    print(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
