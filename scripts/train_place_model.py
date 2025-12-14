#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entraîne un calibrateur simple pour p(place)
============================================

But: améliorer p_place sans complexifier le moteur.

Approche:
- Construire p_win_course à partir des cotes SP (1/cote, normalisé par course).
- Calculer p_place_harville(top3) par course (Harville).
- Apprendre une régression logistique sur:
    logit(p_place_harville), log(field_size), rank_pct, discipline (trot/obstacle)
- Cible: performances.place (bool) (= placé dans les 3).

Sortie:
- config/place_model.json (coefficients + métriques + features)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss


def _safe_prob(x: float) -> float:
    return float(max(1e-12, min(1 - 1e-12, x)))


def logit(p: float) -> float:
    p = _safe_prob(p)
    return math.log(p / (1 - p))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def harville_place_top3(pwins: List[float]) -> List[float]:
    p = [_safe_prob(x) for x in pwins]
    s = sum(p) or 1.0
    p = [x / s for x in p]
    n = len(p)
    out = [0.0] * n
    for i in range(n):
        p1 = p[i]
        p2 = 0.0
        for j in range(n):
            if j == i:
                continue
            denom1 = 1.0 - p[j]
            if denom1 <= 1e-12:
                continue
            p2 += p[j] * (p[i] / denom1)
        p3 = 0.0
        for j in range(n):
            if j == i:
                continue
            denom1 = 1.0 - p[j]
            if denom1 <= 1e-12:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                denom2 = 1.0 - p[j] - p[k]
                if denom2 <= 1e-12:
                    continue
                p3 += p[j] * (p[k] / denom1) * (p[i] / denom2)
        out[i] = float(max(0.0, min(0.999, p1 + p2 + p3)))
    return out


def normalize_discipline(d: str) -> str:
    d = (d or "").lower().strip()
    if "trot" in d or "attel" in d or "mont" in d:
        return "trot"
    if "obst" in d or "haie" in d or "steeple" in d or "cross" in d:
        return "obstacle"
    if "plat" in d or "galop" in d:
        return "plat"
    return "plat"


def fetch_dataset(db_url: str) -> pd.DataFrame:
    con = psycopg2.connect(db_url)
    cur = con.cursor()
    cur.execute(
        """
        SELECT
            p.id_course,
            c.date_course,
            c.discipline,
            c.nombre_partants,
            p.cote_sp,
            p.place,
            p.non_partant,
            p.disqualifie
        FROM performances p
        JOIN courses c ON c.id_course = p.id_course
        WHERE p.cote_sp IS NOT NULL
          AND p.cote_sp > 1
          AND p.place IS NOT NULL
          AND (p.non_partant IS NULL OR p.non_partant = false)
          AND (p.disqualifie IS NULL OR p.disqualifie = false)
          AND c.nombre_partants IS NOT NULL
          AND c.nombre_partants >= 4
        """
    )
    rows = cur.fetchall()
    cur.close()
    con.close()

    df = pd.DataFrame(
        rows,
        columns=[
            "id_course",
            "date_course",
            "discipline",
            "field_size",
            "odds",
            "is_place",
            "non_partant",
            "disqualifie",
        ],
    )
    df["date_course"] = pd.to_datetime(df["date_course"])
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce").astype(float)
    df["field_size"] = pd.to_numeric(df["field_size"], errors="coerce").astype(int)
    df["is_place"] = df["is_place"].astype(int)
    df["discipline_norm"] = df["discipline"].map(normalize_discipline)
    return df.dropna(subset=["odds", "field_size", "date_course"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for course_id, g in df.groupby("id_course", sort=False):
        g = g.copy()
        # Probabilités marché normalisées (1/cote)
        inv = 1.0 / g["odds"].clip(lower=1.01)
        s = float(inv.sum()) or 1.0
        g["p_win_mkt"] = (inv / s).astype(float)

        pwins = g["p_win_mkt"].tolist()
        g["p_place_harville"] = harville_place_top3(pwins)

        g = g.sort_values("odds", ascending=True)
        g["rank_odds"] = np.arange(1, len(g) + 1)
        denom = max(1, len(g) - 1)
        g["rank_pct"] = ((g["rank_odds"] - 1) / denom).astype(float)
        g["log_field"] = np.log(g["field_size"].clip(lower=4).astype(float))
        g["logit_p_place_harville"] = g["p_place_harville"].map(lambda p: logit(float(p)))

        g["is_trot"] = (g["discipline_norm"] == "trot").astype(int)
        g["is_obstacle"] = (g["discipline_norm"] == "obstacle").astype(int)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True) if parts else df
    return out


def time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date_course")
    cut = int(len(df) * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="postgresql://pmu_user:pmu_secure_password_2025@localhost:54624/pmu_database")
    ap.add_argument("--out", default="config/place_model.json")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    df = fetch_dataset(args.db)
    if args.max_rows and args.max_rows > 0:
        df = df.sort_values("date_course").tail(args.max_rows)

    df = build_features(df)
    feat_cols = ["logit_p_place_harville", "log_field", "rank_pct", "is_trot", "is_obstacle"]
    df = df.dropna(subset=feat_cols + ["is_place"])

    train, test = time_split(df, args.test_ratio)
    X_train = train[feat_cols].values
    y_train = train["is_place"].values
    X_test = test[feat_cols].values
    y_test = test["is_place"].values

    model = LogisticRegression(solver="lbfgs", max_iter=2000, n_jobs=1)
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train": {
            "brier": float(brier_score_loss(y_train, p_train)),
            "logloss": float(log_loss(y_train, p_train)),
            "auc": float(roc_auc_score(y_train, p_train)) if len(set(y_train)) > 1 else None,
            "n": int(len(y_train)),
        },
        "test": {
            "brier": float(brier_score_loss(y_test, p_test)),
            "logloss": float(log_loss(y_test, p_test)),
            "auc": float(roc_auc_score(y_test, p_test)) if len(set(y_test)) > 1 else None,
            "n": int(len(y_test)),
        },
    }

    payload = {
        "model": "logistic_regression",
        "target": "is_place_top3",
        "features": feat_cols,
        "intercept": float(model.intercept_[0]),
        "coef": {feat_cols[i]: float(model.coef_[0][i]) for i in range(len(feat_cols))},
        "metrics": metrics,
        "notes": "Calibre p_place à partir de p_place Harville (basé sur cotes), + taille champ + rang + discipline.",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with metrics: test brier={metrics['test']['brier']:.4f}, auc={metrics['test']['auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

