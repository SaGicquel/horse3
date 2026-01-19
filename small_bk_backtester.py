#!/usr/bin/env python3
"""
Small Bankroll Strategy Backtester (100-500€)
=============================================
Teste des stratégies optimisées pour les bankrolls intermédiaires.

Cette tranche est particulière car:
- On peut se permettre plus de variance que micro
- Mais on doit encore être prudent pour ne pas retomber en micro
- C'est la phase de "croissance" vers le full BK

Auteur: Horse3 AI System
Date: 2024-12
"""

import os
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# STRATÉGIES SMALL-BANKROLL (100-500€)
# =============================================================================


@dataclass
class SmallBKStrategyConfig:
    """Configuration d'une stratégie pour small-bankroll."""

    name: str
    description: str

    kelly_fraction: float
    value_cutoff: float
    max_odds: float
    min_proba: float
    max_bets_per_day: int

    min_stake_eur: float = 3.0
    max_stake_pct: float = 0.04
    daily_budget_pct: float = 0.12
    include_ep: bool = False  # Inclure E/P (Gagnant-Placé)?


SMALL_BK_STRATEGIES = {
    # Stratégie 1: Conservateur (comme micro mais un peu plus de marge)
    "conservateur_small": SmallBKStrategyConfig(
        name="Conservateur Small",
        description="Prudent, priorité régularité",
        kelly_fraction=0.10,
        value_cutoff=0.15,
        max_odds=5.0,
        min_proba=0.20,
        max_bets_per_day=2,
        max_stake_pct=0.03,
        daily_budget_pct=0.08,
        include_ep=False,
    ),
    # Stratégie 2: Équilibré (balance croissance/sécurité)
    "equilibre_small": SmallBKStrategyConfig(
        name="Équilibré Small",
        description="Balance croissance et sécurité",
        kelly_fraction=0.15,
        value_cutoff=0.12,
        max_odds=6.0,
        min_proba=0.18,
        max_bets_per_day=3,
        max_stake_pct=0.04,
        daily_budget_pct=0.10,
        include_ep=False,
    ),
    # Stratégie 3: Croissance (pour passer rapidement en full)
    "croissance_small": SmallBKStrategyConfig(
        name="Croissance Small",
        description="Objectif: atteindre 500€ rapidement",
        kelly_fraction=0.20,
        value_cutoff=0.10,
        max_odds=8.0,
        min_proba=0.15,
        max_bets_per_day=4,
        max_stake_pct=0.05,
        daily_budget_pct=0.12,
        include_ep=True,
    ),
    # Stratégie 4: Flat 5€ (comme micro mais avec mise plus haute)
    "flat_5eur": SmallBKStrategyConfig(
        name="Flat 5€",
        description="Mise fixe 5€, simple et efficace",
        kelly_fraction=0.0,
        value_cutoff=0.12,
        max_odds=6.0,
        min_proba=0.18,
        max_bets_per_day=3,
        min_stake_eur=5.0,
        max_stake_pct=1.0,
        daily_budget_pct=0.10,
        include_ep=False,
    ),
    # Stratégie 5: Ultra-Conservateur adapté (version small du full)
    "ultra_conservateur_small": SmallBKStrategyConfig(
        name="Ultra-Conservateur Small",
        description="Version small de la stratégie Full validée",
        kelly_fraction=0.12,
        value_cutoff=0.15,
        max_odds=6.0,
        min_proba=0.18,
        max_bets_per_day=3,
        max_stake_pct=0.03,
        daily_budget_pct=0.10,
        include_ep=False,
    ),
}


# =============================================================================
# CHARGEUR DE DONNÉES
# =============================================================================


def load_predictions(path: str = "data/backtest_predictions_calibrated.csv") -> pd.DataFrame:
    """Charge les prédictions calibrées."""
    print(f"📂 Chargement des données depuis: {path}")
    df = pd.read_csv(path)
    df["date_course"] = pd.to_datetime(df["date_course"])

    required = ["p_final", "cote_sp", "is_win", "race_key", "date_course"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne requise manquante: {col}")

    df = df.dropna(subset=["p_final", "cote_sp", "is_win"])
    df = df[df["cote_sp"] > 1.0]
    df = df.sort_values(["date_course", "race_key"]).reset_index(drop=True)

    print(f"   {len(df):,} prédictions chargées")
    return df


# =============================================================================
# SIMULATEUR SMALL-BANKROLL
# =============================================================================


class SmallBKSimulator:
    """Simulateur adapté aux small-bankrolls (100-500€)."""

    def __init__(self, strategy: SmallBKStrategyConfig, initial_bankroll: float = 200.0):
        self.strategy = strategy
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.history: List[Dict] = []
        self.daily_stats = defaultdict(lambda: {"bets": 0, "stake": 0, "profit": 0})

        # Tracking
        self.reached_full = False  # True si bankroll >= 500€
        self.days_to_full = None
        self.dropped_to_micro = False  # True si bankroll < 50€
        self.drop_date = None

    def reset(self):
        self.bankroll = self.initial_bankroll
        self.history = []
        self.daily_stats = defaultdict(lambda: {"bets": 0, "stake": 0, "profit": 0})
        self.reached_full = False
        self.days_to_full = None
        self.dropped_to_micro = False
        self.drop_date = None

    def _calculate_value(self, p_model: float, odds: float) -> float:
        return p_model * odds - 1

    def _calculate_stake(self, p_model: float, odds: float) -> float:
        if "Flat" in self.strategy.name:
            return self.strategy.min_stake_eur

        if self.strategy.kelly_fraction <= 0:
            return self.strategy.min_stake_eur

        b = odds - 1
        q = 1 - p_model
        kelly_full = (b * p_model - q) / b if b > 0 else 0
        kelly_fractional = max(0, kelly_full * self.strategy.kelly_fraction)

        stake = self.bankroll * kelly_fractional
        stake = min(stake, self.bankroll * self.strategy.max_stake_pct)
        stake = max(self.strategy.min_stake_eur, stake)

        return stake

    def should_bet(self, row: pd.Series, date: str) -> bool:
        if self.dropped_to_micro:
            return False

        if self.bankroll < self.strategy.min_stake_eur:
            self.dropped_to_micro = True
            self.drop_date = date
            return False

        if self.daily_stats[date]["bets"] >= self.strategy.max_bets_per_day:
            return False

        daily_budget = self.bankroll * self.strategy.daily_budget_pct
        if self.daily_stats[date]["stake"] + self.strategy.min_stake_eur > daily_budget:
            return False

        p_model = row["p_final"]
        odds = row["cote_sp"]

        if p_model < self.strategy.min_proba:
            return False
        if odds > self.strategy.max_odds:
            return False

        value = self._calculate_value(p_model, odds)
        if value < self.strategy.value_cutoff:
            return False

        return True

    def place_bet(self, row: pd.Series, date: str, day_index: int) -> Dict:
        p_model = row["p_final"]
        odds = row["cote_sp"]
        is_win = row["is_win"]

        stake = self._calculate_stake(p_model, odds)
        stake = min(stake, self.bankroll)

        if stake < self.strategy.min_stake_eur:
            return {"skipped": True, "reason": "insufficient_funds"}

        if is_win == 1:
            profit = stake * (odds - 1)
        else:
            profit = -stake

        self.bankroll += profit

        # Check transitions
        if self.bankroll < 50:
            self.dropped_to_micro = True
            self.drop_date = date

        if not self.reached_full and self.bankroll >= 500:
            self.reached_full = True
            self.days_to_full = day_index

        bet = {
            "date": date,
            "race_key": row["race_key"],
            "p_model": p_model,
            "odds": odds,
            "stake": stake,
            "stake_pct": stake / self.initial_bankroll * 100,
            "is_win": is_win,
            "profit": profit,
            "bankroll_after": self.bankroll,
            "value": self._calculate_value(p_model, odds),
        }

        self.history.append(bet)
        self.daily_stats[date]["bets"] += 1
        self.daily_stats[date]["stake"] += stake
        self.daily_stats[date]["profit"] += profit

        return bet

    def get_metrics(self) -> Dict[str, Any]:
        if not self.history:
            return {"error": "No bets", "strategy": self.strategy.name}

        df = pd.DataFrame(self.history)

        n_bets = len(df)
        n_wins = int(df["is_win"].sum())
        win_rate = n_wins / n_bets * 100 if n_bets > 0 else 0

        total_staked = df["stake"].sum()
        total_profit = df["profit"].sum()
        roi = total_profit / total_staked * 100 if total_staked > 0 else 0

        # Drawdown
        bankroll_series = df["bankroll_after"].values
        peak = np.maximum.accumulate(np.insert(bankroll_series, 0, self.initial_bankroll))
        drawdown = (peak[1:] - bankroll_series) / peak[1:] * 100
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

        # Semaines positives
        df["date_dt"] = pd.to_datetime(df["date"])
        df["week"] = (
            df["date_dt"].dt.isocalendar().week.astype(str)
            + "-"
            + df["date_dt"].dt.isocalendar().year.astype(str)
        )
        weekly_profits = df.groupby("week")["profit"].sum()
        pct_positive_weeks = (
            (weekly_profits > 0).sum() / len(weekly_profits) * 100 if len(weekly_profits) > 0 else 0
        )

        # Sharpe
        daily_profits = df.groupby("date_dt")["profit"].sum()
        sharpe = (
            daily_profits.mean() / daily_profits.std() * np.sqrt(252)
            if len(daily_profits) > 1 and daily_profits.std() > 0
            else 0
        )

        return {
            "strategy": self.strategy.name,
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": round(self.bankroll, 2),
            "n_bets": n_bets,
            "n_wins": n_wins,
            "win_rate_pct": round(win_rate, 2),
            "total_staked": round(total_staked, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": round(roi, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 3),
            "pct_positive_weeks": round(pct_positive_weeks, 2),
            "reached_full": self.reached_full,
            "days_to_full": self.days_to_full,
            "dropped_to_micro": self.dropped_to_micro,
            "drop_date": self.drop_date,
            "avg_stake": round(df["stake"].mean(), 2),
        }


# =============================================================================
# BACKTESTER
# =============================================================================


def run_small_bk_backtest(
    df: pd.DataFrame, initial_bankroll: float = 200.0, output_dir: str = "backtest_results"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "run_timestamp": datetime.now().isoformat(),
        "initial_bankroll": initial_bankroll,
        "data_period": {
            "start": df["date_course"].min().strftime("%Y-%m-%d"),
            "end": df["date_course"].max().strftime("%Y-%m-%d"),
        },
        "strategies": {},
    }

    max_date = df["date_course"].max()
    periods = {
        "3_months": (max_date - timedelta(days=90), max_date),
        "6_months": (max_date - timedelta(days=180), max_date),
        "1_year": (max_date - timedelta(days=365), max_date),
    }

    all_results = []

    for strat_key, strat_config in SMALL_BK_STRATEGIES.items():
        print(f"\n🎯 Test stratégie: {strat_config.name}")
        results["strategies"][strat_key] = asdict(strat_config)

        for period_name, (start, end) in periods.items():
            period_df = df[(df["date_course"] >= start) & (df["date_course"] <= end)]

            if len(period_df) < 100:
                continue

            print(f"   📊 {period_name}: {len(period_df):,} prédictions")

            sim = SmallBKSimulator(strat_config, initial_bankroll)
            dates = period_df["date_course"].dt.strftime("%Y-%m-%d").unique()

            for day_idx, date in enumerate(sorted(dates)):
                if sim.dropped_to_micro:
                    break

                day_data = period_df[period_df["date_course"].dt.strftime("%Y-%m-%d") == date]

                for _, row in day_data.iterrows():
                    if sim.should_bet(row, date):
                        sim.place_bet(row, date, day_idx)

            metrics = sim.get_metrics()
            metrics["period"] = period_name
            all_results.append(metrics)

    results["all_results"] = all_results

    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(os.path.join(output_dir, "small_bk_comparison.csv"), index=False)

        for period in ["1_year", "6_months", "3_months"]:
            period_data = df_results[df_results["period"] == period].copy()
            if len(period_data) > 0:
                # Score: ROI + régularité - drawdown + bonus full reached
                period_data = period_data.copy()
                period_data["score"] = (
                    period_data["roi_pct"] * 2
                    - period_data["max_drawdown_pct"]
                    + period_data["pct_positive_weeks"]
                    + period_data["reached_full"].apply(lambda x: 50 if x else 0)
                )
                best = period_data.loc[period_data["score"].idxmax()]
                results[f"best_{period}"] = {
                    "strategy": best["strategy"],
                    "roi_pct": best["roi_pct"],
                    "final_bankroll": best["final_bankroll"],
                    "max_drawdown_pct": best["max_drawdown_pct"],
                    "pct_positive_weeks": best["pct_positive_weeks"],
                    "reached_full": best["reached_full"],
                    "days_to_full": best["days_to_full"],
                }

    output_path = os.path.join(output_dir, "small_bk_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n✅ Rapport sauvegardé: {output_path}")

    return results


def print_small_results(results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print(f"📊 RÉSULTATS SMALL-BANKROLL (BK initial: {results['initial_bankroll']}€)")
    print("=" * 80)

    for period in ["1_year", "6_months", "3_months"]:
        key = f"best_{period}"
        if key in results:
            best = results[key]
            status = "🚀 Zone Full!" if best["reached_full"] else "📈 En croissance"
            print(f"\n🏆 Meilleure stratégie ({period}):")
            print(f"   → {best['strategy']}")
            print(f"   • ROI: {best['roi_pct']:+.2f}%")
            print(f"   • Final: {best['final_bankroll']:.2f}€")
            print(f"   • Max Drawdown: {best['max_drawdown_pct']:.2f}%")
            print(f"   • Semaines +: {best['pct_positive_weeks']:.1f}%")
            print(f"   • Status: {status}")

    print("\n" + "=" * 80)


def main():
    print("\n" + "=" * 80)
    print("🏇 SMALL-BANKROLL STRATEGY BACKTESTER - Horse3")
    print("=" * 80)

    df = load_predictions()
    results = run_small_bk_backtest(df, initial_bankroll=200.0)
    print_small_results(results)

    return results


if __name__ == "__main__":
    main()
