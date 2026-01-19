#!/usr/bin/env python3
"""
Micro Bankroll Strategy Backtester
===================================
Teste des stratégies optimisées pour les petits bankrolls (< 100€).

Contraintes spécifiques aux micro-BK:
- Mise minimum 2€ = 2-4% du BK (vs 0.2% pour un BK de 1000€)
- Variance a plus d'impact (une mauvaise série peut tout perdre)
- Besoin de régularité maximale pour construire le capital

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
# STRATÉGIES MICRO-BANKROLL
# =============================================================================


@dataclass
class MicroStrategyConfig:
    """Configuration d'une stratégie pour micro-bankroll."""

    name: str
    description: str

    # Filtres ultra-stricts
    kelly_fraction: float
    value_cutoff: float
    max_odds: float
    min_proba: float
    max_bets_per_day: int

    # Mises adaptées aux petits BK
    min_stake_eur: float = 2.0  # PMU minimum
    max_stake_pct: float = 0.05  # Max 5% par pari
    daily_budget_pct: float = 0.10  # Max 10% par jour


# Stratégies adaptées aux micro-bankrolls
MICRO_STRATEGIES = {
    # Stratégie 1: Hyper-conservateur (1 pari/jour, très sûr)
    "hyper_conservateur": MicroStrategyConfig(
        name="Hyper-Conservateur",
        description="1 pari/jour max, uniquement les coups sûrs",
        kelly_fraction=0.08,
        value_cutoff=0.20,  # Value minimum 20%
        max_odds=4.0,  # Cotes max 4 (favoris)
        min_proba=0.25,  # Proba min 25%
        max_bets_per_day=1,
        max_stake_pct=0.04,
        daily_budget_pct=0.05,
    ),
    # Stratégie 2: Conservateur micro (2-3 paris/jour)
    "conservateur_micro": MicroStrategyConfig(
        name="Conservateur Micro",
        description="2-3 paris/jour, sélection stricte",
        kelly_fraction=0.10,
        value_cutoff=0.18,  # Value minimum 18%
        max_odds=5.0,
        min_proba=0.22,
        max_bets_per_day=3,
        max_stake_pct=0.04,
        daily_budget_pct=0.08,
    ),
    # Stratégie 3: Équilibré micro (similaire à full mais adapté)
    "equilibre_micro": MicroStrategyConfig(
        name="Équilibré Micro",
        description="Équilibre rendement/risque pour petits BK",
        kelly_fraction=0.12,
        value_cutoff=0.15,  # Value minimum 15%
        max_odds=6.0,
        min_proba=0.18,
        max_bets_per_day=3,
        max_stake_pct=0.05,
        daily_budget_pct=0.10,
    ),
    # Stratégie 4: Croissance accélérée (plus risqué)
    "croissance_acceleree": MicroStrategyConfig(
        name="Croissance Accélérée",
        description="Plus de risque pour croître plus vite",
        kelly_fraction=0.15,
        value_cutoff=0.12,
        max_odds=8.0,
        min_proba=0.15,
        max_bets_per_day=4,
        max_stake_pct=0.06,
        daily_budget_pct=0.12,
    ),
    # Stratégie 5: Flat betting (mise fixe)
    "flat_2eur": MicroStrategyConfig(
        name="Flat 2€",
        description="Mise fixe 2€, évite les calculs complexes",
        kelly_fraction=0.0,  # Pas de Kelly
        value_cutoff=0.15,
        max_odds=5.0,
        min_proba=0.20,
        max_bets_per_day=2,
        max_stake_pct=1.0,  # Ignoré en flat
        daily_budget_pct=0.10,
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
# SIMULATEUR MICRO-BANKROLL
# =============================================================================


class MicroBKSimulator:
    """Simulateur adapté aux micro-bankrolls."""

    def __init__(self, strategy: MicroStrategyConfig, initial_bankroll: float = 50.0):
        self.strategy = strategy
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.history: List[Dict] = []
        self.daily_stats = defaultdict(lambda: {"bets": 0, "stake": 0, "profit": 0})
        self.min_stake = 2.0  # Mise minimum PMU

        # Tracking spécial micro-BK
        self.busted = False  # True si bankroll < 2€
        self.bust_date = None
        self.days_to_double = None  # Jours pour doubler le BK
        self.doubled = False

    def reset(self):
        """Réinitialise le simulateur."""
        self.bankroll = self.initial_bankroll
        self.history = []
        self.daily_stats = defaultdict(lambda: {"bets": 0, "stake": 0, "profit": 0})
        self.busted = False
        self.bust_date = None
        self.days_to_double = None
        self.doubled = False

    def _calculate_value(self, p_model: float, odds: float) -> float:
        return p_model * odds - 1

    def _calculate_stake(self, p_model: float, odds: float) -> float:
        """Calcule la mise adaptée aux micro-BK."""
        if self.strategy.name == "Flat 2€":
            return self.min_stake

        # Kelly fractionnaire
        if self.strategy.kelly_fraction <= 0:
            return self.min_stake

        b = odds - 1
        q = 1 - p_model
        kelly_full = (b * p_model - q) / b if b > 0 else 0
        kelly_fractional = max(0, kelly_full * self.strategy.kelly_fraction)

        stake = self.bankroll * kelly_fractional
        stake = min(stake, self.bankroll * self.strategy.max_stake_pct)
        stake = max(self.min_stake, stake)

        return stake

    def should_bet(self, row: pd.Series, date: str) -> bool:
        """Vérifie si on doit parier."""
        if self.busted:
            return False

        if self.bankroll < self.min_stake:
            self.busted = True
            self.bust_date = date
            return False

        # Limites journalières
        if self.daily_stats[date]["bets"] >= self.strategy.max_bets_per_day:
            return False

        daily_budget = self.bankroll * self.strategy.daily_budget_pct
        if self.daily_stats[date]["stake"] + self.min_stake > daily_budget:
            return False

        p_model = row["p_final"]
        odds = row["cote_sp"]

        # Filtres
        if p_model < self.strategy.min_proba:
            return False
        if odds > self.strategy.max_odds:
            return False

        value = self._calculate_value(p_model, odds)
        if value < self.strategy.value_cutoff:
            return False

        return True

    def place_bet(self, row: pd.Series, date: str, day_index: int) -> Dict:
        """Place un pari."""
        p_model = row["p_final"]
        odds = row["cote_sp"]
        is_win = row["is_win"]

        stake = self._calculate_stake(p_model, odds)
        stake = min(stake, self.bankroll)

        if stake < self.min_stake:
            stake = self.min_stake

        if stake > self.bankroll:
            return {"skipped": True, "reason": "insufficient_funds"}

        # Résultat
        if is_win == 1:
            profit = stake * (odds - 1)
        else:
            profit = -stake

        self.bankroll += profit

        # Check bust
        if self.bankroll < self.min_stake:
            self.busted = True
            self.bust_date = date

        # Check double
        if not self.doubled and self.bankroll >= self.initial_bankroll * 2:
            self.doubled = True
            self.days_to_double = day_index

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
        """Calcule les métriques de performance."""
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

        # Survival (jours avant bust)
        unique_dates = df["date"].unique()
        survival_days = len(unique_dates)

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
            "busted": self.busted,
            "bust_date": self.bust_date,
            "survival_days": survival_days,
            "doubled": self.doubled,
            "days_to_double": self.days_to_double,
            "avg_stake": round(df["stake"].mean(), 2),
            "avg_stake_pct": round(df["stake_pct"].mean(), 2),
        }


# =============================================================================
# BACKTESTER
# =============================================================================


def run_micro_backtest(
    df: pd.DataFrame, initial_bankroll: float = 50.0, output_dir: str = "backtest_results"
) -> Dict[str, Any]:
    """Exécute le backtest pour toutes les stratégies micro."""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "run_timestamp": datetime.now().isoformat(),
        "initial_bankroll": initial_bankroll,
        "data_period": {
            "start": df["date_course"].min().strftime("%Y-%m-%d"),
            "end": df["date_course"].max().strftime("%Y-%m-%d"),
            "total_predictions": len(df),
        },
        "strategies": {},
    }

    # Périodes de test
    max_date = df["date_course"].max()
    periods = {
        "3_months": (max_date - timedelta(days=90), max_date),
        "6_months": (max_date - timedelta(days=180), max_date),
        "1_year": (max_date - timedelta(days=365), max_date),
    }

    all_results = []

    for strat_key, strat_config in MICRO_STRATEGIES.items():
        print(f"\n🎯 Test stratégie: {strat_config.name}")
        results["strategies"][strat_key] = asdict(strat_config)

        for period_name, (start, end) in periods.items():
            period_df = df[(df["date_course"] >= start) & (df["date_course"] <= end)]

            if len(period_df) < 100:
                continue

            print(f"   📊 {period_name}: {len(period_df):,} prédictions")

            sim = MicroBKSimulator(strat_config, initial_bankroll)
            dates = period_df["date_course"].dt.strftime("%Y-%m-%d").unique()

            for day_idx, date in enumerate(sorted(dates)):
                if sim.busted:
                    break

                day_data = period_df[period_df["date_course"].dt.strftime("%Y-%m-%d") == date]

                for _, row in day_data.iterrows():
                    if sim.should_bet(row, date):
                        sim.place_bet(row, date, day_idx)

            metrics = sim.get_metrics()
            metrics["period"] = period_name
            all_results.append(metrics)

    results["all_results"] = all_results

    # Trouver la meilleure stratégie
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(os.path.join(output_dir, "micro_bk_comparison.csv"), index=False)

        # Score combiné pour micro-BK: priorité à la survie et régularité
        for period in ["1_year", "6_months", "3_months"]:
            period_data = df_results[df_results["period"] == period].copy()
            if len(period_data) > 0:
                # Ne pas considérer les stratégies qui ont bust
                non_busted = period_data[period_data["busted"] == False]

                if len(non_busted) > 0:
                    non_busted = non_busted.copy()
                    non_busted["score"] = (
                        non_busted["roi_pct"] * 2
                        - non_busted["max_drawdown_pct"]
                        + non_busted["pct_positive_weeks"]
                        + (100 if non_busted["doubled"].any() else 0)
                    )
                    best = non_busted.loc[non_busted["score"].idxmax()]
                    results[f"best_{period}"] = {
                        "strategy": best["strategy"],
                        "roi_pct": best["roi_pct"],
                        "final_bankroll": best["final_bankroll"],
                        "max_drawdown_pct": best["max_drawdown_pct"],
                        "pct_positive_weeks": best["pct_positive_weeks"],
                        "survival_days": best["survival_days"],
                        "doubled": best["doubled"],
                    }

    # Sauvegarder
    output_path = os.path.join(output_dir, "micro_bk_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n✅ Rapport sauvegardé: {output_path}")

    return results


def print_micro_results(results: Dict[str, Any]):
    """Affiche un résumé des résultats micro-BK."""
    print("\n" + "=" * 80)
    print(f"📊 RÉSULTATS MICRO-BANKROLL (BK initial: {results['initial_bankroll']}€)")
    print("=" * 80)

    for period in ["1_year", "6_months", "3_months"]:
        key = f"best_{period}"
        if key in results:
            best = results[key]
            status = "✅ Doublé!" if best["doubled"] else "🔄 En croissance"
            print(f"\n🏆 Meilleure stratégie ({period}):")
            print(f"   → {best['strategy']}")
            print(f"   • ROI: {best['roi_pct']:+.2f}%")
            print(
                f"   • Final: {best['final_bankroll']:.2f}€ (initial: {results['initial_bankroll']}€)"
            )
            print(f"   • Max Drawdown: {best['max_drawdown_pct']:.2f}%")
            print(f"   • Semaines +: {best['pct_positive_weeks']:.1f}%")
            print(f"   • Status: {status}")

    print("\n" + "=" * 80)


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("\n" + "=" * 80)
    print("🏇 MICRO-BANKROLL STRATEGY BACKTESTER - Horse3")
    print("=" * 80)

    # Charger les données
    df = load_predictions()

    # Tester avec BK = 50€
    results = run_micro_backtest(df, initial_bankroll=50.0)

    # Afficher les résultats
    print_micro_results(results)

    return results


if __name__ == "__main__":
    main()
