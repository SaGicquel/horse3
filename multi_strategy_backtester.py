#!/usr/bin/env python3
"""
Multi-Strategy Backtester - Comparaison de Stratégies de Paris
==============================================================
Teste 4 stratégies différentes sur les données historiques:
1. Ultra-Conservateur: Minimise drawdowns, maximise régularité
2. Équilibré: Balance rendement/régularité
3. Rendement Long Terme: Maximise ROI sur 6+ mois
4. Adaptive: Ajuste selon performance récente

Auteur: Horse3 AI System
Date: 2024-12
"""

import os
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION DES STRATÉGIES
# =============================================================================


@dataclass
class StrategyConfig:
    """Configuration d'une stratégie de paris."""

    name: str
    description: str

    # Filtres de sélection
    kelly_fraction: float  # Fraction du Kelly (0.1 = 10%)
    value_cutoff: float  # Value minimum (0.15 = 15%)
    max_odds: float  # Cote maximum
    min_proba: float  # Probabilité minimum
    max_bets_per_day: int  # Nombre max de paris par jour

    # Gestion du capital
    max_stake_pct: float = 0.05  # Max % du bankroll par pari
    daily_budget_pct: float = 0.15  # Budget journalier % du bankroll

    # Types de paris autorisés (simplifié pour le backtest)
    include_place: bool = True
    include_win: bool = True


# Définition des 4 stratégies
STRATEGIES = {
    "ultra_conservateur": StrategyConfig(
        name="Ultra-Conservateur",
        description="Minimise drawdowns, maximise régularité. PLACÉ uniquement.",
        kelly_fraction=0.12,
        value_cutoff=0.15,
        max_odds=6.0,
        min_proba=0.18,
        max_bets_per_day=3,
        max_stake_pct=0.03,
        daily_budget_pct=0.10,
        include_place=True,
        include_win=False,
    ),
    "equilibre": StrategyConfig(
        name="Équilibré",
        description="Balance rendement/régularité. Mix PLACÉ/GAGNANT.",
        kelly_fraction=0.20,
        value_cutoff=0.08,
        max_odds=10.0,
        min_proba=0.12,
        max_bets_per_day=5,
        max_stake_pct=0.04,
        daily_budget_pct=0.12,
        include_place=True,
        include_win=True,
    ),
    "rendement_long_terme": StrategyConfig(
        name="Rendement Long Terme",
        description="Maximise ROI sur 6+ mois. Tous paris.",
        kelly_fraction=0.25,
        value_cutoff=0.05,
        max_odds=15.0,
        min_proba=0.08,
        max_bets_per_day=8,
        max_stake_pct=0.05,
        daily_budget_pct=0.15,
        include_place=True,
        include_win=True,
    ),
    "adaptive": StrategyConfig(
        name="Adaptive",
        description="Ajuste selon performance récente (7 jours).",
        kelly_fraction=0.20,  # Valeur de base
        value_cutoff=0.10,  # Valeur de base
        max_odds=12.0,
        min_proba=0.10,
        max_bets_per_day=5,
        max_stake_pct=0.04,
        daily_budget_pct=0.12,
        include_place=True,
        include_win=True,
    ),
}


# =============================================================================
# CHARGEUR DE DONNÉES
# =============================================================================


def load_predictions(path: str = "data/backtest_predictions_calibrated.csv") -> pd.DataFrame:
    """
    Charge les prédictions calibrées.
    CRITIQUE: Utilise uniquement p_final (probabilité), cote_sp (cote pré-off) et is_win (résultat).
    """
    print(f"📂 Chargement des données depuis: {path}")
    df = pd.read_csv(path)

    # Convertir la date
    df["date_course"] = pd.to_datetime(df["date_course"])

    # Colonnes requises
    required = ["p_final", "cote_sp", "is_win", "race_key", "date_course"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne requise manquante: {col}")

    # Nettoyer
    df = df.dropna(subset=["p_final", "cote_sp", "is_win"])
    df = df[df["cote_sp"] > 1.0]

    # Trier chronologiquement
    df = df.sort_values(["date_course", "race_key"]).reset_index(drop=True)

    print(f"   {len(df):,} prédictions chargées")
    print(f"   Période: {df['date_course'].min()} → {df['date_course'].max()}")

    return df


# =============================================================================
# SIMULATEUR DE PARIS
# =============================================================================


class BettingSimulator:
    """Simule les paris avec une stratégie donnée."""

    def __init__(self, strategy: StrategyConfig, initial_bankroll: float = 1000.0):
        self.strategy = strategy
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.history: List[Dict] = []
        self.daily_stats: Dict[str, Dict] = defaultdict(
            lambda: {"bets": 0, "stake": 0, "profit": 0}
        )

        # Pour la stratégie adaptive
        self.rolling_roi_7d: List[float] = []

    def reset(self):
        """Réinitialise le simulateur."""
        self.bankroll = self.initial_bankroll
        self.history = []
        self.daily_stats = defaultdict(lambda: {"bets": 0, "stake": 0, "profit": 0})
        self.rolling_roi_7d = []

    def _get_adaptive_params(self) -> Tuple[float, float]:
        """Retourne kelly_fraction et value_cutoff adaptés selon performance récente."""
        if len(self.rolling_roi_7d) < 7:
            return self.strategy.kelly_fraction, self.strategy.value_cutoff

        roi_7d = sum(self.rolling_roi_7d[-7:]) / 7

        if roi_7d < -0.05:  # ROI < -5% sur 7 jours
            # Mode conservateur
            return 0.12, 0.15
        elif roi_7d > 0.05:  # ROI > 5% sur 7 jours
            # Mode légèrement plus agressif
            return 0.25, 0.06
        else:
            # Mode équilibré
            return 0.20, 0.10

    def _calculate_value(self, p_model: float, odds: float) -> float:
        """Calcule la value: (p * odds - 1)."""
        return p_model * odds - 1

    def _calculate_kelly(self, p_model: float, odds: float, fraction: float) -> float:
        """Calcule la mise Kelly fractionnelle."""
        if odds <= 1 or p_model <= 0 or p_model >= 1:
            return 0.0

        b = odds - 1
        q = 1 - p_model

        kelly_full = (b * p_model - q) / b
        kelly_fractional = kelly_full * fraction

        return max(0.0, kelly_fractional)

    def should_bet(self, row: pd.Series, date: str) -> bool:
        """Détermine si on doit parier sur ce cheval."""
        p_model = row["p_final"]
        odds = row["cote_sp"]

        # Vérifier les limites journalières
        if self.daily_stats[date]["bets"] >= self.strategy.max_bets_per_day:
            return False

        # Vérifier la proba minimum
        if p_model < self.strategy.min_proba:
            return False

        # Vérifier la cote maximum
        if odds > self.strategy.max_odds:
            return False

        # Calculer la value
        value = self._calculate_value(p_model, odds)

        # Pour la stratégie adaptive, ajuster le seuil
        if self.strategy.name == "Adaptive":
            _, value_cutoff = self._get_adaptive_params()
        else:
            value_cutoff = self.strategy.value_cutoff

        if value < value_cutoff:
            return False

        return True

    def place_bet(self, row: pd.Series, date: str) -> Dict:
        """Place un pari et retourne les détails."""
        p_model = row["p_final"]
        odds = row["cote_sp"]
        is_win = row["is_win"]

        # Calculer la mise
        if self.strategy.name == "Adaptive":
            kelly_fraction, _ = self._get_adaptive_params()
        else:
            kelly_fraction = self.strategy.kelly_fraction

        kelly_stake = self._calculate_kelly(p_model, odds, kelly_fraction)

        # Appliquer les caps
        stake_pct = min(kelly_stake, self.strategy.max_stake_pct)

        # Vérifier le budget journalier
        daily_budget = self.bankroll * self.strategy.daily_budget_pct
        remaining = daily_budget - self.daily_stats[date]["stake"]

        stake = min(self.bankroll * stake_pct, remaining)
        stake = max(2.0, stake)  # Minimum 2€

        if stake > self.bankroll:
            stake = self.bankroll

        if stake < 2.0 or remaining <= 0:
            return {"skipped": True, "reason": "budget"}

        # Calculer le résultat
        if is_win == 1:
            profit = stake * (odds - 1)
        else:
            profit = -stake

        self.bankroll += profit

        # Enregistrer
        bet = {
            "date": date,
            "race_key": row["race_key"],
            "p_model": p_model,
            "odds": odds,
            "stake": stake,
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

    def update_rolling_roi(self, date: str):
        """Met à jour le ROI rolling pour la stratégie adaptive."""
        if date in self.daily_stats:
            stake = self.daily_stats[date]["stake"]
            profit = self.daily_stats[date]["profit"]
            if stake > 0:
                roi = profit / stake
            else:
                roi = 0
            self.rolling_roi_7d.append(roi)

    def get_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques de performance."""
        if not self.history:
            return {"error": "No bets"}

        df = pd.DataFrame(self.history)

        n_bets = len(df)
        n_wins = df["is_win"].sum()
        win_rate = n_wins / n_bets * 100

        total_staked = df["stake"].sum()
        total_profit = df["profit"].sum()
        roi = total_profit / total_staked * 100 if total_staked > 0 else 0

        # Calcul du drawdown
        bankroll_series = df["bankroll_after"].values
        peak = np.maximum.accumulate(bankroll_series)
        drawdown = (peak - bankroll_series) / peak * 100
        max_drawdown = drawdown.max()

        # Sharpe ratio (daily)
        df["date_dt"] = pd.to_datetime(df["date"])
        daily_profits = df.groupby("date_dt")["profit"].sum()

        if len(daily_profits) > 1 and daily_profits.std() > 0:
            sharpe = daily_profits.mean() / daily_profits.std() * np.sqrt(252)
        else:
            sharpe = 0

        # % de semaines positives
        df["week"] = (
            df["date_dt"].dt.isocalendar().week.astype(str)
            + "-"
            + df["date_dt"].dt.isocalendar().year.astype(str)
        )
        weekly_profits = df.groupby("week")["profit"].sum()
        pct_positive_weeks = (
            (weekly_profits > 0).sum() / len(weekly_profits) * 100 if len(weekly_profits) > 0 else 0
        )

        # Temps de récupération moyen
        recovery_times = []
        in_drawdown = False
        dd_start = 0

        for i, dd in enumerate(drawdown):
            if dd > 1 and not in_drawdown:  # Seuil 1%
                in_drawdown = True
                dd_start = i
            elif dd < 0.5 and in_drawdown:
                recovery_times.append(i - dd_start)
                in_drawdown = False

        avg_recovery = np.mean(recovery_times) if recovery_times else 0

        return {
            "strategy": self.strategy.name,
            "n_bets": int(n_bets),
            "n_wins": int(n_wins),
            "win_rate_pct": round(win_rate, 2),
            "total_staked": round(total_staked, 2),
            "total_profit": round(total_profit, 2),
            "roi_pct": round(roi, 2),
            "final_bankroll": round(self.bankroll, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 3),
            "pct_positive_weeks": round(pct_positive_weeks, 2),
            "avg_recovery_bets": round(avg_recovery, 1),
            "avg_stake": round(df["stake"].mean(), 2),
            "avg_odds": round(df["odds"].mean(), 2),
            "avg_value": round(df["value"].mean() * 100, 2),
        }


# =============================================================================
# BACKTESTER MULTI-STRATÉGIES
# =============================================================================


def run_backtest_for_period(
    df: pd.DataFrame, strategy_config: StrategyConfig, period_name: str = "Full"
) -> Dict[str, Any]:
    """Exécute un backtest pour une stratégie sur une période."""
    sim = BettingSimulator(strategy_config)

    dates = df["date_course"].dt.strftime("%Y-%m-%d").unique()

    for date in sorted(dates):
        day_data = df[df["date_course"].dt.strftime("%Y-%m-%d") == date]

        for _, row in day_data.iterrows():
            if sim.should_bet(row, date):
                sim.place_bet(row, date)

        sim.update_rolling_roi(date)

    metrics = sim.get_metrics()
    metrics["period"] = period_name

    return metrics


def run_multi_strategy_backtest(
    df: pd.DataFrame, output_dir: str = "backtest_results"
) -> Dict[str, Any]:
    """Exécute les backtests pour toutes les stratégies."""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "run_timestamp": datetime.now().isoformat(),
        "data_period": {
            "start": df["date_course"].min().strftime("%Y-%m-%d"),
            "end": df["date_course"].max().strftime("%Y-%m-%d"),
            "total_predictions": len(df),
        },
        "strategies": {},
        "periods": {},
    }

    # Définir les périodes de test
    max_date = df["date_course"].max()
    periods = {
        "1_month": (max_date - timedelta(days=30), max_date),
        "3_months": (max_date - timedelta(days=90), max_date),
        "6_months": (max_date - timedelta(days=180), max_date),
        "1_year": (max_date - timedelta(days=365), max_date),
        "2_years": (max_date - timedelta(days=730), max_date),
        "full": (df["date_course"].min(), max_date),
    }

    # Tester chaque stratégie sur chaque période
    all_results = []

    for strat_key, strat_config in STRATEGIES.items():
        print(f"\n🎯 Test stratégie: {strat_config.name}")
        results["strategies"][strat_key] = asdict(strat_config)

        strat_results = {}

        for period_name, (start, end) in periods.items():
            period_df = df[(df["date_course"] >= start) & (df["date_course"] <= end)]

            if len(period_df) < 100:
                print(f"   ⚠️ {period_name}: Pas assez de données ({len(period_df)})")
                continue

            print(f"   📊 {period_name}: {len(period_df):,} prédictions")

            metrics = run_backtest_for_period(period_df, strat_config, period_name)
            strat_results[period_name] = metrics

            all_results.append({"strategy": strat_key, **metrics})

        results["periods"][strat_key] = strat_results

    # Créer un tableau comparatif
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        comparison_df.to_csv(os.path.join(output_dir, "strategy_comparison.csv"), index=False)

        # Trouver la meilleure stratégie pour chaque métrique
        best = {}
        for metric in ["roi_pct", "sharpe_ratio", "max_drawdown_pct", "pct_positive_weeks"]:
            if metric == "max_drawdown_pct":
                # Pour drawdown, on veut le minimum
                best[metric] = comparison_df.loc[
                    comparison_df[metric].idxmin(), ["strategy", "period", metric]
                ].to_dict()
            else:
                best[metric] = comparison_df.loc[
                    comparison_df[metric].idxmax(), ["strategy", "period", metric]
                ].to_dict()

        results["best_by_metric"] = best

        # Score combiné: équilibre entre ROI et régularité
        # Score = ROI - 2*MaxDD + Sharpe*10 + %SemainesPos
        for period in ["1_year", "6_months", "3_months"]:
            period_data = comparison_df[comparison_df["period"] == period].copy()
            if len(period_data) > 0:
                period_data["combined_score"] = (
                    period_data["roi_pct"]
                    - 2 * period_data["max_drawdown_pct"]
                    + period_data["sharpe_ratio"] * 10
                    + period_data["pct_positive_weeks"] * 0.5
                )
                best_combined = period_data.loc[period_data["combined_score"].idxmax()]
                results[f"best_combined_{period}"] = {
                    "strategy": best_combined["strategy"],
                    "combined_score": round(best_combined["combined_score"], 2),
                    "roi_pct": best_combined["roi_pct"],
                    "max_drawdown_pct": best_combined["max_drawdown_pct"],
                    "pct_positive_weeks": best_combined["pct_positive_weeks"],
                }

    # Sauvegarder les résultats
    output_path = os.path.join(output_dir, "multi_strategy_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Rapport sauvegardé: {output_path}")

    return results


# =============================================================================
# AFFICHAGE DES RÉSULTATS
# =============================================================================


def print_results_summary(results: Dict[str, Any]):
    """Affiche un résumé des résultats."""
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ DES RÉSULTATS")
    print("=" * 80)

    for period in ["1_year", "6_months", "3_months"]:
        key = f"best_combined_{period}"
        if key in results:
            best = results[key]
            print(f"\n🏆 Meilleure stratégie ({period}):")
            print(f"   → {best['strategy'].upper()}")
            print(f"   • ROI: {best['roi_pct']:+.2f}%")
            print(f"   • Max Drawdown: {best['max_drawdown_pct']:.2f}%")
            print(f"   • % Semaines +: {best['pct_positive_weeks']:.1f}%")
            print(f"   • Score combiné: {best['combined_score']:.2f}")

    print("\n" + "-" * 80)
    print("📈 Meilleures performances par métrique:")

    if "best_by_metric" in results:
        for metric, data in results["best_by_metric"].items():
            print(f"   • {metric}: {data['strategy']} ({data['period']}) = {data[metric]}")

    print("=" * 80 + "\n")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy Backtester")
    parser.add_argument(
        "--data",
        type=str,
        default="data/backtest_predictions_calibrated.csv",
        help="Chemin vers les prédictions calibrées",
    )
    parser.add_argument("--output", type=str, default="backtest_results", help="Dossier de sortie")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🏇 MULTI-STRATEGY BACKTESTER - Horse3")
    print("=" * 80)

    # Charger les données
    df = load_predictions(args.data)

    # Exécuter les backtests
    results = run_multi_strategy_backtest(df, args.output)

    # Afficher le résumé
    print_results_summary(results)

    return results


if __name__ == "__main__":
    main()
