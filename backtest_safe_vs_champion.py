#!/usr/bin/env python3
"""
BACKTEST COMPARATIF: MODÈLE SAFE VS CHAMPION
=============================================

Compare les performances des deux stratégies sur données historiques.

Usage:
    python backtest_safe_vs_champion.py [--output reports/comparison.md]
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from strategy_backtester import BacktestConfig, BetSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATIONS
# ============================================================================

# Configuration CHAMPION (agressive)
CONFIG_CHAMPION = BacktestConfig(
    temperature=1.254,
    kelly_fraction=0.33,
    value_cutoff=0.0,
    max_stake_pct=0.05,
    initial_bankroll=1000.0,
    output_dir="backtest_results/champion",
)

# Configuration SAFE (conservative)
CONFIG_SAFE = BacktestConfig(
    temperature=1.254,
    kelly_fraction=0.15,  # Réduit
    value_cutoff=0.10,  # Plus strict
    max_stake_pct=0.03,  # Réduit
    initial_bankroll=1000.0,
    output_dir="backtest_results/safe",
)

# Filtres pour stratégie SAFE
SAFE_FILTERS = {
    "max_odds": 6.0,  # Exclure chevaux > cote 6
    "min_proba": 0.15,  # Proba modèle > 15%
    "bet_types": ["PLACE", "EACH_WAY"],  # Pas de GAGNANT seul
    "max_bets_per_day": 3,
}


# ============================================================================
# FILTRAGE SAFE
# ============================================================================


def apply_safe_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les filtres de la stratégie SAFE"""

    initial_count = len(df)

    # Filtre par cote max
    if "odds" in df.columns:
        df = df[df["odds"] <= SAFE_FILTERS["max_odds"]]
    elif "cote" in df.columns:
        df = df[df["cote"] <= SAFE_FILTERS["max_odds"]]

    # Filtre par proba min
    if "prob_calibrated" in df.columns:
        df = df[df["prob_calibrated"] >= SAFE_FILTERS["min_proba"]]
    elif "proba" in df.columns:
        df = df[df["proba"] >= SAFE_FILTERS["min_proba"]]

    # Filtre par value min
    if "value" in df.columns:
        df = df[df["value"] >= CONFIG_SAFE.value_cutoff]

    final_count = len(df)

    logger.info(f"   Filtre SAFE: {initial_count:,} → {final_count:,} opportunités")

    return df


# ============================================================================
# BACKTESTER
# ============================================================================


def run_backtest(df: pd.DataFrame, config: BacktestConfig, strategy_name: str) -> dict:
    """Exécute un backtest complet"""

    logger.info(f"\n{'='*60}")
    logger.info(f"📊 BACKTEST: {strategy_name.upper()}")
    logger.info(f"{'='*60}")

    simulator = BetSimulator(config)

    # Statistiques
    n_bets = 0
    daily_bet_count = {}

    for _, row in df.iterrows():
        # Limite par jour (pour SAFE)
        date = str(row.get("date", row.get("race_date", "")))[:10]
        if date not in daily_bet_count:
            daily_bet_count[date] = 0

        if strategy_name == "SAFE" and daily_bet_count[date] >= SAFE_FILTERS["max_bets_per_day"]:
            continue

        # Probabilité calibrée
        prob = row.get("prob_calibrated", row.get("proba", 0.1))

        # Cote
        odds = row.get("odds", row.get("cote", 5.0))

        # Résultat (1 si gagné/placé, 0 sinon)
        result = int(row.get("is_winner", row.get("victoire", 0)))

        # Placer le pari
        bet_result = simulator.place_bet(
            date=date,
            race_id=str(row.get("race_id", row.get("id_course", ""))),
            horse_id=str(row.get("horse_id", row.get("nom_norm", ""))),
            prob_model=prob,
            odds=odds,
            actual_result=result,
            use_kelly=True,
        )

        if bet_result.get("placed", False):
            n_bets += 1
            daily_bet_count[date] = daily_bet_count.get(date, 0) + 1

    # Métriques finales
    metrics = simulator.get_metrics()
    metrics["strategy"] = strategy_name
    metrics["n_bets"] = n_bets
    metrics["avg_bets_per_day"] = n_bets / max(len(daily_bet_count), 1)

    # Affichage
    logger.info(f"   💰 Bankroll finale: {metrics.get('final_bankroll', 1000):.2f}€")
    logger.info(f"   📈 ROI: {metrics.get('roi', 0)*100:.1f}%")
    logger.info(f"   🎯 Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
    logger.info(f"   📉 Max Drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%")
    logger.info(f"   📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"   🎲 Nombre de paris: {n_bets}")

    return metrics


# ============================================================================
# COMPARAISON
# ============================================================================


def compare_strategies(metrics_champion: dict, metrics_safe: dict) -> str:
    """Génère un rapport de comparaison en Markdown"""

    report = []
    report.append("# 📊 Comparaison Modèle Safe vs Champion\n")
    report.append(f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    report.append("\n## Résumé des Métriques\n")
    report.append("| Métrique | Champion | Safe | Différence |")
    report.append("|----------|----------|------|------------|")

    metrics_to_compare = [
        ("final_bankroll", "Bankroll Finale (€)", "{:.2f}"),
        ("roi", "ROI (%)", "{:.1%}"),
        ("win_rate", "Win Rate (%)", "{:.1%}"),
        ("max_drawdown", "Max Drawdown (%)", "{:.1%}"),
        ("sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
        ("n_bets", "Nombre de Paris", "{:.0f}"),
        ("avg_bets_per_day", "Paris/Jour", "{:.1f}"),
    ]

    for key, label, fmt in metrics_to_compare:
        champ_val = metrics_champion.get(key, 0)
        safe_val = metrics_safe.get(key, 0)

        if isinstance(champ_val, (int, float)) and isinstance(safe_val, (int, float)):
            diff = safe_val - champ_val
            if "rate" in key or "roi" in key or "drawdown" in key:
                diff_str = f"{diff*100:+.1f}pp"
            else:
                diff_str = f"{diff:+.2f}"
        else:
            diff_str = "N/A"

        champ_str = fmt.format(champ_val) if isinstance(champ_val, (int, float)) else str(champ_val)
        safe_str = fmt.format(safe_val) if isinstance(safe_val, (int, float)) else str(safe_val)

        report.append(f"| {label} | {champ_str} | {safe_str} | {diff_str} |")

    # Recommandation
    report.append("\n## 🎯 Recommandation\n")

    safe_roi = metrics_safe.get("roi", 0)
    champ_roi = metrics_champion.get("roi", 0)
    safe_dd = metrics_safe.get("max_drawdown", 1)
    champ_dd = metrics_champion.get("max_drawdown", 1)
    safe_sharpe = metrics_safe.get("sharpe_ratio", 0)

    if safe_sharpe > 0.8 and abs(safe_dd) < abs(champ_dd) * 0.7:
        report.append("> ✅ **Le modèle SAFE est recommandé** : Meilleur ratio rendement/risque")
        report.append(
            f"> - Drawdown réduit de {(1 - abs(safe_dd)/max(abs(champ_dd), 0.01))*100:.0f}%"
        )
        report.append(f"> - Sharpe ratio: {safe_sharpe:.2f}")
    elif safe_roi > 0 and safe_roi > champ_roi:
        report.append("> ✅ **Le modèle SAFE surperforme** : ROI supérieur avec moins de risque")
    elif safe_roi > 0:
        report.append("> ⚠️ **Le modèle SAFE est viable** : ROI positif mais légèrement inférieur")
    else:
        report.append("> ❌ **Besoin d'ajustement** : Le modèle SAFE nécessite des optimisations")

    report.append("\n## Configuration SAFE Utilisée\n")
    report.append("```yaml")
    report.append(f"kelly_fraction: {CONFIG_SAFE.kelly_fraction}")
    report.append(f"value_cutoff: {CONFIG_SAFE.value_cutoff}")
    report.append(f"max_odds: {SAFE_FILTERS['max_odds']}")
    report.append(f"min_proba: {SAFE_FILTERS['min_proba']}")
    report.append(f"max_bets_per_day: {SAFE_FILTERS['max_bets_per_day']}")
    report.append("```\n")

    return "\n".join(report)


# ============================================================================
# MAIN
# ============================================================================


def main(output_path: str = None):
    """Pipeline principal de comparaison"""

    logger.info("🔍 BACKTEST COMPARATIF: SAFE VS CHAMPION")
    logger.info("=" * 70)

    try:
        # 1. Charger les prédictions calibrées
        logger.info("📂 Chargement des prédictions...")
        predictions_path = Path("data/backtest_predictions_calibrated.csv")

        if not predictions_path.exists():
            logger.warning(f"⚠️  Fichier non trouvé: {predictions_path}")
            logger.info("   Tentative avec backtest_predictions.csv...")
            predictions_path = Path("data/backtest_predictions.csv")

        if not predictions_path.exists():
            raise FileNotFoundError(
                "Pas de fichier de prédictions. Générez-le avec generate_predictions_D5.py"
            )

        df = pd.read_csv(predictions_path)
        logger.info(f"✅ Chargé {len(df):,} prédictions")

        # 2. Backtest CHAMPION (toutes les données)
        logger.info("\n📊 Backtest CHAMPION (stratégie agressive)...")
        metrics_champion = run_backtest(df.copy(), CONFIG_CHAMPION, "CHAMPION")

        # 3. Backtest SAFE (données filtrées)
        logger.info("\n📊 Backtest SAFE (stratégie conservative)...")
        df_safe = apply_safe_strategy(df.copy())
        metrics_safe = run_backtest(df_safe, CONFIG_SAFE, "SAFE")

        # 4. Générer rapport
        logger.info("\n📝 Génération du rapport...")
        report = compare_strategies(metrics_champion, metrics_safe)

        # 5. Sauvegarder
        if output_path is None:
            output_path = "reports/safe_vs_champion_comparison.md"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            f.write(report)

        logger.info(f"✅ Rapport sauvegardé: {output_file}")

        # Afficher le rapport
        print("\n" + "=" * 70)
        print(report)
        print("=" * 70)

        return metrics_champion, metrics_safe

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare les stratégies Safe vs Champion")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reports/safe_vs_champion_comparison.md",
        help="Chemin du rapport de sortie",
    )
    args = parser.parse_args()

    main(output_path=args.output)
