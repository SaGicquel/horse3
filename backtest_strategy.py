import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configuration
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PREDICTIONS_PATH = "data/backtest_predictions.csv"


def run_backtest(df, strategy_name, strategy_func, **kwargs):
    """Exécute une stratégie sur l'historique."""
    logger.info(f"🎰 Test Stratégie: {strategy_name}")

    balance = 1000.0  # Bankroll initiale
    history = []
    bets = 0
    wins = 0

    # Grouper par course
    courses = df.groupby("course_id")

    for course_id, group in courses:
        # Appliquer la stratégie pour sélectionner le cheval et la mise
        bet_selection = strategy_func(group, **kwargs)

        if bet_selection:
            cheval_idx, mise = bet_selection
            row = group.loc[cheval_idx]

            # Vérifier résultat
            gagnant = row["position"] == 1
            cote = row["cote_sp"]

            # Mise à jour bankroll
            balance -= mise
            gain = 0
            if gagnant:
                gain = mise * cote
                balance += gain
                wins += 1

            bets += 1
            history.append(
                {
                    "date": row["date"],
                    "balance": balance,
                    "mise": mise,
                    "gain": gain,
                    "resultat": "Gagné" if gagnant else "Perdu",
                }
            )

    # Analyse finale
    if bets == 0:
        logger.warning("   Aucun pari effectué.")
        return None

    final_balance = balance
    roi = (
        (final_balance - 1000) / (bets * 1.0)
    ) * 100  # ROI approximatif (basé sur mise unitaire moyenne 1€)
    # ROI précis = Profit Total / Mises Totales
    total_mises = sum(h["mise"] for h in history)
    profit_total = final_balance - 1000
    roi_percent = (profit_total / total_mises) * 100 if total_mises > 0 else 0

    hit_rate = (wins / bets) * 100

    logger.info(f"   🏁 Balance Finale: {final_balance:.2f}€")
    logger.info(f"   💰 Profit: {profit_total:.2f}€")
    logger.info(f"   📈 ROI: {roi_percent:.2f}%")
    logger.info(f"   🎯 Hit Rate: {hit_rate:.2f}% ({wins}/{bets})")

    return pd.DataFrame(history)


# --- Stratégies ---


def strategy_flat_top1(group, threshold=0.3):
    """Mise 1€ sur le favori du modèle si proba > threshold."""
    # Trouver le cheval avec la max proba
    best_horse = group.loc[group["prob_gnn"].idxmax()]

    if best_horse["prob_gnn"] > threshold:
        return best_horse.name, 1.0  # Index, Mise
    return None


def strategy_value_bet(group, min_edge=0.05, kelly_fraction=0.1):
    """Mise si Value (Proba > 1/Cote). Mise proportionnelle (Kelly)."""
    # Chercher les opportunités de value
    candidates = []
    for idx, row in group.iterrows():
        if row["cote_sp"] > 1.0:
            implied_prob = 1.0 / row["cote_sp"]
            edge = row["prob_gnn"] - implied_prob

            if edge > min_edge:
                candidates.append((idx, edge, row["cote_sp"], row["prob_gnn"]))

    if not candidates:
        return None

    # Prendre la meilleure value
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_idx, edge, cote, prob = candidates[0]

    # Kelly Criterion: f = (bp - q) / b = (p(b+1) - 1) / b
    # b = cote - 1
    # p = prob
    b = cote - 1
    kelly = (prob * b - (1 - prob)) / b

    # Sécurité: Fraction de Kelly
    mise = max(
        0, kelly * kelly_fraction * 1000
    )  # Mise basée sur bankroll fixe 1000 pour simplifier
    mise = min(mise, 50.0)  # Cap mise max
    mise = max(mise, 1.0)  # Mise min

    return best_idx, mise


def main():
    logger.info("🚀 Démarrage Backtesting Phase 10")

    if not pd.io.common.file_exists(PREDICTIONS_PATH):
        logger.error(
            f"❌ Fichier {PREDICTIONS_PATH} introuvable. Lancez generate_predictions_phase10.py d'abord."
        )
        return

    df = pd.read_csv(PREDICTIONS_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Nettoyage cotes
    df = df[df["cote_sp"] > 0]  # On ne peut pas parier sans cote

    logger.info(f"📊 Données chargées: {len(df)} lignes")

    # 1. Stratégie Flat Betting (Top 1 > 30%)
    res_flat = run_backtest(df, "Flat Betting (Top 1, Conf>30%)", strategy_flat_top1, threshold=0.3)

    # 2. Stratégie Flat Betting (Top 1 > 50%)
    res_flat_secure = run_backtest(
        df, "Flat Betting (Top 1, Conf>50%)", strategy_flat_top1, threshold=0.5
    )

    # 3. Stratégie Value Bet (Kelly)
    res_value = run_backtest(df, "Value Betting (Kelly 10%)", strategy_value_bet, min_edge=0.02)

    # 4. Stratégie Value Bet (Kelly Calibré)
    CALIBRATED_PATH = "data/backtest_predictions_calibrated.csv"
    if pd.io.common.file_exists(CALIBRATED_PATH):
        df_calib = pd.read_csv(CALIBRATED_PATH)
        df_calib["date"] = pd.to_datetime(df_calib["date"])
        df_calib = df_calib[df_calib["cote_sp"] > 0]

        # Remplacer prob_gnn par prob_calibrated pour la stratégie
        df_calib["prob_gnn_raw"] = df_calib["prob_gnn"]
        df_calib["prob_gnn"] = df_calib["prob_calibrated"]

        logger.info(f"\n📊 Données Calibrées chargées: {len(df_calib)} lignes")
        res_value_calib = run_backtest(
            df_calib, "Value Betting (Kelly 10% - Calibré)", strategy_value_bet, min_edge=0.02
        )


if __name__ == "__main__":
    main()
