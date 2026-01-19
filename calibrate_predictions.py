import pandas as pd
import numpy as np
import pickle
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import logging
import os
import matplotlib.pyplot as plt

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = "data/backtest_predictions.csv"
MODEL_DIR = "models/phase10"
CALIBRATOR_PATH = f"{MODEL_DIR}/calibrator.pkl"
CALIBRATION_PLOT_PATH = "data/calibration_plot.png"


def reliability_curve(y_true, y_prob, n_bins=10):
    """Calcule la courbe de fiabilité."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1

    bin_sums = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(len(y_prob)):
        idx = min(bin_indices[i], n_bins - 1)
        bin_sums[idx] += y_prob[i]
        bin_true[idx] += y_true[i]
        bin_counts[idx] += 1

    prob_pred = bin_sums / (bin_counts + 1e-10)
    prob_true = bin_true / (bin_counts + 1e-10)

    return prob_pred, prob_true, bin_counts


def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    logger.info(f"📊 Chargement des prédictions depuis {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # Tri par date
    df = df.sort_values("date")

    # Split 50/50 temporel (Calibration / Test)
    # On utilise la première moitié pour apprendre à calibrer
    # Et la seconde pour vérifier si ça améliore les choses
    split_idx = int(len(df) * 0.5)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df["prob_gnn"].values
    y_train = train_df["gagnant"].values

    X_test = test_df["prob_gnn"].values
    y_test = test_df["gagnant"].values

    logger.info(f"Set Calibration: {len(train_df)} lignes")
    logger.info(f"Set Test: {len(test_df)} lignes")

    # 1. Entraînement Isotonic Regression
    logger.info("🔧 Entraînement Isotonic Regression...")
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_reg.fit(X_train, y_train)

    # 2. Prédictions calibrées
    prob_calibrated_test = iso_reg.predict(X_test)
    test_df["prob_calibrated"] = prob_calibrated_test

    # 3. Métriques
    brier_before = brier_score_loss(y_test, X_test)
    brier_after = brier_score_loss(y_test, prob_calibrated_test)

    logger.info(f"📉 Brier Score Avant: {brier_before:.5f}")
    logger.info(
        f"📉 Brier Score Après: {brier_after:.5f} ({'✅ Amélioration' if brier_after < brier_before else '❌ Dégradation'})"
    )

    # 4. Sauvegarde
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump(iso_reg, f)
    logger.info(f"💾 Calibrateur sauvegardé: {CALIBRATOR_PATH}")

    # 5. Visualisation rapide (ASCII)
    logger.info("\n📊 Analyse par bins (Avant vs Après Calibration):")
    logger.info(
        f"{'Bin':<10} | {'Count':<6} | {'Pred Avg':<10} | {'True Avg':<10} | {'Calib Avg':<10}"
    )
    logger.info("-" * 60)

    bins = np.linspace(0, 1, 6)
    for i in range(len(bins) - 1):
        mask = (X_test >= bins[i]) & (X_test < bins[i + 1])
        if mask.sum() > 0:
            count = mask.sum()
            pred_avg = X_test[mask].mean()
            true_avg = y_test[mask].mean()
            calib_avg = prob_calibrated_test[mask].mean()
            logger.info(
                f"{bins[i]:.1f}-{bins[i+1]:.1f} | {count:<6} | {pred_avg:.4f}     | {true_avg:.4f}     | {calib_avg:.4f}"
            )

    # 6. Sauvegarde du dataset de test avec calibration pour backtest
    test_df.to_csv("data/backtest_predictions_calibrated.csv", index=False)
    logger.info("✅ Fichier de test calibré généré: data/backtest_predictions_calibrated.csv")


if __name__ == "__main__":
    main()
