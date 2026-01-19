import pandas as pd
import torch
import pickle
import logging
import os
import argparse
from datetime import datetime
from models.phase9.graph_nn import GNNPredictor, EntityGraphNN
from generate_predictions_phase10 import prepare_features

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "models/phase9/saved_models/gnn_v1.pth"
GRAPH_PATH = "data/phase9/graphs/entity_graph_v1.pkl"
CALIBRATOR_PATH = "models/phase10/calibrator.pkl"
PAPER_LOG_PATH = "data/paper_trading_log.csv"


def load_model_and_resources():
    """Charge le modèle, le graphe et le calibrateur."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Graphe
    with open(GRAPH_PATH, "rb") as f:
        graph_data = pickle.load(f)
    if isinstance(graph_data, dict) and "graph" in graph_data:
        graph_data = graph_data["graph"]

    # Modèle
    metadata = graph_data.metadata()
    hidden_dim = 64
    gnn_backbone = EntityGraphNN(metadata=metadata, hidden_dim=hidden_dim, out_dim=hidden_dim)
    model = GNNPredictor(gnn_model=gnn_backbone, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Calibrateur
    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        with open(CALIBRATOR_PATH, "rb") as f:
            calibrator = pickle.load(f)

    return model, graph_data, calibrator, device


def run_paper_trading_today(input_file="data/ml_features_complete.csv"):
    """Exécute le pipeline de paper trading pour aujourd'hui."""
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"🗞️ Démarrage Paper Trading pour le {today}")

    try:
        if not os.path.exists(input_file):
            logger.error(f"❌ Fichier d'entrée introuvable: {input_file}")
            return

        df = pd.read_csv(input_file)

        # Si le fichier contient plusieurs dates, on prend la plus récente
        if "date_course" in df.columns:
            df["date_course"] = pd.to_datetime(df["date_course"])
            last_date = df["date_course"].max()
            df_today = df[df["date_course"] == last_date].copy()
            logger.info(f"📅 Date traitée: {last_date.date()} ({len(df_today)} partants)")
        else:
            # Si pas de date, on suppose que c'est le fichier du jour
            df_today = df
            logger.info(f"📅 Traitement du fichier complet ({len(df_today)} lignes)")

        if df_today.empty:
            logger.warning("Aucune course trouvée.")
            return

        model, graph_data, calibrator, device = load_model_and_resources()

        bets = []

        for course_id, group in df_today.groupby("id_course"):
            # Prédiction
            X, h_idx, j_idx, t_idx = prepare_features(group.copy(), graph_data, device)

            # Hack dimensions (comme dans generate_predictions)
            x_dict_batch = {
                "cheval": X,
                "jockey": torch.zeros((len(X), 2)).to(device),
                "entraineur": torch.zeros((len(X), 2)).to(device),
            }
            edge_index_dict_batch = {
                k: torch.empty((2, 0), dtype=torch.long).to(device)
                for k in graph_data.edge_index_dict.keys()
            }

            with torch.no_grad():
                out = model(x_dict_batch, edge_index_dict_batch, (h_idx, j_idx, t_idx))
                probs = torch.sigmoid(out).cpu().numpy().flatten()

            # Calibration
            if calibrator:
                probs_calib = calibrator.predict(probs)
            else:
                probs_calib = probs

            group["prob_model"] = probs_calib

            # Stratégie Flat Betting (Top 1)
            best_horse = group.loc[group["prob_model"].idxmax()]

            # Seuil abaissé car calibration a réduit les probas
            # Avec calibration, une proba de 0.15 peut être très bonne si le champ est grand
            # On va prendre le Top 1 sans seuil absolu trop haut, ou un seuil relatif
            if best_horse["prob_model"] > 0.08:  # Seuil ajusté post-calibration (moyenne ~0.09)
                bets.append(
                    {
                        "date": today,
                        "course_id": course_id,
                        "cheval_id": best_horse["id_cheval"],
                        "mise": 10.0,
                        "prob": best_horse["prob_model"],
                        "type": "Flat Top1",
                        "statut": "En cours",
                    }
                )

        # Enregistrement
        if bets:
            df_bets = pd.DataFrame(bets)
            header = not os.path.exists(PAPER_LOG_PATH)
            df_bets.to_csv(PAPER_LOG_PATH, mode="a", header=header, index=False)
            logger.info(f"✅ {len(bets)} paris enregistrés dans {PAPER_LOG_PATH}")
            print(df_bets[["course_id", "cheval_id", "prob", "mise"]])
        else:
            logger.info("Aucun pari sélectionné aujourd'hui.")

    except Exception as e:
        logger.error(f"Erreur Paper Trading: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/ml_features_complete.csv")
    args = parser.parse_args()
    run_paper_trading_today(args.input)
