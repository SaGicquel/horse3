import pandas as pd
import torch
import pickle
import logging
import os
import argparse
from datetime import datetime
from models.phase12.graph_nn_v2 import GNNPredictorV2, EntityGraphNNV2
from generate_predictions_phase10 import prepare_features

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "models/phase12/saved_models/gnn_v2.pth"
GRAPH_PATH = "data/phase9/graphs/entity_graph_v1.pkl"
CALIBRATOR_PATH = "models/phase10/calibrator.pkl"
PAPER_LOG_PATH = "data/paper_trading_log.csv"

# Hyperparamètres V2
HIDDEN_DIM = 128
HEADS = 4


def load_model_and_resources():
    """Charge le modèle V2, le graphe et le calibrateur."""
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

    # Modèle V2
    metadata = graph_data.metadata()
    gnn_backbone = EntityGraphNNV2(
        metadata=metadata, hidden_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM, heads=HEADS
    )
    model = GNNPredictorV2(gnn_model=gnn_backbone, hidden_dim=HIDDEN_DIM)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info(f"✅ Modèle V2 chargé depuis {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle V2: {e}")
        raise e

    model.to(device)
    model.eval()

    # Calibrateur
    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        with open(CALIBRATOR_PATH, "rb") as f:
            calibrator = pickle.load(f)
            logger.info("✅ Calibrateur chargé")

    return model, graph_data, calibrator, device


def run_paper_trading_today(input_file="data/ml_features_complete.csv"):
    """Exécute le pipeline de paper trading pour aujourd'hui avec le modèle V2."""
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"🗞️ Démarrage Paper Trading V2 pour le {today}")

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
            df_today = df
            logger.info(f"📅 Traitement du fichier complet ({len(df_today)} lignes)")

        if df_today.empty:
            logger.warning("Aucune course trouvée.")
            return

        model, graph_data, calibrator, device = load_model_and_resources()

        bets = []

        for course_id, group in df_today.groupby("id_course"):
            # Prédiction
            # prepare_features retourne X (features cheval) et les indices
            X, h_idx, j_idx, t_idx = prepare_features(group.copy(), graph_data, device)

            # Hack dimensions pour V2
            # Le modèle V2 projette tout vers hidden_dim, donc on peut passer n'importe quelle dimension d'entrée
            # tant que c'est cohérent avec ce qu'il a vu à l'entraînement (qui était le graphe complet)
            # Ici on fait de l'inférence inductive pure sur les nœuds existants

            # On doit passer le graphe complet (x_dict, edge_index_dict) au modèle
            # car le GNN a besoin de la structure pour propager l'info
            x_dict = {k: v.to(device) for k, v in graph_data.x_dict.items()}
            edge_index_dict = {k: v.to(device) for k, v in graph_data.edge_index_dict.items()}

            with torch.no_grad():
                # Note: On passe le graphe complet, mais on ne demande la prédiction que pour les indices du batch (h_idx, etc.)
                out = model(x_dict, edge_index_dict, (h_idx, j_idx, t_idx))
                probs = torch.sigmoid(out).cpu().numpy().flatten()

            # Calibration
            if calibrator:
                probs_calibrated = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
            else:
                probs_calibrated = probs

            group["proba_win"] = probs_calibrated

            # Stratégie de pari (Flat Betting > Seuil)
            SEUIL = 0.4  # Seuil ajusté pour V2 (à affiner)

            for i, row in group.iterrows():
                if row["proba_win"] > SEUIL:
                    bets.append(
                        {
                            "date": row["date_course"].date() if "date_course" in row else today,
                            "course_id": course_id,
                            "cheval_id": row["id_cheval"],
                            "cheval_nom": row.get("nom_cheval", "Inconnu"),
                            "mise": 10.0,  # Flat betting
                            "cote": row.get("cote_final", 0.0),
                            "proba_modele": row["proba_win"],
                            "type_pari": "Simple Gagnant",
                            "statut": "En cours",
                        }
                    )

        # Sauvegarde
        if bets:
            new_bets_df = pd.DataFrame(bets)
            logger.info(f"💰 {len(new_bets_df)} paris générés")

            if os.path.exists(PAPER_LOG_PATH):
                existing_log = pd.read_csv(PAPER_LOG_PATH)
                final_log = pd.concat([existing_log, new_bets_df], ignore_index=True)
            else:
                final_log = new_bets_df

            final_log.to_csv(PAPER_LOG_PATH, index=False)
            logger.info(f"✅ Log mis à jour: {PAPER_LOG_PATH}")
            print(new_bets_df[["course_id", "cheval_nom", "proba_modele", "mise"]])
        else:
            logger.info("🚫 Aucun pari qualifié aujourd'hui.")

    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="data/ml_features_complete.csv", help="Fichier d'entrée"
    )
    args = parser.parse_args()

    run_paper_trading_today(args.input)
