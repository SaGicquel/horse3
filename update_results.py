import pandas as pd
import logging
import os
from db_connection import get_connection

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LOG_PATH = "data/paper_trading_log.csv"


def update_pnl():
    """Met à jour le P&L des paris en attente."""
    if not os.path.exists(LOG_PATH):
        logger.warning(f"⚠️ Fichier {LOG_PATH} introuvable.")
        return

    df = pd.read_csv(LOG_PATH)

    if "statut" not in df.columns:
        df["statut"] = "En cours"
    if "gain" not in df.columns:
        df["gain"] = 0.0

    pending_mask = df["statut"] == "En cours"
    pending_bets = df[pending_mask]

    if pending_bets.empty:
        logger.info("✅ Aucun pari en attente.")
        return

    logger.info(f"🔄 Mise à jour de {len(pending_bets)} paris en attente...")

    conn = get_connection()
    cursor = conn.cursor()

    updates_count = 0

    for idx, row in pending_bets.iterrows():
        course_id = row["course_id"]
        cheval_id = row["cheval_id"]
        mise = row["mise"]

        # Chercher le résultat dans la DB
        # On suppose que la table performances contient les résultats
        query = """
            SELECT position_arrivee, cote_sp
            FROM performances
            WHERE id_course = %s AND id_cheval = %s
        """
        cursor.execute(query, (course_id, cheval_id))
        result = cursor.fetchone()

        if result:
            position, cote_sp = result

            # Si la position est NULL, le résultat n'est pas encore là
            if position is None:
                continue

            # Mise à jour
            if position == 1:
                gain = mise * (
                    cote_sp if cote_sp else 1.0
                )  # Fallback cote 1 si manquante (ne devrait pas arriver)
                df.at[idx, "statut"] = "Gagné"
                df.at[idx, "gain"] = gain - mise  # Net profit
                logger.info(
                    f"💰 Gagné! Course {course_id} Cheval {cheval_id} -> +{gain - mise:.2f}€"
                )
            else:
                df.at[idx, "statut"] = "Perdu"
                df.at[idx, "gain"] = -mise
                logger.info(f"❌ Perdu. Course {course_id} Cheval {cheval_id} -> -{mise:.2f}€")

            updates_count += 1

    conn.close()

    if updates_count > 0:
        df.to_csv(LOG_PATH, index=False)
        logger.info(f"✅ {updates_count} paris mis à jour.")
    else:
        logger.info("⏳ Aucune nouvelle information de résultat disponible.")


if __name__ == "__main__":
    update_pnl()
