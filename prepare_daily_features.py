import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from db_connection import get_connection

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DailyFeatureEngineer:
    def __init__(self, target_date):
        self.conn = get_connection()
        self.target_date = target_date
        logger.info(f"🎯 Préparation features pour le {target_date}")

    def get_target_races(self):
        """Récupère les partants du jour cible."""
        query = """
        SELECT
            p.id_performance, p.id_course, p.id_cheval, p.id_jockey, p.id_entraineur,
            c.date_course, c.id_hippodrome, c.distance, c.discipline, c.etat_piste,
            ch.an_naissance, ch.sexe_cheval,
            h.nom_hippodrome, h.type_piste
        FROM performances p
        JOIN courses c ON p.id_course = c.id_course
        JOIN chevaux ch ON p.id_cheval = ch.id_cheval
        LEFT JOIN hippodromes h ON c.id_hippodrome = h.id_hippodrome
        WHERE date(c.date_course) = %s
        AND p.non_partant = FALSE
        """
        return pd.read_sql(query, self.conn, params=(self.target_date,))

    def get_horse_history(self, horse_ids):
        """Récupère l'historique des chevaux engagés."""
        if not horse_ids:
            return pd.DataFrame()

        placeholders = ",".join(["%s"] * len(horse_ids))
        query = f"""
        SELECT
            p.id_cheval, c.date_course, p.position_arrivee, c.allocation, c.distance, h.type_piste
        FROM performances p
        JOIN courses c ON p.id_course = c.id_course
        LEFT JOIN hippodromes h ON c.id_hippodrome = h.id_hippodrome
        WHERE p.id_cheval IN ({placeholders})
        AND c.date_course < %s
        AND p.position_arrivee IS NOT NULL
        ORDER BY c.date_course ASC
        """
        params = list(horse_ids) + [self.target_date]
        return pd.read_sql(query, self.conn, params=params)

    def compute_features(self):
        # 1. Charger les courses du jour
        df_today = self.get_target_races()
        if df_today.empty:
            logger.warning("Aucune course trouvée pour cette date.")
            return pd.DataFrame()

        logger.info(f"📊 {len(df_today)} partants trouvés.")

        # 2. Charger l'historique
        horse_ids = df_today["id_cheval"].unique().tolist()
        logger.info(f"📚 Chargement historique pour {len(horse_ids)} chevaux...")
        df_history = self.get_horse_history(horse_ids)

        # 3. Calculer les features
        # On va faire simple pour l'instant : aligner avec les 6 features du GNN Phase 10
        # 'an_naissance', 'taux_places_carriere', 'gains_carriere', 'jours_depuis_derniere', 'aptitude_piste', 'aptitude_distance'

        features = []

        for idx, row in df_today.iterrows():
            hid = row["id_cheval"]
            hist = df_history[df_history["id_cheval"] == hid]

            # Base
            feat = row.to_dict()

            # Calculs
            if not hist.empty:
                # Jours depuis dernière
                last_date = pd.to_datetime(hist.iloc[-1]["date_course"])
                current_date = pd.to_datetime(row["date_course"])
                feat["jours_depuis_derniere"] = (current_date - last_date).days

                # Taux places (Top 3)
                nb_courses = len(hist)
                nb_places = len(hist[hist["position_arrivee"] <= 3])
                feat["taux_places_carriere"] = nb_places / nb_courses if nb_courses > 0 else 0

                # Gains (Approximation via allocation, ou 0 si pas dispo)
                # On n'a pas le gain exact dans la query history ci-dessus, on met 0 ou on améliore la query
                # Pour l'instant 0 car pas critique pour le test
                feat["gains_carriere"] = 0

                # Aptitudes (Simplifié)
                feat["aptitude_piste"] = 0.5  # Dummy
                feat["aptitude_distance"] = 0.5  # Dummy
            else:
                feat["jours_depuis_derniere"] = 365  # Valeur par défaut
                feat["taux_places_carriere"] = 0
                feat["gains_carriere"] = 0
                feat["aptitude_piste"] = 0
                feat["aptitude_distance"] = 0

            features.append(feat)

        return pd.DataFrame(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--output", type=str, default="data/daily_features.csv")
    args = parser.parse_args()

    engineer = DailyFeatureEngineer(args.date)
    df_features = engineer.compute_features()

    if not df_features.empty:
        df_features.to_csv(args.output, index=False)
        logger.info(f"✅ Features sauvegardées dans {args.output}")
    else:
        logger.warning("❌ Échec calcul features.")


if __name__ == "__main__":
    main()
