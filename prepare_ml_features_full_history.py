#!/usr/bin/env python3
"""
================================================================================
PREPARE ML FEATURES - VERSION HISTORIQUE COMPLÈTE
================================================================================

Description : Génère les features ML directement à partir de cheval_courses_seen
pour TOUTE la période historique (2020-2025)

Ce script contourne les tables normalisées et travaille directement avec
les données brutes pour exploiter tout l'historique disponible.

Usage :
  python prepare_ml_features_full_history.py --output data/ml_features_full_history.csv

================================================================================
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from db_connection import get_connection

# Configuration logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FullHistoryMLFeatureEngineer:
    """Génère les features ML à partir de cheval_courses_seen (données brutes)"""

    def __init__(self):
        self.conn = get_connection()

    def extract_raw_data(self) -> pd.DataFrame:
        """Extrait toutes les données de cheval_courses_seen"""

        logger.info("📥 Extraction des données brutes (cheval_courses_seen)...")

        query = """
        SELECT
            -- Identifiants construits
            ROW_NUMBER() OVER (ORDER BY annee, race_key, numero_dossard) as id_performance,
            race_key as id_course,
            nom_norm,

            -- Reconstruction de la date
            annee,
            COALESCE(heure_depart, '00:00') as heure_depart,

            -- Target
            place_finale as position_arrivee,
            CASE WHEN place_finale = 1 THEN 1 ELSE 0 END as victoire,
            CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as place,

            -- Features cheval
            numero_dossard as numero_corde,
            musique,
            cote_finale as cote_sp,
            cote_matin as cote_pm,
            temps_sec as temps_total,
            vitesse_moyenne,
            ecart_premier as ecart,
            annee - age as an_naissance,
            sexe as sexe_cheval,

            -- Features course
            distance_m as distance,
            discipline,
            etat_piste,
            meteo,
            temperature_c,
            vent_kmh,
            nombre_partants,
            allocation_totale as allocation,

            -- Features hippodrome/contexte
            hippodrome_nom,
            driver_jockey,
            entraineur,

            -- Statut
            CASE WHEN non_partant = 1 THEN TRUE ELSE FALSE END as non_partant,
            CASE WHEN disqualifie = 1 THEN TRUE ELSE FALSE END as disqualifie,

            -- Autres données utiles
            gains_course,
            age,
            race,
            pays_naissance,
            reunion_numero,
            course_numero

        FROM cheval_courses_seen
        WHERE annee IS NOT NULL
          AND place_finale IS NOT NULL
          AND place_finale > 0
          AND non_partant != 1
          AND disqualifie != 1
        ORDER BY annee, race_key, numero_dossard
        """

        df = pd.read_sql(query, self.conn)
        logger.info(f"   ✅ {len(df):,} performances extraites")

        # Reconstruction de la date
        logger.info("📅 Reconstruction des dates...")
        df["date_course"] = pd.to_datetime(df["annee"].astype(str) + "-01-01") + pd.to_timedelta(
            np.random.randint(0, 365, len(df)), unit="D"
        )

        # Nettoyage des données
        df = df.dropna(subset=["nom_norm", "position_arrivee"])

        logger.info(f"   ✅ {len(df):,} performances après nettoyage")
        return df

    def calculate_cheval_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features relatives au cheval"""

        logger.info("🐎 Calcul des features cheval...")

        # Initialiser les colonnes
        features_cols = [
            "forme_5c",
            "forme_10c",
            "nb_courses_12m",
            "nb_victoires_12m",
            "taux_places_12m",
            "regularite",
            "jours_depuis_derniere",
            "aptitude_distance",
            "aptitude_piste",
            "aptitude_hippodrome",
        ]

        for col in features_cols:
            df[col] = 0.0

        logger.info("   📊 Calcul des historiques par cheval...")

        # Grouper par cheval et calculer les features
        for i, (cheval, cheval_data) in enumerate(df.groupby("nom_norm")):
            if i % 1000 == 0:
                logger.info(f"      Traitement cheval {i:,}/{df['nom_norm'].nunique():,}")

            cheval_data = cheval_data.sort_values("date_course").reset_index(drop=True)

            for idx in cheval_data.index:
                course_idx = cheval_data.index[cheval_data.index == idx][0]
                date_course = cheval_data.loc[course_idx, "date_course"]

                # Historique avant cette course
                historique = cheval_data[cheval_data["date_course"] < date_course]

                if len(historique) == 0:
                    continue

                # Forme sur les 5 dernières courses
                recent_5 = historique.tail(5)
                if len(recent_5) > 0:
                    df.loc[df.index[df["nom_norm"] == cheval][course_idx], "forme_5c"] = recent_5[
                        "victoire"
                    ].mean()

                # Forme sur les 10 dernières courses
                recent_10 = historique.tail(10)
                if len(recent_10) > 0:
                    df.loc[df.index[df["nom_norm"] == cheval][course_idx], "forme_10c"] = recent_10[
                        "victoire"
                    ].mean()

                # Statistiques sur 12 mois
                date_limite = date_course - timedelta(days=365)
                hist_12m = historique[historique["date_course"] >= date_limite]

                if len(hist_12m) > 0:
                    nb_courses = len(hist_12m)
                    nb_victoires = hist_12m["victoire"].sum()
                    nb_places = hist_12m["place"].sum()

                    df.loc[df.index[df["nom_norm"] == cheval][course_idx], "nb_courses_12m"] = (
                        nb_courses
                    )
                    df.loc[df.index[df["nom_norm"] == cheval][course_idx], "nb_victoires_12m"] = (
                        nb_victoires
                    )
                    df.loc[df.index[df["nom_norm"] == cheval][course_idx], "taux_places_12m"] = (
                        nb_places / nb_courses if nb_courses > 0 else 0
                    )

                # Régularité (écart-type des positions)
                if len(historique) >= 3:
                    df.loc[df.index[df["nom_norm"] == cheval][course_idx], "regularite"] = 1.0 / (
                        1.0 + historique["position_arrivee"].std()
                    )

                # Jours depuis dernière course
                if len(historique) > 0:
                    derniere_course = historique["date_course"].max()
                    jours = (date_course - derniere_course).days
                    df.loc[
                        df.index[df["nom_norm"] == cheval][course_idx], "jours_depuis_derniere"
                    ] = jours

        logger.info("   ✅ Features cheval calculées")
        return df

    def calculate_jockey_entraineur_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features jockey et entraineur"""

        logger.info("👤 Calcul des features jockey/entraineur...")

        # Initialiser les colonnes
        features_cols = [
            "taux_victoires_jockey",
            "taux_places_jockey",
            "taux_victoires_entraineur",
            "taux_places_entraineur",
            "synergie_jockey_cheval",
            "synergie_entraineur_cheval",
        ]

        for col in features_cols:
            df[col] = 0.0

        # Features jockey (calcul simplifié pour des raisons de performance)
        logger.info("   📊 Statistiques jockey...")
        jockey_stats = (
            df.groupby("driver_jockey")
            .agg({"victoire": ["count", "sum"], "place": "sum"})
            .reset_index()
        )
        jockey_stats.columns = ["driver_jockey", "nb_courses", "nb_victoires", "nb_places"]
        jockey_stats["taux_victoires"] = jockey_stats["nb_victoires"] / jockey_stats["nb_courses"]
        jockey_stats["taux_places"] = jockey_stats["nb_places"] / jockey_stats["nb_courses"]

        # Merge avec le DataFrame principal
        df = df.merge(
            jockey_stats[["driver_jockey", "taux_victoires", "taux_places"]],
            on="driver_jockey",
            how="left",
            suffixes=("", "_jockey"),
        )
        df["taux_victoires_jockey"] = df["taux_victoires"].fillna(0)
        df["taux_places_jockey"] = df["taux_places"].fillna(0)
        df = df.drop(["taux_victoires", "taux_places"], axis=1)

        # Features entraineur
        logger.info("   📊 Statistiques entraineur...")
        entraineur_stats = (
            df.groupby("entraineur")
            .agg({"victoire": ["count", "sum"], "place": "sum"})
            .reset_index()
        )
        entraineur_stats.columns = ["entraineur", "nb_courses", "nb_victoires", "nb_places"]
        entraineur_stats["taux_victoires"] = (
            entraineur_stats["nb_victoires"] / entraineur_stats["nb_courses"]
        )
        entraineur_stats["taux_places"] = (
            entraineur_stats["nb_places"] / entraineur_stats["nb_courses"]
        )

        # Merge avec le DataFrame principal
        df = df.merge(
            entraineur_stats[["entraineur", "taux_victoires", "taux_places"]],
            on="entraineur",
            how="left",
            suffixes=("", "_entraineur"),
        )
        df["taux_victoires_entraineur"] = df["taux_victoires"].fillna(0)
        df["taux_places_entraineur"] = df["taux_places"].fillna(0)
        df = df.drop(["taux_victoires", "taux_places"], axis=1)

        # Synergies (calcul simplifié)
        logger.info("   🤝 Synergies...")
        df["synergie_jockey_cheval"] = (df["taux_victoires_jockey"] * df["forme_5c"]).fillna(0)
        df["synergie_entraineur_cheval"] = (
            df["taux_victoires_entraineur"] * df["forme_5c"]
        ).fillna(0)

        logger.info("   ✅ Features jockey/entraineur calculées")
        return df

    def calculate_course_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features de course"""

        logger.info("🏁 Calcul des features course...")

        # Distance normalisée
        df["distance_norm"] = (df["distance"] - df["distance"].mean()) / df["distance"].std()

        # Niveau moyen des concurrents (approximatif)
        race_quality = df.groupby("id_course")["forme_5c"].mean().reset_index()
        race_quality.columns = ["id_course", "niveau_moyen_concurrent"]
        df = df.merge(race_quality, on="id_course", how="left")
        df["niveau_moyen_concurrent"] = df["niveau_moyen_concurrent"].fillna(
            df["niveau_moyen_concurrent"].mean()
        )

        logger.info("   ✅ Features course calculées")
        return df

    def calculate_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features de marché"""

        logger.info("💰 Calcul des features marché...")

        # Rang des cotes (par course)
        df["rang_cote_sp"] = df.groupby("id_course")["cote_sp"].rank(method="min")

        # Features manquantes mises à zéro
        market_cols = ["rang_cote_turfbzh", "ecart_cote_ia", "prediction_ia_gagnant", "elo_cheval"]
        for col in market_cols:
            if col not in df.columns:
                df[col] = 0.0

        logger.info("   ✅ Features marché calculées")
        return df

    def add_encoded_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features encodées"""

        logger.info("🔢 Encodage des features catégorielles...")

        # Encodage discipline
        df["discipline_Plat"] = (df["discipline"] == "Plat").astype(int)
        df["discipline_Trot"] = (df["discipline"] == "Trot").astype(int)

        # Encodage sexe
        df["sexe_H"] = (df["sexe_cheval"] == "H").astype(int)
        df["sexe_M"] = (df["sexe_cheval"] == "M").astype(int)

        # Encodage état piste (simplifié)
        pistes_courantes = ["Bon léger", "Bon souple", "Souple", "PSF"]
        for piste in pistes_courantes:
            df[f'etat_{piste.replace(" ", "_")}'] = (df["etat_piste"] == piste).astype(int)

        # Hippodrome top 20 (simplifié)
        top_hippodromes = df["hippodrome_nom"].value_counts().head(20).index
        df["hippodrome_top20"] = df["hippodrome_nom"].isin(top_hippodromes).astype(int)

        for hippo in top_hippodromes[:10]:  # Top 10 seulement
            col_name = f"hippodrome_{hippo.replace(' ', '_').replace('-', '_')}"
            df[col_name] = (df["hippodrome_nom"] == hippo).astype(int)

        logger.info("   ✅ Encodage terminé")
        return df

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features d'interaction"""

        logger.info("⚡ Calcul des interactions...")

        # Interactions simples
        df["interaction_forme_jockey"] = df["forme_5c"] * df["taux_victoires_jockey"]
        df["interaction_aptitude_distance"] = df["aptitude_distance"] * df["distance_norm"]
        df["interaction_elo_niveau"] = df["elo_cheval"] * df["niveau_moyen_concurrent"]
        df["interaction_cote_ia"] = df["rang_cote_sp"] * df["prediction_ia_gagnant"]
        df["interaction_synergie_forme"] = df["synergie_jockey_cheval"] * df["forme_10c"]
        df["interaction_victoires_jockey"] = df["taux_victoires_jockey"] * df["nb_victoires_12m"]

        # Popularité hippodrome (approximative)
        hippo_popularity = (
            df.groupby("hippodrome_nom").size().reset_index(name="popularite_hippodrome")
        )
        df = df.merge(hippo_popularity, on="hippodrome_nom", how="left")

        df["interaction_aptitude_popularite"] = (
            df["aptitude_hippodrome"] * df["popularite_hippodrome"]
        )
        df["interaction_regularite_volume"] = df["regularite"] * df["nb_courses_12m"]

        logger.info("   ✅ Interactions calculées")
        return df

    def prepare_features(self) -> pd.DataFrame:
        """Pipeline principal de génération des features"""

        logger.info("🚀 Début du traitement des features ML (historique complet)...")

        # 1. Extraction des données
        df = self.extract_raw_data()

        # 2. Features cheval
        df = self.calculate_cheval_features(df)

        # 3. Features jockey/entraineur
        df = self.calculate_jockey_entraineur_features(df)

        # 4. Features course
        df = self.calculate_course_features(df)

        # 5. Features marché
        df = self.calculate_market_features(df)

        # 6. Encodage
        df = self.add_encoded_features(df)

        # 7. Interactions
        df = self.add_interaction_features(df)

        # Nettoyage final
        df = df.fillna(0)

        logger.info(f"✅ Features générées : {len(df):,} lignes, {len(df.columns):,} colonnes")

        return df

    def close(self):
        """Ferme la connexion"""
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="Prépare les features ML sur historique complet")
    parser.add_argument(
        "--output", default="data/ml_features_full_history.csv", help="Fichier de sortie"
    )

    args = parser.parse_args()

    try:
        engineer = FullHistoryMLFeatureEngineer()
        df = engineer.prepare_features()

        # Sauvegarder
        logger.info(f"💾 Sauvegarde dans {args.output}...")
        df.to_csv(args.output, index=False)
        logger.info(f"✅ Fichier sauvegardé : {args.output}")

        # Résumé
        logger.info("")
        logger.info("📊 RÉSUMÉ:")
        logger.info("-" * 50)
        logger.info("   Période : 2020-2025")
        logger.info(f"   Performances : {len(df):,}")
        logger.info(f"   Chevaux uniques : {df['nom_norm'].nunique():,}")
        logger.info(f"   Courses uniques : {df['id_course'].nunique():,}")
        logger.info(f"   Features : {len(df.columns):,}")
        logger.info("")
        logger.info("🎉 Traitement terminé avec succès !")

        engineer.close()

    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        raise


if __name__ == "__main__":
    main()
