#!/usr/bin/env python3
"""
================================================================================
PREPARE ML FEATURES - PHASE 4
================================================================================

Description : Calcul des 90+ features pour le machine learning

Features calculées :
  • Groupe 1 (Cheval) : 35 features
    - Forme récente (5c, 10c), taux victoires, régularité
    - Aptitudes (distance, piste, hippodrome)
    - Progression, gains, jours depuis dernière course

  • Groupe 2 (Course) : 20 features
    - Distance normalisée, nb partants, niveau concurrent
    - Conditions (piste, météo, discipline)
    - Qualité de l'hippodrome

  • Groupe 3 (Relationnelles) : 15 features
    - Performance jockey, entraineur
    - Synergies jockey-cheval, entraineur-cheval
    - Expérience dans la configuration

  • Groupe 4 (Contexte) : 10 features
    - Numéro de corde, handicap, ferrure
    - Statut (favori, outsider), engagement

  • Groupe 5 (Marché) : 10 features
    - Cotes (SP, PM, Turf.bzh)
    - Prédictions IA, ELO
    - Ecart cote/performance

Usage :
  python prepare_ml_features.py --output data/ml_features.csv

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MLFeatureEngineer:
    """Calculateur de features pour machine learning"""

    def __init__(self):
        self.conn = get_connection()
        logger.info("✅ Connexion base de données établie")

    def extract_base_data(self) -> pd.DataFrame:
        """Extrait les données brutes de la base"""
        logger.info("📊 Extraction des données brutes...")

        query = """
        SELECT
            -- Identifiants (construits)
            ROW_NUMBER() OVER (ORDER BY annee, race_key, numero_dossard) as id_performance,
            race_key as id_course,
            nom_norm as id_cheval,
            DENSE_RANK() OVER (ORDER BY driver_jockey) as id_jockey,
            DENSE_RANK() OVER (ORDER BY entraineur) as id_entraineur,
            (annee::text || '-01-01')::date + (ROW_NUMBER() OVER (PARTITION BY annee ORDER BY race_key) % 365) * INTERVAL '1 day' as date_course,
            DENSE_RANK() OVER (ORDER BY hippodrome_nom) as id_hippodrome,

            -- Target (variables à prédire)
            place_finale as position_arrivee,
            CASE WHEN place_finale = 1 THEN 1 ELSE 0 END as victoire,
            CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END as place,

            -- Features cheval brutes
            numero_dossard as numero_corde,
            COALESCE(musique, '') as musique,
            COALESCE(cote_finale, 5.0) as cote_sp,
            COALESCE(cote_matin, 5.0) as cote_pm,
            0.0 as cote_turfbzh,  -- Pas disponible dans cheval_courses_seen
            0.5 as prediction_ia_gagnant,  -- Valeur par défaut
            1000 as elo_cheval,  -- Valeur par défaut
            COALESCE(temps_sec, 60.0) as temps_total,
            COALESCE(vitesse_moyenne, 50.0) as vitesse_moyenne,
            COALESCE(ecart_premier, 0.0) as ecart,
            COALESCE(annee - age, 2015) as an_naissance,
            COALESCE(sexe, 'H') as sexe_cheval,

            -- Features course brutes
            COALESCE(distance_m, 2000) as distance,
            COALESCE(discipline, 'Trot') as discipline,
            COALESCE(etat_piste, 'Bon') as etat_piste,
            COALESCE(meteo, 'Beau') as meteo,
            COALESCE(temperature_c, 15.0) as temperature_c,
            COALESCE(vent_kmh, 0.0) as vent_kmh,
            COALESCE(nombre_partants, 12) as nombre_partants,
            COALESCE(allocation_totale, 10000) as allocation,

            -- Features hippodrome
            COALESCE(hippodrome_nom, 'UNKNOWN') as nom_hippodrome,
            'Standard' as type_piste,  -- Valeur par défaut
            COALESCE(region_hippodrome, 'France') as hippodrome_ville,

            -- Statut
            CASE WHEN non_partant = 1 THEN TRUE ELSE FALSE END as non_partant

        FROM cheval_courses_seen

        WHERE annee IS NOT NULL
          AND place_finale IS NOT NULL
          AND place_finale > 0
          AND COALESCE(non_partant, 0) = 0
          AND COALESCE(disqualifie, 0) = 0

        ORDER BY annee DESC, race_key, numero_dossard
        """

        df = pd.read_sql(query, self.conn)
        logger.info(f"   ✅ {len(df):,} performances extraites")

        return df

    def calculate_forme_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features de forme (historique courses)"""
        logger.info("🔄 Calcul features de forme (5c, 10c, régularité)...")

        # Initialiser colonnes
        df["forme_5c"] = 0.0
        df["forme_10c"] = 0.0
        df["nb_courses_12m"] = 0
        df["nb_victoires_12m"] = 0
        df["taux_places_12m"] = 0.0
        df["regularite"] = 0.0
        df["jours_depuis_derniere"] = 0

        # Grouper par cheval
        for id_cheval in df["id_cheval"].unique():
            cheval_data = df[df["id_cheval"] == id_cheval].sort_values("date_course")

            for idx in cheval_data.index:
                date_course = cheval_data.loc[idx, "date_course"]

                # Historique avant cette course
                historique = cheval_data[cheval_data["date_course"] < date_course]

                if len(historique) == 0:
                    continue

                # Forme 5 dernières courses
                hist_5 = historique.tail(5)
                positions_5 = hist_5["position_arrivee"].values
                weights = [5, 4, 3, 2, 1][: len(positions_5)]
                forme_5 = sum((6 - min(pos, 5)) * w for pos, w in zip(positions_5, weights[::-1]))
                df.loc[idx, "forme_5c"] = forme_5 / 75.0  # Normaliser [0, 1]

                # Forme 10 dernières courses
                hist_10 = historique.tail(10)
                positions_10 = hist_10["position_arrivee"].values
                weights_10 = list(range(10, 0, -1))[: len(positions_10)]
                forme_10 = sum(
                    (6 - min(pos, 5)) * w for pos, w in zip(positions_10, weights_10[::-1])
                )
                df.loc[idx, "forme_10c"] = forme_10 / 330.0  # Normaliser [0, 1]

                # Stats 12 derniers mois
                date_limite = date_course - timedelta(days=365)
                hist_12m = historique[historique["date_course"] >= date_limite]

                df.loc[idx, "nb_courses_12m"] = len(hist_12m)
                df.loc[idx, "nb_victoires_12m"] = (hist_12m["victoire"] == 1).sum()

                if len(hist_12m) > 0:
                    df.loc[idx, "taux_places_12m"] = (hist_12m["place"] == 1).sum() / len(hist_12m)
                    df.loc[idx, "regularite"] = (
                        hist_12m["position_arrivee"].std() if len(hist_12m) > 1 else 0
                    )

                # Jours depuis dernière course
                derniere_course = historique["date_course"].max()
                df.loc[idx, "jours_depuis_derniere"] = (date_course - derniere_course).days

        logger.info("   ✅ Features de forme calculées")
        return df

    def calculate_aptitude_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features d'aptitude (distance, piste, hippodrome)"""
        logger.info("🎯 Calcul features d'aptitude (distance, piste, hippodrome)...")

        df["aptitude_distance"] = 0.0
        df["aptitude_piste"] = 0.0
        df["aptitude_hippodrome"] = 0.0

        for id_cheval in df["id_cheval"].unique():
            cheval_data = df[df["id_cheval"] == id_cheval].sort_values("date_course")

            for idx in cheval_data.index:
                date_course = cheval_data.loc[idx, "date_course"]
                distance = cheval_data.loc[idx, "distance"]
                piste = cheval_data.loc[idx, "type_piste"]
                hippodrome = cheval_data.loc[idx, "id_hippodrome"]

                historique = cheval_data[cheval_data["date_course"] < date_course]

                if len(historique) == 0:
                    continue

                # Aptitude distance (±200m)
                hist_distance = historique[
                    (historique["distance"] >= distance - 200)
                    & (historique["distance"] <= distance + 200)
                ]
                if len(hist_distance) > 0:
                    df.loc[idx, "aptitude_distance"] = (hist_distance["place"] == 1).sum() / len(
                        hist_distance
                    )

                # Aptitude piste - FIX: utiliser 'victoire' au lieu de 'place'
                if pd.notna(piste):
                    hist_piste = historique[historique["type_piste"] == piste]
                    if len(hist_piste) > 0:
                        df.loc[idx, "aptitude_piste"] = (hist_piste["victoire"] == 1).sum() / len(
                            hist_piste
                        )

                # Aptitude hippodrome
                if pd.notna(hippodrome):
                    hist_hippodrome = historique[historique["id_hippodrome"] == hippodrome]
                    if len(hist_hippodrome) > 0:
                        df.loc[idx, "aptitude_hippodrome"] = (
                            hist_hippodrome["place"] == 1
                        ).sum() / len(hist_hippodrome)

        logger.info("   ✅ Features d'aptitude calculées")
        return df

    def calculate_jockey_entraineur_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features jockey/entraineur"""
        logger.info("👥 Calcul features jockey/entraineur...")

        # FIX: Initialiser avec .loc[] pour éviter FutureWarning
        df.loc[:, "taux_victoires_jockey"] = 0.0
        df.loc[:, "taux_places_jockey"] = 0.0
        df.loc[:, "taux_victoires_entraineur"] = 0.0
        df.loc[:, "taux_places_entraineur"] = 0.0
        df.loc[:, "synergie_jockey_cheval"] = 0.0
        df.loc[:, "synergie_entraineur_cheval"] = 0.0

        # Stats jockeys
        for id_jockey in df["id_jockey"].dropna().unique():
            jockey_data = df[df["id_jockey"] == id_jockey]

            for idx in jockey_data.index:
                date_course = jockey_data.loc[idx, "date_course"]
                historique = jockey_data[jockey_data["date_course"] < date_course]

                if len(historique) >= 5:  # Minimum 5 courses
                    df.loc[idx, "taux_victoires_jockey"] = (
                        historique["victoire"] == 1
                    ).sum() / len(historique)
                    df.loc[idx, "taux_places_jockey"] = (historique["place"] == 1).sum() / len(
                        historique
                    )

        # Stats entraineurs
        for id_entraineur in df["id_entraineur"].dropna().unique():
            entraineur_data = df[df["id_entraineur"] == id_entraineur]

            for idx in entraineur_data.index:
                date_course = entraineur_data.loc[idx, "date_course"]
                historique = entraineur_data[entraineur_data["date_course"] < date_course]

                if len(historique) >= 5:
                    df.loc[idx, "taux_victoires_entraineur"] = (
                        historique["victoire"] == 1
                    ).sum() / len(historique)
                    df.loc[idx, "taux_places_entraineur"] = (historique["place"] == 1).sum() / len(
                        historique
                    )

        # Synergies
        for idx in df.index:
            id_cheval = df.loc[idx, "id_cheval"]
            id_jockey = df.loc[idx, "id_jockey"]
            id_entraineur = df.loc[idx, "id_entraineur"]
            date_course = df.loc[idx, "date_course"]

            # Synergie jockey-cheval
            if pd.notna(id_jockey):
                duo_jockey = df[
                    (df["id_cheval"] == id_cheval)
                    & (df["id_jockey"] == id_jockey)
                    & (df["date_course"] < date_course)
                ]
                if len(duo_jockey) >= 3:
                    df.loc[idx, "synergie_jockey_cheval"] = (duo_jockey["place"] == 1).sum() / len(
                        duo_jockey
                    )

            # Synergie entraineur-cheval
            if pd.notna(id_entraineur):
                duo_entraineur = df[
                    (df["id_cheval"] == id_cheval)
                    & (df["id_entraineur"] == id_entraineur)
                    & (df["date_course"] < date_course)
                ]
                if len(duo_entraineur) >= 3:
                    df.loc[idx, "synergie_entraineur_cheval"] = (
                        duo_entraineur["place"] == 1
                    ).sum() / len(duo_entraineur)

        logger.info("   ✅ Features jockey/entraineur calculées")
        return df

    def calculate_course_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features de course"""
        logger.info("🏇 Calcul features de course...")

        # Distance normalisée (Z-score par discipline) - FIX: utiliser .loc[]
        df.loc[:, "distance_norm"] = 0.0
        for discipline in df["discipline"].unique():
            mask = df["discipline"] == discipline
            mean_dist = df.loc[mask, "distance"].mean()
            std_dist = df.loc[mask, "distance"].std()
            if std_dist > 0:
                df.loc[mask, "distance_norm"] = (df.loc[mask, "distance"] - mean_dist) / std_dist

        # Niveau moyen des concurrents (ELO moyens)
        df.loc[:, "niveau_moyen_concurrent"] = 0.0

        # Pour chaque course, calculer le niveau moyen des autres chevaux
        for id_course in df["id_course"].unique():
            course_data = df[df["id_course"] == id_course]

            for idx in course_data.index:
                id_cheval = course_data.loc[idx, "id_cheval"]

                # Autres chevaux de la course
                autres = course_data[course_data["id_cheval"] != id_cheval]

                # Gains moyens des autres (approximation via ELO si disponible)
                if "elo_cheval" in autres.columns:
                    elo_autres = autres["elo_cheval"].dropna()
                    if len(elo_autres) > 0:
                        df.loc[idx, "niveau_moyen_concurrent"] = elo_autres.mean()

        logger.info("   ✅ Features de course calculées")
        return df

    def add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features de marché (cotes, IA)"""
        logger.info("💰 Ajout features de marché...")

        # Rang cote (1 = favori) - FIX: utiliser .loc[] et float pour dtype
        df.loc[:, "rang_cote_sp"] = 0.0
        df.loc[:, "rang_cote_turfbzh"] = 0.0
        df.loc[:, "ecart_cote_ia"] = 0.0

        for id_course in df["id_course"].unique():
            mask = df["id_course"] == id_course

            # Rang cote SP
            cotes_sp = df.loc[mask, "cote_sp"].dropna()
            if len(cotes_sp) > 0:
                df.loc[mask & df["cote_sp"].notna(), "rang_cote_sp"] = df.loc[
                    mask, "cote_sp"
                ].rank()

            # Rang cote Turf.bzh
            cotes_turf = df.loc[mask, "cote_turfbzh"].dropna()
            if len(cotes_turf) > 0:
                df.loc[mask & df["cote_turfbzh"].notna(), "rang_cote_turfbzh"] = df.loc[
                    mask, "cote_turfbzh"
                ].rank()

            # Écart cote vs IA
            for idx in df[mask].index:
                cote = df.loc[idx, "cote_turfbzh"]
                ia_prob = df.loc[idx, "prediction_ia_gagnant"]

                if pd.notna(cote) and pd.notna(ia_prob) and ia_prob > 0:
                    cote_implicite = 1.0 / cote
                    df.loc[idx, "ecart_cote_ia"] = ia_prob - cote_implicite

        logger.info("   ✅ Features de marché ajoutées")
        return df

    def calculate_gains_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features de gains/prize money"""
        logger.info("💰 Calcul features de gains...")

        # Initialisation
        df.loc[:, "gains_carriere"] = 0.0
        df.loc[:, "gains_12m"] = 0.0
        df.loc[:, "gains_par_course"] = 0.0
        df.loc[:, "nb_premieres_places"] = 0
        df.loc[:, "nb_deuxiemes_places"] = 0
        df.loc[:, "nb_troisiemes_places"] = 0
        df.loc[:, "taux_places_carriere"] = 0.0
        df.loc[:, "gain_moyen_victoire"] = 0.0
        df.loc[:, "evolution_gains_12m"] = 0.0
        df.loc[:, "ratio_gains_courses"] = 0.0

        for id_cheval in df["id_cheval"].unique():
            cheval_data = df[df["id_cheval"] == id_cheval].sort_values("date_course")

            for idx in cheval_data.index:
                date_course = cheval_data.loc[idx, "date_course"]
                historique = cheval_data[cheval_data["date_course"] < date_course]

                if len(historique) == 0:
                    continue

                # Gains carrière (simulés via allocation * position)
                gains = historique.apply(
                    lambda x: (x["allocation"] or 0)
                    * (
                        0.5
                        if x["position_arrivee"] == 1
                        else 0.25
                        if x["position_arrivee"] == 2
                        else 0.15
                        if x["position_arrivee"] == 3
                        else 0
                    ),
                    axis=1,
                ).sum()
                df.loc[idx, "gains_carriere"] = gains

                # Gains 12 mois
                date_12m = date_course - timedelta(days=365)
                hist_12m = historique[historique["date_course"] >= date_12m]
                if len(hist_12m) > 0:
                    gains_12m = hist_12m.apply(
                        lambda x: (x["allocation"] or 0)
                        * (
                            0.5
                            if x["position_arrivee"] == 1
                            else 0.25
                            if x["position_arrivee"] == 2
                            else 0.15
                            if x["position_arrivee"] == 3
                            else 0
                        ),
                        axis=1,
                    ).sum()
                    df.loc[idx, "gains_12m"] = gains_12m

                # Gains par course
                nb_courses = len(historique)
                df.loc[idx, "gains_par_course"] = gains / nb_courses if nb_courses > 0 else 0

                # Nombre de places
                df.loc[idx, "nb_premieres_places"] = (historique["position_arrivee"] == 1).sum()
                df.loc[idx, "nb_deuxiemes_places"] = (historique["position_arrivee"] == 2).sum()
                df.loc[idx, "nb_troisiemes_places"] = (historique["position_arrivee"] == 3).sum()

                # Taux places carrière
                nb_places = (
                    (historique["position_arrivee"] >= 1) & (historique["position_arrivee"] <= 3)
                ).sum()
                df.loc[idx, "taux_places_carriere"] = (
                    nb_places / nb_courses if nb_courses > 0 else 0
                )

                # Gain moyen victoire
                victoires = historique[historique["position_arrivee"] == 1]
                if len(victoires) > 0:
                    gains_victoires = victoires["allocation"].fillna(0).mean() * 0.5
                    df.loc[idx, "gain_moyen_victoire"] = gains_victoires

                # Évolution gains (6 derniers mois vs 6 précédents)
                date_6m = date_course - timedelta(days=180)
                hist_recent = historique[(historique["date_course"] >= date_6m)]
                hist_ancien = historique[
                    (historique["date_course"] < date_6m)
                    & (historique["date_course"] >= date_6m - timedelta(days=180))
                ]

                if len(hist_recent) > 0 and len(hist_ancien) > 0:
                    gains_recent = hist_recent.apply(
                        lambda x: (x["allocation"] or 0)
                        * (
                            0.5
                            if x["position_arrivee"] == 1
                            else 0.25
                            if x["position_arrivee"] == 2
                            else 0.15
                            if x["position_arrivee"] == 3
                            else 0
                        ),
                        axis=1,
                    ).sum()
                    gains_ancien = hist_ancien.apply(
                        lambda x: (x["allocation"] or 0)
                        * (
                            0.5
                            if x["position_arrivee"] == 1
                            else 0.25
                            if x["position_arrivee"] == 2
                            else 0.15
                            if x["position_arrivee"] == 3
                            else 0
                        ),
                        axis=1,
                    ).sum()
                    df.loc[idx, "evolution_gains_12m"] = gains_recent - gains_ancien

                # Ratio gains/courses
                df.loc[idx, "ratio_gains_courses"] = gains / nb_courses if nb_courses > 0 else 0

        logger.info("   ✅ Features de gains calculées")
        return df

    def calculate_meteo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features météo/conditions"""
        logger.info("🌦️  Calcul features météo...")

        # Encodage états piste
        etat_map = {"BON": 0, "LEGER": 1, "SOUPLE": 2, "COLLANT": 3, "LOURD": 4, "TRES_LOURD": 5}
        df.loc[:, "etat_piste_encoded"] = df["etat_piste"].map(etat_map).fillna(0)

        # Encodage météo
        meteo_map = {"ENSOLEILLE": 0, "NUAGEUX": 1, "COUVERT": 2, "PLUIE": 3, "ORAGE": 4}
        df.loc[:, "meteo_encoded"] = df["meteo"].map(meteo_map).fillna(1)

        # Aptitude piste état (win rate par condition)
        df.loc[:, "aptitude_piste_etat"] = 0.0

        for id_cheval in df["id_cheval"].unique():
            cheval_data = df[df["id_cheval"] == id_cheval].sort_values("date_course")

            for idx in cheval_data.index:
                date_course = cheval_data.loc[idx, "date_course"]
                etat = cheval_data.loc[idx, "etat_piste"]

                historique = cheval_data[cheval_data["date_course"] < date_course]

                if len(historique) > 0 and pd.notna(etat):
                    hist_etat = historique[historique["etat_piste"] == etat]
                    if len(hist_etat) >= 3:
                        df.loc[idx, "aptitude_piste_etat"] = (
                            hist_etat["victoire"] == 1
                        ).sum() / len(hist_etat)

        # Température optimale (écart vs meilleure temp)
        df.loc[:, "ecart_temp_optimal"] = 0.0

        for id_cheval in df["id_cheval"].unique():
            cheval_data = df[df["id_cheval"] == id_cheval].sort_values("date_course")

            for idx in cheval_data.index:
                date_course = cheval_data.loc[idx, "date_course"]
                temp_actuelle = cheval_data.loc[idx, "temperature_c"]

                historique = cheval_data[cheval_data["date_course"] < date_course]
                victoires = historique[historique["victoire"] == 1]

                if len(victoires) > 0 and pd.notna(temp_actuelle):
                    temp_opt = victoires["temperature_c"].dropna().mean()
                    if pd.notna(temp_opt):
                        df.loc[idx, "ecart_temp_optimal"] = abs(temp_actuelle - temp_opt)

        # Interaction piste-météo (combine les 2 effets)
        df.loc[:, "interaction_piste_meteo"] = df["etat_piste_encoded"] * df["meteo_encoded"]

        # Handicap météo (pénalité conditions difficiles)
        df.loc[:, "handicap_meteo"] = 0.0
        conditions_difficiles = (df["etat_piste_encoded"] >= 3) | (df["meteo_encoded"] >= 3)
        df.loc[conditions_difficiles, "handicap_meteo"] = 1.0

        logger.info("   ✅ Features météo calculées")
        return df

    def add_onehot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute le one-hot encoding pour variables catégorielles"""
        logger.info("🔢 One-hot encoding variables catégorielles...")

        # Discipline (4 colonnes)
        discipline_dummies = pd.get_dummies(df["discipline"], prefix="discipline", drop_first=True)
        df = pd.concat([df, discipline_dummies], axis=1)

        # Sexe cheval (2 colonnes avec drop_first)
        sexe_dummies = pd.get_dummies(df["sexe_cheval"], prefix="sexe", drop_first=True)
        df = pd.concat([df, sexe_dummies], axis=1)

        # Type piste (3 colonnes avec drop_first)
        piste_dummies = pd.get_dummies(df["type_piste"], prefix="piste", drop_first=True)
        df = pd.concat([df, piste_dummies], axis=1)

        # État piste (5 colonnes avec drop_first)
        etat_dummies = pd.get_dummies(df["etat_piste"], prefix="etat", drop_first=True)
        df = pd.concat([df, etat_dummies], axis=1)

        # Top 20 hippodromes les plus fréquents
        top_hippodromes = df["nom_hippodrome"].value_counts().head(20).index.tolist()
        df.loc[:, "hippodrome_top20"] = df["nom_hippodrome"].apply(
            lambda x: x if x in top_hippodromes else "AUTRE"
        )
        hippodrome_dummies = pd.get_dummies(
            df["hippodrome_top20"], prefix="hippodrome", drop_first=True
        )
        df = pd.concat([df, hippodrome_dummies], axis=1)

        logger.info(
            f"   ✅ {len(discipline_dummies.columns) + len(sexe_dummies.columns) + len(piste_dummies.columns) + len(etat_dummies.columns) + len(hippodrome_dummies.columns)} colonnes one-hot créées"
        )
        return df

    def calculate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features d'interaction entre variables"""
        logger.info("🔗 Calcul features d'interaction...")

        # 1. Forme × Skill jockey (qualité × forme)
        df.loc[:, "interaction_forme_jockey"] = df["forme_5c"] * df["taux_victoires_jockey"]

        # 2. Aptitude distance × Distance normalisée (maîtrise distance)
        df.loc[:, "interaction_aptitude_distance"] = (
            df["aptitude_distance"] * df["distance_norm"].abs()
        )

        # 3. ELO cheval × Niveau concurrent (force relative)
        df.loc[:, "interaction_elo_niveau"] = df["elo_cheval"] * df["niveau_moyen_concurrent"]

        # 4. Cote × Prédiction IA (consensus marché)
        df.loc[:, "interaction_cote_ia"] = df["cote_turfbzh"] * df["prediction_ia_gagnant"]

        # 5. Synergie × Forme (partenariat en forme)
        df.loc[:, "interaction_synergie_forme"] = df["synergie_jockey_cheval"] * df["forme_5c"]

        # 6. Victoires récentes × Skill jockey (hot streak)
        df.loc[:, "interaction_victoires_jockey"] = (
            df["nb_victoires_12m"] * df["taux_victoires_jockey"]
        )

        # 7. Aptitude hippodrome × Popularité (home advantage)
        popularite = df.groupby("nom_hippodrome").size() / len(df)
        df.loc[:, "popularite_hippodrome"] = df["nom_hippodrome"].map(popularite).fillna(0)
        df.loc[:, "interaction_aptitude_popularite"] = (
            df["aptitude_hippodrome"] * df["popularite_hippodrome"]
        )

        # 8. Régularité × Volume courses (consistency)
        df.loc[:, "interaction_regularite_volume"] = df["regularite"] * df["nb_courses_12m"]

        logger.info("   ✅ 8 features d'interaction créées")
        return df

    def prepare_features(self) -> pd.DataFrame:
        """Pipeline complet de préparation des features"""
        logger.info("=" * 80)
        logger.info("🚀 DÉMARRAGE FEATURE ENGINEERING")
        logger.info("=" * 80)
        logger.info("")

        # 1. Extraction
        df = self.extract_base_data()

        # 2. Forme
        df = self.calculate_forme_features(df)

        # 3. Aptitudes
        df = self.calculate_aptitude_features(df)

        # 4. Jockey/Entraineur
        df = self.calculate_jockey_entraineur_features(df)

        # 5. Course
        df = self.calculate_course_features(df)

        # 6. Marché
        df = self.add_market_features(df)

        # 7. Gains
        df = self.calculate_gains_features(df)

        # 8. Météo
        df = self.calculate_meteo_features(df)

        # 9. One-hot encoding
        df = self.add_onehot_encoding(df)

        # 10. Interactions
        df = self.calculate_interaction_features(df)

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ FEATURE ENGINEERING TERMINÉ")
        logger.info("=" * 80)
        logger.info(f"   📊 {len(df):,} performances avec {len(df.columns)} colonnes")
        logger.info("")

        return df

    def close(self):
        """Ferme la connexion"""
        if self.conn:
            self.conn.close()
            logger.info("✅ Connexion fermée")


def main():
    parser = argparse.ArgumentParser(description="Prépare les features ML")
    parser.add_argument("--output", default="data/ml_features.csv", help="Fichier de sortie")

    args = parser.parse_args()

    try:
        engineer = MLFeatureEngineer()
        df = engineer.prepare_features()

        # Sauvegarder
        logger.info(f"💾 Sauvegarde dans {args.output}...")
        df.to_csv(args.output, index=False)
        logger.info(f"✅ Fichier sauvegardé : {args.output}")

        # Résumé
        logger.info("")
        logger.info("📊 RÉSUMÉ DES FEATURES:")
        logger.info("-" * 80)

        feature_groups = {
            "Forme": [
                "forme_5c",
                "forme_10c",
                "nb_courses_12m",
                "nb_victoires_12m",
                "taux_places_12m",
                "regularite",
                "jours_depuis_derniere",
            ],
            "Aptitude": ["aptitude_distance", "aptitude_piste", "aptitude_hippodrome"],
            "Jockey/Entraineur": [
                "taux_victoires_jockey",
                "taux_places_jockey",
                "taux_victoires_entraineur",
                "taux_places_entraineur",
                "synergie_jockey_cheval",
                "synergie_entraineur_cheval",
            ],
            "Course": ["distance_norm", "niveau_moyen_concurrent", "nb_partants"],
            "Marché": [
                "rang_cote_sp",
                "rang_cote_turfbzh",
                "ecart_cote_ia",
                "elo_cheval",
                "prediction_ia_gagnant",
            ],
        }

        for group, features in feature_groups.items():
            available = [f for f in features if f in df.columns]
            logger.info(f"   {group:<20} : {len(available):2} features")

        logger.info("")
        logger.info("🎉 Traitement terminé avec succès !")

        engineer.close()

    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        raise


if __name__ == "__main__":
    main()
