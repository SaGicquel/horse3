#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 BETTING ADVISOR - Système de Recommandation de Paris Intelligent
====================================================================

Ce module analyse chaque course du jour et propose 3 types de paris:
1. PARI SÛR (🟢) - Faible risque, cote < 5, forte probabilité
2. PARI ÉQUILIBRÉ (🟡) - Risque moyen, cote 5-15, bon rapport risque/gain
3. PARI RISQUÉ (🔴) - Haut risque, grosse cote > 15, potentiel élevé

Facteurs analysés:
- Performance historique du cheval
- Forme récente (5 dernières courses)
- Performance jockey/entraineur
- Conditions de course (distance, terrain, hippodrome)
- Tendance des cotes
- Avis des professionnels
- Statistiques par tranche de cote
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ChevalStats:
    """Statistiques détaillées d'un cheval"""

    nom: str
    nb_courses: int = 0
    nb_victoires: int = 0
    nb_places: int = 0  # Top 3
    taux_victoire: float = 0.0
    taux_place: float = 0.0

    # Forme récente (5 dernières courses)
    forme_recente: str = ""
    nb_victoires_recentes: int = 0
    nb_places_recentes: int = 0

    # Performance par conditions
    perf_distance: Dict = field(default_factory=dict)
    perf_hippodrome: Dict = field(default_factory=dict)
    perf_type_course: Dict = field(default_factory=dict)

    # Gains
    gains_total: float = 0
    gains_moyen: float = 0

    # Cote moyenne quand gagne
    cote_moyenne_victoire: float = 0


@dataclass
class JockeyStats:
    """Statistiques du jockey"""

    nom: str
    nb_courses: int = 0
    nb_victoires: int = 0
    taux_victoire: float = 0.0
    taux_place: float = 0.0

    # Performance avec ce cheval
    courses_avec_cheval: int = 0
    victoires_avec_cheval: int = 0


@dataclass
class EntraineurStats:
    """Statistiques de l'entraineur"""

    nom: str
    nb_courses: int = 0
    nb_victoires: int = 0
    taux_victoire: float = 0.0

    # Forme récente
    forme_30j: float = 0.0


@dataclass
class ParticipantAnalyse:
    """Analyse complète d'un participant"""

    nom: str
    numero: int
    cote: float
    cote_reference: float = None

    # Scores par facteur (0-100)
    score_historique: float = 0
    score_forme: float = 0
    score_jockey: float = 0
    score_entraineur: float = 0
    score_conditions: float = 0
    score_cote: float = 0
    score_tendance: float = 0
    score_avis: float = 0

    # Score global pondéré
    score_global: float = 0

    # Probabilité estimée de victoire
    proba_victoire: float = 0
    proba_place: float = 0

    # Value bet
    value_victoire: float = 0
    value_place: float = 0

    # Niveau de confiance
    confiance: str = "FAIBLE"

    # Statistiques détaillées
    stats_cheval: ChevalStats = None
    stats_jockey: JockeyStats = None
    stats_entraineur: EntraineurStats = None

    # Signaux
    signaux_positifs: List[str] = field(default_factory=list)
    signaux_negatifs: List[str] = field(default_factory=list)


@dataclass
class RecommandationPari:
    """Recommandation de pari"""

    type_pari: str  # "SUR", "EQUILIBRE", "RISQUE"
    niveau_risque: int  # 1-5
    participant: ParticipantAnalyse
    race_key: str
    hippodrome: str

    # Mise recommandée
    mise_pct_bankroll: float
    mise_min: float
    mise_max: float

    # Gains potentiels
    gain_potentiel_min: float
    gain_potentiel_max: float

    # Explication
    raison_principale: str
    raisons_secondaires: List[str] = field(default_factory=list)

    # Probabilités
    proba_succes: float = 0
    esperance_gain: float = 0


class BettingAdvisor:
    """Conseiller de paris intelligent"""

    # Poids des facteurs pour le score global
    POIDS = {
        "historique": 0.20,  # Performance globale du cheval
        "forme": 0.25,  # Forme récente (très important)
        "jockey": 0.10,  # Performance du jockey
        "entraineur": 0.10,  # Performance de l'entraineur
        "conditions": 0.10,  # Adéquation aux conditions
        "cote": 0.10,  # Signal des cotes
        "tendance": 0.10,  # Tendance des cotes
        "avis": 0.05,  # Avis professionnel
    }

    def __init__(self, db_connection):
        self.conn = db_connection
        self.cache_chevaux = {}
        self.cache_jockeys = {}
        self.cache_entraineurs = {}
        self.stats_globales = {}
        self._load_global_stats()

    def _load_global_stats(self):
        """Charge les statistiques globales pour les benchmarks"""
        cur = self.conn.cursor()

        # Taux de victoire par tranche de cote
        cur.execute("""
            SELECT
                CASE
                    WHEN cote_finale < 3 THEN '1-3'
                    WHEN cote_finale < 5 THEN '3-5'
                    WHEN cote_finale < 10 THEN '5-10'
                    WHEN cote_finale < 20 THEN '10-20'
                    ELSE '20+'
                END as tranche,
                COUNT(*) as total,
                AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100 as taux_victoire,
                AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) * 100 as taux_place
            FROM cheval_courses_seen
            WHERE cote_finale IS NOT NULL AND race_key LIKE '2025%'
            GROUP BY tranche
        """)

        self.stats_globales["cotes"] = {}
        for row in cur.fetchall():
            self.stats_globales["cotes"][row[0]] = {
                "total": row[1],
                "taux_victoire": row[2] or 0,
                "taux_place": row[3] or 0,
            }

        # Taux moyen global
        cur.execute("""
            SELECT
                AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100,
                AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) * 100
            FROM cheval_courses_seen
            WHERE race_key LIKE '2025%'
        """)
        row = cur.fetchone()
        self.stats_globales["taux_victoire_moyen"] = row[0] or 8.5
        self.stats_globales["taux_place_moyen"] = row[1] or 25.0

        logger.info(
            f"Stats globales chargées: taux victoire moyen = {self.stats_globales['taux_victoire_moyen']:.2f}%"
        )

    def get_cheval_stats(
        self, nom_norm: str, distance: int = None, hippodrome: str = None
    ) -> ChevalStats:
        """Récupère les statistiques complètes d'un cheval"""

        if nom_norm in self.cache_chevaux:
            return self.cache_chevaux[nom_norm]

        cur = self.conn.cursor()
        stats = ChevalStats(nom=nom_norm)

        # Stats globales
        cur.execute(
            """
            SELECT
                COUNT(*) as nb_courses,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as victoires,
                SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) as places,
                SUM(gains_course) as gains_total,
                AVG(CASE WHEN is_win = 1 THEN cote_finale END) as cote_moy_victoire
            FROM cheval_courses_seen
            WHERE nom_norm = %s
        """,
            (nom_norm,),
        )

        row = cur.fetchone()
        if row and row[0]:
            stats.nb_courses = row[0]
            stats.nb_victoires = row[1] or 0
            stats.nb_places = row[2] or 0
            stats.gains_total = row[3] or 0
            stats.cote_moyenne_victoire = row[4] or 0

            if stats.nb_courses > 0:
                stats.taux_victoire = (stats.nb_victoires / stats.nb_courses) * 100
                stats.taux_place = (stats.nb_places / stats.nb_courses) * 100
                stats.gains_moyen = stats.gains_total / stats.nb_courses

        # Forme récente (5 dernières courses)
        cur.execute(
            """
            SELECT place_finale, is_win, musique
            FROM cheval_courses_seen
            WHERE nom_norm = %s
            ORDER BY race_key DESC
            LIMIT 5
        """,
            (nom_norm,),
        )

        forme = []
        for row in cur.fetchall():
            if row[1] == 1:
                forme.append("1")
                stats.nb_victoires_recentes += 1
            elif row[0] and row[0] <= 3:
                forme.append(str(row[0]))
                stats.nb_places_recentes += 1
            elif row[0]:
                forme.append(str(min(row[0], 9)))
            else:
                forme.append("0")

        stats.forme_recente = "".join(forme)

        # Performance par distance (si pertinent)
        if distance:
            cur.execute(
                """
                SELECT
                    COUNT(*),
                    AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100,
                    AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) * 100
                FROM cheval_courses_seen
                WHERE nom_norm = %s
                AND ABS(distance_m - %s) < 200
            """,
                (nom_norm, distance),
            )

            row = cur.fetchone()
            if row and row[0] >= 2:
                stats.perf_distance = {
                    "nb_courses": row[0],
                    "taux_victoire": row[1] or 0,
                    "taux_place": row[2] or 0,
                }

        # Performance sur l'hippodrome (si pertinent)
        if hippodrome:
            cur.execute(
                """
                SELECT
                    COUNT(*),
                    AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100,
                    AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) * 100
                FROM cheval_courses_seen
                WHERE nom_norm = %s
                AND hippodrome_nom = %s
            """,
                (nom_norm, hippodrome),
            )

            row = cur.fetchone()
            if row and row[0] >= 2:
                stats.perf_hippodrome = {
                    "nb_courses": row[0],
                    "taux_victoire": row[1] or 0,
                    "taux_place": row[2] or 0,
                }

        self.cache_chevaux[nom_norm] = stats
        return stats

    def get_jockey_stats(self, jockey_nom: str, cheval_nom: str = None) -> JockeyStats:
        """Récupère les statistiques du jockey"""

        if not jockey_nom:
            return JockeyStats(nom="Inconnu")

        cache_key = f"{jockey_nom}_{cheval_nom or ''}"
        if cache_key in self.cache_jockeys:
            return self.cache_jockeys[cache_key]

        cur = self.conn.cursor()
        stats = JockeyStats(nom=jockey_nom)

        # Stats globales du jockey (30 derniers jours pour la forme)
        cur.execute(
            """
            SELECT
                COUNT(*),
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END),
                AVG(CASE WHEN place_finale <= 3 THEN 1.0 ELSE 0.0 END) * 100
            FROM cheval_courses_seen
            WHERE driver_jockey = %s
            AND race_key >= %s
        """,
            (jockey_nom, (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")),
        )

        row = cur.fetchone()
        if row and row[0]:
            stats.nb_courses = row[0]
            stats.nb_victoires = row[1] or 0
            stats.taux_place = row[2] or 0
            if stats.nb_courses > 0:
                stats.taux_victoire = (stats.nb_victoires / stats.nb_courses) * 100

        # Stats avec ce cheval spécifique
        if cheval_nom:
            cur.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END)
                FROM cheval_courses_seen
                WHERE driver_jockey = %s AND nom_norm = %s
            """,
                (jockey_nom, cheval_nom),
            )

            row = cur.fetchone()
            if row:
                stats.courses_avec_cheval = row[0] or 0
                stats.victoires_avec_cheval = row[1] or 0

        self.cache_jockeys[cache_key] = stats
        return stats

    def get_entraineur_stats(self, entraineur_nom: str) -> EntraineurStats:
        """Récupère les statistiques de l'entraineur"""

        if not entraineur_nom:
            return EntraineurStats(nom="Inconnu")

        if entraineur_nom in self.cache_entraineurs:
            return self.cache_entraineurs[entraineur_nom]

        cur = self.conn.cursor()
        stats = EntraineurStats(nom=entraineur_nom)

        # Stats sur 90 jours
        cur.execute(
            """
            SELECT
                COUNT(*),
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END)
            FROM cheval_courses_seen
            WHERE entraineur = %s
            AND race_key >= %s
        """,
            (entraineur_nom, (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")),
        )

        row = cur.fetchone()
        if row and row[0]:
            stats.nb_courses = row[0]
            stats.nb_victoires = row[1] or 0
            if stats.nb_courses > 0:
                stats.taux_victoire = (stats.nb_victoires / stats.nb_courses) * 100

        # Forme 30 derniers jours
        cur.execute(
            """
            SELECT AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100
            FROM cheval_courses_seen
            WHERE entraineur = %s
            AND race_key >= %s
        """,
            (entraineur_nom, (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")),
        )

        row = cur.fetchone()
        stats.forme_30j = row[0] or 0 if row else 0

        self.cache_entraineurs[entraineur_nom] = stats
        return stats

    def calculer_score_historique(self, stats: ChevalStats) -> Tuple[float, List[str], List[str]]:
        """Calcule le score basé sur l'historique du cheval"""
        score = 50  # Base
        signaux_pos = []
        signaux_neg = []

        if stats.nb_courses == 0:
            signaux_neg.append("🆕 Première course ou données manquantes")
            return 30, signaux_pos, signaux_neg

        # Bonus expérience
        if stats.nb_courses >= 20:
            score += 5
            signaux_pos.append(f"📊 Expérimenté ({stats.nb_courses} courses)")

        # Taux de victoire vs moyenne
        taux_moyen = self.stats_globales.get("taux_victoire_moyen", 8.5)

        if stats.taux_victoire > taux_moyen * 2:
            score += 25
            signaux_pos.append(f"🏆 Taux victoire excellent ({stats.taux_victoire:.1f}%)")
        elif stats.taux_victoire > taux_moyen * 1.5:
            score += 15
            signaux_pos.append(f"✅ Bon taux victoire ({stats.taux_victoire:.1f}%)")
        elif stats.taux_victoire > taux_moyen:
            score += 5
        elif stats.taux_victoire < taux_moyen * 0.5:
            score -= 10
            signaux_neg.append(f"❌ Faible taux victoire ({stats.taux_victoire:.1f}%)")

        # Taux de place
        if stats.taux_place > 40:
            score += 10
            signaux_pos.append(f"📍 Régulièrement placé ({stats.taux_place:.1f}%)")
        elif stats.taux_place < 15:
            score -= 5

        return max(0, min(100, score)), signaux_pos, signaux_neg

    def calculer_score_forme(self, stats: ChevalStats) -> Tuple[float, List[str], List[str]]:
        """Calcule le score basé sur la forme récente"""
        score = 50
        signaux_pos = []
        signaux_neg = []

        if not stats.forme_recente:
            return 50, [], ["⚠️ Pas de données récentes"]

        # Analyser la musique
        forme = stats.forme_recente

        # Victoires récentes
        if stats.nb_victoires_recentes >= 2:
            score += 30
            signaux_pos.append(
                f"🔥 {stats.nb_victoires_recentes} victoires sur 5 dernières courses!"
            )
        elif stats.nb_victoires_recentes == 1:
            score += 15
            signaux_pos.append("✅ Victoire récente")

        # Places
        if stats.nb_places_recentes >= 4:
            score += 20
            signaux_pos.append("📈 Très régulier (4+ places/5)")
        elif stats.nb_places_recentes >= 3:
            score += 10

        # Tendance (amélioration ou dégradation)
        if len(forme) >= 3:
            recent = [int(x) if x.isdigit() else 9 for x in forme[:3]]
            if recent[0] < recent[1] < recent[2]:
                score += 10
                signaux_pos.append("📈 En progression")
            elif recent[0] > recent[1] > recent[2]:
                score -= 10
                signaux_neg.append("📉 En régression")

        # Mauvaise série
        bad_results = sum(1 for x in forme if x in ["0", "7", "8", "9"])
        if bad_results >= 4:
            score -= 20
            signaux_neg.append("⚠️ Mauvaise série récente")

        return max(0, min(100, score)), signaux_pos, signaux_neg

    def calculer_score_jockey(
        self, stats: JockeyStats, cheval_nom: str
    ) -> Tuple[float, List[str], List[str]]:
        """Calcule le score du jockey"""
        score = 50
        signaux_pos = []
        signaux_neg = []

        if not stats or stats.nb_courses == 0:
            return 50, [], []

        # Performance globale
        if stats.taux_victoire > 15:
            score += 20
            signaux_pos.append(f"🏇 Jockey top ({stats.taux_victoire:.1f}% victoires)")
        elif stats.taux_victoire > 10:
            score += 10
            signaux_pos.append(f"✅ Bon jockey ({stats.taux_victoire:.1f}%)")
        elif stats.taux_victoire < 5:
            score -= 10
            signaux_neg.append("⚠️ Jockey peu performant")

        # Combo avec le cheval
        if stats.courses_avec_cheval >= 3 and stats.victoires_avec_cheval >= 1:
            score += 15
            signaux_pos.append(
                f"🤝 Bonne association ({stats.victoires_avec_cheval}/{stats.courses_avec_cheval})"
            )

        return max(0, min(100, score)), signaux_pos, signaux_neg

    def calculer_score_entraineur(
        self, stats: EntraineurStats
    ) -> Tuple[float, List[str], List[str]]:
        """Calcule le score de l'entraineur"""
        score = 50
        signaux_pos = []
        signaux_neg = []

        if not stats or stats.nb_courses == 0:
            return 50, [], []

        # Performance globale
        if stats.taux_victoire > 15:
            score += 15
            signaux_pos.append(f"🎯 Ecurie performante ({stats.taux_victoire:.1f}%)")
        elif stats.taux_victoire > 10:
            score += 5

        # Forme récente
        if stats.forme_30j > 15:
            score += 15
            signaux_pos.append(f"🔥 Ecurie en forme ({stats.forme_30j:.1f}% / 30j)")
        elif stats.forme_30j < 5:
            score -= 10
            signaux_neg.append("⚠️ Ecurie en difficulté")

        return max(0, min(100, score)), signaux_pos, signaux_neg

    def calculer_score_conditions(
        self, stats: ChevalStats, distance: int, hippodrome: str
    ) -> Tuple[float, List[str], List[str]]:
        """Calcule le score d'adéquation aux conditions"""
        score = 50
        signaux_pos = []
        signaux_neg = []

        # Performance sur la distance
        if stats.perf_distance:
            if stats.perf_distance["taux_victoire"] > stats.taux_victoire * 1.5:
                score += 15
                signaux_pos.append(
                    f"📏 Excellent sur cette distance ({stats.perf_distance['taux_victoire']:.1f}%)"
                )
            elif stats.perf_distance["taux_victoire"] < stats.taux_victoire * 0.5:
                score -= 10
                signaux_neg.append("⚠️ Distance peu adaptée")

        # Performance sur l'hippodrome
        if stats.perf_hippodrome:
            if stats.perf_hippodrome["taux_victoire"] > stats.taux_victoire * 1.5:
                score += 15
                signaux_pos.append(
                    f"🏟️ Bon sur cet hippodrome ({stats.perf_hippodrome['taux_victoire']:.1f}%)"
                )
            elif (
                stats.perf_hippodrome["nb_courses"] >= 3
                and stats.perf_hippodrome["taux_victoire"] == 0
            ):
                score -= 10
                signaux_neg.append("⚠️ Jamais gagné ici")

        return max(0, min(100, score)), signaux_pos, signaux_neg

    def calculer_score_cote(self, cote: float) -> Tuple[float, List[str], List[str]]:
        """Calcule le score basé sur la cote"""
        score = 50
        signaux_pos = []
        signaux_neg = []

        if not cote or cote <= 0:
            return 50, [], ["⚠️ Cote non disponible"]

        # Favoris
        if cote < 2:
            score += 25
            signaux_pos.append("⭐ Grand favori")
        elif cote < 3:
            score += 20
            signaux_pos.append("⭐ Favori")
        elif cote < 5:
            score += 10
            signaux_pos.append("👍 Cote intéressante")
        elif cote > 30:
            score -= 15
            signaux_neg.append("⚠️ Outsider (cote élevée)")
        elif cote > 50:
            score -= 25
            signaux_neg.append("❌ Très gros outsider")

        return max(0, min(100, score)), signaux_pos, signaux_neg

    def calculer_score_tendance(
        self, tendance: str, amplitude: float, est_favori: bool
    ) -> Tuple[float, List[str], List[str]]:
        """Calcule le score basé sur la tendance des cotes"""
        score = 50
        signaux_pos = []
        signaux_neg = []

        if tendance == "-":  # Cote baisse = argent entre
            if amplitude and amplitude > 20:
                score += 20
                signaux_pos.append(f"💰 Forte baisse de cote (-{amplitude:.0f}%)")
            elif amplitude and amplitude > 10:
                score += 10
                signaux_pos.append("💰 Baisse de cote")
            else:
                score += 5
        elif tendance == "+":  # Cote monte = argent sort
            if amplitude and amplitude > 20:
                score -= 15
                signaux_neg.append(f"📉 Forte hausse de cote (+{amplitude:.0f}%)")
            elif amplitude and amplitude > 10:
                score -= 10
                signaux_neg.append("📉 Hausse de cote")

        if est_favori:
            score += 10
            signaux_pos.append("⭐ Cheval favori")

        return max(0, min(100, score)), signaux_pos, signaux_neg

    def calculer_score_avis(self, avis: str) -> Tuple[float, List[str], List[str]]:
        """Calcule le score basé sur l'avis professionnel"""
        score = 50
        signaux_pos = []
        signaux_neg = []

        if avis == "POSITIF":
            score += 15
            signaux_pos.append("👍 Avis entraineur positif")
        elif avis == "NEGATIF":
            score -= 15
            signaux_neg.append("👎 Avis entraineur négatif")

        return max(0, min(100, score)), signaux_pos, signaux_neg

    def analyser_participant(
        self, data: dict, distance: int = None, hippodrome: str = None
    ) -> ParticipantAnalyse:
        """Analyse complète d'un participant"""

        nom = data.get("nom_norm") or data.get("nom", "Inconnu")

        analyse = ParticipantAnalyse(
            nom=nom,
            numero=data.get("numero_dossard") or data.get("numero", 0),
            cote=data.get("cote_finale") or data.get("cote", 0),
            cote_reference=data.get("cote_reference"),
        )

        # Récupérer les stats
        analyse.stats_cheval = self.get_cheval_stats(nom, distance, hippodrome)
        analyse.stats_jockey = self.get_jockey_stats(data.get("driver_jockey"), nom)
        analyse.stats_entraineur = self.get_entraineur_stats(data.get("entraineur"))

        # Calculer chaque score
        scores = {}
        all_signaux_pos = []
        all_signaux_neg = []

        # Historique
        score, pos, neg = self.calculer_score_historique(analyse.stats_cheval)
        scores["historique"] = score
        all_signaux_pos.extend(pos)
        all_signaux_neg.extend(neg)
        analyse.score_historique = score

        # Forme
        score, pos, neg = self.calculer_score_forme(analyse.stats_cheval)
        scores["forme"] = score
        all_signaux_pos.extend(pos)
        all_signaux_neg.extend(neg)
        analyse.score_forme = score

        # Jockey
        score, pos, neg = self.calculer_score_jockey(analyse.stats_jockey, nom)
        scores["jockey"] = score
        all_signaux_pos.extend(pos)
        all_signaux_neg.extend(neg)
        analyse.score_jockey = score

        # Entraineur
        score, pos, neg = self.calculer_score_entraineur(analyse.stats_entraineur)
        scores["entraineur"] = score
        all_signaux_pos.extend(pos)
        all_signaux_neg.extend(neg)
        analyse.score_entraineur = score

        # Conditions
        score, pos, neg = self.calculer_score_conditions(analyse.stats_cheval, distance, hippodrome)
        scores["conditions"] = score
        all_signaux_pos.extend(pos)
        all_signaux_neg.extend(neg)
        analyse.score_conditions = score

        # Cote
        score, pos, neg = self.calculer_score_cote(analyse.cote)
        scores["cote"] = score
        all_signaux_pos.extend(pos)
        all_signaux_neg.extend(neg)
        analyse.score_cote = score

        # Tendance
        score, pos, neg = self.calculer_score_tendance(
            data.get("tendance_cote"), data.get("amplitude_tendance"), data.get("est_favori")
        )
        scores["tendance"] = score
        all_signaux_pos.extend(pos)
        all_signaux_neg.extend(neg)
        analyse.score_tendance = score

        # Avis
        score, pos, neg = self.calculer_score_avis(data.get("avis_entraineur"))
        scores["avis"] = score
        all_signaux_pos.extend(pos)
        all_signaux_neg.extend(neg)
        analyse.score_avis = score

        # Score global pondéré
        analyse.score_global = sum(scores[k] * self.POIDS[k] for k in scores)

        # Calculer probabilités
        analyse.proba_victoire = self._score_to_proba(analyse.score_global, analyse.cote)
        analyse.proba_place = min(95, analyse.proba_victoire * 2.5)

        # Value bet
        if analyse.cote and analyse.cote > 0:
            proba_implicite = 100 / analyse.cote
            analyse.value_victoire = analyse.proba_victoire - proba_implicite

        # Confiance
        nb_signaux_pos = len(all_signaux_pos)
        nb_signaux_neg = len(all_signaux_neg)

        if analyse.score_global >= 70 and nb_signaux_pos >= 4:
            analyse.confiance = "HAUTE"
        elif analyse.score_global >= 55 and nb_signaux_pos >= 2:
            analyse.confiance = "MOYENNE"
        else:
            analyse.confiance = "FAIBLE"

        analyse.signaux_positifs = all_signaux_pos
        analyse.signaux_negatifs = all_signaux_neg

        return analyse

    def _score_to_proba(self, score: float, cote: float) -> float:
        """Convertit un score en probabilité"""
        # Base: probabilité implicite de la cote
        if cote and cote > 0:
            proba_base = 100 / cote
        else:
            proba_base = 10

        # Ajustement selon le score
        # Score 50 = pas d'ajustement
        # Score 100 = +50% de la proba
        # Score 0 = -50% de la proba
        adjustment = (score - 50) / 100
        proba = proba_base * (1 + adjustment)

        return max(1, min(95, proba))

    def generer_recommandations(self, race_key: str) -> List[RecommandationPari]:
        """Génère les recommandations de paris pour une course"""

        cur = self.conn.cursor()

        # Récupérer les infos de la course
        cur.execute(
            """
            SELECT DISTINCT
                hippodrome_nom,
                distance_m,
                type_course
            FROM cheval_courses_seen
            WHERE race_key = %s
            LIMIT 1
        """,
            (race_key,),
        )

        course_info = cur.fetchone()
        if not course_info:
            return []

        hippodrome, distance, type_course = course_info

        # Récupérer tous les participants
        cur.execute(
            """
            SELECT
                nom_norm,
                numero_dossard,
                cote_finale,
                cote_reference,
                tendance_cote,
                amplitude_tendance,
                est_favori,
                avis_entraineur,
                driver_jockey,
                entraineur,
                age,
                sexe,
                musique
            FROM cheval_courses_seen
            WHERE race_key = %s
            ORDER BY numero_dossard
        """,
            (race_key,),
        )

        participants = []
        for row in cur.fetchall():
            data = {
                "nom_norm": row[0],
                "numero_dossard": row[1],
                "cote_finale": row[2],
                "cote_reference": row[3],
                "tendance_cote": row[4],
                "amplitude_tendance": row[5],
                "est_favori": row[6],
                "avis_entraineur": row[7],
                "driver_jockey": row[8],
                "entraineur": row[9],
                "age": row[10],
                "sexe": row[11],
                "musique": row[12],
            }

            analyse = self.analyser_participant(data, distance, hippodrome)
            participants.append(analyse)

        # Trier par score
        participants.sort(key=lambda x: x.score_global, reverse=True)

        recommandations = []

        # PARI SÛR (🟢) - Le meilleur candidat avec cote < 5
        for p in participants:
            if (
                p.cote
                and p.cote < 5
                and p.score_global >= 60
                and p.confiance in ["HAUTE", "MOYENNE"]
            ):
                recommandations.append(
                    RecommandationPari(
                        type_pari="SUR",
                        niveau_risque=1,
                        participant=p,
                        race_key=race_key,
                        hippodrome=hippodrome,
                        mise_pct_bankroll=3.0,
                        mise_min=5,
                        mise_max=20,
                        gain_potentiel_min=5 * p.cote,
                        gain_potentiel_max=20 * p.cote,
                        raison_principale=f"Favori solide avec {len(p.signaux_positifs)} signaux positifs",
                        raisons_secondaires=p.signaux_positifs[:3],
                        proba_succes=p.proba_victoire,
                        esperance_gain=(p.proba_victoire / 100 * p.cote - 1) * 100,
                    )
                )
                break

        # PARI ÉQUILIBRÉ (🟡) - Bon rapport risque/gain, cote 5-15
        for p in participants:
            if p.cote and 5 <= p.cote <= 15 and p.score_global >= 55 and p.value_victoire > 0:
                recommandations.append(
                    RecommandationPari(
                        type_pari="EQUILIBRE",
                        niveau_risque=3,
                        participant=p,
                        race_key=race_key,
                        hippodrome=hippodrome,
                        mise_pct_bankroll=2.0,
                        mise_min=3,
                        mise_max=10,
                        gain_potentiel_min=3 * p.cote,
                        gain_potentiel_max=10 * p.cote,
                        raison_principale=f"Value bet avec +{p.value_victoire:.1f}% d'edge",
                        raisons_secondaires=p.signaux_positifs[:3],
                        proba_succes=p.proba_victoire,
                        esperance_gain=(p.proba_victoire / 100 * p.cote - 1) * 100,
                    )
                )
                break

        # PARI RISQUÉ (🔴) - Grosse cote, fort potentiel
        for p in participants:
            if p.cote and p.cote > 15 and p.score_global >= 45 and len(p.signaux_positifs) >= 2:
                recommandations.append(
                    RecommandationPari(
                        type_pari="RISQUE",
                        niveau_risque=5,
                        participant=p,
                        race_key=race_key,
                        hippodrome=hippodrome,
                        mise_pct_bankroll=1.0,
                        mise_min=2,
                        mise_max=5,
                        gain_potentiel_min=2 * p.cote,
                        gain_potentiel_max=5 * p.cote,
                        raison_principale=f"Outsider intéressant à {p.cote:.1f}",
                        raisons_secondaires=p.signaux_positifs[:3],
                        proba_succes=p.proba_victoire,
                        esperance_gain=(p.proba_victoire / 100 * p.cote - 1) * 100,
                    )
                )
                break

        return recommandations

    def generer_rapport_course(self, race_key: str) -> dict:
        """Génère un rapport complet d'analyse pour une course"""

        cur = self.conn.cursor()

        # Infos course
        cur.execute(
            """
            SELECT DISTINCT
                hippodrome_nom,
                distance_m,
                type_course,
                discipline,
                allocation_totale
            FROM cheval_courses_seen
            WHERE race_key = %s
            LIMIT 1
        """,
            (race_key,),
        )

        course_info = cur.fetchone()
        if not course_info:
            return {"error": "Course non trouvée"}

        # Participants analysés
        cur.execute(
            """
            SELECT
                nom_norm, numero_dossard, cote_finale, cote_reference,
                tendance_cote, amplitude_tendance, est_favori, avis_entraineur,
                driver_jockey, entraineur
            FROM cheval_courses_seen
            WHERE race_key = %s
            ORDER BY numero_dossard
        """,
            (race_key,),
        )

        analyses = []
        for row in cur.fetchall():
            data = {
                "nom_norm": row[0],
                "numero_dossard": row[1],
                "cote_finale": row[2],
                "cote_reference": row[3],
                "tendance_cote": row[4],
                "amplitude_tendance": row[5],
                "est_favori": row[6],
                "avis_entraineur": row[7],
                "driver_jockey": row[8],
                "entraineur": row[9],
            }

            analyse = self.analyser_participant(data, course_info[1], course_info[0])
            analyses.append(
                {
                    "nom": analyse.nom,
                    "numero": analyse.numero,
                    "cote": analyse.cote,
                    "score_global": round(analyse.score_global, 1),
                    "score_forme": round(analyse.score_forme, 1),
                    "score_historique": round(analyse.score_historique, 1),
                    "score_jockey": round(analyse.score_jockey, 1),
                    "score_entraineur": round(analyse.score_entraineur, 1),
                    "proba_victoire": round(analyse.proba_victoire, 1),
                    "proba_place": round(analyse.proba_place, 1),
                    "value_victoire": round(analyse.value_victoire, 1),
                    "confiance": analyse.confiance,
                    "signaux_positifs": analyse.signaux_positifs,
                    "signaux_negatifs": analyse.signaux_negatifs,
                    "stats": {
                        "nb_courses": analyse.stats_cheval.nb_courses,
                        "taux_victoire": round(analyse.stats_cheval.taux_victoire, 1),
                        "forme_recente": analyse.stats_cheval.forme_recente,
                    },
                }
            )

        # Trier par score
        analyses.sort(key=lambda x: x["score_global"], reverse=True)

        # Recommandations
        recommandations = self.generer_recommandations(race_key)

        return {
            "race_key": race_key,
            "hippodrome": course_info[0],
            "distance": course_info[1],
            "type_course": course_info[2],
            "discipline": course_info[3],
            "allocation": course_info[4],
            "nb_partants": len(analyses),
            "analyses": analyses,
            "recommandations": [
                {
                    "type": r.type_pari,
                    "niveau_risque": r.niveau_risque,
                    "cheval": r.participant.nom,
                    "numero": r.participant.numero,
                    "cote": r.participant.cote,
                    "mise_recommandee": f"{r.mise_pct_bankroll}% ({r.mise_min}-{r.mise_max}€)",
                    "gain_potentiel": f"{r.gain_potentiel_min:.0f}-{r.gain_potentiel_max:.0f}€",
                    "raison": r.raison_principale,
                    "details": r.raisons_secondaires,
                    "proba_succes": round(r.proba_succes, 1),
                    "esperance": round(r.esperance_gain, 1),
                }
                for r in recommandations
            ],
            "top_3": [
                {
                    "rang": i + 1,
                    "nom": a["nom"],
                    "score": a["score_global"],
                    "cote": a["cote"],
                    "confiance": a["confiance"],
                }
                for i, a in enumerate(analyses[:3])
            ],
        }


def main():
    """Test du module"""
    from db_connection import get_connection

    conn = get_connection()
    advisor = BettingAdvisor(conn)

    # Récupérer une course du jour
    cur = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")

    cur.execute(
        """
        SELECT DISTINCT race_key
        FROM cheval_courses_seen
        WHERE race_key LIKE %s
        LIMIT 3
    """,
        (today + "%",),
    )

    courses = cur.fetchall()

    for (race_key,) in courses:
        print(f"\n{'='*80}")
        print(f"ANALYSE: {race_key}")
        print(f"{'='*80}")

        rapport = advisor.generer_rapport_course(race_key)

        print(f"\n🏟️ {rapport['hippodrome']} - {rapport['distance']}m ({rapport['type_course']})")
        print(f"📊 {rapport['nb_partants']} partants")

        print("\n--- TOP 3 ---")
        for t in rapport["top_3"]:
            print(
                f"  {t['rang']}. {t['nom']} | Score: {t['score']:.1f} | Cote: {t['cote']} | {t['confiance']}"
            )

        print("\n--- RECOMMANDATIONS ---")
        for r in rapport["recommandations"]:
            emoji = "🟢" if r["type"] == "SUR" else ("🟡" if r["type"] == "EQUILIBRE" else "🔴")
            print(f"\n{emoji} PARI {r['type']}")
            print(f"   🐴 {r['cheval']} (n°{r['numero']}) @ {r['cote']}")
            print(f"   💰 Mise: {r['mise_recommandee']}")
            print(f"   🎯 Gain potentiel: {r['gain_potentiel']}")
            print(f"   📊 Proba: {r['proba_succes']}% | Espérance: {r['esperance']}%")
            print(f"   ℹ️ {r['raison']}")

    conn.close()


if __name__ == "__main__":
    main()
