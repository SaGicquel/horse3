#!/usr/bin/env python3
"""
Script de calcul des statistiques agrégées pour chevaux et personnes.
À lancer régulièrement (quotidien ou hebdomadaire) pour maintenir les stats à jour.

Usage:
    python calcul_stats.py [--chevaux] [--personnes] [--all]
    python calcul_stats.py --cheval-id 123
"""

import argparse
import sys
from datetime import date, timedelta
from db_connection import get_connection


class StatsCalculator:
    """Calcule les statistiques agrégées."""

    def __init__(self):
        self.conn = None
        self.cur = None
        self.stats = {
            "chevaux_updated": 0,
            "personnes_updated": 0,
        }

    def connect(self):
        """Connexion BDD."""
        self.conn = get_connection()
        self.cur = self.conn.cursor()

    def close(self):
        """Fermeture BDD."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def calcul_stats_chevaux(self, id_cheval: int = None):
        """
        Calcule les stats pour tous les chevaux ou un cheval spécifique.

        Args:
            id_cheval: ID du cheval (None = tous)
        """
        print("\n🐴 Calcul des statistiques chevaux...")

        where_clause = ""
        params = []

        if id_cheval:
            where_clause = "WHERE ch.id_cheval = %s"
            params = [id_cheval]

        # Stats globales carrière
        query = f"""
            INSERT INTO stats_chevaux (
                id_cheval, date_calcul,
                nb_courses_total, nb_victoires, nb_places,
                tx_victoire, tx_place, gain_total,
                forme_5c, nb_courses_5c, nb_victoires_5c
            )
            SELECT
                ch.id_cheval,
                CURRENT_DATE,
                COUNT(*) as nb_courses,
                SUM(CASE WHEN p.position_arrivee = 1 AND p.non_partant = FALSE THEN 1 ELSE 0 END) as victoires,
                SUM(CASE WHEN p.place = TRUE AND p.non_partant = FALSE THEN 1 ELSE 0 END) as places,
                calcul_tx_victoire(
                    SUM(CASE WHEN p.position_arrivee = 1 AND p.non_partant = FALSE THEN 1 ELSE 0 END)::INTEGER,
                    COUNT(CASE WHEN p.non_partant = FALSE THEN 1 END)::INTEGER
                ),
                calcul_tx_victoire(
                    SUM(CASE WHEN p.place = TRUE AND p.non_partant = FALSE THEN 1 ELSE 0 END)::INTEGER,
                    COUNT(CASE WHEN p.non_partant = FALSE THEN 1 END)::INTEGER
                ),
                SUM(COALESCE(p.gain_course, 0)),
                -- Forme 5 dernières courses (moyenne position)
                (
                    SELECT AVG(subq.position_arrivee)
                    FROM (
                        SELECT p2.position_arrivee
                        FROM performances p2
                        JOIN courses c2 ON p2.id_course = c2.id_course
                        WHERE p2.id_cheval = ch.id_cheval
                        AND p2.non_partant = FALSE
                        AND p2.position_arrivee IS NOT NULL
                        ORDER BY c2.date_course DESC
                        LIMIT 5
                    ) subq
                ),
                -- Nb courses 5 dernières
                (
                    SELECT COUNT(*)
                    FROM (
                        SELECT 1
                        FROM performances p2
                        JOIN courses c2 ON p2.id_course = c2.id_course
                        WHERE p2.id_cheval = ch.id_cheval
                        AND p2.non_partant = FALSE
                        ORDER BY c2.date_course DESC
                        LIMIT 5
                    ) subq
                ),
                -- Nb victoires 5 dernières
                (
                    SELECT COUNT(*)
                    FROM (
                        SELECT 1
                        FROM performances p2
                        JOIN courses c2 ON p2.id_course = c2.id_course
                        WHERE p2.id_cheval = ch.id_cheval
                        AND p2.position_arrivee = 1
                        AND p2.non_partant = FALSE
                        ORDER BY c2.date_course DESC
                        LIMIT 5
                    ) subq
                )
            FROM chevaux ch
            LEFT JOIN performances p ON ch.id_cheval = p.id_cheval
            {where_clause}
            GROUP BY ch.id_cheval
            ON CONFLICT (id_cheval, date_calcul) DO UPDATE SET
                nb_courses_total = EXCLUDED.nb_courses_total,
                nb_victoires = EXCLUDED.nb_victoires,
                nb_places = EXCLUDED.nb_places,
                tx_victoire = EXCLUDED.tx_victoire,
                tx_place = EXCLUDED.tx_place,
                gain_total = EXCLUDED.gain_total,
                forme_5c = EXCLUDED.forme_5c,
                nb_courses_5c = EXCLUDED.nb_courses_5c,
                nb_victoires_5c = EXCLUDED.nb_victoires_5c
        """

        self.cur.execute(query, params)
        self.stats["chevaux_updated"] = self.cur.rowcount
        self.conn.commit()

        print(f"✅ {self.stats['chevaux_updated']} chevaux mis à jour")

    def calcul_aptitudes_chevaux(self, id_cheval: int = None):
        """
        Calcule les aptitudes (distance, piste, hippodrome) pour les chevaux.

        Args:
            id_cheval: ID du cheval (None = tous)
        """
        print("\n📊 Calcul des aptitudes chevaux...")

        # Pour chaque cheval
        where_clause = ""
        params = []

        if id_cheval:
            where_clause = "WHERE id_cheval = %s"
            params = [id_cheval]

        self.cur.execute(f"SELECT id_cheval FROM chevaux {where_clause}", params)
        chevaux = self.cur.fetchall()

        for (cheval_id,) in chevaux:
            # Aptitude distance (% places sur distance ±10%)
            self.cur.execute(
                """
                WITH perf_cheval AS (
                    SELECT
                        p.id_performance,
                        c.distance,
                        p.place
                    FROM performances p
                    JOIN courses c ON p.id_course = c.id_course
                    WHERE p.id_cheval = %s
                    AND p.non_partant = FALSE
                )
                SELECT
                    AVG(CASE WHEN pc2.place THEN 100.0 ELSE 0.0 END) as aptitude_distance
                FROM perf_cheval pc1
                CROSS JOIN perf_cheval pc2
                WHERE ABS(pc2.distance - pc1.distance) <= pc1.distance * 0.1
                LIMIT 1
            """,
                (cheval_id,),
            )

            aptitude_distance = self.cur.fetchone()
            if aptitude_distance and aptitude_distance[0]:
                self.cur.execute(
                    """
                    UPDATE stats_chevaux
                    SET aptitude_distance = %s
                    WHERE id_cheval = %s AND date_calcul = CURRENT_DATE
                """,
                    (round(aptitude_distance[0], 2), cheval_id),
                )

        self.conn.commit()
        print(f"✅ Aptitudes calculées pour {len(chevaux)} chevaux")

    def calcul_stats_personnes(self, periode: str = "12M"):
        """
        Calcule les stats pour jockeys et entraîneurs.

        Args:
            periode: '12M', '3M', '1M'
        """
        print(f"\n👤 Calcul des statistiques personnes (période: {periode})...")

        # Calculer la date limite selon la période
        if periode == "12M":
            date_limite = date.today() - timedelta(days=365)
        elif periode == "3M":
            date_limite = date.today() - timedelta(days=90)
        elif periode == "1M":
            date_limite = date.today() - timedelta(days=30)
        else:
            date_limite = date.today() - timedelta(days=365)

        # Stats jockeys
        self.cur.execute(
            """
            INSERT INTO stats_personnes (
                id_personne, date_calcul, periode,
                nb_courses, nb_victoires, nb_places,
                tx_victoire, tx_place,
                nb_courses_plat, tx_victoire_plat,
                nb_courses_obstacle, tx_victoire_obstacle,
                nb_courses_trot, tx_victoire_trot
            )
            SELECT
                per.id_personne,
                CURRENT_DATE,
                %s,
                COUNT(*) as nb_courses,
                SUM(CASE WHEN p.position_arrivee = 1 AND p.non_partant = FALSE THEN 1 ELSE 0 END) as victoires,
                SUM(CASE WHEN p.place = TRUE AND p.non_partant = FALSE THEN 1 ELSE 0 END) as places,
                calcul_tx_victoire(
                    SUM(CASE WHEN p.position_arrivee = 1 AND p.non_partant = FALSE THEN 1 ELSE 0 END)::INTEGER,
                    COUNT(CASE WHEN p.non_partant = FALSE THEN 1 END)::INTEGER
                ),
                calcul_tx_victoire(
                    SUM(CASE WHEN p.place = TRUE AND p.non_partant = FALSE THEN 1 ELSE 0 END)::INTEGER,
                    COUNT(CASE WHEN p.non_partant = FALSE THEN 1 END)::INTEGER
                ),
                -- Plat
                COUNT(CASE WHEN c.discipline = 'Plat' THEN 1 END),
                calcul_tx_victoire(
                    SUM(CASE WHEN c.discipline = 'Plat' AND p.position_arrivee = 1 AND p.non_partant = FALSE THEN 1 ELSE 0 END)::INTEGER,
                    COUNT(CASE WHEN c.discipline = 'Plat' AND p.non_partant = FALSE THEN 1 END)::INTEGER
                ),
                -- Obstacle
                COUNT(CASE WHEN c.discipline = 'Obstacle' THEN 1 END),
                calcul_tx_victoire(
                    SUM(CASE WHEN c.discipline = 'Obstacle' AND p.position_arrivee = 1 AND p.non_partant = FALSE THEN 1 ELSE 0 END)::INTEGER,
                    COUNT(CASE WHEN c.discipline = 'Obstacle' AND p.non_partant = FALSE THEN 1 END)::INTEGER
                ),
                -- Trot
                COUNT(CASE WHEN c.discipline = 'Trot' THEN 1 END),
                calcul_tx_victoire(
                    SUM(CASE WHEN c.discipline = 'Trot' AND p.position_arrivee = 1 AND p.non_partant = FALSE THEN 1 ELSE 0 END)::INTEGER,
                    COUNT(CASE WHEN c.discipline = 'Trot' AND p.non_partant = FALSE THEN 1 END)::INTEGER
                )
            FROM personnes per
            LEFT JOIN performances p ON per.id_personne = p.id_jockey
            LEFT JOIN courses c ON p.id_course = c.id_course
            WHERE per.type = 'JOCKEY'
            AND c.date_course >= %s
            GROUP BY per.id_personne
            ON CONFLICT (id_personne, date_calcul, periode) DO UPDATE SET
                nb_courses = EXCLUDED.nb_courses,
                nb_victoires = EXCLUDED.nb_victoires,
                nb_places = EXCLUDED.nb_places,
                tx_victoire = EXCLUDED.tx_victoire,
                tx_place = EXCLUDED.tx_place,
                nb_courses_plat = EXCLUDED.nb_courses_plat,
                tx_victoire_plat = EXCLUDED.tx_victoire_plat,
                nb_courses_obstacle = EXCLUDED.nb_courses_obstacle,
                tx_victoire_obstacle = EXCLUDED.tx_victoire_obstacle,
                nb_courses_trot = EXCLUDED.nb_courses_trot,
                tx_victoire_trot = EXCLUDED.tx_victoire_trot
        """,
            (periode, date_limite),
        )

        nb_jockeys = self.cur.rowcount

        # Stats entraîneurs
        self.cur.execute(
            """
            INSERT INTO stats_personnes (
                id_personne, date_calcul, periode,
                nb_courses, nb_victoires, nb_places,
                tx_victoire, tx_place
            )
            SELECT
                per.id_personne,
                CURRENT_DATE,
                %s,
                COUNT(*) as nb_courses,
                SUM(CASE WHEN p.position_arrivee = 1 AND p.non_partant = FALSE THEN 1 ELSE 0 END) as victoires,
                SUM(CASE WHEN p.place = TRUE AND p.non_partant = FALSE THEN 1 ELSE 0 END) as places,
                calcul_tx_victoire(
                    SUM(CASE WHEN p.position_arrivee = 1 AND p.non_partant = FALSE THEN 1 ELSE 0 END)::INTEGER,
                    COUNT(CASE WHEN p.non_partant = FALSE THEN 1 END)::INTEGER
                ),
                calcul_tx_victoire(
                    SUM(CASE WHEN p.place = TRUE AND p.non_partant = FALSE THEN 1 ELSE 0 END)::INTEGER,
                    COUNT(CASE WHEN p.non_partant = FALSE THEN 1 END)::INTEGER
                )
            FROM personnes per
            LEFT JOIN performances p ON per.id_personne = p.id_entraineur
            LEFT JOIN courses c ON p.id_course = c.id_course
            WHERE per.type = 'ENTRAINEUR'
            AND c.date_course >= %s
            GROUP BY per.id_personne
            ON CONFLICT (id_personne, date_calcul, periode) DO UPDATE SET
                nb_courses = EXCLUDED.nb_courses,
                nb_victoires = EXCLUDED.nb_victoires,
                nb_places = EXCLUDED.nb_places,
                tx_victoire = EXCLUDED.tx_victoire,
                tx_place = EXCLUDED.tx_place
        """,
            (periode, date_limite),
        )

        nb_entraineurs = self.cur.rowcount

        self.conn.commit()
        self.stats["personnes_updated"] = nb_jockeys + nb_entraineurs

        print(f"✅ {nb_jockeys} jockeys mis à jour")
        print(f"✅ {nb_entraineurs} entraîneurs mis à jour")

    def show_stats(self):
        """Affiche les stats du calcul."""
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ DES CALCULS")
        print("=" * 70)
        for key, value in self.stats.items():
            print(f"   {key:25s} : {value:6d}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Calcul des statistiques agrégées")
    parser.add_argument("--chevaux", action="store_true", help="Calculer stats chevaux")
    parser.add_argument("--personnes", action="store_true", help="Calculer stats personnes")
    parser.add_argument("--all", action="store_true", help="Calculer toutes les stats")
    parser.add_argument("--cheval-id", type=int, help="ID cheval spécifique")
    parser.add_argument(
        "--periode",
        type=str,
        default="12M",
        choices=["1M", "3M", "12M"],
        help="Période pour stats personnes",
    )

    args = parser.parse_args()

    if not any([args.chevaux, args.personnes, args.all, args.cheval_id]):
        parser.print_help()
        sys.exit(1)

    calc = StatsCalculator()
    calc.connect()

    try:
        if args.all or args.chevaux or args.cheval_id:
            calc.calcul_stats_chevaux(id_cheval=args.cheval_id)
            if not args.cheval_id:  # Aptitudes = calcul lourd, skip si ID spécifique
                calc.calcul_aptitudes_chevaux()

        if args.all or args.personnes:
            calc.calcul_stats_personnes(periode=args.periode)

        calc.show_stats()
        print("\n✅ Calculs terminés avec succès !")

    except Exception as e:
        print(f"\n❌ Erreur lors des calculs : {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        calc.close()


if __name__ == "__main__":
    main()
