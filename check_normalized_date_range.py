#!/usr/bin/env python3
"""
Script pour vérifier la plage de dates dans les tables normalisées.
"""

import sys
from db_connection import get_connection


def check_normalized_date_range():
    """Vérifie la plage de dates dans la table courses (normalisée)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Vérifier la plage de dates dans la table courses normalisée
        query = """
        SELECT
            MIN(date_course) as date_min,
            MAX(date_course) as date_max,
            COUNT(*) as total_courses,
            COUNT(DISTINCT date_course) as total_days
        FROM courses;
        """

        cursor.execute(query)
        result = cursor.fetchone()

        if result:
            date_min, date_max, total_courses, total_days = result
            print("📅 Plage de dates dans la table courses (normalisée) :")
            print(f"   Date minimum : {date_min}")
            print(f"   Date maximum : {date_max}")
            print(f"   Total courses : {total_courses:,}")
            print(f"   Total jours uniques : {total_days:,}")

            # Calculer la différence en jours
            if date_min and date_max:
                diff_days = (date_max - date_min).days
                print(f"   Période couverte : {diff_days:,} jours")

        # Vérifier aussi le nombre de performances
        query_perf = """
        SELECT
            COUNT(*) as total_performances,
            COUNT(DISTINCT p.id_cheval) as total_chevaux,
            COUNT(DISTINCT p.id_jockey) as total_jockeys,
            COUNT(DISTINCT p.id_entraineur) as total_entraineurs
        FROM performances p
        JOIN courses c ON p.id_course = c.id_course
        WHERE p.non_partant = FALSE
          AND p.position_arrivee IS NOT NULL
          AND p.position_arrivee > 0;
        """

        cursor.execute(query_perf)
        result_perf = cursor.fetchone()

        if result_perf:
            total_perf, total_chevaux, total_jockeys, total_entraineurs = result_perf
            print("\n📊 Données de performances :")
            print(f"   Total performances valides : {total_perf:,}")
            print(f"   Chevaux uniques : {total_chevaux:,}")
            print(f"   Jockeys uniques : {total_jockeys:,}")
            print(f"   Entraineurs uniques : {total_entraineurs:,}")

        cursor.close()
        conn.close()

        return result

    except Exception as e:
        print(f"❌ Erreur lors de la vérification : {e}")
        return None


if __name__ == "__main__":
    check_normalized_date_range()
