#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilitaire de Statistiques - Base de données PMU
Affiche rapidement les statistiques de la base de données
"""

import sqlite3
from datetime import datetime, date, timedelta

DB_PATH = "data/database.db"


def print_separator():
    print("=" * 80)


def print_subseparator():
    print("-" * 80)


def main():
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()

        print_separator()
        print("📊 STATISTIQUES DE LA BASE DE DONNÉES PMU")
        print_separator()
        print()

        # Statistiques générales
        print("📈 STATISTIQUES GÉNÉRALES")
        print_subseparator()

        cur.execute("SELECT COUNT(*) FROM chevaux")
        nb_chevaux = cur.fetchone()[0]
        print(f"🐴 Chevaux enregistrés: {nb_chevaux:,}")

        cur.execute("SELECT COUNT(DISTINCT race) FROM chevaux WHERE race IS NOT NULL")
        nb_races = cur.fetchone()[0]
        print(f"🏁 Races différentes: {nb_races}")

        cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
        nb_participations = cur.fetchone()[0]
        print(f"📋 Participations totales: {nb_participations:,}")

        if nb_chevaux > 0:
            moy_courses = nb_participations / nb_chevaux
            print(f"📊 Moyenne courses/cheval: {moy_courses:.1f}")

        print()

        # Dates
        print("📅 PÉRIODE COUVERTE")
        print_subseparator()

        cur.execute("""
            SELECT
                MIN(substr(race_key, 1, 10)) as min_date,
                MAX(substr(race_key, 1, 10)) as max_date
            FROM cheval_courses_seen
        """)
        result = cur.fetchone()
        min_date, max_date = result[0], result[1]

        if min_date and max_date:
            print(f"📆 Première course: {min_date}")
            print(f"📆 Dernière course: {max_date}")

            # Calculer le nombre de jours
            try:
                d1 = datetime.strptime(min_date, "%Y-%m-%d").date()
                d2 = datetime.strptime(max_date, "%Y-%m-%d").date()
                nb_jours = (d2 - d1).days + 1
                print(f"📊 Période: {nb_jours} jours")
            except:
                pass

            # Vérifier si à jour
            try:
                derniere_date = datetime.strptime(max_date, "%Y-%m-%d").date()
                aujourd_hui = date.today()
                jours_retard = (aujourd_hui - derniere_date).days

                if jours_retard == 0:
                    print("✅ Base à jour (dernière course: aujourd'hui)")
                elif jours_retard == 1:
                    print("⚠️  Base avec 1 jour de retard (dernière course: hier)")
                elif jours_retard <= 7:
                    print(f"⚠️  Base avec {jours_retard} jours de retard")
                else:
                    print(f"❌ Base avec {jours_retard} jours de retard")
            except:
                pass
        else:
            print("❌ Aucune course enregistrée")

        print()

        # Statistiques par date récente
        print("📊 ACTIVITÉ RÉCENTE (10 derniers jours)")
        print_subseparator()

        cur.execute("""
            SELECT
                substr(race_key, 1, 10) as date_course,
                COUNT(*) as nb_courses
            FROM cheval_courses_seen
            GROUP BY date_course
            ORDER BY date_course DESC
            LIMIT 10
        """)

        dates_recentes = cur.fetchall()
        if dates_recentes:
            for date_course, nb_courses in dates_recentes:
                print(f"  {date_course}: {nb_courses:3d} participations")
        else:
            print("  Aucune donnée")

        print()

        # Top races
        print("🏇 TOP 10 RACES")
        print_subseparator()

        cur.execute("""
            SELECT race, COUNT(*) as nb
            FROM chevaux
            WHERE race IS NOT NULL
            GROUP BY race
            ORDER BY nb DESC
            LIMIT 10
        """)

        top_races = cur.fetchall()
        if top_races:
            for i, (race, nb) in enumerate(top_races, 1):
                print(f"  {i:2d}. {race:30s} {nb:5,} chevaux")
        else:
            print("  Aucune donnée")

        print()

        # Chevaux avec le plus de courses
        print("🏆 TOP 10 CHEVAUX (plus de participations)")
        print_subseparator()

        cur.execute("""
            SELECT nom, nombre_courses_total, nombre_victoires_total
            FROM chevaux
            WHERE nombre_courses_total > 0
            ORDER BY nombre_courses_total DESC
            LIMIT 10
        """)

        top_chevaux = cur.fetchall()
        if top_chevaux:
            for i, (nom, total, victoires) in enumerate(top_chevaux, 1):
                taux = (victoires / total * 100) if total > 0 else 0
                print(
                    f"  {i:2d}. {nom:30s} {total:3d} courses, {victoires:2d} victoires ({taux:4.1f}%)"
                )
        else:
            print("  Aucune donnée")

        print()

        # Répartition par sexe
        print("👫 RÉPARTITION PAR SEXE")
        print_subseparator()

        cur.execute("""
            SELECT sexe, COUNT(*) as nb
            FROM chevaux
            WHERE sexe IS NOT NULL
            GROUP BY sexe
            ORDER BY nb DESC
        """)

        sexes = cur.fetchall()
        if sexes:
            total_sexe = sum(nb for _, nb in sexes)
            for sexe, nb in sexes:
                pct = (nb / total_sexe * 100) if total_sexe > 0 else 0
                sexe_label = {"M": "Mâles", "F": "Femelles", "H": "Hongres"}.get(sexe, sexe)
                print(f"  {sexe_label:15s} {nb:6,} ({pct:5.1f}%)")
        else:
            print("  Aucune donnée")

        print()
        print_separator()
        print(f"🗄️  Base de données: {DB_PATH}")
        print_separator()

        con.close()

    except sqlite3.Error as e:
        print(f"❌ Erreur base de données: {e}")
    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()
