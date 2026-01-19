#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de création des chevaux orphelins
Crée les chevaux manquants qui ont des courses dans cheval_courses_seen
"""

import sqlite3
import sys

DB_PATH = "data/database.db"


def clean_orphans():
    """Crée les chevaux manquants (orphelins)"""
    print("=" * 80)
    print("✨ CRÉATION DES CHEVAUX MANQUANTS (ORPHELINS)")
    print("=" * 80)
    print()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Trouver les orphelins avec leurs statistiques
    cur.execute("""
        SELECT
            ccs.nom_norm,
            COUNT(*) as nb_courses,
            SUM(ccs.is_win) as nb_victoires,
            SUM(CASE WHEN ccs.annee = 2025 THEN 1 ELSE 0 END) as nb_courses_2025,
            SUM(CASE WHEN ccs.annee = 2025 AND ccs.is_win = 1 THEN 1 ELSE 0 END) as nb_victoires_2025,
            MIN(ccs.race_key) as premiere_course
        FROM cheval_courses_seen ccs
        LEFT JOIN chevaux c ON LOWER(c.nom) = ccs.nom_norm
        WHERE c.id_cheval IS NULL
        GROUP BY ccs.nom_norm
        ORDER BY nb_courses DESC
    """)

    orphelins = cur.fetchall()

    if not orphelins:
        print("✅ Aucun cheval orphelin trouvé")
        print("   Tous les chevaux de cheval_courses_seen existent dans la table chevaux")
        con.close()
        return

    print(f"⚠️  {len(orphelins)} chevaux orphelins trouvés:")
    print("   (Ces chevaux ont des courses mais n'existent pas dans la table 'chevaux')")
    print()

    for nom, nb_courses, nb_victoires, nb_2025, nbv_2025, premiere in orphelins[:20]:
        print(f"   • {nom}")
        print(f"      → {nb_courses} course(s), {nb_victoires or 0} victoire(s)")
        print(f"      → 2025: {nb_2025} course(s), {nbv_2025 or 0} victoire(s)")
        print(f"      → Première course: {premiere}")
        print()

    if len(orphelins) > 20:
        print(f"   ... et {len(orphelins) - 20} autres")

    print()
    print("Ces chevaux vont être créés dans la table 'chevaux' avec leurs statistiques.")
    print()

    response = input("Voulez-vous créer ces chevaux manquants ? (O/n): ").strip().lower()

    if response == "n":
        print("\n❌ Création annulée")
        con.close()
        return

    # Créer les chevaux manquants
    print("\n✨ Création en cours...")

    nb_created = 0
    for nom, nb_courses, nb_victoires, nb_2025, nbv_2025, premiere in orphelins:
        cur.execute(
            """
            INSERT INTO chevaux (
                nom,
                nombre_courses_total,
                nombre_victoires_total,
                nombre_courses_2025,
                nombre_victoires_2025,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (nom, nb_courses, nb_victoires or 0, nb_2025, nbv_2025 or 0),
        )
        nb_created += 1

    con.commit()

    print(f"\n✅ {nb_created} chevaux créés avec succès dans la table 'chevaux'")
    print("   Ils ont maintenant un id_cheval et leurs statistiques sont renseignées")

    # Vérifier qu'il n'y a plus d'orphelins
    cur.execute("""
        SELECT COUNT(DISTINCT ccs.nom_norm)
        FROM cheval_courses_seen ccs
        LEFT JOIN chevaux c ON LOWER(c.nom) = ccs.nom_norm
        WHERE c.id_cheval IS NULL
    """)

    remaining = cur.fetchone()[0]

    if remaining == 0:
        print("\n✅ Parfait ! Plus aucun orphelin dans la base")
    else:
        print(f"\n⚠️  Il reste encore {remaining} orphelins (vérifiez la normalisation des noms)")

    con.close()


if __name__ == "__main__":
    try:
        clean_orphans()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
