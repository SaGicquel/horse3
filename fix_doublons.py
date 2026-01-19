#!/usr/bin/env python3
"""
Script de nettoyage des doublons dans la table chevaux.
Les doublons sont causés par la différence de casse (majuscules/minuscules).
"""

import sqlite3
from datetime import datetime


def fix_doublons():
    """
    Fusionne les doublons en gardant l'entrée la plus complète.

    Stratégie:
    1. Pour chaque groupe de doublons (même nom en ignorant la casse)
    2. Garder l'entrée avec le plus d'informations (entraîneur, courses, etc.)
    3. Transférer les participations vers l'entrée conservée
    4. Supprimer les autres entrées
    """

    conn = sqlite3.connect("data/database.db")
    cursor = conn.cursor()

    print("=" * 70)
    print("   NETTOYAGE DES DOUBLONS")
    print("=" * 70)

    # Étape 1: Identifier tous les doublons
    print("\n1️⃣  Identification des doublons...")

    cursor.execute("""
        SELECT LOWER(nom) as nom_lower, COUNT(*) as count
        FROM chevaux
        GROUP BY nom_lower
        HAVING COUNT(*) > 1
        ORDER BY count DESC
    """)

    doublons = cursor.fetchall()
    print(f"   ✓ {len(doublons)} groupes de doublons trouvés")

    if not doublons:
        print("\n✅ Aucun doublon à traiter!")
        conn.close()
        return

    # Étape 2: Traiter chaque groupe de doublons
    print("\n2️⃣  Traitement des doublons...")

    total_supprime = 0
    total_fusionne = 0

    for nom_lower, count in doublons:
        # Récupérer toutes les entrées pour ce nom
        cursor.execute(
            """
            SELECT id_cheval, nom, race, sexe, date_naissance,
                   nombre_courses_total, nombre_victoires_total,
                   entraineur_courant, jockey_habituel,
                   dernier_poids_couru, created_at
            FROM chevaux
            WHERE LOWER(nom) = ?
            ORDER BY
                -- Priorité à l'entrée avec entraîneur
                CASE WHEN entraineur_courant IS NOT NULL THEN 0 ELSE 1 END,
                -- Puis par nombre de courses
                nombre_courses_total DESC,
                -- Puis par date de création (plus récent = mieux)
                created_at DESC
        """,
            (nom_lower,),
        )

        entries = cursor.fetchall()

        if len(entries) <= 1:
            continue

        # Garder la première (la meilleure selon nos critères)
        id_a_garder = entries[0][0]
        nom_a_garder = entries[0][1]  # Le nom à garder
        ids_a_supprimer = [entry[0] for entry in entries[1:]]
        noms_a_supprimer = [entry[1] for entry in entries[1:]]

        # Transférer les participations vers le nom à garder
        for nom_ancien in noms_a_supprimer:
            # Mettre à jour les participations (utilise nom_norm, pas id)
            cursor.execute(
                """
                UPDATE cheval_courses_seen
                SET nom_norm = ?
                WHERE nom_norm = ?
            """,
                (nom_a_garder.lower(), nom_ancien.lower()),
            )

        # Supprimer les doublons
        cursor.execute(
            f"""
            DELETE FROM chevaux
            WHERE id_cheval IN ({','.join('?' * len(ids_a_supprimer))})
        """,
            ids_a_supprimer,
        )

        total_supprime += len(ids_a_supprimer)
        total_fusionne += 1

        if total_fusionne % 100 == 0:
            print(f"   Traité: {total_fusionne}/{len(doublons)} groupes...")

    # Étape 3: Recalculer les statistiques pour les chevaux fusionnés
    print("\n3️⃣  Recalcul des statistiques...")

    cursor.execute("""
        UPDATE chevaux
        SET nombre_courses_total = (
            SELECT COUNT(*)
            FROM cheval_courses_seen
            WHERE LOWER(cheval_courses_seen.nom_norm) = LOWER(chevaux.nom)
        )
    """)

    # Étape 4: Commit et vérification
    conn.commit()

    print("\n4️⃣  Vérification finale...")

    cursor.execute("""
        SELECT COUNT(*) as count
        FROM chevaux
        GROUP BY LOWER(nom)
        HAVING COUNT(*) > 1
    """)

    doublons_restants = cursor.fetchall()

    print("\n" + "=" * 70)
    print("   RÉSULTAT")
    print("=" * 70)
    print(f"\n✅ {total_fusionne} groupes de doublons traités")
    print(f"✅ {total_supprime} entrées supprimées")

    if doublons_restants:
        print(f"\n⚠️  {len(doublons_restants)} doublons restants (nécessitent traitement manuel)")
    else:
        print("\n✅ AUCUN DOUBLON RESTANT!")

    # Statistiques finales
    cursor.execute("SELECT COUNT(*) FROM chevaux")
    total_chevaux = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT LOWER(nom)) FROM chevaux")
    noms_uniques = cursor.fetchone()[0]

    print("\n📊 Base nettoyée:")
    print(f"   Total chevaux: {total_chevaux:,}")
    print(f"   Noms uniques: {noms_uniques:,}")

    conn.close()

    print("\n✅ Nettoyage terminé avec succès!")


if __name__ == "__main__":
    try:
        # Backup avant traitement
        print("⚠️  IMPORTANT: Il est recommandé de faire une sauvegarde avant!")
        print("   cp data/database.db data/database.db.backup")

        response = input("\nContinuer? (o/n): ")
        if response.lower() != "o":
            print("❌ Annulé")
            exit(0)

        fix_doublons()

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback

        traceback.print_exc()
