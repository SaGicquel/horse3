#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour mettre à jour les rapports détaillés depuis rapports_json existants
"""

import sys
from db_connection import get_connection
from scraper_pmu_simple import extract_rapports_detailles


def update_rapports(date_iso=None):
    """
    Met à jour les colonnes rapport_quarte, rapport_quinte, etc.
    depuis les rapports_json existants
    """
    con = get_connection()
    cur = con.cursor()

    # Récupérer toutes les courses avec rapports_json
    if date_iso:
        cur.execute(
            """
            SELECT race_key, rapports_json
            FROM cheval_courses_seen
            WHERE race_key LIKE %s
            AND rapports_json IS NOT NULL
        """,
            (f"{date_iso}|%",),
        )
    else:
        cur.execute("""
            SELECT race_key, rapports_json
            FROM cheval_courses_seen
            WHERE rapports_json IS NOT NULL
            AND (rapport_quarte IS NULL OR rapport_quinte IS NULL)
        """)

    rows = cur.fetchall()
    total = len(rows)

    print(f"📊 Mise à jour de {total} courses...")
    print("=" * 80)

    updated_quarte = 0
    updated_quinte = 0
    updated_multi = 0
    updated_pick5 = 0
    updated_total = 0

    for i, (race_key, rapports_json) in enumerate(rows, 1):
        if i % 100 == 0:
            print(f"  Traité {i}/{total}...")

        # Extraire les rapports
        rapports_detail = extract_rapports_detailles(rapports_json)

        # Mettre à jour
        updates = []
        params = []

        if rapports_detail.get("rapport_quarte"):
            updates.append("rapport_quarte = %s")
            params.append(rapports_detail["rapport_quarte"])
            updated_quarte += 1

        if rapports_detail.get("rapport_quinte"):
            updates.append("rapport_quinte = %s")
            params.append(rapports_detail["rapport_quinte"])
            updated_quinte += 1

        if rapports_detail.get("rapport_multi"):
            updates.append("rapport_multi = %s")
            params.append(rapports_detail["rapport_multi"])
            updated_multi += 1

        if rapports_detail.get("rapport_pick5"):
            updates.append("rapport_pick5 = %s")
            params.append(rapports_detail["rapport_pick5"])
            updated_pick5 += 1

        if rapports_detail.get("montant_enjeux_total"):
            updates.append("montant_enjeux_total = %s")
            params.append(int(rapports_detail["montant_enjeux_total"]))
            updated_total += 1

        if updates:
            params.append(race_key)
            sql = f"""
                UPDATE cheval_courses_seen
                SET {', '.join(updates)}
                WHERE race_key = %s
            """
            cur.execute(sql, params)

    con.commit()

    print()
    print("=" * 80)
    print("✅ MISE À JOUR TERMINÉE")
    print("=" * 80)
    print(f"  • Courses avec Quarté+: {updated_quarte}")
    print(f"  • Courses avec Quinté+: {updated_quinte}")
    print(f"  • Courses avec Multi: {updated_multi}")
    print(f"  • Courses avec Pick5: {updated_pick5}")
    print(f"  • Courses avec montant enjeux: {updated_total}")

    con.close()


if __name__ == "__main__":
    date_iso = sys.argv[1] if len(sys.argv) > 1 else None

    if date_iso:
        print(f"📅 Mise à jour pour le {date_iso}")
    else:
        print("📅 Mise à jour pour toutes les dates")

    update_rapports(date_iso)
