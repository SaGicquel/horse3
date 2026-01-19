#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vérification complète après l'orchestrateur
Affiche les statistiques sur les colonnes enrichies
"""

import psycopg2

# Configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 54624,
    "database": "pmubdd",
    "user": "postgres",
    "password": "okokok",
}


def verify_enrichment():
    """Vérifie l'enrichissement des données"""

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("=" * 80)
    print("VÉRIFICATION DE L'ENRICHISSEMENT - 2025-11-04")
    print("=" * 80)
    print()

    # 1. Statistiques globales
    cur.execute("""
        SELECT
            COUNT(*) as total_participations,
            COUNT(DISTINCT nom_norm) as chevaux_uniques
        FROM cheval_courses_seen
        WHERE race_key LIKE '2025-11-04%'
    """)
    total_parts, chevaux_uniques = cur.fetchone()
    print("📊 STATISTIQUES GLOBALES")
    print(f"   Total participations: {total_parts}")
    print(f"   Chevaux uniques: {chevaux_uniques}")
    print()

    # 2. SCRAPER 1 - Métadonnées
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE temperature_c IS NOT NULL) as temp,
            COUNT(*) FILTER (WHERE vent_kmh IS NOT NULL) as vent,
            COUNT(*) FILTER (WHERE meteo_code IS NOT NULL) as meteo,
            COUNT(*) FILTER (WHERE draw_stalle IS NOT NULL) as stalle,
            COUNT(*) FILTER (WHERE autostart_num IS NOT NULL) as autostart
        FROM cheval_courses_seen
        WHERE race_key LIKE '2025-11-04%'
    """)
    temp, vent, meteo, stalle, autostart = cur.fetchone()
    print("📋 SCRAPER 1 - Métadonnées course")
    print(f"   Température: {temp}/{total_parts} ({temp*100//total_parts}%)")
    print(f"   Vent: {vent}/{total_parts} ({vent*100//total_parts if total_parts else 0}%)")
    print(f"   Météo: {meteo}/{total_parts} ({meteo*100//total_parts if total_parts else 0}%)")
    print(f"   Stalle: {stalle}/{total_parts} ({stalle*100//total_parts if total_parts else 0}%)")
    print(
        f"   Autostart: {autostart}/{total_parts} ({autostart*100//total_parts if total_parts else 0}%)"
    )
    print()

    # 3. SCRAPER 2 - Détails cheval
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE num_pmu IS NOT NULL) as num,
            COUNT(*) FILTER (WHERE poids_porte_kg IS NOT NULL) as poids,
            COUNT(*) FILTER (WHERE ferrure IS NOT NULL) as ferrure,
            COUNT(*) FILTER (WHERE materiel IS NOT NULL) as materiel,
            COUNT(*) FILTER (WHERE days_off IS NOT NULL) as days_off
        FROM cheval_courses_seen
        WHERE race_key LIKE '2025-11-04%'
    """)
    num, poids, ferrure, materiel, days_off = cur.fetchone()
    print("🐴 SCRAPER 2 - Détails cheval")
    print(f"   Numéro PMU: {num}/{total_parts} ({num*100//total_parts if total_parts else 0}%)")
    print(
        f"   Poids porté: {poids}/{total_parts} ({poids*100//total_parts if total_parts else 0}%)"
    )
    print(
        f"   Ferrure: {ferrure}/{total_parts} ({ferrure*100//total_parts if total_parts else 0}%)"
    )
    print(
        f"   Matériel: {materiel}/{total_parts} ({materiel*100//total_parts if total_parts else 0}%)"
    )
    print(
        f"   Days off: {days_off}/{total_parts} ({days_off*100//total_parts if total_parts else 0}%)"
    )
    print()

    # 4. SCRAPER 4 - Conditions jour
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE pace_debut IS NOT NULL) as pace_d,
            COUNT(*) FILTER (WHERE pace_milieu IS NOT NULL) as pace_m,
            COUNT(*) FILTER (WHERE pace_fin IS NOT NULL) as pace_f,
            COUNT(*) FILTER (WHERE penetrometre IS NOT NULL) as penetro
        FROM cheval_courses_seen
        WHERE race_key LIKE '2025-11-04%'
    """)
    pace_d, pace_m, pace_f, penetro = cur.fetchone()
    print("🏁 SCRAPER 4 - Conditions jour")
    print(
        f"   Pace début: {pace_d}/{total_parts} ({pace_d*100//total_parts if total_parts else 0}%)"
    )
    print(
        f"   Pace milieu: {pace_m}/{total_parts} ({pace_m*100//total_parts if total_parts else 0}%)"
    )
    print(f"   Pace fin: {pace_f}/{total_parts} ({pace_f*100//total_parts if total_parts else 0}%)")
    print(
        f"   Pénétromètre: {penetro}/{total_parts} ({penetro*100//total_parts if total_parts else 0}%)"
    )
    print()

    # 5. SCRAPER 5 - Cotes marché
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE cote_evolution_pct IS NOT NULL) as evol,
            COUNT(*) FILTER (WHERE tendance_marche IS NOT NULL) as tendance,
            COUNT(*) FILTER (WHERE est_favori = TRUE) as favoris,
            COUNT(*) FILTER (WHERE est_outsider = TRUE) as outsiders
        FROM cheval_courses_seen
        WHERE race_key LIKE '2025-11-04%'
    """)
    evol, tendance, favoris, outsiders = cur.fetchone()
    print("💰 SCRAPER 5 - Cotes marché")
    print(
        f"   Évolution cote: {evol}/{total_parts} ({evol*100//total_parts if total_parts else 0}%)"
    )
    print(
        f"   Tendance: {tendance}/{total_parts} ({tendance*100//total_parts if total_parts else 0}%)"
    )
    print(f"   Favoris: {favoris}")
    print(f"   Outsiders: {outsiders}")
    print()

    # 6. SCRAPER 7 - Features ML
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE indice_confiance IS NOT NULL) as confiance,
            COUNT(*) FILTER (WHERE probabilite_victoire IS NOT NULL) as proba_win,
            COUNT(*) FILTER (WHERE kelly_criterion IS NOT NULL) as kelly,
            COUNT(*) FILTER (WHERE is_value_bet = TRUE) as value_bets
        FROM cheval_courses_seen
        WHERE race_key LIKE '2025-11-04%'
    """)
    confiance, proba_win, kelly, value_bets = cur.fetchone()
    print("🤖 SCRAPER 7 - Features ML")
    print(
        f"   Indice confiance: {confiance}/{total_parts} ({confiance*100//total_parts if total_parts else 0}%)"
    )
    print(
        f"   Probabilité victoire: {proba_win}/{total_parts} ({proba_win*100//total_parts if total_parts else 0}%)"
    )
    print(
        f"   Kelly criterion: {kelly}/{total_parts} ({kelly*100//total_parts if total_parts else 0}%)"
    )
    print(f"   Value bets: {value_bets}")
    print()

    # 7. Exemple de ligne enrichie
    cur.execute("""
        SELECT
            nom_norm,
            temperature_c,
            vent_kmh,
            num_pmu,
            poids_porte_kg,
            ferrure,
            pace_debut,
            indice_confiance
        FROM cheval_courses_seen
        WHERE race_key LIKE '2025-11-04%'
          AND temperature_c IS NOT NULL
          AND num_pmu IS NOT NULL
        LIMIT 1
    """)
    row = cur.fetchone()
    if row:
        print("📝 EXEMPLE DE LIGNE ENRICHIE")
        print(f"   Cheval: {row[0]}")
        print(f"   Température: {row[1]}°C")
        print(f"   Vent: {row[2]} km/h")
        print(f"   Numéro PMU: {row[3]}")
        print(f"   Poids: {row[4]} kg")
        print(f"   Ferrure: {row[5]}")
        print(f"   Pace début: {row[6]}")
        print(f"   Confiance ML: {row[7]}")

    cur.close()
    conn.close()

    print()
    print("=" * 80)
    print("✅ VÉRIFICATION TERMINÉE")
    print("=" * 80)


if __name__ == "__main__":
    verify_enrichment()
