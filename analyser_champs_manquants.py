#!/usr/bin/env python3
"""
Script pour comparer les champs de la BDD vides avec les champs disponibles dans l'API PMU
et générer un plan d'action pour compléter le scraper
"""

import json
import psycopg2
from db_connection import get_connection
from collections import defaultdict

# Mapping des colonnes BDD vers les champs API PMU
CHAMPS_API_MAPPING = {
    # Infos course de base (déjà récupérées)
    "discipline": "discipline",
    "specialite": "specialite",
    "distance_m": "distance",
    "corde": "corde",
    "allocation_totale": "montantPrix",
    "course_nom": "libelle",
    "conditions_course": "conditions",
    "heure_depart": "heureDepart",
    # Infos hippodrome (partiellement récupérées)
    "hippodrome_code": "hippodrome.codeHippodrome",
    "hippodrome_nom": "hippodrome.libelleLong",
    "code_hippodrome": "hippodrome.codeHippodrome",
    "nom_hippodrome": "hippodrome.libelleLong",
    "pays_hippodrome": "hippodrome.pays",  # À vérifier dans API
    "region_hippodrome": "hippodrome.region",  # À vérifier
    # Infos course avancées (MANQUANTES - 0% rempli)
    "type_depart": "parcours",  # GP = Grand Parcours
    "type_piste": "typePiste",  # Peut être déduit
    "etat_piste": "reunion.etatPiste",  # Dans les données réunion
    "meteo": "reunion.meteo",  # Dans les données réunion
    "profil_piste": "reunion.profilPiste",  # À vérifier
    "classe_course": "categorieParticularite",  # ou autre champ
    "prix_course": "libelle",  # Déjà récupéré normalement
    # Allocations détaillées (MANQUANTES - 0%)
    "allocation_premier": "montantOffert1er",  # ✅ Disponible!
    "allocation_deuxieme": "montantOffert2eme",  # ✅ Disponible!
    "allocation_troisieme": "montantOffert3eme",  # ✅ Disponible!
    "montant_enjeux_total": "rapports.enjeux",  # Dans rapports
    # Commentaires (MANQUANTS - 0%)
    "commentaire_apres_course": "commentaireApresCourse.texte",  # ✅ Disponible!
    "commentaire_avant_course": "commentaireAvantCourse.texte",  # À vérifier
    # Participants - Infos de base
    "numero_dossard": "participant.numero",
    "driver_jockey": "participant.driver.nom",
    "entraineur": "participant.entraineur.nom",
    "proprietaire": "participant.proprietaire.nom",
    "eleveur": "participant.eleveur.nom",  # ✅ À vérifier
    # Participants - Identifiants PMU (MANQUANTS - 0%)
    "id_cheval_pmu": "participant.idCheval",  # ✅ Disponible!
    "id_driver_pmu": "participant.driver.id",  # ✅ Disponible!
    "id_jockey_pmu": "participant.jockey.id",  # ✅ Disponible!
    "id_entraineur_pmu": "participant.entraineur.id",  # ✅ Disponible!
    # Participants - Physique (majorité vides)
    "age": "participant.age",
    "sexe": "participant.sexe",
    "robe": "participant.robe",
    "race": "participant.race",
    "pays_naissance": "participant.paysNaissance",  # ✅ À vérifier
    "origine_complete": "participant.origine",  # ✅ À vérifier
    # Participants - Équipement (majorité vides)
    "equipement": "participant.equipement",
    "deferrage": "participant.deferrage",
    "ferrure": "participant.ferrure",  # ✅ À vérifier
    "materiel": "participant.materiel",  # ✅ À vérifier
    "oeilleres": "participant.oeilleres",
    # Participants - Handicap & poids (majorité vides)
    "handicap_distance": "participant.handicapDistance",
    "handicap_valeur": "participant.handicapValeur",  # ✅ À vérifier
    "poids_kg": "participant.poids",  # ✅ À vérifier
    "poids_porte_kg": "participant.poidsPorte",  # ✅ À vérifier
    "decharge_kg": "participant.decharge",  # ✅ À vérifier
    # Participants - Cotes (majorité vides)
    "cote_matin": "participant.coteMatin",  # ✅ À vérifier dans cotes
    "cote_finale": "participant.coteFinale",  # ✅ Dans rapports/cotes
    "cote_evolution_pct": "participant.coteEvolution",  # Calculé
    "probabilite_implicite": "calculé depuis cote_finale",  # 1/cote
    "tendance_marche": "participant.tendance",  # ✅ À vérifier
    # Résultats course (majorité vides)
    "place_finale": "participant.place.place",
    "statut_arrivee": "participant.place.statut",  # ✅ À vérifier
    "temps_str": "participant.performance.temps",  # ✅ À vérifier
    "temps_sec": "participant.performance.tempsSecondes",  # Calculé
    "temps_total_s": "participant.performance.tempsTotal",  # ✅ À vérifier
    "ecarts": "participant.performance.ecarts",  # ✅ À vérifier
    "ecart_premier": "participant.performance.ecartPremier",  # ✅ À vérifier
    "ecart_precedent": "participant.performance.ecartPrecedent",  # ✅ À vérifier
    # Vitesses (MANQUANTES - 0%)
    "vitesse_moyenne": "participant.performance.vitesseMoyenne",  # Calculé
    "vitesse_fin_course": "participant.performance.vitesseFinale",  # ✅ À vérifier
    "reduction_km": "participant.performance.reductionKm",
    "reduction_km_sec": "participant.performance.reductionKmSec",
    # Spécifique trot (MANQUANTS - 0%)
    "autostart_ligne": "participant.autostartLigne",  # ✅ À vérifier
    "autostart_num": "participant.autostartNumero",  # ✅ À vérifier
    # Gains (majorité vides)
    "gains_course": "participant.gainsObtenus",  # ✅ À vérifier
    "gains_carriere": "participant.gainsCarriere",  # ✅ À vérifier
    # Météo détaillée (MANQUANTES - 0%)
    "meteo_code": "reunion.meteo.code",  # ✅ À vérifier structure
    "temperature_c": "reunion.meteo.temperature",  # ✅ À vérifier
    "vent_kmh": "reunion.meteo.vent",  # ✅ À vérifier
    "penetrometre": "reunion.penetrometre",  # ✅ À vérifier
    # Autres (MANQUANTS - 0%)
    "num_pmu": "participant.numero",  # Même que numero_dossard
    "note_journaliste": "participant.noteJournaliste",  # ✅ À vérifier pronostics
    "observations": "participant.observations",  # ✅ À vérifier
    "pmu_reunion_id": "reunion.numOfficiel",
    "pmu_course_id": "course.numOrdre",
    "days_off": "calculé depuis dernière course",  # Calculé
    "statut_sante": "participant.statutSante",  # ✅ À vérifier
    # Casaques (MANQUANTES - 0%)
    "couleurs_casaque_driver": "participant.driver.couleursCasaque",  # ✅ À vérifier
    "couleurs_casaque_jockey": "participant.jockey.couleursCasaque",  # ✅ À vérifier
    # Rapports détaillés (majorité manquants)
    "rapport_quarte": "rapports.QUARTE_PLUS",
    "rapport_quinte": "rapports.QUINTE_PLUS",
    "rapport_multi": "rapports.MULTI",
    "rapport_pick5": "rapports.PICK5",
}


def analyser_champs_manquants():
    """Analyse complète des champs manquants"""

    print("=" * 80)
    print("ANALYSE DES CHAMPS MANQUANTS ET PLAN D'ACTION")
    print("=" * 80)

    conn = get_connection()
    cur = conn.cursor()

    # Récupérer les stats des colonnes
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    total_rows = cur.fetchone()[0]

    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'cheval_courses_seen'
        ORDER BY ordinal_position
    """)
    colonnes = cur.fetchall()

    # Analyser chaque colonne
    colonnes_vides = []

    for col_name, col_type in colonnes:
        cur.execute(f"""
            SELECT COUNT(*)
            FROM cheval_courses_seen
            WHERE {col_name} IS NULL
        """)
        null_count = cur.fetchone()[0]

        unknown_count = 0
        if "text" in col_type or "character" in col_type:
            cur.execute(f"""
                SELECT COUNT(*)
                FROM cheval_courses_seen
                WHERE {col_name} = 'UNKNOWN'
            """)
            unknown_count = cur.fetchone()[0]

        empty_count = null_count + unknown_count
        pct_empty = (empty_count / total_rows * 100) if total_rows > 0 else 0

        if pct_empty > 50:  # Plus de 50% vide
            colonnes_vides.append(
                {
                    "colonne": col_name,
                    "pct_vide": pct_empty,
                    "null": null_count,
                    "unknown": unknown_count,
                    "champ_api": CHAMPS_API_MAPPING.get(col_name, "❓ NON IDENTIFIÉ"),
                }
            )

    # Trier par pourcentage vide
    colonnes_vides.sort(key=lambda x: x["pct_vide"], reverse=True)

    # Grouper par catégorie
    categories = {
        "course": [],
        "hippodrome": [],
        "participant": [],
        "resultats": [],
        "rapports": [],
        "meteo": [],
        "identifiants": [],
        "autres": [],
    }

    for col in colonnes_vides:
        name = col["colonne"]
        if any(k in name for k in ["hippodrome", "pays_hippodrome", "region"]):
            categories["hippodrome"].append(col)
        elif any(
            k in name
            for k in [
                "driver",
                "jockey",
                "entraineur",
                "proprietaire",
                "eleveur",
                "age",
                "sexe",
                "robe",
                "race",
                "poids",
                "equipement",
                "deferrage",
            ]
        ):
            categories["participant"].append(col)
        elif any(k in name for k in ["rapport", "enjeux", "allocation"]):
            categories["rapports"].append(col)
        elif any(
            k in name for k in ["place", "temps", "vitesse", "ecart", "gains", "statut_arrivee"]
        ):
            categories["resultats"].append(col)
        elif any(k in name for k in ["meteo", "temperature", "vent", "penetro", "etat_piste"]):
            categories["meteo"].append(col)
        elif any(k in name for k in ["id_", "pmu_", "code_", "num_"]):
            categories["identifiants"].append(col)
        elif any(
            k in name
            for k in ["type_", "classe", "profil", "parcours", "commentaire", "prix_course"]
        ):
            categories["course"].append(col)
        else:
            categories["autres"].append(col)

    # Afficher par catégorie
    print(f"\nTotal lignes: {total_rows:,}")
    print(f"Total colonnes >50% vides: {len(colonnes_vides)}")

    categories_ordre = [
        "course",
        "hippodrome",
        "participant",
        "resultats",
        "rapports",
        "meteo",
        "identifiants",
        "autres",
    ]

    for cat_name in categories_ordre:
        cols = categories[cat_name]
        if not cols:
            continue

        print(f"\n{'='*80}")
        print(f"📦 {cat_name.upper()} ({len(cols)} champs)")
        print("=" * 80)

        for col in cols:
            emoji = (
                "✅"
                if col["champ_api"] != "❓ NON IDENTIFIÉ" and "✅" in col["champ_api"]
                else "🔴"
                if col["pct_vide"] > 90
                else "🟠"
            )
            print(f"\n{emoji} {col['colonne']:35s} {100-col['pct_vide']:5.1f}% rempli")
            print(f"   API: {col['champ_api']}")

    cur.close()
    conn.close()

    # Générer le plan d'action
    print(f"\n{'='*80}")
    print("🎯 PLAN D'ACTION POUR COMPLÉTER LE SCRAPER")
    print("=" * 80)

    print("""
PRIORITÉ 1 - Champs facilement disponibles (API directe)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Allocations détaillées (course_data):
   ✅ allocation_premier = course_data.get('montantOffert1er')
   ✅ allocation_deuxieme = course_data.get('montantOffert2eme')
   ✅ allocation_troisieme = course_data.get('montantOffert3eme')

2. Commentaire après course:
   ✅ commentaire_apres_course = course_data.get('commentaireApresCourse', {}).get('texte')

3. Identifiants PMU participants:
   ✅ id_cheval_pmu = participant.get('idCheval')
   ✅ id_driver_pmu = participant.get('driver', {}).get('id')
   ✅ id_jockey_pmu = participant.get('jockey', {}).get('id')
   ✅ id_entraineur_pmu = participant.get('entraineur', {}).get('id')

4. Identifiants course/réunion:
   ✅ pmu_reunion_id = reunion_data.get('numOfficiel')
   ✅ pmu_course_id = course_data.get('numOrdre')

PRIORITÉ 2 - Champs nécessitant un appel API supplémentaire
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. Cotes détaillées:
   🔍 Appeler /cotes pour récupérer cote_matin, cote_finale, tendance

6. Météo détaillée:
   🔍 Vérifier si disponible dans reunion_data (temperature, vent, etc.)

7. Performances détaillées:
   🔍 Vérifier dans données historiques (temps_total_s, vitesses, etc.)

PRIORITÉ 3 - Champs calculés
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

8. Vitesses:
   📊 vitesse_moyenne = (distance_m / 1000) / (temps_sec / 3600)
   📊 probabilite_implicite = 1 / cote_finale

9. Jours de repos:
   📊 days_off = date_course - date_derniere_course

PRIORITÉ 4 - Champs optionnels/avancés
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

10. Analyses ML:
    🤖 combinaison_winrate, biais_stalle, effet_topographie
    → À calculer après enrichissement complet

11. Pronostics:
    📝 note_journaliste, probabilite_victoire
    → Nécessite accès pronostics PMU
""")

    print("\n" + "=" * 80)
    print("✅ Analyse terminée")
    print("=" * 80)


if __name__ == "__main__":
    analyser_champs_manquants()
