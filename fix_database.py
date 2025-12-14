#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de correction automatique des problèmes de la base de données
Propose des corrections pour les anomalies détectées
"""

import sqlite3
import sys

DB_PATH = "data/database.db"

def print_section(title):
    """Affiche un titre de section"""
    print("\n" + "=" * 80)
    print(f"🔧 {title}")
    print("=" * 80)

def fix_coherence_issues(con):
    """Corrige les incohérences de compteurs"""
    print_section("CORRECTION DES INCOHÉRENCES DE COMPTEURS")
    
    cur = con.cursor()
    
    # Recalculer depuis cheval_courses_seen
    print("\n📊 Recalcul des compteurs depuis l'historique...")
    
    cur.execute("""
        UPDATE chevaux
        SET
            nombre_courses_total = (
                SELECT COUNT(*) FROM cheval_courses_seen s
                WHERE s.nom_norm = LOWER(chevaux.nom)
            ),
            nombre_victoires_total = (
                SELECT COALESCE(SUM(is_win),0) FROM cheval_courses_seen s
                WHERE s.nom_norm = LOWER(chevaux.nom)
            ),
            nombre_courses_2025 = (
                SELECT COUNT(*) FROM cheval_courses_seen s
                WHERE s.nom_norm = LOWER(chevaux.nom) AND s.annee = 2025
            ),
            nombre_victoires_2025 = (
                SELECT COALESCE(SUM(is_win),0) FROM cheval_courses_seen s
                WHERE s.nom_norm = LOWER(chevaux.nom) AND s.annee = 2025
            )
    """)
    
    rows_updated = cur.rowcount
    con.commit()
    
    print(f"✅ {rows_updated} chevaux mis à jour")
    
    # Vérifier les incohérences restantes
    cur.execute("""
        SELECT COUNT(*) FROM chevaux
        WHERE nombre_victoires_total > nombre_courses_total
        AND nombre_courses_total IS NOT NULL
        AND nombre_victoires_total IS NOT NULL
    """)
    
    still_incoherent = cur.fetchone()[0]
    
    if still_incoherent > 0:
        print(f"⚠️  {still_incoherent} incohérences persistent après correction")
    else:
        print("✅ Toutes les incohérences ont été corrigées")

def fix_orphans(con):
    """Crée les chevaux manquants pour les enregistrements orphelins"""
    print_section("CRÉATION DES CHEVAUX MANQUANTS (ORPHELINS)")
    
    cur = con.cursor()
    
    # Trouver les orphelins avec leurs stats
    cur.execute("""
        SELECT 
            ccs.nom_norm,
            COUNT(*) as nb_courses,
            SUM(ccs.is_win) as nb_victoires,
            SUM(CASE WHEN ccs.annee = 2025 THEN 1 ELSE 0 END) as nb_courses_2025,
            SUM(CASE WHEN ccs.annee = 2025 AND ccs.is_win = 1 THEN 1 ELSE 0 END) as nb_victoires_2025
        FROM cheval_courses_seen ccs
        LEFT JOIN chevaux c ON LOWER(c.nom) = ccs.nom_norm
        WHERE c.id_cheval IS NULL
        GROUP BY ccs.nom_norm
    """)
    
    orphelins = cur.fetchall()
    
    if not orphelins:
        print("\n✅ Aucun orphelin à créer")
        return
    
    print(f"\n⚠️  {len(orphelins)} chevaux orphelins détectés")
    print("Ces chevaux ont des courses enregistrées mais n'existent pas dans la table 'chevaux'")
    print("→ Ils vont être créés avec leurs statistiques calculées depuis l'historique")
    
    # Créer les chevaux manquants
    nb_created = 0
    for nom_norm, nb_courses, nb_victoires, nb_courses_2025, nb_victoires_2025 in orphelins:
        print(f"\n   • Création: {nom_norm}")
        print(f"      → {nb_courses} courses, {nb_victoires} victoires")
        
        # Insérer le cheval avec les stats calculées
        cur.execute("""
            INSERT INTO chevaux (
                nom, 
                nombre_courses_total, 
                nombre_victoires_total,
                nombre_courses_2025,
                nombre_victoires_2025,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (nom_norm, nb_courses, nb_victoires or 0, nb_courses_2025, nb_victoires_2025 or 0))
        
        nb_created += 1
    
    con.commit()
    
    print(f"\n✅ {nb_created} chevaux orphelins créés avec succès")
    print(f"   Ils apparaissent maintenant dans la table 'chevaux' avec leurs statistiques")

def fix_duplicate_horses(con):
    """Fusionne les doublons de chevaux"""
    print_section("TRAITEMENT DES DOUBLONS DE CHEVAUX")
    
    cur = con.cursor()
    
    # Trouver les doublons stricts (même nom + date de naissance)
    cur.execute("""
        SELECT LOWER(nom) as nom_norm, date_naissance, COUNT(*) as cnt
        FROM chevaux
        WHERE date_naissance IS NOT NULL
        GROUP BY nom_norm, date_naissance
        HAVING COUNT(*) > 1
    """)
    
    doublons = cur.fetchall()
    
    if not doublons:
        print("\n✅ Aucun doublon strict à traiter")
        return
    
    print(f"\n⚠️  {len(doublons)} doublons stricts détectés")
    print("Ces doublons seront fusionnés (le plus complet sera gardé)")
    
    for nom_norm, date_naiss, cnt in doublons:
        # Récupérer tous les IDs de ce doublon
        cur.execute("""
            SELECT id_cheval, nom, race, sexe, nombre_courses_total
            FROM chevaux
            WHERE LOWER(nom) = ? AND date_naissance = ?
            ORDER BY nombre_courses_total DESC NULLS LAST, id_cheval ASC
        """, (nom_norm, date_naiss))
        
        chevaux = cur.fetchall()
        
        if len(chevaux) <= 1:
            continue
        
        # Garder le premier (le plus complet)
        to_keep = chevaux[0][0]
        to_delete = [ch[0] for ch in chevaux[1:]]
        
        print(f"\n   • {nom_norm} (né {date_naiss}):")
        print(f"      → Garde ID {to_keep}")
        print(f"      → Supprime IDs {to_delete}")
        
        # Mettre à jour les références dans cheval_courses_seen
        for id_to_delete in to_delete:
            cur.execute("""
                UPDATE cheval_courses_seen
                SET nom_norm = (SELECT LOWER(nom) FROM chevaux WHERE id_cheval = ?)
                WHERE nom_norm = (SELECT LOWER(nom) FROM chevaux WHERE id_cheval = ?)
            """, (to_keep, id_to_delete))
        
        # Supprimer les doublons
        cur.execute("""
            DELETE FROM chevaux WHERE id_cheval IN ({})
        """.format(','.join('?' * len(to_delete))), to_delete)
    
    con.commit()
    print(f"\n✅ {len(doublons)} doublons fusionnés")

def normalize_all_races(con):
    """Normalise toutes les races dans la base"""
    print_section("NORMALISATION DES RACES")
    
    cur = con.cursor()
    
    # Mapping des normalisations
    normalizations = {
        'PUR-SANG': 'PUR SANG',
        'PUR SANG ARABE': 'ARABE',
        'ANGLO-ARABE': 'ANGLO ARABE',
        'ANGLO-ARABE DE COMPLEMENT': 'ANGLO ARABE DE COMPLEMENT',
        'TROTTEUR FRANÇAIS': 'TROTTEUR FRANCAIS',
        'TROTTEUR FRANÇAIS ÉTRANGER': 'TROTTEUR ETRANGER',
    }
    
    total_updated = 0
    
    for old_race, new_race in normalizations.items():
        cur.execute("""
            UPDATE chevaux SET race = ? WHERE race = ?
        """, (new_race, old_race))
        
        updated = cur.rowcount
        if updated > 0:
            print(f"   • '{old_race}' → '{new_race}': {updated} chevaux")
            total_updated += updated
    
    con.commit()
    
    if total_updated > 0:
        print(f"\n✅ {total_updated} chevaux normalisés")
    else:
        print("\n✅ Toutes les races sont déjà normalisées")

def main():
    print("=" * 80)
    print("🔧 CORRECTION AUTOMATIQUE DE LA BASE DE DONNÉES")
    print("=" * 80)
    print()
    print("Ce script va:")
    print("  1. Recalculer les compteurs depuis l'historique")
    print("  2. Créer les chevaux manquants (orphelins) avec leurs stats")
    print("  3. Fusionner les doublons de chevaux")
    print("  4. Normaliser les noms de races")
    print()
    
    response = input("⚠️  Continuer ? (o/N): ").strip().lower()
    
    if response != 'o':
        print("\n❌ Correction annulée")
        return
    
    print("\n🔧 Début des corrections...")
    
    try:
        con = sqlite3.connect(DB_PATH)
        
        # Backup avant modifications
        print("\n💾 Création d'un backup...")
        backup_con = sqlite3.connect(f"{DB_PATH}.backup_auto")
        con.backup(backup_con)
        backup_con.close()
        print("✅ Backup créé: data/database.db.backup_auto")
        
        # Corrections
        fix_coherence_issues(con)
        fix_orphans(con)
        fix_duplicate_horses(con)
        normalize_all_races(con)
        
        con.close()
        
        print("\n" + "=" * 80)
        print("✅ CORRECTIONS TERMINÉES")
        print("=" * 80)
        print()
        print("💡 Lancez `python verify_database.py` pour vérifier le résultat")
        print()
        
    except Exception as e:
        print(f"\n❌ Erreur lors des corrections: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
