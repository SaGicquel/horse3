#!/usr/bin/env python3
"""
Script de nettoyage des doublons de chevaux
- Fusionne les performances vers l'ID le plus ancien (qui a généralement plus de données)
- Supprime les entrées en double
- Crée une contrainte unique pour éviter les futurs doublons
"""

import logging
from datetime import datetime
from db_connection import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyser_doublons(cur):
    """Analyse les doublons existants"""
    
    # Doublons par nom + parents (même cheval, année différente)
    cur.execute('''
        SELECT nom_cheval, nom_pere, nom_mere,
               array_agg(id_cheval ORDER BY id_cheval) as ids,
               array_agg(an_naissance ORDER BY id_cheval) as annees,
               COUNT(*) as nb
        FROM chevaux
        WHERE nom_pere IS NOT NULL AND nom_mere IS NOT NULL
        GROUP BY nom_cheval, nom_pere, nom_mere
        HAVING COUNT(*) > 1
    ''')
    doublons_parents = cur.fetchall()
    
    # Doublons par nom + code_pmu
    cur.execute('''
        SELECT nom_cheval, code_pmu,
               array_agg(id_cheval ORDER BY id_cheval) as ids,
               COUNT(*) as nb
        FROM chevaux
        WHERE code_pmu IS NOT NULL
        GROUP BY nom_cheval, code_pmu
        HAVING COUNT(*) > 1
    ''')
    doublons_pmu = cur.fetchall()
    
    return doublons_parents, doublons_pmu


def fusionner_chevaux(cur, id_principal, ids_doublons):
    """Fusionne les performances des doublons vers l'ID principal"""
    
    for id_doublon in ids_doublons:
        if id_doublon == id_principal:
            continue
            
        # Compter les performances à migrer
        cur.execute('SELECT COUNT(*) FROM performances WHERE id_cheval = %s', (id_doublon,))
        nb_perfs = cur.fetchone()[0]
        
        if nb_perfs > 0:
            # Migrer les performances (éviter les doublons de course)
            cur.execute('''
                UPDATE performances 
                SET id_cheval = %s 
                WHERE id_cheval = %s
                AND id_course NOT IN (
                    SELECT id_course FROM performances WHERE id_cheval = %s
                )
            ''', (id_principal, id_doublon, id_principal))
            migrees = cur.rowcount
            
            # Supprimer les performances en doublon restantes
            cur.execute('DELETE FROM performances WHERE id_cheval = %s', (id_doublon,))
            
            logger.debug(f"  Migré {migrees} performances de {id_doublon} vers {id_principal}")
        
        # Supprimer le cheval en doublon
        cur.execute('DELETE FROM chevaux WHERE id_cheval = %s', (id_doublon,))


def nettoyer_doublons(dry_run=True):
    """Nettoie tous les doublons"""
    
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # Stats avant
        cur.execute('SELECT COUNT(*) FROM chevaux')
        chevaux_avant = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM performances')
        perfs_avant = cur.fetchone()[0]
        
        logger.info("=" * 60)
        logger.info("🔍 ANALYSE DES DOUBLONS")
        logger.info("=" * 60)
        logger.info(f"Chevaux avant: {chevaux_avant:,}")
        logger.info(f"Performances avant: {perfs_avant:,}")
        
        # Analyser
        doublons_parents, doublons_pmu = analyser_doublons(cur)
        
        logger.info(f"\n📊 Doublons trouvés:")
        logger.info(f"   - Par nom + parents: {len(doublons_parents)} groupes")
        logger.info(f"   - Par nom + code_pmu: {len(doublons_pmu)} groupes")
        
        # Calculer le nombre total de lignes en double
        total_doublons = sum(row[5] - 1 for row in doublons_parents)
        logger.info(f"   - Total lignes à supprimer: {total_doublons}")
        
        if dry_run:
            logger.info("\n⚠️  MODE DRY-RUN - Aucune modification")
            logger.info("   Relancer avec --execute pour appliquer")
            
            # Afficher quelques exemples
            logger.info("\n📝 Exemples de doublons à fusionner:")
            for row in doublons_parents[:10]:
                nom, pere, mere, ids, annees, nb = row
                logger.info(f"   {nom} (père: {pere})")
                logger.info(f"      IDs: {ids}, Années: {annees}")
            
            return
        
        # Mode exécution
        logger.info("\n🔧 NETTOYAGE EN COURS...")
        
        # 1. Fusionner les doublons par parents
        logger.info("\n1️⃣ Fusion des doublons (même nom + parents)...")
        compteur = 0
        for row in doublons_parents:
            nom, pere, mere, ids, annees, nb = row
            id_principal = ids[0]  # Le plus ancien ID
            
            fusionner_chevaux(cur, id_principal, ids[1:])
            compteur += 1
            
            if compteur % 500 == 0:
                logger.info(f"   Traité {compteur}/{len(doublons_parents)} groupes...")
        
        logger.info(f"   ✅ {compteur} groupes fusionnés")
        
        # 2. Vérifier et nettoyer doublons code_pmu restants
        logger.info("\n2️⃣ Vérification doublons code_pmu restants...")
        cur.execute('''
            SELECT nom_cheval, code_pmu,
                   array_agg(id_cheval ORDER BY id_cheval) as ids
            FROM chevaux
            WHERE code_pmu IS NOT NULL
            GROUP BY nom_cheval, code_pmu
            HAVING COUNT(*) > 1
        ''')
        doublons_pmu_restants = cur.fetchall()
        
        for row in doublons_pmu_restants:
            nom, code_pmu, ids = row
            fusionner_chevaux(cur, ids[0], ids[1:])
        
        logger.info(f"   ✅ {len(doublons_pmu_restants)} doublons code_pmu nettoyés")
        
        # 3. Mettre à jour les données du cheval principal avec les meilleures infos
        logger.info("\n3️⃣ Mise à jour des métadonnées...")
        
        # Stats après
        cur.execute('SELECT COUNT(*) FROM chevaux')
        chevaux_apres = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM performances')
        perfs_apres = cur.fetchone()[0]
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 RÉSULTAT DU NETTOYAGE")
        logger.info("=" * 60)
        logger.info(f"Chevaux: {chevaux_avant:,} → {chevaux_apres:,} (supprimés: {chevaux_avant - chevaux_apres:,})")
        logger.info(f"Performances: {perfs_avant:,} → {perfs_apres:,}")
        
        # 4. Créer une contrainte unique pour éviter les futurs doublons
        logger.info("\n4️⃣ Création de la contrainte unique...")
        
        # Supprimer l'ancienne contrainte si elle existe
        cur.execute('''
            SELECT constraint_name FROM information_schema.table_constraints 
            WHERE table_name = 'chevaux' AND constraint_type = 'UNIQUE'
        ''')
        for row in cur.fetchall():
            cur.execute(f'ALTER TABLE chevaux DROP CONSTRAINT IF EXISTS {row[0]}')
        
        # Créer un index unique sur nom + parents (permet NULL)
        cur.execute('''
            CREATE UNIQUE INDEX IF NOT EXISTS idx_chevaux_unique_identity
            ON chevaux (nom_cheval, COALESCE(nom_pere, ''), COALESCE(nom_mere, ''))
        ''')
        logger.info("   ✅ Index unique créé: idx_chevaux_unique_identity")
        
        # Commit
        conn.commit()
        logger.info("\n✅ NETTOYAGE TERMINÉ AVEC SUCCÈS!")
        
        # Vérification finale
        logger.info("\n🔍 Vérification finale...")
        cur.execute('''
            SELECT COUNT(*) FROM (
                SELECT nom_cheval, nom_pere, nom_mere
                FROM chevaux
                GROUP BY nom_cheval, nom_pere, nom_mere
                HAVING COUNT(*) > 1
            ) sub
        ''')
        doublons_restants = cur.fetchone()[0]
        logger.info(f"   Doublons restants: {doublons_restants}")
        
        if doublons_restants == 0:
            logger.info("   ✅ BASE 100% PROPRE - AUCUN DOUBLON!")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Erreur: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def verifier_integrite():
    """Vérifie l'intégrité après nettoyage"""
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        logger.info("\n🔍 VÉRIFICATION D'INTÉGRITÉ")
        logger.info("=" * 60)
        
        # Performances orphelines
        cur.execute('''
            SELECT COUNT(*) FROM performances p
            LEFT JOIN chevaux c ON p.id_cheval = c.id_cheval
            WHERE c.id_cheval IS NULL
        ''')
        orphelines = cur.fetchone()[0]
        logger.info(f"Performances orphelines: {orphelines}")
        
        # Doublons restants
        cur.execute('''
            SELECT COUNT(*) FROM (
                SELECT nom_cheval, nom_pere, nom_mere
                FROM chevaux
                WHERE nom_pere IS NOT NULL
                GROUP BY nom_cheval, nom_pere, nom_mere
                HAVING COUNT(*) > 1
            ) sub
        ''')
        doublons = cur.fetchone()[0]
        logger.info(f"Doublons chevaux: {doublons}")
        
        # Stats générales
        cur.execute('SELECT COUNT(*) FROM chevaux')
        chevaux = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM performances')
        perfs = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM courses')
        courses = cur.fetchone()[0]
        
        logger.info(f"\n📊 Stats finales:")
        logger.info(f"   Chevaux: {chevaux:,}")
        logger.info(f"   Performances: {perfs:,}")
        logger.info(f"   Courses: {courses:,}")
        
        if orphelines == 0 and doublons == 0:
            logger.info("\n✅ BASE DE DONNÉES INTÈGRE!")
        else:
            logger.warning("\n⚠️ Problèmes détectés!")
            
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        print("🚀 Exécution du nettoyage...")
        nettoyer_doublons(dry_run=False)
        verifier_integrite()
    elif len(sys.argv) > 1 and sys.argv[1] == '--check':
        verifier_integrite()
    else:
        print("Usage:")
        print("  python nettoyer_doublons.py          # Dry-run (analyse seule)")
        print("  python nettoyer_doublons.py --execute  # Exécuter le nettoyage")
        print("  python nettoyer_doublons.py --check    # Vérifier l'intégrité")
        print()
        nettoyer_doublons(dry_run=True)
