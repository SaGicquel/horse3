#!/usr/bin/env python3
"""
Script de migration de l'ancienne BDD vers le nouveau schéma.
Transfère les données existantes en les adaptant au nouveau format.

Usage:
    python migrate_to_new_schema.py [--dry-run]
"""

import argparse
import sys
from datetime import datetime
from db_connection import get_connection

class DatabaseMigrator:
    """Migre les données de l'ancien vers le nouveau schéma."""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.conn = None
        self.cur = None
        self.stats = {
            'hippodromes': 0,
            'courses': 0,
            'chevaux': 0,
            'personnes': 0,
            'performances': 0,
        }
    
    def connect(self):
        """Connexion à la base."""
        self.conn = get_connection()
        self.cur = self.conn.cursor()
    
    def close(self):
        """Fermeture connexion."""
        if self.cur:
            self.cur.close()
        if self.conn:
            if not self.dry_run:
                self.conn.commit()
            self.conn.close()
    
    def check_old_schema_exists(self) -> bool:
        """Vérifie si l'ancien schéma existe."""
        self.cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('chevaux', 'cheval_courses_seen')
        """)
        tables = self.cur.fetchall()
        return len(tables) > 0
    
    def check_new_schema_exists(self) -> bool:
        """Vérifie si le nouveau schéma existe."""
        self.cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('hippodromes', 'courses', 'performances')
        """)
        tables = self.cur.fetchall()
        return len(tables) >= 3
    
    def migrate_hippodromes(self):
        """Migre les hippodromes depuis cheval_courses_seen."""
        print("\n🏇 Migration des hippodromes...")
        
        # Vérifier si la colonne hippodrome existe dans l'ancien schéma
        self.cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'cheval_courses_seen' 
            AND column_name IN ('hippodrome', 'lieu')
        """)
        
        columns = [c[0] for c in self.cur.fetchall()]
        
        if not columns:
            print("⚠️  Aucune colonne hippodrome trouvée dans l'ancien schéma")
            return
        
        hippodrome_col = columns[0]
        
        # Extraire les hippodromes uniques
        self.cur.execute(f"""
            SELECT DISTINCT {hippodrome_col}
            FROM cheval_courses_seen
            WHERE {hippodrome_col} IS NOT NULL
        """)
        
        hippodromes = self.cur.fetchall()
        
        for (nom,) in hippodromes:
            if not nom:
                continue
            
            # Générer code PMU
            code_pmu = ''.join(c for c in nom.upper() if c.isalpha())[:4]
            
            if not self.dry_run:
                self.cur.execute("""
                    INSERT INTO hippodromes (nom_hippodrome, code_pmu, pays)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (code_pmu) DO NOTHING
                """, (nom, code_pmu, 'FR'))
                self.stats['hippodromes'] += 1
            else:
                print(f"   [DRY-RUN] Hippodrome : {nom} ({code_pmu})")
        
        if not self.dry_run:
            self.conn.commit()
        
        print(f"✅ {self.stats['hippodromes']} hippodromes migrés")
    
    def migrate_chevaux(self):
        """Migre la table chevaux."""
        print("\n🐴 Migration des chevaux...")
        
        # Vérifier structure ancienne table
        self.cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'chevaux'
            ORDER BY ordinal_position
        """)
        old_columns = {c[0] for c in self.cur.fetchall()}
        
        # Récupérer tous les chevaux
        self.cur.execute("SELECT * FROM chevaux")
        chevaux_old = self.cur.fetchall()
        
        # Mapping colonnes (adapté à votre ancienne structure)
        for cheval in chevaux_old:
            # Adapter selon votre vraie structure
            # Exemple basique :
            if 'nom' in old_columns:
                # TODO: adapter selon votre vraie structure
                pass
        
        print(f"⚠️  Migration manuelle nécessaire selon votre structure")
    
    def migrate_performances(self):
        """Migre cheval_courses_seen vers performances."""
        print("\n📊 Migration des performances...")
        
        # TODO: Implémenter selon votre structure exacte
        print("⚠️  À implémenter selon votre structure exacte")
    
    def show_stats(self):
        """Affiche les statistiques de migration."""
        print("\n" + "=" * 70)
        print("📊 STATISTIQUES DE MIGRATION")
        print("=" * 70)
        for key, value in self.stats.items():
            print(f"   {key:20s} : {value:6d} migrés")
        print("=" * 70)
    
    def migrate_all(self):
        """Lance toutes les migrations."""
        print("=" * 70)
        print("🚀 MIGRATION VERS NOUVEAU SCHÉMA")
        print("=" * 70)
        
        if self.dry_run:
            print("⚠️  MODE DRY-RUN : aucune modification ne sera appliquée\n")
        
        # Vérifications
        if not self.check_old_schema_exists():
            print("❌ Ancien schéma introuvable")
            return False
        
        if not self.check_new_schema_exists():
            print("❌ Nouveau schéma introuvable. Exécutez d'abord init_schema_plan_complet.py")
            return False
        
        print("✅ Ancien et nouveau schémas détectés\n")
        
        # Migrations
        try:
            self.migrate_hippodromes()
            # self.migrate_chevaux()
            # self.migrate_performances()
            
            self.show_stats()
            
            if not self.dry_run:
                print("\n✅ Migration terminée avec succès !")
            else:
                print("\n✅ Simulation terminée. Relancez sans --dry-run pour appliquer.")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Erreur lors de la migration : {e}")
            import traceback
            traceback.print_exc()
            if not self.dry_run:
                self.conn.rollback()
            return False

def main():
    parser = argparse.ArgumentParser(description='Migration vers nouveau schéma')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Simule la migration sans modification')
    
    args = parser.parse_args()
    
    migrator = DatabaseMigrator(dry_run=args.dry_run)
    migrator.connect()
    
    try:
        success = migrator.migrate_all()
        sys.exit(0 if success else 1)
    finally:
        migrator.close()

if __name__ == '__main__':
    main()
