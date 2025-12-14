#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires pour l'enrichissement des données hippiques
"""

import unittest
import sqlite3
import tempfile
import os
from datetime import date

from enrichment.normalization import (
    normalize_name, extract_birth_year, normalize_country, normalize_sex
)
from enrichment.calculations import (
    parse_time_str, compute_reduction_km, compute_annual_gains,
    compute_total_gains, compute_records, format_reduction_km,
    is_valid_finish_for_gains
)
from enrichment.matching import (
    HorseMatcher, MatchResult, levenshtein_distance
)
from enrichment.migrations import run_migrations


class TestNormalization(unittest.TestCase):
    """Tests de normalisation des noms"""
    
    def test_normalize_name_basic(self):
        """Test normalisation basique"""
        self.assertEqual(normalize_name("Élégant D'Avril"), "ELEGANT DAVRIL")
        self.assertEqual(normalize_name("BLACK SAXON (FR)"), "BLACK SAXON")
        self.assertEqual(normalize_name("L'As-du-Jour"), "LAS DU JOUR")
        self.assertEqual(normalize_name("  Saint  Martin  "), "SAINT MARTIN")
    
    def test_normalize_name_accents(self):
        """Test suppression accents"""
        self.assertEqual(normalize_name("Étoile"), "ETOILE")
        self.assertEqual(normalize_name("Château"), "CHATEAU")
        self.assertEqual(normalize_name("Noël"), "NOEL")
    
    def test_normalize_name_special_chars(self):
        """Test caractères spéciaux"""
        self.assertEqual(normalize_name("Jean-Paul.Martin"), "JEAN PAUL MARTIN")
        self.assertEqual(normalize_name("L'Étoile_des_Prés"), "LETOILE DES PRES")
    
    def test_normalize_name_country_suffix(self):
        """Test suppression suffixe pays"""
        self.assertEqual(normalize_name("CHAMPION (GB)"), "CHAMPION")
        self.assertEqual(normalize_name("WINNER (USA)"), "WINNER")
        self.assertEqual(normalize_name("STAR (FR)"), "STAR")
    
    def test_normalize_name_empty(self):
        """Test valeurs vides"""
        self.assertIsNone(normalize_name(None))
        self.assertIsNone(normalize_name(""))
        self.assertIsNone(normalize_name("   "))
    
    def test_extract_birth_year(self):
        """Test extraction année de naissance"""
        self.assertEqual(extract_birth_year("2018-04-27"), 2018)
        self.assertEqual(extract_birth_year("27/04/2018"), 2018)
        self.assertEqual(extract_birth_year("20180427"), 2018)
        self.assertEqual(extract_birth_year("2018"), 2018)
        self.assertIsNone(extract_birth_year("invalide"))
        self.assertIsNone(extract_birth_year(None))
    
    def test_normalize_country(self):
        """Test normalisation pays"""
        self.assertEqual(normalize_country("france"), "FR")
        self.assertEqual(normalize_country("ROYAUME-UNI"), "GB")
        self.assertEqual(normalize_country("usa"), "US")
        self.assertEqual(normalize_country("FR"), "FR")
    
    def test_normalize_sex(self):
        """Test normalisation sexe"""
        self.assertEqual(normalize_sex("HONGRE"), "H")
        self.assertEqual(normalize_sex("male"), "M")
        self.assertEqual(normalize_sex("Femelle"), "F")
        self.assertEqual(normalize_sex("JUMENT"), "F")
        self.assertIsNone(normalize_sex(None))


class TestCalculations(unittest.TestCase):
    """Tests des calculs hippiques"""
    
    def test_parse_time_str_minutes_seconds(self):
        """Test parsing format minutes'secondes"dixièmes"""
        self.assertAlmostEqual(parse_time_str("1'12\"8"), 72.8)
        self.assertAlmostEqual(parse_time_str("1'11\""), 71.0)
        self.assertAlmostEqual(parse_time_str("1'11"), 71.0)
        self.assertAlmostEqual(parse_time_str("2'05\"5"), 125.5)
    
    def test_parse_time_str_colon_format(self):
        """Test parsing format 1:12.8"""
        self.assertAlmostEqual(parse_time_str("1:12.8"), 72.8)
        self.assertAlmostEqual(parse_time_str("1:11"), 71.0)
    
    def test_parse_time_str_decimal(self):
        """Test parsing format décimal"""
        self.assertAlmostEqual(parse_time_str("68.7"), 68.7)
        self.assertAlmostEqual(parse_time_str("72.3"), 72.3)
    
    def test_parse_time_str_invalid(self):
        """Test valeurs invalides"""
        self.assertIsNone(parse_time_str(None))
        self.assertIsNone(parse_time_str(""))
        self.assertIsNone(parse_time_str("invalide"))
    
    def test_compute_reduction_km_provided(self):
        """Test réduction fournie par API"""
        self.assertEqual(compute_reduction_km(None, 2400, 30.5), 30.5)
    
    def test_compute_reduction_km_calculated(self):
        """Test réduction calculée"""
        # 72.8 secondes pour 2400m = 30.333... s/km
        result = compute_reduction_km(72.8, 2400, None)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 30.333, places=2)
    
    def test_compute_reduction_km_invalid(self):
        """Test valeurs invalides"""
        self.assertIsNone(compute_reduction_km(None, None, None))
        self.assertIsNone(compute_reduction_km(72.8, 0, None))
    
    def test_compute_annual_gains(self):
        """Test calcul gains annuels"""
        perfs = [
            {'allocation_eur': 1000, 'finish_status': '1', 'race_date': '2025-10-15'},
            {'allocation_eur': 500, 'finish_status': '2', 'race_date': '2025-09-20'},
            {'allocation_eur': 300, 'finish_status': 'NP', 'race_date': '2025-08-10'},  # Exclu
            {'allocation_eur': 200, 'finish_status': '4', 'race_date': '2024-12-05'},  # 2024
        ]
        self.assertEqual(compute_annual_gains(perfs, 2025), 1500.0)
        self.assertEqual(compute_annual_gains(perfs, 2024), 200.0)
        self.assertEqual(compute_annual_gains(perfs, 2023), 0.0)
    
    def test_compute_total_gains(self):
        """Test calcul gains totaux"""
        perfs = [
            {'allocation_eur': 1000, 'finish_status': '1'},
            {'allocation_eur': 500, 'finish_status': '2'},
            {'allocation_eur': 300, 'finish_status': 'NP'},  # Exclu
            {'allocation_eur': 200, 'finish_status': '5'},
        ]
        self.assertEqual(compute_total_gains(perfs), 1700.0)
    
    def test_compute_records(self):
        """Test calcul records attelé/monté"""
        perfs = [
            {'reduction_km_sec': 30.5, 'discipline': 'attelé', 'race_date': '2025-10-15', 
             'venue': 'VINCENNES', 'race_code': 'R1C1'},
            {'reduction_km_sec': 28.2, 'discipline': 'attelé', 'race_date': '2025-09-20',
             'venue': 'VINCENNES', 'race_code': 'R1C2'},  # Meilleur
            {'reduction_km_sec': 35.0, 'discipline': 'monté', 'race_date': '2025-08-10',
             'venue': 'LONGCHAMP', 'race_code': 'R1C3'},
        ]
        
        record_attele, record_monte = compute_records(perfs)
        
        self.assertIsNotNone(record_attele)
        self.assertAlmostEqual(record_attele.reduction_km_sec, 28.2)
        self.assertEqual(record_attele.venue, 'VINCENNES')
        
        self.assertIsNotNone(record_monte)
        self.assertAlmostEqual(record_monte.reduction_km_sec, 35.0)
        self.assertEqual(record_monte.venue, 'LONGCHAMP')
    
    def test_format_reduction_km(self):
        """Test formatage réduction"""
        self.assertEqual(format_reduction_km(72.8), "1'12\"8")
        self.assertEqual(format_reduction_km(30.5), "30\"5")
        self.assertEqual(format_reduction_km(125.3), "2'05\"3")
    
    def test_is_valid_finish_for_gains(self):
        """Test validation statut d'arrivée"""
        self.assertTrue(is_valid_finish_for_gains('1'))
        self.assertTrue(is_valid_finish_for_gains('2'))
        self.assertTrue(is_valid_finish_for_gains('5'))
        self.assertFalse(is_valid_finish_for_gains('NP'))
        self.assertFalse(is_valid_finish_for_gains('DAI'))
        self.assertFalse(is_valid_finish_for_gains(''))


class TestMatching(unittest.TestCase):
    """Tests du matching PMU ↔ IFCE"""
    
    def setUp(self):
        """Créer une base de données temporaire pour les tests"""
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.conn = sqlite3.connect(self.db_path)
        run_migrations(self.conn, enable_fts=False)
        
        # Insérer des chevaux de test
        cur = self.conn.cursor()
        cur.executemany("""
            INSERT INTO ifce_horses (name, name_norm, sex, birth_year, country, breed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            ('Black Saxon', 'BLACK SAXON', 'H', 2018, 'FR', 'TROTTEUR FRANÇAIS'),
            ('Élégant D\'Avril', 'ELEGANT DAVRIL', 'M', 2019, 'FR', 'TROTTEUR FRANÇAIS'),
            ('Champion Star', 'CHAMPION STAR', 'M', 2020, 'GB', 'PUR SANG'),
            ('Lady Winner', 'LADY WINNER', 'F', 2019, 'FR', 'TROTTEUR FRANÇAIS'),
        ])
        self.conn.commit()
        
        self.matcher = HorseMatcher(self.conn, enable_fuzzy=False)
    
    def tearDown(self):
        """Nettoyer la base de données temporaire"""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_match_strategy_a_exact(self):
        """Test matching strict (A) avec match exact"""
        result = self.matcher.match_horse(
            "BLACK SAXON", birth_year=2018, sex='H', country='FR'
        )
        self.assertIsNotNone(result.ifce_horse_key)
        self.assertEqual(result.match_stage, 'A')
        self.assertEqual(result.confidence, 1.0)
    
    def test_match_strategy_a_with_accent(self):
        """Test matching strict (A) avec accents"""
        result = self.matcher.match_horse(
            "Élégant D'Avril", birth_year=2019, sex='M', country='FR'
        )
        self.assertIsNotNone(result.ifce_horse_key)
        self.assertEqual(result.match_stage, 'A')
    
    def test_match_strategy_b_year_only(self):
        """Test matching souple (B) avec année uniquement"""
        result = self.matcher.match_horse(
            "LADY WINNER", birth_year=2019
        )
        self.assertIsNotNone(result.ifce_horse_key)
        self.assertEqual(result.match_stage, 'B')
        self.assertGreater(result.confidence, 0.7)
    
    def test_match_strategy_b_country_only(self):
        """Test matching souple (B) avec pays uniquement"""
        result = self.matcher.match_horse(
            "CHAMPION STAR", country='GB'
        )
        self.assertIsNotNone(result.ifce_horse_key)
        self.assertEqual(result.match_stage, 'B')
    
    def test_match_not_found(self):
        """Test cheval non trouvé"""
        result = self.matcher.match_horse(
            "UNKNOWN HORSE", birth_year=2020, sex='M', country='FR'
        )
        self.assertIsNone(result.ifce_horse_key)
        self.assertEqual(result.match_stage, 'none')
    
    def test_levenshtein_distance(self):
        """Test distance de Levenshtein"""
        self.assertEqual(levenshtein_distance("CHAT", "CHAT"), 0)
        self.assertEqual(levenshtein_distance("CHAT", "CHATS"), 1)
        self.assertEqual(levenshtein_distance("CHAT", "CHIEN"), 3)
        self.assertEqual(levenshtein_distance("", "ABC"), 3)
    
    def test_cache_alias(self):
        """Test cache d'alias"""
        # Créer un alias
        self.matcher.cache_alias("BLACK SAXON (FR)", 1)
        
        # Vérifier qu'il est utilisé
        result = self.matcher.match_horse("BLACK SAXON (FR)")
        self.assertIsNotNone(result.ifce_horse_key)
        self.assertEqual(result.match_stage, 'A')
        self.assertEqual(result.confidence, 1.0)
        self.assertIn("Cache", result.details)


class TestMigrations(unittest.TestCase):
    """Tests des migrations"""
    
    def test_migrations_create_tables(self):
        """Test création des tables"""
        db_fd, db_path = tempfile.mkstemp()
        conn = sqlite3.connect(db_path)
        
        run_migrations(conn, enable_fts=False)
        
        cur = conn.cursor()
        
        # Vérifier que les tables existent
        tables = [
            'ifce_horses', 'pmu_horses', 'performances',
            'horse_year_stats', 'horse_totals', 'horse_aliases'
        ]
        
        for table in tables:
            cur.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table,))
            self.assertIsNotNone(cur.fetchone(), f"Table {table} devrait exister")
        
        conn.close()
        os.close(db_fd)
        os.unlink(db_path)
    
    def test_migrations_idempotent(self):
        """Test que les migrations sont idempotentes (peuvent être relancées)"""
        db_fd, db_path = tempfile.mkstemp()
        conn = sqlite3.connect(db_path)
        
        # Lancer 2 fois
        run_migrations(conn, enable_fts=False)
        run_migrations(conn, enable_fts=False)
        
        # Pas d'erreur = succès
        self.assertTrue(True)
        
        conn.close()
        os.close(db_fd)
        os.unlink(db_path)


def run_tests():
    """Lance tous les tests"""
    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    suite.addTests(loader.loadTestsFromTestCase(TestNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestCalculations))
    suite.addTests(loader.loadTestsFromTestCase(TestMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestMigrations))
    
    # Lancer les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES TESTS")
    print("=" * 70)
    print(f"Tests exécutés : {result.testsRun}")
    print(f"✅ Succès : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Échecs : {len(result.failures)}")
    print(f"💥 Erreurs : {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
