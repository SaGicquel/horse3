#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires minimaux pour vérifier la résolution des données hippodrome.
"""

import sys
import unittest

sys.path.insert(0, ".")

from scraper_pmu_simple import resolve_hippodrome_identifiers, _hippo_str  # noqa: E402
from scrapers.metadata_course import MetadataCourseScraper  # noqa: E402


class HippodromeResolutionTests(unittest.TestCase):
    """Vérifie les fallbacks de récupération code/nom d'hippodrome."""

    def test_resolve_hippodrome_identifiers_codehippodrome(self):
        hippo_payload = {
            "codeHippodrome": "fon",
            "libelleLong": "HIPPODROME DE FONTAINEBLEAU",
        }
        code, name = resolve_hippodrome_identifiers(hippo_payload)
        self.assertEqual(code, "FON")
        self.assertEqual(name, "HIPPODROME DE FONTAINEBLEAU")

    def test_hippo_str_nested_payload(self):
        historical_item = {"hippodrome": {"codeHippodrome": "VIC", "libelleCourt": "VINCENNES"}}
        self.assertEqual(_hippo_str(historical_item), "VIC")

    def test_metadata_course_extract_uses_codehippodrome(self):
        scraper = MetadataCourseScraper(verbose=False)
        course_data = {
            "hippodrome": {"codeHippodrome": "LON", "libelleLong": "LONGCHAMP"},
            "libelle": "PRIX TEST",
            "conditions": None,
            "heureDepart": "14:20:00",
        }
        reunion_data = {"hippodrome": {"codeHippodrome": "LON"}}

        metadata = scraper.extract_metadata("2025-11-03", 1, 2, course_data, reunion_data)

        self.assertEqual(metadata["course_id"], "FR-LON-2025-11-03-R1C2")
        self.assertEqual(metadata["meeting_id"], "FR-LON-2025-11-03-R1")
        self.assertEqual(metadata["heure_locale"], "14:20")


if __name__ == "__main__":
    unittest.main()
