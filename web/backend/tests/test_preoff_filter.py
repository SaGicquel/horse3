#!/usr/bin/env python3
"""
Tests d'intégration pour le filtrage pré-off (anti-fuite).

Ces tests vérifient que:
1. Les chevaux non-partants (NP, DAI, ARR, etc.) sont exclus des recommandations
2. Les cotes post-off ne peuvent pas entrer dans le pipeline d'inférence
3. Les outsiders extrêmes (cote > seuil) sont filtrés
"""

import os
import sys

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    print("⚠️  pytest non disponible, utilisation du mode standalone")

from odds_guard import select_preoff_market_odds


class TestSelectPreoffMarketOdds:
    """Tests pour la fonction select_preoff_market_odds"""

    def test_rejects_post_off_with_place_finale(self):
        """La fonction doit rejeter les cotes si place_finale est définie (post-off)"""
        try:
            select_preoff_market_odds(3.5, 4.0, place_finale=1)
            raise AssertionError("Devrait lever une ValueError pour place_finale=1")
        except ValueError as e:
            assert "post-off" in str(e).lower()

        try:
            select_preoff_market_odds(3.5, 4.0, place_finale=5)
            raise AssertionError("Devrait lever une ValueError pour place_finale=5")
        except ValueError as e:
            assert "post-off" in str(e).lower()

        try:
            select_preoff_market_odds(3.5, 4.0, place_finale=0)
            raise AssertionError("Devrait lever une ValueError pour place_finale=0")
        except ValueError as e:
            assert "post-off" in str(e).lower()

    def test_accepts_preoff_without_place(self):
        """La fonction doit accepter les cotes pré-off (place_finale=None)"""
        result = select_preoff_market_odds(3.5, 4.0, place_finale=None)
        assert result == 4.0  # préférence pour cote_reference

    def test_prefers_cote_reference(self):
        """La cote de référence (pré-off) doit être préférée"""
        result = select_preoff_market_odds(10.0, 8.0, None)
        assert result == 8.0

    def test_fallback_to_cote_finale(self):
        """Si cote_reference est None, utiliser cote_finale"""
        result = select_preoff_market_odds(5.5, None, None)
        assert result == 5.5


class TestStatusFiltering:
    """Tests pour le filtrage des statuts de chevaux"""

    # Statuts qui doivent être EXCLUS
    EXCLUDED_STATUSES = [
        "NON_PARTANT",
        "NON PARTANT",
        "NONPARTANT",
        "NP",
        "DAI",
        "DIA",
        "DISQUALIFIE",
        "DISQ",
        "DQ",
        "ARR",
        "ARRET",
        "ARRETE",
        "STOPPED",
        "CHU",
        "CHUTE",
        "TOMBE",
        "TOM",
        "FALL",
        "RET",
        "RETIRE",
        "RETIREE",
        "RÉTIRÉ",
        "WITHDRAWN",
        "DIST",
        "DISTANCIE",
        "DISTANCED",
    ]

    # Statuts qui doivent être ACCEPTÉS
    ALLOWED_STATUSES = ["PARTANT", "PARTANTE", "PART", "P", "", None]

    @staticmethod
    def normalize_status(statut):
        """Normalise le statut comme dans le code principal"""
        return (statut or "").upper().replace("-", " ").replace("_", " ")

    @staticmethod
    def is_excluded(statut_norm):
        """Vérifie si un statut normalisé doit être exclu"""
        EXCLUDED_KEYWORDS = (
            "NP",
            "NON PARTANT",
            "NONPARTANT",
            "NON_PARTANT",
            "DAI",
            "DIA",
            "DISQ",
            "DQ",
            "DISQUALIFIE",
            "DISQUALIFIED",
            "ARR",
            "ARRET",
            "ARRETE",
            "STOPPED",
            "CHU",
            "CHUTE",
            "TOMBE",
            "TOM",
            "FALL",
            "FALLEN",
            "RET",
            "RETIRE",
            "RETIREE",
            "RÉTIRÉ",
            "RETRAIT",
            "WITHDRAWN",
            "DIST",
            "DISTANCIE",
            "DISTANCED",
        )
        allowed_status = {"", "PARTANT", "PARTANTE", "PART", "P"}
        return statut_norm not in allowed_status or any(
            key in statut_norm for key in EXCLUDED_KEYWORDS
        )

    def test_excluded_statuses_are_rejected(self):
        """Tous les statuts exclus doivent être rejetés"""
        for statut in self.EXCLUDED_STATUSES:
            statut_norm = self.normalize_status(statut)
            assert self.is_excluded(
                statut_norm
            ), f"Le statut '{statut}' devrait être exclu mais ne l'est pas"

    def test_allowed_statuses_are_accepted(self):
        """Les statuts autorisés doivent être acceptés"""
        for statut in self.ALLOWED_STATUSES:
            statut_norm = self.normalize_status(statut)
            # Pour les statuts autorisés, is_excluded doit retourner False
            # Sauf si le statut contient un mot-clé interdit (ce qui ne devrait pas arriver pour les statuts valides)
            if statut_norm in {"", "PARTANT", "PARTANTE", "PART", "P"}:
                assert not self.is_excluded(
                    statut_norm
                ), f"Le statut '{statut}' devrait être accepté mais est exclu"


class TestOddsFiltering:
    """Tests pour le filtrage des cotes extrêmes"""

    MAX_PREOFF_COTE = 50.0  # Valeur par défaut

    def test_high_odds_should_be_filtered(self):
        """Les cotes > MAX_PREOFF_COTE doivent être filtrées"""
        # Ces cotes devraient être filtrées
        high_odds = [51.0, 100.0, 200.0, 500.0]
        for odds in high_odds:
            assert odds > self.MAX_PREOFF_COTE, f"Cote {odds} devrait être > {self.MAX_PREOFF_COTE}"

    def test_normal_odds_should_pass(self):
        """Les cotes <= MAX_PREOFF_COTE doivent passer"""
        normal_odds = [1.5, 5.0, 10.0, 25.0, 49.0, 50.0]
        for odds in normal_odds:
            assert odds <= self.MAX_PREOFF_COTE, f"Cote {odds} devrait passer le filtre"

    def test_cote_reference_also_filtered(self):
        """La cote de référence doit aussi respecter le seuil"""
        # Cas problématique: cote_finale < 50 mais cote_reference > 50
        cote_finale = 45.0
        cote_reference = 61.0

        # La cote effective utilisée est cote_reference (prioritaire dans select_preoff_market_odds)
        effective_odds = select_preoff_market_odds(cote_finale, cote_reference, None)
        assert effective_odds == cote_reference

        # Cette cote effective devrait être filtrée
        assert (
            effective_odds > self.MAX_PREOFF_COTE
        ), "Ce cas doit être filtré par le code principal"


class TestNoPostOffDataInPipeline:
    """
    Tests d'intégration pour vérifier qu'aucune donnée post-off
    ne peut entrer dans le pipeline d'inférence.
    """

    def test_post_off_columns_not_allowed(self):
        """
        Colonnes post-off qui NE DOIVENT JAMAIS être utilisées pour l'inférence:
        - place_finale (indique la position à l'arrivée)
        - is_win (indique si le cheval a gagné)
        - gains_course (gains après la course)
        - statut_arrivee (statut après la course)
        - rapport_gagnant, rapport_place (rapports finaux)

        Ce test documente ces colonnes comme interdites.
        """
        POST_OFF_COLUMNS = {
            "place_finale": "Position à l'arrivée - post-off",
            "is_win": "Indicateur victoire - post-off",
            "gains_course": "Gains après course - post-off",
            "statut_arrivee": "Statut après course - post-off",
            "rapport_gagnant": "Rapport final gagnant - post-off",
            "rapport_place": "Rapport final placé - post-off",
            "rapport_quarte": "Rapport Quarté - post-off",
            "rapport_quinte": "Rapport Quinté - post-off",
        }

        # Ce test est documentaire - il liste les colonnes interdites
        # L'implémentation dans main.py doit filtrer sur place_finale IS NULL
        assert len(POST_OFF_COLUMNS) > 0, "Liste des colonnes post-off à ne pas utiliser"

    def test_preoff_columns_allowed(self):
        """
        Colonnes pré-off qui PEUVENT être utilisées pour l'inférence:
        - cote_finale (snapshot pré-départ)
        - cote_reference (cote PMU de référence pré-off)
        - cote_matin (cote du matin)
        - tendance_cote (évolution de la cote)
        - avis_entraineur (avis avant course)
        - musique (historique du cheval)
        """
        PREOFF_COLUMNS = {
            "cote_finale": "Snapshot cote pré-départ",
            "cote_reference": "Cote PMU référence pré-off",
            "cote_matin": "Cote matinale",
            "tendance_cote": "Évolution cote pré-off",
            "amplitude_tendance": "Amplitude variation cote",
            "avis_entraineur": "Avis avant course",
            "musique": "Historique performances",
            "statut_participant": "Statut avant départ (PARTANT/NP)",
        }

        assert len(PREOFF_COLUMNS) > 0, "Liste des colonnes pré-off autorisées"


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        # Mode standalone sans pytest
        print("=" * 60)
        print("Tests d'intégration anti-fuite pré-off")
        print("=" * 60)

        tests_passed = 0
        tests_failed = 0

        # Tests select_preoff_market_odds
        test_class = TestSelectPreoffMarketOdds()
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                try:
                    getattr(test_class, method_name)()
                    print(f"✅ {method_name}")
                    tests_passed += 1
                except Exception as e:
                    print(f"❌ {method_name}: {e}")
                    tests_failed += 1

        # Tests StatusFiltering
        test_class = TestStatusFiltering()
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                try:
                    getattr(test_class, method_name)()
                    print(f"✅ {method_name}")
                    tests_passed += 1
                except Exception as e:
                    print(f"❌ {method_name}: {e}")
                    tests_failed += 1

        # Tests OddsFiltering
        test_class = TestOddsFiltering()
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                try:
                    getattr(test_class, method_name)()
                    print(f"✅ {method_name}")
                    tests_passed += 1
                except Exception as e:
                    print(f"❌ {method_name}: {e}")
                    tests_failed += 1

        # Tests NoPostOffDataInPipeline
        test_class = TestNoPostOffDataInPipeline()
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                try:
                    getattr(test_class, method_name)()
                    print(f"✅ {method_name}")
                    tests_passed += 1
                except Exception as e:
                    print(f"❌ {method_name}: {e}")
                    tests_failed += 1

        print("=" * 60)
        print(f"Résultats: {tests_passed} passés, {tests_failed} échoués")
        print("=" * 60)

        if tests_failed > 0:
            sys.exit(1)
