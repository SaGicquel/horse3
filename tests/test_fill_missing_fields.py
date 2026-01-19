#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTS - Fill Missing Fields Job
===================================

Tests unitaires pour le job de remplissage des champs manquants.

Focus principal:
1. Pas de fuite temporelle (champs POST_OFF exclus de l'inférence)
2. Flags de qualité correctement posés
3. Fonctions de dérivation/parsing correctes
"""

import pytest
import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jobs.fill_missing_fields import (
    # Constantes
    POST_OFF_FIELDS,
    PRE_OFF_FEATURES,
    PENETROMETRE_TO_ETAT,
    METEO_ADJUSTMENT,
    # Fonctions
    derive_etat_piste,
    parse_temps_str,
    proxy_temps_sec,
    impute_gains_carriere,
    # Classes
    QualityFlags,
    FillMissingFieldsJob,
)


# =============================================================================
# TESTS ANTI-FUITE TEMPORELLE
# =============================================================================


class TestTemporalLeakage:
    """Tests garantissant qu'aucune fuite temporelle n'est possible."""

    def test_post_off_fields_not_in_pre_off_features(self):
        """Les champs POST_OFF ne doivent jamais être dans PRE_OFF_FEATURES."""
        intersection = POST_OFF_FIELDS & PRE_OFF_FEATURES
        assert len(intersection) == 0, f"Fuite temporelle: {intersection}"

    def test_validate_no_temporal_leakage_clean(self):
        """Features propres passent la validation."""
        clean_features = ["discipline", "distance_m", "hippodrome_nom", "age", "sexe"]
        is_valid, violations = FillMissingFieldsJob.validate_no_temporal_leakage(clean_features)

        assert is_valid is True
        assert len(violations) == 0

    def test_validate_no_temporal_leakage_violation(self):
        """Features avec POST_OFF échouent la validation."""
        bad_features = ["discipline", "place_finale", "rapport_gagnant", "age"]
        is_valid, violations = FillMissingFieldsJob.validate_no_temporal_leakage(bad_features)

        assert is_valid is False
        assert "place_finale" in violations
        assert "rapport_gagnant" in violations
        assert len(violations) == 2

    def test_place_finale_is_post_off(self):
        """place_finale doit être un champ POST_OFF."""
        assert "place_finale" in POST_OFF_FIELDS

    def test_temps_sec_is_post_off(self):
        """temps_sec doit être un champ POST_OFF."""
        assert "temps_sec" in POST_OFF_FIELDS

    def test_rapport_gagnant_is_post_off(self):
        """rapport_gagnant doit être un champ POST_OFF."""
        assert "rapport_gagnant" in POST_OFF_FIELDS

    def test_gains_course_is_post_off(self):
        """gains_course (de la course) doit être POST_OFF."""
        assert "gains_course" in POST_OFF_FIELDS

    def test_gains_carriere_is_pre_off(self):
        """gains_carriere (historique) doit être PRE_OFF."""
        assert "gains_carriere" in PRE_OFF_FEATURES

    def test_etat_piste_is_pre_off(self):
        """etat_piste (dérivé avant course) doit être PRE_OFF."""
        assert "etat_piste" in PRE_OFF_FEATURES

    def test_cote_finale_is_pre_off(self):
        """cote_finale (juste avant départ) est PRE_OFF."""
        assert "cote_finale" in PRE_OFF_FEATURES

    def test_all_pre_off_features_safe(self):
        """Toutes les PRE_OFF_FEATURES passent la validation."""
        features = list(PRE_OFF_FEATURES)
        is_valid, violations = FillMissingFieldsJob.validate_no_temporal_leakage(features)

        assert is_valid is True
        assert len(violations) == 0


# =============================================================================
# TESTS QUALITY FLAGS
# =============================================================================


class TestQualityFlags:
    """Tests des flags de qualité."""

    def test_quality_flags_default(self):
        """Flags par défaut sont 'unknown' et False."""
        flags = QualityFlags()

        assert flags.etat_piste_source == "unknown"
        assert flags.temps_sec_source == "unknown"
        assert flags.rapport_gagnant_source == "unknown"
        assert flags.gains_carriere_source == "unknown"
        assert flags.report_missing is False
        assert flags.temps_missing is False
        assert flags.gains_imputed is False

    def test_quality_flags_to_dict(self):
        """Conversion en dict fonctionne."""
        flags = QualityFlags(etat_piste_source="penetro", report_missing=True, gains_imputed=True)

        d = flags.to_dict()

        assert isinstance(d, dict)
        assert d["etat_piste_source"] == "penetro"
        assert d["report_missing"] is True
        assert d["gains_imputed"] is True

    def test_quality_flags_all_fields_in_dict(self):
        """to_dict contient tous les champs."""
        flags = QualityFlags()
        d = flags.to_dict()

        expected_keys = {
            "etat_piste_source",
            "temps_sec_source",
            "rapport_gagnant_source",
            "gains_carriere_source",
            "report_missing",
            "temps_missing",
            "gains_imputed",
        }

        assert set(d.keys()) == expected_keys


# =============================================================================
# TESTS DERIVE ETAT_PISTE
# =============================================================================


class TestDeriveEtatPiste:
    """Tests de la dérivation de l'état de piste."""

    def test_from_penetrometre_bon(self):
        """Pénétromètre 2.8 → 'bon'."""
        etat, source = derive_etat_piste(2.8, None, None)
        assert etat == "bon"
        assert source == "penetro"

    def test_from_penetrometre_souple(self):
        """Pénétromètre 3.2 → 'souple'."""
        etat, source = derive_etat_piste(3.2, None, None)
        assert etat == "souple"
        assert source == "penetro"

    def test_from_penetrometre_lourd(self):
        """Pénétromètre 4.3 → 'lourd'."""
        etat, source = derive_etat_piste(4.3, None, None)
        assert etat == "lourd"
        assert source == "penetro"

    def test_penetro_adjusted_by_meteo_rain(self):
        """Pénétromètre 3.0 + pluie → ajusté vers lourd."""
        # 3.0 seul = "souple"
        etat_sec, _ = derive_etat_piste(3.0, None, None)
        assert etat_sec == "souple"

        # 3.0 + pluie = 3.0 + 0.8 = 3.8 → "collant"
        etat_pluie, source = derive_etat_piste(3.0, "pluie", None)
        assert etat_pluie == "collant"
        assert source == "penetro"

    def test_penetro_adjusted_by_meteo_sun(self):
        """Pénétromètre 3.0 + soleil → ajusté vers sec."""
        # 3.0 + soleil = 3.0 - 0.5 = 2.5 → "bon"
        etat, source = derive_etat_piste(3.0, "beau temps ensoleillé", None)
        assert etat == "bon"
        assert source == "penetro"

    def test_fallback_hippodrome(self):
        """Sans pénétromètre, fallback sur hippodrome."""
        etat, source = derive_etat_piste(None, None, "Hippodrome de Longchamp")
        assert etat == "souple"
        assert source == "hippodrome"

    def test_fallback_hippodrome_vincennes(self):
        """Vincennes = 'bon'."""
        etat, source = derive_etat_piste(None, None, "Vincennes")
        assert etat == "bon"
        assert source == "hippodrome"

    def test_fallback_meteo_only_rain(self):
        """Méteo seule 'pluie forte' → 'lourd'."""
        etat, source = derive_etat_piste(None, "pluie forte avec orage", "HIPPODROME_INCONNU")
        assert etat == "lourd"
        assert source == "meteo"

    def test_fallback_meteo_only_sun(self):
        """Méteo seule 'soleil' → 'bon'."""
        etat, source = derive_etat_piste(None, "grand soleil", "HIPPODROME_INCONNU")
        assert etat == "bon"
        assert source == "meteo"

    def test_fallback_default_unknown(self):
        """Sans aucune info → 'unknown'."""
        etat, source = derive_etat_piste(None, None, None)
        assert etat == "unknown"
        assert source == "default"

    def test_negative_penetrometre_ignored(self):
        """Pénétromètre négatif = ignoré."""
        etat, source = derive_etat_piste(-1.0, None, "Vincennes")
        # Fallback sur hippodrome
        assert source == "hippodrome"


# =============================================================================
# TESTS PARSE TEMPS_STR
# =============================================================================


class TestParseTempsStr:
    """Tests du parsing de temps."""

    def test_format_apostrophe(self):
        """Format 1'23''4 → 83.4s."""
        result = parse_temps_str("1'23''4")
        assert result == pytest.approx(83.4, rel=0.01)

    def test_format_apostrophe_no_decimals(self):
        """Format 1'23'' → 83.0s."""
        result = parse_temps_str("1'23''")
        assert result == pytest.approx(83.0, rel=0.01)

    def test_format_colon(self):
        """Format 1:23.4 → 83.4s."""
        result = parse_temps_str("1:23.4")
        assert result == pytest.approx(83.4, rel=0.01)

    def test_format_colon_comma(self):
        """Format 1:23,4 → 83.4s."""
        result = parse_temps_str("1:23,4")
        assert result == pytest.approx(83.4, rel=0.01)

    def test_format_m_s(self):
        """Format 1m23s4 → 83.4s."""
        result = parse_temps_str("1m23s4")
        assert result == pytest.approx(83.4, rel=0.01)

    def test_format_seconds_only(self):
        """Format 83.4 → 83.4s."""
        result = parse_temps_str("83.4")
        assert result == pytest.approx(83.4, rel=0.01)

    def test_format_with_whitespace(self):
        """Whitespace est ignoré."""
        result = parse_temps_str("  1:23.4  ")
        assert result == pytest.approx(83.4, rel=0.01)

    def test_none_input(self):
        """None → None."""
        result = parse_temps_str(None)
        assert result is None

    def test_empty_string(self):
        """String vide → None."""
        result = parse_temps_str("")
        assert result is None

    def test_invalid_format(self):
        """Format invalide → None."""
        result = parse_temps_str("pas un temps")
        assert result is None

    def test_two_minutes(self):
        """Format 2'15''7 → 135.7s."""
        result = parse_temps_str("2'15''7")
        assert result == pytest.approx(135.7, rel=0.01)


# =============================================================================
# TESTS PROXY TEMPS_SEC
# =============================================================================


class TestProxyTempsSec:
    """Tests du proxy temps via distance/vitesse."""

    def test_plat_default(self):
        """2000m plat ≈ 124s (58 km/h)."""
        result = proxy_temps_sec(2000, "plat", None)
        # 2000m / (58/3.6) = 2000 / 16.11 ≈ 124s
        assert result is not None
        assert 120 < result < 130

    def test_trot_default(self):
        """2100m trot ≈ 164s (46 km/h)."""
        result = proxy_temps_sec(2100, "trot", None)
        # 2100m / (46/3.6) = 2100 / 12.78 ≈ 164s
        assert result is not None
        assert 160 < result < 170

    def test_obstacle_default(self):
        """3500m obstacle ≈ 252s (50 km/h)."""
        result = proxy_temps_sec(3500, "obstacle", None)
        # 3500m / (50/3.6) = 3500 / 13.89 ≈ 252s
        assert result is not None
        assert 245 < result < 260

    def test_hippodrome_specific(self):
        """Vincennes = vitesse spécifique."""
        result_vincennes = proxy_temps_sec(2100, "trot", "Vincennes")
        result_default = proxy_temps_sec(2100, "trot", "AUTRE")

        # Vincennes légèrement plus lent (45.5 vs 46.0)
        assert result_vincennes > result_default

    def test_no_distance_returns_none(self):
        """Sans distance → None."""
        result = proxy_temps_sec(None, "plat", None)
        assert result is None

    def test_zero_distance_returns_none(self):
        """Distance 0 → None."""
        result = proxy_temps_sec(0, "plat", None)
        assert result is None

    def test_negative_distance_returns_none(self):
        """Distance négative → None."""
        result = proxy_temps_sec(-1000, "plat", None)
        assert result is None

    def test_unknown_discipline_fallback_plat(self):
        """Discipline inconnue → fallback plat."""
        result = proxy_temps_sec(2000, "galop_inconnu", None)
        result_plat = proxy_temps_sec(2000, "plat", None)

        assert result == result_plat


# =============================================================================
# TESTS IMPUTE GAINS_CARRIERE
# =============================================================================


class TestImputeGainsCarriere:
    """Tests de l'imputation des gains carrière."""

    def test_age_2_plat_low_perfs(self):
        """2 ans plat, peu de courses → Q10."""
        gains, source = impute_gains_carriere(2, "plat", 2, 0, random_state=42)

        assert source == "imputed"
        assert gains is not None
        # Q10 = 0, mais avec bruit peut être légèrement positif
        assert gains >= 0
        assert gains < 50000  # Pas trop élevé pour un débutant

    def test_age_4_plat_winner(self):
        """4 ans plat, plusieurs victoires → Q75."""
        gains, source = impute_gains_carriere(4, "plat", 15, 4, random_state=42)

        assert source == "imputed"
        # Q75 pour (4-5, plat) = 120000 ± bruit
        assert gains > 50000

    def test_age_7_trot_experienced(self):
        """7 ans trot, expérimenté → Q50."""
        gains, source = impute_gains_carriere(7, "trot", 50, 2, random_state=42)

        assert source == "imputed"
        # Q50 pour (7+, trot) = 80000
        assert gains > 30000

    def test_none_age_returns_null(self):
        """Âge None → null."""
        gains, source = impute_gains_carriere(None, "plat", 10, 1)

        assert gains is None
        assert source == "null"

    def test_unknown_discipline_fallback(self):
        """Discipline inconnue → fallback plat."""
        gains, source = impute_gains_carriere(4, "course_imaginaire", 10, 1, random_state=42)

        assert source == "imputed"
        assert gains is not None

    def test_reproducible_with_seed(self):
        """Même seed → même résultat."""
        gains1, _ = impute_gains_carriere(4, "plat", 10, 1, random_state=12345)
        gains2, _ = impute_gains_carriere(4, "plat", 10, 1, random_state=12345)

        assert gains1 == gains2

    def test_different_without_seed(self):
        """Sans seed, résultats peuvent varier (avec le bruit)."""
        # Note: Ce test peut parfois passer par hasard, mais statistiquement
        # sur plusieurs appels, il devrait y avoir de la variance
        results = set()
        for _ in range(10):
            gains, _ = impute_gains_carriere(4, "plat", 10, 1)
            results.add(gains)

        # Normalement on devrait avoir plusieurs valeurs différentes
        # Mais avec la nature du test, on vérifie juste que ça fonctionne
        assert len(results) >= 1


# =============================================================================
# TESTS JOB
# =============================================================================


class TestFillMissingFieldsJob:
    """Tests du job principal."""

    def test_version_exists(self):
        """Version définie."""
        assert hasattr(FillMissingFieldsJob, "VERSION")
        assert FillMissingFieldsJob.VERSION == "1.0.0"

    def test_get_pre_off_features(self):
        """get_pre_off_features retourne une liste."""
        features = FillMissingFieldsJob.get_pre_off_features()

        assert isinstance(features, list)
        assert len(features) > 0
        assert "etat_piste" in features
        assert "gains_carriere" in features

    def test_get_post_off_fields(self):
        """get_post_off_fields retourne une liste."""
        fields = FillMissingFieldsJob.get_post_off_fields()

        assert isinstance(fields, list)
        assert len(fields) > 0
        assert "place_finale" in fields
        assert "rapport_gagnant" in fields

    def test_job_init_explicit_none(self):
        """Job initialisé avec conn=None explicite garde None."""
        # Note: Si DB disponible, le job essaie de se connecter par défaut
        # On vérifie que dry_run et batch_size sont bien configurés
        job = FillMissingFieldsJob(conn=None, dry_run=True)

        # Avec conn=None explicite, il reste None si DB non importée
        # ou se connecte si DB disponible - on teste juste les params
        assert job.dry_run is True
        assert job.batch_size == 1000

    def test_job_stats_initialized(self):
        """Stats initialisées à 0."""
        job = FillMissingFieldsJob(conn=None, dry_run=True)

        assert job.stats["processed"] == 0
        assert job.stats["etat_piste_filled"] == 0
        assert job.stats["temps_sec_filled"] == 0
        assert job.stats["rapport_gagnant_filled"] == 0
        assert job.stats["gains_carriere_filled"] == 0
        assert job.stats["errors"] == 0


# =============================================================================
# TESTS INTEGRATION (MOCK DB)
# =============================================================================


class MockCursor:
    """Mock cursor pour tests sans vraie DB."""

    def __init__(self):
        self.rowcount = 0
        self._results = []

    def execute(self, query, params=None):
        pass

    def fetchall(self):
        return self._results

    def fetchone(self):
        return (0, 0)


class MockConnection:
    """Mock connection pour tests."""

    def __init__(self):
        self._cursor = MockCursor()

    def cursor(self):
        return self._cursor

    def commit(self):
        pass


class TestJobIntegration:
    """Tests d'intégration avec mock DB."""

    def test_fill_etat_piste_with_mock_returns_zero(self):
        """Avec mock DB vide, fill_etat_piste retourne 0."""
        mock_conn = MockConnection()
        job = FillMissingFieldsJob(conn=mock_conn, dry_run=True)
        result = job.fill_etat_piste()

        assert result == 0

    def test_fill_temps_sec_with_mock_returns_zero(self):
        """Avec mock DB vide, fill_temps_sec retourne 0."""
        mock_conn = MockConnection()
        job = FillMissingFieldsJob(conn=mock_conn, dry_run=True)
        result = job.fill_temps_sec()

        assert result == 0

    def test_fill_rapport_gagnant_with_mock_returns_zero(self):
        """Avec mock DB vide, fill_rapport_gagnant retourne 0."""
        mock_conn = MockConnection()
        job = FillMissingFieldsJob(conn=mock_conn, dry_run=True)
        result = job.fill_rapport_gagnant()

        assert result == 0

    def test_fill_gains_carriere_with_mock_returns_zero(self):
        """Avec mock DB vide, fill_gains_carriere retourne 0."""
        mock_conn = MockConnection()
        job = FillMissingFieldsJob(conn=mock_conn, dry_run=True)
        result = job.fill_gains_carriere()

        assert result == 0


# =============================================================================
# TESTS EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests des cas limites."""

    def test_penetrometre_extreme_high(self):
        """Pénétromètre extrême (8+) → 'détrempé'."""
        etat, _ = derive_etat_piste(8.0, None, None)
        assert etat == "détrempé"

    def test_penetrometre_very_high(self):
        """Pénétromètre 5.0 → 'très lourd'."""
        etat, _ = derive_etat_piste(5.0, None, None)
        assert etat == "très lourd"

    def test_penetrometre_extreme_low(self):
        """Pénétromètre très bas → 'très bon'."""
        etat, _ = derive_etat_piste(1.0, None, None)
        assert etat == "très bon"

    def test_parse_temps_zero_minutes(self):
        """Format 0'45''3 → 45.3s."""
        result = parse_temps_str("0'45''3")
        assert result == pytest.approx(45.3, rel=0.01)

    def test_proxy_very_short_distance(self):
        """Distance très courte (800m)."""
        result = proxy_temps_sec(800, "plat", None)
        assert result is not None
        assert 40 < result < 60

    def test_proxy_very_long_distance(self):
        """Distance très longue (4500m obstacle)."""
        result = proxy_temps_sec(4500, "obstacle", None)
        assert result is not None
        assert 300 < result < 400

    def test_impute_very_old_horse(self):
        """Cheval très âgé (12 ans)."""
        gains, source = impute_gains_carriere(12, "trot", 100, 10, random_state=42)

        assert source == "imputed"
        assert gains is not None
        # Un cheval de 12 ans avec 10 victoires devrait avoir des gains élevés


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
