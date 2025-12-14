# -*- coding: utf-8 -*-
"""
Package d'enrichissement des données hippiques
Modules : normalisation, calculs, matching, migrations
"""

from enrichment.normalization import (
    normalize_name,
    extract_birth_year,
    normalize_country,
    normalize_sex
)

from enrichment.calculations import (
    parse_time_str,
    compute_reduction_km,
    compute_annual_gains,
    compute_total_gains,
    compute_records,
    format_reduction_km,
    is_valid_finish_for_gains,
    PerformanceRecord
)

from enrichment.matching import (
    HorseMatcher,
    MatchResult,
    levenshtein_distance
)

from enrichment.migrations import (
    run_migrations,
    column_exists,
    table_exists
)

__version__ = '1.0.0'
__all__ = [
    # Normalisation
    'normalize_name',
    'extract_birth_year',
    'normalize_country',
    'normalize_sex',
    # Calculs
    'parse_time_str',
    'compute_reduction_km',
    'compute_annual_gains',
    'compute_total_gains',
    'compute_records',
    'format_reduction_km',
    'is_valid_finish_for_gains',
    'PerformanceRecord',
    # Matching
    'HorseMatcher',
    'MatchResult',
    'levenshtein_distance',
    # Migrations
    'run_migrations',
    'column_exists',
    'table_exists',
]
