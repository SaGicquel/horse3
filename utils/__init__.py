# -*- coding: utf-8 -*-
"""
Utils module pour le projet Horse3
Contient les utilitaires communs pour les scrapers
"""

from .rate_limiter import AdaptiveRateLimiter
from .circuit_breaker import CircuitBreaker
from .scraper_common import (
    resolve_hippodrome_identifiers,
    normalize_name,
    normalize_race,
    map_sexe,
    get_json_safe,
    ScraperSession,
)

__all__ = [
    "AdaptiveRateLimiter",
    "CircuitBreaker",
    "resolve_hippodrome_identifiers",
    "normalize_name",
    "normalize_race",
    "map_sexe",
    "get_json_safe",
    "ScraperSession",
]
