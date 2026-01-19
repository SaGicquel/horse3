# -*- coding: utf-8 -*-
"""
Module de calculs hippiques
Temps, réductions kilométriques, gains, records
"""

import re
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from datetime import date


@dataclass
class PerformanceRecord:
    """Record de performance (meilleur temps)"""

    reduction_km_sec: float
    date: str
    venue: str
    race_code: str
    discipline: str


def parse_time_str(time_str: str | None) -> float | None:
    """
    Parse diverses notations de temps hippiques vers secondes.

    Formats supportés :
        - "1'12\"8" → 72.8 secondes
        - "1'11\"" → 71.0 secondes
        - "1'11" → 71.0 secondes
        - "68.7" → 68.7 secondes
        - "1:12.8" → 72.8 secondes

    Args:
        time_str: Chaîne représentant un temps

    Returns:
        Temps en secondes (float) ou None si impossible à parser

    Examples:
        >>> parse_time_str("1'12\"8")
        72.8
        >>> parse_time_str("1'11")
        71.0
        >>> parse_time_str("68.7")
        68.7
    """
    if not time_str:
        return None

    time_str = str(time_str).strip()
    if not time_str:
        return None

    # Format "1'12"8" ou "1'12"" ou "1'12'" ou "1'12"
    # Groupe 1: minutes, Groupe 2: secondes entières, Groupe 3: dixièmes (optionnel)
    # Support de ' et ' (apostrophe normale et typographique)
    # Support de " et " (guillemet normal et typographique)
    match = re.match(r"^(\d+)['\u2019](\d+)[\"'\u2019\u201d]*(\d)?$", time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        tenths = int(match.group(3)) if match.group(3) else 0
        return minutes * 60 + seconds + tenths / 10.0

    # Format "1:12.8" ou "1:12"
    match = re.match(r"^(\d+):(\d+)(?:\.(\d))?$", time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        tenths = int(match.group(3)) if match.group(3) else 0
        return minutes * 60 + seconds + tenths / 10.0

    # Format décimal direct "68.7" ou "68,7"
    time_str_clean = time_str.replace(",", ".")
    try:
        return float(time_str_clean)
    except ValueError:
        pass

    return None


def compute_reduction_km(
    time_sec: float | None, distance_m: int | None, reduction_provided: float | None = None
) -> float | None:
    """
    Calcule la réduction kilométrique en secondes.

    Si fournie par l'API PMU, on l'utilise.
    Sinon : reduction_km_sec = (time_sec / distance_m) * 1000

    Args:
        time_sec: Temps de la course en secondes
        distance_m: Distance en mètres
        reduction_provided: Réduction fournie par l'API (optionnel, en secondes)

    Returns:
        Réduction kilométrique en secondes ou None

    Examples:
        >>> compute_reduction_km(72.8, 2400, None)
        30.333...
        >>> compute_reduction_km(None, 2400, 30.5)
        30.5
    """
    # Si fournie par l'API, priorité absolue
    if reduction_provided is not None:
        return float(reduction_provided)

    # Sinon calcul
    if time_sec is not None and distance_m is not None and distance_m > 0:
        return (time_sec / distance_m) * 1000.0

    return None


def compute_annual_gains(performances: List[Dict], year: int) -> float:
    """
    Calcule les gains annuels (€) pour une année donnée.

    Règles :
        - Somme des allocations où finish_status IN ('1','2','3','4','5')
        - Exclusion : 'DAI', 'NP', 'RET', 'TNP', 'DISQ', 'ARR', 'DIST', 'TOMB'
        - L'allocation est déjà partagée en cas d'ex-aequo (si l'API est correcte)

    Args:
        performances: Liste de dicts avec clés 'allocation_eur', 'finish_status', 'race_date'
        year: Année ciblée

    Returns:
        Total des gains en euros
    """
    total = 0.0

    for perf in performances:
        # Vérifier l'année
        race_date = perf.get("race_date", "")
        if not race_date or not race_date.startswith(str(year)):
            continue

        # Vérifier le statut d'arrivée
        finish = str(perf.get("finish_status", "")).upper().strip()
        if not finish:
            continue

        # Exclusions explicites
        excluded = {"DAI", "NP", "RET", "TNP", "DISQ", "ARR", "DIST", "TOMB", "NON PARTANT"}
        if finish in excluded:
            continue

        # Inclusion : places 1-5 (numérique ou texte)
        if finish in ("1", "2", "3", "4", "5"):
            allocation = perf.get("allocation_eur", 0)
            if allocation:
                total += float(allocation)

    return total


def compute_total_gains(performances: List[Dict]) -> float:
    """
    Calcule les gains totaux (€) sur toute la carrière.

    Args:
        performances: Liste de dicts avec clés 'allocation_eur', 'finish_status'

    Returns:
        Total des gains en euros
    """
    total = 0.0

    for perf in performances:
        finish = str(perf.get("finish_status", "")).upper().strip()
        if not finish:
            continue

        # Exclusions
        excluded = {"DAI", "NP", "RET", "TNP", "DISQ", "ARR", "DIST", "TOMB", "NON PARTANT"}
        if finish in excluded:
            continue

        # Places payées (1-5 généralement)
        if finish in ("1", "2", "3", "4", "5"):
            allocation = perf.get("allocation_eur", 0)
            if allocation:
                total += float(allocation)

    return total


def compute_records(
    performances: List[Dict],
) -> Tuple[Optional[PerformanceRecord], Optional[PerformanceRecord]]:
    """
    Calcule les records attelé et monté (meilleure réduction kilométrique).

    Args:
        performances: Liste de dicts avec clés :
            - reduction_km_sec
            - discipline ('attelé', 'monté', etc.)
            - race_date, venue, race_code

    Returns:
        Tuple (record_attele, record_monte) où chaque record est un PerformanceRecord ou None
    """
    record_attele: Optional[PerformanceRecord] = None
    record_monte: Optional[PerformanceRecord] = None

    for perf in performances:
        reduction = perf.get("reduction_km_sec")
        if reduction is None:
            continue

        reduction = float(reduction)
        discipline = str(perf.get("discipline", "")).lower().strip()

        # Record attelé
        if "attel" in discipline:
            if record_attele is None or reduction < record_attele.reduction_km_sec:
                record_attele = PerformanceRecord(
                    reduction_km_sec=reduction,
                    date=perf.get("race_date", ""),
                    venue=perf.get("venue", ""),
                    race_code=perf.get("race_code", ""),
                    discipline=discipline,
                )

        # Record monté
        elif "mont" in discipline or "plat" in discipline or "haies" in discipline:
            if record_monte is None or reduction < record_monte.reduction_km_sec:
                record_monte = PerformanceRecord(
                    reduction_km_sec=reduction,
                    date=perf.get("race_date", ""),
                    venue=perf.get("venue", ""),
                    race_code=perf.get("race_code", ""),
                    discipline=discipline,
                )

    return record_attele, record_monte


def format_reduction_km(reduction_sec: float) -> str:
    """
    Formate une réduction kilométrique en notation lisible.

    Args:
        reduction_sec: Réduction en secondes (ex: 30.5)

    Returns:
        Format "1'10\"5" si > 60s, sinon "30\"5"

    Examples:
        >>> format_reduction_km(72.8)
        "1'12\"8"
        >>> format_reduction_km(30.5)
        "30\"5"
    """
    if reduction_sec >= 60:
        minutes = int(reduction_sec // 60)
        seconds = reduction_sec % 60
        sec_int = int(seconds)
        tenths = int(round((seconds - sec_int) * 10))
        return f"{minutes}'{sec_int:02d}\"{tenths}"
    else:
        sec_int = int(reduction_sec)
        tenths = int(round((reduction_sec - sec_int) * 10))
        return f'{sec_int}"{tenths}'


def is_valid_finish_for_gains(finish_status: str) -> bool:
    """
    Détermine si un statut d'arrivée donne droit à des gains.

    Args:
        finish_status: Statut ('1', '2', 'DAI', 'NP', etc.)

    Returns:
        True si le cheval a terminé dans les places payées
    """
    finish = str(finish_status).upper().strip()

    # Exclusions explicites
    excluded = {"DAI", "NP", "RET", "TNP", "DISQ", "ARR", "DIST", "TOMB", "NON PARTANT", ""}
    if finish in excluded:
        return False

    # Places payées (généralement 1-5)
    return finish in ("1", "2", "3", "4", "5")


if __name__ == "__main__":
    # Tests rapides
    print("Tests parse_time_str :")
    print("-" * 60)
    test_times = ["1'12\"8", "1'11\"", "1'11", "68.7", "1:12.8", "invalide"]
    for t in test_times:
        parsed = parse_time_str(t)
        print(f"{t:15} → {parsed}")

    print("\n" + "=" * 60)
    print("Tests compute_reduction_km :")
    print("-" * 60)
    print(f"72.8s / 2400m = {compute_reduction_km(72.8, 2400):.2f}s/km")
    print(f"Fournie 30.5 = {compute_reduction_km(None, 2400, 30.5)}s/km")

    print("\n" + "=" * 60)
    print("Tests format_reduction_km :")
    print("-" * 60)
    print(f"72.8s → {format_reduction_km(72.8)}")
    print(f"30.5s → {format_reduction_km(30.5)}")

    print("\n" + "=" * 60)
    print("Tests gains :")
    print("-" * 60)
    perfs = [
        {"allocation_eur": 1000, "finish_status": "1", "race_date": "2025-10-15"},
        {"allocation_eur": 500, "finish_status": "2", "race_date": "2025-09-20"},
        {"allocation_eur": 300, "finish_status": "NP", "race_date": "2025-08-10"},
        {"allocation_eur": 200, "finish_status": "4", "race_date": "2024-12-05"},
    ]
    print(f"Gains 2025 : {compute_annual_gains(perfs, 2025)}€")
    print(f"Gains totaux : {compute_total_gains(perfs)}€")
