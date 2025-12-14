# -*- coding: utf-8 -*-
"""
Normalisation des textes (noms chevaux, personnes, hippodromes).
Conservé côté Python pour ETL, doublé par core.normalize_name côté SQL.
"""
import re
import unicodedata
from typing import Optional


def strip_accents(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    return ''.join(ch for ch in s if not unicodedata.combining(ch))


def normalize_name(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = strip_accents(s)
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_country(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = strip_accents(s).upper().strip()
    if len(s) > 3:
        return s[:3]
    return s


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def birth_year_from_date(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    m = re.match(r"(\d{4})-\d{2}-\d{2}", date_str)
    if m:
        return int(m.group(1))
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", date_str)
    if m:
        return int(m.group(3))
    return None
