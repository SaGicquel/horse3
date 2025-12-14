# -*- coding: utf-8 -*-
"""
Module de normalisation des noms de chevaux
Utilisé pour le matching PMU ↔ IFCE et la déduplication
"""

import re
import unicodedata


def normalize_name(name: str | None) -> str | None:
    """
    Normalisation universelle des noms de chevaux pour matching/déduplication.
    
    Étapes :
    1. Uppercase
    2. Suppression des accents/diacritiques
    3. Suppression du suffixe pays final " (XX)" collé au nom
    4. Suppression des caractères spéciaux : . ' ' - _
    5. Compactage des espaces multiples
    
    Exemples :
        "Élégant D'Avril" → "ELEGANT DAVRIL"
        "BLACK SAXON (FR)" → "BLACK SAXON"
        "L'As-du-Jour" → "LAS DU JOUR"
        "  Saint  Martin  " → "SAINT MARTIN"
    
    Args:
        name: Nom du cheval à normaliser
        
    Returns:
        Nom normalisé ou None si vide
    """
    if not name:
        return None
    
    # 1. Uppercase
    name = name.upper()
    
    # 2. Suppression accents/diacritiques
    # NFD = décompose les caractères accentués (é → e + accent)
    name = unicodedata.normalize('NFD', name)
    # Garde uniquement les caractères non "combining" (supprime les accents)
    name = ''.join(ch for ch in name if not unicodedata.combining(ch))
    
    # 3. Suppression suffixe pays " (XX)" en fin de chaîne
    # Exemples : " (FR)", " (GB)", " (USA)"
    name = re.sub(r'\s*\([A-Z]{2,3}\)\s*$', '', name)
    
    # 4. Suppression caractères spéciaux . ' ' - _
    name = name.replace('.', ' ')  # Point → espace pour garder séparation
    name = name.replace("'", '')
    name = name.replace('-', ' ')  # Tiret → espace pour garder séparation
    name = name.replace('_', ' ')
    
    # 5. Compactage espaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name if name else None


def extract_birth_year(date_str: str | None) -> int | None:
    """
    Extrait l'année de naissance depuis différents formats de date.
    
    Formats supportés :
        - "YYYY-MM-DD" → YYYY
        - "DD/MM/YYYY" → YYYY
        - "YYYYMMDD" → YYYY
        - "YYYY" → YYYY
    
    Args:
        date_str: Chaîne représentant une date
        
    Returns:
        Année (int) ou None si impossible à parser
    """
    if not date_str:
        return None
    
    date_str = str(date_str).strip()
    
    # Format YYYY-MM-DD ou YYYY/MM/DD
    match = re.match(r'^(\d{4})[-/]', date_str)
    if match:
        return int(match.group(1))
    
    # Format DD/MM/YYYY ou DD-MM-YYYY
    match = re.match(r'^\d{2}[-/](\d{2})[-/](\d{4})$', date_str)
    if match:
        return int(match.group(2))
    
    # Format YYYYMMDD
    if re.match(r'^\d{8}$', date_str):
        return int(date_str[:4])
    
    # Format YYYY seul
    if re.match(r'^\d{4}$', date_str):
        return int(date_str)
    
    return None


def normalize_country(country: str | None) -> str | None:
    """
    Normalise les codes pays pour faciliter le matching.
    
    Exemples :
        "france" → "FR"
        "ROYAUME-UNI" → "GB"
        "usa" → "US"
        
    Args:
        country: Nom ou code pays
        
    Returns:
        Code pays normalisé (2-3 lettres) ou None
    """
    if not country:
        return None
    
    country = country.strip().upper()
    
    # Mapping noms courants → codes ISO
    country_map = {
        'FRANCE': 'FR',
        'FRA': 'FR',
        'ROYAUME-UNI': 'GB',
        'UK': 'GB',
        'ANGLETERRE': 'GB',
        'GRANDE-BRETAGNE': 'GB',
        'IRLANDE': 'IE',
        'IRE': 'IE',
        'ALLEMAGNE': 'DE',
        'DEU': 'DE',
        'GER': 'DE',
        'ITALIE': 'IT',
        'ITA': 'IT',
        'ESPAGNE': 'ES',
        'ESP': 'ES',
        'BELGIQUE': 'BE',
        'BEL': 'BE',
        'PAYS-BAS': 'NL',
        'NLD': 'NL',
        'SUISSE': 'CH',
        'SUI': 'CH',
        'ETATS-UNIS': 'US',
        'USA': 'US',
        'AUSTRALIE': 'AU',
        'AUS': 'AU',
        'JAPON': 'JP',
        'JPN': 'JP',
    }
    
    # Chercher dans le mapping
    if country in country_map:
        return country_map[country]
    
    # Si déjà un code ISO (2-3 lettres), le garder
    if re.match(r'^[A-Z]{2,3}$', country):
        return country
    
    return country  # Retourner tel quel si pas reconnu


def normalize_sex(sex: str | None) -> str | None:
    """
    Normalise le sexe du cheval : H (hongre), M (mâle), F (femelle).
    
    Args:
        sex: Sexe du cheval (divers formats)
        
    Returns:
        'H', 'M', 'F' ou None
    """
    if not sex:
        return None
    
    sex = sex.strip().upper()
    
    if sex in ('H', 'HONGRE', 'GELDING'):
        return 'H'
    elif sex in ('M', 'MALE', 'MÂLE', 'STALLION', 'ETALON'):
        return 'M'
    elif sex in ('F', 'FEMELLE', 'FEMALE', 'MARE', 'JUMENT'):
        return 'F'
    
    return None


if __name__ == '__main__':
    # Tests rapides
    test_cases = [
        "Élégant D'Avril",
        "BLACK SAXON (FR)",
        "L'As-du-Jour",
        "  Saint  Martin  ",
        "jean-paul.martin",
        "Étoile_Des_Prés (GB)",
    ]
    
    print("Tests de normalisation :")
    print("-" * 60)
    for name in test_cases:
        normalized = normalize_name(name)
        print(f"{name:30} → {normalized}")
    
    print("\n" + "=" * 60)
    print("Tests extraction année :")
    print("-" * 60)
    dates = ["2018-04-27", "27/04/2018", "20180427", "2018", "invalide"]
    for d in dates:
        year = extract_birth_year(d)
        print(f"{d:15} → {year}")
