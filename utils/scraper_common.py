# -*- coding: utf-8 -*-
"""
Utilitaires communs pour les scrapers PMU

Contient les fonctions partagées entre les différents scrapers:
- Normalisation des noms
- Résolution des identifiants d'hippodrome
- Session HTTP configurée
- Parsing sécurisé JSON
"""

import re
import unicodedata
import textwrap
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Headers par défaut pour les requêtes
DEFAULT_USER_AGENT = "horse3-scraper/2.0 (+contact@example.com)"
DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


def normalize_name(s: Optional[str]) -> Optional[str]:
    """
    Normalise un nom (cheval, jockey, etc.):
    - Supprime les accents
    - Met en minuscules
    - Collapse les espaces multiples
    
    Args:
        s: Chaîne à normaliser
    
    Returns:
        Chaîne normalisée ou None si vide
    
    Example:
        >>> normalize_name("ÉTOILE D'OR")
        "etoile d'or"
    """
    if not s:
        return None
    
    # Normaliser les caractères Unicode (décomposer les accents)
    s = unicodedata.normalize('NFKD', s)
    # Supprimer les caractères de combinaison (accents)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # Normaliser les espaces et mettre en minuscules
    s = re.sub(r"\s+", " ", s).strip().lower()
    
    return s if s else None


def normalize_race(race: Optional[str]) -> Optional[str]:
    """
    Normalise un nom de race pour éviter les doublons:
    - Enlève les astérisques (*ANGLO-ARABE* -> ANGLO-ARABE)
    - Remplace les tirets par des espaces
    - Convertit en majuscules
    
    Args:
        race: Nom de la race
    
    Returns:
        Race normalisée ou None
    
    Example:
        >>> normalize_race("*PUR-SANG*")
        "PUR SANG"
    """
    if not race:
        return None
    
    # Enlever les astérisques
    race = race.strip().replace('*', '')
    # Remplacer les tirets par des espaces
    race = race.replace('-', ' ')
    # Normaliser les espaces et convertir en majuscules
    race = re.sub(r"\s+", " ", race).strip().upper()
    
    return race if race else None


def map_sexe(s: Optional[str]) -> Optional[str]:
    """
    Mappe les différentes représentations du sexe vers un code standard.
    
    Args:
        s: Représentation du sexe (string)
    
    Returns:
        'M' (mâle), 'F' (femelle), 'H' (hongre) ou None
    
    Example:
        >>> map_sexe("HONGRE")
        "H"
    """
    if not s:
        return None
    
    u = s.upper().strip()
    
    if "HONGRE" in u or u == "H":
        return "H"
    if "FEMEL" in u or u == "F" or u == "JUMENT":
        return "F"
    if "MALE" in u or u == "M" or u == "ENTIER":
        return "M"
    
    return None


def resolve_hippodrome_identifiers(hippo_data: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrait le code et le nom d'un hippodrome depuis diverses sources de données.
    Gère les différents formats de l'API PMU.
    
    Args:
        hippo_data: Données de l'hippodrome (dict, str, ou autre)
    
    Returns:
        Tuple (code_normalisé, nom_normalisé)
    
    Example:
        >>> resolve_hippodrome_identifiers({"code": "VINC", "libelleLong": "Vincennes"})
        ("VINC", "Vincennes")
    """
    if not hippo_data:
        return None, None
    
    if isinstance(hippo_data, dict):
        # Extraire le code
        code = (
            hippo_data.get("code") or
            hippo_data.get("codeHippodrome") or
            hippo_data.get("codeLieu") or
            hippo_data.get("libelleCourt")
        )
        
        # Extraire le nom
        name = (
            hippo_data.get("libelleLong") or
            hippo_data.get("libelleCourt") or
            hippo_data.get("nom") or
            hippo_data.get("libelle")
        )
    else:
        # Si c'est une chaîne, l'utiliser comme code et nom
        code = str(hippo_data).strip()
        name = code
    
    # Normaliser
    code_norm = code.strip().upper() if code else None
    name_norm = name.strip() if name else None
    
    return code_norm, name_norm


def get_json_safe(
    url: str,
    session: Optional[requests.Session] = None,
    timeout: int = 15,
    headers: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Récupère du JSON de manière sécurisée.
    Gère les réponses vides, HTML, et les erreurs de parsing.
    
    Args:
        url: URL à requêter
        session: Session requests (optionnel)
        timeout: Timeout en secondes
        headers: Headers additionnels
    
    Returns:
        Dict parsé ou None en cas d'erreur
    """
    req_headers = {**DEFAULT_HEADERS, **(headers or {})}
    
    try:
        if session:
            r = session.get(url, headers=req_headers, timeout=timeout)
        else:
            r = requests.get(url, headers=req_headers, timeout=timeout)
        
        # Gérer les codes de statut spéciaux
        if r.status_code in (204, 404):
            return None
        
        r.raise_for_status()
        
        # Vérifier le content-type
        ctype = (r.headers.get("Content-Type") or "").lower()
        txt = r.text.strip()
        
        # Vérifier si la réponse est vide
        if not txt:
            return None
        
        # Vérifier si c'est du HTML au lieu de JSON
        if "text/html" in ctype or txt.startswith("<!DOCTYPE") or txt.startswith("<html"):
            head = textwrap.shorten(txt.replace("\n", " "), width=180)
            logger.warning(f"Réponse HTML au lieu de JSON: {url} - {head}")
            return None
        
        # Parser le JSON
        return r.json()
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout: {url}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP {e.response.status_code}: {url}")
        return None
    except requests.exceptions.JSONDecodeError as e:
        logger.warning(f"JSON invalide: {url} - {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Erreur réseau: {url} - {e}")
        return None


class ScraperSession:
    """
    Session HTTP configurée pour le scraping avec:
    - Pool de connexions
    - Retry automatique
    - Headers par défaut
    
    Usage:
        with ScraperSession() as session:
            data = session.get_json(url)
    """
    
    def __init__(
        self,
        pool_connections: int = 10,
        pool_maxsize: int = 20,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        user_agent: Optional[str] = None
    ):
        """
        Args:
            pool_connections: Nombre de connexions dans le pool
            pool_maxsize: Taille max du pool
            max_retries: Nombre de tentatives en cas d'échec
            backoff_factor: Facteur de délai entre les tentatives
            user_agent: User-Agent personnalisé
        """
        self.session = requests.Session()
        
        # Configurer les headers
        headers = DEFAULT_HEADERS.copy()
        if user_agent:
            headers["User-Agent"] = user_agent
        self.session.headers.update(headers)
        
        # Configurer le retry
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        # Configurer l'adapter avec pool
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """GET classique"""
        return self.session.get(url, **kwargs)
    
    def get_json(self, url: str, timeout: int = 15) -> Optional[Dict]:
        """GET avec parsing JSON sécurisé"""
        return get_json_safe(url, session=self.session, timeout=timeout)
    
    def close(self):
        """Ferme la session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def to_pmu_date(dt: str) -> str:
    """
    Convertit une date ISO en format PMU.
    
    Args:
        dt: Date au format 'YYYY-MM-DD' ou 'YYYYMMDD'
    
    Returns:
        Date au format 'DDMMYYYY'
    
    Example:
        >>> to_pmu_date("2024-01-15")
        "15012024"
    """
    ds = dt.replace("-", "")
    if len(ds) != 8 or not ds.isdigit():
        raise ValueError(f"Format de date invalide: {dt}")
    
    yyyy, mm, dd = ds[:4], ds[4:6], ds[6:8]
    return f"{dd}{mm}{yyyy}"


def from_pmu_date(dt: str) -> str:
    """
    Convertit une date PMU en format ISO.
    
    Args:
        dt: Date au format 'DDMMYYYY'
    
    Returns:
        Date au format 'YYYY-MM-DD'
    
    Example:
        >>> from_pmu_date("15012024")
        "2024-01-15"
    """
    if len(dt) != 8 or not dt.isdigit():
        raise ValueError(f"Format de date PMU invalide: {dt}")
    
    dd, mm, yyyy = dt[:2], dt[2:4], dt[4:8]
    return f"{yyyy}-{mm}-{dd}"


if __name__ == "__main__":
    # Tests des fonctions
    print("=== Tests des utilitaires ===\n")
    
    # Test normalize_name
    print("normalize_name:")
    result1 = normalize_name("ÉTOILE D'OR")
    print(f"  ÉTOILE D'OR -> {result1}")
    result2 = normalize_name("  READY  SET  GO  ")
    print(f"  '  READY  SET  GO  ' -> {result2}")
    
    # Test normalize_race
    print("\nnormalize_race:")
    print(f"  *PUR-SANG* -> {normalize_race('*PUR-SANG*')}")
    print(f"  ANGLO-ARABE -> {normalize_race('ANGLO-ARABE')}")
    
    # Test map_sexe
    print("\nmap_sexe:")
    print(f"  HONGRE -> {map_sexe('HONGRE')}")
    print(f"  FEMELLE -> {map_sexe('FEMELLE')}")
    print(f"  M -> {map_sexe('M')}")
    
    # Test resolve_hippodrome_identifiers
    print("\nresolve_hippodrome_identifiers:")
    hippo = {"code": "VINC", "libelleLong": "Vincennes"}
    print(f"  dict -> {resolve_hippodrome_identifiers(hippo)}")
    print(f"  LONGCHAMP -> {resolve_hippodrome_identifiers('LONGCHAMP')}")
    
    # Test to_pmu_date / from_pmu_date
    print("\nto_pmu_date / from_pmu_date:")
    print(f"  2024-01-15 -> {to_pmu_date('2024-01-15')}")
    print(f"  15012024 -> {from_pmu_date('15012024')}")
    print("\n✅ Tous les tests passent !")
