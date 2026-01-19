# -*- coding: utf-8 -*-
"""
Scraper PMU COMPLET - Course du jour avec toutes les infos
Points clés:
- Scrape toutes les courses du jour automatiquement
- Récupère TOUTES les informations disponibles (participants + performances détaillées)
- Gère les doublons par nom + date de naissance
- Enrichit la table 'chevaux' avec données complètes
- Fallback online/offline pour robustesse
- VERSION MULTI-THREADÉE pour performance optimale (5-10x plus rapide)
- Rate limiting adaptatif et circuit breaker pour robustesse
"""

import time
import re
import unicodedata
import requests
import textwrap
import json
import logging
from datetime import datetime, date
from typing import Iterable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from db_connection import get_connection

# Import des utilitaires (optionnel, fallback si non disponible)
try:
    from utils.rate_limiter import AdaptiveRateLimiter
    from utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("[INFO] Module utils non disponible, mode legacy")

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Config
# ---------------------------

# Utilise PostgreSQL via db_connection.py
# DB_PATH = "data/database.db"  # Deprecated: SQLite remplacé par PostgreSQL

UA = "horse2-enricher/1.4 (+contact: youremail@example.com)"
COMMON_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.pmu.fr/turf/",
    "Connection": "keep-alive",
}

BASE = "https://online.turfinfo.api.pmu.fr/rest/client/7"
FALLBACK_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/7"

# tokens pour détecter une victoire selon 'place.abrege'
VICTORY_TOKENS = {"1", "1er", "V", "V1"}

# Multi-threading config - OPTIMISÉ POUR VITESSE MAX
MAX_WORKERS = 16  # Nombre de threads parallèles (doublé pour vitesse)
RATE_LIMIT_DELAY = 0.05  # Délai de base entre requêtes (réduit pour vitesse)

# Thread-safe lock pour les écritures DB
db_lock = threading.Lock()

# Rate limiter et circuit breaker globaux
if UTILS_AVAILABLE:
    rate_limiter = AdaptiveRateLimiter(
        min_delay=0.05, max_delay=5.0, backoff_factor=1.3, consecutive_success_threshold=5
    )
    circuit_breaker = CircuitBreaker(failure_threshold=10, success_threshold=3, reset_timeout=60.0)
    logger.info("✅ Rate limiter et circuit breaker activés")
else:
    rate_limiter = None
    circuit_breaker = None

# ---------------------------
# Utils
# ---------------------------


def to_pmu_date(dt: str) -> str:
    """
    'YYYY-MM-DD' ou 'YYYYMMDD' -> 'DDMMYYYY'
    """
    ds = dt.replace("-", "")
    if len(ds) != 8 or not ds.isdigit():
        raise ValueError(f"Bad date: {dt}")
    yyyy, mm, dd = ds[:4], ds[4:6], ds[6:8]
    return f"{dd}{mm}{yyyy}"


def norm(s: str | None) -> str | None:
    """
    Normalisation nom: strip accents, normalize apostrophes, collapse spaces, lower.
    """
    if not s:
        return None
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Normaliser les apostrophes (typographiques vers standard)
    s = s.replace("'", "'").replace("'", "'").replace("`", "'")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def normalize_race(race: str | None) -> str | None:
    """
    Normalise un nom de race pour éviter les doublons:
    - Enlève les astérisques (*ANGLO-ARABE* -> ANGLO-ARABE)
    - Remplace les tirets par des espaces (PUR-SANG -> PUR SANG)
    - Convertit en majuscules et enlève les espaces multiples
    """
    if not race:
        return None
    # Enlever les astérisques
    race = race.strip().replace("*", "")
    # Remplacer les tirets par des espaces
    race = race.replace("-", " ")
    # Normaliser les espaces et convertir en majuscules
    race = re.sub(r"\s+", " ", race).strip().upper()
    return race if race else None


def get_json(url: str, timeout=8):
    """
    GET JSON robuste avec rate limiting et circuit breaker.
    Évite JSONDecodeError si HTML/empty.
    Timeout réduit à 8s pour éviter les blocages.
    """
    # Vérifier le circuit breaker
    if circuit_breaker and not circuit_breaker.can_execute():
        logger.warning(f"Circuit breaker ouvert, requête ignorée: {url}")
        return None

    # Appliquer le rate limiting
    if rate_limiter:
        rate_limiter.wait()

    try:
        r = requests.get(url, headers=COMMON_HEADERS, timeout=timeout)

        if r.status_code in (204, 404):
            if rate_limiter:
                rate_limiter.success()
            return None

        r.raise_for_status()

        ctype = (r.headers.get("Content-Type") or "").lower()
        txt = r.text.strip()

        if not txt:
            if rate_limiter:
                rate_limiter.success()
            return None

        if "text/html" in ctype or txt.startswith("<!DOCTYPE html") or txt.startswith("<html"):
            head = textwrap.shorten(txt.replace("\n", " "), width=180)
            logger.warning(f"Non-JSON response ({r.status_code}) from {url}: {head}")
            if rate_limiter:
                rate_limiter.error(r.status_code)
            return None

        result = r.json()

        # Succès
        if rate_limiter:
            rate_limiter.success()
        if circuit_breaker:
            circuit_breaker.record_success()

        return result

    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP {e.response.status_code}: {url}")
        if rate_limiter:
            rate_limiter.error(e.response.status_code)
        if circuit_breaker:
            circuit_breaker.record_failure()
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout: {url}")
        if rate_limiter:
            rate_limiter.error()
        if circuit_breaker:
            circuit_breaker.record_failure()
        return None
    except requests.JSONDecodeError:
        head = textwrap.shorten(txt.replace("\n", " "), width=180)
        logger.warning(f"JSON parse failed for {url}: {head}")
        if rate_limiter:
            rate_limiter.error()
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue: {url} - {e}")
        if rate_limiter:
            rate_limiter.error()
        if circuit_breaker:
            circuit_breaker.record_failure()
        return None


# ---------------------------
# Découverte programme
# ---------------------------


def discover_reunions(date_iso: str) -> list[int]:
    """
    Liste des réunions (R1..R15) ayant des courses pour 'date_iso'.
    Essaie online puis offline en fallback.
    """
    d = to_pmu_date(date_iso)
    res = []
    for base in (BASE, FALLBACK_BASE):
        for r in range(1, 16):
            data = get_json(f"{base}/programme/{d}/R{r}")
            if data and data.get("courses"):
                res.append(r)
            # Le rate limiting est géré par get_json si disponible
            if not rate_limiter:
                time.sleep(0.35)
        if res:
            break
    return sorted(set(res))


def discover_courses(date_iso: str, reunion: int) -> list[int]:
    """
    Liste des numéros de courses pour une réunion.
    """
    d = to_pmu_date(date_iso)
    for base in (BASE, FALLBACK_BASE):
        data = get_json(f"{base}/programme/{d}/R{reunion}")
        if data and data.get("courses"):
            nums = []
            for c in data["courses"]:
                n = c.get("numOrdre") or c.get("numExterne") or c.get("num")
                if isinstance(n, int):
                    nums.append(n)
            return sorted(set(nums)) or list(range(1, 13))
    return []


# ---------------------------
# Mapping / helpers
# ---------------------------


def map_sexe(s: str | None) -> str | None:
    if not s:
        return None
    u = s.upper()
    if "HONGRE" in u:
        return "H"
    if "FEMEL" in u:
        return "F"
    if "MALE" in u or u == "M":
        return "M"
    return None


def _is_win(place: dict) -> bool:
    p = place.get("place")
    if p == 1:  # numérique
        return True
    abr = str(place.get("abrege") or "").strip()
    if abr in VICTORY_TOKENS:
        return True
    if abr.replace(" ", "").lower() in {"1e", "1er"}:
        return True
    if place.get("estVainqueur") is True:
        return True
    return False


def _abbr_token(place: dict) -> str:
    """
    Token musique: place numérique/abrégé sinon '%s'
    """
    abr = str(place.get("abrege") or "").strip()
    p = place.get("place")
    if isinstance(p, int) and p > 0:
        return str(p)
    if abr:
        return abr
    return "%s"


def _parse_hist_date_any(d):
    """
    Essaie plusieurs formats rencontrés; renvoie (iso, year) ou (None, None)
    """
    if not d:
        return None, None
    s = str(d)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"):
        try:
            dt = datetime.strptime(s, fmt).date()
            return dt.isoformat(), dt.year
        except ValueError:
            pass
    return None, None


def resolve_hippodrome_identifiers(hippo_data) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract a normalized (code, name) tuple from heterogeneous hippodrome payloads.
    """
    if not hippo_data:
        return None, None

    if isinstance(hippo_data, dict):
        code = (
            hippo_data.get("code")
            or hippo_data.get("codeHippodrome")
            or hippo_data.get("codeLieu")
            or hippo_data.get("libelleCourt")
        )
        name = (
            hippo_data.get("libelleLong")
            or hippo_data.get("libelleCourt")
            or hippo_data.get("nom")
            or hippo_data.get("libelle")
        )
    else:
        code = str(hippo_data).strip()
        name = code

    code_norm = code.strip().upper() if code else None
    name_norm = name.strip() if name else None
    return code_norm, name_norm


def _hippo_str(item) -> str:
    """
    Renvoie un identifiant lisible d'hippodrome depuis divers schémas possibles.
    """
    hippo = item
    if isinstance(item, dict):
        hippo = item.get("hippodrome") or item.get("hippo") or item
    code, name = resolve_hippodrome_identifiers(hippo)
    if code:
        return code
    if name:
        return name.upper()
    return ""


def extract_rapports_detailles(rapports_json: str) -> dict:
    """
    Extrait les rapports détaillés depuis le JSON des rapports PMU.
    Format attendu : liste d'objets avec typePari (SIMPLE_GAGNANT, COUPLE, TRIO, QUARTE_PLUS, QUINTE_PLUS, etc.)
    Retourne un dict avec rapport_quarte, rapport_quinte, rapport_multi, rapport_pick5, montant_enjeux_total
    """
    if not rapports_json:
        return {}

    try:
        rapports_data = json.loads(rapports_json)
    except:
        return {}

    result = {}

    # Format : liste d'objets avec typePari
    if isinstance(rapports_data, list):
        total_enjeux = 0

        for item in rapports_data:
            if not isinstance(item, dict):
                continue

            type_pari = item.get("typePari") or item.get("type") or ""
            rapports = item.get("rapports") or []

            # Extraire le dividende (premier élément de la liste rapports généralement)
            dividende = None
            if rapports and len(rapports) > 0:
                premier_rapport = rapports[0]
                if isinstance(premier_rapport, dict):
                    dividende = premier_rapport.get("dividende") or premier_rapport.get(
                        "dividendePourUnEuro"
                    )
                    if not dividende:
                        dividende = premier_rapport.get("rapport")

            # Mapping des types de paris
            if "QUARTE" in type_pari.upper() or type_pari == "QUARTE_PLUS":
                if dividende:
                    result["rapport_quarte"] = float(dividende)

            elif "QUINTE" in type_pari.upper() or type_pari == "QUINTE_PLUS":
                if dividende:
                    result["rapport_quinte"] = float(dividende)

            elif "MULTI" in type_pari.upper():
                if dividende:
                    result["rapport_multi"] = float(dividende)

            elif "PICK5" in type_pari.upper() or "PICK_5" in type_pari.upper():
                if dividende:
                    result["rapport_pick5"] = float(dividende)

            # Calculer montant enjeux (si disponible)
            montant = item.get("montantEnjeux") or item.get("enjeux") or item.get("montantTotal")
            if montant:
                if isinstance(montant, (int, float)):
                    total_enjeux += montant

        if total_enjeux > 0:
            result["montant_enjeux_total"] = total_enjeux

    # Format alternatif : dict direct (ancien format)
    elif isinstance(rapports_data, dict):
        # Quarté+
        quarte = rapports_data.get("rapportQuarte") or rapports_data.get("quarte")
        if quarte:
            if isinstance(quarte, dict):
                result["rapport_quarte"] = quarte.get("rapport") or quarte.get("dividende")
                result["montant_enjeux_quarte"] = quarte.get("montantEnjeux") or quarte.get(
                    "enjeux"
                )
            else:
                result["rapport_quarte"] = float(quarte) if quarte else None

        # Quinté+
        quinte = rapports_data.get("rapportQuinte") or rapports_data.get("quinte")
        if quinte:
            if isinstance(quinte, dict):
                result["rapport_quinte"] = quinte.get("rapport") or quinte.get("dividende")
                result["montant_enjeux_quinte"] = quinte.get("montantEnjeux") or quinte.get(
                    "enjeux"
                )
            else:
                result["rapport_quinte"] = float(quinte) if quinte else None

        # Multi
        multi = rapports_data.get("rapportMulti") or rapports_data.get("multi")
        if multi:
            if isinstance(multi, dict):
                result["rapport_multi"] = multi.get("rapport") or multi.get("dividende")
                result["montant_enjeux_multi"] = multi.get("montantEnjeux") or multi.get("enjeux")
            else:
                result["rapport_multi"] = float(multi) if multi else None

        # Pick5
        pick5 = rapports_data.get("rapportPick5") or rapports_data.get("pick5")
        if pick5:
            if isinstance(pick5, dict):
                result["rapport_pick5"] = pick5.get("rapport") or pick5.get("dividende")
                result["montant_enjeux_pick5"] = pick5.get("montantEnjeux") or pick5.get("enjeux")
            else:
                result["rapport_pick5"] = float(pick5) if pick5 else None

        # Montant total
        total_enjeux = (
            rapports_data.get("montantEnjeuxTotal")
            or rapports_data.get("enjeuxTotal")
            or rapports_data.get("montantTotal")
        )
        if total_enjeux:
            result["montant_enjeux_total"] = float(total_enjeux) if total_enjeux else 0

    return result


def extract_info_hippodrome(hippo_data) -> dict:
    """
    Extrait les informations enrichies sur l'hippodrome.
    """
    result = {}

    if isinstance(hippo_data, dict):
        result["pays_hippodrome"] = (
            hippo_data.get("pays") or hippo_data.get("country") or hippo_data.get("codePays")
        )
        result["region_hippodrome"] = (
            hippo_data.get("region") or hippo_data.get("departement") or hippo_data.get("zone")
        )

    return result


def extract_info_personnel(personnel_data) -> dict:
    """
    Extrait les informations enrichies sur le personnel (driver/jockey/entraîneur).
    Retourne couleurs casaque et IDs PMU.
    """
    result = {}

    if isinstance(personnel_data, dict):
        # ID PMU
        result["id_pmu"] = (
            personnel_data.get("id") or personnel_data.get("idPmu") or personnel_data.get("numPmu")
        )

        # Couleurs casaque
        casaque = personnel_data.get("casaque") or personnel_data.get("colors") or {}
        if isinstance(casaque, dict):
            result["couleurs_casaque"] = casaque.get("description") or casaque.get("libelle")
        elif isinstance(casaque, str):
            result["couleurs_casaque"] = casaque

    return result


def calculate_vitesse(distance_m: int, temps_sec: float) -> float:
    """
    Calcule la vitesse moyenne en km/h.
    """
    if not distance_m or not temps_sec or temps_sec <= 0:
        return None

    distance_km = distance_m / 1000.0
    temps_heures = temps_sec / 3600.0

    return distance_km / temps_heures


def calculate_ecarts(participants_list: list, place_finale: int, temps_sec: float) -> dict:
    """
    Calcule les écarts avec le 1er et le précédent.
    Retourne: {ecart_premier: X, ecart_precedent: Y}
    """
    result = {}

    if not participants_list or not place_finale or not temps_sec:
        return result

    # Trouver le 1er
    premier = None
    precedent = None

    for p in participants_list:
        place = p.get("ordreArrivee") or p.get("place") or p.get("rang")
        if place == 1:
            premier_temps = p.get("tempsSecondes") or p.get("tempsSec")
            if premier_temps:
                premier = premier_temps
                break

    # Trouver le précédent (place - 1)
    if place_finale > 1:
        for p in participants_list:
            place = p.get("ordreArrivee") or p.get("place") or p.get("rang")
            if place == (place_finale - 1):
                precedent_temps = p.get("tempsSecondes") or p.get("tempsSec")
                if precedent_temps:
                    precedent = precedent_temps
                    break

    # Calculer les écarts
    if premier:
        result["ecart_premier"] = temps_sec - premier

    if precedent:
        result["ecart_precedent"] = temps_sec - precedent

    return result


def extract_allocations_detaillees(course_data: dict, place_finale: int) -> dict:
    """
    Extrait les allocations détaillées (1er, 2ème, 3ème) depuis les données de course.
    """
    result = {}

    if not course_data:
        return result

    # Chercher dans allocations ou repartition
    repartition = course_data.get("repartitionAllocation") or course_data.get("allocation") or {}

    if isinstance(repartition, dict):
        result["allocation_premier"] = (
            repartition.get("premier") or repartition.get("1") or repartition.get("1er")
        )
        result["allocation_deuxieme"] = (
            repartition.get("deuxieme") or repartition.get("2") or repartition.get("2eme")
        )
        result["allocation_troisieme"] = (
            repartition.get("troisieme") or repartition.get("3") or repartition.get("3eme")
        )

    # Calculer le pourcentage de l'allocation totale
    total_allocation = course_data.get("montantPrix") or course_data.get("allocation")
    if total_allocation and place_finale and place_finale <= 3:
        allocation_place = result.get(
            f"allocation_{['premier', 'deuxieme', 'troisieme'][place_finale - 1]}"
        )
        if allocation_place:
            result["pourcentage_allocation_course"] = (allocation_place / total_allocation) * 100.0

    return result


def _race_key(item: dict, reunion: int | None, course: int | None, fallback_date_iso: str):
    """
    Clé stable: DATE|R{r}|C{c}|HIPPO; la date tombe en fallback sur 'date_iso' si absente.
    """
    d_iso, _ = _parse_hist_date_any(item.get("date") or item.get("dateReunion") or item.get("jour"))
    if not d_iso:
        d_iso = fallback_date_iso  # Fallback vital
    hippo = _hippo_str(item) or "%s"
    r = item.get("numReunion") or reunion or "%s"
    c = item.get("numCourse") or item.get("course") or course or "%s"
    return f"{d_iso}|R{r}|C{c}|{hippo}"


# ---------------------------
# Upserts BDD - Structure simplifiée (2 tables)
# ---------------------------


def upsert_cheval(
    cur,
    nom,
    date_naissance=None,
    sexe=None,
    robe=None,
    race=None,
    pays_naissance=None,
    entraineur=None,
    driver=None,
    num_pmu=None,
    nom_pere=None,
    nom_mere=None,
    proprietaire=None,
    eleveur=None,
    id_cheval_pmu=None,
    url_fiche_pmu=None,
):
    """
    Upsert complet d'un cheval avec gestion des doublons.
    Adapté au schéma actuel de la table 'chevaux' avec contrainte unique sur (nom_cheval, an_naissance, sexe_cheval).
    """
    key = norm(nom)
    race = normalize_race(race)

    # Extraire l'année de naissance depuis date_naissance
    an_naissance = None
    if date_naissance:
        try:
            an_naissance = int(date_naissance.split("-")[0])
        except:
            pass

    # Mapper sexe PMU vers format DB (1 caractère: M, F, H)
    sexe_db = None
    if sexe:
        sexe_str = sexe.upper() if isinstance(sexe, str) else str(sexe).upper()
        if "HONGRE" in sexe_str:
            sexe_db = "H"
        elif "FEMELLE" in sexe_str or sexe_str == "F":
            sexe_db = "F"
        elif "MALE" in sexe_str or sexe_str == "M" or "ENTIER" in sexe_str:
            sexe_db = "M"
        else:
            # Premier caractère en dernier recours
            sexe_db = sexe_str[:1] if sexe_str else None

    # Créer un pattern de recherche qui tolère les variations d'apostrophes
    # Remplacer les apostrophes par un pattern LIKE qui matche tous les types
    search_pattern = key.replace("'", "_") if key else None

    # Recherche par nom_cheval (avec tolérance apostrophes via LIKE pattern)
    cur.execute(
        """
        SELECT id_cheval, nom_cheval, sexe_cheval, robe, nom_pere, nom_mere,
               proprietaire, eleveur, code_pmu, an_naissance
        FROM chevaux
        WHERE LOWER(nom_cheval) LIKE %s
        LIMIT 1
    """,
        (search_pattern,),
    )

    row = cur.fetchone()

    if row:
        # Mise à jour des infos existantes (si nouvelles valeurs fournies)
        id_ch = row[0]
        updates = []
        params = []

        # Champs de base - mise à jour si fourni et différent
        if sexe_db and row[2] != sexe_db:
            updates.append("sexe_cheval=%s")
            params.append(sexe_db)
        if robe and row[3] != robe:
            updates.append("robe=%s")
            params.append(robe)
        if nom_pere and row[4] != nom_pere:
            updates.append("nom_pere=%s")
            params.append(nom_pere)
        if nom_mere and row[5] != nom_mere:
            updates.append("nom_mere=%s")
            params.append(nom_mere)
        if proprietaire and row[6] != proprietaire:
            updates.append("proprietaire=%s")
            params.append(proprietaire)
        if eleveur and row[7] != eleveur:
            updates.append("eleveur=%s")
            params.append(eleveur)
        if id_cheval_pmu and row[8] != id_cheval_pmu:
            updates.append("code_pmu=%s")
            params.append(id_cheval_pmu)
        # Ne pas mettre à jour an_naissance si déjà défini (évite violation contrainte unique)
        if an_naissance and row[9] is None:
            updates.append("an_naissance=%s")
            params.append(an_naissance)

        if updates:
            updates.append("updated_at=CURRENT_TIMESTAMP")
            params.append(id_ch)
            try:
                cur.execute("SAVEPOINT upsert_update_sp")
                cur.execute(f"UPDATE chevaux SET {', '.join(updates)} WHERE id_cheval=%s", params)
                cur.execute("RELEASE SAVEPOINT upsert_update_sp")
            except Exception:
                # Rollback et ignorer les erreurs de mise à jour
                try:
                    cur.execute("ROLLBACK TO SAVEPOINT upsert_update_sp")
                except:
                    pass

        return id_ch
    else:
        # Nouveau cheval - Vérifier d'abord s'il existe avec même nom + parents
        # (évite les doublons même si l'année diffère dans l'API)
        try:
            cur.execute(
                """
                SELECT id_cheval FROM chevaux
                WHERE nom_cheval = %s
                AND COALESCE(nom_pere, '') = COALESCE(%s, '')
                AND COALESCE(nom_mere, '') = COALESCE(%s, '')
                LIMIT 1
            """,
                (nom, nom_pere, nom_mere),
            )
            existing = cur.fetchone()

            if existing:
                # Cheval existe déjà - mettre à jour les infos manquantes
                id_ch = existing[0]
                cur.execute(
                    """
                    UPDATE chevaux SET
                        sexe_cheval = COALESCE(sexe_cheval, %s),
                        robe = COALESCE(robe, %s),
                        proprietaire = COALESCE(proprietaire, %s),
                        eleveur = COALESCE(eleveur, %s),
                        code_pmu = COALESCE(code_pmu, %s),
                        an_naissance = COALESCE(an_naissance, %s),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id_cheval = %s
                """,
                    (sexe_db, robe, proprietaire, eleveur, id_cheval_pmu, an_naissance, id_ch),
                )
                return id_ch

            # Vraiment nouveau cheval - insérer
            cur.execute("SAVEPOINT upsert_cheval_sp")
            cur.execute(
                """
                INSERT INTO chevaux (nom_cheval, sexe_cheval, robe, nom_pere, nom_mere,
                                     proprietaire, eleveur, code_pmu, an_naissance,
                                     created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING id_cheval
            """,
                (
                    nom,
                    sexe_db,
                    robe,
                    nom_pere,
                    nom_mere,
                    proprietaire,
                    eleveur,
                    id_cheval_pmu,
                    an_naissance,
                ),
            )

            result = cur.fetchone()
            cur.execute("RELEASE SAVEPOINT upsert_cheval_sp")
            return result[0] if result else None
        except Exception as e:
            # Rollback au SAVEPOINT pour nettoyer la transaction
            try:
                cur.execute("ROLLBACK TO SAVEPOINT upsert_cheval_sp")
            except:
                pass
            # Essayer de récupérer l'ID existant par nom ou code_pmu
            try:
                cur.execute(
                    """
                    SELECT id_cheval FROM chevaux
                    WHERE LOWER(nom_cheval) LIKE %s OR code_pmu=%s
                    LIMIT 1
                """,
                    (search_pattern, id_cheval_pmu),
                )
                row = cur.fetchone()
                return row[0] if row else None
            except:
                return None


def upsert_cheval_counts(cur, nom, perf):
    """
    Met à jour les compteurs à partir d'agrégats (TOTAL & 2025) calculés depuis l'historique.

    NOTE: Fonction simplifiée - les colonnes de statistiques (nombre_courses_total, etc.)
    n'existent pas dans le schéma actuel de la table 'chevaux'.
    Les statistiques de performances sont stockées dans cheval_courses_seen à la place.

    Cette fonction fait un simple passthrough pour le moment.
    Pour activer les compteurs, ajouter les colonnes à la table chevaux.
    """
    # Pour l'instant, on ne fait rien car les colonnes n'existent pas
    # Les statistiques sont calculées à partir de cheval_courses_seen
    pass


# ---------------------------
# Fetch par course
# ---------------------------


def fetch_participants(date_iso: str, reunion: int, course: int):
    d = to_pmu_date(date_iso)
    for base in (BASE, FALLBACK_BASE):
        url = f"{base}/programme/{d}/R{reunion}/C{course}/participants"
        data = get_json(url)
        if data and data.get("participants"):
            return data["participants"]
    return []


def fetch_performances(
    date_iso: str, reunion: int, course: int, cur, hippodrome_actuel: str, max_hist: int = 12
):
    """
    - Construit 'musique' (N premiers tokens) depuis l'historique
    - N'enregistre que la course ACTUELLE (pas l'historique) dans 'cheval_courses_seen'
    - Agrège nb_courses / nb_victoires (all & 2025) depuis l'historique complet
    Renvoie: nom_norm -> { 'musique', 'nbc', 'nbv', 'nbc_2025', 'nbv_2025', 'dernieres_perfs' }
    """
    d = to_pmu_date(date_iso)
    data = None
    for base in (BASE, FALLBACK_BASE):
        url = f"{base}/programme/{d}/R{reunion}/C{course}/performances-detaillees/pretty"
        data = get_json(url)
        if data and data.get("participants"):
            break
    if not data:
        return {}

    result = {}
    for p in data.get("participants", []):
        nom = p.get("nomCheval")
        if not nom:
            continue
        key_nom = norm(nom)
        hist = p.get("coursesCourues", []) or []

        tokens = []
        nb_courses_total = 0
        nb_victoires_total = 0
        nb_courses_2025 = 0
        nb_victoires_2025 = 0
        dernieres_perfs = []

        # Parcourir l'historique pour construire la musique et les stats
        for idx, h in enumerate(hist):
            # Trouver la place du cheval dans cette course (itsHim = true)
            place_info = None
            place_num = None

            participants = h.get("participants", [])
            for part in participants:
                if part.get("itsHim"):
                    place_data = part.get("place", {}) or {}
                    place_num = place_data.get("place")
                    place_info = place_data
                    break

            # Si pas trouvé dans participants, essayer la place directe (ancien format)
            if not place_info:
                place_info = h.get("place", {}) or {}
                if not place_num:
                    place_num = place_info.get("place")

            # musique: N premiers tokens
            if idx < max_hist:
                tokens.append(_abbr_token(place_info))

            # Compter les courses et victoires
            nb_courses_total += 1
            is_win = _is_win(place_info)
            if is_win:
                nb_victoires_total += 1

            # Compter pour 2025
            _iso, year = _parse_hist_date_any(
                h.get("date") or h.get("dateReunion") or h.get("jour")
            )
            if year == 2025:
                nb_courses_2025 += 1
                if is_win:
                    nb_victoires_2025 += 1

            # Stocker les dernières performances (5 max) avec détails
            if idx < 5:
                hippo = _hippo_str(h) or "%s"

                # Extraire place lisible
                place_str = str(place_num) if place_num else _abbr_token(place_info)

                dernieres_perfs.append(
                    {
                        "date": _iso or "%s",
                        "hippodrome": hippo,
                        "place": place_str,
                        "is_win": is_win,
                    }
                )

        result[key_nom] = {
            "musique": "".join(tokens) or None,
            "nbc": nb_courses_total,
            "nbv": nb_victoires_total,
            "nbc_2025": nb_courses_2025,
            "nbv_2025": nb_victoires_2025,
            "dernieres_perfs": dernieres_perfs,
        }
    return result


def enrich_from_course(cur, date_iso: str, reunion: int, course: int, sleep_s=None):
    """
    Version SIMPLIFIÉE - Écrit directement dans les 2 tables principales :
    - chevaux : infos de base sur le cheval
    - cheval_courses_seen : toutes les infos de la course
    """
    if sleep_s is None:
        sleep_s = RATE_LIMIT_DELAY

    # ========================================
    # 1. RÉCUPÉRER LES DÉTAILS DE LA COURSE
    # ========================================
    d = to_pmu_date(date_iso)
    course_data = None

    for base in (BASE, FALLBACK_BASE):
        url = f"{base}/programme/{d}/R{reunion}/C{course}"
        course_data = get_json(url)
        if course_data:
            break

    if not course_data:
        print(f"  ⚠️  Pas de données pour R{reunion}C{course}")
        return

    # Récupérer les données de réunion (pour météo, pénétromètre, etc.)
    reunion_data = {}
    for base in (BASE, FALLBACK_BASE):
        url_reunion = f"{base}/programme/{d}"
        programme_data = get_json(url_reunion)
        if programme_data and "programme" in programme_data:
            reunions = programme_data["programme"].get("reunions", [])
            # Trouver la bonne réunion
            for r in reunions:
                if r.get("numOfficiel") == reunion or r.get("numExterneReunion") == reunion:
                    reunion_data = r
                    break
            if reunion_data:
                break

    # ========================================
    # 2. EXTRAIRE INFOS RÉUNION & COURSE
    # ========================================
    hippo = course_data.get("hippodrome") or reunion_data.get("hippodrome") or {}

    # Hippodrome
    venue_code, venue_name = resolve_hippodrome_identifiers(hippo)
    hippo_info = extract_info_hippodrome(hippo)

    hippodrome_actuel = venue_code or "UNKNOWN"
    pays_hippodrome = hippo_info.get("pays_hippodrome")
    region_hippodrome = hippo_info.get("region_hippodrome")

    # Météo & piste (réunion)
    weather = reunion_data.get("meteo")
    track_condition = course_data.get("etatPiste") or reunion_data.get("etatPiste")

    # Infos course
    discipline = course_data.get("discipline")
    specialty = course_data.get("specialite")
    distance_m = course_data.get("distance")
    start_method = course_data.get("typeDepart")
    rope_side = course_data.get("corde")
    track_surface = course_data.get("typePiste")
    total_allocation = course_data.get("montantPrix") or course_data.get("allocation")
    race_name = course_data.get("libelle") or course_data.get("libelleLong")
    race_conditions = course_data.get("conditions")
    race_type = course_data.get("typeCourse")
    start_time = course_data.get("heureDepart")
    pmu_reunion_id = reunion_data.get("numOfficiel")
    pmu_course_id = course_data.get("numOrdre")  # Correction: numOrdre au lieu de numOfficiel

    # ========================================
    # PHASE 1 - NOUVEAUX CHAMPS
    # ========================================

    # Allocations détaillées
    allocation_premier = course_data.get("montantOffert1er")
    allocation_deuxieme = course_data.get("montantOffert2eme")
    allocation_troisieme = course_data.get("montantOffert3eme")

    # Commentaires
    commentaire_avant_data = course_data.get("commentaireAvantCourse")
    if isinstance(commentaire_avant_data, dict):
        commentaire_avant_course = commentaire_avant_data.get("texte")
    elif isinstance(commentaire_avant_data, str):
        commentaire_avant_course = commentaire_avant_data
    else:
        commentaire_avant_course = None

    commentaire_apres_data = course_data.get("commentaireApresCourse")
    if isinstance(commentaire_apres_data, dict):
        commentaire_apres_course = commentaire_apres_data.get("texte")
    elif isinstance(commentaire_apres_data, str):
        commentaire_apres_course = commentaire_apres_data
    else:
        commentaire_apres_course = None

    # Classe de course
    classe_course = course_data.get("categorieParticularite")

    # Prix course (nom complet)
    prix_course = course_data.get("libelle") or race_name

    # Météo détaillée
    meteo_data = reunion_data.get("meteo", {}) or {}

    if isinstance(meteo_data, dict):
        meteo_code = meteo_data.get("nebulositeCode") or meteo_data.get("code")
        temperature_c = meteo_data.get("temperature")
        vent_kmh = (
            meteo_data.get("forceVent") or meteo_data.get("vitesseVent") or meteo_data.get("vent")
        )
    elif isinstance(meteo_data, str):
        meteo_code = meteo_data
        temperature_c = None
        vent_kmh = None
    else:
        meteo_code = None
        temperature_c = None
        vent_kmh = None

    penetrometre = reunion_data.get("penetrometre")
    profil_piste = reunion_data.get("profilPiste")

    # ========================================
    # PHASE 5 - DONNÉES COURSE SUPPLÉMENTAIRES (NOUVEAU!)
    # ========================================
    duree_course = course_data.get("dureeCourse")
    course_trackee = course_data.get("courseTrackee", False)
    replay_disponible = course_data.get("replayDisponible", False)

    # État piste
    etat_piste = track_condition
    meteo = (
        weather
        if isinstance(weather, str)
        else (weather.get("libelle") if isinstance(weather, dict) else None)
    )

    # Incidents de course
    incidents = course_data.get("incidents")
    incidents_json = json.dumps(incidents, ensure_ascii=False) if incidents else None

    # ========================================
    # 3. RÉCUPÉRER LES RAPPORTS (si course terminée)
    # ========================================
    rapports_json = None
    rapports_detailles = {}
    for base in (BASE, FALLBACK_BASE):
        url_rapports = f"{base}/programme/{d}/R{reunion}/C{course}/rapports-definitifs"
        rapports_data = get_json(url_rapports)
        if rapports_data:
            rapports_json = json.dumps(rapports_data, ensure_ascii=False)
            rapports_detailles = extract_rapports_detailles(rapports_json)
            break

    # ========================================
    # 4. RÉCUPÉRER LES PARTANTS
    # ========================================
    plist = fetch_participants(date_iso, reunion, course)
    if not plist:
        print(f"  ⚠️  Pas de partants pour R{reunion}C{course}")
        return

    # performances -> remplit cheval_courses_seen avec la course actuelle uniquement
    perf_map = fetch_performances(date_iso, reunion, course, cur, hippodrome_actuel, max_hist=12)

    # ========================================
    # 5. PARCOURIR PARTICIPANTS ET ÉCRIRE
    # ========================================
    participants = plist

    if not participants:
        print(f"  ⚠️  Aucun participant pour R{reunion}C{course}")
        return

    print(f"  📋 {len(participants)} participants trouvés")

    # ========================================
    # 6. TRAITER CHAQUE PARTICIPANT
    # ========================================
    for p in participants:
        nom = p.get("nom", "").strip()
        if not nom:
            continue

        nom_normalize = norm(nom)
        # Format du race_key : date|R#|C#|HIPPODROME (compatible avec migration)
        race_key = f"{date_iso}|R{reunion}|C{course}|{hippodrome_actuel}"

        # ID cheval PMU et URL fiche
        id_cheval_pmu = p.get("idCheval")
        url_fiche_pmu = (
            f"https://www.pmu.fr/turf/chevaux/{id_cheval_pmu}" if id_cheval_pmu else None
        )

        # Sexe & robe (peuvent être des dict)
        sexe = p.get("sexe")
        if isinstance(sexe, dict):
            sexe = sexe.get("libelleCourt") or sexe.get("code")

        robe_raw = p.get("robe")
        if isinstance(robe_raw, dict):
            robe = robe_raw.get("libelleCourt") or robe_raw.get("libelleLong")
        else:
            robe = robe_raw

        # Personnel (peuvent être des dict ou des strings)
        entraineur_data = p.get("entraineur")
        if isinstance(entraineur_data, dict):
            entraineur = entraineur_data.get("nom") or entraineur_data.get("libelleCourt")
            info_entraineur = extract_info_personnel(entraineur_data)
            id_entraineur_pmu = info_entraineur.get("id_pmu")
        else:
            entraineur = entraineur_data if isinstance(entraineur_data, str) else None
            id_entraineur_pmu = None

        driver_data = p.get("driver") or p.get("jockey")
        if isinstance(driver_data, dict):
            driver = driver_data.get("nom") or driver_data.get("libelleCourt")
            info_driver = extract_info_personnel(driver_data)
            id_driver_pmu = info_driver.get("id_pmu")
            couleurs_casaque_driver = info_driver.get("couleurs_casaque")
        else:
            driver = driver_data if isinstance(driver_data, str) else None
            id_driver_pmu = None
            couleurs_casaque_driver = None

        # Pour jockey séparé (si présent)
        jockey_data = p.get("jockey")
        if isinstance(jockey_data, dict) and jockey_data != driver_data:
            info_jockey = extract_info_personnel(jockey_data)
            id_jockey_pmu = info_jockey.get("id_pmu")
            couleurs_casaque_jockey = info_jockey.get("couleurs_casaque")
        else:
            id_jockey_pmu = id_driver_pmu
            couleurs_casaque_jockey = couleurs_casaque_driver

        proprietaire = p.get("proprietaire")
        if isinstance(proprietaire, dict):
            proprietaire = proprietaire.get("nom") or proprietaire.get("libelleCourt")

        eleveur = p.get("eleveur")
        if isinstance(eleveur, dict):
            eleveur = eleveur.get("nom") or eleveur.get("libelleCourt")

        # Calcul date de naissance depuis âge
        age = p.get("age")
        date_naissance = None
        if age and isinstance(age, int):
            annee_course = int(date_iso.split("-")[0])
            annee_naissance = annee_course - age
            date_naissance = f"{annee_naissance}-01-01"

        # Race et pays
        race_cheval = p.get("race")
        pays_origine = p.get("paysNaissance") or p.get("paysOrigine") or p.get("pays")

        # Pedigree
        nom_pere = p.get("nomPere") or p.get("pere")
        nom_mere = p.get("nomMere") or p.get("mere")

        # Numéro de dossard/corde
        numero_dossard = p.get("numPmu") or p.get("numero") or p.get("numeroCorde")
        horse_num_pmu = p.get("numPmu")

        # Statistiques carrière
        musique_api = p.get("musique")
        nombre_courses = p.get("nombreCourses")
        nombre_victoires = p.get("nombreVictoires")

        # ========================================
        # PHASE 2 - STATISTIQUES DÉTAILLÉES
        # ========================================
        nombre_places = p.get("nombrePlaces")
        nombre_places_second = p.get("nombrePlacesSecond")
        nombre_places_troisieme = p.get("nombrePlacesTroisieme")

        # Gains détaillés
        gains_carriere = None
        gains_victoires = None
        gains_place = None
        gains_annee_en_cours = None
        gains_annee_precedente = None

        gains_participant = p.get("gainsParticipant")
        if gains_participant and isinstance(gains_participant, dict):
            gains_carriere = gains_participant.get("gainsCarriere")
            gains_victoires = gains_participant.get("gainsVictoires")
            gains_place = gains_participant.get("gainsPlace")
            gains_annee_en_cours = gains_participant.get("gainsAnneeEnCours")
            gains_annee_precedente = gains_participant.get("gainsAnneePrecedente")

        # Indicateurs participant
        indicateur_inedit = p.get("indicateurInedit")
        driver_change = p.get("driverChange")
        jument_pleine = p.get("jumentPleine")
        incident = p.get("incident")

        # Équipement et handicap
        deferrage = p.get("deferrage") or p.get("deferre")
        oeilleres = p.get("oeilleres")
        handicap_distance = p.get("handicapDistance")
        poids = p.get("poids")

        # Construire le champ equipment
        equipment_parts = []
        if oeilleres:
            equipment_parts.append(f"Œillères: {oeilleres}")
        if deferrage:
            equipment_parts.append(f"Déferré: {deferrage}")
        equipment = ", ".join(equipment_parts) if equipment_parts else None

        # Cotes
        cote_prob_matin = p.get("coteProbableMatin") or p.get("coteMatin")
        cote_finale = p.get("coteDirecte") or p.get("rapportDirect")

        # ========================================
        # PHASE 3 - COTES DÉTAILLÉES ET TENDANCES (NOUVEAU!)
        # ========================================
        # Dernier rapport direct (cote actuelle avec tendance)
        rapport_direct = p.get("dernierRapportDirect", {}) or {}
        if rapport_direct:
            # Utiliser cote directe de rapport_direct si pas déjà définie
            if not cote_finale:
                cote_finale = rapport_direct.get("rapport")
            tendance_cote = (rapport_direct.get("indicateurTendance") or "").strip()
            amplitude_tendance = rapport_direct.get("nombreIndicateurTendance")
            est_favori = rapport_direct.get("favoris", False)
            grosse_prise = rapport_direct.get("grossePrise", False)
        else:
            tendance_cote = None
            amplitude_tendance = None
            est_favori = False
            grosse_prise = False

        # Rapport de référence (cote matin/veille)
        rapport_ref = p.get("dernierRapportReference", {}) or {}
        if rapport_ref:
            cote_reference = rapport_ref.get("rapport")
            # Utiliser comme cote matin si pas définie
            if not cote_prob_matin:
                cote_prob_matin = cote_reference
        else:
            cote_reference = None

        # ========================================
        # PHASE 4 - INDICATEURS STRATÉGIQUES (NOUVEAU!)
        # ========================================
        avis_entraineur = p.get("avisEntraineur")  # NEUTRE, POSITIF, NEGATIF
        allure = p.get("allure")  # TROT, GALOP...
        statut_participant = p.get("statut")  # PARTANT, NON_PARTANT...
        supplement = p.get("supplement", 0)
        engagement = p.get("engagement", False)
        poids_condition_monte_change = p.get("poidsConditionMonteChange", False)
        url_casaque = p.get("urlCasaque")

        # Commentaire après course au niveau participant (peut être plus précis)
        commentaire_participant = p.get("commentaireApresCourse", {})
        if isinstance(commentaire_participant, dict):
            commentaire_participant_texte = commentaire_participant.get("texte")
            source_commentaire = commentaire_participant.get("source")
        else:
            commentaire_participant_texte = None
            source_commentaire = None

        # Résultats (si disponibles)
        place_actuelle = p.get("ordreArrivee") or p.get("place") or p.get("rang")
        place_finale = (
            int(place_actuelle) if place_actuelle and str(place_actuelle).isdigit() else None
        )

        # Statut d'arrivée
        statut_arrivee = p.get("statutArrivee")  # Arrivé, DAI, NP, etc.
        is_non_runner = 1 if (statut_arrivee == "NP") else 0
        is_disqualified = 1 if ("disqualif" in (statut_arrivee or "").lower()) else 0

        # Temps et réduction kilométrique
        temps_str = p.get("tempsObtenu")
        temps_sec = p.get("tempsSecondes")
        reduction_km = p.get("reductionKilometrique")

        # Écarts
        ecarts = p.get("ecarts")

        # Calculer les écarts détaillés et vitesses
        ecarts_detail = calculate_ecarts(participants, place_finale, temps_sec)
        ecart_premier = ecarts_detail.get("ecart_premier")
        ecart_precedent = ecarts_detail.get("ecart_precedent")

        # Calculer les vitesses
        vitesse_moyenne = (
            calculate_vitesse(distance_m, temps_sec) if distance_m and temps_sec else None
        )
        vitesse_fin_course = vitesse_moyenne  # On peut améliorer plus tard avec données détaillées

        # Extraire allocations détaillées
        allocations_detail = (
            extract_allocations_detaillees(course_data, place_finale) if place_finale else {}
        )

        # Gains pour cette course
        gains_course = p.get("gainsCourse") or p.get("gainsEpreuve")

        # Notes
        observations = p.get("observations") or p.get("remarques")

        # Nombre de partants total dans la course
        num_runners = (
            course_data.get("nombreDeclaresPartants")
            or course_data.get("nombrePartants")
            or len(plist)
        )

        # Musique depuis perf_map ou API
        musique_cheval = None
        pf = perf_map.get(nom_normalize)
        if pf:
            musique_cheval = musique_api or pf.get("musique")
        elif musique_api:
            musique_cheval = musique_api

        # is_win
        is_win = 1 if (place_finale == 1) else 0

        # ========================================
        # A. UPSERT dans chevaux avec pedigree + URL (récupère id_cheval)
        # ========================================
        id_cheval = upsert_cheval(
            cur,
            nom,
            date_naissance,
            sexe,
            robe,
            race_cheval,
            pays_origine,
            entraineur,
            driver,
            num_pmu=horse_num_pmu,
            nom_pere=nom_pere,
            nom_mere=nom_mere,
            proprietaire=proprietaire,
            eleveur=eleveur,
            id_cheval_pmu=id_cheval_pmu,
            url_fiche_pmu=url_fiche_pmu,
        )

        # Compteurs/musique depuis perf_map
        pf = perf_map.get(nom_normalize)
        if pf:
            # La musique a déjà été calculée ci-dessus
            upsert_cheval_counts(cur, nom, pf)

        # ========================================
        # B. ÉCRIRE dans cheval_courses_seen avec TOUTES les infos + id_cheval
        # ========================================
        # Extraire les valeurs des nouvelles données
        rapport_quarte = rapports_detailles.get("rapport_quarte")
        rapport_quinte = rapports_detailles.get("rapport_quinte")
        rapport_multi = rapports_detailles.get("rapport_multi")
        rapport_pick5 = rapports_detailles.get("rapport_pick5")
        montant_enjeux_total = rapports_detailles.get("montant_enjeux_total")

        # Les allocations sont déjà extraites avant la boucle (lignes 817-819)
        # allocation_premier, allocation_deuxieme, allocation_troisieme sont déjà définis
        pourcentage_allocation_course = allocations_detail.get("pourcentage_allocation_course")

        # Nom complet du prix (avec sponsors si disponible)
        prix_course = race_name  # On utilise déjà race_name, peut être enrichi plus tard

        cur.execute(
            """
            INSERT INTO cheval_courses_seen (
                id_cheval, nom_norm, race_key, annee, is_win,
                reunion_numero, course_numero, hippodrome_code, hippodrome_nom,
                discipline, specialite, distance_m, type_depart, corde, type_piste,
                allocation_totale, course_nom, conditions_course, type_course,
                heure_depart, etat_piste, meteo,
                numero_dossard, driver_jockey, entraineur, proprietaire, eleveur,
                age, sexe, robe, race, pays_naissance, poids_kg,
                nom_pere, nom_mere,
                equipement, deferrage, handicap_distance,
                cote_matin, cote_finale,
                place_finale, statut_arrivee, temps_str, temps_sec, reduction_km_sec, ecarts,
                gains_course, gains_carriere, nombre_partants,
                non_partant, disqualifie, observations,
                musique,
                pmu_reunion_id, pmu_course_id,
                rapports_json, incidents_json,
                rapport_quarte, rapport_quinte, rapport_multi, rapport_pick5, montant_enjeux_total,
                allocation_premier, allocation_deuxieme, allocation_troisieme, pourcentage_allocation_course,
                pays_hippodrome, region_hippodrome,
                ecart_premier, ecart_precedent, vitesse_moyenne, vitesse_fin_course,
                couleurs_casaque_driver, couleurs_casaque_jockey,
                id_driver_pmu, id_jockey_pmu, id_entraineur_pmu,
                prix_course,
                commentaire_avant_course, commentaire_apres_course, classe_course,
                meteo_code, temperature_c, vent_kmh, penetrometre, profil_piste,
                nombre_places, nombre_places_second, nombre_places_troisieme,
                gains_victoires, gains_place, gains_annee_en_cours, gains_annee_precedente,
                indicateur_inedit, driver_change, jument_pleine, incident,
                cote_reference, tendance_cote, amplitude_tendance, est_favori, grosse_prise,
                avis_entraineur, allure, statut_participant, supplement, engagement,
                poids_condition_monte_change, url_casaque, source_commentaire,
                duree_course, course_trackee, replay_disponible
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s,
                %s, %s,
                %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (nom_norm, race_key) DO UPDATE SET
                id_cheval                    = COALESCE(EXCLUDED.id_cheval, cheval_courses_seen.id_cheval),
                annee                        = COALESCE(EXCLUDED.annee, cheval_courses_seen.annee),
                is_win                      = COALESCE(EXCLUDED.is_win, cheval_courses_seen.is_win),
                reunion_numero              = COALESCE(EXCLUDED.reunion_numero, cheval_courses_seen.reunion_numero),
                course_numero               = COALESCE(EXCLUDED.course_numero, cheval_courses_seen.course_numero),
                hippodrome_code             = COALESCE(EXCLUDED.hippodrome_code, cheval_courses_seen.hippodrome_code),
                hippodrome_nom              = COALESCE(EXCLUDED.hippodrome_nom, cheval_courses_seen.hippodrome_nom),
                discipline                  = COALESCE(EXCLUDED.discipline, cheval_courses_seen.discipline),
                specialite                  = COALESCE(EXCLUDED.specialite, cheval_courses_seen.specialite),
                distance_m                  = COALESCE(EXCLUDED.distance_m, cheval_courses_seen.distance_m),
                type_depart                 = COALESCE(EXCLUDED.type_depart, cheval_courses_seen.type_depart),
                corde                       = COALESCE(EXCLUDED.corde, cheval_courses_seen.corde),
                type_piste                  = COALESCE(EXCLUDED.type_piste, cheval_courses_seen.type_piste),
                allocation_totale           = COALESCE(EXCLUDED.allocation_totale, cheval_courses_seen.allocation_totale),
                course_nom                  = COALESCE(EXCLUDED.course_nom, cheval_courses_seen.course_nom),
                conditions_course           = COALESCE(EXCLUDED.conditions_course, cheval_courses_seen.conditions_course),
                type_course                 = COALESCE(EXCLUDED.type_course, cheval_courses_seen.type_course),
                heure_depart                = COALESCE(EXCLUDED.heure_depart, cheval_courses_seen.heure_depart),
                etat_piste                  = COALESCE(EXCLUDED.etat_piste, cheval_courses_seen.etat_piste),
                meteo                       = COALESCE(EXCLUDED.meteo, cheval_courses_seen.meteo),
                numero_dossard              = COALESCE(EXCLUDED.numero_dossard, cheval_courses_seen.numero_dossard),
                driver_jockey               = COALESCE(EXCLUDED.driver_jockey, cheval_courses_seen.driver_jockey),
                entraineur                  = COALESCE(EXCLUDED.entraineur, cheval_courses_seen.entraineur),
                proprietaire                = COALESCE(EXCLUDED.proprietaire, cheval_courses_seen.proprietaire),
                eleveur                     = COALESCE(EXCLUDED.eleveur, cheval_courses_seen.eleveur),
                age                         = COALESCE(EXCLUDED.age, cheval_courses_seen.age),
                sexe                        = COALESCE(EXCLUDED.sexe, cheval_courses_seen.sexe),
                robe                        = COALESCE(EXCLUDED.robe, cheval_courses_seen.robe),
                race                        = COALESCE(EXCLUDED.race, cheval_courses_seen.race),
                pays_naissance              = COALESCE(EXCLUDED.pays_naissance, cheval_courses_seen.pays_naissance),
                poids_kg                    = COALESCE(EXCLUDED.poids_kg, cheval_courses_seen.poids_kg),
                nom_pere                    = COALESCE(EXCLUDED.nom_pere, cheval_courses_seen.nom_pere),
                nom_mere                    = COALESCE(EXCLUDED.nom_mere, cheval_courses_seen.nom_mere),
                equipement                  = COALESCE(EXCLUDED.equipement, cheval_courses_seen.equipement),
                deferrage                   = COALESCE(EXCLUDED.deferrage, cheval_courses_seen.deferrage),
                handicap_distance           = COALESCE(EXCLUDED.handicap_distance, cheval_courses_seen.handicap_distance),
                cote_matin                  = COALESCE(EXCLUDED.cote_matin, cheval_courses_seen.cote_matin),
                cote_finale                 = COALESCE(EXCLUDED.cote_finale, cheval_courses_seen.cote_finale),
                place_finale                = COALESCE(EXCLUDED.place_finale, cheval_courses_seen.place_finale),
                statut_arrivee              = COALESCE(EXCLUDED.statut_arrivee, cheval_courses_seen.statut_arrivee),
                temps_str                   = COALESCE(EXCLUDED.temps_str, cheval_courses_seen.temps_str),
                temps_sec                   = COALESCE(EXCLUDED.temps_sec, cheval_courses_seen.temps_sec),
                reduction_km_sec            = COALESCE(EXCLUDED.reduction_km_sec, cheval_courses_seen.reduction_km_sec),
                ecarts                      = COALESCE(EXCLUDED.ecarts, cheval_courses_seen.ecarts),
                gains_course                = COALESCE(EXCLUDED.gains_course, cheval_courses_seen.gains_course),
                gains_carriere              = COALESCE(EXCLUDED.gains_carriere, cheval_courses_seen.gains_carriere),
                nombre_partants             = COALESCE(EXCLUDED.nombre_partants, cheval_courses_seen.nombre_partants),
                non_partant                 = COALESCE(EXCLUDED.non_partant, cheval_courses_seen.non_partant),
                disqualifie                 = COALESCE(EXCLUDED.disqualifie, cheval_courses_seen.disqualifie),
                observations                = COALESCE(EXCLUDED.observations, cheval_courses_seen.observations),
                musique                     = COALESCE(EXCLUDED.musique, cheval_courses_seen.musique),
                pmu_reunion_id              = COALESCE(EXCLUDED.pmu_reunion_id, cheval_courses_seen.pmu_reunion_id),
                pmu_course_id               = COALESCE(EXCLUDED.pmu_course_id, cheval_courses_seen.pmu_course_id),
                rapports_json               = COALESCE(EXCLUDED.rapports_json, cheval_courses_seen.rapports_json),
                incidents_json              = COALESCE(EXCLUDED.incidents_json, cheval_courses_seen.incidents_json),
                rapport_quarte              = COALESCE(EXCLUDED.rapport_quarte, cheval_courses_seen.rapport_quarte),
                rapport_quinte              = COALESCE(EXCLUDED.rapport_quinte, cheval_courses_seen.rapport_quinte),
                rapport_multi               = COALESCE(EXCLUDED.rapport_multi, cheval_courses_seen.rapport_multi),
                rapport_pick5               = COALESCE(EXCLUDED.rapport_pick5, cheval_courses_seen.rapport_pick5),
                montant_enjeux_total        = COALESCE(EXCLUDED.montant_enjeux_total, cheval_courses_seen.montant_enjeux_total),
                allocation_premier          = COALESCE(EXCLUDED.allocation_premier, cheval_courses_seen.allocation_premier),
                allocation_deuxieme         = COALESCE(EXCLUDED.allocation_deuxieme, cheval_courses_seen.allocation_deuxieme),
                allocation_troisieme        = COALESCE(EXCLUDED.allocation_troisieme, cheval_courses_seen.allocation_troisieme),
                pourcentage_allocation_course = COALESCE(EXCLUDED.pourcentage_allocation_course, cheval_courses_seen.pourcentage_allocation_course),
                pays_hippodrome             = COALESCE(EXCLUDED.pays_hippodrome, cheval_courses_seen.pays_hippodrome),
                region_hippodrome           = COALESCE(EXCLUDED.region_hippodrome, cheval_courses_seen.region_hippodrome),
                ecart_premier               = COALESCE(EXCLUDED.ecart_premier, cheval_courses_seen.ecart_premier),
                ecart_precedent             = COALESCE(EXCLUDED.ecart_precedent, cheval_courses_seen.ecart_precedent),
                vitesse_moyenne             = COALESCE(EXCLUDED.vitesse_moyenne, cheval_courses_seen.vitesse_moyenne),
                vitesse_fin_course          = COALESCE(EXCLUDED.vitesse_fin_course, cheval_courses_seen.vitesse_fin_course),
                couleurs_casaque_driver     = COALESCE(EXCLUDED.couleurs_casaque_driver, cheval_courses_seen.couleurs_casaque_driver),
                couleurs_casaque_jockey     = COALESCE(EXCLUDED.couleurs_casaque_jockey, cheval_courses_seen.couleurs_casaque_jockey),
                id_driver_pmu               = COALESCE(EXCLUDED.id_driver_pmu, cheval_courses_seen.id_driver_pmu),
                id_jockey_pmu               = COALESCE(EXCLUDED.id_jockey_pmu, cheval_courses_seen.id_jockey_pmu),
                id_entraineur_pmu           = COALESCE(EXCLUDED.id_entraineur_pmu, cheval_courses_seen.id_entraineur_pmu),
                prix_course                 = COALESCE(EXCLUDED.prix_course, cheval_courses_seen.prix_course),
                commentaire_avant_course    = COALESCE(EXCLUDED.commentaire_avant_course, cheval_courses_seen.commentaire_avant_course),
                commentaire_apres_course    = COALESCE(EXCLUDED.commentaire_apres_course, cheval_courses_seen.commentaire_apres_course),
                classe_course               = COALESCE(EXCLUDED.classe_course, cheval_courses_seen.classe_course),
                meteo_code                  = COALESCE(EXCLUDED.meteo_code, cheval_courses_seen.meteo_code),
                temperature_c               = COALESCE(EXCLUDED.temperature_c, cheval_courses_seen.temperature_c),
                vent_kmh                    = COALESCE(EXCLUDED.vent_kmh, cheval_courses_seen.vent_kmh),
                penetrometre                = COALESCE(EXCLUDED.penetrometre, cheval_courses_seen.penetrometre),
                profil_piste                = COALESCE(EXCLUDED.profil_piste, cheval_courses_seen.profil_piste),
                nombre_places               = COALESCE(EXCLUDED.nombre_places, cheval_courses_seen.nombre_places),
                nombre_places_second        = COALESCE(EXCLUDED.nombre_places_second, cheval_courses_seen.nombre_places_second),
                nombre_places_troisieme     = COALESCE(EXCLUDED.nombre_places_troisieme, cheval_courses_seen.nombre_places_troisieme),
                gains_victoires             = COALESCE(EXCLUDED.gains_victoires, cheval_courses_seen.gains_victoires),
                gains_place                 = COALESCE(EXCLUDED.gains_place, cheval_courses_seen.gains_place),
                gains_annee_en_cours        = COALESCE(EXCLUDED.gains_annee_en_cours, cheval_courses_seen.gains_annee_en_cours),
                gains_annee_precedente      = COALESCE(EXCLUDED.gains_annee_precedente, cheval_courses_seen.gains_annee_precedente),
                indicateur_inedit           = COALESCE(EXCLUDED.indicateur_inedit, cheval_courses_seen.indicateur_inedit),
                driver_change               = COALESCE(EXCLUDED.driver_change, cheval_courses_seen.driver_change),
                jument_pleine               = COALESCE(EXCLUDED.jument_pleine, cheval_courses_seen.jument_pleine),
                incident                    = COALESCE(EXCLUDED.incident, cheval_courses_seen.incident),
                cote_reference              = COALESCE(EXCLUDED.cote_reference, cheval_courses_seen.cote_reference),
                tendance_cote               = COALESCE(EXCLUDED.tendance_cote, cheval_courses_seen.tendance_cote),
                amplitude_tendance          = COALESCE(EXCLUDED.amplitude_tendance, cheval_courses_seen.amplitude_tendance),
                est_favori                  = COALESCE(EXCLUDED.est_favori, cheval_courses_seen.est_favori),
                grosse_prise                = COALESCE(EXCLUDED.grosse_prise, cheval_courses_seen.grosse_prise),
                avis_entraineur             = COALESCE(EXCLUDED.avis_entraineur, cheval_courses_seen.avis_entraineur),
                allure                      = COALESCE(EXCLUDED.allure, cheval_courses_seen.allure),
                statut_participant          = COALESCE(EXCLUDED.statut_participant, cheval_courses_seen.statut_participant),
                supplement                  = COALESCE(EXCLUDED.supplement, cheval_courses_seen.supplement),
                engagement                  = COALESCE(EXCLUDED.engagement, cheval_courses_seen.engagement),
                poids_condition_monte_change = COALESCE(EXCLUDED.poids_condition_monte_change, cheval_courses_seen.poids_condition_monte_change),
                url_casaque                 = COALESCE(EXCLUDED.url_casaque, cheval_courses_seen.url_casaque),
                source_commentaire          = COALESCE(EXCLUDED.source_commentaire, cheval_courses_seen.source_commentaire),
                duree_course                = COALESCE(EXCLUDED.duree_course, cheval_courses_seen.duree_course),
                course_trackee              = COALESCE(EXCLUDED.course_trackee, cheval_courses_seen.course_trackee),
                replay_disponible           = COALESCE(EXCLUDED.replay_disponible, cheval_courses_seen.replay_disponible)
        """,
            (
                id_cheval,
                nom_normalize,
                race_key,
                int(date_iso.split("-")[0]),
                is_win,
                reunion,
                course,
                hippodrome_actuel,
                venue_name,
                discipline,
                specialty,
                distance_m,
                start_method,
                rope_side,
                track_surface,
                total_allocation,
                race_name,
                race_conditions,
                race_type,
                start_time,
                track_condition,
                meteo,
                numero_dossard,
                driver,
                entraineur,
                proprietaire,
                eleveur,
                age,
                sexe,
                robe,
                race_cheval,
                pays_origine,
                poids,
                nom_pere,
                nom_mere,
                equipment,
                deferrage,
                handicap_distance,
                cote_prob_matin,
                cote_finale,
                place_finale,
                statut_arrivee,
                temps_str,
                temps_sec,
                reduction_km,
                ecarts,
                gains_course,
                gains_carriere,
                num_runners,
                is_non_runner,
                is_disqualified,
                observations,
                musique_cheval,
                pmu_reunion_id,
                pmu_course_id,
                rapports_json,
                incidents_json,
                rapport_quarte,
                rapport_quinte,
                rapport_multi,
                rapport_pick5,
                montant_enjeux_total,
                allocation_premier,
                allocation_deuxieme,
                allocation_troisieme,
                pourcentage_allocation_course,
                pays_hippodrome,
                region_hippodrome,
                ecart_premier,
                ecart_precedent,
                vitesse_moyenne,
                vitesse_fin_course,
                couleurs_casaque_driver,
                couleurs_casaque_jockey,
                id_driver_pmu,
                id_jockey_pmu,
                id_entraineur_pmu,
                prix_course,
                commentaire_avant_course,
                commentaire_participant_texte or commentaire_apres_course,
                classe_course,
                meteo_code,
                temperature_c,
                vent_kmh,
                penetrometre,
                profil_piste,
                nombre_places,
                nombre_places_second,
                nombre_places_troisieme,
                gains_victoires,
                gains_place,
                gains_annee_en_cours,
                gains_annee_precedente,
                indicateur_inedit,
                driver_change,
                jument_pleine,
                incident,
                # Nouvelles colonnes Phase 3-4
                cote_reference,
                tendance_cote,
                amplitude_tendance,
                est_favori,
                grosse_prise,
                avis_entraineur,
                allure,
                statut_participant,
                supplement,
                engagement,
                poids_condition_monte_change,
                url_casaque,
                source_commentaire,
                duree_course,
                course_trackee,
                replay_disponible,
            ),
        )

        # Log enrichi
        print(
            f"  ✓ {nom} (#{numero_dossard or 'N/A'}, âge:{age or 'N/A'}, "
            + f"sexe:{sexe or 'N/A'}, place:{place_finale or 'N/A'})"
        )

    time.sleep(sleep_s)


# ---------------------------
# Setup & Recalc
# ---------------------------


def _column_exists(cur, table: str, col: str) -> bool:
    """Vérifie si une colonne existe dans une table PostgreSQL"""
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name=%s AND column_name=%s
    """,
        (table, col),
    )
    return cur.fetchone() is not None


def db_setup(con):
    cur = con.cursor()
    # index pour LOWER(nom_cheval)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chevaux_nom_lower ON chevaux(LOWER(nom_cheval));")
    # Index sur cheval_courses_seen
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ccs_nom ON cheval_courses_seen(nom_norm);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ccs_year ON cheval_courses_seen(annee);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ccs_discipline ON cheval_courses_seen(discipline);")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ccs_hippodrome ON cheval_courses_seen(hippodrome_code);"
    )

    # ========================================
    # COLONNES CHEVAUX - Statistiques par discipline
    # ========================================
    print("  🔧 Vérification colonnes chevaux...")
    cols_chevaux = [
        ("nombre_courses_2025", "INTEGER"),
        ("nombre_victoires_2025", "INTEGER"),
        ("dernieres_performances", "TEXT"),
        # Statistiques par discipline
        ("nombre_courses_trot", "INTEGER"),
        ("nombre_victoires_trot", "INTEGER"),
        ("nombre_courses_plat", "INTEGER"),
        ("nombre_victoires_plat", "INTEGER"),
        ("nombre_courses_obstacle", "INTEGER"),
        ("nombre_victoires_obstacle", "INTEGER"),
        ("gains_trot", "BIGINT"),
        ("gains_plat", "BIGINT"),
        ("gains_obstacle", "BIGINT"),
        ("meilleur_place_trot", "INTEGER"),
        ("meilleur_place_plat", "INTEGER"),
        ("meilleur_place_obstacle", "INTEGER"),
        # Records
        ("record_distance_trot", "INTEGER"),
        ("record_distance_plat", "INTEGER"),
        ("reduction_km_record_trot", "REAL"),
        ("reduction_km_record_plat", "REAL"),
        ("date_record_trot", "DATE"),
        ("date_record_plat", "DATE"),
        ("hippodrome_record", "TEXT"),
        # Forme récente
        ("forme_recente_30j", "TEXT"),  # JSON: {courses: X, victoires: Y}
        ("forme_recente_60j", "TEXT"),
        ("forme_recente_90j", "TEXT"),
        ("serie_victoires", "INTEGER"),
        ("serie_places", "INTEGER"),
        ("serie_defaites", "INTEGER"),
        # Gains annuels
        ("gains_annuels_par_annee", "TEXT"),  # JSON: {2024: X, 2025: Y}
    ]

    for col, dtype in cols_chevaux:
        if not _column_exists(cur, "chevaux", col):
            cur.execute(f"ALTER TABLE chevaux ADD COLUMN {col} {dtype};")
            print(f"    ✓ Ajout colonne chevaux.{col}")

    # ========================================
    # COLONNES CHEVAL_COURSES_SEEN - Enrichissements
    # ========================================
    print("  🔧 Vérification colonnes cheval_courses_seen...")
    cols_courses = [
        # Rapports détaillés (extraits depuis rapports_json)
        ("rapport_quarte", "REAL"),
        ("rapport_quinte", "REAL"),
        ("rapport_multi", "REAL"),
        ("rapport_pick5", "REAL"),
        ("montant_enjeux_total", "BIGINT"),
        # Allocations détaillées
        ("allocation_premier", "INTEGER"),
        ("allocation_deuxieme", "INTEGER"),
        ("allocation_troisieme", "INTEGER"),
        ("pourcentage_allocation_course", "REAL"),
        # Hippodrome enrichi
        ("pays_hippodrome", "TEXT"),
        ("region_hippodrome", "TEXT"),
        # Analyse de performance
        ("ecart_premier", "REAL"),  # En secondes
        ("ecart_precedent", "REAL"),  # En secondes
        ("vitesse_moyenne", "REAL"),  # km/h
        ("vitesse_fin_course", "REAL"),  # km/h
        # Personnel enrichi
        ("couleurs_casaque_driver", "TEXT"),
        ("couleurs_casaque_jockey", "TEXT"),
        ("id_driver_pmu", "TEXT"),
        ("id_jockey_pmu", "TEXT"),
        ("id_entraineur_pmu", "TEXT"),
        # Statistiques course
        ("note_journaliste", "REAL"),
        ("commentaire_avant_course", "TEXT"),
        ("commentaire_apres_course", "TEXT"),
        ("prix_course", "TEXT"),  # Nom complet avec sponsors
    ]

    for col, dtype in cols_courses:
        if not _column_exists(cur, "cheval_courses_seen", col):
            cur.execute(f"ALTER TABLE cheval_courses_seen ADD COLUMN {col} {dtype};")
            print(f"    ✓ Ajout colonne cheval_courses_seen.{col}")

    con.commit()
    print("  ✅ Schéma de base de données optimisé!")


def recalc_totals_from_seen(con):
    """
    Recalcule les compteurs dans 'chevaux' depuis 'cheval_courses_seen' (vérité terrain).
    À lancer après un batch si tu veux corriger des valeurs héritées.
    """
    cur = con.cursor()
    cur.execute("""
      UPDATE chevaux
      SET
        nombre_courses_total = (
          SELECT COUNT(*) FROM cheval_courses_seen s
          WHERE s.nom_norm = LOWER(chevaux.nom_cheval)
        ),
        nombre_victoires_total = (
          SELECT COALESCE(SUM(is_win),0) FROM cheval_courses_seen s
          WHERE s.nom_norm = LOWER(chevaux.nom_cheval)
        ),
        nombre_courses_2025 = (
          SELECT COUNT(*) FROM cheval_courses_seen s
          WHERE s.nom_norm = LOWER(chevaux.nom_cheval) AND s.annee = 2025
        ),
        nombre_victoires_2025 = (
          SELECT COALESCE(SUM(is_win),0) FROM cheval_courses_seen s
          WHERE s.nom_norm = LOWER(chevaux.nom_cheval) AND s.annee = 2025
        );
    """)
    con.commit()


# ---------------------------
# Runner
# ---------------------------


def run(date_iso: str, recalc_after: bool = False, use_threading: bool = True):
    """
    Version multi-threadée optimisée.
    use_threading=True : parallélise les courses (5-10x plus rapide)
    use_threading=False : version séquentielle classique
    """
    con = get_connection()
    db_setup(con)

    reunions = discover_reunions(date_iso)
    if not reunions:
        print(f"[INFO] Aucun programme trouvé pour {date_iso}")
        con.close()
        return

    if not use_threading:
        # Mode séquentiel classique (pour debug ou comparaison)
        cur = con.cursor()
        for r in reunions:
            courses = discover_courses(date_iso, r)
            for c in courses:
                print(f"[INFO] {date_iso} R{r} C{c}")
                try:
                    enrich_from_course(cur, date_iso, r, c)
                    con.commit()
                except requests.HTTPError as e:
                    print(f"[WARN] HTTP error on {date_iso} R{r} C{c}: {e}")
                except Exception as e:
                    print(f"[WARN] Unexpected error on {date_iso} R{r} C{c}: {e}")
    else:
        # Mode multi-threadé (RAPIDE!)
        def process_course(args):
            """Traite une course avec sa propre connexion DB (thread-safe)"""
            r, c = args
            thread_con = None
            try:
                # Chaque thread a sa propre connexion pour éviter les conflits de transaction
                thread_con = get_connection()
                cur = thread_con.cursor()

                print(f"[INFO] {date_iso} R{r} C{c}")
                enrich_from_course(cur, date_iso, r, c)

                thread_con.commit()
                return (r, c, "success")
            except requests.HTTPError as e:
                if thread_con:
                    thread_con.rollback()
                print(f"[WARN] HTTP error on {date_iso} R{r} C{c}: {e}")
                return (r, c, f"http_error: {e}")
            except Exception as e:
                if thread_con:
                    thread_con.rollback()
                print(f"[WARN] Unexpected error on {date_iso} R{r} C{c}: {e}")
                return (r, c, f"error: {e}")
            finally:
                if thread_con:
                    thread_con.close()

        # Créer la liste de toutes les tâches (réunion, course)
        tasks = []
        for r in reunions:
            courses = discover_courses(date_iso, r)
            for c in courses:
                tasks.append((r, c))

        print(f"\n🚀 Lancement de {len(tasks)} courses en parallèle ({MAX_WORKERS} threads)...\n")

        # Exécuter en parallèle avec ThreadPoolExecutor
        success_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_course, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    r, c, status = future.result()
                    if status == "success":
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"[ERROR] Task failed: {e}")

        print(f"\n✅ Terminé: {success_count} succès, {error_count} erreurs")

    if recalc_after:
        print("\n📊 Recalcul des totaux depuis l'historique...")
        recalc_totals_from_seen(con)

    con.close()


if __name__ == "__main__":
    import sys

    # Scraper les courses du JOUR ACTUEL
    today = date.today().isoformat()  # Format: YYYY-MM-DD

    # Vérifier si l'utilisateur veut désactiver le multi-threading
    use_threads = "--no-threads" not in sys.argv

    if use_threads:
        print(f"🏇 Début du scraping MULTI-THREADÉ pour {today}")
        print(f"⚡ Utilisation de {MAX_WORKERS} threads en parallèle")
    else:
        print(f"🏇 Début du scraping SÉQUENTIEL pour {today}")

    start_time = time.time()
    run(today, recalc_after=True, use_threading=use_threads)
    elapsed = time.time() - start_time

    print(f"\n✅ Scraping terminé pour {today} en {elapsed:.1f}s")
    print("📊 Récapitulation des données dans la base de données...")

    # Afficher un récapitulatif
    con = get_connection()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM chevaux")
    nb_chevaux = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cheval_courses_seen")
    nb_courses = cur.fetchone()[0]
    con.close()

    print(f"   • Nombre de chevaux: {nb_chevaux}")
    print(f"   • Nombre de courses distinctes: {nb_courses}")
    print(f"   • Temps écoulé: {elapsed:.1f}s")

    if use_threads:
        print(f"\n💡 Pour désactiver le multi-threading: python {sys.argv[0]} --no-threads")

    # Exemples de dates supplémentaires (optionnel):
    # for d in ("2025-10-29", "2025-10-28"):
    #     run(d, recalc_after=False)
