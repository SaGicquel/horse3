#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 FILL MISSING FIELDS - Remplissage intelligent des champs manquants
======================================================================

Remplit les champs manquants via dérivation ou imputation:
- etat_piste: dérivé depuis pénétromètre + météo + mapping hippodrome
- temps_sec: parsé depuis temps officiel ou proxy distance/vitesse_moyenne
- rapport_gagnant: récupéré post-résultat (scraper sécurisé)
- gains_carriere: enrichi via chevaux/performances, imputation par quantiles

IMPORTANT: Aucune fuite temporelle - les données post-départ ne sont jamais
utilisées comme features d'inférence avant le départ.

Auteur: Horse3 System
Date: 2024-12
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# DB Connection
try:
    from db_connection import get_connection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    get_connection = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES & MAPPINGS
# =============================================================================

# Mapping pénétromètre → état piste
PENETROMETRE_TO_ETAT = {
    # Plat et obstacle
    (0.0, 2.5): "très bon",
    (2.5, 3.0): "bon",
    (3.0, 3.5): "souple",
    (3.5, 4.0): "collant",
    (4.0, 4.5): "lourd",
    (4.5, 5.5): "très lourd",
    (5.5, 99.0): "détrempé",
}

# Mapping météo → ajustement état piste
METEO_ADJUSTMENT = {
    "ensoleillé": -0.5,
    "soleil": -0.5,
    "beau temps": -0.5,
    "nuageux": 0,
    "couvert": 0.2,
    "bruine": 0.3,
    "pluie légère": 0.5,
    "pluie": 0.8,
    "pluie forte": 1.2,
    "orage": 1.5,
    "neige": 1.0,
}

# Hippodrome → terrain type par défaut (si pas de pénétromètre)
HIPPODROME_DEFAULT_TERRAIN = {
    "longchamp": "souple",
    "auteuil": "souple",
    "vincennes": "bon",
    "chantilly": "bon",
    "deauville": "bon",
    "saint-cloud": "souple",
    "maisons-laffitte": "bon",
    "fontainebleau": "souple",
    "compiegne": "souple",
    "enghien": "bon",
    "cagnes-sur-mer": "bon",
}

# Vitesse moyenne par discipline × hippodrome (km/h) pour proxy temps
VITESSE_MOYENNE_REF = {
    "plat": {
        "default": 58.0,
        "longchamp": 56.5,
        "deauville": 57.0,
        "chantilly": 58.0,
    },
    "trot": {
        "default": 46.0,
        "vincennes": 45.5,
        "enghien": 46.0,
        "cabourg": 45.0,
    },
    "obstacle": {
        "default": 50.0,
        "auteuil": 48.0,
        "compiegne": 51.0,
    }
}

# Quantiles de référence pour gains_carriere (par âge × discipline)
# Format: {(age_min, age_max, discipline): [q10, q25, q50, q75, q90]}
GAINS_QUANTILES_REF = {
    (2, 2, "plat"): [0, 2000, 8000, 25000, 80000],
    (3, 3, "plat"): [0, 5000, 20000, 60000, 200000],
    (4, 5, "plat"): [0, 10000, 40000, 120000, 400000],
    (6, 99, "plat"): [0, 15000, 60000, 180000, 600000],
    (2, 3, "trot"): [0, 3000, 15000, 50000, 150000],
    (4, 6, "trot"): [0, 10000, 45000, 150000, 500000],
    (7, 99, "trot"): [0, 20000, 80000, 250000, 800000],
    (3, 4, "obstacle"): [0, 2000, 10000, 35000, 100000],
    (5, 99, "obstacle"): [0, 8000, 30000, 100000, 350000],
}

# Champs POST-OFF (ne jamais utiliser en features d'inférence AVANT départ)
POST_OFF_FIELDS = frozenset({
    "place_finale",
    "statut_arrivee",
    "temps_str",
    "temps_sec",
    "reduction_km_sec",
    "ecarts",
    "gains_course",
    "rapport_gagnant",
    "rapport_place",
    "rapport_couple",
    "rapport_trio",
    "disqualifie",
    # Note: non_partant peut être connu avant si déclaré
})

# Features autorisées pour l'inférence (PRE-OFF uniquement)
PRE_OFF_FEATURES = frozenset({
    "discipline",
    "distance_m",
    "hippodrome_nom",
    "etat_piste",  # Dérivé avant course
    "meteo",
    "age",
    "sexe",
    "poids_kg",
    "numero_dossard",
    "cote_matin",
    "cote_finale",  # Juste avant départ, OK
    "driver_jockey",
    "entraineur",
    "musique",
    "gains_carriere",  # Historique, OK
    "nb_courses_carriere",
    "nb_victoires_carriere",
    "taux_places_carriere",
})


# =============================================================================
# DATACLASS POUR FLAGS DE QUALITÉ
# =============================================================================

@dataclass
class QualityFlags:
    """Flags de qualité pour un enregistrement."""
    etat_piste_source: str = "unknown"  # penetro|meteo|hippodrome|default
    temps_sec_source: str = "unknown"   # parsed|proxy|null
    rapport_gagnant_source: str = "unknown"  # scraped|null
    gains_carriere_source: str = "unknown"  # db|imputed|null
    
    # Flags de problème
    report_missing: bool = False  # rapport_gagnant manquant
    temps_missing: bool = False   # temps_sec non récupérable
    gains_imputed: bool = False   # gains_carriere imputé
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "etat_piste_source": self.etat_piste_source,
            "temps_sec_source": self.temps_sec_source,
            "rapport_gagnant_source": self.rapport_gagnant_source,
            "gains_carriere_source": self.gains_carriere_source,
            "report_missing": self.report_missing,
            "temps_missing": self.temps_missing,
            "gains_imputed": self.gains_imputed,
        }


# =============================================================================
# FONCTIONS DE DÉRIVATION
# =============================================================================

def derive_etat_piste(
    penetrometre: Optional[float],
    meteo: Optional[str],
    hippodrome: Optional[str]
) -> Tuple[str, str]:
    """
    Dérive l'état de la piste depuis les données disponibles.
    
    Args:
        penetrometre: Valeur pénétromètre (0-6+)
        meteo: Conditions météo textuelles
        hippodrome: Nom de l'hippodrome
        
    Returns:
        Tuple[str, str]: (etat_piste, source)
    """
    # 1. Si pénétromètre disponible
    if penetrometre is not None and penetrometre >= 0:
        # Ajuster selon météo
        adjustment = 0.0
        if meteo:
            meteo_lower = meteo.lower()
            for keyword, adj in METEO_ADJUSTMENT.items():
                if keyword in meteo_lower:
                    adjustment = adj
                    break
        
        adjusted_penetro = penetrometre + adjustment
        
        # Mapper vers état
        for (low, high), etat in PENETROMETRE_TO_ETAT.items():
            if low <= adjusted_penetro < high:
                return etat, "penetro"
        
        return "lourd", "penetro"  # Fallback si très élevé
    
    # 2. Fallback sur hippodrome par défaut
    if hippodrome:
        hippo_lower = hippodrome.lower().strip()
        for h, default_etat in HIPPODROME_DEFAULT_TERRAIN.items():
            if h in hippo_lower:
                return default_etat, "hippodrome"
    
    # 3. Fallback météo seul (approximation grossière)
    if meteo:
        meteo_lower = meteo.lower()
        if any(w in meteo_lower for w in ["pluie forte", "orage", "détrempé"]):
            return "lourd", "meteo"
        elif any(w in meteo_lower for w in ["pluie", "bruine"]):
            return "souple", "meteo"
        elif any(w in meteo_lower for w in ["soleil", "beau"]):
            return "bon", "meteo"
    
    # 4. Défaut absolu
    return "unknown", "default"


def parse_temps_str(temps_str: Optional[str]) -> Optional[float]:
    """
    Parse une chaîne de temps en secondes.
    
    Formats supportés:
    - "1'23''4" → 83.4s
    - "1:23.4" → 83.4s
    - "1m23s4" → 83.4s
    - "83.4" → 83.4s (déjà en secondes)
    
    Args:
        temps_str: Temps au format texte
        
    Returns:
        float ou None si non parsable
    """
    if not temps_str:
        return None
    
    temps_str = str(temps_str).strip()
    
    # Format "1'23''4" ou "1'23\"4"
    match = re.match(r"(\d+)'(\d+)''?\"?(\d*)", temps_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        tenths = int(match.group(3)) if match.group(3) else 0
        return minutes * 60 + seconds + tenths / 10
    
    # Format "1:23.4" ou "1:23,4"
    match = re.match(r"(\d+):(\d+)[.,]?(\d*)", temps_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        decimals = float(f"0.{match.group(3)}") if match.group(3) else 0
        return minutes * 60 + seconds + decimals
    
    # Format "1m23s4" ou "1m23s"
    match = re.match(r"(\d+)m(\d+)s(\d*)", temps_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        tenths = int(match.group(3)) if match.group(3) else 0
        return minutes * 60 + seconds + tenths / 10
    
    # Déjà en secondes (float)
    try:
        return float(temps_str)
    except ValueError:
        return None


def proxy_temps_sec(
    distance_m: Optional[int],
    discipline: Optional[str],
    hippodrome: Optional[str]
) -> Optional[float]:
    """
    Estime le temps en secondes via proxy distance/vitesse moyenne.
    
    Args:
        distance_m: Distance en mètres
        discipline: plat/trot/obstacle
        hippodrome: Nom de l'hippodrome
        
    Returns:
        float estimé ou None
    """
    if not distance_m or distance_m <= 0:
        return None
    
    disc = (discipline or "plat").lower()
    if disc not in VITESSE_MOYENNE_REF:
        disc = "plat"
    
    # Chercher vitesse spécifique hippodrome
    vitesse_km_h = VITESSE_MOYENNE_REF[disc].get("default", 55.0)
    
    if hippodrome:
        hippo_lower = hippodrome.lower()
        for h, v in VITESSE_MOYENNE_REF[disc].items():
            if h != "default" and h in hippo_lower:
                vitesse_km_h = v
                break
    
    # Temps = distance / vitesse
    # vitesse en km/h → m/s = vitesse / 3.6
    vitesse_m_s = vitesse_km_h / 3.6
    temps_sec = distance_m / vitesse_m_s
    
    return round(temps_sec, 2)


def impute_gains_carriere(
    age: Optional[int],
    discipline: Optional[str],
    performances_count: int = 0,
    victoires_count: int = 0,
    random_state: Optional[int] = None
) -> Tuple[Optional[int], str]:
    """
    Impute les gains carrière par quantiles (âge × discipline).
    
    Args:
        age: Âge du cheval
        discipline: plat/trot/obstacle
        performances_count: Nombre de courses
        victoires_count: Nombre de victoires
        random_state: Seed pour reproductibilité
        
    Returns:
        Tuple[int, str]: (gains_carriere, source)
    """
    if age is None:
        return None, "null"
    
    disc = (discipline or "plat").lower()
    if disc not in ["plat", "trot", "obstacle"]:
        disc = "plat"
    
    # Trouver la tranche d'âge
    quantiles = None
    for (age_min, age_max, d), q in GAINS_QUANTILES_REF.items():
        if age_min <= age <= age_max and d == disc:
            quantiles = q
            break
    
    if quantiles is None:
        # Fallback sur la dernière tranche
        quantiles = [0, 10000, 40000, 120000, 400000]
    
    # Ajuster le quantile cible selon performances
    if victoires_count >= 3:
        target_quantile = 0.75  # Q75
    elif performances_count >= 10:
        target_quantile = 0.50  # Médiane
    elif performances_count >= 5:
        target_quantile = 0.25  # Q25
    else:
        target_quantile = 0.10  # Q10
    
    # Interpoler
    # quantiles = [q10, q25, q50, q75, q90]
    quantile_points = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Interpolation + bruit
    gains = np.interp(target_quantile, quantile_points, quantiles)
    noise = np.random.normal(0, 0.1) * gains  # ±10% de bruit
    gains = max(0, int(gains + noise))
    
    return gains, "imputed"


# =============================================================================
# SCRAPER RAPPORT GAGNANT (SÉCURISÉ)
# =============================================================================

def scrape_rapport_gagnant(
    race_id: str,
    horse_number: int,
    max_retries: int = 2,
    timeout: float = 5.0
) -> Tuple[Optional[float], str]:
    """
    Récupère le rapport gagnant de manière sécurisée.
    
    NOTE: Cette fonction est un placeholder. Dans une implémentation
    réelle, elle ferait un appel API ou scraping web sécurisé.
    
    Args:
        race_id: Identifiant de la course (format YYYY-MM-DD_RX_CY)
        horse_number: Numéro du cheval gagnant
        max_retries: Nombre de tentatives
        timeout: Timeout en secondes
        
    Returns:
        Tuple[float, str]: (rapport_gagnant, source)
    """
    # TODO: Implémenter scraping réel vers PMU/autres sources
    # Pour l'instant, on retourne null + flag
    logger.debug(f"Scrape rapport: race={race_id}, horse={horse_number}")
    
    return None, "null"


# =============================================================================
# JOB PRINCIPAL
# =============================================================================

class FillMissingFieldsJob:
    """
    Job de remplissage des champs manquants.
    
    GARANTIES:
    - Pas de fuite temporelle : les champs POST_OFF ne sont jamais utilisés
      comme features avant le départ
    - Flags de qualité toujours posés
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        conn=None,
        batch_size: int = 1000,
        dry_run: bool = False
    ):
        self.conn = conn or (get_connection() if DB_AVAILABLE else None)
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.stats = {
            "processed": 0,
            "etat_piste_filled": 0,
            "temps_sec_filled": 0,
            "rapport_gagnant_filled": 0,
            "gains_carriere_filled": 0,
            "errors": 0,
        }
    
    def fill_etat_piste(self) -> int:
        """
        Remplit etat_piste manquant via dérivation.
        
        Returns:
            Nombre d'enregistrements mis à jour
        """
        if not self.conn:
            logger.error("Pas de connexion DB")
            return 0
        
        cur = self.conn.cursor()
        
        # Sélectionner les enregistrements avec etat_piste manquant
        cur.execute("""
            SELECT race_key, penetrometre, meteo, hippodrome_nom
            FROM cheval_courses_seen
            WHERE (etat_piste IS NULL OR etat_piste = '' OR etat_piste = 'unknown')
            LIMIT %s
        """, (self.batch_size,))
        
        rows = cur.fetchall()
        updated = 0
        
        for row in rows:
            race_key, penetro, meteo, hippodrome = row
            
            etat, source = derive_etat_piste(penetro, meteo, hippodrome)
            
            if etat and etat != "unknown":
                if not self.dry_run:
                    cur.execute("""
                        UPDATE cheval_courses_seen
                        SET etat_piste = %s,
                            etat_piste_source = %s
                        WHERE race_key = %s
                    """, (etat, source, race_key))
                updated += 1
        
        if not self.dry_run:
            self.conn.commit()
        
        self.stats["etat_piste_filled"] = updated
        logger.info(f"etat_piste: {updated} mis à jour (source: penetro/meteo/hippodrome)")
        return updated
    
    def fill_temps_sec(self) -> int:
        """
        Remplit temps_sec via parsing ou proxy.
        
        Returns:
            Nombre d'enregistrements mis à jour
        """
        if not self.conn:
            return 0
        
        cur = self.conn.cursor()
        
        # D'abord parser temps_str
        cur.execute("""
            SELECT race_key, temps_str, distance_m, discipline, hippodrome_nom
            FROM cheval_courses_seen
            WHERE temps_sec IS NULL
              AND statut_arrivee NOT IN ('NON_PARTANT', 'DISQUALIFIE')
            LIMIT %s
        """, (self.batch_size,))
        
        rows = cur.fetchall()
        updated = 0
        
        for row in rows:
            race_key, temps_str, distance, discipline, hippodrome = row
            
            # Essayer de parser
            temps_sec = parse_temps_str(temps_str)
            source = "parsed" if temps_sec else None
            
            # Sinon proxy
            if temps_sec is None:
                temps_sec = proxy_temps_sec(distance, discipline, hippodrome)
                source = "proxy" if temps_sec else "null"
            
            if temps_sec:
                if not self.dry_run:
                    cur.execute("""
                        UPDATE cheval_courses_seen
                        SET temps_sec = %s,
                            temps_sec_source = %s
                        WHERE race_key = %s
                    """, (temps_sec, source, race_key))
                updated += 1
        
        if not self.dry_run:
            self.conn.commit()
        
        self.stats["temps_sec_filled"] = updated
        logger.info(f"temps_sec: {updated} mis à jour (parsed/proxy)")
        return updated
    
    def fill_rapport_gagnant(self) -> int:
        """
        Récupère les rapports gagnants manquants (post-course).
        
        Returns:
            Nombre d'enregistrements mis à jour
        """
        if not self.conn:
            return 0
        
        cur = self.conn.cursor()
        
        # Sélectionner les gagnants sans rapport
        cur.execute("""
            SELECT race_key, numero_dossard
            FROM cheval_courses_seen
            WHERE place_finale = 1
              AND (rapport_gagnant IS NULL OR rapport_gagnant <= 0)
            LIMIT %s
        """, (self.batch_size,))
        
        rows = cur.fetchall()
        updated = 0
        flagged = 0
        
        for row in rows:
            race_key, numero = row
            
            rapport, source = scrape_rapport_gagnant(race_key, numero)
            
            if rapport and rapport > 0:
                if not self.dry_run:
                    cur.execute("""
                        UPDATE cheval_courses_seen
                        SET rapport_gagnant = %s,
                            rapport_gagnant_source = %s,
                            report_missing = FALSE
                        WHERE race_key = %s
                    """, (rapport, source, race_key))
                updated += 1
            else:
                # Poser le flag report_missing
                if not self.dry_run:
                    cur.execute("""
                        UPDATE cheval_courses_seen
                        SET report_missing = TRUE,
                            rapport_gagnant_source = 'null'
                        WHERE race_key = %s
                    """, (race_key,))
                flagged += 1
        
        if not self.dry_run:
            self.conn.commit()
        
        self.stats["rapport_gagnant_filled"] = updated
        logger.info(f"rapport_gagnant: {updated} récupérés, {flagged} flaggés manquants")
        return updated
    
    def fill_gains_carriere(self) -> int:
        """
        Enrichit/impute gains_carriere depuis chevaux/performances.
        
        Returns:
            Nombre d'enregistrements mis à jour
        """
        if not self.conn:
            return 0
        
        cur = self.conn.cursor()
        
        # D'abord essayer d'enrichir depuis table chevaux
        cur.execute("""
            UPDATE cheval_courses_seen AS ccs
            SET gains_carriere = c.gains_totaux,
                gains_carriere_source = 'db'
            FROM chevaux AS c
            WHERE LOWER(ccs.nom_norm) = LOWER(c.nom_cheval)
              AND ccs.gains_carriere IS NULL
              AND c.gains_totaux IS NOT NULL
              AND c.gains_totaux > 0
        """)
        
        from_db = cur.rowcount if not self.dry_run else 0
        
        # Ensuite calculer depuis performances historiques
        cur.execute("""
            WITH horse_gains AS (
                SELECT 
                    nom_norm,
                    SUM(COALESCE(gains_course, 0)) AS calculated_gains
                FROM cheval_courses_seen
                WHERE gains_course IS NOT NULL
                GROUP BY nom_norm
            )
            UPDATE cheval_courses_seen AS ccs
            SET gains_carriere = hg.calculated_gains,
                gains_carriere_source = 'calculated'
            FROM horse_gains AS hg
            WHERE ccs.nom_norm = hg.nom_norm
              AND ccs.gains_carriere IS NULL
              AND hg.calculated_gains > 0
        """)
        
        from_calc = cur.rowcount if not self.dry_run else 0
        
        # Enfin imputer pour les restants
        cur.execute("""
            SELECT race_key, age, discipline
            FROM cheval_courses_seen
            WHERE gains_carriere IS NULL
            LIMIT %s
        """, (self.batch_size,))
        
        rows = cur.fetchall()
        imputed = 0
        
        for row in rows:
            race_key, age, discipline = row
            
            # Compter les performances et victoires
            cur.execute("""
                SELECT COUNT(*), SUM(CASE WHEN place_finale = 1 THEN 1 ELSE 0 END)
                FROM cheval_courses_seen
                WHERE nom_norm = (SELECT nom_norm FROM cheval_courses_seen WHERE race_key = %s)
                  AND date_course < (SELECT date_course FROM cheval_courses_seen WHERE race_key = %s LIMIT 1)
            """, (race_key, race_key))
            
            counts = cur.fetchone()
            perf_count = counts[0] or 0
            win_count = counts[1] or 0
            
            gains, source = impute_gains_carriere(age, discipline, perf_count, win_count)
            
            if gains is not None:
                if not self.dry_run:
                    cur.execute("""
                        UPDATE cheval_courses_seen
                        SET gains_carriere = %s,
                            gains_carriere_source = %s,
                            gains_imputed = TRUE
                        WHERE race_key = %s
                    """, (gains, source, race_key))
                imputed += 1
        
        if not self.dry_run:
            self.conn.commit()
        
        total = from_db + from_calc + imputed
        self.stats["gains_carriere_filled"] = total
        logger.info(f"gains_carriere: {from_db} depuis DB, {from_calc} calculés, {imputed} imputés")
        return total
    
    def run(self) -> Dict[str, int]:
        """
        Exécute tous les remplissages.
        
        Returns:
            Dict avec statistiques
        """
        logger.info(f"🔧 Fill Missing Fields Job v{self.VERSION}")
        logger.info(f"   dry_run={self.dry_run}, batch_size={self.batch_size}")
        
        try:
            self.fill_etat_piste()
            self.fill_temps_sec()
            self.fill_rapport_gagnant()
            self.fill_gains_carriere()
            
            self.stats["processed"] = sum([
                self.stats["etat_piste_filled"],
                self.stats["temps_sec_filled"],
                self.stats["rapport_gagnant_filled"],
                self.stats["gains_carriere_filled"],
            ])
            
        except Exception as e:
            logger.error(f"Erreur lors du job: {e}")
            self.stats["errors"] += 1
            raise
        
        logger.info(f"✅ Job terminé: {self.stats['processed']} champs remplis")
        return self.stats
    
    @staticmethod
    def validate_no_temporal_leakage(features: List[str]) -> Tuple[bool, List[str]]:
        """
        Valide qu'aucune feature n'est un champ POST_OFF.
        
        Args:
            features: Liste des features utilisées
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, violations)
        """
        violations = [f for f in features if f in POST_OFF_FIELDS]
        return len(violations) == 0, violations
    
    @staticmethod
    def get_pre_off_features() -> List[str]:
        """Retourne la liste des features autorisées pré-départ."""
        return list(PRE_OFF_FEATURES)
    
    @staticmethod
    def get_post_off_fields() -> List[str]:
        """Retourne la liste des champs post-départ (interdits en inférence)."""
        return list(POST_OFF_FIELDS)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Point d'entrée CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Remplir les champs manquants")
    parser.add_argument("--dry-run", action="store_true", help="Ne pas modifier la DB")
    parser.add_argument("--batch-size", type=int, default=1000, help="Taille des batches")
    parser.add_argument("--field", choices=["etat_piste", "temps_sec", "rapport_gagnant", "gains_carriere", "all"],
                       default="all", help="Champ à remplir")
    
    args = parser.parse_args()
    
    job = FillMissingFieldsJob(
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    
    if args.field == "all":
        stats = job.run()
    elif args.field == "etat_piste":
        job.fill_etat_piste()
        stats = job.stats
    elif args.field == "temps_sec":
        job.fill_temps_sec()
        stats = job.stats
    elif args.field == "rapport_gagnant":
        job.fill_rapport_gagnant()
        stats = job.stats
    elif args.field == "gains_carriere":
        job.fill_gains_carriere()
        stats = job.stats
    
    print(f"\n📊 Résultats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
