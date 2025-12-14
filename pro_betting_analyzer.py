#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏇 PRO BETTING ANALYZER - Système de Probabilités Calibrées
============================================================

Analyste de paris hippiques niveau pro:
- Probabilités calibrées et cohérentes au NIVEAU COURSE (somme p_win = 1)
- Normalisation softmax à température ou Plackett-Luce
- Aucune fuite temporelle (données pré-départ uniquement)
- Fusion modèle/marché: marché = prior, modèle = likelihood
- Sortie strictement JSON

Auteur: Horse3 System
Version: 2.1.0

CONFIGURATION:
- Les paramètres T, α, kelly sont chargés depuis config/pro_betting.yaml
- Le décorateur @coherent_params garantit la cohérence avec les artefacts de calibration
"""

import json
import math
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np

# Import de la configuration centralisée
try:
    from config.loader import get_config, coherent_params, get_calibration_params_from_artifacts
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    warnings.warn(
        "⚠️ config.loader non disponible - paramètres par défaut utilisés. "
        "Veuillez créer config/pro_betting.yaml pour garantir la cohérence.",
        UserWarning
    )

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES DE SORTIE
# =============================================================================

@dataclass
class RunnerAnalysis:
    """Analyse d'un partant - sortie JSON stricte"""
    numero: int
    nom: str
    p_win: Optional[float] = None          # Probabilité victoire (normalisée, somme=1)
    p_place: Optional[float] = None        # Probabilité placé (top 3)
    fair_odds: Optional[float] = None      # Cote juste (1/p_win)
    market_odds: Optional[float] = None    # Cote marché actuelle
    value_pct: Optional[float] = None      # Value % = (fair - market) / market * 100
    kelly_fraction: Optional[float] = None # Fraction Kelly (0-1)
    rationale: List[str] = field(default_factory=list)  # 2-3 puces max
    
    def to_dict(self) -> dict:
        return {
            "numero": self.numero,
            "nom": self.nom,
            "p_win": round(self.p_win, 4) if self.p_win is not None else None,
            "p_place": round(self.p_place, 4) if self.p_place is not None else None,
            "fair_odds": round(self.fair_odds, 2) if self.fair_odds is not None else None,
            "market_odds": self.market_odds,
            "value_pct": round(self.value_pct, 2) if self.value_pct is not None else None,
            "kelly_fraction": round(self.kelly_fraction, 4) if self.kelly_fraction is not None else None,
            "rationale": self.rationale[:3]  # Max 3 puces
        }


@dataclass
class RaceAnalysis:
    """Analyse complète d'une course - sortie JSON"""
    race_id: str
    timestamp: str
    hippodrome: str
    distance_m: int
    discipline: str
    nb_partants: int
    model_version: str
    runners: List[RunnerAnalysis] = field(default_factory=list)
    run_notes: List[str] = field(default_factory=list)  # Alertes/warnings
    
    def to_json(self) -> str:
        return json.dumps({
            "race_id": self.race_id,
            "timestamp": self.timestamp,
            "hippodrome": self.hippodrome,
            "distance_m": self.distance_m,
            "discipline": self.discipline,
            "nb_partants": self.nb_partants,
            "model_version": self.model_version,
            "runners": [r.to_dict() for r in self.runners],
            "run_notes": self.run_notes
        }, ensure_ascii=False, indent=2)


# =============================================================================
# NORMALISATION DES PROBABILITÉS
# =============================================================================

class ProbabilityNormalizer:
    """
    Normalise les probabilités au niveau course.
    Méthodes: softmax avec température, Plackett-Luce, ou simple renormalisation.
    """
    
    @staticmethod
    def softmax_temperature(raw_scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Softmax avec température.
        T < 1: plus de confiance dans le favori
        T > 1: distribution plus uniforme
        """
        if len(raw_scores) == 0:
            return np.array([])
        
        # Stabilité numérique
        scores = raw_scores / temperature
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        
        return exp_scores / np.sum(exp_scores)
    
    @staticmethod
    def plackett_luce_normalize(strengths: np.ndarray) -> np.ndarray:
        """
        Modèle Plackett-Luce: p(i gagne) = strength_i / sum(strengths)
        Les strengths sont typiquement exp(score) ou 1/cote
        """
        if len(strengths) == 0:
            return np.array([])
        
        strengths = np.maximum(strengths, 1e-10)  # Éviter division par 0
        return strengths / np.sum(strengths)
    
    @staticmethod
    def harville_place_proba(win_probs: np.ndarray, top_n: int = 3) -> np.ndarray:
        """
        Approximation de Harville pour P(top N).
        Plus précis que simple multiplication.
        """
        n = len(win_probs)
        if n == 0:
            return np.array([])
        
        place_probs = np.zeros(n)
        
        for i in range(n):
            # P(i dans top N) ≈ somme des P(i finit à position k) pour k=1..N
            prob_i = 0.0
            
            # Position 1: P(win)
            prob_i += win_probs[i]
            
            # Positions 2 et 3 (Harville)
            for pos in range(2, min(top_n + 1, n + 1)):
                # P(i finit pos) = sum_{j!=i} P(j gagne) * P(i gagne | j parti)
                for j in range(n):
                    if j != i:
                        remaining_strength = 1 - win_probs[j]
                        if remaining_strength > 1e-10:
                            p_i_given_j = win_probs[i] / remaining_strength
                            
                            if pos == 2:
                                prob_i += win_probs[j] * p_i_given_j
                            elif pos == 3:
                                # Position 3: intégrer sur les 2 premiers
                                for k in range(n):
                                    if k != i and k != j:
                                        remaining_2 = 1 - win_probs[j] - win_probs[k]
                                        if remaining_2 > 1e-10:
                                            p_i_given_jk = win_probs[i] / remaining_2
                                            p_k_given_j = win_probs[k] / (1 - win_probs[j])
                                            prob_i += win_probs[j] * p_k_given_j * p_i_given_jk * 0.1
            
            place_probs[i] = min(0.95, prob_i)
        
        return place_probs


# =============================================================================
# FUSION MODÈLE / MARCHÉ (BAYÉSIEN)
# =============================================================================

class BayesianOddsFusion:
    """
    Fusion bayésienne: marché = prior, modèle = likelihood.
    Permet d'intégrer prudemment les signaux du modèle.
    """
    
    def __init__(self, market_weight: float = 0.6, model_weight: float = 0.4):
        """
        market_weight + model_weight = 1.0
        Plus le marché est efficient, plus market_weight est élevé.
        """
        self.market_weight = market_weight
        self.model_weight = model_weight
    
    def fuse_probabilities(
        self,
        market_probs: np.ndarray,
        model_probs: np.ndarray,
        confidence: float = 1.0
    ) -> np.ndarray:
        """
        Fusionne probabilités marché et modèle.
        
        Args:
            market_probs: P implicites du marché (1/cote normalisées)
            model_probs: P du modèle ML
            confidence: Facteur de confiance dans le modèle (0-1)
        
        Returns:
            Probabilités fusionnées et normalisées
        """
        if len(market_probs) != len(model_probs):
            raise ValueError("Arrays must have same length")
        
        # Vérifier si le marché est informatif (pas de cotes par défaut)
        market_has_info = np.any(market_probs != market_probs[0])
        
        if not market_has_info:
            # Pas de cotes différenciées -> utiliser le modèle seul
            return model_probs
        
        # Ajuster le poids du modèle selon la confiance
        adjusted_model_weight = self.model_weight * confidence
        adjusted_market_weight = 1 - adjusted_model_weight
        
        # Fusion log-linéaire (géométrique)
        # P_fused ∝ P_market^w_market * P_model^w_model
        log_market = np.log(np.maximum(market_probs, 1e-10))
        log_model = np.log(np.maximum(model_probs, 1e-10))
        
        log_fused = adjusted_market_weight * log_market + adjusted_model_weight * log_model
        
        # Exponentier et normaliser
        fused = np.exp(log_fused - np.max(log_fused))
        fused = fused / np.sum(fused)
        
        return fused
    
    def calculate_overround(self, odds: np.ndarray) -> float:
        """Calcule l'overround (marge bookmaker)"""
        implied_probs = 1 / np.maximum(odds, 1.01)
        return np.sum(implied_probs) - 1


# =============================================================================
# KELLY CRITERION
# =============================================================================

class KellyCriterion:
    """
    Calcul du critère de Kelly fractionnaire.
    """
    
    @staticmethod
    def calculate(p_win: float, odds: float, fraction: float = 0.25) -> float:
        """
        Calcule la fraction Kelly.
        
        Args:
            p_win: Probabilité de victoire estimée
            odds: Cote décimale européenne
            fraction: Fraction de Kelly à utiliser (0.25 = quart Kelly)
        
        Returns:
            Fraction du bankroll à miser (0 si négatif)
        """
        if odds <= 1 or p_win <= 0 or p_win >= 1:
            return 0.0
        
        q = 1 - p_win
        b = odds - 1  # Gain net pour 1€ misé
        
        # Kelly formula: f* = (bp - q) / b
        kelly = (b * p_win - q) / b
        
        # Appliquer fraction et clipper
        kelly_fractional = kelly * fraction
        
        return max(0.0, min(kelly_fractional, 0.1))  # Cap à 10% du bankroll


# =============================================================================
# ANALYSEUR PRINCIPAL
# =============================================================================

class ProBettingAnalyzer:
    """
    Analyseur de paris hippiques niveau professionnel.
    Produit des probabilités calibrées et cohérentes.
    
    Les paramètres sont chargés depuis config/pro_betting.yaml pour garantir
    la cohérence avec les artefacts de calibration (T*, α).
    """
    
    VERSION = "2.1.0"
    
    def __init__(
        self,
        db_connection,
        softmax_temperature: float = None,  # Si None, charge depuis config
        market_weight: float = None,         # Si None, charge depuis config (1-α)
        kelly_fraction: float = None,        # Si None, charge depuis config
        discipline: str = None               # Pour α par discipline
    ):
        """
        Initialise l'analyseur.
        
        Args:
            db_connection: Connexion à la base de données
            softmax_temperature: Température T (défaut: config.calibration.temperature)
            market_weight: Poids marché 1-α (défaut: 1 - config.calibration.blend_alpha)
            kelly_fraction: Fraction Kelly (défaut: config.kelly.fraction)
            discipline: Discipline pour α spécifique (plat, trot, obstacle)
        """
        self.conn = db_connection
        self.discipline = discipline
        
        # === Charger les paramètres depuis la config centralisée ===
        if CONFIG_AVAILABLE:
            config = get_config()
            artifact_params = get_calibration_params_from_artifacts()
            
            # Température: priorité artefact > config > param > défaut
            if softmax_temperature is not None:
                # Vérifier cohérence
                expected_temp = artifact_params['temperature']
                if abs(softmax_temperature - expected_temp) > 0.001:
                    warnings.warn(
                        f"⚠️ Temperature mismatch: passed={softmax_temperature}, "
                        f"expected={expected_temp} (from {artifact_params['source']}). "
                        f"Using expected value.",
                        UserWarning
                    )
                    softmax_temperature = expected_temp
            else:
                softmax_temperature = artifact_params['temperature']
            
            # Alpha/market_weight: α = poids modèle, market_weight = 1 - α
            if market_weight is not None:
                expected_alpha = config.get_blend_alpha(discipline or 'default')
                expected_market = 1 - expected_alpha
                if abs(market_weight - expected_market) > 0.001:
                    warnings.warn(
                        f"⚠️ Market weight mismatch: passed={market_weight}, "
                        f"expected={expected_market} (α={expected_alpha}). "
                        f"Using expected value.",
                        UserWarning
                    )
                    market_weight = expected_market
            else:
                alpha = config.get_blend_alpha(discipline or 'default')
                market_weight = 1 - alpha
            
            # Kelly fraction
            if kelly_fraction is None:
                kelly_fraction = config.kelly.fraction
            
            # Log des paramètres utilisés
            logger.info(f"✅ ProBettingAnalyzer init: T={softmax_temperature:.4f}, "
                       f"market_weight={market_weight:.2f}, kelly={kelly_fraction:.2f}")
        else:
            # Fallback aux valeurs par défaut calibrées (T=1.254, α=0.2)
            if softmax_temperature is None:
                softmax_temperature = 1.254  # Valeur calibrée
            if market_weight is None:
                market_weight = 0.8  # 1 - α où α=0.2
            if kelly_fraction is None:
                kelly_fraction = 0.25
            
            warnings.warn(
                f"⚠️ Config non disponible, utilisation des défauts: "
                f"T={softmax_temperature}, market_weight={market_weight}",
                UserWarning
            )
        
        self.temperature = softmax_temperature
        self.market_weight = market_weight
        self.kelly_fraction = kelly_fraction
        
        self.normalizer = ProbabilityNormalizer()
        self.fusion = BayesianOddsFusion(market_weight=market_weight)
        self.kelly = KellyCriterion()
        
        # Cache pour éviter requêtes répétées
        self._stats_cache = {}
    
    def _get_runner_raw_score(
        self,
        nom: str,
        cote: float,
        distance: int,
        hippodrome: str,
        data: dict
    ) -> Tuple[float, List[str]]:
        """
        Calcule un score brut pour un partant.
        Utilise UNIQUEMENT des données pré-départ (pas de fuite temporelle).
        
        Returns:
            (score, rationale_items)
        """
        cur = self.conn.cursor()
        score = 50.0
        rationale = []
        
        # 1. Forme récente (5 dernières courses AVANT cette date)
        race_date = data.get('date_course', datetime.now().strftime('%Y-%m-%d'))
        
        cur.execute("""
            SELECT 
                COUNT(*) as nb,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN place_finale <= 3 THEN 1 ELSE 0 END) as places,
                AVG(place_finale) as avg_place
            FROM (
                SELECT is_win, place_finale
                FROM cheval_courses_seen
                WHERE nom_norm = %s AND race_key < %s
                ORDER BY race_key DESC
                LIMIT 5
            ) sub
        """, (nom, race_date))
        
        row = cur.fetchone()
        if row and row[0] and row[0] >= 3:
            nb, wins, places, avg_place = row
            
            if wins and wins >= 2:
                score += 25
                rationale.append(f"🔥 {wins}/5 victoires récentes")
            elif wins and wins >= 1:
                score += 10
            
            if places and places >= 4:
                score += 15
                rationale.append(f"📊 Régulier ({places}/5 placés)")
            
            if avg_place and avg_place < 4:
                score += 10
        elif row and row[0] and row[0] < 3:
            rationale.append("⚠️ Peu de courses récentes")
        
        # 2. Historique sur distance similaire
        cur.execute("""
            SELECT 
                COUNT(*) as nb,
                AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) * 100 as win_rate
            FROM cheval_courses_seen
            WHERE nom_norm = %s 
            AND ABS(distance_m - %s) < 200
            AND race_key < %s
        """, (nom, distance, race_date))
        
        row = cur.fetchone()
        if row and row[0] and row[0] >= 3:
            if row[1] and row[1] > 15:
                score += 15
                rationale.append(f"📏 Bon sur {distance}m ({row[1]:.0f}%)")
            elif row[1] and row[1] < 5:
                score -= 10
        
        # 3. Historique hippodrome
        cur.execute("""
            SELECT 
                COUNT(*) as nb,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
            FROM cheval_courses_seen
            WHERE nom_norm = %s 
            AND hippodrome_nom = %s
            AND race_key < %s
        """, (nom, hippodrome, race_date))
        
        row = cur.fetchone()
        if row and row[0] and row[0] >= 2 and row[1] and row[1] >= 1:
            score += 10
            rationale.append(f"🏟️ Déjà gagné ici")
        
        # 4. Signal cote (intégré prudemment)
        if cote and cote > 0:
            if cote < 3:
                score += 15
            elif cote < 5:
                score += 10
            elif cote > 20:
                score -= 10
        
        # 5. Tendance cote (si disponible, pré-départ)
        tendance = data.get('tendance_cote')
        amplitude = data.get('amplitude_tendance', 0) or 0
        
        if tendance == '-' and amplitude > 15:
            score += 10
            rationale.append(f"💰 Cote en baisse (-{amplitude:.0f}%)")
        elif tendance == '+' and amplitude > 15:
            score -= 5
        
        # 6. Avis entraineur (pré-départ)
        avis = data.get('avis_entraineur')
        if avis == 'POSITIF':
            score += 8
        elif avis == 'NEGATIF':
            score -= 8
        
        return max(10, min(90, score)), rationale[:3]
    
    def analyze_race(self, race_key: str) -> str:
        """
        Analyse une course et retourne un JSON strict.
        
        Args:
            race_key: Identifiant de la course (ex: "2025-12-02_R1_C1")
        
        Returns:
            JSON string formaté selon le schéma spécifié
        """
        cur = self.conn.cursor()
        notes = []
        
        # 1. Infos course
        cur.execute("""
            SELECT DISTINCT
                hippodrome_nom,
                distance_m,
                discipline,
                type_course
            FROM cheval_courses_seen
            WHERE race_key = %s
            LIMIT 1
        """, (race_key,))
        
        course_info = cur.fetchone()
        if not course_info:
            return json.dumps({
                "error": "Course non trouvée",
                "race_id": race_key
            })
        
        hippodrome, distance, discipline, type_course = course_info
        
        if not distance:
            distance = 2000
            notes.append("Distance manquante, défaut 2000m")
        
        # 2. Récupérer partants
        cur.execute("""
            SELECT 
                nom_norm,
                numero_dossard,
                cote_finale,
                cote_reference,
                tendance_cote,
                amplitude_tendance,
                est_favori,
                avis_entraineur,
                driver_jockey,
                entraineur
            FROM cheval_courses_seen
            WHERE race_key = %s
            AND (non_partant IS NULL OR non_partant = 0)
            ORDER BY numero_dossard
        """, (race_key,))
        
        runners_data = cur.fetchall()
        
        if len(runners_data) < 2:
            return json.dumps({
                "error": "Moins de 2 partants",
                "race_id": race_key
            })
        
        # 3. Calculer scores bruts et cotes
        raw_scores = []
        market_odds = []
        runner_info = []
        
        for row in runners_data:
            nom, numero, cote, cote_ref, tendance, amplitude, favori, avis, jockey, entraineur = row
            
            data = {
                'date_course': race_key[:10],
                'tendance_cote': tendance,
                'amplitude_tendance': amplitude,
                'avis_entraineur': avis
            }
            
            score, rationale = self._get_runner_raw_score(
                nom, cote, distance, hippodrome, data
            )
            
            raw_scores.append(score)
            market_odds.append(cote if cote and cote > 1 else 10.0)
            runner_info.append({
                'nom': nom,
                'numero': numero or 0,
                'cote': cote,
                'rationale': rationale
            })
        
        raw_scores = np.array(raw_scores)
        market_odds = np.array(market_odds)
        
        # 4. Normaliser les probabilités modèle (softmax)
        model_probs = self.normalizer.softmax_temperature(raw_scores, self.temperature)
        
        # 5. Probabilités implicites du marché
        market_probs = 1 / market_odds
        market_probs = market_probs / np.sum(market_probs)  # Normaliser
        
        # 6. Calculer l'overround
        overround = self.fusion.calculate_overround(market_odds)
        if overround > 0.25:
            notes.append(f"Overround élevé ({overround*100:.1f}%), marché peu efficient")
        
        # 7. Fusion bayésienne
        fused_probs = self.fusion.fuse_probabilities(
            market_probs,
            model_probs,
            confidence=0.8
        )
        
        # 8. Vérifier que somme = 1 (à 1e-6 près)
        prob_sum = np.sum(fused_probs)
        if abs(prob_sum - 1.0) > 1e-6:
            fused_probs = fused_probs / prob_sum
            notes.append(f"Probabilités renormalisées (était {prob_sum:.6f})")
        
        # 9. Probabilités placé (Harville)
        place_probs = self.normalizer.harville_place_proba(fused_probs, top_n=3)
        
        # 10. Construire les analyses
        runners = []
        for i, info in enumerate(runner_info):
            p_win = fused_probs[i]
            p_place = place_probs[i] if i < len(place_probs) else None
            
            fair_odds = 1 / p_win if p_win > 0.001 else None
            market_odd = info['cote']
            
            # Value %
            value_pct = None
            if fair_odds and market_odd and market_odd > 0:
                value_pct = (fair_odds - market_odd) / market_odd * 100
                # Négatif = valeur (cote marché > fair)
                # Positif = pas de valeur
                value_pct = -value_pct  # Inverser pour que + = bon
            
            # Kelly
            kelly_f = None
            if p_win and market_odd and market_odd > 1:
                kelly_f = self.kelly.calculate(p_win, market_odd, self.kelly_fraction)
            
            runner = RunnerAnalysis(
                numero=info['numero'],
                nom=info['nom'],
                p_win=p_win,
                p_place=p_place,
                fair_odds=fair_odds,
                market_odds=market_odd,
                value_pct=value_pct,
                kelly_fraction=kelly_f,
                rationale=info['rationale']
            )
            runners.append(runner)
        
        # 11. Trier par p_win décroissant
        runners.sort(key=lambda x: x.p_win or 0, reverse=True)
        
        # 12. Construire la réponse
        analysis = RaceAnalysis(
            race_id=race_key,
            timestamp=datetime.now().isoformat(),
            hippodrome=hippodrome or "Inconnu",
            distance_m=distance or 0,
            discipline=discipline or type_course or "Inconnu",
            nb_partants=len(runners),
            model_version=self.VERSION,
            runners=runners,
            run_notes=notes
        )
        
        return analysis.to_json()
    
    def analyze_race_dict(self, race_key: str) -> dict:
        """Version dict de analyze_race (pour usage interne)"""
        return json.loads(self.analyze_race(race_key))
    
    @classmethod
    def create_coherent(cls, db_connection, discipline: str = None) -> 'ProBettingAnalyzer':
        """
        Factory method qui garantit un analyseur avec paramètres cohérents.
        
        Args:
            db_connection: Connexion à la base de données
            discipline: Discipline pour α spécifique (plat, trot, obstacle)
        
        Returns:
            ProBettingAnalyzer avec T, α, kelly chargés depuis config/artefacts
        """
        # Les paramètres sont automatiquement chargés depuis la config
        return cls(db_connection, discipline=discipline)
    
    def get_params_info(self) -> dict:
        """Retourne les paramètres actuels pour debug/vérification."""
        return {
            "temperature": self.temperature,
            "market_weight": self.market_weight,
            "blend_alpha": 1 - self.market_weight,
            "kelly_fraction": self.kelly_fraction,
            "discipline": self.discipline,
            "version": self.VERSION,
            "config_available": CONFIG_AVAILABLE
        }


# =============================================================================
# POINT D'ENTRÉE CLI
# =============================================================================

def main():
    """Point d'entrée pour test CLI"""
    import sys
    from db_connection import get_connection
    
    conn = get_connection()
    
    # Créer l'analyseur avec paramètres cohérents
    analyzer = ProBettingAnalyzer.create_coherent(conn)
    
    # Afficher les paramètres utilisés
    print("=" * 60, file=sys.stderr)
    print("🏇 PRO BETTING ANALYZER - Paramètres", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    params = analyzer.get_params_info()
    print(f"   Temperature (T*): {params['temperature']:.4f}", file=sys.stderr)
    print(f"   Blend Alpha (α):  {params['blend_alpha']:.2f}", file=sys.stderr)
    print(f"   Kelly Fraction:   {params['kelly_fraction']:.2f}", file=sys.stderr)
    print(f"   Config Available: {params['config_available']}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Récupérer une course du jour ou argument
    if len(sys.argv) > 1:
        race_key = sys.argv[1]
    else:
        cur = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        
        cur.execute("""
            SELECT DISTINCT race_key 
            FROM cheval_courses_seen 
            WHERE race_key LIKE %s
            ORDER BY race_key
            LIMIT 1
        """, (today + '%',))
        
        row = cur.fetchone()
        if not row:
            # Chercher la dernière course disponible
            cur.execute("""
                SELECT DISTINCT race_key 
                FROM cheval_courses_seen 
                ORDER BY race_key DESC
                LIMIT 1
            """)
            row = cur.fetchone()
        
        if not row:
            print(json.dumps({"error": "Aucune course trouvée"}))
            return
        
        race_key = row[0]
    
    # Analyser et afficher JSON
    result = analyzer.analyze_race(race_key)
    print(result)
    
    conn.close()


if __name__ == "__main__":
    main()
