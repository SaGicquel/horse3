#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 BETTING PORTFOLIO OPTIMIZER - Optimisation de Portefeuille de Paris
======================================================================

Construit un portefeuille optimal qui maximise l'EV sous contrainte de risque:
- Kelly fractionnaire par pari (via pari_math) avec profils: SUR/STANDARD/AMBITIEUX
- EV parimutuel avec takeout (via pari_math)
- Knapsack souple pour respecter budget et caps
- Pénalisation des corrélations (écurie/jockey/course)
- Filtre value_cutoff (≥5% par défaut)
- Gestion du drawdown
- Arrondi à 0.50€ et caps stricts

Auteur: Horse3 Pro System
Version: 2.0.0
"""

import json
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
import numpy as np

# Import des fonctions EV parimutuel centralisées
try:
    from pari_math import (
        ev_parimutuel_win,
        kelly_fraction as pari_kelly_fraction,
        kelly_fraction_raw,
        DEFAULT_TAKEOUT_RATE
    )
    PARI_MATH_AVAILABLE = True
except ImportError:
    PARI_MATH_AVAILABLE = False
    DEFAULT_TAKEOUT_RATE = 0.16

# Import config centralisée
try:
    from config.loader import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# PROFILS KELLY
# =============================================================================

KELLY_PROFILES = {
    "SUR": 0.25,        # 1/4 Kelly - Conservateur
    "STANDARD": 0.33,   # 1/3 Kelly - Équilibré (défaut)
    "AMBITIEUX": 0.50,  # 1/2 Kelly - Agressif
}


# =============================================================================
# CONFIGURATION PAR DÉFAUT
# =============================================================================

def _get_default_portfolio_config() -> dict:
    """Charge la config depuis pro_betting.yaml ou utilise les défauts."""
    if CONFIG_AVAILABLE:
        try:
            config = get_config()
            
            # Utiliser betting_defaults si disponible
            kelly_frac = config.betting_defaults.get_kelly_fraction()
            
            return {
                "kelly_profile": config.betting_defaults.kelly_profile_default,
                "kelly_fraction": kelly_frac,
                "custom_kelly_fraction": config.betting_defaults.custom_kelly_fraction,
                "max_bets": 10,
                "max_stake_per_bet": config.betting_defaults.cap_per_bet,  # 2%
                "value_cutoff": config.betting_defaults.value_cutoff,  # 5%
                "daily_budget_rate": config.betting_defaults.daily_budget_rate,  # 12%
                "max_unit_bets_per_race": config.betting_defaults.max_unit_bets_per_race,  # 2
                "rounding_increment_eur": config.betting_defaults.rounding_increment_eur,  # 0.50
                "correlation_penalty": config.portfolio.correlation_penalty,
                "max_same_race": config.portfolio.max_same_race,
                "max_same_jockey": config.portfolio.max_same_jockey,
                "max_same_trainer": config.portfolio.max_same_trainer,
                "confidence_level": config.portfolio.confidence_level,
                "markets_mode": config.markets.mode,
                "parimutuel": config.markets.mode == "parimutuel",
                "takeout_rate": config.markets.takeout_rate
            }
        except Exception as e:
            logger.warning(f"Erreur chargement config: {e}, utilisation défauts")
    
    return {
        "kelly_profile": "STANDARD",
        "kelly_fraction": float(os.getenv("BETTING_KELLY_FRACTION", 0.33)),
        "custom_kelly_fraction": float(os.getenv("BETTING_KELLY_FRACTION", 0.33)),
        "max_bets": 10,
        "max_stake_per_bet": 0.02,  # Cap 2% par pari
        "value_cutoff": float(os.getenv("BETTING_VALUE_CUTOFF", 0.05)),        # Cutoff 5%
        "daily_budget_rate": 0.12,   # 12% bankroll/jour
        "max_unit_bets_per_race": 2,
        "rounding_increment_eur": 0.5,
        "correlation_penalty": 0.3,
        "max_same_race": 2,
        "max_same_jockey": 3,
        "max_same_trainer": 3,
        "confidence_level": 0.95,
        "markets_mode": "parimutuel",
        "parimutuel": True,
        "takeout_rate": DEFAULT_TAKEOUT_RATE,
        "drawdown_limit_rate": 0.30,   # Cible DD95 <= 30% bankroll
        "drawdown_confidence": 0.95    # Niveau de confiance pour DD
    }


DEFAULT_PORTFOLIO_CONFIG = _get_default_portfolio_config()


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class BetCandidate:
    """Candidat au portefeuille"""
    horse_id: str
    name: str
    market: str  # WIN, PLACE, etc.
    p: float     # Probabilité estimée
    odds: float  # Cote
    ev: float    # Expected Value = p * odds - 1
    variance_est: float = 0.0
    corr_group: str = None  # Groupe de corrélation (race_id, trainer, jockey)
    race_id: str = None
    jockey: str = None
    trainer: str = None
    parimutuel: bool = True
    takeout_rate: float = DEFAULT_TAKEOUT_RATE
    
    # Calculés
    kelly_raw: float = 0.0
    kelly_adjusted: float = 0.0
    stake: float = 0.0
    ev_stake: float = 0.0
    selected: bool = False
    exclusion_reason: str = None
    
    def __post_init__(self):
        # Calculer Kelly brut
        if self.odds > 1 and 0 < self.p < 1:
            if PARI_MATH_AVAILABLE:
                self.kelly_raw = kelly_fraction_raw(self.p, self.odds)
            else:
                q = 1 - self.p
                b = self.odds - 1
                self.kelly_raw = max(0, (b * self.p - q) / b)
        
        # Variance estimée si non fournie
        if self.variance_est <= 0:
            self.variance_est = self.p * (1 - self.p)
        
        # Recalculer l'EV avec pari_math si disponible et mode parimutuel
        if PARI_MATH_AVAILABLE and self.parimutuel and self.ev == 0:
            self.ev = ev_parimutuel_win(self.p, self.odds, self.takeout_rate)


@dataclass
class PortfolioResult:
    """Résultat de l'optimisation"""
    budget_today: float
    kelly_fraction: float
    selection: List[Dict]
    excluded: List[Dict]
    summary: Dict
    run_notes: List[str]
    # Nouvelles propriétés pour l'UI
    profile_used: str = "STANDARD"
    kelly_fraction_effective: float = 0.33
    caps: Dict = field(default_factory=dict)
    budget_left: float = 0.0
    
    def to_json(self) -> str:
        return json.dumps({
            "budget_today": self.budget_today,
            "kelly_fraction": self.kelly_fraction,
            "profile_used": self.profile_used,
            "kelly_fraction_effective": self.kelly_fraction_effective,
            "caps": self.caps,
            "budget_left": self.budget_left,
            "selection": self.selection,
            "excluded": self.excluded,
            "summary": self.summary,
            "run_notes": self.run_notes
        }, ensure_ascii=False, indent=2)


# =============================================================================
# CALCULS FINANCIERS
# =============================================================================

class RiskCalculator:
    """Calculs de risque pour le portefeuille"""
    
    @staticmethod
    def expected_drawdown_95(stakes: List[float], probs: List[float], odds: List[float]) -> float:
        """
        Estime le drawdown au 95e percentile via simulation Monte Carlo.
        
        En simplifié: on calcule la perte potentielle si tous les paris perdent,
        puis on ajuste pour le niveau de confiance.
        """
        if not stakes:
            return 0.0
        
        n = len(stakes)
        total_stake = sum(stakes)
        
        # Simulation simplifiée: calcul analytique
        # Expected loss = sum(stake_i * (1 - p_i))
        expected_loss = sum(s * (1 - p) for s, p in zip(stakes, probs))
        
        # Variance de la perte
        # Var(Loss) = sum(stake_i^2 * p_i * (1-p_i))
        var_loss = sum(s**2 * p * (1-p) for s, p in zip(stakes, probs))
        std_loss = np.sqrt(var_loss)
        
        # Drawdown 95%: percentile basé sur approximation normale
        # Pour 95%, z ≈ 1.645
        drawdown_95 = expected_loss + 1.645 * std_loss
        
        return min(drawdown_95, total_stake)  # Cap au total misé
    
    @staticmethod
    def sharpe_ratio(ev_total: float, var_total: float, stake_total: float) -> float:
        """Calcule un ratio de Sharpe simplifié pour le portefeuille"""
        if var_total <= 0 or stake_total <= 0:
            return 0.0
        
        std_total = np.sqrt(var_total)
        return ev_total / std_total if std_total > 0 else 0.0
    
    @staticmethod
    def portfolio_variance(
        stakes: List[float], 
        variances: List[float],
        correlations: np.ndarray = None
    ) -> float:
        """
        Calcule la variance du portefeuille.
        
        Sans corrélations: Var = sum(stake_i^2 * var_i)
        Avec corrélations: Var = w'Σw (forme matricielle)
        """
        n = len(stakes)
        if n == 0:
            return 0.0
        
        stakes = np.array(stakes)
        variances = np.array(variances)
        
        if correlations is None:
            # Indépendance assumée
            return np.sum(stakes**2 * variances)
        else:
            # Avec matrice de corrélation
            std_devs = np.sqrt(variances)
            cov_matrix = np.outer(std_devs, std_devs) * correlations
            return stakes @ cov_matrix @ stakes


# =============================================================================
# OPTIMISEUR DE PORTEFEUILLE
# =============================================================================

class BettingPortfolioOptimizer:
    """
    Optimise un portefeuille de paris sous contraintes de risque.
    
    Utilise pari_math pour les calculs EV parimutuel si disponible.
    """
    
    VERSION = "1.1.0"
    
    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_PORTFOLIO_CONFIG, **(config or {})}
        self.risk_calc = RiskCalculator()
    
    def _build_correlation_matrix(self, candidates: List[BetCandidate]) -> np.ndarray:
        """
        Construit une matrice de corrélation approximative.
        
        Hypothèses:
        - Même course: corrélation = -1/(n-1) (compétition)
        - Même jockey/écurie: corrélation = 0.3 (style similaire)
        - Sinon: 0 (indépendants)
        """
        n = len(candidates)
        corr = np.eye(n)  # Diagonale = 1
        
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = candidates[i], candidates[j]
                
                # Même course = corrélation négative (un seul gagnant)
                if ci.race_id and cj.race_id and ci.race_id == cj.race_id:
                    # Corrélation négative: si un gagne, l'autre perd
                    corr[i, j] = -0.5
                    corr[j, i] = -0.5
                
                # Même jockey ou écurie = légère corrélation positive
                elif (ci.jockey and cj.jockey and ci.jockey == cj.jockey) or \
                     (ci.trainer and cj.trainer and ci.trainer == cj.trainer):
                    corr[i, j] = 0.2
                    corr[j, i] = 0.2
                
                # Même groupe de corrélation
                elif ci.corr_group and cj.corr_group and ci.corr_group == cj.corr_group:
                    corr[i, j] = self.config["correlation_penalty"]
                    corr[j, i] = self.config["correlation_penalty"]
        
        return corr
    
    def _apply_concentration_limits(
        self, 
        candidates: List[BetCandidate]
    ) -> Tuple[List[BetCandidate], List[str]]:
        """
        Applique les limites de concentration.
        Retourne (candidats filtrés, notes)
        """
        notes = []
        
        # Compter par groupe
        race_counts = {}
        jockey_counts = {}
        trainer_counts = {}
        
        filtered = []
        
        for c in candidates:
            # Vérifier limites
            race_ok = True
            jockey_ok = True
            trainer_ok = True
            
            if c.race_id:
                race_counts[c.race_id] = race_counts.get(c.race_id, 0)
                if race_counts[c.race_id] >= self.config["max_same_race"]:
                    race_ok = False
                    c.exclusion_reason = f"max_same_race ({self.config['max_same_race']})"
            
            if c.jockey:
                jockey_counts[c.jockey] = jockey_counts.get(c.jockey, 0)
                if jockey_counts[c.jockey] >= self.config["max_same_jockey"]:
                    jockey_ok = False
                    c.exclusion_reason = f"max_same_jockey ({self.config['max_same_jockey']})"
            
            if c.trainer:
                trainer_counts[c.trainer] = trainer_counts.get(c.trainer, 0)
                if trainer_counts[c.trainer] >= self.config["max_same_trainer"]:
                    trainer_ok = False
                    c.exclusion_reason = f"max_same_trainer ({self.config['max_same_trainer']})"
            
            if race_ok and jockey_ok and trainer_ok:
                filtered.append(c)
                if c.race_id:
                    race_counts[c.race_id] += 1
                if c.jockey:
                    jockey_counts[c.jockey] += 1
                if c.trainer:
                    trainer_counts[c.trainer] += 1
        
        excluded_count = len(candidates) - len(filtered)
        if excluded_count > 0:
            notes.append(f"{excluded_count} paris exclus pour concentration")
        
        return filtered, notes
    
    def _kelly_allocation(
        self,
        candidates: List[BetCandidate],
        bankroll: float,
        budget_today: float
    ) -> List[BetCandidate]:
        """
        Applique l'allocation Kelly fractionnaire avec caps et arrondi.
        
        Formules:
        - Kelly plein: f* = (p*(o-1) - (1-p)) / (o-1)
        - Stake rate proposé: rate = min(kelly_fraction * f*, cap_per_bet)
        - Stake €: stake = bankroll * rate, arrondi au rounding_increment_eur
        
        Utilise pari_math.kelly_fraction si disponible pour le mode parimutuel.
        """
        kelly_frac = self.config["kelly_fraction"]
        cap_per_bet = self.config["max_stake_per_bet"]  # 2% par défaut
        parimutuel = self.config.get("parimutuel", True)
        takeout = self.config.get("takeout_rate", DEFAULT_TAKEOUT_RATE)
        rounding = self.config.get("rounding_increment_eur", 0.5)
        
        for c in candidates:
            # Kelly avec pari_math si disponible
            if PARI_MATH_AVAILABLE and parimutuel:
                kelly_adjusted_raw = pari_kelly_fraction(
                    c.p, c.odds, kelly_frac, cap_per_bet, parimutuel, takeout
                )
            else:
                # Fallback: Kelly classique
                kelly_adjusted_raw = c.kelly_raw * kelly_frac
            
            # Appliquer le cap
            c.kelly_adjusted = min(kelly_adjusted_raw, cap_per_bet)
            
            # Stake en euros AVANT arrondi
            stake_raw = bankroll * c.kelly_adjusted
            
            # Arrondir au pas rounding_increment_eur (0.50€ par défaut)
            if rounding > 0:
                c.stake = round(stake_raw / rounding) * rounding
            else:
                c.stake = round(stake_raw, 2)
            
            # Si stake arrondi dépasse le cap, réduire
            max_stake_eur = bankroll * cap_per_bet
            if c.stake > max_stake_eur:
                c.stake = math.floor(max_stake_eur / rounding) * rounding if rounding > 0 else round(max_stake_eur, 2)
            
            # EV du stake
            c.ev_stake = c.stake * c.ev
        
        return candidates
    
    def _knapsack_selection(
        self,
        candidates: List[BetCandidate],
        budget: float,
        max_bets: int
    ) -> List[BetCandidate]:
        """
        Sélection type knapsack: maximiser EV sous contrainte de budget.
        
        Algorithme glouton: trier par EV/stake (efficiency), puis prendre
        jusqu'à épuisement du budget ou max_bets.
        """
        # Trier par ratio EV/stake (efficiency) décroissant
        # Puis par variance croissante (robustesse) en cas d'égalité
        candidates_with_stake = [c for c in candidates if c.stake > 0]
        
        candidates_with_stake.sort(
            key=lambda c: (
                -c.ev_stake / max(c.stake, 0.01),  # Efficiency
                c.variance_est  # Robustesse (variance faible = mieux)
            )
        )
        
        selected = []
        remaining_budget = budget
        
        for c in candidates_with_stake:
            if len(selected) >= max_bets:
                c.exclusion_reason = f"max_bets ({max_bets})"
                continue
            
            if c.stake <= remaining_budget:
                c.selected = True
                selected.append(c)
                remaining_budget -= c.stake
            else:
                # Réduire le stake pour rentrer dans le budget
                if remaining_budget >= 1:  # Minimum 1€
                    c.stake = round(remaining_budget, 2)
                    c.ev_stake = c.stake * c.ev
                    c.selected = True
                    selected.append(c)
                    remaining_budget = 0
                else:
                    c.exclusion_reason = "budget_insuffisant"
        
        return selected
    
    def optimize(
        self,
        bets: List[Dict],
        bankroll: float,
        budget_today: float = None,
        kelly_profile: str = None
    ) -> PortfolioResult:
        """
        Optimise le portefeuille de paris.
        
        Args:
            bets: Liste de paris candidats
            bankroll: Bankroll total
            budget_today: Budget du jour (défaut: daily_budget_rate * bankroll = 12%)
            kelly_profile: Profil Kelly (SUR, STANDARD, AMBITIEUX, PERSONNALISE)
        
        Returns:
            PortfolioResult avec sélection optimale
        """
        # Déterminer le profil Kelly et la fraction effective
        profile = kelly_profile or self.config.get("kelly_profile", "STANDARD")
        profile = profile.upper()
        
        if profile == "PERSONNALISE":
            kelly_frac = self.config.get("custom_kelly_fraction", 0.33)
        elif profile in KELLY_PROFILES:
            kelly_frac = KELLY_PROFILES[profile]
        else:
            kelly_frac = KELLY_PROFILES["STANDARD"]
            profile = "STANDARD"
        
        # Mettre à jour la config avec la fraction effective
        self.config["kelly_fraction"] = kelly_frac
        
        # Budget du jour par défaut = daily_budget_rate * bankroll (12%)
        daily_rate = self.config.get("daily_budget_rate", 0.12)
        if budget_today is None:
            budget_today = bankroll * daily_rate
        
        cap_per_bet = self.config.get("max_stake_per_bet", 0.02)
        rounding = self.config.get("rounding_increment_eur", 0.5)
        
        run_notes = []
        run_notes.append(f"Profil: {profile} ({kelly_frac:.0%} Kelly)")
        run_notes.append(f"Budget jour: {budget_today:.2f}€ ({daily_rate:.0%} de {bankroll:.0f}€)")
        run_notes.append(f"Cap/pari: {cap_per_bet:.0%} = {bankroll * cap_per_bet:.2f}€")
        
        # 1. Convertir en BetCandidate
        parimutuel = self.config.get("parimutuel", True)
        takeout = self.config.get("takeout_rate", DEFAULT_TAKEOUT_RATE)
        
        candidates = []
        for b in bets:
            try:
                c = BetCandidate(
                    horse_id=b.get("horse_id", "unknown"),
                    name=b.get("name", b.get("horse_id", "unknown")),
                    market=b.get("market", "WIN"),
                    p=float(b.get("p", 0)),
                    odds=float(b.get("odds", 1)),
                    ev=float(b.get("ev", 0)),
                    variance_est=float(b.get("variance_est", 0)),
                    corr_group=b.get("corr_group"),
                    race_id=b.get("race_id"),
                    jockey=b.get("jockey"),
                    trainer=b.get("trainer"),
                    parimutuel=parimutuel,
                    takeout_rate=takeout
                )
                candidates.append(c)
            except (ValueError, TypeError) as e:
                run_notes.append(f"Pari ignoré (données invalides): {b.get('horse_id', '?')}")
        
        run_notes.append(f"{len(candidates)} paris candidats reçus")
        if parimutuel:
            run_notes.append(f"mode=parimutuel, takeout={takeout:.0%}")
        
        # 2. Filtrer par value_cutoff (≥5% par défaut)
        value_cutoff = self.config["value_cutoff"]
        filtered = []
        excluded_value = 0
        
        for c in candidates:
            # value <= 0 => stake = 0
            if c.ev <= 0:
                c.exclusion_reason = f"value ({c.ev:.2%}) ≤ 0 → stake=0"
                excluded_value += 1
            elif c.ev < value_cutoff:
                c.exclusion_reason = f"value ({c.ev:.2%}) < cutoff ({value_cutoff:.0%})"
                excluded_value += 1
            else:
                filtered.append(c)
        
        if excluded_value > 0:
            run_notes.append(f"{excluded_value} paris exclus (value < {value_cutoff:.0%} ou ≤0)")
        
        # 3. Appliquer limites de concentration
        filtered, conc_notes = self._apply_concentration_limits(filtered)
        run_notes.extend(conc_notes)
        
        # 4. Allocation Kelly
        filtered = self._kelly_allocation(filtered, bankroll, budget_today)
        
        # 5. Sélection knapsack
        max_bets = self.config["max_bets"]
        selected = self._knapsack_selection(filtered, budget_today, max_bets)
        
        # 6. Calculs de risque + rescaling si drawdown dépasse la borne
        def compute_metrics(sel: List[BetCandidate]):
            if not sel:
                return 0, 0, 0, 0, 0, np.eye(0)
            stakes = [c.stake for c in sel]
            probs = [c.p for c in sel]
            odds = [c.odds for c in sel]
            variances = [c.variance_est for c in sel]
            corr_matrix = self._build_correlation_matrix(sel)
            ev_total = sum(c.ev_stake for c in sel)
            stake_total = sum(stakes)
            var_total = self.risk_calc.portfolio_variance(stakes, variances, corr_matrix)
            drawdown_95 = self.risk_calc.expected_drawdown_95(stakes, probs, odds)
            sharpe = self.risk_calc.sharpe_ratio(ev_total, var_total, stake_total)
            return ev_total, stake_total, var_total, drawdown_95, sharpe, corr_matrix
        
        ev_total, stake_total, var_total, drawdown_95, sharpe, corr_matrix = compute_metrics(selected)
        
        dd_limit = bankroll * self.config.get("drawdown_limit_rate", 0.30)
        rounding = self.config.get("rounding_increment_eur", 0.5)
        cap_per_bet = self.config.get("max_stake_per_bet", 0.02)
        
        if selected and drawdown_95 > dd_limit and drawdown_95 > 0:
            scale = dd_limit / drawdown_95
            run_notes.append(f"Drawdown95 {drawdown_95:.2f}€ > limite {dd_limit:.2f}€ → scaling {scale:.2f}")
            for c in selected:
                c.stake = max(0, c.stake * scale)
                if rounding > 0:
                    c.stake = round(c.stake / rounding) * rounding
                max_cap_eur = bankroll * cap_per_bet
                if c.stake > max_cap_eur:
                    c.stake = math.floor(max_cap_eur / rounding) * rounding if rounding > 0 else max_cap_eur
                c.ev_stake = c.stake * c.ev
            # Recalcul après scaling
            ev_total, stake_total, var_total, drawdown_95, sharpe, corr_matrix = compute_metrics(selected)
        elif selected and drawdown_95 > 0:
            run_notes.append(f"Drawdown95 {drawdown_95:.2f}€ ≤ limite {dd_limit:.2f}€")
        
        if selected and all(c.stake <= bankroll * cap_per_bet * 1.01 for c in selected):
            run_notes.append(f"cap {cap_per_bet:.0%} respecté")
        if not selected:
            run_notes.append("Aucun pari sélectionné")
        
        # 7. Construire la sortie
        selection_output = []
        for c in selected:
            selection_output.append({
                "horse_id": c.horse_id,
                "name": c.name,
                "market": c.market,
                "odds": c.odds,
                "p": round(c.p, 4),
                "ev": round(c.ev, 4),
                "stake": c.stake,
                "ev_stake": round(c.ev_stake, 2),
                "kelly_raw": round(c.kelly_raw, 4),
                "kelly_adjusted": round(c.kelly_adjusted, 4),
                "reason": "value>=cutoff & concentration_ok"
            })
        
        # Trier par EV décroissante puis variance croissante
        selection_output.sort(key=lambda x: (-x["ev_stake"], x["ev"]))
        
        # Exclus
        excluded_output = []
        for c in candidates:
            if not c.selected and c.exclusion_reason:
                excluded_output.append({
                    "horse_id": c.horse_id,
                    "name": c.name,
                    "ev": round(c.ev, 4),
                    "reason": c.exclusion_reason
                })
        
        # Calculer budget_left
        budget_left = budget_today - stake_total
        
        return PortfolioResult(
            budget_today=budget_today,
            kelly_fraction=kelly_frac,
            profile_used=profile,
            kelly_fraction_effective=kelly_frac,
            caps={
                "cap_per_bet": cap_per_bet,
                "cap_per_bet_eur": round(bankroll * cap_per_bet, 2),
                "daily_budget_rate": daily_rate,
                "daily_budget_eur": round(budget_today, 2),
                "rounding_increment_eur": rounding,
                "max_unit_bets_per_race": self.config.get("max_unit_bets_per_race", 2)
            },
            budget_left=round(budget_left, 2),
            selection=selection_output,
            excluded=excluded_output,
            summary={
                "ev_total": round(ev_total, 2),
                "stake_total": round(stake_total, 2),
                "nb_bets": len(selected),
                "expected_drawdown_95": round(drawdown_95, 2),
                "portfolio_variance": round(var_total, 4),
                "sharpe_ratio": round(sharpe, 3),
                "roi_expected": round(ev_total / stake_total * 100, 2) if stake_total > 0 else 0
            },
            run_notes=run_notes
        )
    
    def optimize_from_pronostics(
        self,
        pronostics: List[Dict],
        bankroll: float,
        budget_today: float = None
    ) -> PortfolioResult:
        """
        Optimise à partir de sorties du RacePronosticGenerator.
        
        Args:
            pronostics: Liste de résultats de generate_pronostic_dict()
            bankroll: Bankroll total
            budget_today: Budget du jour
        """
        # Extraire tous les paris recommandés
        all_bets = []
        
        for prono in pronostics:
            race_id = prono.get("race_id", "")
            
            for runner in prono.get("runners", []):
                # Ne prendre que les value bets
                value = runner.get("value_win")
                if value is not None and value > 0:
                    all_bets.append({
                        "horse_id": runner.get("horse_id", runner.get("name")),
                        "name": runner.get("name"),
                        "market": "WIN",
                        "p": runner.get("p_win", 0),
                        "odds": runner.get("market_odds_win", 10),
                        "ev": value,
                        "variance_est": runner.get("p_win", 0.1) * (1 - runner.get("p_win", 0.1)),
                        "race_id": race_id,
                        "jockey": runner.get("jockey"),
                        "trainer": runner.get("trainer")
                    })
        
        return self.optimize(all_bets, bankroll, budget_today)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Point d'entrée CLI - exemple avec données de test"""
    import sys
    
    # Données de test
    test_bets = [
        {"horse_id": "cheval_1", "name": "Missil Va I Ve", "market": "WIN", 
         "p": 0.54, "odds": 5.4, "ev": 1.92, "race_id": "R1", "jockey": "J1"},
        {"horse_id": "cheval_2", "name": "Mohawk", "market": "WIN", 
         "p": 0.78, "odds": 6.4, "ev": 3.96, "race_id": "R2", "jockey": "J2"},
        {"horse_id": "cheval_3", "name": "Lady's Os", "market": "WIN", 
         "p": 0.16, "odds": 62.0, "ev": 8.89, "race_id": "R2", "jockey": "J3"},
        {"horse_id": "cheval_4", "name": "Kocktail Love", "market": "WIN", 
         "p": 0.013, "odds": 145.0, "ev": 0.82, "race_id": "R2", "jockey": "J4"},
        {"horse_id": "cheval_5", "name": "Malvasia", "market": "WIN", 
         "p": 0.17, "odds": 2.6, "ev": -0.57, "race_id": "R1", "jockey": "J5"},  # Value négative
        {"horse_id": "cheval_6", "name": "Test Low", "market": "WIN", 
         "p": 0.05, "odds": 15.0, "ev": 0.03, "race_id": "R3", "jockey": "J1"},  # Value < cutoff
    ]
    
    # Si argument JSON fourni
    if len(sys.argv) > 1 and sys.argv[1] != "--test":
        try:
            input_data = json.loads(sys.argv[1])
            bets = input_data.get("bets", input_data)
            bankroll = input_data.get("bankroll", 1000)
            budget = input_data.get("budget_today", bankroll * 0.2)
        except json.JSONDecodeError:
            print(json.dumps({"error": "JSON invalide"}))
            return
    else:
        bets = test_bets
        bankroll = 1000
        budget = 200
    
    # Optimiser
    optimizer = BettingPortfolioOptimizer()
    result = optimizer.optimize(bets, bankroll=bankroll, budget_today=budget)
    
    print(result.to_json())


if __name__ == "__main__":
    main()
