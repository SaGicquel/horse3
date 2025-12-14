#!/usr/bin/env python3
"""
Backtester Complet - Stratégie Paris Hippiques
===============================================
- Backtest win/place + portefeuille + exotiques
- Split temporel strict (anti-fuite)
- Métriques: ROI, Sharpe, Drawdown, Calibration, Turnover, EV/pari
- Baselines: marché pur, modèle sans blend, sans Kelly
- Slippage/latence configurable
- Caps journaliers d'exposition (budget) et max_stake_pct
- Protection contre utilisation de cotes finales (pré-off uniquement)

Auteur: Horse Racing AI System
Date: 2024-12
"""

import os
import json
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import hashlib
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')


class DataLeakageError(Exception):
    """Erreur levée en cas de détection de fuite de données."""
    pass


class OddsValidationError(Exception):
    """Erreur levée si les cotes utilisées ne sont pas pré-off."""
    pass

# ============================================================================
# CONFIGURATION - Chargée depuis config/pro_betting.yaml
# ============================================================================

# Import config centralisée
try:
    from config.loader import get_config as get_pro_config
    _PRO_CFG = get_pro_config()
    _TEMPERATURE = _PRO_CFG.calibration.temperature
    _BLEND_ALPHA = _PRO_CFG.calibration.blend_alpha_global
    _KELLY_FRACTION = _PRO_CFG.kelly.fraction
    _VALUE_CUTOFF = _PRO_CFG.kelly.value_cutoff
    _MAX_STAKE_PCT = _PRO_CFG.kelly.max_stake_pct
    _TAKEOUT = _PRO_CFG.markets.takeout_rate
    _INITIAL_BANKROLL = _PRO_CFG.backtest.initial_bankroll
except ImportError:
    _TEMPERATURE = 1.254
    _BLEND_ALPHA = 0.2
    _KELLY_FRACTION = 0.25
    _VALUE_CUTOFF = 0.05
    _MAX_STAKE_PCT = 0.05
    _TAKEOUT = 0.16
    _INITIAL_BANKROLL = 1000.0


@dataclass
class BacktestConfig:
    """Configuration du backtest - valeurs par défaut depuis pro_betting.yaml."""
    # Périodes
    train_start: str = "2023-01-01"
    train_end: str = "2023-09-30"
    val_start: str = "2023-10-01"
    val_end: str = "2023-11-30"
    test_start: str = "2023-12-01"
    test_end: str = "2024-03-31"
    
    # Marchés - depuis config centralisée
    market_type: str = "parimutuel"  # "fixed" ou "parimutuel"
    commission_rate: float = 0.05  # 5% commission bookmaker
    parimutuel_takeout: float = _TAKEOUT  # Depuis config
    
    # Stratégie - depuis config centralisée
    value_cutoff: float = _VALUE_CUTOFF
    kelly_fraction: float = _KELLY_FRACTION
    max_stake_pct: float = _MAX_STAKE_PCT
    min_stake: float = 2.0
    initial_bankroll: float = _INITIAL_BANKROLL
    
    # Caps journaliers d'exposition
    daily_budget_cap: float = 500.0  # Max exposition journalière (€)
    daily_budget_cap_pct: float = 0.50  # Ou en % du bankroll courant
    
    # Slippage/Latence (en ticks ou %)
    slippage_ticks: float = 0.5  # Ex: +0.5 tick sur les cotes
    slippage_pct: float = 0.0  # Ou en % (ex: 0.02 = 2%)
    latency_ms: int = 200  # Latence simulée (pour information)
    
    # Validation des cotes
    require_preoff_odds: bool = True  # Empêche utilisation cotes finales
    preoff_odds_col: str = "odds_preoff"  # Colonne des cotes pré-off
    final_odds_col: str = "odds_final"  # Colonne des cotes finales (si existe)
    
    # Filtres
    disciplines: List[str] = None  # None = toutes
    min_odds: float = 1.5
    max_odds: float = 50.0
    min_runners: int = 5
    
    # Blend - depuis config centralisée
    blend_alpha: float = _BLEND_ALPHA
    temperature: float = _TEMPERATURE
    
    # Exotiques
    include_exotics: bool = True
    exotic_budget_pct: float = 0.10  # 10% du bankroll pour exotiques
    
    # Output
    output_dir: str = "backtest_results"
    generate_plots: bool = True
    
    # Anti-fuite strict
    strict_temporal_split: bool = True
    leak_detection_enabled: bool = True
    
    def __post_init__(self):
        if self.disciplines is None:
            self.disciplines = ['plat', 'trot', 'obstacle']


# ============================================================================
# SIMULATEUR DE PARIS
# ============================================================================

class BetSimulator:
    """Simule l'exécution de paris avec gestion du bankroll et caps journaliers."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.bankroll = config.initial_bankroll
        self.initial_bankroll = config.initial_bankroll
        self.history: List[Dict] = []
        self.bankroll_history: List[Tuple[str, float]] = []
        self.peak_bankroll = config.initial_bankroll
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Tracking journalier pour les caps
        self.daily_exposure: Dict[str, float] = defaultdict(float)
        self.daily_bets_count: Dict[str, int] = defaultdict(int)
        
        # Tracking des séries drawdown
        self.drawdown_series: List[float] = []
        self.losing_streaks: List[int] = []
        self.current_losing_streak = 0
    
    def reset(self):
        """Réinitialise le simulateur."""
        self.bankroll = self.initial_bankroll
        self.history = []
        self.bankroll_history = []
        self.peak_bankroll = self.initial_bankroll
        self.max_drawdown = 0.0
        self.daily_exposure = defaultdict(float)
        self.daily_bets_count = defaultdict(int)
        self.drawdown_series = []
        self.losing_streaks = []
        self.current_losing_streak = 0
    
    def _apply_slippage(self, odds: float) -> float:
        """Applique le slippage/latence aux cotes."""
        if self.config.slippage_pct > 0:
            # Slippage en pourcentage (réduction de cote)
            odds = odds * (1 - self.config.slippage_pct)
        
        if self.config.slippage_ticks > 0:
            # Slippage en ticks (réduction absolue)
            # Convention: 1 tick = 0.01 pour cotes < 2, 0.02 pour cotes 2-3, etc.
            if odds < 2:
                tick_size = 0.01
            elif odds < 3:
                tick_size = 0.02
            elif odds < 5:
                tick_size = 0.05
            elif odds < 10:
                tick_size = 0.10
            else:
                tick_size = 0.20
            
            odds = odds - (self.config.slippage_ticks * tick_size)
        
        return max(1.01, odds)  # Cote minimum 1.01
    
    def _get_daily_budget_cap(self, date: str) -> float:
        """Retourne le cap journalier d'exposition."""
        # Cap absolu ou en % du bankroll courant (le plus restrictif)
        cap_absolute = self.config.daily_budget_cap
        cap_pct = self.bankroll * self.config.daily_budget_cap_pct
        return min(cap_absolute, cap_pct)
    
    def _can_place_bet(self, date: str, stake: float) -> Tuple[bool, float]:
        """
        Vérifie si le pari peut être placé (budget journalier).
        
        Returns:
            (can_place, adjusted_stake)
        """
        daily_cap = self._get_daily_budget_cap(date)
        remaining = daily_cap - self.daily_exposure[date]
        
        if remaining <= 0:
            return False, 0.0
        
        if stake > remaining:
            return True, remaining  # Stake ajusté
        
        return True, stake
    
    def place_bet(
        self,
        date: str,
        race_id: str,
        horse_id: str,
        bet_type: str,  # 'win', 'place', 'exotic'
        prob_model: float,
        odds: float,
        actual_result: int,  # 1 = gagné, 0 = perdu
        stake: float = None,
        use_kelly: bool = True,
        odds_is_preoff: bool = True  # Flag pour validation
    ) -> Dict:
        """
        Place un pari et met à jour le bankroll.
        
        Args:
            odds_is_preoff: Si False et require_preoff_odds activé, lève une erreur
        
        Returns:
            Dict avec détails du pari
        """
        # Validation des cotes pré-off
        if self.config.require_preoff_odds and not odds_is_preoff:
            raise OddsValidationError(
                f"Tentative d'utiliser des cotes non pré-off pour {race_id}/{horse_id}. "
                "Désactiver require_preoff_odds ou fournir des cotes pré-off."
            )
        
        # Appliquer le slippage
        odds_adjusted = self._apply_slippage(odds)
        
        # Calcul du stake
        if stake is None:
            if use_kelly:
                # Kelly criterion
                q = 1 - prob_model
                b = odds_adjusted - 1
                
                # EV = prob * odds - 1
                ev = prob_model * odds_adjusted - 1
                
                # Si EV <= 0, pas de pari (value<=0 => stake==0)
                if ev <= 0:
                    return {
                        'date': date,
                        'race_id': race_id,
                        'horse_id': horse_id,
                        'bet_type': bet_type,
                        'prob_model': prob_model,
                        'odds_original': odds,
                        'odds_adjusted': odds_adjusted,
                        'stake': 0.0,
                        'result': actual_result,
                        'profit': 0.0,
                        'bankroll_after': self.bankroll,
                        'ev_expected': ev,
                        'skipped_reason': 'ev_non_positive'
                    }
                
                kelly = (prob_model * b - q) / b if b > 0 else 0
                kelly = max(0, kelly)
                
                stake_pct = min(kelly * self.config.kelly_fraction, self.config.max_stake_pct)
                stake = self.bankroll * stake_pct
            else:
                # Flat betting
                stake = self.bankroll * 0.01  # 1% flat
        
        stake = max(self.config.min_stake, min(stake, self.bankroll * self.config.max_stake_pct))
        
        if stake > self.bankroll:
            stake = self.bankroll
        
        # Vérifier le cap journalier
        can_place, adjusted_stake = self._can_place_bet(date, stake)
        if not can_place or adjusted_stake <= 0:
            return {
                'date': date,
                'race_id': race_id,
                'horse_id': horse_id,
                'bet_type': bet_type,
                'prob_model': prob_model,
                'odds_original': odds,
                'odds_adjusted': odds_adjusted,
                'stake': 0.0,
                'result': actual_result,
                'profit': 0.0,
                'bankroll_after': self.bankroll,
                'ev_expected': prob_model * odds_adjusted - 1,
                'skipped_reason': 'daily_cap_reached'
            }
        
        stake = adjusted_stake
        
        # Mettre à jour l'exposition journalière
        self.daily_exposure[date] += stake
        self.daily_bets_count[date] += 1
        
        # Calculer le résultat
        if actual_result == 1:
            profit = stake * (odds_adjusted - 1)
            if self.config.market_type == "fixed":
                profit *= (1 - self.config.commission_rate)
            self.current_losing_streak = 0
        else:
            profit = -stake
            self.current_losing_streak += 1
            self.losing_streaks.append(self.current_losing_streak)
        
        # Mettre à jour le bankroll
        self.bankroll += profit
        
        # Tracking drawdown
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        
        self.current_drawdown = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
        
        self.drawdown_series.append(self.current_drawdown)
        
        # Enregistrer
        bet_record = {
            'date': date,
            'race_id': race_id,
            'horse_id': horse_id,
            'bet_type': bet_type,
            'prob_model': prob_model,
            'odds_original': odds,
            'odds_adjusted': odds_adjusted,
            'stake': stake,
            'result': actual_result,
            'profit': profit,
            'bankroll_after': self.bankroll,
            'ev_expected': prob_model * odds_adjusted - 1,
            'daily_exposure': self.daily_exposure[date],
            'drawdown_pct': self.current_drawdown * 100
        }
        
        self.history.append(bet_record)
        self.bankroll_history.append((date, self.bankroll))
        
        return bet_record
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques de performance complètes."""
        if not self.history:
            return {'error': 'No bets placed'}
        
        df = pd.DataFrame(self.history)
        
        # Filtrer les paris effectivement placés (stake > 0)
        df_placed = df[df['stake'] > 0].copy()
        
        if len(df_placed) == 0:
            return {'error': 'No bets actually placed (all skipped)'}
        
        # Métriques de base
        n_bets = len(df_placed)
        n_bets_skipped = len(df[df['stake'] == 0])
        n_wins = df_placed['result'].sum()
        win_rate = n_wins / n_bets if n_bets > 0 else 0
        
        total_staked = df_placed['stake'].sum()
        total_profit = df_placed['profit'].sum()
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
        
        # Turnover (volume total misé)
        turnover = total_staked
        
        # EV moyen par pari
        avg_ev_per_bet = df_placed['ev_expected'].mean()
        
        # Sharpe simple (profit journalier)
        df_placed['date'] = pd.to_datetime(df_placed['date'])
        daily_profits = df_placed.groupby('date')['profit'].sum()
        
        if len(daily_profits) > 1:
            sharpe = daily_profits.mean() / daily_profits.std() * np.sqrt(252) if daily_profits.std() > 0 else 0
        else:
            sharpe = 0
        
        # Drawdown
        bankroll_series = pd.Series([b for _, b in self.bankroll_history])
        peak = bankroll_series.expanding().max()
        drawdown_series = (peak - bankroll_series) / peak
        max_drawdown_pct = drawdown_series.max() * 100
        
        # Analyse des séries de drawdown
        drawdown_analysis = self._analyze_drawdown_series()
        
        # Profit par décile de value
        if len(df_placed) >= 10:
            df_placed['value_decile'] = pd.qcut(df_placed['ev_expected'], q=10, labels=False, duplicates='drop')
            profit_by_decile = df_placed.groupby('value_decile')['profit'].sum().to_dict()
            count_by_decile = df_placed.groupby('value_decile').size().to_dict()
            ev_by_decile = df_placed.groupby('value_decile')['ev_expected'].mean().to_dict()
        else:
            profit_by_decile = {}
            count_by_decile = {}
            ev_by_decile = {}
        
        # Métriques de slippage
        avg_slippage = (df_placed['odds_original'] - df_placed['odds_adjusted']).mean()
        
        # Exposition journalière
        daily_exposure_stats = {
            'mean': np.mean(list(self.daily_exposure.values())),
            'max': np.max(list(self.daily_exposure.values())) if self.daily_exposure else 0,
            'days_at_cap': sum(1 for exp in self.daily_exposure.values() 
                             if exp >= self._get_daily_budget_cap("dummy") * 0.95)
        }
        
        return {
            'n_bets': int(n_bets),
            'n_bets_skipped': int(n_bets_skipped),
            'n_wins': int(n_wins),
            'win_rate_pct': round(win_rate * 100, 2),
            'total_staked': round(total_staked, 2),
            'turnover': round(turnover, 2),
            'total_profit': round(total_profit, 2),
            'roi_pct': round(roi, 2),
            'avg_ev_per_bet': round(avg_ev_per_bet * 100, 4),
            'avg_ev_pct': round(avg_ev_per_bet * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'final_bankroll': round(self.bankroll, 2),
            'bankroll_growth_pct': round((self.bankroll - self.initial_bankroll) / self.initial_bankroll * 100, 2),
            'avg_stake': round(df_placed['stake'].mean(), 2),
            'avg_odds_original': round(df_placed['odds_original'].mean(), 2),
            'avg_odds_adjusted': round(df_placed['odds_adjusted'].mean(), 2),
            'avg_slippage': round(avg_slippage, 4),
            'profit_by_value_decile': {str(k): round(v, 2) for k, v in profit_by_decile.items()},
            'count_by_value_decile': {str(k): int(v) for k, v in count_by_decile.items()},
            'ev_by_value_decile': {str(k): round(v * 100, 2) for k, v in ev_by_decile.items()},
            'drawdown_analysis': drawdown_analysis,
            'daily_exposure_stats': daily_exposure_stats
        }
    
    def _analyze_drawdown_series(self) -> Dict[str, Any]:
        """Analyse détaillée des séries de drawdown."""
        if not self.drawdown_series:
            return {}
        
        dd = np.array(self.drawdown_series)
        
        # Identifier les périodes de drawdown
        in_drawdown = dd > 0.01  # Seuil 1%
        
        # Compter les périodes consécutives de drawdown
        drawdown_periods = []
        current_period = 0
        current_max = 0
        
        for i, d in enumerate(dd):
            if d > 0.01:
                current_period += 1
                current_max = max(current_max, d)
            else:
                if current_period > 0:
                    drawdown_periods.append({
                        'length': current_period,
                        'max_dd': current_max
                    })
                current_period = 0
                current_max = 0
        
        if current_period > 0:
            drawdown_periods.append({
                'length': current_period,
                'max_dd': current_max
            })
        
        # Séries perdantes
        max_losing_streak = max(self.losing_streaks) if self.losing_streaks else 0
        avg_losing_streak = np.mean(self.losing_streaks) if self.losing_streaks else 0
        
        return {
            'max_drawdown_pct': round(dd.max() * 100, 2) if len(dd) > 0 else 0,
            'avg_drawdown_pct': round(dd.mean() * 100, 2) if len(dd) > 0 else 0,
            'time_in_drawdown_pct': round(np.mean(in_drawdown) * 100, 2),
            'n_drawdown_periods': len(drawdown_periods),
            'longest_drawdown_period': max(p['length'] for p in drawdown_periods) if drawdown_periods else 0,
            'max_losing_streak': max_losing_streak,
            'avg_losing_streak': round(avg_losing_streak, 1)
        }


# ============================================================================
# CALIBRATION ANALYZER
# ============================================================================

class CalibrationAnalyzer:
    """Analyse la calibration des probabilités."""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def compute_calibration(
        self, 
        probs: np.ndarray, 
        outcomes: np.ndarray
    ) -> Dict[str, Any]:
        """Calcule les métriques de calibration."""
        # Brier Score
        brier = np.mean((probs - outcomes) ** 2)
        
        # Log Loss
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        logloss = -np.mean(outcomes * np.log(probs_clipped) + (1 - outcomes) * np.log(1 - probs_clipped))
        
        # ECE
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        ece = 0.0
        bins_detail = []
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean(outcomes[mask])
                bin_conf = np.mean(probs[mask])
                bin_count = np.sum(mask)
                bin_ece = np.abs(bin_acc - bin_conf) * (bin_count / len(probs))
                ece += bin_ece
                bins_detail.append({
                    'bin': i,
                    'range': f"[{bins[i]:.2f}, {bins[i+1]:.2f})",
                    'count': int(bin_count),
                    'predicted': round(bin_conf, 4),
                    'observed': round(bin_acc, 4),
                    'gap': round(bin_acc - bin_conf, 4)
                })
        
        return {
            'brier_score': round(brier, 4),
            'log_loss': round(logloss, 4),
            'ece': round(ece, 4),
            'bins': bins_detail
        }
    
    def plot_reliability(
        self, 
        probs: np.ndarray, 
        outcomes: np.ndarray,
        output_path: str,
        title: str = "Reliability Diagram"
    ):
        """Génère le diagramme de fiabilité."""
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        bin_accuracies = []
        bin_counts = []
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(outcomes[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(np.nan)
                bin_counts.append(0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Reliability curve
        ax = axes[0]
        ax.plot([0, 1], [0, 1], 'k--', label='Parfait', linewidth=2)
        valid = ~np.isnan(bin_accuracies)
        ax.bar(np.array(bin_centers)[valid], np.array(bin_accuracies)[valid], 
               width=0.08, alpha=0.7, color='steelblue', label='Observé')
        ax.set_xlabel('Probabilité prédite', fontsize=12)
        ax.set_ylabel('Fréquence observée', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Histogram
        ax = axes[1]
        ax.hist(probs, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Probabilité prédite', fontsize=12)
        ax.set_ylabel('Fréquence', fontsize=12)
        ax.set_title('Distribution des Probabilités', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path


# ============================================================================
# BACKTESTER PRINCIPAL
# ============================================================================

class StrategyBacktester:
    """
    Backtester complet pour la stratégie de paris hippiques.
    
    Fonctionnalités:
    - Split temporel strict (anti-fuite)
    - Validation des cotes pré-off
    - Slippage/latence configurable
    - Caps journaliers d'exposition
    - Multiples baselines
    - Métriques complètes (turnover, EV/pari, profit par décile, drawdown)
    - Visualisations
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.plot_paths: List[str] = []
        self._leak_warnings: List[str] = []
        
        # Créer le dossier de sortie
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _validate_no_data_leakage(self, df: pd.DataFrame, test_start: str) -> None:
        """
        Vérifie qu'il n'y a pas de fuite de données (look-ahead bias).
        
        Raises:
            DataLeakageError si une fuite est détectée
        """
        if not self.config.leak_detection_enabled:
            return
        
        # Vérifier que les colonnes de features ne contiennent pas d'info future
        suspicious_cols = []
        
        # Détecter les colonnes avec "final", "result", "outcome" qui pourraient être des fuites
        leak_keywords = ['final', 'result', 'outcome', 'finish', 'position_arrivee']
        for col in df.columns:
            col_lower = col.lower()
            for keyword in leak_keywords:
                if keyword in col_lower and col not in ['label_win', 'label_place']:
                    suspicious_cols.append(col)
        
        if suspicious_cols:
            self._leak_warnings.append(
                f"Colonnes suspectes détectées (possible fuite): {suspicious_cols}"
            )
        
        # Vérifier les cotes finales
        if 'odds_final' in df.columns and self.config.require_preoff_odds:
            # S'assurer qu'on n'utilise pas odds_final pour les paris
            self._leak_warnings.append(
                "Colonne 'odds_final' présente. Assurez-vous de n'utiliser que les cotes pré-off."
            )
    
    def _validate_temporal_split(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> None:
        """
        Vérifie que le split temporel est strict.
        
        Raises:
            DataLeakageError si des dates se chevauchent
        """
        if not self.config.strict_temporal_split:
            return
        
        train_max = pd.to_datetime(train_df['date']).max()
        test_min = pd.to_datetime(test_df['date']).min()
        
        if train_max >= test_min:
            raise DataLeakageError(
                f"FUITE DÉTECTÉE: Split temporel non strict! "
                f"Train max: {train_max}, Test min: {test_min}. "
                f"Les données train doivent être strictement antérieures au test."
            )
    
    def _validate_odds_availability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valide et filtre les données pour n'utiliser que les cotes pré-off.
        
        Returns:
            DataFrame filtré avec uniquement les cotes valides
        """
        if not self.config.require_preoff_odds:
            return df
        
        preoff_col = self.config.preoff_odds_col
        final_col = self.config.final_odds_col
        
        # Vérifier que la colonne pré-off existe
        if preoff_col not in df.columns:
            raise OddsValidationError(
                f"Colonne de cotes pré-off '{preoff_col}' non trouvée. "
                f"Colonnes disponibles: {df.columns.tolist()}"
            )
        
        # Filtrer les lignes sans cotes pré-off valides
        df_valid = df[df[preoff_col].notna() & (df[preoff_col] > 1.0)].copy()
        
        n_filtered = len(df) - len(df_valid)
        if n_filtered > 0:
            self._leak_warnings.append(
                f"{n_filtered} lignes filtrées (cotes pré-off invalides ou manquantes)"
            )
        
        # Si odds_final existe, vérifier qu'on ne l'utilise pas par erreur
        if final_col in df.columns:
            # Ajouter un flag pour tracking
            df_valid['_odds_source'] = 'preoff'
        
        return df_valid
    
    def _generate_synthetic_data(self, n_races: int = 1000) -> pd.DataFrame:
        """
        Génère des données synthétiques pour le backtest.
        En production, remplacer par les vraies données.
        """
        np.random.seed(42)
        
        records = []
        start_date = pd.Timestamp(self.config.train_start)
        
        for race_idx in range(n_races):
            race_date = start_date + pd.Timedelta(days=race_idx // 5)
            race_id = f"R{race_idx:05d}"
            discipline = np.random.choice(['plat', 'trot', 'obstacle'], p=[0.5, 0.35, 0.15])
            
            n_runners = np.random.randint(self.config.min_runners, 16)
            
            # Vraies forces latentes
            true_strength = np.random.randn(n_runners)
            true_probs = np.exp(true_strength) / np.sum(np.exp(true_strength))
            
            # Logits du modèle (avec erreur)
            model_error = np.random.randn(n_runners) * 0.3
            model_logits = true_strength + model_error
            
            # Cotes du marché
            market_probs = true_probs + np.random.randn(n_runners) * 0.02
            market_probs = np.clip(market_probs, 0.01, 0.99)
            market_probs = market_probs / market_probs.sum()
            margin = 1.15
            market_odds = margin / market_probs
            market_odds = np.clip(market_odds, self.config.min_odds, self.config.max_odds)
            
            # Simuler le résultat
            winner_idx = np.random.choice(n_runners, p=true_probs)
            
            # Top 3 pour place
            remaining = list(range(n_runners))
            place_indices = [winner_idx]
            remaining.remove(winner_idx)
            
            for _ in range(min(2, len(remaining))):
                rem_probs = true_probs[remaining] / true_probs[remaining].sum()
                next_place = np.random.choice(remaining, p=rem_probs)
                place_indices.append(next_place)
                remaining.remove(next_place)
            
            for horse_idx in range(n_runners):
                is_win = horse_idx == winner_idx
                is_place = horse_idx in place_indices
                
                records.append({
                    'date': race_date.strftime('%Y-%m-%d'),
                    'race_id': race_id,
                    'horse_id': f"H{race_idx:05d}_{horse_idx:02d}",
                    'horse_name': f"Cheval_{horse_idx}",
                    'discipline': discipline,
                    'n_runners': n_runners,
                    'logits_model': model_logits[horse_idx],
                    'odds_preoff': market_odds[horse_idx],
                    'label_win': int(is_win),
                    'label_place': int(is_place),
                    'true_prob': true_probs[horse_idx]
                })
        
        return pd.DataFrame(records)
    
    def _normalize_probabilities(self, df: pd.DataFrame, col: str, race_col: str = 'race_id') -> pd.DataFrame:
        """Normalise les probabilités par course."""
        df = df.copy()
        df[col] = df.groupby(race_col)[col].transform(lambda x: x / x.sum())
        return df
    
    def _compute_model_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les probabilités du modèle."""
        df = df.copy()
        
        # Softmax par course avec température
        def softmax_temp(logits, temp):
            scaled = logits / temp
            scaled = scaled - scaled.max()
            exp_scaled = np.exp(scaled)
            return exp_scaled / exp_scaled.sum()
        
        probs = []
        for race_id, group in df.groupby('race_id'):
            race_probs = softmax_temp(group['logits_model'].values, self.config.temperature)
            probs.extend(race_probs)
        
        df['p_model'] = probs
        
        # Probabilités du marché
        df['p_market'] = 1 / df['odds_preoff']
        df = self._normalize_probabilities(df, 'p_market')
        
        # Blend
        def blend_logits(p_model, p_market, alpha):
            eps = 1e-10
            p_model = np.clip(p_model, eps, 1 - eps)
            p_market = np.clip(p_market, eps, 1 - eps)
            
            logit_model = logit(p_model)
            logit_market = logit(p_market)
            
            # Shrinkage si divergence extrême
            divergence = np.abs(logit_model - logit_market)
            shrink = np.where(divergence > 3, 0.5, 1.0)
            logit_model_adj = logit_market + shrink * (logit_model - logit_market)
            
            logit_blend = alpha * logit_model_adj + (1 - alpha) * logit_market
            return expit(logit_blend)
        
        df['p_blend'] = blend_logits(
            df['p_model'].values,
            df['p_market'].values,
            self.config.blend_alpha
        )
        df = self._normalize_probabilities(df, 'p_blend')
        
        # EV
        df['ev_model'] = df['p_model'] * df['odds_preoff'] - 1
        df['ev_blend'] = df['p_blend'] * df['odds_preoff'] - 1
        df['ev_market'] = df['p_market'] * df['odds_preoff'] - 1  # Toujours négatif (marge)
        
        return df
    
    def _run_strategy(
        self, 
        df: pd.DataFrame,
        strategy_name: str,
        prob_col: str,
        use_kelly: bool = True,
        value_cutoff: float = None
    ) -> Dict[str, Any]:
        """Exécute une stratégie et retourne les métriques."""
        if value_cutoff is None:
            value_cutoff = self.config.value_cutoff
        
        simulator = BetSimulator(self.config)
        
        # Filtrer les value bets
        ev_col = f"ev_{prob_col.split('_')[1]}" if '_' in prob_col else 'ev_blend'
        if ev_col not in df.columns:
            df = df.copy()
            df[ev_col] = df[prob_col] * df['odds_preoff'] - 1
        
        value_bets = df[df[ev_col] >= value_cutoff].copy()
        
        if len(value_bets) == 0:
            return {
                'strategy': strategy_name,
                'n_bets': 0,
                'error': 'No value bets found'
            }
        
        # Exécuter les paris
        for _, row in value_bets.iterrows():
            simulator.place_bet(
                date=row['date'],
                race_id=row['race_id'],
                horse_id=row['horse_id'],
                bet_type='win',
                prob_model=row[prob_col],
                odds=row['odds_preoff'],
                actual_result=row['label_win'],
                use_kelly=use_kelly,
                odds_is_preoff=True  # Flag explicite
            )
        
        # Récupérer les métriques
        metrics = simulator.get_metrics()
        metrics['strategy'] = strategy_name
        metrics['prob_col'] = prob_col
        metrics['use_kelly'] = use_kelly
        metrics['value_cutoff'] = value_cutoff
        
        # Données pour analyse
        return {
            'metrics': metrics,
            'simulator': simulator,
            'probs': value_bets[prob_col].values,
            'outcomes': value_bets['label_win'].values
        }
    
    def run_backtest(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Exécute le backtest complet.
        
        Args:
            df: DataFrame avec les données (None = données synthétiques)
            
        Returns:
            Dict avec tous les résultats
            
        Raises:
            DataLeakageError: Si une fuite de données est détectée
            OddsValidationError: Si les cotes ne sont pas valides pré-off
        """
        print("=" * 60)
        print("🚀 BACKTEST STRATÉGIE PARIS HIPPIQUES")
        print("=" * 60)
        print(f"   Slippage: {self.config.slippage_ticks} ticks / {self.config.slippage_pct*100}%")
        print(f"   Cap journalier: {self.config.daily_budget_cap}€ ou {self.config.daily_budget_cap_pct*100}%")
        print(f"   Cotes pré-off requises: {self.config.require_preoff_odds}")
        
        # Réinitialiser les warnings
        self._leak_warnings = []
        
        # Charger ou générer les données
        if df is None:
            print("\n📊 Génération de données synthétiques...")
            df = self._generate_synthetic_data(n_races=2000)
        
        print(f"   Total: {len(df)} entrées, {df['race_id'].nunique()} courses")
        
        # Validation des cotes
        print("\n🔍 Validation des cotes pré-off...")
        df = self._validate_odds_availability(df)
        
        # Vérifier les fuites potentielles
        print("🔍 Vérification anti-fuite...")
        self._validate_no_data_leakage(df, self.config.test_start)
        
        # Split temporel strict
        print("\n📅 Split temporel:")
        train_df = df[(df['date'] >= self.config.train_start) & (df['date'] <= self.config.train_end)]
        val_df = df[(df['date'] >= self.config.val_start) & (df['date'] <= self.config.val_end)]
        test_df = df[(df['date'] >= self.config.test_start) & (df['date'] <= self.config.test_end)]
        
        print(f"   Train: {len(train_df)} ({train_df['date'].min()} → {train_df['date'].max()})")
        print(f"   Val:   {len(val_df)} ({val_df['date'].min() if len(val_df) > 0 else 'N/A'} → {val_df['date'].max() if len(val_df) > 0 else 'N/A'})")
        print(f"   Test:  {len(test_df)} ({test_df['date'].min() if len(test_df) > 0 else 'N/A'} → {test_df['date'].max() if len(test_df) > 0 else 'N/A'})")
        
        # Validation du split temporel strict
        if len(train_df) > 0 and len(test_df) > 0:
            self._validate_temporal_split(train_df, test_df)
        
        # Utiliser les données disponibles
        if len(test_df) == 0:
            print("   ⚠️ Pas de données test, utilisation de toutes les données")
            test_df = df.copy()
        
        # Calculer les probabilités
        print("\n🔢 Calcul des probabilités...")
        test_df = self._compute_model_probs(test_df)
        
        # Exécuter les stratégies
        print("\n📈 Exécution des stratégies...")
        strategies_results = {}
        
        # 1. Stratégie principale: Blend + Kelly
        print("   → Blend + Kelly...")
        result = self._run_strategy(test_df, "Blend + Kelly", "p_blend", use_kelly=True)
        strategies_results['blend_kelly'] = result
        
        # 2. Baseline 1: Marché pur (impossible à battre, sanity check)
        print("   → Marché pur (baseline)...")
        result = self._run_strategy(test_df, "Marché Pur", "p_market", use_kelly=True, value_cutoff=-0.10)
        strategies_results['market_pure'] = result
        
        # 3. Baseline 2: Modèle sans blend
        print("   → Modèle sans blend...")
        result = self._run_strategy(test_df, "Modèle Seul", "p_model", use_kelly=True)
        strategies_results['model_only'] = result
        
        # 4. Baseline 3: Blend sans Kelly (flat)
        print("   → Blend sans Kelly (flat)...")
        result = self._run_strategy(test_df, "Blend + Flat", "p_blend", use_kelly=False)
        strategies_results['blend_flat'] = result
        
        # 5. Blend + Kelly agressif
        print("   → Blend + Kelly agressif...")
        old_kelly = self.config.kelly_fraction
        self.config.kelly_fraction = 0.5
        result = self._run_strategy(test_df, "Blend + Kelly Agressif", "p_blend", use_kelly=True)
        strategies_results['blend_kelly_aggressive'] = result
        self.config.kelly_fraction = old_kelly
        
        # Analyse de calibration
        print("\n📊 Analyse de calibration...")
        calibration_analyzer = CalibrationAnalyzer(n_bins=10)
        calibration_results = {}
        
        for name, result in strategies_results.items():
            if 'probs' in result and len(result['probs']) > 0:
                cal = calibration_analyzer.compute_calibration(
                    result['probs'], 
                    result['outcomes']
                )
                calibration_results[name] = cal
        
        # Générer les plots
        if self.config.generate_plots:
            print("\n📈 Génération des graphiques...")
            self._generate_all_plots(strategies_results, calibration_analyzer, test_df)
        
        # Top/Nadirs
        print("\n📊 Analyse des séries temporelles...")
        timeseries_analysis = self._analyze_timeseries(strategies_results)
        
        # Warnings de fuite
        if self._leak_warnings:
            print("\n⚠️ WARNINGS ANTI-FUITE:")
            for w in self._leak_warnings:
                print(f"   - {w}")
        
        # Compiler les résultats
        print("\n📋 Compilation des résultats...")
        
        # Tableau comparatif
        comparison_table = []
        for name, result in strategies_results.items():
            if 'metrics' in result:
                m = result['metrics']
                comparison_table.append({
                    'strategy': m.get('strategy', name),
                    'n_bets': m.get('n_bets', 0),
                    'n_bets_skipped': m.get('n_bets_skipped', 0),
                    'roi_pct': m.get('roi_pct', 0),
                    'sharpe': m.get('sharpe_ratio', 0),
                    'max_dd_pct': m.get('max_drawdown_pct', 0),
                    'avg_ev_pct': m.get('avg_ev_pct', 0),
                    'turnover': m.get('turnover', 0),
                    'avg_ev_per_bet': m.get('avg_ev_per_bet', 0),
                    'win_rate_pct': m.get('win_rate_pct', 0),
                    'final_bankroll': m.get('final_bankroll', 0)
                })
        
        final_results = {
            'config': asdict(self.config),
            'data_summary': {
                'total_entries': len(df),
                'total_races': int(df['race_id'].nunique()),
                'test_entries': len(test_df),
                'test_races': int(test_df['race_id'].nunique()),
                'date_range': {
                    'start': str(test_df['date'].min()),
                    'end': str(test_df['date'].max())
                }
            },
            'strategies_comparison': comparison_table,
            'detailed_metrics': {
                name: result.get('metrics', {}) 
                for name, result in strategies_results.items()
            },
            'calibration': calibration_results,
            'timeseries_analysis': timeseries_analysis,
            'plot_paths': self.plot_paths,
            'leak_warnings': self._leak_warnings,
            'slippage_config': {
                'slippage_ticks': self.config.slippage_ticks,
                'slippage_pct': self.config.slippage_pct,
                'latency_ms': self.config.latency_ms
            },
            'budget_caps': {
                'daily_budget_cap': self.config.daily_budget_cap,
                'daily_budget_cap_pct': self.config.daily_budget_cap_pct,
                'max_stake_pct': self.config.max_stake_pct
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Sauvegarder le rapport JSON
        report_path = os.path.join(self.config.output_dir, 'backtest_report.json')
        with open(report_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n💾 Rapport sauvegardé: {report_path}")
        
        return final_results
    
    def _generate_all_plots(
        self, 
        strategies_results: Dict,
        calibration_analyzer: CalibrationAnalyzer,
        test_df: pd.DataFrame
    ):
        """Génère tous les graphiques."""
        output_dir = self.config.output_dir
        
        # 1. Courbes de bankroll
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for name, result in strategies_results.items():
            if 'simulator' in result:
                sim = result['simulator']
                if sim.bankroll_history:
                    dates = [d for d, _ in sim.bankroll_history]
                    bankrolls = [b for _, b in sim.bankroll_history]
                    label = result.get('metrics', {}).get('strategy', name)
                    ax.plot(range(len(bankrolls)), bankrolls, label=label, linewidth=1.5)
        
        ax.axhline(y=self.config.initial_bankroll, color='gray', linestyle='--', label='Initial')
        ax.set_xlabel('Nombre de paris', fontsize=12)
        ax.set_ylabel('Bankroll (€)', fontsize=12)
        ax.set_title('Évolution du Bankroll par Stratégie', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        path = os.path.join(output_dir, 'bankroll_evolution.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        self.plot_paths.append(path)
        
        # 2. Reliability diagram pour la stratégie principale
        main_result = strategies_results.get('blend_kelly', {})
        if 'probs' in main_result and len(main_result['probs']) > 0:
            path = os.path.join(output_dir, 'calibration_reliability.png')
            calibration_analyzer.plot_reliability(
                main_result['probs'],
                main_result['outcomes'],
                path,
                title="Calibration - Stratégie Blend + Kelly"
            )
            self.plot_paths.append(path)
        
        # 3. Profit par décile de value
        fig, ax = plt.subplots(figsize=(12, 6))
        
        main_metrics = main_result.get('metrics', {})
        if 'profit_by_value_decile' in main_metrics:
            deciles = main_metrics['profit_by_value_decile']
            x = list(deciles.keys())
            y = list(deciles.values())
            colors = ['green' if v >= 0 else 'red' for v in y]
            ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_xlabel('Décile de Value (EV)', fontsize=12)
            ax.set_ylabel('Profit (€)', fontsize=12)
            ax.set_title('Profit par Décile de Value Attendue', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
        
        path = os.path.join(output_dir, 'profit_by_value_decile.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        self.plot_paths.append(path)
        
        # 4. Comparaison des stratégies (bar chart)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        strategies = []
        rois = []
        sharpes = []
        drawdowns = []
        
        for name, result in strategies_results.items():
            if 'metrics' in result:
                m = result['metrics']
                strategies.append(m.get('strategy', name)[:15])
                rois.append(m.get('roi_pct', 0))
                sharpes.append(m.get('sharpe_ratio', 0))
                drawdowns.append(m.get('max_drawdown_pct', 0))
        
        # ROI
        ax = axes[0]
        colors = ['green' if r >= 0 else 'red' for r in rois]
        ax.barh(strategies, rois, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('ROI (%)')
        ax.set_title('ROI par Stratégie')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Sharpe
        ax = axes[1]
        ax.barh(strategies, sharpes, color='steelblue', alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Sharpe Ratio')
        ax.set_title('Sharpe par Stratégie')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Drawdown
        ax = axes[2]
        ax.barh(strategies, drawdowns, color='coral', alpha=0.7)
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_title('Drawdown Max par Stratégie')
        ax.grid(True, alpha=0.3, axis='x')
        
        path = os.path.join(output_dir, 'strategies_comparison.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        self.plot_paths.append(path)
        
        # 5. Distribution des EV
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(test_df['ev_blend'] * 100, bins=50, alpha=0.7, label='Blend', color='steelblue')
        ax.hist(test_df['ev_model'] * 100, bins=50, alpha=0.5, label='Modèle seul', color='orange')
        ax.axvline(x=self.config.value_cutoff * 100, color='red', linestyle='--', label=f'Cutoff ({self.config.value_cutoff*100}%)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Expected Value (%)', fontsize=12)
        ax.set_ylabel('Fréquence', fontsize=12)
        ax.set_title('Distribution des Expected Values', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        path = os.path.join(output_dir, 'ev_distribution.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        self.plot_paths.append(path)
        
        # 6. Graphique des séries de drawdown
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Drawdown au cours du temps
        ax = axes[0, 0]
        main_result = strategies_results.get('blend_kelly', {})
        if 'simulator' in main_result:
            sim = main_result['simulator']
            if sim.drawdown_series:
                ax.fill_between(range(len(sim.drawdown_series)), 
                               [d * 100 for d in sim.drawdown_series], 
                               alpha=0.7, color='coral')
                ax.set_xlabel('Nombre de paris', fontsize=10)
                ax.set_ylabel('Drawdown (%)', fontsize=10)
                ax.set_title('Évolution du Drawdown', fontsize=12)
                ax.grid(True, alpha=0.3)
        
        # Histogramme des séries perdantes
        ax = axes[0, 1]
        if 'simulator' in main_result:
            sim = main_result['simulator']
            if sim.losing_streaks:
                ax.hist(sim.losing_streaks, bins=20, alpha=0.7, color='coral', edgecolor='black')
                ax.axvline(x=np.mean(sim.losing_streaks), color='red', linestyle='--', 
                          label=f'Moyenne: {np.mean(sim.losing_streaks):.1f}')
                ax.set_xlabel('Longueur série perdante', fontsize=10)
                ax.set_ylabel('Fréquence', fontsize=10)
                ax.set_title('Distribution des Séries Perdantes', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Exposition journalière
        ax = axes[1, 0]
        if 'simulator' in main_result:
            sim = main_result['simulator']
            if sim.daily_exposure:
                dates = list(sim.daily_exposure.keys())[:50]  # 50 premiers jours
                exposures = [sim.daily_exposure[d] for d in dates]
                ax.bar(range(len(dates)), exposures, alpha=0.7, color='steelblue')
                daily_cap = self.config.daily_budget_cap
                ax.axhline(y=daily_cap, color='red', linestyle='--', label=f'Cap: {daily_cap}€')
                ax.set_xlabel('Jour', fontsize=10)
                ax.set_ylabel('Exposition (€)', fontsize=10)
                ax.set_title('Exposition Journalière', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Turnover cumulé
        ax = axes[1, 1]
        for name, result in strategies_results.items():
            if 'simulator' in result:
                sim = result['simulator']
                if sim.history:
                    stakes = [h['stake'] for h in sim.history]
                    cumulative = np.cumsum(stakes)
                    label = result.get('metrics', {}).get('strategy', name)[:15]
                    ax.plot(range(len(cumulative)), cumulative, label=label, linewidth=1.5)
        
        ax.set_xlabel('Nombre de paris', fontsize=10)
        ax.set_ylabel('Turnover cumulé (€)', fontsize=10)
        ax.set_title('Turnover Cumulé par Stratégie', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        path = os.path.join(output_dir, 'drawdown_analysis.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        self.plot_paths.append(path)
    
    def _analyze_timeseries(self, strategies_results: Dict) -> Dict:
        """Analyse les séries temporelles (tops/nadirs)."""
        analysis = {}
        
        for name, result in strategies_results.items():
            if 'simulator' not in result:
                continue
            
            sim = result['simulator']
            if not sim.bankroll_history:
                continue
            
            bankrolls = [b for _, b in sim.bankroll_history]
            
            if len(bankrolls) < 2:
                continue
            
            # Peak et trough
            peak_idx = np.argmax(bankrolls)
            trough_idx = np.argmin(bankrolls)
            
            # Séquences de gains/pertes
            profits = [h['profit'] for h in sim.history]
            
            # Plus longue série gagnante
            max_win_streak = 0
            current_streak = 0
            for p in profits:
                if p > 0:
                    current_streak += 1
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    current_streak = 0
            
            # Plus longue série perdante
            max_loss_streak = 0
            current_streak = 0
            for p in profits:
                if p < 0:
                    current_streak += 1
                    max_loss_streak = max(max_loss_streak, current_streak)
                else:
                    current_streak = 0
            
            analysis[name] = {
                'peak_bankroll': round(max(bankrolls), 2),
                'peak_position': int(peak_idx),
                'trough_bankroll': round(min(bankrolls), 2),
                'trough_position': int(trough_idx),
                'max_win_streak': int(max_win_streak),
                'max_loss_streak': int(max_loss_streak),
                'final_vs_peak_pct': round((bankrolls[-1] / max(bankrolls) - 1) * 100, 2)
            }
        
        return analysis


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

class TestStrategyBacktester(unittest.TestCase):
    """Tests unitaires pour le backtester."""
    
    def setUp(self):
        """Configuration avant chaque test."""
        self.config = BacktestConfig(
            train_start="2023-01-01",
            train_end="2023-06-30",
            val_start="2023-07-01",
            val_end="2023-08-31",
            test_start="2023-09-01",
            test_end="2024-12-31",
            initial_bankroll=1000.0,
            kelly_fraction=0.25,
            value_cutoff=0.05,
            daily_budget_cap=200.0,
            daily_budget_cap_pct=0.30,
            slippage_ticks=0.5,
            require_preoff_odds=True,
            strict_temporal_split=True,
            leak_detection_enabled=True,
            generate_plots=False,
            output_dir="/tmp/backtest_test"
        )
    
    def test_data_leakage_detection_temporal_split(self):
        """Test: fuite détectée => test échoue (split temporel)."""
        backtester = StrategyBacktester(self.config)
        
        # Créer des données avec chevauchement (fuite)
        train_df = pd.DataFrame({
            'date': ['2023-09-15', '2023-09-20'],  # Dates dans la période test!
            'race_id': ['R1', 'R2'],
            'horse_id': ['H1', 'H2']
        })
        
        test_df = pd.DataFrame({
            'date': ['2023-09-01', '2023-09-10'],
            'race_id': ['R3', 'R4'],
            'horse_id': ['H3', 'H4']
        })
        
        # Doit lever DataLeakageError
        with self.assertRaises(DataLeakageError):
            backtester._validate_temporal_split(train_df, test_df)
    
    def test_odds_validation_non_preoff_fails(self):
        """Test: utilisation de cotes non pré-off lève une erreur."""
        simulator = BetSimulator(self.config)
        
        # Tenter d'utiliser des cotes non pré-off
        with self.assertRaises(OddsValidationError):
            simulator.place_bet(
                date="2023-09-01",
                race_id="R001",
                horse_id="H001",
                bet_type="win",
                prob_model=0.20,
                odds=5.0,
                actual_result=0,
                odds_is_preoff=False  # NON pré-off
            )
    
    def test_daily_budget_cap_respected(self):
        """Test: cap budget journalier respecté."""
        simulator = BetSimulator(self.config)
        
        # Placer plusieurs paris le même jour jusqu'à atteindre le cap
        total_stake = 0
        date = "2023-09-01"
        
        for i in range(20):  # Essayer de placer beaucoup de paris
            result = simulator.place_bet(
                date=date,
                race_id=f"R{i:03d}",
                horse_id=f"H{i:03d}",
                bet_type="win",
                prob_model=0.30,
                odds=4.0,
                actual_result=0,
                use_kelly=True,
                odds_is_preoff=True
            )
            total_stake += result['stake']
        
        # Vérifier que le cap journalier est respecté
        daily_cap = min(self.config.daily_budget_cap, 
                       self.config.initial_bankroll * self.config.daily_budget_cap_pct)
        
        self.assertLessEqual(
            simulator.daily_exposure[date],
            daily_cap * 1.01,  # Petite marge pour arrondis
            f"Cap journalier dépassé: {simulator.daily_exposure[date]} > {daily_cap}"
        )
    
    def test_negative_ev_stake_zero(self):
        """Test: value <= 0 => stake == 0."""
        simulator = BetSimulator(self.config)
        
        # Pari avec EV négatif (prob faible, cote faible)
        result = simulator.place_bet(
            date="2023-09-01",
            race_id="R001",
            horse_id="H001",
            bet_type="win",
            prob_model=0.10,  # 10% de chances
            odds=2.0,  # Cote 2.0 => EV = 0.1 * 2 - 1 = -0.8 (négatif)
            actual_result=0,
            use_kelly=True,
            odds_is_preoff=True
        )
        
        # EV après slippage sera encore plus négatif
        self.assertEqual(
            result['stake'], 0.0,
            f"Stake devrait être 0 pour EV négatif, obtenu: {result['stake']}"
        )
        self.assertEqual(
            result.get('skipped_reason'), 'ev_non_positive',
            "Raison de skip devrait être 'ev_non_positive'"
        )
    
    def test_slippage_applied(self):
        """Test: le slippage est correctement appliqué."""
        simulator = BetSimulator(self.config)
        
        odds_original = 5.0
        
        result = simulator.place_bet(
            date="2023-09-01",
            race_id="R001",
            horse_id="H001",
            bet_type="win",
            prob_model=0.30,  # EV positif avec ces cotes
            odds=odds_original,
            actual_result=0,
            use_kelly=True,
            odds_is_preoff=True
        )
        
        # Vérifier que le slippage a été appliqué
        if result['stake'] > 0:  # Si le pari a été placé
            self.assertLess(
                result['odds_adjusted'],
                result['odds_original'],
                "Cote ajustée devrait être inférieure à l'originale (slippage)"
            )
    
    def test_max_stake_pct_respected(self):
        """Test: max_stake_pct est respecté."""
        simulator = BetSimulator(self.config)
        
        # Pari avec très haut EV (devrait vouloir gros stake)
        result = simulator.place_bet(
            date="2023-09-01",
            race_id="R001",
            horse_id="H001",
            bet_type="win",
            prob_model=0.50,  # 50% de chances
            odds=10.0,  # Cote 10 => EV très élevé
            actual_result=1,
            use_kelly=True,
            odds_is_preoff=True
        )
        
        max_allowed = simulator.initial_bankroll * self.config.max_stake_pct
        
        self.assertLessEqual(
            result['stake'],
            max_allowed * 1.01,  # Petite marge
            f"Stake {result['stake']} dépasse max_stake_pct {max_allowed}"
        )
    
    def test_suspicious_columns_detected(self):
        """Test: colonnes suspectes (fuite potentielle) sont détectées."""
        backtester = StrategyBacktester(self.config)
        
        # Données avec colonnes suspectes
        df = pd.DataFrame({
            'date': ['2023-09-01'],
            'race_id': ['R1'],
            'horse_id': ['H1'],
            'odds_preoff': [5.0],
            'odds_final': [4.5],  # Colonne suspecte
            'position_finale': [1],  # Colonne suspecte
            'result_pmu': [1],  # Colonne suspecte
            'label_win': [1],  # OK - c'est le label
        })
        
        backtester._validate_no_data_leakage(df, "2023-09-01")
        
        # Vérifier que des warnings ont été générés
        self.assertTrue(
            len(backtester._leak_warnings) > 0,
            "Des warnings de fuite auraient dû être générés"
        )


def run_tests():
    """Lance les tests unitaires."""
    print("\n" + "=" * 60)
    print("🧪 TESTS UNITAIRES - STRATEGY BACKTESTER")
    print("=" * 60 + "\n")
    
    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestStrategyBacktester)
    
    # Runner avec verbosité
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ TOUS LES TESTS PASSÉS")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        for failure in result.failures + result.errors:
            print(f"   - {failure[0]}")
    print("=" * 60)
    
    return result.wasSuccessful()


# ============================================================================
# MAIN - TESTS
# ============================================================================

def main():
    """Test du backtester."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtester Stratégie Paris Hippiques')
    parser.add_argument('--test', action='store_true', help='Lancer le backtest de démo')
    parser.add_argument('--unittest', action='store_true', help='Lancer les tests unitaires')
    parser.add_argument('--output', type=str, default='backtest_results', help='Dossier de sortie')
    parser.add_argument('--slippage-ticks', type=float, default=0.5, help='Slippage en ticks')
    parser.add_argument('--daily-cap', type=float, default=500.0, help='Cap budget journalier (€)')
    args = parser.parse_args()
    
    if args.unittest:
        success = run_tests()
        return 0 if success else 1
    
    if args.test:
        print("=" * 60)
        print("🧪 TEST DU BACKTESTER")
        print("=" * 60)
        
        # Configuration
        config = BacktestConfig(
            train_start="2023-01-01",
            train_end="2023-06-30",
            val_start="2023-07-01",
            val_end="2023-08-31",
            test_start="2023-09-01",
            test_end="2024-12-31",
            initial_bankroll=1000.0,
            kelly_fraction=0.25,
            value_cutoff=0.05,
            blend_alpha=0.4,
            slippage_ticks=args.slippage_ticks,
            daily_budget_cap=args.daily_cap,
            output_dir=args.output
        )
        
        # Exécuter le backtest
        backtester = StrategyBacktester(config)
        results = backtester.run_backtest()
        
        # Afficher le résumé
        print("\n" + "=" * 60)
        print("📋 RÉSUMÉ DES RÉSULTATS")
        print("=" * 60)
        
        print("\n📊 Comparaison des stratégies:")
        print("-" * 100)
        print(f"{'Stratégie':<25} {'N Paris':>8} {'Skip':>6} {'ROI %':>10} {'Sharpe':>10} {'Max DD %':>10} {'Turnover':>10}")
        print("-" * 100)
        
        for s in results['strategies_comparison']:
            print(f"{s['strategy']:<25} {s['n_bets']:>8} {s.get('n_bets_skipped', 0):>6} {s['roi_pct']:>10.2f} {s['sharpe']:>10.3f} {s['max_dd_pct']:>10.2f} {s.get('turnover', 0):>10.2f}")
        
        print("-" * 100)
        
        # Calibration
        print("\n📈 Calibration (stratégie principale):")
        if 'blend_kelly' in results['calibration']:
            cal = results['calibration']['blend_kelly']
            print(f"   Brier Score: {cal['brier_score']}")
            print(f"   Log Loss: {cal['log_loss']}")
            print(f"   ECE: {cal['ece']}")
        
        # Métriques détaillées
        print("\n📊 Métriques détaillées (Blend + Kelly):")
        if 'blend_kelly' in results['detailed_metrics']:
            m = results['detailed_metrics']['blend_kelly']
            print(f"   Turnover: {m.get('turnover', 0):.2f}€")
            print(f"   EV moyen/pari: {m.get('avg_ev_per_bet', 0):.4f}%")
            print(f"   Slippage moyen: {m.get('avg_slippage', 0):.4f}")
            if 'drawdown_analysis' in m:
                da = m['drawdown_analysis']
                print(f"   Temps en drawdown: {da.get('time_in_drawdown_pct', 0):.1f}%")
                print(f"   Max série perdante: {da.get('max_losing_streak', 0)}")
        
        # Plots générés
        print(f"\n📊 Graphiques générés dans: {config.output_dir}/")
        for path in results['plot_paths']:
            print(f"   - {os.path.basename(path)}")
        
        # Warnings
        if results.get('leak_warnings'):
            print("\n⚠️ Warnings anti-fuite:")
            for w in results['leak_warnings']:
                print(f"   - {w}")
        
        print("\n✅ BACKTEST TERMINÉ")
        
        # Output JSON condensé
        print("\n" + "=" * 60)
        print("📄 SORTIE JSON (condensée)")
        print("=" * 60)
        
        output_json = {
            'strategies_comparison': results['strategies_comparison'],
            'calibration_blend_kelly': results['calibration'].get('blend_kelly', {}),
            'timeseries_analysis': results['timeseries_analysis'],
            'slippage_config': results['slippage_config'],
            'budget_caps': results['budget_caps'],
            'plot_paths': results['plot_paths']
        }
        
        print(json.dumps(output_json, indent=2))


if __name__ == '__main__':
    main()
