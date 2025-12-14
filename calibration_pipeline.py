#!/usr/bin/env python3
"""
Pipeline de Calibration et Normalisation par Course
====================================================
- Normalisation softmax avec température apprise
- Calibration Platt + Isotonic avec sélection automatique
- Blend modèle-marché avec α optimisé par CV
- **Correction biais favori/outsider (gamma)**
- **Alpha dynamique (temps, volume)**
- Évaluation: Brier, ECE, NLL, courbes de fiabilité
- Walk-forward par date

Auteur: Horse Racing AI System
Date: 2024-12
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib

import numpy as np
import pandas as pd

# Import du loader de configuration centralisé
try:
    from config.loader import update_config_from_calibration, get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️ config.loader non disponible - les params ne seront pas exportés vers config/pro_betting.yaml")
from scipy.special import softmax, expit, logit
from scipy.optimize import minimize_scalar, minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Pour serveur sans display

# Import du module de correction de biais marché
try:
    from market_debiaser import (
        MarketDebiasConfig, 
        MarketGammaCorrector, 
        DynamicAlphaCalculator,
        DynamicMarketModelBlender,
        DebiasMetrics
    )
    DEBIASER_AVAILABLE = True
except ImportError:
    DEBIASER_AVAILABLE = False
    print("⚠️ market_debiaser non disponible - blend classique utilisé")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CalibrationConfig:
    """Configuration de la pipeline de calibration."""
    # Chemins
    artifacts_dir: str = "calibration_artifacts"
    
    # Softmax
    temperature_init: float = 1.5
    temperature_bounds: Tuple[float, float] = (0.5, 5.0)
    
    # Calibration
    n_bins_ece: int = 10
    min_samples_cluster: int = 200  # Min samples pour calibration par cluster
    
    # Blend
    alpha_grid: List[float] = None  # Grille pour α
    alpha_cv_folds: int = 5
    
    # Correction biais favori/outsider (gamma)
    use_gamma_correction: bool = True  # Activer correction p_mkt^γ
    gamma_init: float = 1.0
    gamma_bounds: Tuple[float, float] = (0.5, 2.0)
    gamma_by_cluster: bool = True  # γ par discipline/hippodrome
    
    # Alpha dynamique (temps/volume)
    use_dynamic_alpha: bool = True  # α = f(pool, time)
    alpha_bounds: Tuple[float, float] = (0.3, 0.9)
    alpha_base: float = 0.5  # a0
    alpha_pool_coef: float = 0.05  # a1 · log(pool)
    alpha_time_coef: float = 0.01  # a2 · minutes_to_off
    default_pool_size: float = 50000.0
    default_minutes_to_off: float = 30.0
    
    # Walk-forward
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    min_train_samples: int = 500
    
    # Évaluation
    profit_sim_stake: float = 10.0  # Stake pour simulation
    value_threshold: float = 0.05  # Seuil EV pour parier
    
    def __post_init__(self):
        if self.alpha_grid is None:
            self.alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ============================================================================
# MÉTRIQUES DE CALIBRATION
# ============================================================================

class CalibrationMetrics:
    """Calcul des métriques de calibration."""
    
    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Brier Score (MSE des probabilités)."""
        return float(np.mean((y_prob - y_true) ** 2))
    
    @staticmethod
    def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
        """Negative Log-Likelihood."""
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
    
    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        n_bins: int = 10
    ) -> Tuple[float, List[Dict]]:
        """
        Expected Calibration Error avec détail par bin.
        
        Returns:
            ece: float - ECE global
            bins_detail: List[Dict] - Détail par bin
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        bins_detail = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                bin_count = np.sum(mask)
                bin_ece = np.abs(bin_acc - bin_conf) * (bin_count / len(y_prob))
                ece += bin_ece
                bins_detail.append({
                    'bin': i,
                    'range': f"[{bins[i]:.2f}, {bins[i+1]:.2f})",
                    'count': int(bin_count),
                    'accuracy': float(bin_acc),
                    'confidence': float(bin_conf),
                    'gap': float(bin_acc - bin_conf)
                })
        
        return float(ece), bins_detail
    
    @staticmethod
    def reliability_curve(
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Courbe de fiabilité (reliability diagram).
        
        Returns:
            bin_centers: np.ndarray
            bin_accuracies: np.ndarray
            bin_counts: np.ndarray
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_accuracies = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_accuracies[i] = np.mean(y_true[mask])
                bin_counts[i] = np.sum(mask)
            else:
                bin_accuracies[i] = np.nan
        
        return bin_centers, bin_accuracies, bin_counts


# ============================================================================
# NORMALISATION SOFTMAX AVEC TEMPÉRATURE
# ============================================================================

class TemperatureSoftmax:
    """Normalisation softmax avec température apprise."""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.temperature: float = config.temperature_init
        self.fitted: bool = False
    
    def _softmax_with_temp(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Softmax avec température sur un groupe (une course)."""
        scaled = logits / temperature
        # Stabilité numérique
        scaled = scaled - np.max(scaled)
        exp_scaled = np.exp(scaled)
        return exp_scaled / np.sum(exp_scaled)
    
    def _nll_loss(
        self, 
        temperature: float, 
        logits_by_race: List[np.ndarray],
        labels_by_race: List[np.ndarray]
    ) -> float:
        """NLL pour optimiser la température."""
        total_nll = 0.0
        total_samples = 0
        
        for logits, labels in zip(logits_by_race, labels_by_race):
            probs = self._softmax_with_temp(logits, temperature)
            # NLL = -sum(y * log(p))
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            nll = -np.sum(labels * np.log(probs))
            total_nll += nll
            total_samples += len(labels)
        
        return total_nll / total_samples if total_samples > 0 else 1e10
    
    def fit(
        self, 
        df: pd.DataFrame,
        logits_col: str = 'logits_model',
        race_col: str = 'race_id',
        label_col: str = 'label_win'
    ) -> 'TemperatureSoftmax':
        """
        Apprend la température optimale.
        
        Args:
            df: DataFrame avec logits, race_id, labels
            
        Returns:
            self
        """
        # Grouper par course
        logits_by_race = []
        labels_by_race = []
        
        for race_id, group in df.groupby(race_col):
            if group[label_col].sum() == 1:  # Course avec un gagnant unique
                logits_by_race.append(group[logits_col].values)
                labels_by_race.append(group[label_col].values)
        
        if len(logits_by_race) < 10:
            print(f"⚠️  Pas assez de courses ({len(logits_by_race)}) pour apprendre T")
            self.temperature = self.config.temperature_init
            self.fitted = True
            return self
        
        # Optimiser T
        result = minimize_scalar(
            lambda t: self._nll_loss(t, logits_by_race, labels_by_race),
            bounds=self.config.temperature_bounds,
            method='bounded'
        )
        
        self.temperature = result.x
        self.fitted = True
        print(f"✅ Température optimale apprise: T = {self.temperature:.4f}")
        
        return self
    
    def transform(
        self, 
        df: pd.DataFrame,
        logits_col: str = 'logits_model',
        race_col: str = 'race_id'
    ) -> np.ndarray:
        """
        Transforme les logits en probabilités normalisées par course.
        
        Returns:
            np.ndarray: Probabilités (même ordre que df)
        """
        if not self.fitted:
            raise ValueError("TemperatureSoftmax non entraîné. Appeler fit() d'abord.")
        
        # Réinitialiser l'index pour éviter les problèmes
        df = df.reset_index(drop=True)
        probs = np.zeros(len(df))
        
        for race_id, group in df.groupby(race_col):
            idx = group.index.values
            logits = group[logits_col].values
            race_probs = self._softmax_with_temp(logits, self.temperature)
            probs[idx] = race_probs
        
        return probs
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        logits_col: str = 'logits_model',
        race_col: str = 'race_id',
        label_col: str = 'label_win'
    ) -> np.ndarray:
        """Fit puis transform."""
        self.fit(df, logits_col, race_col, label_col)
        return self.transform(df, logits_col, race_col)
    
    def save(self, path: str):
        """Sauvegarde le scaler."""
        with open(path, 'wb') as f:
            pickle.dump({
                'temperature': self.temperature,
                'config': self.config
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'TemperatureSoftmax':
        """Charge un scaler sauvegardé."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        scaler = cls(data['config'])
        scaler.temperature = data['temperature']
        scaler.fitted = True
        return scaler


# ============================================================================
# CALIBRATEURS (Platt + Isotonic)
# ============================================================================

class PlattCalibrator:
    """Calibration Platt (régression logistique sur les probas)."""
    
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.fitted = False
    
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> 'PlattCalibrator':
        """Fit sur les probabilités."""
        # Transformer en logits pour la régression
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        logits = logit(probs_clipped).reshape(-1, 1)
        self.model.fit(logits, labels)
        self.fitted = True
        return self
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Calibre les probabilités."""
        if not self.fitted:
            raise ValueError("PlattCalibrator non entraîné.")
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        logits = logit(probs_clipped).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load(cls, path: str) -> 'PlattCalibrator':
        calibrator = cls()
        with open(path, 'rb') as f:
            calibrator.model = pickle.load(f)
        calibrator.fitted = True
        return calibrator


class IsotonicCalibrator:
    """Calibration Isotonique (régression isotonique)."""
    
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
        self.fitted = False
    
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibrator':
        """Fit sur les probabilités."""
        self.model.fit(probs, labels)
        self.fitted = True
        return self
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Calibre les probabilités."""
        if not self.fitted:
            raise ValueError("IsotonicCalibrator non entraîné.")
        return self.model.predict(probs)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load(cls, path: str) -> 'IsotonicCalibrator':
        calibrator = cls()
        with open(path, 'rb') as f:
            calibrator.model = pickle.load(f)
        calibrator.fitted = True
        return calibrator


# ============================================================================
# BLEND MODÈLE-MARCHÉ
# ============================================================================

class MarketModelBlender:
    """Blend des probabilités modèle et marché dans l'espace logit."""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.alpha: float = 0.5  # Poids du modèle
        self.alpha_by_cluster: Dict[str, float] = {}
        self.fitted = False
    
    def _blend_logits(
        self, 
        p_model: np.ndarray, 
        p_market: np.ndarray, 
        alpha: float
    ) -> np.ndarray:
        """
        Blend dans l'espace logit.
        logit(p_blend) = α * logit(p_model) + (1-α) * logit(p_market)
        """
        eps = 1e-10
        p_model = np.clip(p_model, eps, 1 - eps)
        p_market = np.clip(p_market, eps, 1 - eps)
        
        logit_model = logit(p_model)
        logit_market = logit(p_market)
        
        # Shrinkage si divergence extrême
        divergence = np.abs(logit_model - logit_market)
        shrink_factor = np.where(divergence > 3, 0.5, 1.0)
        logit_model_adj = logit_market + shrink_factor * (logit_model - logit_market)
        
        logit_blend = alpha * logit_model_adj + (1 - alpha) * logit_market
        
        return expit(logit_blend)
    
    def _cv_score(
        self, 
        alpha: float, 
        p_model: np.ndarray, 
        p_market: np.ndarray, 
        labels: np.ndarray,
        dates: np.ndarray
    ) -> float:
        """Score CV pour un alpha donné (Brier score moyen)."""
        unique_dates = np.unique(dates)
        n_dates = len(unique_dates)
        
        if n_dates < self.config.alpha_cv_folds:
            # Pas assez de dates, utiliser tout le dataset
            p_blend = self._blend_logits(p_model, p_market, alpha)
            return CalibrationMetrics.brier_score(labels, p_blend)
        
        # Time-series CV
        fold_size = n_dates // self.config.alpha_cv_folds
        scores = []
        
        for i in range(self.config.alpha_cv_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.config.alpha_cv_folds - 1 else n_dates
            val_dates = unique_dates[val_start:val_end]
            
            val_mask = np.isin(dates, val_dates)
            if np.sum(val_mask) < 10:
                continue
            
            p_blend = self._blend_logits(
                p_model[val_mask], 
                p_market[val_mask], 
                alpha
            )
            score = CalibrationMetrics.brier_score(labels[val_mask], p_blend)
            scores.append(score)
        
        return np.mean(scores) if scores else 1.0
    
    def fit(
        self, 
        df: pd.DataFrame,
        p_model_col: str = 'p_model_norm',
        p_market_col: str = 'p_market',
        label_col: str = 'label_win',
        date_col: str = 'date',
        cluster_col: Optional[str] = None
    ) -> 'MarketModelBlender':
        """
        Apprend α optimal par CV.
        
        Args:
            df: DataFrame avec probabilités et labels
            cluster_col: Colonne pour calibration par cluster (ex: 'discipline')
        """
        # Vérifier présence des colonnes
        if p_market_col not in df.columns:
            print("⚠️  Pas de cotes marché, α fixé à 1.0 (modèle seul)")
            self.alpha = 1.0
            self.fitted = True
            return self
        
        p_model = df[p_model_col].values
        p_market = df[p_market_col].values
        labels = df[label_col].values
        dates = df[date_col].values if date_col in df.columns else np.arange(len(df))
        
        # Grid search pour α global
        best_alpha = 0.5
        best_score = float('inf')
        
        for alpha in self.config.alpha_grid:
            score = self._cv_score(alpha, p_model, p_market, labels, dates)
            if score < best_score:
                best_score = score
                best_alpha = alpha
        
        self.alpha = best_alpha
        print(f"✅ α optimal global: {self.alpha:.2f} (Brier CV: {best_score:.4f})")
        
        # Calibration par cluster si demandé
        if cluster_col and cluster_col in df.columns:
            for cluster, group in df.groupby(cluster_col):
                if len(group) < self.config.min_samples_cluster:
                    continue
                
                p_model_c = group[p_model_col].values
                p_market_c = group[p_market_col].values
                labels_c = group[label_col].values
                dates_c = group[date_col].values if date_col in group.columns else np.arange(len(group))
                
                best_alpha_c = self.alpha
                best_score_c = float('inf')
                
                for alpha in self.config.alpha_grid:
                    score = self._cv_score(alpha, p_model_c, p_market_c, labels_c, dates_c)
                    if score < best_score_c:
                        best_score_c = score
                        best_alpha_c = alpha
                
                self.alpha_by_cluster[cluster] = best_alpha_c
                print(f"  → Cluster '{cluster}': α = {best_alpha_c:.2f}")
        
        self.fitted = True
        return self
    
    def transform(
        self, 
        df: pd.DataFrame,
        p_model_col: str = 'p_model_norm',
        p_market_col: str = 'p_market',
        cluster_col: Optional[str] = None
    ) -> np.ndarray:
        """Applique le blend."""
        if not self.fitted:
            raise ValueError("MarketModelBlender non entraîné.")
        
        # Réinitialiser l'index
        df = df.reset_index(drop=True)
        p_model = df[p_model_col].values
        
        if p_market_col not in df.columns:
            return p_model
        
        p_market = df[p_market_col].values
        p_blend = np.zeros(len(df))
        
        if cluster_col and cluster_col in df.columns and self.alpha_by_cluster:
            for cluster, group in df.groupby(cluster_col):
                idx = group.index.values
                alpha = self.alpha_by_cluster.get(cluster, self.alpha)
                p_blend[idx] = self._blend_logits(
                    p_model[idx], p_market[idx], alpha
                )
        else:
            p_blend = self._blend_logits(p_model, p_market, self.alpha)
        
        return p_blend
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'alpha_by_cluster': self.alpha_by_cluster
            }, f)
    
    @classmethod
    def load(cls, path: str, config: CalibrationConfig = None) -> 'MarketModelBlender':
        if config is None:
            config = CalibrationConfig()
        blender = cls(config)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        blender.alpha = data['alpha']
        blender.alpha_by_cluster = data['alpha_by_cluster']
        blender.fitted = True
        return blender


# ============================================================================
# SIMULATION DE PROFIT
# ============================================================================

class ProfitSimulator:
    """Simulation de profit sur paris win/place."""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
    
    def simulate_flat_betting(
        self, 
        df: pd.DataFrame,
        p_col: str = 'p_calibrated',
        odds_col: str = 'odds_market',
        label_col: str = 'label_win',
        stake: float = None
    ) -> Dict[str, Any]:
        """
        Simule des paris flat sur les value bets.
        
        Returns:
            Dict avec métriques de profit
        """
        if stake is None:
            stake = self.config.profit_sim_stake
        
        df = df.copy()
        
        # Calculer EV
        df['ev'] = df[p_col] * df[odds_col] - 1
        
        # Filtrer value bets
        value_bets = df[df['ev'] > self.config.value_threshold]
        
        if len(value_bets) == 0:
            return {
                'n_bets': 0,
                'total_staked': 0,
                'total_return': 0,
                'profit': 0,
                'roi': 0,
                'win_rate': 0,
                'avg_odds': 0,
                'avg_ev': 0
            }
        
        n_bets = len(value_bets)
        total_staked = n_bets * stake
        
        # Calculer les gains
        wins = value_bets[label_col].values
        odds = value_bets[odds_col].values
        returns = wins * odds * stake
        total_return = np.sum(returns)
        
        profit = total_return - total_staked
        roi = (profit / total_staked) * 100 if total_staked > 0 else 0
        
        return {
            'n_bets': int(n_bets),
            'total_staked': float(total_staked),
            'total_return': float(total_return),
            'profit': float(profit),
            'roi': float(roi),
            'win_rate': float(np.mean(wins) * 100),
            'avg_odds': float(np.mean(odds)),
            'avg_ev': float(np.mean(value_bets['ev']) * 100)
        }
    
    def simulate_kelly_betting(
        self, 
        df: pd.DataFrame,
        p_col: str = 'p_calibrated',
        odds_col: str = 'odds_market',
        label_col: str = 'label_win',
        bankroll: float = 1000,
        kelly_fraction: float = 0.25,
        max_stake_pct: float = 0.05
    ) -> Dict[str, Any]:
        """Simule des paris Kelly fractionnel."""
        df = df.copy().sort_values('date' if 'date' in df.columns else df.index.name or df.index[0])
        
        current_bankroll = bankroll
        n_bets = 0
        total_staked = 0
        total_return = 0
        bankroll_history = [bankroll]
        
        for _, row in df.iterrows():
            p = row[p_col]
            odds = row[odds_col]
            label = row[label_col]
            
            # Kelly criterion
            q = 1 - p
            b = odds - 1
            kelly = (p * b - q) / b if b > 0 else 0
            
            if kelly <= 0:
                continue
            
            # Fractionnel + cap
            stake_pct = min(kelly * kelly_fraction, max_stake_pct)
            stake = current_bankroll * stake_pct
            
            n_bets += 1
            total_staked += stake
            
            if label == 1:
                win_amount = stake * odds
                total_return += win_amount
                current_bankroll += win_amount - stake
            else:
                current_bankroll -= stake
            
            bankroll_history.append(current_bankroll)
        
        profit = total_return - total_staked
        roi = (profit / total_staked) * 100 if total_staked > 0 else 0
        
        # Calcul drawdown
        peak = bankroll
        max_drawdown = 0
        for b in bankroll_history:
            if b > peak:
                peak = b
            dd = (peak - b) / peak
            if dd > max_drawdown:
                max_drawdown = dd
        
        return {
            'n_bets': int(n_bets),
            'total_staked': float(total_staked),
            'total_return': float(total_return),
            'profit': float(profit),
            'roi': float(roi),
            'final_bankroll': float(current_bankroll),
            'bankroll_growth': float((current_bankroll - bankroll) / bankroll * 100),
            'max_drawdown_pct': float(max_drawdown * 100)
        }


# ============================================================================
# PIPELINE PRINCIPALE
# ============================================================================

class CalibrationPipeline:
    """
    Pipeline complète de calibration et normalisation.
    
    Étapes:
    1. Walk-forward split par date
    2. Normalisation softmax avec T appris
    3. Calibration (Platt vs Isotonic)
    4. Blend modèle-marché avec α optimisé
       - Option: Correction gamma (biais favori/outsider)
       - Option: Alpha dynamique (temps/volume)
    5. Évaluation et génération d'artefacts
    """
    
    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.scaler: Optional[TemperatureSoftmax] = None
        self.calibrator_platt: Optional[PlattCalibrator] = None
        self.calibrator_isotonic: Optional[IsotonicCalibrator] = None
        self.blender: Optional[MarketModelBlender] = None
        self.dynamic_blender: Optional['DynamicMarketModelBlender'] = None  # Nouveau blender dynamique
        self.best_calibrator_type: str = 'isotonic'
        self.calibrators_by_cluster: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.artifacts_paths: Dict[str, str] = {}
    
    def _walk_forward_split(
        self, 
        df: pd.DataFrame, 
        date_col: str = 'date'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split walk-forward par date."""
        if date_col not in df.columns:
            # Fallback: split par index
            n = len(df)
            train_end = int(n * self.config.train_ratio)
            val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
            return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
        
        df = df.sort_values(date_col)
        dates = df[date_col].unique()
        n_dates = len(dates)
        
        train_end = int(n_dates * self.config.train_ratio)
        val_end = int(n_dates * (self.config.train_ratio + self.config.val_ratio))
        
        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]
        
        train_df = df[df[date_col].isin(train_dates)]
        val_df = df[df[date_col].isin(val_dates)]
        test_df = df[df[date_col].isin(test_dates)]
        
        print(f"📊 Split walk-forward:")
        print(f"   Train: {len(train_df)} samples ({len(train_dates)} dates)")
        print(f"   Val:   {len(val_df)} samples ({len(val_dates)} dates)")
        print(f"   Test:  {len(test_df)} samples ({len(test_dates)} dates)")
        
        return train_df, val_df, test_df
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare les données (conversion cotes → probas marché)."""
        df = df.copy()
        
        # Convertir cotes en probabilités si présentes
        if 'odds_market_preoff' in df.columns and 'p_market' not in df.columns:
            df['p_market'] = 1 / df['odds_market_preoff']
            # Normaliser par course
            df['p_market'] = df.groupby('race_id')['p_market'].transform(
                lambda x: x / x.sum()
            )
        
        return df
    
    def _select_best_calibrator(
        self, 
        val_df: pd.DataFrame,
        p_col: str = 'p_model_norm',
        label_col: str = 'label_win'
    ) -> str:
        """Sélectionne le meilleur calibrateur sur validation."""
        probs = val_df[p_col].values
        labels = val_df[label_col].values
        
        # Évaluer Platt
        p_platt = self.calibrator_platt.transform(probs)
        metrics_platt = {
            'brier': CalibrationMetrics.brier_score(labels, p_platt),
            'nll': CalibrationMetrics.log_loss(labels, p_platt),
            'ece': CalibrationMetrics.expected_calibration_error(labels, p_platt, self.config.n_bins_ece)[0]
        }
        
        # Évaluer Isotonic
        p_isotonic = self.calibrator_isotonic.transform(probs)
        metrics_isotonic = {
            'brier': CalibrationMetrics.brier_score(labels, p_isotonic),
            'nll': CalibrationMetrics.log_loss(labels, p_isotonic),
            'ece': CalibrationMetrics.expected_calibration_error(labels, p_isotonic, self.config.n_bins_ece)[0]
        }
        
        print(f"\n📈 Comparaison calibrateurs (validation):")
        print(f"   Platt    - Brier: {metrics_platt['brier']:.4f}, ECE: {metrics_platt['ece']:.4f}, NLL: {metrics_platt['nll']:.4f}")
        print(f"   Isotonic - Brier: {metrics_isotonic['brier']:.4f}, ECE: {metrics_isotonic['ece']:.4f}, NLL: {metrics_isotonic['nll']:.4f}")
        
        # Sélectionner par Brier score
        if metrics_isotonic['brier'] < metrics_platt['brier']:
            print("   ✅ Isotonic sélectionné")
            return 'isotonic'
        else:
            print("   ✅ Platt sélectionné")
            return 'platt'
    
    def _generate_plots(
        self, 
        test_df: pd.DataFrame,
        p_col: str = 'p_final',
        label_col: str = 'label_win',
        output_dir: str = None
    ) -> List[str]:
        """Génère les graphiques de calibration."""
        if output_dir is None:
            output_dir = self.config.artifacts_dir
        
        os.makedirs(output_dir, exist_ok=True)
        plot_paths = []
        
        probs = test_df[p_col].values
        labels = test_df[label_col].values
        
        # 1. Reliability diagram
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Courbe de fiabilité
        bin_centers, bin_accuracies, bin_counts = CalibrationMetrics.reliability_curve(
            labels, probs, self.config.n_bins_ece
        )
        
        ax = axes[0]
        ax.plot([0, 1], [0, 1], 'k--', label='Parfaitement calibré')
        valid_bins = ~np.isnan(bin_accuracies)
        ax.bar(bin_centers[valid_bins], bin_accuracies[valid_bins], 
               width=0.08, alpha=0.7, label='Observé')
        ax.set_xlabel('Probabilité prédite')
        ax.set_ylabel('Fréquence observée')
        ax.set_title('Courbe de Fiabilité (Reliability Diagram)')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Histogramme des probabilités
        ax = axes[1]
        ax.hist(probs, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Probabilité prédite')
        ax.set_ylabel('Fréquence')
        ax.set_title('Distribution des Probabilités')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(output_dir, 'calibration_reliability.png')
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths.append(path)
        
        # 2. ECE par bin
        fig, ax = plt.subplots(figsize=(10, 5))
        ece, bins_detail = CalibrationMetrics.expected_calibration_error(
            labels, probs, self.config.n_bins_ece
        )
        
        gaps = [b['gap'] for b in bins_detail]
        ranges = [b['range'] for b in bins_detail]
        colors = ['green' if g >= 0 else 'red' for g in gaps]
        
        ax.bar(range(len(gaps)), gaps, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(gaps)))
        ax.set_xticklabels(ranges, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Bin de probabilité')
        ax.set_ylabel('Gap (Observé - Prédit)')
        ax.set_title(f'Analyse ECE par Bin (ECE global = {ece:.4f})')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = os.path.join(output_dir, 'calibration_ece_bins.png')
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths.append(path)
        
        return plot_paths
    
    def fit(
        self, 
        df: pd.DataFrame,
        logits_col: str = 'logits_model',
        odds_col: str = 'odds_market_preoff',
        race_col: str = 'race_id',
        label_win_col: str = 'label_win',
        label_place_col: str = 'label_place',
        date_col: str = 'date',
        cluster_col: Optional[str] = None  # Ex: 'discipline', 'hippodrome'
    ) -> 'CalibrationPipeline':
        """
        Entraîne la pipeline complète.
        
        Args:
            df: DataFrame avec les données historiques
            cluster_col: Colonne pour calibration par cluster
        """
        print("=" * 60)
        print("🚀 PIPELINE DE CALIBRATION - ENTRAÎNEMENT")
        print("=" * 60)
        
        # Préparer les données
        df = self._prepare_data(df)
        
        # Split walk-forward
        train_df, val_df, test_df = self._walk_forward_split(df, date_col)
        
        if len(train_df) < self.config.min_train_samples:
            raise ValueError(f"Pas assez de données d'entraînement: {len(train_df)} < {self.config.min_train_samples}")
        
        # 1. Normalisation softmax avec température
        print("\n📐 Étape 1: Normalisation Softmax")
        self.scaler = TemperatureSoftmax(self.config)
        train_df['p_model_norm'] = self.scaler.fit_transform(train_df, logits_col, race_col, label_win_col)
        val_df['p_model_norm'] = self.scaler.transform(val_df, logits_col, race_col)
        test_df['p_model_norm'] = self.scaler.transform(test_df, logits_col, race_col)
        
        # 2. Calibration (train sur train, sélection sur val)
        print("\n🎯 Étape 2: Calibration")
        
        # Fit Platt
        self.calibrator_platt = PlattCalibrator()
        self.calibrator_platt.fit(
            train_df['p_model_norm'].values, 
            train_df[label_win_col].values
        )
        
        # Fit Isotonic
        self.calibrator_isotonic = IsotonicCalibrator()
        self.calibrator_isotonic.fit(
            train_df['p_model_norm'].values, 
            train_df[label_win_col].values
        )
        
        # Sélectionner le meilleur
        self.best_calibrator_type = self._select_best_calibrator(val_df, 'p_model_norm', label_win_col)
        
        # Appliquer calibration
        calibrator = self.calibrator_isotonic if self.best_calibrator_type == 'isotonic' else self.calibrator_platt
        train_df['p_calibrated'] = calibrator.transform(train_df['p_model_norm'].values)
        val_df['p_calibrated'] = calibrator.transform(val_df['p_model_norm'].values)
        test_df['p_calibrated'] = calibrator.transform(test_df['p_model_norm'].values)
        
        # 3. Blend modèle-marché
        print("\n🔀 Étape 3: Blend Modèle-Marché")
        
        # Vérifier si on utilise le blend dynamique avec correction gamma
        use_dynamic = (
            DEBIASER_AVAILABLE and 
            (self.config.use_gamma_correction or self.config.use_dynamic_alpha) and
            'p_market' in train_df.columns
        )
        
        if use_dynamic:
            print("   📊 Mode: Blend DYNAMIQUE avec correction biais")
            
            # Créer la config pour le debiaser
            debias_config = MarketDebiasConfig(
                gamma_init=self.config.gamma_init,
                gamma_bounds=self.config.gamma_bounds,
                gamma_by_cluster=self.config.gamma_by_cluster,
                alpha_bounds=self.config.alpha_bounds,
                alpha_base=self.config.alpha_base,
                alpha_pool_coef=self.config.alpha_pool_coef,
                alpha_time_coef=self.config.alpha_time_coef,
                default_pool_size=self.config.default_pool_size,
                default_minutes_to_off=self.config.default_minutes_to_off,
                n_bins_ece=self.config.n_bins_ece,
                min_samples_cluster=self.config.min_samples_cluster
            )
            
            self.dynamic_blender = DynamicMarketModelBlender(debias_config)
            combined_train_val = pd.concat([train_df, val_df])
            
            # Fit le blender dynamique
            self.dynamic_blender.fit(
                combined_train_val,
                p_model_col='p_calibrated',
                odds_col=odds_col,
                race_col=race_col,
                label_col=label_win_col,
                date_col=date_col,
                pool_col='pool_size',
                time_col='minutes_to_off',
                cluster_col=cluster_col
            )
            
            # Transform sur test
            test_df = self.dynamic_blender.transform(
                test_df,
                p_model_col='p_calibrated',
                odds_col=odds_col,
                race_col=race_col,
                pool_col='pool_size',
                time_col='minutes_to_off',
                cluster_col=cluster_col
            )
            test_df['p_final'] = test_df['p_win_blend']
            
            # Évaluation spécifique du debiaser
            debias_results = self.dynamic_blender.evaluate(
                test_df,
                p_model_col='p_calibrated',
                odds_col=odds_col,
                label_col=label_win_col
            )
            self.metrics['debias'] = debias_results
            self.dynamic_blender.print_evaluation_report(debias_results)
            
        elif 'p_market' in train_df.columns:
            print("   📊 Mode: Blend CLASSIQUE (α statique)")
            self.blender = MarketModelBlender(self.config)
            # Utiliser les probas calibrées pour le blend
            combined_train_val = pd.concat([train_df, val_df])
            self.blender.fit(
                combined_train_val,
                p_model_col='p_calibrated',
                p_market_col='p_market',
                label_col=label_win_col,
                date_col=date_col,
                cluster_col=cluster_col
            )
            
            test_df['p_final'] = self.blender.transform(
                test_df, 
                p_model_col='p_calibrated',
                p_market_col='p_market',
                cluster_col=cluster_col
            )
        else:
            print("   ⚠️  Pas de cotes marché, utilisation des probas calibrées seules")
            test_df['p_final'] = test_df['p_calibrated']
        
        # Re-normaliser par course après blend
        test_df['p_final'] = test_df.groupby(race_col)['p_final'].transform(lambda x: x / x.sum())
        
        # 4. Évaluation sur test
        print("\n📊 Étape 4: Évaluation sur Test")
        self._evaluate(test_df, 'p_final', label_win_col, odds_col)
        
        # 5. Générer plots
        print("\n📈 Génération des graphiques...")
        self.artifacts_paths['plots'] = self._generate_plots(test_df, 'p_final', label_win_col)
        
        # 6. Sauvegarder artefacts
        self._save_artifacts()
        
        return self
    
    def _evaluate(
        self, 
        test_df: pd.DataFrame,
        p_col: str,
        label_col: str,
        odds_col: str
    ):
        """Évalue la pipeline sur le test set."""
        probs = test_df[p_col].values
        labels = test_df[label_col].values
        
        # Métriques de calibration
        brier = CalibrationMetrics.brier_score(labels, probs)
        nll = CalibrationMetrics.log_loss(labels, probs)
        ece, bins_detail = CalibrationMetrics.expected_calibration_error(
            labels, probs, self.config.n_bins_ece
        )
        
        self.metrics['calibration'] = {
            'brier_score': brier,
            'log_loss': nll,
            'ece': ece,
            'ece_bins': bins_detail
        }
        
        print(f"   Brier Score: {brier:.4f}")
        print(f"   Log Loss:    {nll:.4f}")
        print(f"   ECE:         {ece:.4f}")
        
        # Simulation de profit
        if odds_col in test_df.columns:
            simulator = ProfitSimulator(self.config)
            
            # Flat betting
            flat_results = simulator.simulate_flat_betting(
                test_df, p_col, odds_col, label_col
            )
            self.metrics['profit_flat'] = flat_results
            print(f"\n   📈 Simulation Flat Betting:")
            print(f"      Paris: {flat_results['n_bets']}, ROI: {flat_results['roi']:.1f}%")
            
            # Kelly betting
            kelly_results = simulator.simulate_kelly_betting(
                test_df, p_col, odds_col, label_col
            )
            self.metrics['profit_kelly'] = kelly_results
            print(f"   📈 Simulation Kelly:")
            print(f"      Paris: {kelly_results['n_bets']}, ROI: {kelly_results['roi']:.1f}%, Drawdown Max: {kelly_results['max_drawdown_pct']:.1f}%")
        
        # Stats globales
        self.metrics['stats'] = {
            'n_samples_test': len(test_df),
            'n_races_test': test_df['race_id'].nunique() if 'race_id' in test_df.columns else 0,
            'win_rate_observed': float(np.mean(labels) * 100),
            'avg_prob_predicted': float(np.mean(probs) * 100),
            'temperature': self.scaler.temperature if self.scaler else None,
            'alpha_blend': self.blender.alpha if self.blender else None,
            'calibrator_type': self.best_calibrator_type,
            'use_dynamic_blend': self.dynamic_blender is not None,
            'gamma': self.dynamic_blender.gamma_corrector.gamma if self.dynamic_blender else None,
            'gamma_by_cluster': self.dynamic_blender.gamma_corrector.gamma_by_cluster if self.dynamic_blender else {}
        }
    
    def _save_artifacts(self):
        """Sauvegarde tous les artefacts."""
        output_dir = self.config.artifacts_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Scaler (température)
        if self.scaler:
            path = os.path.join(output_dir, f'scaler_temperature_{timestamp}.pkl')
            self.scaler.save(path)
            self.artifacts_paths['scaler'] = path
        
        # Calibrateurs
        if self.calibrator_platt:
            path = os.path.join(output_dir, f'calibrator_platt_{timestamp}.pkl')
            self.calibrator_platt.save(path)
            self.artifacts_paths['calibrator_platt'] = path
        
        if self.calibrator_isotonic:
            path = os.path.join(output_dir, f'calibrator_isotonic_{timestamp}.pkl')
            self.calibrator_isotonic.save(path)
            self.artifacts_paths['calibrator_isotonic'] = path
        
        # Blender
        if self.blender:
            path = os.path.join(output_dir, f'blender_{timestamp}.pkl')
            self.blender.save(path)
            self.artifacts_paths['blender'] = path
        
        # Dynamic Blender (avec correction gamma)
        if self.dynamic_blender:
            dynamic_dir = os.path.join(output_dir, f'dynamic_blender_{timestamp}')
            self.dynamic_blender.save(dynamic_dir)
            self.artifacts_paths['dynamic_blender'] = dynamic_dir
        
        # Rapport JSON
        report = {
            'timestamp': timestamp,
            'config': asdict(self.config),
            'metrics': self.metrics,
            'artifacts': self.artifacts_paths,
            'best_calibrator': self.best_calibrator_type,
            'temperature': self.scaler.temperature if self.scaler else None,
            'alpha': self.blender.alpha if self.blender else None,
            'alpha_by_cluster': self.blender.alpha_by_cluster if self.blender else {},
            'use_dynamic_blend': self.dynamic_blender is not None,
            'gamma': self.dynamic_blender.gamma_corrector.gamma if self.dynamic_blender else None,
            'gamma_by_cluster': self.dynamic_blender.gamma_corrector.gamma_by_cluster if self.dynamic_blender else {},
            'alpha_coefficients': {
                'a0': self.dynamic_blender.alpha_calculator.a0 if self.dynamic_blender else None,
                'a1': self.dynamic_blender.alpha_calculator.a1 if self.dynamic_blender else None,
                'a2': self.dynamic_blender.alpha_calculator.a2 if self.dynamic_blender else None
            } if self.dynamic_blender else None
        }
        
        report_path = os.path.join(output_dir, f'calibration_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.artifacts_paths['report'] = report_path
        
        print(f"\n💾 Artefacts sauvegardés dans: {output_dir}")
        for name, path in self.artifacts_paths.items():
            if isinstance(path, list):
                for p in path:
                    print(f"   - {p}")
            else:
                print(f"   - {path}")
        
        # === NOUVEAU: Exporter vers config/pro_betting.yaml ===
        self._export_to_config()
    
    def _export_to_config(self):
        """
        Exporte les paramètres optimaux vers config/pro_betting.yaml.
        Garantit la cohérence entre artefacts et configuration YAML.
        """
        if not CONFIG_AVAILABLE:
            print("⚠️  config.loader non disponible - export vers YAML ignoré")
            return
        
        try:
            # Récupérer les paramètres optimaux
            temperature = self.scaler.temperature if self.scaler else 1.0
            
            # Gérer les deux types de blenders
            if self.dynamic_blender:
                alpha_global = self.dynamic_blender.alpha_calculator.a0
                alpha_by_discipline = self.dynamic_blender.gamma_corrector.gamma_by_cluster
                gamma_global = self.dynamic_blender.gamma_corrector.gamma
                gamma_by_discipline = self.dynamic_blender.gamma_corrector.gamma_by_cluster
            else:
                alpha_global = self.blender.alpha if self.blender else 0.5
                alpha_by_discipline = self.blender.alpha_by_cluster if self.blender else {}
                gamma_global = 1.0
                gamma_by_discipline = {}
            
            calibrator_type = self.best_calibrator_type
            
            # Récupérer les métriques
            metrics = {}
            if 'calibration' in self.metrics:
                metrics['brier'] = self.metrics['calibration'].get('brier_score', 0)
                metrics['log_loss'] = self.metrics['calibration'].get('log_loss', 0)
                metrics['ece'] = self.metrics['calibration'].get('ece', 0)
            
            # Appeler la fonction d'update centralisée
            update_config_from_calibration(
                temperature=temperature,
                alpha_global=alpha_global,
                alpha_by_discipline=alpha_by_discipline,
                calibrator_type=calibrator_type,
                metrics=metrics
            )
            
            print(f"\n📝 Configuration pro_betting.yaml mise à jour:")
            print(f"   - temperature: {temperature:.4f}")
            print(f"   - blend_alpha_global: {alpha_global:.2f}")
            print(f"   - blend_alpha_by_discipline: {alpha_by_discipline}")
            print(f"   - calibrator_type: {calibrator_type}")
            
            # Afficher params gamma si dynamic blender
            if self.dynamic_blender:
                print(f"   - gamma_global: {gamma_global:.2f}")
                print(f"   - gamma_by_discipline: {gamma_by_discipline}")
                print(f"   - alpha_coefficients: a0={self.dynamic_blender.alpha_calculator.a0:.3f}, "
                      f"a1={self.dynamic_blender.alpha_calculator.a1:.3f}, "
                      f"a2={self.dynamic_blender.alpha_calculator.a2:.3f}")
            
        except Exception as e:
            print(f"⚠️  Erreur lors de l'export vers config: {e}")
    
    def transform(
        self, 
        df: pd.DataFrame,
        logits_col: str = 'logits_model',
        race_col: str = 'race_id',
        odds_col: str = 'odds_market_preoff',
        cluster_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Applique la pipeline sur de nouvelles données.
        
        Returns:
            DataFrame avec colonnes ajoutées: p_model_norm, p_calibrated, p_final
        """
        df = df.copy()
        df = self._prepare_data(df)
        
        # Softmax normalisé
        df['p_model_norm'] = self.scaler.transform(df, logits_col, race_col)
        
        # Calibration
        calibrator = self.calibrator_isotonic if self.best_calibrator_type == 'isotonic' else self.calibrator_platt
        df['p_calibrated'] = calibrator.transform(df['p_model_norm'].values)
        
        # Blend - avec dynamic blender si disponible
        if self.dynamic_blender and odds_col in df.columns:
            df = self.dynamic_blender.transform(
                df,
                p_model_col='p_calibrated',
                odds_col=odds_col,
                race_col=race_col,
                pool_col='pool_size',
                time_col='minutes_to_off',
                cluster_col=cluster_col
            )
            df['p_final'] = df['p_win_blend']
        elif self.blender and 'p_market' in df.columns:
            df['p_final'] = self.blender.transform(
                df, 
                p_model_col='p_calibrated',
                p_market_col='p_market',
                cluster_col=cluster_col
            )
        else:
            df['p_final'] = df['p_calibrated']
        
        # Re-normaliser par course
        df['p_final'] = df.groupby(race_col)['p_final'].transform(lambda x: x / x.sum())
        
        return df
    
    @classmethod
    def load(cls, artifacts_dir: str, report_name: str = None) -> 'CalibrationPipeline':
        """
        Charge une pipeline sauvegardée.
        
        Args:
            artifacts_dir: Dossier des artefacts
            report_name: Nom du rapport (dernier si None)
        """
        # Trouver le rapport
        if report_name is None:
            reports = sorted([f for f in os.listdir(artifacts_dir) if f.startswith('calibration_report')])
            if not reports:
                raise ValueError(f"Aucun rapport trouvé dans {artifacts_dir}")
            report_name = reports[-1]
        
        report_path = os.path.join(artifacts_dir, report_name)
        with open(report_path) as f:
            report = json.load(f)
        
        # Reconstruire la config
        config = CalibrationConfig(**{k: v for k, v in report['config'].items() 
                                      if k in CalibrationConfig.__dataclass_fields__})
        
        pipeline = cls(config)
        
        # Charger les artefacts
        artifacts = report['artifacts']
        
        if 'scaler' in artifacts:
            pipeline.scaler = TemperatureSoftmax.load(artifacts['scaler'])
        
        if 'calibrator_platt' in artifacts:
            pipeline.calibrator_platt = PlattCalibrator.load(artifacts['calibrator_platt'])
        
        if 'calibrator_isotonic' in artifacts:
            pipeline.calibrator_isotonic = IsotonicCalibrator.load(artifacts['calibrator_isotonic'])
        
        if 'blender' in artifacts:
            pipeline.blender = MarketModelBlender.load(artifacts['blender'], config)
        
        # Charger dynamic blender si disponible
        if 'dynamic_blender' in artifacts and DEBIASER_AVAILABLE:
            try:
                pipeline.dynamic_blender = DynamicMarketModelBlender.load(artifacts['dynamic_blender'])
            except Exception as e:
                print(f"⚠️ Impossible de charger dynamic_blender: {e}")
        
        pipeline.best_calibrator_type = report['best_calibrator']
        pipeline.metrics = report['metrics']
        pipeline.artifacts_paths = artifacts
        
        return pipeline


# ============================================================================
# GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES POUR TEST
# ============================================================================

def generate_synthetic_data(
    n_races: int = 500,
    horses_per_race: int = 12,
    seed: int = 42
) -> pd.DataFrame:
    """
    Génère des données synthétiques pour tester la pipeline.
    
    Simule:
    - Logits de modèle (avec bruit)
    - Cotes de marché (corrélées aux vraies probas)
    - Résultats (victoires, places)
    """
    np.random.seed(seed)
    
    records = []
    
    for race_id in range(n_races):
        date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=race_id // 5)
        discipline = np.random.choice(['plat', 'trot', 'obstacle'], p=[0.5, 0.35, 0.15])
        hippodrome = np.random.choice(['PAR', 'VIN', 'DEA', 'CHA', 'AUT'])
        
        n_horses = np.random.randint(8, horses_per_race + 1)
        
        # Vraies "forces" des chevaux (latentes)
        true_strength = np.random.randn(n_horses)
        true_probs = softmax(true_strength)
        
        # Logits du modèle (avec bruit et biais)
        model_logits = true_strength + np.random.randn(n_horses) * 0.5
        
        # Cotes du marché (inversées des vraies probas + bruit + marge bookmaker)
        market_probs_raw = true_probs + np.random.randn(n_horses) * 0.03
        market_probs_raw = np.clip(market_probs_raw, 0.01, 0.99)
        market_probs_raw = market_probs_raw / market_probs_raw.sum()
        
        # Ajouter marge bookmaker (15%)
        margin = 1.15
        market_odds = margin / market_probs_raw
        
        # Simuler le résultat
        winner_idx = np.random.choice(n_horses, p=true_probs)
        
        # Simuler les places (top 3)
        remaining_probs = true_probs.copy()
        remaining_probs[winner_idx] = 0
        remaining_probs = remaining_probs / remaining_probs.sum()
        place_2_idx = np.random.choice(n_horses, p=remaining_probs)
        remaining_probs[place_2_idx] = 0
        remaining_probs = remaining_probs / remaining_probs.sum()
        place_3_idx = np.random.choice(n_horses, p=remaining_probs)
        
        for horse_idx in range(n_horses):
            is_win = horse_idx == winner_idx
            is_place = horse_idx in [winner_idx, place_2_idx, place_3_idx]
            
            records.append({
                'race_id': f'R{race_id:05d}',
                'horse_id': f'H{race_id:05d}_{horse_idx:02d}',
                'date': date,
                'discipline': discipline,
                'hippodrome': hippodrome,
                'logits_model': model_logits[horse_idx],
                'odds_market_preoff': market_odds[horse_idx],
                'label_win': int(is_win),
                'label_place': int(is_place),
                'true_prob': true_probs[horse_idx]  # Pour vérification
            })
    
    df = pd.DataFrame(records)
    print(f"📊 Données synthétiques générées: {len(df)} lignes, {n_races} courses")
    return df


# ============================================================================
# MAIN - TESTS
# ============================================================================

def main():
    """Test de la pipeline avec données synthétiques."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline de Calibration')
    parser.add_argument('--test', action='store_true', help='Lancer les tests')
    parser.add_argument('--data', type=str, help='Chemin vers le dataset')
    parser.add_argument('--output', type=str, default='calibration_artifacts', help='Dossier de sortie')
    args = parser.parse_args()
    
    if args.test or args.data is None:
        print("=" * 60)
        print("🧪 TEST DE LA PIPELINE DE CALIBRATION")
        print("=" * 60)
        
        # Générer données synthétiques
        df = generate_synthetic_data(n_races=500, horses_per_race=12)
        
        # Configuration
        config = CalibrationConfig(
            artifacts_dir=args.output,
            temperature_init=1.5,
            min_samples_cluster=100
        )
        
        # Entraîner la pipeline
        pipeline = CalibrationPipeline(config)
        pipeline.fit(
            df,
            logits_col='logits_model',
            odds_col='odds_market_preoff',
            race_col='race_id',
            label_win_col='label_win',
            date_col='date',
            cluster_col='discipline'  # Calibration par discipline
        )
        
        # Afficher le rapport final
        print("\n" + "=" * 60)
        print("📋 RAPPORT FINAL")
        print("=" * 60)
        
        report = {
            'metrics': pipeline.metrics,
            'artifacts': pipeline.artifacts_paths,
            'parameters': {
                'temperature': pipeline.scaler.temperature if pipeline.scaler else None,
                'alpha_global': pipeline.blender.alpha if pipeline.blender else None,
                'alpha_by_cluster': pipeline.blender.alpha_by_cluster if pipeline.blender else {},
                'calibrator': pipeline.best_calibrator_type
            }
        }
        
        print(json.dumps(report, indent=2, default=str))
        
        # Test de chargement
        print("\n🔄 Test de chargement de la pipeline...")
        loaded_pipeline = CalibrationPipeline.load(args.output)
        
        # Test de transformation sur nouvelles données
        new_df = generate_synthetic_data(n_races=50, horses_per_race=10, seed=999)
        result_df = loaded_pipeline.transform(
            new_df,
            logits_col='logits_model',
            race_col='race_id',
            cluster_col='discipline'
        )
        
        print(f"\n✅ Transformation sur nouvelles données: {len(result_df)} lignes")
        print(f"   Colonnes ajoutées: p_model_norm, p_calibrated, p_final")
        
        # Vérification normalisation
        sums = result_df.groupby('race_id')['p_final'].sum()
        print(f"   Somme p_final par course: min={sums.min():.4f}, max={sums.max():.4f}, mean={sums.mean():.4f}")
        
        print("\n✅ TOUS LES TESTS PASSÉS")
    
    else:
        # Charger vraies données
        print(f"📂 Chargement des données depuis: {args.data}")
        df = pd.read_csv(args.data)
        
        config = CalibrationConfig(artifacts_dir=args.output)
        pipeline = CalibrationPipeline(config)
        pipeline.fit(df)
        
        print(f"\n✅ Pipeline entraînée. Artefacts dans: {args.output}")


if __name__ == '__main__':
    main()
