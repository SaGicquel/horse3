#!/usr/bin/env python3
"""
Module de Correction du Biais Favori/Outsider et Blend Dynamique
================================================================

Ce module implémente:
1. Correction "puissance" des probabilités marché: p_mkt_i ∝ q_i^γ
2. Blend dynamique avec α fonction du temps et du volume
3. Métriques de contrôle (Brier, ECE, logloss, lift EV)

Auteur: Horse Racing AI System
Date: 2024-12
"""

import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar, minimize
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class MarketDebiasConfig:
    """Configuration pour la correction de biais marché et blend dynamique."""

    # Correction gamma (biais favori/outsider)
    gamma_init: float = 1.0  # gamma=1 = pas de correction
    gamma_bounds: Tuple[float, float] = (0.5, 2.0)  # Bornes pour γ
    gamma_grid: List[float] = field(
        default_factory=lambda: [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    )
    gamma_cv_folds: int = 5
    gamma_by_cluster: bool = True  # Apprendre γ par discipline/hippodrome

    # Alpha dynamique
    alpha_bounds: Tuple[float, float] = (0.3, 0.9)  # Bornes pour α
    alpha_base: float = 0.5  # a0 - baseline
    alpha_pool_coef: float = 0.05  # a1 - coefficient log(pool)
    alpha_time_coef: float = 0.01  # a2 - coefficient minutes_to_off
    alpha_ema_span: int = 20  # Lissage EMA pour α

    # Pool par défaut (proxy si non disponible)
    default_pool_size: float = 50000.0  # Pool moyen typique
    default_minutes_to_off: float = 30.0  # Minutes par défaut avant départ

    # Correction insider (optionnel)
    enable_insider_correction: bool = False
    insider_threshold: float = 0.15  # Seuil d'inefficience pour correction

    # Évaluation
    n_bins_ece: int = 10
    min_samples_cluster: int = 200

    # Shrinkage pour blend
    shrinkage_threshold: float = 3.0  # Seuil logit pour shrinkage


# ============================================================================
# MÉTRIQUES DE CALIBRATION
# ============================================================================


class DebiasMetrics:
    """Métriques pour évaluer la correction de biais."""

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
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """Expected Calibration Error."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                bin_count = np.sum(mask)
                ece += np.abs(bin_acc - bin_conf) * (bin_count / len(y_prob))

        return float(ece)

    @staticmethod
    def expected_value_lift(
        y_true: np.ndarray,
        p_blend: np.ndarray,
        p_market: np.ndarray,
        odds: np.ndarray,
        value_threshold: float = 0.05,
    ) -> Dict[str, float]:
        """
        Calcule le lift EV du blend vs marché seul.

        Returns:
            Dict avec ev_blend, ev_market, lift_pct
        """
        eps = 1e-10

        # Fair odds
        fair_odds_blend = 1.0 / np.clip(p_blend, eps, 1.0)
        fair_odds_market = 1.0 / np.clip(p_market, eps, 1.0)

        # Value = (p_est * odds - 1)
        value_blend = p_blend * odds - 1
        value_market = p_market * odds - 1

        # Sélection des paris avec value > threshold
        mask_blend = value_blend > value_threshold
        mask_market = value_market > value_threshold

        # EV réalisé
        ev_blend = 0.0
        if np.sum(mask_blend) > 0:
            ev_blend = np.mean((y_true[mask_blend] * odds[mask_blend] - 1))

        ev_market = 0.0
        if np.sum(mask_market) > 0:
            ev_market = np.mean((y_true[mask_market] * odds[mask_market] - 1))

        lift = (ev_blend - ev_market) * 100 if ev_market != 0 else 0.0

        return {
            "ev_blend": float(ev_blend),
            "ev_market": float(ev_market),
            "lift_pct": float(lift),
            "n_bets_blend": int(np.sum(mask_blend)),
            "n_bets_market": int(np.sum(mask_market)),
        }


# ============================================================================
# CORRECTION GAMMA (BIAIS FAVORI/OUTSIDER)
# ============================================================================


class MarketGammaCorrector:
    """
    Correction du biais favori/outsider par transformation en puissance.

    Les marchés de paris ont tendance à:
    - Surestimer les favoris (trop de volume)
    - Sous-estimer les outsiders

    Correction: p_mkt_corr_i ∝ q_i^γ
    - γ < 1: favorise les outsiders (corrige la surestimation des favoris)
    - γ > 1: favorise les favoris
    - γ = 1: pas de correction
    """

    def __init__(self, config: MarketDebiasConfig):
        self.config = config
        self.gamma: float = config.gamma_init
        self.gamma_by_cluster: Dict[str, float] = {}
        self.fitted: bool = False

    def _odds_to_probs(self, odds: np.ndarray) -> np.ndarray:
        """Convertit les cotes en probabilités implicites normalisées."""
        probs = 1.0 / np.clip(odds, 1.01, 1000.0)
        return probs / np.sum(probs)

    def _apply_gamma_correction(self, probs: np.ndarray, gamma: float) -> np.ndarray:
        """
        Applique la correction en puissance.
        p_corr_i ∝ p_i^γ, puis renormalise.
        """
        eps = 1e-10
        probs = np.clip(probs, eps, 1.0)

        # Transformation en puissance
        probs_gamma = np.power(probs, gamma)

        # Renormalisation
        total = np.sum(probs_gamma)
        if total > 0:
            return probs_gamma / total
        else:
            return probs

    def _cv_score_gamma(
        self,
        gamma: float,
        probs_by_race: List[np.ndarray],
        labels_by_race: List[np.ndarray],
        dates_by_race: List[Any],
    ) -> float:
        """Score CV pour un gamma donné (Brier score moyen)."""
        unique_dates = np.unique(dates_by_race)
        n_dates = len(unique_dates)

        if n_dates < self.config.gamma_cv_folds:
            # Pas assez de dates, score sur tout
            all_probs = []
            all_labels = []
            for probs, labels in zip(probs_by_race, labels_by_race):
                p_corr = self._apply_gamma_correction(probs, gamma)
                all_probs.extend(p_corr)
                all_labels.extend(labels)
            return DebiasMetrics.brier_score(np.array(all_labels), np.array(all_probs))

        # Time-series CV
        fold_size = n_dates // self.config.gamma_cv_folds
        scores = []

        for i in range(self.config.gamma_cv_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.config.gamma_cv_folds - 1 else n_dates
            val_dates = set(unique_dates[val_start:val_end])

            all_probs = []
            all_labels = []

            for probs, labels, date in zip(probs_by_race, labels_by_race, dates_by_race):
                if date in val_dates:
                    p_corr = self._apply_gamma_correction(probs, gamma)
                    all_probs.extend(p_corr)
                    all_labels.extend(labels)

            if len(all_probs) >= 10:
                score = DebiasMetrics.brier_score(np.array(all_labels), np.array(all_probs))
                scores.append(score)

        return np.mean(scores) if scores else 1.0

    def fit(
        self,
        df: pd.DataFrame,
        odds_col: str = "odds_market_preoff",
        race_col: str = "race_id",
        label_col: str = "label_win",
        date_col: str = "date",
        cluster_col: Optional[str] = None,
    ) -> "MarketGammaCorrector":
        """
        Apprend γ optimal par CV.

        Args:
            df: DataFrame avec cotes, race_id, labels
            cluster_col: Colonne pour γ par cluster (ex: 'discipline', 'hippodrome')
        """
        print("\n🎯 Apprentissage γ (correction biais favori/outsider)")

        # Grouper par course
        probs_by_race = []
        labels_by_race = []
        dates_by_race = []

        for race_id, group in df.groupby(race_col):
            if odds_col not in group.columns:
                continue
            if group[label_col].sum() != 1:  # Course avec un seul gagnant
                continue

            odds = group[odds_col].values
            if np.any(np.isnan(odds)) or np.any(odds <= 1):
                continue

            probs = self._odds_to_probs(odds)
            labels = group[label_col].values
            date = group[date_col].iloc[0] if date_col in group.columns else race_id

            probs_by_race.append(probs)
            labels_by_race.append(labels)
            dates_by_race.append(date)

        if len(probs_by_race) < 50:
            print(f"   ⚠️ Pas assez de courses ({len(probs_by_race)}) pour apprendre γ")
            self.gamma = self.config.gamma_init
            self.fitted = True
            return self

        print(f"   📊 {len(probs_by_race)} courses pour CV")

        # Grid search pour γ global
        best_gamma = 1.0
        best_score = float("inf")

        for gamma in self.config.gamma_grid:
            score = self._cv_score_gamma(gamma, probs_by_race, labels_by_race, dates_by_race)
            if score < best_score:
                best_score = score
                best_gamma = gamma

        self.gamma = best_gamma
        print(f"   ✅ γ optimal global: {self.gamma:.2f} (Brier CV: {best_score:.4f})")

        # Calibration par cluster si demandé
        if cluster_col and cluster_col in df.columns and self.config.gamma_by_cluster:
            print(f"   📂 Apprentissage γ par {cluster_col}...")

            for cluster, group_df in df.groupby(cluster_col):
                if len(group_df) < self.config.min_samples_cluster:
                    continue

                # Grouper par course dans ce cluster
                probs_c = []
                labels_c = []
                dates_c = []

                for race_id, race_group in group_df.groupby(race_col):
                    if odds_col not in race_group.columns:
                        continue
                    if race_group[label_col].sum() != 1:
                        continue

                    odds = race_group[odds_col].values
                    if np.any(np.isnan(odds)) or np.any(odds <= 1):
                        continue

                    probs = self._odds_to_probs(odds)
                    labels = race_group[label_col].values
                    date = (
                        race_group[date_col].iloc[0] if date_col in race_group.columns else race_id
                    )

                    probs_c.append(probs)
                    labels_c.append(labels)
                    dates_c.append(date)

                if len(probs_c) < 30:
                    continue

                # Grid search pour ce cluster
                best_gamma_c = self.gamma
                best_score_c = float("inf")

                for gamma in self.config.gamma_grid:
                    score = self._cv_score_gamma(gamma, probs_c, labels_c, dates_c)
                    if score < best_score_c:
                        best_score_c = score
                        best_gamma_c = gamma

                self.gamma_by_cluster[cluster] = best_gamma_c
                print(f"      → {cluster}: γ = {best_gamma_c:.2f}")

        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        odds_col: str = "odds_market_preoff",
        race_col: str = "race_id",
        cluster_col: Optional[str] = None,
    ) -> np.ndarray:
        """
        Applique la correction γ aux probabilités marché.

        Returns:
            np.ndarray: Probabilités marché corrigées (p_market_debiased)
        """
        if not self.fitted:
            raise ValueError("MarketGammaCorrector non entraîné. Appeler fit() d'abord.")

        df = df.reset_index(drop=True)
        p_debiased = np.zeros(len(df))

        for race_id, group in df.groupby(race_col):
            idx = group.index.values

            if odds_col not in group.columns:
                # Fallback: pas de correction
                p_debiased[idx] = 1.0 / len(idx)
                continue

            odds = group[odds_col].values
            if np.any(np.isnan(odds)) or np.any(odds <= 1):
                p_debiased[idx] = 1.0 / len(idx)
                continue

            # Probabilités implicites normalisées
            probs = self._odds_to_probs(odds)

            # Déterminer γ à utiliser
            gamma = self.gamma
            if cluster_col and cluster_col in group.columns and self.gamma_by_cluster:
                cluster = group[cluster_col].iloc[0]
                gamma = self.gamma_by_cluster.get(cluster, self.gamma)

            # Appliquer correction
            p_corr = self._apply_gamma_correction(probs, gamma)
            p_debiased[idx] = p_corr

        return p_debiased

    def fit_transform(
        self,
        df: pd.DataFrame,
        odds_col: str = "odds_market_preoff",
        race_col: str = "race_id",
        label_col: str = "label_win",
        date_col: str = "date",
        cluster_col: Optional[str] = None,
    ) -> np.ndarray:
        """Fit puis transform."""
        self.fit(df, odds_col, race_col, label_col, date_col, cluster_col)
        return self.transform(df, odds_col, race_col, cluster_col)

    def save(self, path: str):
        """Sauvegarde le correcteur."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "gamma": self.gamma,
                    "gamma_by_cluster": self.gamma_by_cluster,
                    "config": self.config,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "MarketGammaCorrector":
        """Charge un correcteur sauvegardé."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        corrector = cls(data["config"])
        corrector.gamma = data["gamma"]
        corrector.gamma_by_cluster = data["gamma_by_cluster"]
        corrector.fitted = True
        return corrector


# ============================================================================
# ALPHA DYNAMIQUE
# ============================================================================


class DynamicAlphaCalculator:
    """
    Calcule α dynamiquement en fonction du temps restant et de la taille du pool.

    Formule: α = clamp(a0 + a1·log(pool) - a2·minutes_to_off, [α_min, α_max])

    Intuition:
    - Plus le pool est grand, plus le marché est informatif → α plus bas (plus de poids marché)
    - Plus on est proche du départ, plus le marché est informatif → α plus bas
    - Avec un petit pool ou loin du départ, on fait plus confiance au modèle → α plus haut
    """

    def __init__(self, config: MarketDebiasConfig):
        self.config = config
        self.a0: float = config.alpha_base
        self.a1: float = config.alpha_pool_coef
        self.a2: float = config.alpha_time_coef
        self.alpha_min: float = config.alpha_bounds[0]
        self.alpha_max: float = config.alpha_bounds[1]
        self.fitted: bool = False

        # EMA pour lissage
        self._alpha_history: List[float] = []

    def _compute_alpha(self, pool_size: float, minutes_to_off: float) -> float:
        """
        Calcule α pour une observation.

        α = clamp(a0 + a1·log(pool) - a2·minutes_to_off, [α_min, α_max])
        """
        # Log du pool (avec floor pour éviter log(0))
        log_pool = np.log(max(pool_size, 100.0))

        # Calcul brut
        alpha_raw = self.a0 + self.a1 * log_pool - self.a2 * minutes_to_off

        # Clamp dans les bornes
        alpha = np.clip(alpha_raw, self.alpha_min, self.alpha_max)

        return float(alpha)

    def _compute_alpha_vectorized(
        self, pool_sizes: np.ndarray, minutes_to_off: np.ndarray
    ) -> np.ndarray:
        """Version vectorisée du calcul d'α."""
        # Log du pool
        log_pools = np.log(np.maximum(pool_sizes, 100.0))

        # Calcul brut
        alpha_raw = self.a0 + self.a1 * log_pools - self.a2 * minutes_to_off

        # Clamp
        alpha = np.clip(alpha_raw, self.alpha_min, self.alpha_max)

        return alpha

    def fit(
        self,
        df: pd.DataFrame,
        p_model_col: str = "p_model",
        p_market_col: str = "p_market_debiased",
        label_col: str = "label_win",
        pool_col: str = "pool_size",
        time_col: str = "minutes_to_off",
        date_col: str = "date",
    ) -> "DynamicAlphaCalculator":
        """
        Apprend les coefficients optimaux (a0, a1, a2) par CV.
        """
        print("\n🔧 Apprentissage coefficients α dynamique")

        # Vérifier colonnes disponibles
        has_pool = pool_col in df.columns
        has_time = time_col in df.columns

        if not has_pool and not has_time:
            print("   ⚠️ Pas de colonnes pool/time, utilisation de α fixe")
            self.fitted = True
            return self

        # Préparer les données
        df = df.copy()

        if not has_pool:
            df[pool_col] = self.config.default_pool_size
            print(f"   ℹ️ Pool par défaut: {self.config.default_pool_size}")

        if not has_time:
            df[time_col] = self.config.default_minutes_to_off
            print(f"   ℹ️ Minutes par défaut: {self.config.default_minutes_to_off}")

        # Filtrer données valides
        valid_mask = ~df[p_model_col].isna() & ~df[p_market_col].isna() & ~df[label_col].isna()
        df = df[valid_mask].copy()

        if len(df) < 100:
            print(f"   ⚠️ Pas assez de données ({len(df)}), paramètres par défaut")
            self.fitted = True
            return self

        # Optimisation des coefficients
        def objective(params):
            a0, a1, a2 = params
            self.a0, self.a1, self.a2 = a0, a1, a2

            alphas = self._compute_alpha_vectorized(df[pool_col].values, df[time_col].values)

            # Blend avec ces alphas
            p_blend = self._blend_with_alphas(
                df[p_model_col].values, df[p_market_col].values, alphas
            )

            # Brier score
            return DebiasMetrics.brier_score(df[label_col].values, p_blend)

        # Grid search initial
        best_params = (self.a0, self.a1, self.a2)
        best_score = float("inf")

        for a0 in [0.4, 0.5, 0.6, 0.7]:
            for a1 in [0.02, 0.05, 0.08]:
                for a2 in [0.005, 0.01, 0.02]:
                    score = objective((a0, a1, a2))
                    if score < best_score:
                        best_score = score
                        best_params = (a0, a1, a2)

        self.a0, self.a1, self.a2 = best_params
        print(f"   ✅ Coefficients optimaux: a0={self.a0:.3f}, a1={self.a1:.3f}, a2={self.a2:.3f}")
        print(f"      Brier CV: {best_score:.4f}")

        self.fitted = True
        return self

    def _blend_with_alphas(
        self, p_model: np.ndarray, p_market: np.ndarray, alphas: np.ndarray
    ) -> np.ndarray:
        """Blend en logit-space avec α variable."""
        eps = 1e-10
        p_model = np.clip(p_model, eps, 1 - eps)
        p_market = np.clip(p_market, eps, 1 - eps)

        logit_model = logit(p_model)
        logit_market = logit(p_market)

        logit_blend = alphas * logit_model + (1 - alphas) * logit_market

        return expit(logit_blend)

    def get_alpha(self, pool_size: float = None, minutes_to_off: float = None) -> float:
        """Calcule α pour une observation."""
        if pool_size is None:
            pool_size = self.config.default_pool_size
        if minutes_to_off is None:
            minutes_to_off = self.config.default_minutes_to_off

        return self._compute_alpha(pool_size, minutes_to_off)

    def get_alpha_batch(
        self, df: pd.DataFrame, pool_col: str = "pool_size", time_col: str = "minutes_to_off"
    ) -> np.ndarray:
        """Calcule α pour un batch."""
        pool_sizes = (
            df[pool_col].values
            if pool_col in df.columns
            else np.full(len(df), self.config.default_pool_size)
        )
        minutes = (
            df[time_col].values
            if time_col in df.columns
            else np.full(len(df), self.config.default_minutes_to_off)
        )

        # Remplacer NaN par défauts
        pool_sizes = np.where(np.isnan(pool_sizes), self.config.default_pool_size, pool_sizes)
        minutes = np.where(np.isnan(minutes), self.config.default_minutes_to_off, minutes)

        return self._compute_alpha_vectorized(pool_sizes, minutes)

    def save(self, path: str):
        """Sauvegarde le calculateur."""
        with open(path, "wb") as f:
            pickle.dump({"a0": self.a0, "a1": self.a1, "a2": self.a2, "config": self.config}, f)

    @classmethod
    def load(cls, path: str) -> "DynamicAlphaCalculator":
        """Charge un calculateur sauvegardé."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        calc = cls(data["config"])
        calc.a0 = data["a0"]
        calc.a1 = data["a1"]
        calc.a2 = data["a2"]
        calc.fitted = True
        return calc


# ============================================================================
# BLEND DYNAMIQUE AVEC CORRECTION
# ============================================================================


class DynamicMarketModelBlender:
    """
    Blend modèle-marché avec:
    1. Correction gamma du biais favori/outsider
    2. Alpha dynamique fonction du temps/volume
    3. Blend en logit-space avec shrinkage

    Pipeline:
    1. Cotes → Probas implicites (normaliser)
    2. Correction gamma: p_mkt_corr ∝ p_mkt^γ
    3. α dynamique = f(pool, time)
    4. Blend: logit(p*) = α·logit(p_model) + (1-α)·logit(p_mkt_corr)
    5. Renormaliser par course
    """

    def __init__(self, config: MarketDebiasConfig = None):
        self.config = config or MarketDebiasConfig()

        # Composants
        self.gamma_corrector = MarketGammaCorrector(self.config)
        self.alpha_calculator = DynamicAlphaCalculator(self.config)

        # Métriques
        self.metrics: Dict[str, Any] = {}
        self.fitted: bool = False

    def _blend_logit_with_shrinkage(
        self, p_model: np.ndarray, p_market_corr: np.ndarray, alphas: np.ndarray
    ) -> np.ndarray:
        """
        Blend en logit-space avec shrinkage pour divergences extrêmes.
        """
        eps = 1e-10
        p_model = np.clip(p_model, eps, 1 - eps)
        p_market_corr = np.clip(p_market_corr, eps, 1 - eps)

        logit_model = logit(p_model)
        logit_market = logit(p_market_corr)

        # Shrinkage si divergence > seuil
        divergence = np.abs(logit_model - logit_market)
        shrink_factor = np.where(
            divergence > self.config.shrinkage_threshold,
            self.config.shrinkage_threshold / divergence,
            1.0,
        )

        # Ajuster logit_model vers logit_market si divergence forte
        logit_model_adj = logit_market + shrink_factor * (logit_model - logit_market)

        # Blend
        logit_blend = alphas * logit_model_adj + (1 - alphas) * logit_market

        return expit(logit_blend)

    def fit(
        self,
        df: pd.DataFrame,
        p_model_col: str = "p_model_calibrated",
        odds_col: str = "odds_market_preoff",
        race_col: str = "race_id",
        label_col: str = "label_win",
        date_col: str = "date",
        pool_col: str = "pool_size",
        time_col: str = "minutes_to_off",
        cluster_col: Optional[str] = None,  # 'discipline' ou 'hippodrome'
    ) -> "DynamicMarketModelBlender":
        """
        Entraîne le blender complet:
        1. Apprend γ par CV
        2. Apprend coefficients α dynamique
        """
        print("=" * 60)
        print("🚀 DYNAMIC MARKET-MODEL BLENDER - ENTRAÎNEMENT")
        print("=" * 60)

        # 1. Correction gamma
        self.gamma_corrector.fit(
            df,
            odds_col=odds_col,
            race_col=race_col,
            label_col=label_col,
            date_col=date_col,
            cluster_col=cluster_col,
        )

        # Appliquer correction gamma pour avoir p_market_debiased
        df = df.copy()
        df["p_market_debiased"] = self.gamma_corrector.transform(
            df, odds_col=odds_col, race_col=race_col, cluster_col=cluster_col
        )

        # 2. Alpha dynamique
        self.alpha_calculator.fit(
            df,
            p_model_col=p_model_col,
            p_market_col="p_market_debiased",
            label_col=label_col,
            pool_col=pool_col,
            time_col=time_col,
            date_col=date_col,
        )

        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        p_model_col: str = "p_model_calibrated",
        odds_col: str = "odds_market_preoff",
        race_col: str = "race_id",
        pool_col: str = "pool_size",
        time_col: str = "minutes_to_off",
        cluster_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Applique le blend complet.

        Returns:
            DataFrame avec colonnes ajoutées:
            - p_market_debiased: Probas marché corrigées (γ)
            - alpha_dynamic: α calculé pour chaque observation
            - p_win_blend: Probabilités finales blendées
            - fair_odds: Cotes équitables (1/p_win_blend)
            - value: Value (p_win_blend * odds - 1)
        """
        if not self.fitted:
            raise ValueError("DynamicMarketModelBlender non entraîné. Appeler fit() d'abord.")

        df = df.copy().reset_index(drop=True)

        # 1. Correction gamma
        df["p_market_debiased"] = self.gamma_corrector.transform(
            df, odds_col=odds_col, race_col=race_col, cluster_col=cluster_col
        )

        # 2. Alpha dynamique
        df["alpha_dynamic"] = self.alpha_calculator.get_alpha_batch(
            df, pool_col=pool_col, time_col=time_col
        )

        # 3. Blend en logit-space avec shrinkage
        p_blend_raw = self._blend_logit_with_shrinkage(
            df[p_model_col].values, df["p_market_debiased"].values, df["alpha_dynamic"].values
        )

        # 4. Renormaliser par course
        df["p_win_blend_raw"] = p_blend_raw
        df["p_win_blend"] = df.groupby(race_col)["p_win_blend_raw"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else x
        )

        # 5. Calculs dérivés
        eps = 1e-10
        df["fair_odds"] = 1.0 / np.clip(df["p_win_blend"].values, eps, 1.0)

        if odds_col in df.columns:
            df["value"] = df["p_win_blend"] * df[odds_col] - 1

        # Cleanup
        df.drop(columns=["p_win_blend_raw"], inplace=True, errors="ignore")

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        p_model_col: str = "p_model_calibrated",
        odds_col: str = "odds_market_preoff",
        race_col: str = "race_id",
        label_col: str = "label_win",
        date_col: str = "date",
        pool_col: str = "pool_size",
        time_col: str = "minutes_to_off",
        cluster_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fit puis transform."""
        self.fit(
            df,
            p_model_col,
            odds_col,
            race_col,
            label_col,
            date_col,
            pool_col,
            time_col,
            cluster_col,
        )
        return self.transform(df, p_model_col, odds_col, race_col, pool_col, time_col, cluster_col)

    def evaluate(
        self,
        df: pd.DataFrame,
        p_model_col: str = "p_model_calibrated",
        odds_col: str = "odds_market_preoff",
        label_col: str = "label_win",
        value_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Évalue le blender avec métriques de contrôle.

        Returns:
            Dict avec Brier, ECE, logloss, lift EV
        """
        results = {}

        # Vérifier colonnes
        if "p_win_blend" not in df.columns:
            raise ValueError("Exécuter transform() avant evaluate()")

        p_blend = df["p_win_blend"].values
        p_market = df["p_market_debiased"].values if "p_market_debiased" in df.columns else None
        p_model = df[p_model_col].values
        labels = df[label_col].values
        odds = df[odds_col].values if odds_col in df.columns else None

        # Métriques calibration - Blend
        results["blend"] = {
            "brier": DebiasMetrics.brier_score(labels, p_blend),
            "ece": DebiasMetrics.expected_calibration_error(
                labels, p_blend, self.config.n_bins_ece
            ),
            "logloss": DebiasMetrics.log_loss(labels, p_blend),
        }

        # Métriques calibration - Modèle seul
        results["model_only"] = {
            "brier": DebiasMetrics.brier_score(labels, p_model),
            "ece": DebiasMetrics.expected_calibration_error(
                labels, p_model, self.config.n_bins_ece
            ),
            "logloss": DebiasMetrics.log_loss(labels, p_model),
        }

        # Métriques calibration - Marché seul (corrigé)
        if p_market is not None:
            results["market_only"] = {
                "brier": DebiasMetrics.brier_score(labels, p_market),
                "ece": DebiasMetrics.expected_calibration_error(
                    labels, p_market, self.config.n_bins_ece
                ),
                "logloss": DebiasMetrics.log_loss(labels, p_market),
            }

        # Lift EV
        if odds is not None and p_market is not None:
            results["ev_lift"] = DebiasMetrics.expected_value_lift(
                labels, p_blend, p_market, odds, value_threshold
            )

        # Statistiques α
        if "alpha_dynamic" in df.columns:
            alphas = df["alpha_dynamic"].values
            results["alpha_stats"] = {
                "mean": float(np.mean(alphas)),
                "std": float(np.std(alphas)),
                "min": float(np.min(alphas)),
                "max": float(np.max(alphas)),
                "p25": float(np.percentile(alphas, 25)),
                "p75": float(np.percentile(alphas, 75)),
            }

        # Statistiques γ
        results["gamma"] = {
            "global": self.gamma_corrector.gamma,
            "by_cluster": self.gamma_corrector.gamma_by_cluster,
        }

        self.metrics = results
        return results

    def print_evaluation_report(self, results: Dict[str, Any] = None):
        """Affiche un rapport d'évaluation formaté."""
        if results is None:
            results = self.metrics

        print("\n" + "=" * 60)
        print("📊 RAPPORT D'ÉVALUATION - BLEND DYNAMIQUE")
        print("=" * 60)

        print("\n🎯 Métriques de Calibration:")
        print("-" * 40)
        print(f"{'Source':<20} {'Brier':<12} {'ECE':<12} {'LogLoss':<12}")
        print("-" * 40)

        for source in ["blend", "model_only", "market_only"]:
            if source in results:
                m = results[source]
                label = {
                    "blend": "Blend",
                    "model_only": "Modèle seul",
                    "market_only": "Marché corrigé",
                }[source]
                print(f"{label:<20} {m['brier']:<12.4f} {m['ece']:<12.4f} {m['logloss']:<12.4f}")

        if "ev_lift" in results:
            print("\n💰 Lift Expected Value:")
            print("-" * 40)
            ev = results["ev_lift"]
            print(f"   EV Blend:  {ev['ev_blend']:.4f} ({ev['n_bets_blend']} paris)")
            print(f"   EV Marché: {ev['ev_market']:.4f} ({ev['n_bets_market']} paris)")
            print(f"   Lift:      {ev['lift_pct']:.1f}%")

        if "alpha_stats" in results:
            print("\n📈 Statistiques α dynamique:")
            print("-" * 40)
            a = results["alpha_stats"]
            print(f"   Moyenne: {a['mean']:.3f} ± {a['std']:.3f}")
            print(f"   Range:   [{a['min']:.3f}, {a['max']:.3f}]")
            print(f"   IQR:     [{a['p25']:.3f}, {a['p75']:.3f}]")

        if "gamma" in results:
            print("\n🔧 Paramètre γ (correction biais):")
            print("-" * 40)
            print(f"   Global: {results['gamma']['global']:.2f}")
            if results["gamma"]["by_cluster"]:
                for cluster, g in results["gamma"]["by_cluster"].items():
                    print(f"   {cluster}: {g:.2f}")

        print("\n" + "=" * 60)

    def save(self, output_dir: str):
        """Sauvegarde tous les composants."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Gamma corrector
        self.gamma_corrector.save(os.path.join(output_dir, f"gamma_corrector_{timestamp}.pkl"))

        # Alpha calculator
        self.alpha_calculator.save(os.path.join(output_dir, f"alpha_calculator_{timestamp}.pkl"))

        # Métriques
        with open(os.path.join(output_dir, f"blend_metrics_{timestamp}.json"), "w") as f:
            import json

            # Convertir numpy types pour JSON
            def convert(o):
                if isinstance(o, np.floating):
                    return float(o)
                if isinstance(o, np.integer):
                    return int(o)
                return o

            json.dump(self.metrics, f, default=convert, indent=2)

        print(f"✅ Artefacts sauvegardés dans {output_dir}")

    @classmethod
    def load(cls, output_dir: str, timestamp: str = None) -> "DynamicMarketModelBlender":
        """Charge un blender sauvegardé."""
        import glob

        # Trouver les fichiers les plus récents si timestamp non spécifié
        if timestamp is None:
            gamma_files = glob.glob(os.path.join(output_dir, "gamma_corrector_*.pkl"))
            if not gamma_files:
                raise FileNotFoundError(f"Aucun fichier gamma_corrector trouvé dans {output_dir}")
            gamma_file = max(gamma_files)
            timestamp = gamma_file.split("_")[-1].replace(".pkl", "")

        blender = cls()
        blender.gamma_corrector = MarketGammaCorrector.load(
            os.path.join(output_dir, f"gamma_corrector_{timestamp}.pkl")
        )
        blender.alpha_calculator = DynamicAlphaCalculator.load(
            os.path.join(output_dir, f"alpha_calculator_{timestamp}.pkl")
        )
        blender.fitted = True

        return blender


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================


def demo_usage():
    """Démo d'utilisation du module."""

    # Générer données synthétiques
    np.random.seed(42)
    n_races = 100
    n_runners_per_race = 8

    data = []
    for race_id in range(n_races):
        # Vraies probabilités
        true_probs = np.random.dirichlet(np.ones(n_runners_per_race) * 2)

        # Cotes marché (avec biais favori)
        market_probs = true_probs**0.85  # Biais: favoris surestimés
        market_probs /= market_probs.sum()
        overround = 1.15
        market_odds = overround / market_probs

        # Probas modèle (bruit)
        model_logits = np.log(true_probs + 1e-6) + np.random.randn(n_runners_per_race) * 0.5
        model_probs = np.exp(model_logits) / np.exp(model_logits).sum()

        # Gagnant
        winner_idx = np.random.choice(n_runners_per_race, p=true_probs)

        for i in range(n_runners_per_race):
            data.append(
                {
                    "race_id": f"R{race_id}",
                    "date": f"2024-01-{(race_id % 30) + 1:02d}",
                    "discipline": np.random.choice(["plat", "trot", "obstacle"]),
                    "p_model_calibrated": model_probs[i],
                    "odds_market_preoff": market_odds[i],
                    "label_win": 1 if i == winner_idx else 0,
                    "pool_size": np.random.uniform(30000, 150000),
                    "minutes_to_off": np.random.uniform(5, 60),
                }
            )

    df = pd.DataFrame(data)

    # Créer et entraîner le blender
    config = MarketDebiasConfig()
    blender = DynamicMarketModelBlender(config)

    # Fit + Transform
    df_result = blender.fit_transform(
        df,
        p_model_col="p_model_calibrated",
        odds_col="odds_market_preoff",
        race_col="race_id",
        label_col="label_win",
        date_col="date",
        pool_col="pool_size",
        time_col="minutes_to_off",
        cluster_col="discipline",
    )

    # Évaluation
    results = blender.evaluate(
        df_result,
        p_model_col="p_model_calibrated",
        odds_col="odds_market_preoff",
        label_col="label_win",
    )

    blender.print_evaluation_report(results)

    # Afficher quelques résultats
    print("\n📋 Exemple de résultats (5 premières lignes):")
    print(
        df_result[
            [
                "race_id",
                "p_model_calibrated",
                "p_market_debiased",
                "alpha_dynamic",
                "p_win_blend",
                "fair_odds",
                "value",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    demo_usage()
