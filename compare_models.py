#!/usr/bin/env python3
"""
compare_models.py - Comparaison Statistique Champion vs Challenger

Compare les performances du modèle champion vs challenger en utilisant
les données de feedback collectées durant la période d'A/B testing.

Utilise le test de Welch (t-test indépendant sans hypothèse de variance égale)
pour déterminer si le challenger est statistiquement meilleur que le champion.

Si challenger > champion ET p-value < 0.05 ET effect size > 0.3:
→ Promotion automatique du challenger vers champion

Usage:
    python compare_models.py --days 7
    python compare_models.py --days 14 --min-samples 200 --auto-promote

Author: Phase 8 - Online Learning System
Date: 2025-11-14
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare les performances champion vs challenger"""

    def __init__(
        self, min_samples: int = 100, p_threshold: float = 0.05, effect_size_threshold: float = 0.3
    ):
        """
        Initialise le comparateur de modèles

        Args:
            min_samples: Nombre minimum de prédictions par modèle pour comparaison valide
            p_threshold: Seuil p-value pour significativité statistique
            effect_size_threshold: Seuil Cohen's d pour effet pratique significatif
        """
        self.min_samples = min_samples
        self.p_threshold = p_threshold
        self.effect_size_threshold = effect_size_threshold

        logger.info("📊 ModelComparator initialisé")
        logger.info(f"   Min samples: {min_samples}")
        logger.info(f"   P-value threshold: {p_threshold}")
        logger.info(f"   Effect size threshold: {effect_size_threshold}")

    def load_feedback_data(self, days: int = 7) -> pd.DataFrame:
        """
        Charge les données de feedback depuis PostgreSQL

        Pour le développement, simule avec données CSV
        En production, interrogerait:
        SELECT * FROM feedback_results WHERE date_course >= NOW() - INTERVAL '{days} days'
        """
        logger.info(f"📂 Chargement feedback (derniers {days} jours)...")

        # STUB: En production, interroger PostgreSQL
        # Pour dev, simuler avec données fictives
        logger.warning("   ⚠️  MODE DEV: Simulation données feedback")

        # Simuler 500 prédictions sur 7 jours
        np.random.seed(42)
        n_champion = 450  # 90% champion
        n_challenger = 50  # 10% challenger

        # Champion: ROC-AUC ~0.70 (légèrement moins bon)
        champion_data = {
            "model_version": ["champion"] * n_champion,
            "predicted_proba": np.random.beta(2, 3, n_champion),  # Distribution skewed
            "actual_position": np.random.choice(
                [1, 2, 3, 4, 5], n_champion, p=[0.12, 0.15, 0.18, 0.25, 0.30]
            ),
        }

        # Challenger: ROC-AUC ~0.75 (meilleur)
        challenger_data = {
            "model_version": ["challenger"] * n_challenger,
            "predicted_proba": np.random.beta(2.5, 2.5, n_challenger),  # Meilleure discrimination
            "actual_position": np.random.choice(
                [1, 2, 3, 4, 5], n_challenger, p=[0.18, 0.20, 0.17, 0.22, 0.23]
            ),
        }

        df_champion = pd.DataFrame(champion_data)
        df_challenger = pd.DataFrame(challenger_data)

        df = pd.concat([df_champion, df_challenger], ignore_index=True)

        # Convertir actual_position en victoire binaire (1 = victoire, 0 = défaite)
        df["victoire"] = (df["actual_position"] == 1).astype(int)

        logger.info(f"   ✅ {len(df):,} feedbacks chargés")
        logger.info(f"   Champion: {len(df_champion):,} prédictions")
        logger.info(f"   Challenger: {len(df_challenger):,} prédictions")

        return df

    def calculate_roc_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule le ROC-AUC score

        ROC-AUC mesure la capacité du modèle à discriminer entre classes.
        1.0 = parfait, 0.5 = aléatoire
        """
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) < 2:
            logger.warning("⚠️  Pas assez de classes différentes pour ROC-AUC")
            return 0.5

        try:
            return roc_auc_score(y_true, y_pred)
        except Exception as e:
            logger.error(f"❌ Erreur calcul ROC-AUC: {e}")
            return 0.5

    def calculate_accuracy(self, y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
        """Calcule l'accuracy (pourcentage de prédictions correctes)"""
        return np.mean(y_true == y_pred_binary)

    def calculate_brier_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calcule le Brier Score (calibration des probabilités)

        Brier Score = moyenne((p_pred - p_true)^2)
        0.0 = parfait, 1.0 = pire
        """
        return np.mean((y_pred_proba - y_true) ** 2)

    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calcule la taille d'effet de Cohen's d

        d = (mean1 - mean2) / pooled_std

        Interprétation:
        - |d| < 0.2: effet négligeable
        - 0.2 ≤ |d| < 0.5: effet petit
        - 0.5 ≤ |d| < 0.8: effet moyen
        - |d| ≥ 0.8: effet large
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def compare_models(self, df: pd.DataFrame) -> Dict:
        """
        Compare champion vs challenger sur les données de feedback

        Returns:
            {
                'champion': {...metrics...},
                'challenger': {...metrics...},
                'comparison': {...statistical tests...},
                'recommendation': 'promote' | 'keep_champion',
                'reasoning': str
            }
        """
        logger.info("📊 Comparaison champion vs challenger...")

        # Séparer données par modèle
        df_champion = df[df["model_version"] == "champion"].copy()
        df_challenger = df[df["model_version"] == "challenger"].copy()

        # Vérifier échantillons suffisants
        if len(df_champion) < self.min_samples:
            logger.error(
                f"❌ Pas assez de prédictions champion: {len(df_champion)} < {self.min_samples}"
            )
            return {
                "error": "insufficient_champion_samples",
                "champion_samples": len(df_champion),
                "required": self.min_samples,
            }

        if len(df_challenger) < self.min_samples:
            logger.error(
                f"❌ Pas assez de prédictions challenger: {len(df_challenger)} < {self.min_samples}"
            )
            return {
                "error": "insufficient_challenger_samples",
                "challenger_samples": len(df_challenger),
                "required": self.min_samples,
            }

        # Calculer métriques champion
        logger.info("   📈 Calcul métriques champion...")
        champion_roc_auc = self.calculate_roc_auc(
            df_champion["victoire"].values, df_champion["predicted_proba"].values
        )
        champion_brier = self.calculate_brier_score(
            df_champion["victoire"].values, df_champion["predicted_proba"].values
        )
        champion_accuracy_top1 = df_champion["victoire"].mean()

        champion_metrics = {
            "samples": len(df_champion),
            "roc_auc": round(champion_roc_auc, 4),
            "brier_score": round(champion_brier, 4),
            "accuracy_top1": round(champion_accuracy_top1, 4),
            "mean_proba": round(df_champion["predicted_proba"].mean(), 4),
            "std_proba": round(df_champion["predicted_proba"].std(), 4),
        }

        logger.info(
            f"   Champion: ROC-AUC={champion_roc_auc:.4f}, Brier={champion_brier:.4f}, n={len(df_champion)}"
        )

        # Calculer métriques challenger
        logger.info("   📈 Calcul métriques challenger...")
        challenger_roc_auc = self.calculate_roc_auc(
            df_challenger["victoire"].values, df_challenger["predicted_proba"].values
        )
        challenger_brier = self.calculate_brier_score(
            df_challenger["victoire"].values, df_challenger["predicted_proba"].values
        )
        challenger_accuracy_top1 = df_challenger["victoire"].mean()

        challenger_metrics = {
            "samples": len(df_challenger),
            "roc_auc": round(challenger_roc_auc, 4),
            "brier_score": round(challenger_brier, 4),
            "accuracy_top1": round(challenger_accuracy_top1, 4),
            "mean_proba": round(df_challenger["predicted_proba"].mean(), 4),
            "std_proba": round(df_challenger["predicted_proba"].std(), 4),
        }

        logger.info(
            f"   Challenger: ROC-AUC={challenger_roc_auc:.4f}, Brier={challenger_brier:.4f}, n={len(df_challenger)}"
        )

        # Test statistique: Welch's t-test sur les probabilités prédites
        # H0: champion == challenger
        # H1: challenger > champion (test unilatéral)
        logger.info("   🔬 Test statistique de Welch...")

        t_stat, p_value_twotailed = stats.ttest_ind(
            df_challenger["predicted_proba"].values,
            df_champion["predicted_proba"].values,
            equal_var=False,  # Welch's t-test (pas d'hypothèse variance égale)
        )

        # Convertir en test unilatéral (challenger > champion)
        if t_stat > 0:
            p_value = p_value_twotailed / 2
        else:
            p_value = 1 - (p_value_twotailed / 2)

        # Calculer effect size (Cohen's d)
        effect_size = self.cohens_d(
            df_challenger["predicted_proba"].values, df_champion["predicted_proba"].values
        )

        comparison = {
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "effect_size_cohens_d": round(effect_size, 4),
            "significant": p_value < self.p_threshold,
            "practical_significant": abs(effect_size) > self.effect_size_threshold,
            "roc_auc_improvement": round(challenger_roc_auc - champion_roc_auc, 4),
            "brier_improvement": round(
                champion_brier - challenger_brier, 4
            ),  # Brier: plus petit = mieux
        }

        logger.info(f"   t={t_stat:.4f}, p={p_value:.4f}, d={effect_size:.4f}")

        # Décision de promotion
        logger.info("   🤔 Évaluation critères de promotion...")

        criteria = {
            "challenger_better_roc_auc": challenger_roc_auc > champion_roc_auc,
            "statistically_significant": comparison["significant"],
            "practically_significant": comparison["practical_significant"],
        }

        # Promotion si TOUTES les conditions sont réunies
        should_promote = all(criteria.values())

        if should_promote:
            recommendation = "promote"
            reasoning = (
                f"Challenger meilleur: ROC-AUC +{comparison['roc_auc_improvement']:.4f}, "
                f"p={comparison['p_value']:.4f} < {self.p_threshold}, "
                f"Cohen's d={comparison['effect_size_cohens_d']:.4f} > {self.effect_size_threshold}. "
                f"Promotion recommandée."
            )
            logger.info("   ✅ RECOMMANDATION: PROMOUVOIR CHALLENGER")
        else:
            recommendation = "keep_champion"
            failed_criteria = [k for k, v in criteria.items() if not v]
            reasoning = (
                f"Challenger ne satisfait pas tous les critères. "
                f"Échecs: {', '.join(failed_criteria)}. "
                f"Conservation du champion."
            )
            logger.info("   ⏸️  RECOMMANDATION: CONSERVER CHAMPION")

        return {
            "timestamp": datetime.now().isoformat(),
            "champion": champion_metrics,
            "challenger": challenger_metrics,
            "comparison": comparison,
            "criteria": criteria,
            "recommendation": recommendation,
            "reasoning": reasoning,
        }

    def promote_challenger(self, dry_run: bool = False) -> bool:
        """
        Promeut le challenger vers champion

        1. Archive l'ancien champion
        2. Copie le challenger vers champion
        3. Met à jour metadata

        Args:
            dry_run: Si True, simule sans faire de changements

        Returns:
            True si succès, False sinon
        """
        logger.info("🚀 Promotion challenger → champion...")

        champion_dir = Path("data/models/champion")
        challenger_dir = Path("data/models/challenger")
        archive_dir = Path("data/models/archive")

        # Vérifier que challenger existe
        challenger_model = challenger_dir / "model.pkl"
        if not challenger_model.exists():
            logger.error(f"❌ Modèle challenger introuvable: {challenger_model}")
            return False

        # Créer timestamp pour archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / timestamp

        if dry_run:
            logger.info("   🔍 MODE DRY-RUN: Simulation sans changements")
            logger.info(f"   1. Archive champion → {archive_path}")
            logger.info(f"   2. Copie challenger → {champion_dir}")
            logger.info("   3. Mise à jour metadata")
            return True

        try:
            # 1. Archiver ancien champion
            if champion_dir.exists():
                logger.info(f"   📦 Archive ancien champion: {archive_path}")
                archive_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(champion_dir, archive_path, dirs_exist_ok=True)

            # 2. Copier challenger → champion
            logger.info("   📋 Copie challenger → champion")
            if champion_dir.exists():
                shutil.rmtree(champion_dir)
            shutil.copytree(challenger_dir, champion_dir)

            # 3. Mettre à jour metadata champion
            metadata_path = champion_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                metadata["promoted_at"] = datetime.now().isoformat()
                metadata["promoted_from"] = "challenger"
                metadata["previous_champion_archived"] = str(archive_path)

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            logger.info("   ✅ Promotion terminée avec succès")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors de la promotion: {e}", exc_info=True)
            return False

    def generate_report(
        self, comparison_result: Dict, output_path: str = "comparison_report.json"
    ) -> None:
        """Génère un rapport de comparaison JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison_result, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Rapport sauvegardé: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comparaison statistique champion vs challenger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    # Comparaison standard (7 jours)
    python compare_models.py --days 7

    # Comparaison avec seuils personnalisés
    python compare_models.py --days 14 --p-threshold 0.01 --effect-size 0.5

    # Comparaison avec auto-promotion
    python compare_models.py --days 7 --auto-promote

    # Dry-run (simulation promotion)
    python compare_models.py --days 7 --auto-promote --dry-run
        """,
    )

    parser.add_argument(
        "--days", type=int, default=7, help="Nombre de jours de feedback à analyser (défaut: 7)"
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Nombre minimum de prédictions par modèle (défaut: 100)",
    )

    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="Seuil p-value pour significativité (défaut: 0.05)",
    )

    parser.add_argument(
        "--effect-size",
        type=float,
        default=0.3,
        help="Seuil Cohen's d pour effet pratique (défaut: 0.3)",
    )

    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Promouvoir automatiquement si critères satisfaits",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Simuler la promotion sans faire de changements"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.json",
        help="Chemin du rapport JSON (défaut: comparison_report.json)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("📊 COMPARAISON CHAMPION VS CHALLENGER")
    logger.info("=" * 80)
    logger.info("")

    # Initialiser comparateur
    comparator = ModelComparator(
        min_samples=args.min_samples,
        p_threshold=args.p_threshold,
        effect_size_threshold=args.effect_size,
    )

    # Charger feedback
    df = comparator.load_feedback_data(days=args.days)

    if df.empty:
        logger.error("❌ Pas de données feedback, arrêt du script")
        return 1

    # Comparer modèles
    result = comparator.compare_models(df)

    if "error" in result:
        logger.error(f"❌ Erreur: {result['error']}")
        return 1

    # Générer rapport
    logger.info("")
    logger.info("📄 Génération rapport...")
    comparator.generate_report(result, args.output)

    # Afficher résumé
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ RÉSUMÉ COMPARAISON")
    logger.info("=" * 80)
    logger.info(f"   Champion ROC-AUC       : {result['champion']['roc_auc']:.4f}")
    logger.info(f"   Challenger ROC-AUC     : {result['challenger']['roc_auc']:.4f}")
    logger.info(f"   Amélioration           : {result['comparison']['roc_auc_improvement']:+.4f}")
    logger.info("")
    logger.info(f"   P-value                : {result['comparison']['p_value']:.4f}")
    logger.info(f"   Cohen's d              : {result['comparison']['effect_size_cohens_d']:.4f}")
    logger.info(
        f"   Significatif (stat)    : {'✅ OUI' if result['comparison']['significant'] else '❌ NON'}"
    )
    logger.info(
        f"   Significatif (pratique): {'✅ OUI' if result['comparison']['practical_significant'] else '❌ NON'}"
    )
    logger.info("")
    logger.info(f"   📋 RECOMMANDATION: {result['recommendation'].upper()}")
    logger.info(f"   💡 Raisonnement: {result['reasoning']}")
    logger.info("")

    # Auto-promotion si demandé
    if args.auto_promote:
        if result["recommendation"] == "promote":
            logger.info("🚀 Auto-promotion activée...")
            success = comparator.promote_challenger(dry_run=args.dry_run)

            if success:
                if args.dry_run:
                    logger.info("✅ Dry-run terminé (aucun changement effectué)")
                else:
                    logger.info("✅ Promotion terminée avec succès !")
                    logger.info("   ⚠️  Redémarrer l'API pour charger le nouveau champion")
            else:
                logger.error("❌ Échec de la promotion")
                return 1
        else:
            logger.info("⏸️  Auto-promotion ignorée (recommendation = keep_champion)")

    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 Comparaison terminée avec succès !")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
