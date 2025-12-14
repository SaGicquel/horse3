#!/usr/bin/env python3
"""
detect_drift.py - Détection de Drift des Features ML

Détecte les changements de distribution entre les données d'entraînement (baseline)
et les données de production (feedback récent) pour les 62 features ML.

Tests statistiques:
- Kolmogorov-Smirnov (KS): mesure la distance maximale entre deux distributions
- Jensen-Shannon Divergence (JS): mesure la divergence entre deux distributions

Seuils:
- KS > 0.3 OU JS > 0.15 → Drift CRITIQUE (alerte immédiate)
- KS > 0.2 OU JS > 0.1 → Drift WARNING (surveillance)
- Sinon → Pas de drift détecté

Usage:
    python detect_drift.py --baseline data/ml_features_complete.csv --days 7
    python detect_drift.py --output drift_report.json --threshold-ks 0.25

Author: Phase 8 - Online Learning System
Date: 2025-11-14
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Détecteur de drift pour features ML"""
    
    # 62 features ML utilisées pour l'entraînement
    ML_FEATURES = [
        # Forme (7)
        'forme_5c', 'forme_10c', 'nb_courses_12m', 'nb_victoires_12m',
        'nb_places_12m', 'jours_depuis_derniere_course', 'regularite_places',
        
        # Aptitude (3)
        'aptitude_distance', 'aptitude_piste', 'aptitude_hippodrome',
        
        # Jockey/Entraineur (6)
        'taux_victoires_jockey', 'taux_places_jockey', 
        'taux_victoires_entraineur', 'taux_places_entraineur',
        'synergie_jockey_cheval', 'synergie_entraineur_cheval',
        
        # Course (3)
        'distance_norm', 'nb_partants', 'niveau_moyen_concurrent',
        
        # Marché (5)
        'cote_sp', 'cote_turfbzh', 'rang_cote_sp', 'rang_cote_turfbzh',
        'ecart_cote_ia',
        
        # Handicap (2)
        'poids_cheval', 'diff_poids_median',
        
        # Méteo/Track (4)
        'piste_tres_bonne', 'piste_bonne', 'piste_souple', 'piste_tres_souple',
        
        # Encoded categoricals (32)
        'specialite_ATTELE', 'specialite_MONTE', 'specialite_PLAT', 
        'specialite_OBSTACLES', 'sexe_F', 'sexe_H', 'sexe_M',
        'discipline_ATTELE', 'discipline_GALOP_PLAT', 'discipline_GALOP_OBSTACLES',
        'discipline_TROT_ATTELE', 'discipline_TROT_MONTE',
        'hippodrome_AUTEUIL', 'hippodrome_CHANTILLY', 'hippodrome_DEAUVILLE',
        'hippodrome_LONGCHAMP', 'hippodrome_MAISONS-LAFFITTE', 
        'hippodrome_SAINT-CLOUD', 'hippodrome_VINCENNES',
        'type_piste_CORDE_DROITE', 'type_piste_CORDE_GAUCHE', 
        'type_piste_DROITE', 'type_piste_LIGNE_DROITE',
        'categorie_course_APPRENTIS', 'categorie_course_CONDITIONS',
        'categorie_course_GROUPE_1', 'categorie_course_GROUPE_2',
        'categorie_course_GROUPE_3', 'categorie_course_HANDICAP',
        'categorie_course_LISTED', 'categorie_course_RECLAMERS',
        'categorie_course_VENTE'
    ]
    
    def __init__(
        self,
        threshold_ks: float = 0.3,
        threshold_js: float = 0.15,
        warning_threshold_ks: float = 0.2,
        warning_threshold_js: float = 0.1
    ):
        """
        Initialise le détecteur de drift
        
        Args:
            threshold_ks: Seuil critique pour KS test (défaut 0.3)
            threshold_js: Seuil critique pour JS divergence (défaut 0.15)
            warning_threshold_ks: Seuil warning pour KS (défaut 0.2)
            warning_threshold_js: Seuil warning pour JS (défaut 0.1)
        """
        self.threshold_ks = threshold_ks
        self.threshold_js = threshold_js
        self.warning_threshold_ks = warning_threshold_ks
        self.warning_threshold_js = warning_threshold_js
        
        logger.info("🔍 DriftDetector initialisé")
        logger.info(f"   Seuils KS: warning={warning_threshold_ks}, critical={threshold_ks}")
        logger.info(f"   Seuils JS: warning={warning_threshold_js}, critical={threshold_js}")
    
    def load_baseline(self, filepath: str) -> pd.DataFrame:
        """Charge les données baseline (training data)"""
        logger.info(f"📂 Chargement baseline depuis {filepath}...")
        df = pd.read_csv(filepath)
        logger.info(f"   ✅ {len(df):,} lignes chargées")
        return df
    
    def load_production_data(self, days: int = 7) -> pd.DataFrame:
        """
        Charge les données de production depuis feedback_results
        
        Pour le développement, on simule avec les données récentes
        En production, cela interrogerait PostgreSQL:
        SELECT * FROM feedback_results WHERE date_course >= NOW() - INTERVAL '{days} days'
        """
        logger.info(f"📊 Chargement données production (derniers {days} jours)...")
        
        # STUB: En production, interroger PostgreSQL
        # Pour dev, simuler avec un échantillon aléatoire du baseline
        logger.warning("   ⚠️  MODE DEV: Simulation données production avec baseline sample")
        
        try:
            baseline_path = "data/ml_features_complete.csv"
            df = pd.read_csv(baseline_path)
            
            # Simuler données de production: 500 lignes récentes avec drift artificiel
            sample_size = min(500, len(df) // 10)
            df_prod = df.sample(n=sample_size, random_state=42).copy()
            
            # Ajouter du drift artificiel sur quelques features pour test
            np.random.seed(42)
            df_prod['forme_5c'] = df_prod['forme_5c'] * np.random.uniform(0.8, 1.2, len(df_prod))
            df_prod['cote_sp'] = df_prod['cote_sp'] * np.random.uniform(0.9, 1.3, len(df_prod))
            
            logger.info(f"   ✅ {len(df_prod):,} lignes production simulées")
            return df_prod
            
        except FileNotFoundError:
            logger.error(f"   ❌ Fichier baseline introuvable: {baseline_path}")
            return pd.DataFrame()
    
    def compute_ks_test(
        self,
        baseline_values: np.ndarray,
        production_values: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calcule le test de Kolmogorov-Smirnov
        
        Returns:
            (ks_statistic, p_value)
        """
        # Supprimer les NaN
        baseline_clean = baseline_values[~np.isnan(baseline_values)]
        production_clean = production_values[~np.isnan(production_values)]
        
        if len(baseline_clean) < 10 or len(production_clean) < 10:
            return 0.0, 1.0
        
        ks_stat, p_value = ks_2samp(baseline_clean, production_clean)
        return ks_stat, p_value
    
    def compute_js_divergence(
        self,
        baseline_values: np.ndarray,
        production_values: np.ndarray,
        bins: int = 50
    ) -> float:
        """
        Calcule la divergence de Jensen-Shannon
        
        Returns:
            js_divergence (0 = identique, 1 = totalement différent)
        """
        # Supprimer les NaN
        baseline_clean = baseline_values[~np.isnan(baseline_values)]
        production_clean = production_values[~np.isnan(production_values)]
        
        if len(baseline_clean) < 10 or len(production_clean) < 10:
            return 0.0
        
        # Créer des histogrammes avec les mêmes bins
        min_val = float(min(baseline_clean.min(), production_clean.min()))
        max_val = float(max(baseline_clean.max(), production_clean.max()))
        
        # Éviter division par zéro
        if max_val - min_val < 1e-6:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        hist_baseline, _ = np.histogram(baseline_clean, bins=bin_edges, density=True)
        hist_production, _ = np.histogram(production_clean, bins=bin_edges, density=True)
        
        # Normaliser pour avoir des probabilités
        hist_baseline = hist_baseline / hist_baseline.sum()
        hist_production = hist_production / hist_production.sum()
        
        # Ajouter epsilon pour éviter log(0)
        epsilon = 1e-10
        hist_baseline = hist_baseline + epsilon
        hist_production = hist_production + epsilon
        
        # Renormaliser
        hist_baseline = hist_baseline / hist_baseline.sum()
        hist_production = hist_production / hist_production.sum()
        
        # Calculer JS divergence
        js_div = jensenshannon(hist_baseline, hist_production)
        
        return float(js_div)
    
    def detect_feature_drift(
        self,
        feature_name: str,
        baseline_values: np.ndarray,
        production_values: np.ndarray
    ) -> Dict:
        """
        Détecte le drift pour une feature spécifique
        
        Returns:
            {
                'feature': str,
                'ks_statistic': float,
                'ks_pvalue': float,
                'js_divergence': float,
                'drift_detected': bool,
                'severity': 'none' | 'warning' | 'critical',
                'baseline_mean': float,
                'production_mean': float,
                'baseline_std': float,
                'production_std': float,
                'baseline_samples': int,
                'production_samples': int
            }
        """
        # Calcul KS test
        ks_stat, ks_pval = self.compute_ks_test(baseline_values, production_values)
        
        # Calcul JS divergence
        js_div = self.compute_js_divergence(baseline_values, production_values)
        
        # Déterminer severity
        severity = 'none'
        drift_detected = False
        
        if ks_stat >= self.threshold_ks or js_div >= self.threshold_js:
            severity = 'critical'
            drift_detected = True
        elif ks_stat >= self.warning_threshold_ks or js_div >= self.warning_threshold_js:
            severity = 'warning'
            drift_detected = True
        
        # Statistiques descriptives
        baseline_clean = baseline_values[~np.isnan(baseline_values)]
        production_clean = production_values[~np.isnan(production_values)]
        
        return {
            'feature': feature_name,
            'ks_statistic': round(float(ks_stat), 4),
            'ks_pvalue': round(float(ks_pval), 4),
            'js_divergence': round(float(js_div), 4),
            'drift_detected': drift_detected,
            'severity': severity,
            'baseline_mean': round(float(baseline_clean.mean()), 4) if len(baseline_clean) > 0 else 0.0,
            'production_mean': round(float(production_clean.mean()), 4) if len(production_clean) > 0 else 0.0,
            'baseline_std': round(float(baseline_clean.std()), 4) if len(baseline_clean) > 0 else 0.0,
            'production_std': round(float(production_clean.std()), 4) if len(production_clean) > 0 else 0.0,
            'baseline_samples': int(len(baseline_clean)),
            'production_samples': int(len(production_clean))
        }
    
    def detect_all_features(
        self,
        df_baseline: pd.DataFrame,
        df_production: pd.DataFrame
    ) -> List[Dict]:
        """Détecte le drift pour toutes les features ML"""
        logger.info("🔍 Détection drift sur 62 features ML...")
        
        results = []
        
        for i, feature in enumerate(self.ML_FEATURES, 1):
            if feature not in df_baseline.columns:
                logger.warning(f"   ⚠️  Feature manquante dans baseline: {feature}")
                continue
            
            if feature not in df_production.columns:
                logger.warning(f"   ⚠️  Feature manquante dans production: {feature}")
                continue
            
            baseline_values = df_baseline[feature].values
            production_values = df_production[feature].values
            
            drift_info = self.detect_feature_drift(
                feature, baseline_values, production_values
            )
            results.append(drift_info)
            
            # Log si drift détecté
            if drift_info['drift_detected']:
                severity_emoji = "🔴" if drift_info['severity'] == 'critical' else "🟡"
                logger.warning(
                    f"   {severity_emoji} [{i:2d}/62] {feature:<30} | "
                    f"KS={drift_info['ks_statistic']:.3f} | "
                    f"JS={drift_info['js_divergence']:.3f} | "
                    f"{drift_info['severity'].upper()}"
                )
        
        logger.info(f"   ✅ Détection terminée pour {len(results)} features")
        return results
    
    def generate_report(
        self,
        drift_results: List[Dict],
        output_path: str = "drift_report.json"
    ) -> Dict:
        """
        Génère un rapport de drift complet
        
        Returns:
            {
                'timestamp': str,
                'total_features': int,
                'features_with_drift': int,
                'critical_drifts': int,
                'warning_drifts': int,
                'drift_percentage': float,
                'features': List[Dict]
            }
        """
        total_features = len(drift_results)
        features_with_drift = sum(1 for r in drift_results if r['drift_detected'])
        critical_drifts = sum(1 for r in drift_results if r['severity'] == 'critical')
        warning_drifts = sum(1 for r in drift_results if r['severity'] == 'warning')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_features': total_features,
            'features_with_drift': features_with_drift,
            'critical_drifts': critical_drifts,
            'warning_drifts': warning_drifts,
            'drift_percentage': round(100 * features_with_drift / total_features, 2) if total_features > 0 else 0.0,
            'features': drift_results
        }
        
        # Sauvegarder JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Rapport sauvegardé: {output_path}")
        
        return report
    
    def export_prometheus_metrics(self, drift_results: List[Dict]) -> str:
        """
        Exporte les métriques au format Prometheus
        
        Returns:
            Texte au format Prometheus
        """
        lines = [
            "# HELP feature_drift_ks_statistic Kolmogorov-Smirnov statistic per feature",
            "# TYPE feature_drift_ks_statistic gauge",
        ]
        
        for result in drift_results:
            feature = result['feature']
            ks_stat = result['ks_statistic']
            lines.append(f'feature_drift_ks_statistic{{feature="{feature}"}} {ks_stat}')
        
        lines.extend([
            "",
            "# HELP feature_drift_js_divergence Jensen-Shannon divergence per feature",
            "# TYPE feature_drift_js_divergence gauge",
        ])
        
        for result in drift_results:
            feature = result['feature']
            js_div = result['js_divergence']
            lines.append(f'feature_drift_js_divergence{{feature="{feature}"}} {js_div}')
        
        lines.extend([
            "",
            "# HELP drift_alerts_total Total number of drift alerts",
            "# TYPE drift_alerts_total counter",
        ])
        
        critical_count = sum(1 for r in drift_results if r['severity'] == 'critical')
        warning_count = sum(1 for r in drift_results if r['severity'] == 'warning')
        
        lines.append(f'drift_alerts_total{{severity="critical"}} {critical_count}')
        lines.append(f'drift_alerts_total{{severity="warning"}} {warning_count}')
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Détection de drift des features ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    # Détection standard (7 derniers jours)
    python detect_drift.py --baseline data/ml_features_complete.csv
    
    # Détection avec seuils personnalisés
    python detect_drift.py --baseline data/ml_features_complete.csv --threshold-ks 0.25 --threshold-js 0.12
    
    # Détection sur 14 jours avec output custom
    python detect_drift.py --baseline data/ml_features_complete.csv --days 14 --output reports/drift_14d.json
    
    # Export métriques Prometheus
    python detect_drift.py --baseline data/ml_features_complete.csv --prometheus-output metrics/drift.prom
        """
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        default='data/ml_features_complete.csv',
        help='Chemin vers fichier CSV baseline (défaut: data/ml_features_complete.csv)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Nombre de jours de données production à analyser (défaut: 7)'
    )
    
    parser.add_argument(
        '--threshold-ks',
        type=float,
        default=0.3,
        help='Seuil critique pour KS statistic (défaut: 0.3)'
    )
    
    parser.add_argument(
        '--threshold-js',
        type=float,
        default=0.15,
        help='Seuil critique pour JS divergence (défaut: 0.15)'
    )
    
    parser.add_argument(
        '--warning-threshold-ks',
        type=float,
        default=0.2,
        help='Seuil warning pour KS statistic (défaut: 0.2)'
    )
    
    parser.add_argument(
        '--warning-threshold-js',
        type=float,
        default=0.1,
        help='Seuil warning pour JS divergence (défaut: 0.1)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='drift_report.json',
        help='Chemin du rapport JSON (défaut: drift_report.json)'
    )
    
    parser.add_argument(
        '--prometheus-output',
        type=str,
        help='Chemin pour export Prometheus (optionnel)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🔍 DÉTECTION DRIFT DES FEATURES ML")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialiser détecteur
    detector = DriftDetector(
        threshold_ks=args.threshold_ks,
        threshold_js=args.threshold_js,
        warning_threshold_ks=args.warning_threshold_ks,
        warning_threshold_js=args.warning_threshold_js
    )
    
    # Charger données
    df_baseline = detector.load_baseline(args.baseline)
    df_production = detector.load_production_data(days=args.days)
    
    if df_baseline.empty:
        logger.error("❌ Baseline vide, arrêt du script")
        return 1
    
    if df_production.empty:
        logger.error("❌ Données production vides, arrêt du script")
        return 1
    
    # Détection drift
    drift_results = detector.detect_all_features(df_baseline, df_production)
    
    # Générer rapport
    logger.info("")
    logger.info("📊 GÉNÉRATION RAPPORT...")
    report = detector.generate_report(drift_results, args.output)
    
    # Afficher résumé
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ RÉSUMÉ DÉTECTION DRIFT")
    logger.info("=" * 80)
    logger.info(f"   Features analysées      : {report['total_features']}")
    logger.info(f"   Features avec drift     : {report['features_with_drift']}")
    logger.info(f"   Drifts CRITIQUES        : {report['critical_drifts']}")
    logger.info(f"   Drifts WARNING          : {report['warning_drifts']}")
    logger.info(f"   Pourcentage drift       : {report['drift_percentage']:.1f}%")
    logger.info("")
    
    if report['critical_drifts'] > 0:
        logger.warning("🔴 ALERTE: Drift critique détecté ! Retraining recommandé.")
    elif report['warning_drifts'] > 0:
        logger.warning("🟡 WARNING: Drift modéré détecté. Surveillance accrue recommandée.")
    else:
        logger.info("✅ Aucun drift significatif détecté.")
    
    # Export Prometheus
    if args.prometheus_output:
        logger.info("")
        logger.info(f"📊 Export métriques Prometheus: {args.prometheus_output}")
        metrics_text = detector.export_prometheus_metrics(drift_results)
        
        prom_path = Path(args.prometheus_output)
        prom_path.parent.mkdir(parents=True, exist_ok=True)
        prom_path.write_text(metrics_text, encoding='utf-8')
        
        logger.info("   ✅ Métriques exportées")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 Détection drift terminée avec succès !")
    logger.info("=" * 80)
    
    # Exit code selon severity
    if report['critical_drifts'] > 0:
        return 2  # Exit code 2 = drift critique
    elif report['warning_drifts'] > 0:
        return 1  # Exit code 1 = drift warning
    else:
        return 0  # Exit code 0 = pas de drift


if __name__ == "__main__":
    sys.exit(main())
