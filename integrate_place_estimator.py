#!/usr/bin/env python3
"""
Intégration du module d'estimation p(place) avec le système de paris exotiques
==============================================================================

Ce script intègre le nouveau PlaceProbabilityEstimator avec:
- exotic_ticket_generator.py
- race_pronostic_generator.py
- pro_betting_analyzer.py

Usage:
    # Via CLI
    python integrate_place_estimator.py --race-key "2025-12-06|R1|C1|PAR"
    
    # Via import
    from integrate_place_estimator import generate_exotic_analysis
    result = generate_exotic_analysis(p_win_blend, discipline='plat')

Auteur: Horse Racing AI System
Date: 2024-12
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import argparse
from datetime import datetime

# Imports locaux
try:
    from place_probability_estimator import (
        PlaceProbabilityEstimator,
        PlaceEstimatorConfig,
        PlaceEstimatorSelector,
        CalibrationMetrics,
        EVStabilityValidator,
        TemperatureLearner,
        estimate_exotic_probs,
        compare_estimators_on_race
    )
except ImportError:
    print("⚠️  Module place_probability_estimator.py non trouvé")
    raise

try:
    from exotic_ticket_generator import ExoticTicketGenerator, ExoticConfig
except ImportError:
    ExoticTicketGenerator = None
    print("⚠️  Module exotic_ticket_generator.py non disponible")

try:
    from pro_betting_analyzer import ProBettingAnalyzer, ProbabilityNormalizer
except ImportError:
    ProBettingAnalyzer = None
    print("⚠️  Module pro_betting_analyzer.py non disponible")

try:
    from config.loader import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


# =============================================================================
# FONCTIONS D'INTÉGRATION
# =============================================================================

def load_place_estimator_config() -> PlaceEstimatorConfig:
    """Charge la configuration depuis pro_betting.yaml."""
    config = PlaceEstimatorConfig()
    
    if CONFIG_AVAILABLE:
        try:
            cfg = get_config()
            config.num_simulations = cfg.simulation.num_simulations
            config.takeout_rate = cfg.markets.takeout_rate
            
            # Charger les paramètres place_estimators si disponibles
            # (via lecture directe du YAML car pas encore dans le loader)
            import yaml
            yaml_path = Path(__file__).parent / "config" / "pro_betting.yaml"
            if yaml_path.exists():
                with open(yaml_path) as f:
                    raw = yaml.safe_load(f)
                
                pe = raw.get('place_estimators', {})
                config.temperature_default = pe.get('temperature_default', 1.0)
                config.temperature_plat = pe.get('temperature_plat', 0.95)
                config.temperature_trot = pe.get('temperature_trot', 1.05)
                config.temperature_obstacle = pe.get('temperature_obstacle', 1.10)
                config.henery_gamma = pe.get('henery_gamma', 0.81)
                config.stern_lambda = pe.get('stern_lambda', 0.15)
                config.lbs_iterations = pe.get('lbs_iterations', 100)
                
        except Exception as e:
            print(f"⚠️  Erreur chargement config: {e}")
    
    return config


def generate_exotic_analysis(
    p_win_blend: np.ndarray,
    discipline: str = 'plat',
    structure: str = 'trio',
    horse_names: List[str] = None,
    fixed_odds: Dict[Tuple, float] = None,
    include_comparison: bool = True,
    include_stability: bool = True
) -> Dict[str, Any]:
    """
    Génère une analyse complète pour paris exotiques.
    
    Args:
        p_win_blend: Probabilités de victoire (blend modèle/marché)
        discipline: 'plat', 'trot', 'obstacle'
        structure: 'exacta', 'trio', 'quarte', 'quinte'
        horse_names: Noms des chevaux
        fixed_odds: Cotes fixes si disponibles
        include_comparison: Inclure comparaison des estimateurs
        include_stability: Inclure test de stabilité EV
    
    Returns:
        Dict complet avec p_place, combo_probs, EV, validation
    """
    # Normaliser
    p_win = np.array(p_win_blend)
    if not np.isclose(p_win.sum(), 1.0, atol=0.01):
        p_win = p_win / p_win.sum()
    
    n_horses = len(p_win)
    if horse_names is None:
        horse_names = [f"#{i+1}" for i in range(n_horses)]
    
    # Charger config
    config = load_place_estimator_config()
    
    # Créer l'estimateur principal
    estimator = PlaceProbabilityEstimator(
        p_win,
        discipline=discipline,
        config=config,
        horse_names=horse_names
    )
    
    result = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'discipline': discipline,
            'structure': structure,
            'n_horses': n_horses,
            'n_simulations': config.num_simulations,
            'takeout_rate': config.takeout_rate
        },
        'input': {
            'p_win_blend': {
                horse_names[i]: round(float(p_win[i]), 4)
                for i in range(n_horses)
            }
        }
    }
    
    # 1. Probabilités de place
    print("📊 Calcul des probabilités de place...")
    place_data = estimator.estimate_place_probs(top_n=3, method='auto')
    result['p_place'] = {
        horse_names[i]: round(float(place_data['p_place'][i]), 4)
        for i in range(n_horses)
    }
    result['place_details'] = place_data['by_horse']
    
    # 2. Probabilités de combinaisons
    print(f"🎲 Simulation de {config.num_simulations} arrivées ({structure})...")
    combo_data = estimator.estimate_combo_probs(structure)
    result['combo_stats'] = {
        'n_combos_observed': combo_data['n_combos'],
        'coverage': round(combo_data['coverage'], 4)
    }
    result['top_combos'] = combo_data['top_combos'][:30]
    
    # 3. EV des tickets
    print("💰 Calcul des Expected Values...")
    ev_data = estimator.calculate_tickets_ev(
        combo_data['combo_probs'], 
        structure,
        fixed_odds
    )
    result['ev_analysis'] = {
        'n_tickets': ev_data['n_tickets'],
        'n_positive_ev': ev_data['n_positive_ev'],
        'positive_ev_rate': round(ev_data['n_positive_ev'] / ev_data['n_tickets'] * 100, 2)
    }
    result['best_tickets'] = ev_data['positive_ev_tickets'][:20]
    
    # 4. Comparaison des estimateurs (optionnel)
    if include_comparison:
        print("🔬 Comparaison des estimateurs...")
        comparison = estimator.compare_estimators()
        result['estimator_comparison'] = comparison
    
    # 5. Validation stabilité (optionnel)
    if include_stability:
        print("✅ Validation de la stabilité EV...")
        validator = EVStabilityValidator(p_win, config)
        stability = validator.check_stability([5000, 10000, 20000], structure)
        result['stability'] = {
            'recommended_n': stability['recommended_n'],
            'is_stable': stability['is_stable']
        }
    
    # 6. Packs de tickets recommandés
    print("📦 Génération des packs de tickets...")
    result['packs'] = generate_ticket_packs(
        ev_data['positive_ev_tickets'],
        config
    )
    
    return result


def generate_ticket_packs(positive_ev_tickets: List[Dict], 
                          config: PlaceEstimatorConfig) -> Dict[str, Any]:
    """
    Génère des packs de tickets (SÛR, ÉQUILIBRÉ, RISQUÉ).
    
    Args:
        positive_ev_tickets: Tickets avec EV positif
        config: Configuration
    
    Returns:
        Dict avec les 3 packs
    """
    if not positive_ev_tickets:
        return {'error': 'Aucun ticket à EV positif'}
    
    # Classifier par profil de risque
    sure_tickets = []
    balanced_tickets = []
    risky_tickets = []
    
    for ticket in positive_ev_tickets:
        prob = ticket['prob']
        ev = ticket['ev']
        
        # Critères de classification
        if prob >= 0.003 and ev >= 0.05:  # ≥0.3% prob, ≥5% EV
            sure_tickets.append(ticket)
        elif prob >= 0.001 and ev >= 0.10:  # ≥0.1% prob, ≥10% EV
            balanced_tickets.append(ticket)
        elif ev >= 0.15:  # ≥15% EV (peu importe prob)
            risky_tickets.append(ticket)
    
    def make_pack(tickets: List[Dict], label: str, max_tickets: int = 10) -> Dict:
        if not tickets:
            return {'label': label, 'tickets': [], 'total_prob': 0, 'expected_ev': 0}
        
        selected = tickets[:max_tickets]
        total_prob = sum(t['prob'] for t in selected)
        avg_ev = np.mean([t['ev'] for t in selected])
        
        return {
            'label': label,
            'n_tickets': len(selected),
            'tickets': selected,
            'total_prob_pct': round(total_prob * 100, 2),
            'expected_ev_pct': round(avg_ev * 100, 2),
            'risk_profile': 'LOW' if label == 'SÛR' else ('MEDIUM' if label == 'ÉQUILIBRÉ' else 'HIGH')
        }
    
    return {
        'SÛR': make_pack(sure_tickets, 'SÛR', 5),
        'ÉQUILIBRÉ': make_pack(balanced_tickets, 'ÉQUILIBRÉ', 8),
        'RISQUÉ': make_pack(risky_tickets, 'RISQUÉ', 10)
    }


def integrate_with_exotic_generator(
    p_win_blend: np.ndarray,
    discipline: str = 'plat',
    budget: float = 100.0,
    structure: str = 'trio_ordre',
    horse_names: List[str] = None
) -> Dict[str, Any]:
    """
    Intègre le nouvel estimateur avec ExoticTicketGenerator existant.
    
    Utilise les p_place et combo_probs améliorés comme input.
    """
    if ExoticTicketGenerator is None:
        return {'error': 'ExoticTicketGenerator non disponible'}
    
    config = load_place_estimator_config()
    
    # Utiliser le nouvel estimateur pour les probas
    estimator = PlaceProbabilityEstimator(
        p_win_blend,
        discipline=discipline,
        config=config,
        horse_names=horse_names
    )
    
    # Obtenir les probabilités améliorées
    place_data = estimator.estimate_place_probs(top_n=3)
    
    # Utiliser le générateur existant avec nos probas
    exotic_config = ExoticConfig(
        num_simulations=config.num_simulations,
        budget=budget,
        structure=structure,
        takeout_rate=config.takeout_rate
    )
    
    generator = ExoticTicketGenerator(exotic_config)
    
    # Générer les tickets avec le générateur existant
    result = generator.generate(
        probabilities=list(p_win_blend),
        horse_names=horse_names,
        structure=structure,
        budget=budget
    )
    
    # Enrichir avec nos analyses
    result['enhanced_p_place'] = {
        name: round(float(place_data['p_place'][i]), 4)
        for i, name in enumerate(horse_names or [f"#{i}" for i in range(len(p_win_blend))])
    }
    result['estimator_used'] = type(estimator.estimator).__name__
    result['temperature'] = estimator.simulator.temperature
    
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Estimation de p(place) et probabilités d'ordre pour exotiques"
    )
    parser.add_argument(
        '--race-key', '-r',
        help="Clé de course (format: YYYY-MM-DD|Rx|Cx|HIP)"
    )
    parser.add_argument(
        '--discipline', '-d',
        default='plat',
        choices=['plat', 'trot', 'obstacle'],
        help="Discipline"
    )
    parser.add_argument(
        '--structure', '-s',
        default='trio',
        choices=['exacta', 'trio', 'quarte', 'quinte'],
        help="Type de pari"
    )
    parser.add_argument(
        '--simulations', '-n',
        type=int,
        default=20000,
        help="Nombre de simulations"
    )
    parser.add_argument(
        '--output', '-o',
        help="Fichier de sortie JSON"
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help="Mode démonstration avec données exemple"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Données de démonstration
        print("\n" + "=" * 70)
        print("🏇 DÉMONSTRATION - Estimation p(place) et Exotiques")
        print("=" * 70)
        
        p_win = np.array([0.22, 0.18, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02])
        horse_names = [
            "GOLDEN STAR", "SILVER BOLT", "BRONZE FLASH", "IRON SPIRIT",
            "COPPER DREAM", "STEEL HEART", "CHROME FIRE", "PLATINUM RUN",
            "DIAMOND DUST", "RUBY LIGHT"
        ]
        
        result = generate_exotic_analysis(
            p_win,
            discipline=args.discipline,
            structure=args.structure,
            horse_names=horse_names,
            include_comparison=True,
            include_stability=True
        )
        
        # Afficher les résultats
        print("\n📊 PROBABILITÉS DE PLACE (Top 3)")
        print("-" * 40)
        for name, prob in result['p_place'].items():
            bar = "█" * int(prob * 50)
            print(f"  {name:15s}: {prob*100:5.1f}% {bar}")
        
        print(f"\n🎲 COMBINAISONS ({args.structure.upper()})")
        print("-" * 40)
        print(f"  Combos observés: {result['combo_stats']['n_combos_observed']}")
        for i, combo in enumerate(result['top_combos'][:10]):
            names = " - ".join(combo['combo'])
            print(f"  {i+1:2d}. {names}: {combo['prob_pct']:.2f}%")
        
        print(f"\n💰 EXPECTED VALUES")
        print("-" * 40)
        print(f"  Tickets EV+: {result['ev_analysis']['n_positive_ev']} / {result['ev_analysis']['n_tickets']}")
        print(f"  Taux EV+: {result['ev_analysis']['positive_ev_rate']:.1f}%")
        
        if result['best_tickets']:
            print("\n  Top 5 tickets:")
            for t in result['best_tickets'][:5]:
                names = " - ".join(t['combo'])
                print(f"    {names}: EV={t['ev_pct']:+.1f}%")
        
        print(f"\n📦 PACKS DE TICKETS")
        print("-" * 40)
        for pack_name, pack in result['packs'].items():
            if isinstance(pack, dict) and 'n_tickets' in pack:
                print(f"  {pack_name}: {pack['n_tickets']} tickets, "
                      f"prob={pack['total_prob_pct']:.1f}%, EV={pack['expected_ev_pct']:+.1f}%")
        
        if result.get('stability'):
            print(f"\n✅ VALIDATION")
            print("-" * 40)
            print(f"  N recommandé: {result['stability']['recommended_n']}")
            print(f"  Stable: {'Oui ✓' if result['stability']['is_stable'] else 'Non ✗'}")
        
        # Sauvegarder si demandé
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n📁 Résultats sauvegardés: {args.output}")
        
    else:
        print("Utilisez --demo pour une démonstration, ou --race-key pour analyser une course réelle")


if __name__ == "__main__":
    main()
