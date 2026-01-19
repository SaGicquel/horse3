"""
🧠 AI Results Analyzer - Module d'Analyse IA des Résultats de Paris
====================================================================
Utilise Gemini (Google) ou OpenAI (GPT-4) pour:
- Analyser les patterns de succès/échec des paris
- Identifier les conditions favorables
- Suggérer des améliorations au modèle
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AIConfig:
    """Configuration pour les APIs IA."""

    provider: Literal["openai", "gemini", "auto"] = "auto"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-pro"
    temperature: float = 0.3  # Plus bas = plus précis
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "AIConfig":
        """Charge la configuration depuis les variables d'environnement."""
        return cls(
            provider=os.getenv("AI_PROVIDER", "auto"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            gemini_api_key=os.getenv("GOOGLE_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            temperature=float(os.getenv("AI_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("AI_MAX_TOKENS", "4096")),
        )


# =============================================================================
# PROVIDERS ABSTRAITS
# =============================================================================


class AIProvider(ABC):
    """Interface abstraite pour les providers IA."""

    @abstractmethod
    def analyze(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Analyse avec le prompt et le contexte donné."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Vérifie si le provider est disponible."""
        pass


class OpenAIProvider(AIProvider):
    """Provider OpenAI (GPT-4)."""

    def __init__(self, config: AIConfig):
        self.config = config
        self.client = None
        if config.openai_api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=config.openai_api_key)
            except ImportError:
                logger.warning("openai package not installed. Run: pip install openai")

    def is_available(self) -> bool:
        return self.client is not None

    def analyze(self, prompt: str, context: Dict[str, Any] = None) -> str:
        if not self.is_available():
            raise RuntimeError("OpenAI client not initialized")

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
        ]

        if context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Contexte des données:\n```json\n{json.dumps(context, indent=2, ensure_ascii=False)}\n```",
                }
            )

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    def _get_system_prompt(self) -> str:
        return """Tu es un expert en analyse de données pour les courses hippiques.
Tu analyses les résultats des paris pour identifier:
- Les patterns de succès et d'échec
- Les conditions qui favorisent les gains
- Les suggestions d'amélioration pour le modèle de prédiction

Tes réponses doivent être:
- Basées sur les données fournies
- Structurées avec des sections claires
- Actionables avec des recommandations concrètes
- En français

Utilise le format markdown pour structurer tes réponses."""


class GeminiProvider(AIProvider):
    """Provider Google Gemini."""

    def __init__(self, config: AIConfig):
        self.config = config
        self.model = None
        if config.gemini_api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=config.gemini_api_key)
                self.model = genai.GenerativeModel(config.gemini_model)
            except ImportError:
                logger.warning(
                    "google-generativeai package not installed. Run: pip install google-generativeai"
                )

    def is_available(self) -> bool:
        return self.model is not None

    def analyze(self, prompt: str, context: Dict[str, Any] = None) -> str:
        if not self.is_available():
            raise RuntimeError("Gemini model not initialized")

        full_prompt = self._get_system_prompt() + "\n\n"

        if context:
            full_prompt += f"Contexte des données:\n```json\n{json.dumps(context, indent=2, ensure_ascii=False)}\n```\n\n"

        full_prompt += f"Question/Analyse demandée:\n{prompt}"

        response = self.model.generate_content(
            full_prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            },
        )
        return response.text

    def _get_system_prompt(self) -> str:
        return """Tu es un expert en analyse de données pour les courses hippiques.
Tu analyses les résultats des paris pour identifier:
- Les patterns de succès et d'échec
- Les conditions qui favorisent les gains
- Les suggestions d'amélioration pour le modèle de prédiction

Tes réponses doivent être:
- Basées sur les données fournies
- Structurées avec des sections claires
- Actionables avec des recommandations concrètes
- En français

Utilise le format markdown pour structurer tes réponses."""


# =============================================================================
# ANALYSEUR PRINCIPAL
# =============================================================================


class AIResultsAnalyzer:
    """
    Analyseur IA des résultats de paris.

    Utilise automatiquement le meilleur provider disponible (Gemini ou OpenAI).
    """

    def __init__(self, config: AIConfig = None):
        self.config = config or AIConfig.from_env()
        self.provider = self._init_provider()

    def _init_provider(self) -> Optional[AIProvider]:
        """Initialise le provider selon la configuration."""
        if self.config.provider == "openai":
            provider = OpenAIProvider(self.config)
            if provider.is_available():
                logger.info("Using OpenAI provider")
                return provider

        if self.config.provider == "gemini":
            provider = GeminiProvider(self.config)
            if provider.is_available():
                logger.info("Using Gemini provider")
                return provider

        # Auto: essayer Gemini d'abord (moins cher), puis OpenAI
        if self.config.provider == "auto":
            gemini = GeminiProvider(self.config)
            if gemini.is_available():
                logger.info("Auto-selected Gemini provider")
                return gemini

            openai = OpenAIProvider(self.config)
            if openai.is_available():
                logger.info("Auto-selected OpenAI provider")
                return openai

        logger.warning("No AI provider available. Set OPENAI_API_KEY or GOOGLE_API_KEY.")
        return None

    def is_available(self) -> bool:
        """Vérifie si un provider IA est disponible."""
        return self.provider is not None and self.provider.is_available()

    def analyze_bets_performance(self, bets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse la performance d'une liste de paris.

        Args:
            bets: Liste de paris avec leurs résultats

        Returns:
            Analyse IA structurée
        """
        if not self.is_available():
            return {"error": "No AI provider available", "analysis": None}

        # Calculer les stats de base
        stats = self._compute_basic_stats(bets)

        prompt = f"""Analyse ces résultats de paris hippiques et fournis:

1. **Résumé de Performance** - ROI, taux de réussite, tendances
2. **Patterns Identifiés** - Conditions favorables/défavorables
3. **Points d'Amélioration** - Suggestions concrètes
4. **Score de Confiance** - Évaluation de la qualité des prédictions

Statistiques calculées: {json.dumps(stats, indent=2)}
"""

        try:
            analysis = self.provider.analyze(prompt, {"bets": bets[:50], "stats": stats})
            return {
                "success": True,
                "provider": type(self.provider).__name__,
                "stats": stats,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return {"error": str(e), "stats": stats, "analysis": None}

    def suggest_model_improvements(
        self,
        model_metrics: Dict[str, float],
        recent_predictions: List[Dict],
        feature_importances: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Suggère des améliorations pour le modèle de prédiction.

        Args:
            model_metrics: Métriques du modèle (AUC, ROI, etc.)
            recent_predictions: Prédictions récentes avec résultats
            feature_importances: Importance des features
        """
        if not self.is_available():
            return {"error": "No AI provider available"}

        context = {
            "model_metrics": model_metrics,
            "predictions_sample": recent_predictions[:30],
            "feature_importances": feature_importances or {},
        }

        prompt = """Analyse ce modèle de prédiction hippique et suggère des améliorations:

1. **Évaluation des Métriques** - Points forts et faiblesses
2. **Features Importantes** - Analyse des features les plus prédictives
3. **Suggestions de Nouvelles Features** - Données à ajouter
4. **Optimisation des Hyperparamètres** - Pistes à explorer
5. **Stratégie de Betting** - Ajustements recommandés
"""

        try:
            analysis = self.provider.analyze(prompt, context)
            return {
                "success": True,
                "provider": type(self.provider).__name__,
                "suggestions": analysis,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"AI suggestion error: {e}")
            return {"error": str(e)}

    def analyze_losing_streak(self, losing_bets: List[Dict]) -> Dict[str, Any]:
        """
        Analyse spécifique d'une série de paris perdants.
        """
        if not self.is_available():
            return {"error": "No AI provider available"}

        prompt = """Ces paris ont tous été perdants. Analyse pourquoi et suggère des ajustements:

1. **Points Communs** - Qu'ont ces paris en commun?
2. **Erreurs Potentielles** - Où le modèle s'est-il trompé?
3. **Signaux d'Alerte** - À quoi faire attention à l'avenir?
4. **Recommandations** - Comment éviter ces pertes?
"""

        try:
            analysis = self.provider.analyze(prompt, {"losing_bets": losing_bets})
            return {"success": True, "analysis": analysis, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            return {"error": str(e)}

    def _compute_basic_stats(self, bets: List[Dict]) -> Dict[str, Any]:
        """Calcule les statistiques de base sur les paris."""
        if not bets:
            return {}

        total = len(bets)
        wins = sum(1 for b in bets if b.get("result") == "WIN" or b.get("won", False))
        total_stake = sum(b.get("stake", 0) for b in bets)
        total_returns = sum(b.get("returns", 0) for b in bets)

        return {
            "total_bets": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / total * 100, 2) if total > 0 else 0,
            "total_stake": round(total_stake, 2),
            "total_returns": round(total_returns, 2),
            "profit": round(total_returns - total_stake, 2),
            "roi": round((total_returns - total_stake) / total_stake * 100, 2)
            if total_stake > 0
            else 0,
            "avg_stake": round(total_stake / total, 2) if total > 0 else 0,
            "avg_odds": round(sum(b.get("odds", 0) for b in bets) / total, 2) if total > 0 else 0,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_analyzer() -> AIResultsAnalyzer:
    """Factory pour obtenir un analyseur configuré."""
    return AIResultsAnalyzer()


def quick_analyze(bets: List[Dict]) -> str:
    """Analyse rapide de paris pour usage en ligne de commande."""
    analyzer = get_analyzer()
    if not analyzer.is_available():
        return "❌ Aucune API IA configurée. Définissez OPENAI_API_KEY ou GOOGLE_API_KEY."

    result = analyzer.analyze_bets_performance(bets)
    if result.get("error"):
        return f"❌ Erreur: {result['error']}"

    return result.get("analysis", "Pas d'analyse disponible")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Results Analyzer")
    parser.add_argument("--test", action="store_true", help="Test avec données fictives")
    parser.add_argument("--check", action="store_true", help="Vérifier la configuration")
    args = parser.parse_args()

    analyzer = get_analyzer()

    if args.check:
        print("🔍 Vérification de la configuration IA...")
        print(
            f"   OPENAI_API_KEY: {'✅ Défini' if os.getenv('OPENAI_API_KEY') else '❌ Non défini'}"
        )
        print(
            f"   GOOGLE_API_KEY: {'✅ Défini' if os.getenv('GOOGLE_API_KEY') else '❌ Non défini'}"
        )
        print(
            f"   Provider actif: {type(analyzer.provider).__name__ if analyzer.provider else 'Aucun'}"
        )
        print(f"   Disponible: {'✅' if analyzer.is_available() else '❌'}")

    elif args.test:
        print("🧪 Test avec données fictives...")
        test_bets = [
            {"name": "Cheval A", "stake": 10, "odds": 3.5, "result": "WIN", "returns": 35},
            {"name": "Cheval B", "stake": 15, "odds": 2.1, "result": "LOSE", "returns": 0},
            {"name": "Cheval C", "stake": 20, "odds": 4.0, "result": "WIN", "returns": 80},
            {"name": "Cheval D", "stake": 10, "odds": 5.5, "result": "LOSE", "returns": 0},
            {"name": "Cheval E", "stake": 25, "odds": 1.8, "result": "WIN", "returns": 45},
        ]

        if analyzer.is_available():
            result = analyzer.analyze_bets_performance(test_bets)
            print("\n📊 Résultat de l'analyse:")
            print(result.get("analysis", "Pas d'analyse"))
        else:
            print("❌ Aucune API IA configurée")

    else:
        parser.print_help()
