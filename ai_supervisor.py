"""
AI Supervisor - Race Analysis & Anomaly Detection
==================================================
Step A: Generate structured prompts with ML predictions + race stats
Step B: Analyze with LLM to detect anomalies and validate predictions
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RaceContext:
    course_id: str
    date: str
    hippodrome: str
    distance: int
    discipline: str
    nombre_partants: int
    conditions: Optional[str] = None
    meteo: Optional[str] = None
    terrain: Optional[str] = None


@dataclass
class HorseAnalysis:
    cheval_id: str
    nom: str
    numero: int
    cote_sp: float
    prob_model: float
    rang_model: int
    forme_5c: float = 0.0
    forme_10c: float = 0.0
    nb_courses_12m: int = 0
    nb_victoires_12m: int = 0
    taux_victoires_jockey: float = 0.0
    musique: Optional[str] = None
    derniere_course_jours: Optional[int] = None

    @property
    def value_edge(self) -> float:
        if self.cote_sp > 0:
            implied_prob = 1.0 / self.cote_sp
            return self.prob_model - implied_prob
        return 0.0


@dataclass
class SupervisorResult:
    course_id: str
    timestamp: str
    analysis: str
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    provider: Optional[str] = None


class AiSupervisor:
    def __init__(self, provider=None):
        self.provider = provider or self._init_provider()

    def _init_provider(self):
        try:
            from ai_results_analyzer import AIConfig, OpenAIProvider, GeminiProvider

            config = AIConfig.from_env()

            if config.provider == "openai" or config.provider == "auto":
                provider = OpenAIProvider(config)
                if provider.is_available():
                    logger.info("AiSupervisor: Using OpenAI")
                    return provider

            if config.provider == "gemini" or config.provider == "auto":
                provider = GeminiProvider(config)
                if provider.is_available():
                    logger.info("AiSupervisor: Using Gemini")
                    return provider

            logger.warning("AiSupervisor: No AI provider available")
            return None
        except ImportError as e:
            logger.warning(f"AiSupervisor: Could not import providers: {e}")
            return None

    def generate_prompt(
        self,
        race: RaceContext,
        horses: List[HorseAnalysis],
        feature_importances: Optional[Dict[str, float]] = None,
    ) -> str:
        prompt_parts = [
            "## Analyse de Course - Validation des Prédictions ML",
            "",
            f"**Course**: {race.course_id}",
            f"**Date**: {race.date}",
            f"**Hippodrome**: {race.hippodrome}",
            f"**Distance**: {race.distance}m | **Discipline**: {race.discipline}",
            f"**Partants**: {race.nombre_partants}",
        ]

        if race.terrain:
            prompt_parts.append(f"**Terrain**: {race.terrain}")
        if race.meteo:
            prompt_parts.append(f"**Météo**: {race.meteo}")

        prompt_parts.extend(["", "### Prédictions du Modèle ML", ""])
        prompt_parts.append("| # | Cheval | Cote | Prob ML | Rang | Forme 5c | V/12m | Edge |")
        prompt_parts.append("|---|--------|------|---------|------|----------|-------|------|")

        sorted_horses = sorted(horses, key=lambda h: h.rang_model)
        for h in sorted_horses:
            edge_pct = h.value_edge * 100
            edge_str = f"+{edge_pct:.1f}%" if edge_pct > 0 else f"{edge_pct:.1f}%"
            prompt_parts.append(
                f"| {h.numero} | {h.nom[:15]} | {h.cote_sp:.1f} | {h.prob_model:.1%} | "
                f"{h.rang_model} | {h.forme_5c:.0%} | {h.nb_victoires_12m} | {edge_str} |"
            )

        if feature_importances:
            prompt_parts.extend(["", "### Features Importantes (Top 10)", ""])
            sorted_fi = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in sorted_fi:
                prompt_parts.append(f"- **{feat}**: {imp:.3f}")

        prompt_parts.extend(
            [
                "",
                "### Tâches d'Analyse",
                "",
                "1. **Détection d'Anomalies**: Identifier les incohérences entre:",
                "   - La forme récente et le classement du modèle",
                "   - La cote marché et la probabilité ML",
                "   - Les statistiques historiques et la performance attendue",
                "",
                "2. **Validation des Top 3**: Pour chaque favori du modèle:",
                "   - Points forts justifiant la sélection",
                "   - Risques ou signaux d'alerte",
                "",
                "3. **Recommandations**:",
                "   - Niveau de confiance (Haute/Moyenne/Faible)",
                "   - Chevaux à surveiller (value potentielle)",
                "   - Mises en garde spécifiques",
            ]
        )

        return "\n".join(prompt_parts)

    def detect_rule_based_anomalies(self, horses: List[HorseAnalysis]) -> List[Dict[str, Any]]:
        anomalies = []

        for h in horses:
            if h.rang_model <= 3 and h.forme_5c < 0.3:
                anomalies.append(
                    {
                        "type": "FORME_FAIBLE_FAVORI",
                        "cheval": h.nom,
                        "numero": h.numero,
                        "detail": f"Favori modèle (rang {h.rang_model}) avec forme faible ({h.forme_5c:.0%})",
                        "severity": "HIGH",
                    }
                )

            edge = h.value_edge
            if edge > 0.15:
                anomalies.append(
                    {
                        "type": "VALUE_EXTREME",
                        "cheval": h.nom,
                        "numero": h.numero,
                        "detail": f"Edge très élevé ({edge:.1%}) - vérifier données",
                        "severity": "MEDIUM",
                    }
                )

            if h.rang_model <= 3 and h.nb_victoires_12m == 0 and h.nb_courses_12m >= 5:
                anomalies.append(
                    {
                        "type": "FAVORI_SANS_VICTOIRE",
                        "cheval": h.nom,
                        "numero": h.numero,
                        "detail": f"Favori (rang {h.rang_model}) sans victoire sur {h.nb_courses_12m} courses",
                        "severity": "MEDIUM",
                    }
                )

            if h.derniere_course_jours and h.derniere_course_jours > 90 and h.rang_model <= 5:
                anomalies.append(
                    {
                        "type": "LONGUE_ABSENCE",
                        "cheval": h.nom,
                        "numero": h.numero,
                        "detail": f"Absence de {h.derniere_course_jours} jours pour un favori",
                        "severity": "MEDIUM",
                    }
                )

        return anomalies

    def _extract_and_verify_claims(
        self, analysis: str, horses: List[HorseAnalysis]
    ) -> Dict[str, Any]:
        """
        Step C: Vérification factuelle (Cross-check).
        Extrait les affirmations du LLM et les compare aux données structurées.
        """
        report = {"checked": 0, "hallucinations": 0, "details": []}

        # Mapping nom -> objet cheval
        horse_map = {}
        for h in horses:
            horse_map[h.nom.lower()] = h
            # Partials match support (e.g. "Bold" for "Bold Eagle")
            parts = h.nom.lower().split()
            if len(parts) > 1:
                horse_map[parts[0]] = h

        # Mapping numero -> objet cheval
        number_map = {h.numero: h for h in horses}

        lines = analysis.lower().split("\n")

        # Regex patterns
        # "cote de 5.2", "cote: 5.2", "cote est de 5.2"
        odds_pattern = re.compile(r"cote\s*(?:est)?\s*(?:de|:)?\s*(\d+[.,]?\d*)")

        # "cheval n°5", "#5", "numero 5"
        number_pattern = re.compile(r"(?:#|n°|numéro|numero)\s*(\d+)")

        for line in lines:
            # Identifier le cheval mentionné dans la ligne
            mentioned_horse = None

            # 1. Recherche par numéro explicite (#1, n°1)
            num_match = number_pattern.search(line)
            if num_match:
                try:
                    num = int(num_match.group(1))
                    if num in number_map:
                        mentioned_horse = number_map[num]
                except ValueError:
                    pass

            # 2. Recherche par nom (si pas trouvé par numéro)
            if not mentioned_horse:
                for name, h in horse_map.items():
                    # On cherche le nom entouré d'espaces ou de ponctuation
                    if re.search(r"\b" + re.escape(name) + r"\b", line):
                        mentioned_horse = h
                        break

            if mentioned_horse:
                # Vérifier les cotes
                match = odds_pattern.search(line)
                if match:
                    try:
                        claimed_odds = float(match.group(1).replace(",", "."))
                        real_odds = mentioned_horse.cote_sp

                        report["checked"] += 1

                        # Tolérance de 10% ou 0.5 point
                        if abs(claimed_odds - real_odds) > max(0.5, real_odds * 0.1):
                            report["hallucinations"] += 1
                            report["details"].append(
                                {
                                    "type": "ODDS_MISMATCH",
                                    "claim": f"Cote {claimed_odds}",
                                    "fact": f"Cote {real_odds}",
                                    "horse": mentioned_horse.nom,
                                }
                            )
                    except ValueError:
                        pass

                # Vérifier si le LLM dit qu'il est "favori" alors qu'il ne l'est pas
                if (
                    "favori" in line
                    and mentioned_horse.rang_model > 3
                    and mentioned_horse.cote_sp > 10
                ):
                    # On vérifie que la phrase ne dit pas "n'est pas favori"
                    if "pas favori" not in line and "outsider" not in line:
                        report["checked"] += 1
                        report["hallucinations"] += 1
                        report["details"].append(
                            {
                                "type": "FALSE_FAVORITE",
                                "claim": "Identifié comme favori",
                                "fact": f"Rang {mentioned_horse.rang_model}, Cote {mentioned_horse.cote_sp}",
                                "horse": mentioned_horse.nom,
                            }
                        )

        return report

    def _analyze_sentiment(self, analysis: str) -> float:
        """
        Analyse simpliste du sentiment du rapport LLM.
        Retourne un score entre -1.0 (Négatif) et 1.0 (Positif).
        """
        keywords_pos = ["confiant", "solide", "excellent", "valeur", "surclassé", "base"]
        keywords_neg = ["risque", "attention", "méfiance", "doute", "incertain", "faible"]

        score = 0.0
        lower_analysis = analysis.lower()

        for kw in keywords_pos:
            score += lower_analysis.count(kw) * 0.1
        for kw in keywords_neg:
            score -= lower_analysis.count(kw) * 0.15

        return max(-1.0, min(1.0, score))

    def _calculate_decision_score(
        self,
        ml_confidence: float,
        sentiment_score: float,
        anomalies: List[Dict[str, Any]],
        verification_report: Dict[str, Any],
    ) -> float:
        """
        Step D: Calcul du score final de décision.
        Aggrège ML, Sentiment LLM et Pénalités.
        """
        # Poids
        W_ML = 0.6
        W_SENTIMENT = 0.4

        # Base score (ML + Sentiment)
        # Sentiment (-1 à 1) est normalisé vers (0.5 à 1.5) pour moduler le ML
        sentiment_modulator = 1.0 + (sentiment_score * 0.2)
        score = ml_confidence * sentiment_modulator

        # Pénalités Anomalies
        high_severity = sum(1 for a in anomalies if a.get("severity") == "HIGH")
        medium_severity = sum(1 for a in anomalies if a.get("severity") == "MEDIUM")

        penalty = (high_severity * 0.15) + (medium_severity * 0.05)

        # Pénalité Hallucinations (Step C)
        hallucinations = verification_report.get("hallucinations", 0)
        penalty += hallucinations * 0.2

        final_score = max(0.0, min(1.0, score - penalty))
        return final_score

    def analyze(
        self,
        race: RaceContext,
        horses: List[HorseAnalysis],
        feature_importances: Optional[Dict[str, float]] = None,
    ) -> SupervisorResult:
        prompt = self.generate_prompt(race, horses, feature_importances)
        rule_anomalies = self.detect_rule_based_anomalies(horses)

        if self.provider and self.provider.is_available():
            try:
                context = {
                    "course": {
                        "id": race.course_id,
                        "hippodrome": race.hippodrome,
                        "distance": race.distance,
                        "discipline": race.discipline,
                    },
                    "predictions": [
                        {
                            "numero": h.numero,
                            "nom": h.nom,
                            "cote": h.cote_sp,
                            "prob_ml": round(h.prob_model, 4),
                            "rang": h.rang_model,
                            "forme_5c": h.forme_5c,
                            "edge": round(h.value_edge, 4),
                        }
                        for h in sorted(horses, key=lambda x: x.rang_model)
                    ],
                    "anomalies_detectees": rule_anomalies,
                }

                llm_analysis = self.provider.analyze(prompt, context)
                provider_name = type(self.provider).__name__

                recommendations = self._extract_recommendations(llm_analysis)

                # Step C: Vérification
                verification_report = self._extract_and_verify_claims(llm_analysis, horses)

                # Step D: Décision
                sentiment = self._analyze_sentiment(llm_analysis)
                base_ml_conf = self._calculate_confidence(
                    horses, []
                )  # ML confidence pure sans pénalités anomalies

                final_confidence = self._calculate_decision_score(
                    base_ml_conf, sentiment, rule_anomalies, verification_report
                )

                return SupervisorResult(
                    course_id=race.course_id,
                    timestamp=datetime.now().isoformat(),
                    analysis=llm_analysis,
                    anomalies=rule_anomalies,
                    recommendations=recommendations,
                    confidence_score=final_confidence,
                    provider=provider_name,
                )
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")

        fallback_analysis = self._generate_fallback_analysis(race, horses, rule_anomalies)
        confidence = self._calculate_confidence(horses, rule_anomalies)

        return SupervisorResult(
            course_id=race.course_id,
            timestamp=datetime.now().isoformat(),
            analysis=fallback_analysis,
            anomalies=rule_anomalies,
            recommendations=[],
            confidence_score=confidence,
            provider="rule_based",
        )

    def _extract_recommendations(self, analysis: str) -> List[str]:
        recommendations = []
        lines = analysis.split("\n")
        in_reco_section = False

        for line in lines:
            lower = line.lower()
            if "recommandation" in lower or "conseil" in lower:
                in_reco_section = True
                continue
            if in_reco_section and line.strip().startswith(("-", "*", "•")):
                recommendations.append(line.strip().lstrip("-*• "))
            if in_reco_section and line.strip().startswith("#"):
                in_reco_section = False

        return recommendations[:5]

    def _calculate_confidence(
        self, horses: List[HorseAnalysis], anomalies: List[Dict[str, Any]]
    ) -> float:
        base_confidence = 0.7

        high_severity = sum(1 for a in anomalies if a.get("severity") == "HIGH")
        medium_severity = sum(1 for a in anomalies if a.get("severity") == "MEDIUM")

        penalty = (high_severity * 0.15) + (medium_severity * 0.05)

        if horses:
            top3 = [h for h in horses if h.rang_model <= 3]
            avg_forme = sum(h.forme_5c for h in top3) / len(top3) if top3 else 0
            if avg_forme > 0.5:
                base_confidence += 0.1

        return max(0.1, min(1.0, base_confidence - penalty))

    def _generate_fallback_analysis(
        self, race: RaceContext, horses: List[HorseAnalysis], anomalies: List[Dict[str, Any]]
    ) -> str:
        lines = [
            f"## Analyse Course {race.course_id}",
            f"*{race.hippodrome} - {race.distance}m {race.discipline}*",
            "",
            "### Top 3 Prédictions ML",
        ]

        top3 = sorted(horses, key=lambda h: h.rang_model)[:3]
        for h in top3:
            edge_str = f"+{h.value_edge:.1%}" if h.value_edge > 0 else f"{h.value_edge:.1%}"
            lines.append(
                f"- **#{h.numero} {h.nom}**: Prob {h.prob_model:.1%}, Cote {h.cote_sp:.1f}, Edge {edge_str}"
            )

        if anomalies:
            lines.extend(["", "### ⚠️ Anomalies Détectées"])
            for a in anomalies:
                severity_icon = "🔴" if a["severity"] == "HIGH" else "🟡"
                lines.append(f"- {severity_icon} **{a['type']}**: {a['detail']}")

        lines.extend(["", "*Analyse rule-based (LLM non disponible)*"])

        return "\n".join(lines)
