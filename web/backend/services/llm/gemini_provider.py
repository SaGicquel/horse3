"""
🤖 Gemini Provider - Structured Outputs Implementation
======================================================

Implémentation du provider Google Gemini avec support des sorties structurées.
Utilise response_mime_type="application/json" + response_schema.

Auteur: Agent IA Pipeline
Date: 2024-12-21
"""

from __future__ import annotations

import json
import time
import logging
from datetime import datetime
from typing import Type, TypeVar, Any, Optional

from pydantic import BaseModel, ValidationError

from .base import (
    LLMProvider,
    LLMProviderType,
    LLMConfig,
    LLMTrace,
    LLMCallStatus,
    LLMValidationError,
    LLMBudgetExceededError,
    LLMTimeoutError,
    LLMAPIError,
)

# Logger
logger = logging.getLogger("agent_ia.llm.gemini")

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# PRICING (USD per 1M tokens) - Décembre 2024
# =============================================================================

GEMINI_PRICING = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Preview gratuit
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estime le coût d'un appel en USD"""
    # Normaliser le nom du modèle
    model_key = model.replace("models/", "")
    pricing = GEMINI_PRICING.get(model_key, GEMINI_PRICING.get("gemini-1.5-flash"))
    cost_in = (tokens_in / 1_000_000) * pricing["input"]
    cost_out = (tokens_out / 1_000_000) * pricing["output"]
    return cost_in + cost_out


# =============================================================================
# GEMINI PROVIDER
# =============================================================================


class GeminiProvider(LLMProvider):
    """
    Provider Google Gemini avec support des sorties structurées.

    Utilise response_mime_type="application/json" et response_schema
    pour garantir que la sortie respecte le schéma.
    """

    provider_type = LLMProviderType.GEMINI

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = None

    @property
    def client(self):
        """Lazy loading du client Gemini"""
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
        return self._client

    def _get_model(self, model_name: str):
        """Récupère le modèle Gemini"""
        return self.client.GenerativeModel(model_name)

    def _pydantic_to_gemini_schema(self, schema: Type[BaseModel]) -> dict:
        """
        Convertit un schéma Pydantic en format Gemini.
        Gemini ne supporte pas $defs, donc on nettoie le schéma.
        """
        json_schema = schema.model_json_schema()

        # Supprimer les champs non supportés par Gemini
        if "$defs" in json_schema:
            del json_schema["$defs"]
        if "definitions" in json_schema:
            del json_schema["definitions"]

        return json_schema

    def _get_schema_description(self, schema: Type[BaseModel]) -> str:
        """Génère une description textuelle du schéma pour le prompt"""
        json_schema = schema.model_json_schema()
        props = json_schema.get("properties", {})

        lines = [f"Schéma JSON attendu ({schema.__name__}):"]
        lines.append("{")

        for name, prop in props.items():
            ptype = prop.get("type", "any")
            desc = prop.get("description", "")
            required = name in json_schema.get("required", [])
            req_mark = "*" if required else ""

            if "enum" in prop:
                ptype = f"enum: {prop['enum']}"

            lines.append(f'  "{name}"{req_mark}: {ptype}  // {desc}')

        lines.append("}")
        return "\n".join(lines)

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        schema: Type[T],
        cfg: LLMConfig,
    ) -> tuple[T, LLMTrace]:
        """
        Génère une sortie structurée avec Gemini.

        Utilise response_mime_type="application/json" sans response_schema
        car Gemini a des limitations sur les schémas complexes ($defs).
        On ajoute le schéma dans le prompt à la place.
        """
        # Vérifier le budget
        can_proceed, reason = self._check_budget()
        if not can_proceed:
            raise LLMBudgetExceededError(reason)

        # Initialiser la trace
        trace = LLMTrace(
            provider=self.provider_type,
            model=model,
            step_name=cfg.step_name,
            run_id=cfg.run_id,
            prompt_hash=self.compute_prompt_hash(prompt),
            schema_name=schema.__name__,
            schema_version="1.0.0",
        )

        attempt = 0
        last_error: Optional[str] = None
        last_validation_errors: list[str] = []

        # Ajouter la description du schéma au prompt
        schema_desc = self._get_schema_description(schema)
        enhanced_prompt = f"{prompt}\n\n{schema_desc}\n\nRéponds UNIQUEMENT avec un JSON valide."

        while attempt < cfg.max_retries:
            attempt += 1
            trace.attempt = attempt

            try:
                # Construire le prompt (avec correction si retry)
                current_prompt = enhanced_prompt
                if attempt > 1 and last_validation_errors:
                    current_prompt = self._build_retry_prompt(
                        original_prompt=enhanced_prompt,
                        validation_errors=last_validation_errors,
                        schema=schema,
                    )

                # Configuration de génération - SANS response_schema
                generation_config = {
                    "temperature": cfg.temperature,
                    "max_output_tokens": cfg.max_tokens,
                    "top_p": cfg.top_p,
                    "response_mime_type": "application/json",
                }

                # Appel API
                start_time = time.time()

                gemini_model = self._get_model(model)
                response = gemini_model.generate_content(
                    current_prompt,
                    generation_config=generation_config,
                )

                # Métriques
                latency_ms = int((time.time() - start_time) * 1000)
                trace.latency_ms = latency_ms

                # Tokens (estimation car Gemini ne donne pas toujours les counts)
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    trace.tokens_in = getattr(response.usage_metadata, "prompt_token_count", 0)
                    trace.tokens_out = getattr(response.usage_metadata, "candidates_token_count", 0)
                else:
                    # Estimation
                    trace.tokens_in = self.estimate_tokens(current_prompt)
                    trace.tokens_out = self.estimate_tokens(response.text) if response.text else 0

                trace.cost_estimate_usd = estimate_cost(model, trace.tokens_in, trace.tokens_out)

                # Extraire le contenu
                content = response.text

                if not content:
                    last_error = "Empty response from Gemini"
                    last_validation_errors = [last_error]
                    logger.warning(f"[Gemini] Empty response attempt {attempt}")
                    continue

                # Parser le JSON avec tentative de réparation
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    # Tentative de réparation du JSON tronqué
                    repaired_content = self._repair_json(content)
                    if repaired_content:
                        try:
                            data = json.loads(repaired_content)
                            logger.info(f"[Gemini] JSON repaired successfully on attempt {attempt}")
                        except json.JSONDecodeError as e2:
                            last_error = f"JSON parse error: {e2}"
                            last_validation_errors = [last_error]
                            logger.warning(f"[Gemini] JSON parse error attempt {attempt}: {e2}")
                            continue
                    else:
                        last_error = f"JSON parse error: {e}"
                        last_validation_errors = [last_error]
                        logger.warning(f"[Gemini] JSON parse error attempt {attempt}: {e}")
                        continue

                # Valider avec Pydantic
                try:
                    result = schema.model_validate(data)

                    # Succès!
                    trace.status = (
                        LLMCallStatus.RETRY_SUCCESS if attempt > 1 else LLMCallStatus.SUCCESS
                    )
                    trace.finalize()

                    # Enregistrer l'usage
                    self._record_usage(trace.tokens_total, trace.cost_estimate_usd)

                    logger.info(
                        f"[Gemini] {model} | step={cfg.step_name} | "
                        f"tokens={trace.tokens_total} | cost=${trace.cost_estimate_usd:.4f} | "
                        f"latency={latency_ms}ms | attempt={attempt}"
                    )

                    return result, trace

                except ValidationError as e:
                    last_validation_errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                    last_error = f"Validation errors: {last_validation_errors}"
                    logger.warning(
                        f"[Gemini] Validation error attempt {attempt}: {last_validation_errors}"
                    )
                    continue

            except Exception as e:
                last_error = str(e)
                logger.error(f"[Gemini] API error attempt {attempt}: {e}")

                # Timeout spécifique
                if "timeout" in str(e).lower() or "deadline" in str(e).lower():
                    trace.status = LLMCallStatus.TIMEOUT
                    trace.error_message = last_error
                    trace.finalize()
                    raise LLMTimeoutError(f"Gemini timeout after {cfg.timeout_seconds}s", trace)

                continue

        # Échec après tous les retries
        trace.status = (
            LLMCallStatus.VALIDATION_ERROR if last_validation_errors else LLMCallStatus.FAILED
        )
        trace.error_message = last_error
        trace.validation_errors = last_validation_errors
        trace.finalize()

        raise LLMValidationError(
            f"Failed after {cfg.max_retries} attempts: {last_error}",
            errors=last_validation_errors,
            trace=trace,
        )

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        cfg: LLMConfig,
    ) -> tuple[str, LLMTrace]:
        """Génère une sortie texte libre"""
        # Vérifier le budget
        can_proceed, reason = self._check_budget()
        if not can_proceed:
            raise LLMBudgetExceededError(reason)

        # Initialiser la trace
        trace = LLMTrace(
            provider=self.provider_type,
            model=model,
            step_name=cfg.step_name,
            run_id=cfg.run_id,
            prompt_hash=self.compute_prompt_hash(prompt),
        )

        try:
            start_time = time.time()

            generation_config = {
                "temperature": cfg.temperature,
                "max_output_tokens": cfg.max_tokens,
                "top_p": cfg.top_p,
            }

            gemini_model = self._get_model(model)
            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            # Métriques
            trace.latency_ms = int((time.time() - start_time) * 1000)

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                trace.tokens_in = getattr(response.usage_metadata, "prompt_token_count", 0)
                trace.tokens_out = getattr(response.usage_metadata, "candidates_token_count", 0)
            else:
                trace.tokens_in = self.estimate_tokens(prompt)
                trace.tokens_out = self.estimate_tokens(response.text) if response.text else 0

            trace.cost_estimate_usd = estimate_cost(model, trace.tokens_in, trace.tokens_out)

            content = response.text or ""

            trace.status = LLMCallStatus.SUCCESS
            trace.finalize()

            self._record_usage(trace.tokens_total, trace.cost_estimate_usd)

            logger.info(
                f"[Gemini] {model} | text | "
                f"tokens={trace.tokens_total} | cost=${trace.cost_estimate_usd:.4f}"
            )

            return content, trace

        except Exception as e:
            trace.status = LLMCallStatus.FAILED
            trace.error_message = str(e)
            trace.finalize()
            raise LLMAPIError(str(e), trace=trace)

    def _build_retry_prompt(
        self,
        original_prompt: str,
        validation_errors: list[str],
        schema: Type[BaseModel],
    ) -> str:
        """Construit le prompt de retry après erreur de validation"""
        errors_str = "\n".join(f"- {e}" for e in validation_errors)

        schema_desc = schema.model_json_schema().get("description", schema.__name__)

        return f"""{original_prompt}

---
⚠️ CORRECTION REQUISE
Ton JSON précédent contenait des erreurs:
{errors_str}

Schéma attendu: {schema_desc}

Réponds UNIQUEMENT avec le JSON corrigé, sans texte."""

    def _repair_json(self, content: str) -> Optional[str]:
        """
        Tente de réparer un JSON tronqué ou malformé.
        Stratégies:
        1. Fermer les chaînes non terminées
        2. Équilibrer les accolades/crochets
        3. Supprimer les virgules trailing
        """
        if not content or not content.strip():
            return None

        content = content.strip()

        # Trouver le début du JSON (peut commencer par du texte)
        json_start = content.find("{")
        if json_start == -1:
            json_start = content.find("[")
        if json_start == -1:
            return None

        content = content[json_start:]

        # Compter les caractères de structure
        open_braces = content.count("{")
        close_braces = content.count("}")
        open_brackets = content.count("[")
        close_brackets = content.count("]")

        # Vérifier si on est au milieu d'une chaîne (nombre impair de ")
        in_string = False
        escaped = False
        last_char_idx = 0

        for i, char in enumerate(content):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                last_char_idx = i

        # Si on est dans une chaîne non fermée, on la ferme
        if in_string:
            # Trouver un bon point de coupure (avant le dernier " non fermé)
            # On coupe après le dernier caractère valide et on ferme
            content = content[: last_char_idx + 1] + '"'

        # Supprimer les virgules trailing avant fermeture
        import re

        content = re.sub(r",\s*([}\]])", r"\1", content)

        # Équilibrer les accolades
        missing_braces = open_braces - close_braces
        missing_brackets = open_brackets - close_brackets

        if missing_braces > 0:
            content = content.rstrip(",\n\r\t ") + ("}" * missing_braces)
        if missing_brackets > 0:
            content = content.rstrip(",\n\r\t ") + ("]" * missing_brackets)

        # Tenter de parser pour valider
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            # Dernière tentative: couper à la dernière structure valide
            # Trouver le dernier } ou ] et couper là
            for i in range(len(content) - 1, -1, -1):
                if content[i] in "}]":
                    truncated = content[: i + 1]
                    try:
                        json.loads(truncated)
                        return truncated
                    except json.JSONDecodeError:
                        continue
            return None
