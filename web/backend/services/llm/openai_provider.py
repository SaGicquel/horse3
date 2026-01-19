"""
🤖 OpenAI Provider - Structured Outputs Implementation
======================================================

Implémentation du provider OpenAI avec support des Structured Outputs.
Utilise le mode JSON Schema pour garantir la conformité du schéma.

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
logger = logging.getLogger("agent_ia.llm.openai")

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# PRICING (USD per 1M tokens) - Décembre 2024
# =============================================================================

OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estime le coût d'un appel en USD"""
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING.get("gpt-4o-mini"))
    cost_in = (tokens_in / 1_000_000) * pricing["input"]
    cost_out = (tokens_out / 1_000_000) * pricing["output"]
    return cost_in + cost_out


# =============================================================================
# OPENAI PROVIDER
# =============================================================================


class OpenAIProvider(LLMProvider):
    """
    Provider OpenAI avec support des Structured Outputs.

    Utilise response_format avec json_schema pour garantir
    que la sortie respecte le schéma Pydantic.
    """

    provider_type = LLMProviderType.OPENAI

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self._client = None

    @property
    def client(self):
        """Lazy loading du client OpenAI"""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def _pydantic_to_json_schema(self, schema: Type[BaseModel]) -> dict:
        """Convertit un schéma Pydantic en JSON Schema pour OpenAI"""
        json_schema = schema.model_json_schema()

        # OpenAI Structured Outputs nécessite un format spécifique
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": json_schema,
            },
        }

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        schema: Type[T],
        cfg: LLMConfig,
    ) -> tuple[T, LLMTrace]:
        """
        Génère une sortie structurée avec Structured Outputs.

        Retry automatique si validation Pydantic échoue.
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

        while attempt < cfg.max_retries:
            attempt += 1
            trace.attempt = attempt

            try:
                # Construire le prompt (avec correction si retry)
                current_prompt = prompt
                if attempt > 1 and last_validation_errors:
                    current_prompt = self._build_retry_prompt(
                        original_prompt=prompt,
                        validation_errors=last_validation_errors,
                        schema=schema,
                    )

                # Appel API
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Tu es un assistant qui répond UNIQUEMENT en JSON valide.",
                        },
                        {"role": "user", "content": current_prompt},
                    ],
                    response_format=self._pydantic_to_json_schema(schema),
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    top_p=cfg.top_p,
                    timeout=cfg.timeout_seconds,
                )

                # Métriques
                latency_ms = int((time.time() - start_time) * 1000)
                trace.latency_ms = latency_ms

                # Tokens
                if response.usage:
                    trace.tokens_in = response.usage.prompt_tokens
                    trace.tokens_out = response.usage.completion_tokens
                    trace.cost_estimate_usd = estimate_cost(
                        model, trace.tokens_in, trace.tokens_out
                    )

                # Extraire le contenu
                content = response.choices[0].message.content

                # Parser le JSON
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    last_error = f"JSON parse error: {e}"
                    last_validation_errors = [last_error]
                    logger.warning(f"[OpenAI] JSON parse error attempt {attempt}: {e}")
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
                        f"[OpenAI] {model} | step={cfg.step_name} | "
                        f"tokens={trace.tokens_total} | cost=${trace.cost_estimate_usd:.4f} | "
                        f"latency={latency_ms}ms | attempt={attempt}"
                    )

                    return result, trace

                except ValidationError as e:
                    last_validation_errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                    last_error = f"Validation errors: {last_validation_errors}"
                    logger.warning(
                        f"[OpenAI] Validation error attempt {attempt}: {last_validation_errors}"
                    )
                    continue

            except Exception as e:
                last_error = str(e)
                logger.error(f"[OpenAI] API error attempt {attempt}: {e}")

                # Timeout spécifique
                if "timeout" in str(e).lower():
                    trace.status = LLMCallStatus.TIMEOUT
                    trace.error_message = last_error
                    trace.finalize()
                    raise LLMTimeoutError(f"OpenAI timeout after {cfg.timeout_seconds}s", trace)

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

            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                timeout=cfg.timeout_seconds,
            )

            # Métriques
            trace.latency_ms = int((time.time() - start_time) * 1000)

            if response.usage:
                trace.tokens_in = response.usage.prompt_tokens
                trace.tokens_out = response.usage.completion_tokens
                trace.cost_estimate_usd = estimate_cost(model, trace.tokens_in, trace.tokens_out)

            content = response.choices[0].message.content or ""

            trace.status = LLMCallStatus.SUCCESS
            trace.finalize()

            self._record_usage(trace.tokens_total, trace.cost_estimate_usd)

            logger.info(
                f"[OpenAI] {model} | text | "
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

        # Obtenir la description du schéma
        schema_desc = schema.model_json_schema().get("description", schema.__name__)

        return f"""{original_prompt}

---
⚠️ CORRECTION REQUISE
Ton JSON précédent contenait des erreurs:
{errors_str}

Schéma attendu: {schema_desc}

Réponds UNIQUEMENT avec le JSON corrigé, sans texte."""
