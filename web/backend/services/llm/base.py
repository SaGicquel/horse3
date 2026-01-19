"""
🤖 LLM Provider - Base Interface & Types
========================================

Interface commune pour tous les providers LLM (OpenAI, Gemini, etc.)
Assure l'interchangeabilité et la traçabilité.

Auteur: Agent IA Pipeline
Date: 2024-12-21
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TypeVar, Generic, Type
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================


class LLMProviderType(str, Enum):
    """Types de providers LLM disponibles"""

    OPENAI = "openai"
    GEMINI = "gemini"


class LLMCallStatus(str, Enum):
    """Statut d'un appel LLM"""

    SUCCESS = "SUCCESS"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RETRY_SUCCESS = "RETRY_SUCCESS"
    FAILED = "FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    TIMEOUT = "TIMEOUT"


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class LLMConfig:
    """Configuration pour un appel LLM"""

    # Modèle
    temperature: float = 0.0  # Déterministe par défaut
    max_tokens: int = 4096
    top_p: float = 1.0

    # Timeouts
    timeout_seconds: int = 60
    max_retries: int = 3  # Retries pour validation errors (Gemini needs more)

    # Budget guards
    max_tokens_per_call: int = 16000

    # Méta
    step_name: str = ""  # A, B, C, D
    run_id: Optional[str] = None


@dataclass
class LLMBudgetConfig:
    """Limites budgétaires pour un run complet"""

    max_llm_calls_per_run: int = 10
    max_tokens_per_run: int = 100000
    max_latency_per_call_ms: int = 30000  # 30 secondes
    max_cost_per_run_usd: float = 1.0

    # Compteurs (mis à jour pendant le run)
    current_calls: int = 0
    current_tokens: int = 0
    current_cost_usd: float = 0.0

    def can_proceed(self) -> tuple[bool, Optional[str]]:
        """Vérifie si on peut faire un nouvel appel LLM"""
        if self.current_calls >= self.max_llm_calls_per_run:
            return False, f"Max calls exceeded: {self.current_calls}/{self.max_llm_calls_per_run}"
        if self.current_tokens >= self.max_tokens_per_run:
            return False, f"Max tokens exceeded: {self.current_tokens}/{self.max_tokens_per_run}"
        if self.current_cost_usd >= self.max_cost_per_run_usd:
            return (
                False,
                f"Max cost exceeded: ${self.current_cost_usd:.4f}/${self.max_cost_per_run_usd}",
            )
        return True, None

    def record_usage(self, tokens: int, cost_usd: float):
        """Enregistre l'usage d'un appel"""
        self.current_calls += 1
        self.current_tokens += tokens
        self.current_cost_usd += cost_usd


# =============================================================================
# TRACE (OBSERVABILITÉ)
# =============================================================================


class LLMTrace(BaseModel):
    """
    Trace complète d'un appel LLM pour audit et debug.
    Stocké dans agent_steps.output_json ou meta JSONB.
    """

    # Identifiants
    trace_id: str = Field(default_factory=lambda: str(uuid4())[:12])
    run_id: Optional[str] = None
    step_name: str = ""  # A, B, C, D

    # Provider
    provider: LLMProviderType
    model: str

    # Prompt
    prompt_version: str = "1.0.0"
    prompt_hash: str = ""  # SHA256[:16] du prompt
    inputs_hash: Optional[str] = None  # Hash des inputs de replay

    # Schema
    schema_name: Optional[str] = None
    schema_version: Optional[str] = None

    # Tokens & Cost
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_total: int = 0
    cost_estimate_usd: float = 0.0

    # Performance
    latency_ms: int = 0
    attempt: int = 1  # 1, 2, 3...

    # Résultat
    status: LLMCallStatus = LLMCallStatus.SUCCESS
    error_message: Optional[str] = None
    validation_errors: Optional[list[str]] = None

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None

    def finalize(self):
        """Finalise la trace avec les calculs dérivés"""
        self.finished_at = datetime.utcnow()
        self.tokens_total = self.tokens_in + self.tokens_out


# =============================================================================
# INTERFACE PROVIDER
# =============================================================================

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """
    Interface abstraite pour les providers LLM.

    Tous les providers doivent implémenter:
    - generate_structured(): Sortie JSON validée par Pydantic
    - generate_text(): Sortie texte libre

    Usage:
        provider = OpenAIProvider(api_key="...")
        result = provider.generate_structured(
            model="gpt-4o-mini",
            prompt="Analyse ce rapport...",
            schema=StepBOutput,
            cfg=LLMConfig(temperature=0)
        )
    """

    provider_type: LLMProviderType

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._budget: Optional[LLMBudgetConfig] = None

    def set_budget(self, budget: LLMBudgetConfig):
        """Configure les limites budgétaires"""
        self._budget = budget

    @abstractmethod
    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        schema: Type[T],
        cfg: LLMConfig,
    ) -> tuple[T, LLMTrace]:
        """
        Génère une sortie structurée validée par le schéma Pydantic.

        Args:
            model: Nom du modèle
            prompt: Prompt complet
            schema: Classe Pydantic pour la validation
            cfg: Configuration

        Returns:
            Tuple (instance du schéma validée, trace d'exécution)

        Raises:
            LLMValidationError: Si validation échoue après retries
            LLMBudgetExceededError: Si budget dépassé
            LLMTimeoutError: Si timeout
        """
        ...

    @abstractmethod
    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        cfg: LLMConfig,
    ) -> tuple[str, LLMTrace]:
        """
        Génère une sortie texte libre.

        Args:
            model: Nom du modèle
            prompt: Prompt complet
            cfg: Configuration

        Returns:
            Tuple (texte généré, trace d'exécution)
        """
        ...

    def _check_budget(self) -> tuple[bool, Optional[str]]:
        """Vérifie le budget disponible"""
        if self._budget is None:
            return True, None
        return self._budget.can_proceed()

    def _record_usage(self, tokens: int, cost: float):
        """Enregistre l'usage"""
        if self._budget:
            self._budget.record_usage(tokens, cost)

    @staticmethod
    def compute_prompt_hash(prompt: str) -> str:
        """Calcule le hash SHA256 du prompt (16 premiers chars)"""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimation grossière du nombre de tokens (1 token ≈ 4 chars)"""
        return len(text) // 4


# =============================================================================
# EXCEPTIONS
# =============================================================================


class LLMError(Exception):
    """Erreur de base pour les appels LLM"""

    pass


class LLMValidationError(LLMError):
    """Erreur de validation du schéma Pydantic"""

    def __init__(self, message: str, errors: list[str], trace: LLMTrace):
        super().__init__(message)
        self.errors = errors
        self.trace = trace


class LLMBudgetExceededError(LLMError):
    """Budget dépassé"""

    def __init__(self, message: str, trace: Optional[LLMTrace] = None):
        super().__init__(message)
        self.trace = trace


class LLMTimeoutError(LLMError):
    """Timeout de l'appel LLM"""

    def __init__(self, message: str, trace: LLMTrace):
        super().__init__(message)
        self.trace = trace


class LLMAPIError(LLMError):
    """Erreur API (rate limit, auth, etc.)"""

    def __init__(
        self, message: str, status_code: Optional[int] = None, trace: Optional[LLMTrace] = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.trace = trace


# =============================================================================
# FACTORY
# =============================================================================


def get_provider(
    provider_type: LLMProviderType,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """
    Factory pour obtenir un provider LLM.

    Args:
        provider_type: Type de provider (openai, gemini)
        api_key: Clé API (optionnel, lit depuis env sinon)

    Returns:
        Instance du provider
    """
    import os

    if provider_type == LLMProviderType.OPENAI:
        from .openai_provider import OpenAIProvider

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found")
        return OpenAIProvider(api_key=key)

    elif provider_type == LLMProviderType.GEMINI:
        from .gemini_provider import GeminiProvider

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found")
        return GeminiProvider(api_key=key)

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
