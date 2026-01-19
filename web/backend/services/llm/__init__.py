"""
🤖 LLM Module - Agent IA
========================

Module isolé pour tous les appels LLM (OpenAI, Gemini).
Aucune logique LLM n'est diluée dans les services métier.

Structure:
- base.py: Interface LLMProvider + types communs
- schemas.py: Modèles Pydantic pour Step B/C/D
- prompt_store.py: Prompts versionnés avec hashing
- openai_provider.py: Implémentation OpenAI
- gemini_provider.py: Implémentation Gemini

Usage:
    from services.llm import get_provider, LLMProviderType, LLMConfig
    from services.llm.schemas import StepBOutput

    provider = get_provider(LLMProviderType.OPENAI)
    result, trace = provider.generate_structured(
        model="gpt-4o-mini",
        prompt="...",
        schema=StepBOutput,
        cfg=LLMConfig(temperature=0, step_name="B")
    )

Auteur: Agent IA Pipeline
Date: 2024-12-21
"""

# Base
from .base import (
    LLMProvider,
    LLMProviderType,
    LLMConfig,
    LLMBudgetConfig,
    LLMTrace,
    LLMCallStatus,
    LLMError,
    LLMValidationError,
    LLMBudgetExceededError,
    LLMTimeoutError,
    LLMAPIError,
    get_provider,
)

# Schemas
from .schemas import (
    StepBOutput,
    StepCOutput,
    StepDOutput,
    ConfidenceLevel,
    RecommendationAction,
    RiskAssessment,
    get_schema_for_step,
    SCHEMA_REGISTRY,
)

# Prompts
from .prompt_store import (
    get_prompt,
    format_prompt,
    list_prompts,
    compute_prompt_hash,
    PromptMetadata,
    PROMPT_REGISTRY,
)

# Providers (import explicite pour éviter circular)
# from .openai_provider import OpenAIProvider
# from .gemini_provider import GeminiProvider

__all__ = [
    # Base
    "LLMProvider",
    "LLMProviderType",
    "LLMConfig",
    "LLMBudgetConfig",
    "LLMTrace",
    "LLMCallStatus",
    "LLMError",
    "LLMValidationError",
    "LLMBudgetExceededError",
    "LLMTimeoutError",
    "LLMAPIError",
    "get_provider",
    # Schemas
    "StepBOutput",
    "StepCOutput",
    "StepDOutput",
    "ConfidenceLevel",
    "RecommendationAction",
    "RiskAssessment",
    "get_schema_for_step",
    "SCHEMA_REGISTRY",
    # Prompts
    "get_prompt",
    "format_prompt",
    "list_prompts",
    "compute_prompt_hash",
    "PromptMetadata",
    "PROMPT_REGISTRY",
]
