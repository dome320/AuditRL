from auditrl.config import Settings, get_settings
from auditrl.llm.base import (
    ComputationProposal,
    EvidenceCandidate,
    EvidenceSelection,
    ExtractedClaim,
    ExtractedFigure,
    ExtractionResult,
    LLMClient,
    Operand,
)
from auditrl.llm.heuristic import HeuristicClient
from auditrl.llm.litellm_client import LiteLLMClient
from auditrl.llm.prompts import PROMPT_VERSION


def get_client(model: str | None = None, settings: Settings | None = None) -> LLMClient:
    """`model` is any LiteLLM model string, or "heuristic" for the offline rule-based client."""
    settings = settings or get_settings()
    model = model or settings.model
    if model == "heuristic":
        return HeuristicClient()
    return LiteLLMClient(model, temperature=settings.temperature, max_tokens=settings.max_tokens)


__all__ = [
    "PROMPT_VERSION",
    "ComputationProposal",
    "EvidenceCandidate",
    "EvidenceSelection",
    "ExtractedClaim",
    "ExtractedFigure",
    "ExtractionResult",
    "HeuristicClient",
    "LLMClient",
    "LiteLLMClient",
    "Operand",
    "get_client",
]
