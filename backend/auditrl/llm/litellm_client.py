import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from auditrl.llm import prompts
from auditrl.llm.base import (
    ComputationProposal,
    EvidenceCandidate,
    EvidenceSelection,
    ExtractedFigure,
    ExtractionResult,
    LLMClient,
    RationaleOut,
)

# (prompt_name, messages) -> (content, prompt_tokens, completion_tokens, cost_usd)
Transport = Callable[[str, list[dict[str, str]]], tuple[str, int, int, float]]


class LLMResponseError(RuntimeError):
    pass


def parse_json_object(content: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.S)
    if fenced:
        content = fenced.group(1)
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(content[start : end + 1])


def format_candidates(candidates: list[EvidenceCandidate]) -> str:
    return (
        "\n".join(f"[{c.passage_id}] ({c.document_name}) {c.text}" for c in candidates) or "(none)"
    )


class LiteLLMClient(LLMClient):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        transport: Transport | None = None,
        max_retries: int = 1,
    ) -> None:
        super().__init__()
        self.name = model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.transport = transport or self._litellm_transport
        self.max_retries = max_retries

    def _litellm_transport(
        self, prompt_name: str, messages: list[dict[str, str]]
    ) -> tuple[str, int, int, float]:
        import litellm

        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:
            cost = 0.0
        return (
            content,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            cost,
        )

    def _call[T: BaseModel](self, prompt_name: str, template: str, schema: type[T], **kw: Any) -> T:
        user = template.format(schema=json.dumps(schema.model_json_schema()), **kw)
        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            content, p_tok, c_tok, cost = self.transport(prompt_name, messages)
            self.usage.add(prompt_name, p_tok, c_tok, cost)
            try:
                return schema.model_validate(parse_json_object(content))
            except (ValueError, ValidationError) as exc:
                last_error = exc
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"That was not valid JSON for the schema ({exc}). Reply again.",
                    },
                ]
        raise LLMResponseError(f"{prompt_name}: {last_error}")

    def extract_claims(self, chunk: str) -> ExtractionResult:
        return self._call("extract_claims", prompts.EXTRACT_CLAIMS, ExtractionResult, chunk=chunk)

    def locate_evidence(
        self, claim_text: str, figure: ExtractedFigure, candidates: list[EvidenceCandidate]
    ) -> EvidenceSelection:
        if not candidates:
            return EvidenceSelection(found=False, reason="no candidate passages")
        return self._call(
            "locate_evidence",
            prompts.LOCATE_EVIDENCE,
            EvidenceSelection,
            claim_text=claim_text,
            value_text=figure.value_text,
            metric=figure.metric,
            period=figure.period or "unspecified",
            entity=figure.entity or "unspecified",
            candidates=format_candidates(candidates),
        )

    def propose_computation(
        self, claim_text: str, figures: list[ExtractedFigure], candidates: list[EvidenceCandidate]
    ) -> ComputationProposal | None:
        proposal = self._call(
            "propose_computation",
            prompts.PROPOSE_COMPUTATION,
            ComputationProposal,
            claim_text=claim_text,
            figures=json.dumps([f.model_dump() for f in figures]),
            candidates=format_candidates(candidates),
        )
        return proposal if proposal.applicable else None

    def write_rationale(self, claim_text: str, verdict: str, facts: list[str]) -> str:
        out = self._call(
            "write_rationale",
            prompts.WRITE_RATIONALE,
            RationaleOut,
            claim_text=claim_text,
            verdict=verdict,
            facts="\n".join(f"- {f}" for f in facts),
        )
        return out.rationale.strip()
