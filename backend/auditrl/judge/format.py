"""Prompt and output format for the claim judge, shared by training data, inference, and rewards.

The judge sees one claim plus retrieved evidence passages labeled E1..Ek and returns a JSON
verdict. Training examples and live audits render evidence through the same function so
the fine-tuned model sees identical inputs in both settings.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from auditrl.llm.litellm_client import parse_json_object

JUDGE_PROMPT_VERSION = "judge-v1"

ErrorTypeLiteral = Literal[
    "arithmetic_error",
    "misstated_figure",
    "wrong_period",
    "unit_error",
    "citation_mismatch",
    "missing_source",
]

SYSTEM = (
    "You audit claims from financial research reports against source filings. Judge only "
    "from the evidence passages provided. Answer with a single JSON object and nothing else."
)

USER = """Claim: {claim}
Cited source: {cited}

Evidence passages:
{evidence}

Decide whether the evidence supports the claim.
- "valid": every figure in the claim matches the evidence (allowing for rounding) and any
  cited source is the right filing.
- "invalid": the evidence contradicts the claim. Give the error_type ({error_types}),
  the evidence ids you relied on, and a correction: the claim rewritten with the right
  figure, period, unit, or source.
- "uncertain": the evidence does not contain what is needed to decide. Give no correction.
Percentages must be recomputed from the evidence values. Table values are scaled by the
stated units.

Reply with JSON: {{"verdict": ..., "error_type": ... or null, "evidence": ["E1", ...],
"correction": ... or null, "rationale": "one or two sentences"}}"""

ERROR_TYPES = ", ".join(ErrorTypeLiteral.__args__)  # type: ignore[attr-defined]


@dataclass
class EvidenceItem:
    """A retrieved passage as the judge sees it. `eid` is its local label (E1, E2, ...)."""

    eid: str
    document: str
    text: str
    row_label: str | None = None
    headers: list[str] | None = None
    cells: list[str] | None = None
    unit_scale: float = 1.0
    passage_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JudgeOutput(BaseModel):
    verdict: Literal["valid", "invalid", "uncertain"]
    error_type: ErrorTypeLiteral | None = None
    evidence: list[str] = Field(default_factory=list)
    correction: str | None = None
    rationale: str = ""

    @field_validator("error_type", "correction", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        return None if v in ("", "null", "none", "None") else v

    def normalized(self) -> "JudgeOutput":
        if self.verdict != "invalid":
            return self.model_copy(update={"error_type": None, "correction": None})
        return self


def _scale_note(scale: float) -> str:
    return {1e3: " (in thousands)", 1e6: " (in millions)", 1e9: " (in billions)"}.get(scale, "")


def render_item(item: EvidenceItem, max_chars: int = 600) -> str:
    if item.row_label and item.cells:
        headers = item.headers or []
        cols = [
            f"{headers[i] if i < len(headers) and headers[i] else f'col{i + 1}'}: {c}"
            for i, c in enumerate(item.cells)
        ]
        body = f"{item.row_label}{_scale_note(item.unit_scale)} | " + " | ".join(cols)
    else:
        body = " ".join(item.text.split())[:max_chars]
    return f"[{item.eid}] ({item.document}) {body}"


def build_messages(
    claim: str, cited_source: str | None, evidence: list[EvidenceItem]
) -> list[dict[str, str]]:
    user = USER.format(
        claim=claim.strip(),
        cited=cited_source or "none",
        evidence="\n".join(render_item(e) for e in evidence) or "(no passages retrieved)",
        error_types=ERROR_TYPES,
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def render_target(out: JudgeOutput) -> str:
    out = out.normalized()
    return json.dumps(
        {
            "verdict": out.verdict,
            "error_type": out.error_type,
            "evidence": out.evidence,
            "correction": out.correction,
            "rationale": out.rationale,
        },
        ensure_ascii=False,
    )


def parse_output(text: str) -> JudgeOutput:
    """Raises ValueError (or pydantic ValidationError, a ValueError) on malformed output."""
    return JudgeOutput.model_validate(parse_json_object(text)).normalized()
