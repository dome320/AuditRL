from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

FigureRole = Literal["input", "result", "standalone"]
ClaimTypeLiteral = Literal["arithmetic", "figure", "citation", "qualitative"]


class ExtractedFigure(BaseModel):
    value_text: str = Field(
        description="The number exactly as written in the claim, e.g. '$85.2 billion'"
    )
    metric: str = Field(description="What the number measures, e.g. 'services revenue'")
    period: str | None = Field(default=None, description="Fiscal period, e.g. 'fiscal 2023'")
    entity: str | None = None
    role: FigureRole = "standalone"


class ExtractedClaim(BaseModel):
    text: str = Field(description="Verbatim span copied from the report")
    claim_type: ClaimTypeLiteral
    checkable: bool
    cited_source: str | None = None
    figures: list[ExtractedFigure] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    claims: list[ExtractedClaim]


class EvidenceCandidate(BaseModel):
    passage_id: int
    document_name: str
    text: str
    row_label: str | None = None
    headers: list[str] | None = None
    cells: list[str] | None = None
    unit_scale: float = 1.0


class EvidenceSelection(BaseModel):
    found: bool
    passage_id: int | None = None
    quote: str = Field(default="", description="Verbatim substring of the chosen passage")
    value_text: str | None = Field(
        default=None, description="The value exactly as it appears in the quote"
    )
    reason: str = ""


class Operand(BaseModel):
    name: str
    value_text: str
    source: Literal["claim", "evidence"]
    passage_id: int | None = None


class ComputationProposal(BaseModel):
    applicable: bool = True
    operation: str = Field(description="growth | margin | sum | difference | ratio | cagr | other")
    expression: str = Field(description="Arithmetic over operand names, e.g. '(b - a) / a * 100'")
    operands: list[Operand]
    result_value_text: str = Field(description="The stated result exactly as written in the claim")


class RationaleOut(BaseModel):
    rationale: str


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    by_prompt: dict[str, int] = field(default_factory=dict)

    def add(self, prompt: str, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost
        self.calls += 1
        self.by_prompt[prompt] = self.by_prompt.get(prompt, 0) + 1


class LLMClient(ABC):
    """Proposes structured judgments; deterministic code in `auditrl.verify` decides verdicts."""

    name: str

    def __init__(self) -> None:
        self.usage = Usage()

    @abstractmethod
    def extract_claims(self, chunk: str) -> ExtractionResult: ...

    @abstractmethod
    def locate_evidence(
        self, claim_text: str, figure: ExtractedFigure, candidates: list[EvidenceCandidate]
    ) -> EvidenceSelection: ...

    @abstractmethod
    def propose_computation(
        self, claim_text: str, figures: list[ExtractedFigure], candidates: list[EvidenceCandidate]
    ) -> ComputationProposal | None: ...

    @abstractmethod
    def write_rationale(self, claim_text: str, verdict: str, facts: list[str]) -> str: ...
