from dataclasses import dataclass, field
from typing import Any

from auditrl.models import ErrorType, Verdict


@dataclass
class Evidence:
    passage_id: int
    quote: str


@dataclass
class CheckResult:
    """Outcome of one deterministic check on one claim.

    `decisive` checks determine the claim verdict; non-decisive ones (e.g. an input figure
    with no evidence behind a verified computation) are kept for transparency only.
    """

    kind: str
    verdict: Verdict
    decisive: bool = True
    error_type: ErrorType | None = None
    facts: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    replacements: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.5
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "verdict": self.verdict.value,
            "decisive": self.decisive,
            "error_type": self.error_type.value if self.error_type else None,
            "facts": self.facts,
            "evidence": [{"passage_id": e.passage_id, "quote": e.quote} for e in self.evidence],
            "replacements": [list(r) for r in self.replacements],
            "confidence": self.confidence,
            "details": self.details,
        }


def uncertain(kind: str, reason: str, decisive: bool = True, **details: Any) -> CheckResult:
    return CheckResult(
        kind=kind,
        verdict=Verdict.uncertain,
        decisive=decisive,
        facts=[reason],
        confidence=0.3,
        details=details,
    )
