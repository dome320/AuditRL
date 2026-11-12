from typing import Any

from pydantic import ValidationError

from auditrl.judge.format import EvidenceItem, JudgeOutput, build_messages, parse_output
from auditrl.llm.base import Usage
from auditrl.llm.litellm_client import Transport


class JudgeClient:
    """Calls a judge model through LiteLLM: a frontier model, the untuned base model, or a
    fine-tuned adapter served behind an OpenAI-compatible endpoint (`api_base`)."""

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        transport: Transport | None = None,
        max_retries: int = 1,
    ) -> None:
        self.name = model
        self.model = model
        self.api_base = api_base or None
        self.api_key = api_key or None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.transport = transport or self._litellm_transport
        self.max_retries = max_retries
        self.usage = Usage()

    def _litellm_transport(
        self, prompt_name: str, messages: list[dict[str, str]]
    ) -> tuple[str, int, int, float]:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
            kwargs["api_key"] = self.api_key or "EMPTY"
        response = litellm.completion(**kwargs)
        usage = getattr(response, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:
            cost = 0.0
        return (
            response.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            cost,
        )

    def judge(
        self, claim: str, cited_source: str | None, evidence: list[EvidenceItem]
    ) -> tuple[JudgeOutput | None, str]:
        """Returns (parsed output or None if unparseable after retries, last raw response)."""
        messages = build_messages(claim, cited_source, evidence)
        content = ""
        for _ in range(self.max_retries + 1):
            content, p_tok, c_tok, cost = self.transport("judge", messages)
            self.usage.add("judge", p_tok, c_tok, cost)
            try:
                return parse_output(content), content
            except (ValueError, ValidationError) as exc:
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"Invalid JSON for the format ({exc}). Reply again.",
                    },
                ]
        return None, content
