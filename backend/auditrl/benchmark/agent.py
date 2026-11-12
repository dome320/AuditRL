"""A financial research agent whose reports (and their natural mistakes) form the benchmark.

The agent reads the task's filings through tools, may use a calculator, and writes a short
results review. Every message and tool call is kept as a trace (spec: collect traces).
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from auditrl.verify.safe_eval import UnsafeExpression, evaluate

AGENT_PROMPT_VERSION = "agent-v1"

SYSTEM = (
    "You are a sell-side equity research analyst writing a concise results review for "
    "institutional clients. Work only from the filings you are given."
)

USER = """Write a results review of {company} ({ticker}) for fiscal {fy}.

Filings available: {titles}.

Requirements:
- 200-350 words of prose in 2-4 paragraphs; no tables, headings, or bullet lists.
- Cover revenue and its growth versus fiscal {prior}, gross, operating and net margins,
  net income, diluted EPS, operating cash flow and capital expenditures where reported.
- State the fiscal year for every figure.
- Express dollar amounts in billions with one decimal place (e.g. $12.3 billion) and
  per-share amounts in dollars (e.g. $6.42).
- Cite the filing for key figures as (Source: <filing title>).
- Do not make investment recommendations or price targets."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the titles of the filings available for this task.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Return the full text of a filing by its title.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate an arithmetic expression, e.g. '(416161 - 391035) / 391035 * 100'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


@dataclass
class Task:
    task_id: str
    ticker: str
    cik: int
    company: str
    fiscal_year: int
    sources: dict[str, str]  # file name -> document title

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        return cls(**d)


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


Completion = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], AssistantTurn]


def litellm_completion(model: str, temperature: float) -> Completion:
    def complete(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> AssistantTurn:
        import litellm

        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
        resp = litellm.completion(**kwargs)
        msg = resp.choices[0].message
        calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]
        usage = getattr(resp, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:
            cost = 0.0
        return AssistantTurn(
            content=msg.content,
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
        )

    return complete


@dataclass
class Trace:
    task: dict[str, Any]
    model: str
    mode: str
    prompt_version: str
    temperature: float
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    report: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    started_at: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchAgent:
    def __init__(
        self,
        model: str,
        *,
        complete: Completion | None = None,
        mode: Literal["tools", "context"] = "tools",
        temperature: float = 0.7,
        max_steps: int = 10,
    ) -> None:
        self.model = model
        self.mode = mode
        self.temperature = temperature
        self.max_steps = max_steps
        self.complete = complete or litellm_completion(model, temperature)

    def run(self, task: Task, documents: dict[str, str]) -> Trace:
        """`documents` maps document title -> text."""
        started = time.perf_counter()
        titles = list(documents)
        user = USER.format(
            company=task.company,
            ticker=task.ticker,
            fy=task.fiscal_year,
            prior=task.fiscal_year - 1,
            titles="; ".join(titles),
        )
        if self.mode == "context":
            user += "\n\n" + "\n\n".join(f"=== {t} ===\n{documents[t]}" for t in titles)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        tools = TOOLS if self.mode == "tools" else None
        tool_log: list[dict[str, Any]] = []
        totals = AssistantTurn(content=None)
        report = ""
        error: str | None = None
        try:
            report = self._loop(messages, tools, documents, tool_log, totals)
        except Exception as exc:  # provider/network errors end the episode with a recorded error
            error = f"{type(exc).__name__}: {exc}"

        return Trace(
            task=asdict(task),
            model=self.model,
            mode=self.mode,
            prompt_version=AGENT_PROMPT_VERSION,
            temperature=self.temperature,
            messages=messages,
            tool_calls=tool_log,
            report=report,
            prompt_tokens=totals.prompt_tokens,
            completion_tokens=totals.completion_tokens,
            cost_usd=round(totals.cost_usd, 6),
            latency_s=round(time.perf_counter() - started, 3),
            started_at=datetime.now(UTC).isoformat(),
            error=error or (None if report else "agent produced no report"),
        )

    def _loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        documents: dict[str, str],
        tool_log: list[dict[str, Any]],
        totals: AssistantTurn,
    ) -> str:
        for step in range(self.max_steps):
            final_step = step == self.max_steps - 1
            if final_step and tools:
                messages.append({"role": "user", "content": "Write the final report now."})
            turn = self.complete(messages, None if final_step else tools)
            totals.prompt_tokens += turn.prompt_tokens
            totals.completion_tokens += turn.completion_tokens
            totals.cost_usd += turn.cost_usd
            assistant: dict[str, Any] = {"role": "assistant", "content": turn.content}
            if turn.tool_calls:
                assistant["tool_calls"] = turn.tool_calls
            messages.append(assistant)
            if not turn.tool_calls:
                return (turn.content or "").strip()
            for call in turn.tool_calls:
                result = self._run_tool(
                    call["function"]["name"], call["function"]["arguments"], documents
                )
                tool_log.append(
                    {
                        "step": step,
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                        "result": result,
                    }
                )
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        return ""

    @staticmethod
    def _run_tool(name: str, arguments: str, documents: dict[str, str]) -> str:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return "error: arguments must be a JSON object"
        if name == "list_documents":
            return "\n".join(documents)
        if name == "read_document":
            title = str(args.get("title", ""))
            if title in documents:
                return documents[title]
            matches = [t for t in documents if title.lower() in t.lower()]
            return (
                documents[matches[0]]
                if len(matches) == 1
                else f"error: no document titled {title!r}"
            )
        if name == "calculate":
            try:
                return repr(round(evaluate(str(args.get("expression", "")), {}), 6))
            except (UnsafeExpression, SyntaxError, ZeroDivisionError, OverflowError) as exc:
                return f"error: {exc}"
        return f"error: unknown tool {name!r}"
