"""Synthetic judge examples built from real XBRL facts with deterministically injected errors.

Each example is a claim sentence about a real company and fiscal year, the evidence the
auditor's retrieval returns for it from that task's two filings, and a target verdict that
is exact by construction. If retrieval misses the rows needed to decide, the target is
`uncertain`: the judge should learn to abstain rather than guess, exactly as at inference.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from auditrl.benchmark.agent import Task
from auditrl.benchmark.dataset import load_filings
from auditrl.benchmark.edgar import Filing
from auditrl.benchmark.metrics import BY_KEY
from auditrl.intake import parse_path
from auditrl.judge.evidence import index_from_parsed, retrieve
from auditrl.judge.format import EvidenceItem, JudgeOutput

PHRASES = {
    "revenue": ["revenue", "total revenue", "net sales"],
    "cost_of_revenue": ["cost of revenue", "cost of sales"],
    "gross_profit": ["gross profit"],
    "rnd": ["research and development expense", "R&D spending"],
    "sga": ["SG&A expense", "selling, general and administrative expense"],
    "operating_income": ["operating income", "income from operations"],
    "net_income": ["net income", "net earnings"],
    "eps_diluted": ["diluted EPS", "diluted earnings per share"],
    "operating_cash_flow": ["operating cash flow", "cash from operations"],
    "capex": ["capital expenditures", "capex"],
}
MARGIN_NUMERATORS = {
    "gross margin": "gross_profit",
    "operating margin": "operating_income",
    "net margin": "net_income",
}
UNSUPPORTED = [
    ("Share repurchases totaled {v} in fiscal {fy}.", 0.04),
    ("Backlog stood at {v} at the end of fiscal {fy}.", 0.3),
    ("Dividends paid amounted to {v} in fiscal {fy}.", 0.03),
    ("Remaining performance obligations reached {v} at fiscal {fy} year end.", 0.5),
    ("Stock-based compensation expense was {v} in fiscal {fy}.", 0.02),
]
MONEY_TEMPLATES = [
    "{Co} reported {phrase} of {v} in fiscal {fy}.",
    "{Phrase} was {v} in fiscal {fy}.",
    "In fiscal {fy}, {phrase} came in at {v}.",
    "{Phrase} reached {v} in fiscal {fy}, compared with {pv} in fiscal {py}.",
]


@dataclass
class Example:
    claim: str
    cited_source: str | None
    evidence: list[EvidenceItem]
    target: JudgeOutput
    kind: str
    meta: dict = field(default_factory=dict)


def money(v: float) -> str:
    if abs(v) >= 1e9:
        return f"${v / 1e9:,.1f} billion"
    return f"${v / 1e6:,.0f} million"


def per_share(v: float) -> str:
    return f"${v:.2f}"


def pct(v: float) -> str:
    return f"{v:.1f}%"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


class TaskGenerator:
    def __init__(self, bench: Path, task: Task, rng: random.Random, top_k: int = 8) -> None:
        self.task, self.rng, self.top_k = task, rng, top_k
        src = bench / "sources" / task.ticker
        self.index = index_from_parsed([parse_path(src / name) for name in task.sources])
        filings = {f.fiscal_year: f for f in load_filings(src / "facts.json")}
        self.fy = task.fiscal_year
        self.cur: Filing = filings[self.fy]
        titles = {title: name for name, title in task.sources.items()}
        by_year = {int(name.rsplit("FY", 1)[1].split(".")[0]): t for t, name in titles.items()}
        self.title_cur, self.title_prev = by_year[self.fy], by_year[self.fy - 1]

    # --- facts and evidence -------------------------------------------------

    def value(self, metric: str, year: int) -> float | None:
        """The value as the rendered filing shows it (money rounded to millions), so gold
        figures are exactly derivable from the evidence the judge sees."""
        fact = self.cur.value(metric, year)
        if fact is None:
            return None
        return fact.value if BY_KEY[metric].per_share else round(fact.value / 1e6) * 1e6

    def _rows(self, evidence: list[EvidenceItem], metric: str, year: int) -> list[EvidenceItem]:
        """Retrieved rows reporting `metric` for `year`, current filing first."""
        label = BY_KEY[metric].label
        col = f"Fiscal {year}"
        rows = [
            e
            for e in evidence
            if e.row_label == label and e.headers and col in e.headers
            and e.cells and e.cells[e.headers.index(col)] not in ("", "—")
        ]  # fmt: skip
        return sorted(rows, key=lambda e: e.document != self.title_cur)

    def _cell(self, row: EvidenceItem, year: int) -> str:
        assert row.headers and row.cells
        return row.cells[row.headers.index(f"Fiscal {year}")]

    def _example(
        self,
        claim: str,
        needs: list[tuple[str, int]],
        build: Callable[[list[EvidenceItem]], JudgeOutput],
        kind: str,
        cited: str | None = None,
    ) -> Example:
        evidence = retrieve(self.index, claim, k=self.top_k)
        rows = []
        for metric, year in needs:
            found = self._rows(evidence, metric, year)
            if not found:
                what = f"{BY_KEY[metric].label.lower()} for fiscal {year}"
                target = JudgeOutput(
                    verdict="uncertain",
                    rationale=f"The retrieved passages do not report {what}.",
                )
                return Example(claim, cited, evidence, target, f"{kind}:retrieval_miss")
            rows.append(found[0])
        target = build(rows)
        if target.verdict != "uncertain" and not target.evidence:
            target = target.model_copy(
                update={"evidence": list(dict.fromkeys(r.eid for r in rows))}
            )
        return Example(claim, cited, evidence, target, kind)

    # --- generators ---------------------------------------------------------

    def money_claim(self, variant: str) -> Example | None:
        metric = self.rng.choice([m for m in PHRASES if m != "eps_diluted"])
        true, prev = self.value(metric, self.fy), self.value(metric, self.fy - 1)
        if not true or true <= 0 or not prev or prev <= 0:
            return None
        stated = true
        if variant == "misstated":
            stated = true * self.rng.choice(
                [self.rng.uniform(0.8, 0.96), self.rng.uniform(1.04, 1.2)]
            )
        elif variant == "wrong_period":
            if abs(prev - true) / true < 0.02:
                return None
            stated = prev
        if variant in ("misstated", "wrong_period") and money(stated) == money(true):
            return None
        stated_text = money(stated)
        if variant == "unit_error":
            if true < 1e9:
                return None
            stated_text = f"${true / 1e9:,.1f} million"
        phrase = self.rng.choice(PHRASES[metric])
        templates = MONEY_TEMPLATES if variant != "wrong_period" else MONEY_TEMPLATES[:3]
        template = self.rng.choice(templates)
        fill = dict(Co=self.task.company, phrase=phrase, Phrase=_cap(phrase), fy=self.fy,
                    py=self.fy - 1, pv=money(prev))  # fmt: skip
        claim = template.format(v=stated_text, **fill)
        correction = template.format(v=money(true), **fill)
        needs = [(metric, self.fy)] + ([(metric, self.fy - 1)] if "{pv}" in template else [])
        label = BY_KEY[metric].label

        def build(rows: list[EvidenceItem]) -> JudgeOutput:
            row = rows[0]
            cell = self._cell(row, self.fy)
            src = f"{row.document} reports {label.lower()} of {cell} million for fiscal {self.fy}"
            if variant == "valid":
                return JudgeOutput(
                    verdict="valid", rationale=f"{src}, which matches {stated_text}."
                )
            if variant == "wrong_period":
                why = f"{src} ({money(true)}); {stated_text} is the fiscal {self.fy - 1} figure."
            elif variant == "unit_error":
                why = (
                    f"{src}, i.e. {money(true)}; "
                    f"the claim's {stated_text} is off by a factor of 1,000."
                )
            else:
                why = f"{src}, i.e. {money(true)}, not {stated_text}."
            return JudgeOutput(
                verdict="invalid",
                error_type="misstated_figure" if variant == "misstated" else variant,  # type: ignore[arg-type]
                correction=correction,
                rationale=why,
            )

        return self._example(claim, needs, build, f"money:{variant}")

    def eps_claim(self, variant: str) -> Example | None:
        true = self.value("eps_diluted", self.fy)
        if true is None or true <= 0:
            return None
        stated = true if variant == "valid" else round(true * self.rng.uniform(1.05, 1.25), 2)
        if variant != "valid" and per_share(stated) == per_share(true):
            return None
        phrase = self.rng.choice(PHRASES["eps_diluted"])
        template = self.rng.choice(
            ["{Phrase} was {v} in fiscal {fy}.", "{Co} earned {v} of {phrase} in fiscal {fy}."]
        )
        fill = dict(Co=self.task.company, phrase=phrase, Phrase=_cap(phrase), fy=self.fy)
        claim, correction = (
            template.format(v=per_share(stated), **fill),
            template.format(v=per_share(true), **fill),
        )

        def build(rows: list[EvidenceItem]) -> JudgeOutput:
            cell = self._cell(rows[0], self.fy)
            src = f"{rows[0].document} reports diluted EPS of {cell} for fiscal {self.fy}"
            if variant == "valid":
                return JudgeOutput(verdict="valid", rationale=f"{src}, matching the claim.")
            return JudgeOutput(
                verdict="invalid",
                error_type="misstated_figure",
                correction=correction,
                rationale=f"{src}, not {per_share(stated)}.",
            )

        return self._example(claim, [("eps_diluted", self.fy)], build, f"eps:{variant}")

    def margin_claim(self, variant: str) -> Example | None:
        name = self.rng.choice(list(MARGIN_NUMERATORS))
        num_key = MARGIN_NUMERATORS[name]
        num, den = self.value(num_key, self.fy), self.value("revenue", self.fy)
        if not num or not den or num <= 0:
            return None
        true = num / den * 100
        stated = true
        if variant == "wrong":
            stated = true + self.rng.choice([-1, 1]) * self.rng.uniform(1.0, 4.0)
            if pct(stated) == pct(true):
                return None
        template = self.rng.choice(
            [
                "{Name} was {p} in fiscal {fy}.",
                "In fiscal {fy}, {co} posted a {name} of {p}.",
                "{Co}'s {name} reached {p} for fiscal {fy}.",
            ]
        )
        fill = dict(
            Name=_cap(name), name=name, Co=self.task.company, co=self.task.company, fy=self.fy
        )
        claim, correction = (
            template.format(p=pct(stated), **fill),
            template.format(p=pct(true), **fill),
        )

        def build(rows: list[EvidenceItem]) -> JudgeOutput:
            a, b = self._cell(rows[0], self.fy), self._cell(rows[1], self.fy)
            calc = f"{BY_KEY[num_key].label} of {a} divided by total revenue of {b} is {true:.2f}%"
            if variant == "valid":
                return JudgeOutput(
                    verdict="valid", rationale=f"{calc}, consistent with {pct(stated)}."
                )
            return JudgeOutput(
                verdict="invalid",
                error_type="arithmetic_error",
                correction=correction,
                rationale=f"{calc}, not {pct(stated)}.",
            )

        return self._example(
            claim, [(num_key, self.fy), ("revenue", self.fy)], build, f"margin:{variant}"
        )

    def growth_claim(self, variant: str) -> Example | None:
        metric = self.rng.choice(
            ["revenue", "net_income", "operating_income", "operating_cash_flow"]
        )
        cur, prev = self.value(metric, self.fy), self.value(metric, self.fy - 1)
        if not cur or not prev or prev <= 0 or cur <= 0:
            return None
        true = (cur - prev) / prev * 100
        if abs(true) < 0.5:
            return None
        stated, direction = abs(true), "rose" if true > 0 else "fell"
        if variant == "wrong_size":
            stated = abs(true) + self.rng.choice([-1, 1]) * self.rng.uniform(1.0, 5.0)
            if stated <= 0 or pct(stated) == pct(abs(true)):
                return None
        elif variant == "wrong_direction":
            direction = "fell" if true > 0 else "rose"
        phrase = self.rng.choice(PHRASES[metric])
        template = "{Phrase} {d} {p} in fiscal {fy} from fiscal {py}."
        fill = dict(Phrase=_cap(phrase), fy=self.fy, py=self.fy - 1)
        claim = template.format(d=direction, p=pct(stated), **fill)
        correction = template.format(d="rose" if true > 0 else "fell", p=pct(abs(true)), **fill)

        def build(rows: list[EvidenceItem]) -> JudgeOutput:
            a, b = self._cell(rows[0], self.fy), self._cell(rows[1], self.fy - 1)
            calc = (
                f"{BY_KEY[metric].label} moved from {b} in fiscal {self.fy - 1} to {a} in fiscal "
                f"{self.fy}, a change of {true:+.2f}%"
            )
            if variant == "valid":
                return JudgeOutput(verdict="valid", rationale=f"{calc}, consistent with the claim.")
            return JudgeOutput(
                verdict="invalid",
                error_type="arithmetic_error",
                correction=correction,
                rationale=f"{calc}; the claim states it {direction} {pct(stated)}.",
            )

        needs = [(metric, self.fy), (metric, self.fy - 1)]
        return self._example(claim, needs, build, f"growth:{variant}")

    def citation_claim(self, variant: str) -> Example | None:
        metric = self.rng.choice(["revenue", "net_income", "operating_income", "gross_profit"])
        year = self.fy - 1 if variant == "prior_year_valid" else self.fy
        true = self.value(metric, year)
        if not true or true <= 0:
            return None
        cited = self.title_prev if variant == "mismatch" else self.title_cur
        phrase = self.rng.choice(PHRASES[metric])
        base = f"{_cap(phrase)} was {money(true)} in fiscal {year}"
        claim = f"{base} (Source: {cited})."
        correction = f"{base} (Source: {self.title_cur})."

        def build(rows: list[EvidenceItem]) -> JudgeOutput:
            row = rows[0]
            if variant == "mismatch":
                return JudgeOutput(
                    verdict="invalid",
                    error_type="citation_mismatch",
                    correction=correction,
                    rationale=(
                        f"The fiscal {year} figure is reported in {self.title_cur}; "
                        f"{self.title_prev} covers fiscal {self.fy - 1} and {self.fy - 2}."
                    ),
                )
            return JudgeOutput(
                verdict="valid",
                rationale=(
                    f"{row.document} reports {self._cell(row, year)} million for fiscal "
                    f"{year}, and the cited filing covers that year."
                ),
            )

        ex = self._example(claim, [(metric, year)], build, f"citation:{variant}", cited=cited)
        if variant == "mismatch" and ex.target.verdict == "invalid":
            cur_rows = [
                r for r in self._rows(ex.evidence, metric, year) if r.document == self.title_cur
            ]
            if not cur_rows:
                ex.target = JudgeOutput(
                    verdict="uncertain",
                    rationale=f"The retrieved passages do not include {self.title_cur}.",
                )
                ex.kind += ":retrieval_miss"
        return ex

    def uncertain_claim(self, variant: str) -> Example | None:
        revenue = self.value("revenue", self.fy)
        if not revenue:
            return None
        if variant == "unsupported":
            template, frac = self.rng.choice(UNSUPPORTED)
            claim = template.format(
                v=money(revenue * frac * self.rng.uniform(0.5, 1.5)), fy=self.fy
            )
            evidence = retrieve(self.index, claim, k=self.top_k)
            target = JudgeOutput(
                verdict="uncertain",
                rationale="None of the provided filings' passages report this figure.",
            )
            return Example(claim, None, evidence, target, "uncertain:unsupported")
        year = self.fy - 3
        metric = self.rng.choice(["revenue", "net_income", "operating_income"])
        phrase = self.rng.choice(PHRASES[metric])
        claim = (
            f"{_cap(phrase)} was {money(revenue * self.rng.uniform(0.6, 0.95))} in fiscal {year}."
        )
        evidence = retrieve(self.index, claim, k=self.top_k)
        target = JudgeOutput(
            verdict="uncertain",
            rationale=(
                f"The provided filings cover fiscal {self.fy - 2} to {self.fy}; "
                f"none reports fiscal {year}."
            ),
        )
        return Example(claim, None, evidence, target, "uncertain:uncovered_year")


# (generator method, variant, weight)
MIX: list[tuple[str, str, float]] = [
    ("money_claim", "valid", 7),
    ("money_claim", "misstated", 3),
    ("money_claim", "wrong_period", 3),
    ("money_claim", "unit_error", 2),
    ("eps_claim", "valid", 2),
    ("eps_claim", "misstated", 1),
    ("margin_claim", "valid", 3),
    ("margin_claim", "wrong", 2),
    ("growth_claim", "valid", 2),
    ("growth_claim", "wrong_size", 1.5),
    ("growth_claim", "wrong_direction", 1),
    ("citation_claim", "valid", 1),
    ("citation_claim", "prior_year_valid", 1),
    ("citation_claim", "mismatch", 2),
    ("uncertain_claim", "unsupported", 2),
    ("uncertain_claim", "uncovered_year", 1.5),
]


def generate_for_task(bench: Path, task: Task, n: int, seed: int, top_k: int = 8) -> list[Example]:
    rng = random.Random(f"{seed}:{task.task_id}")
    gen = TaskGenerator(bench, task, rng, top_k=top_k)
    weights = [w for *_, w in MIX]
    out: list[Example] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        method, variant, _ = rng.choices(MIX, weights=weights)[0]
        ex = getattr(gen, method)(variant)
        if ex is None or ex.claim in seen:
            continue
        seen.add(ex.claim)
        ex.meta = {"task_id": task.task_id, "ticker": task.ticker, "fiscal_year": task.fiscal_year}
        out.append(ex)
    return out
