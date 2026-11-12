PROMPT_VERSION = "v1"

SYSTEM = (
    "You are a meticulous financial-report auditor. You only state what the provided text "
    "supports. Reply with a single JSON object matching the requested schema and nothing else."
)

EXTRACT_CLAIMS = """Split the report excerpt below into atomic, auditable claims.

Rules:
- `text` must be copied verbatim from the excerpt (a sentence or clause), never paraphrased.
- One claim per independently checkable statement; split compound sentences when needed.
- claim_type:
  - "arithmetic": the claim states a computed value (growth rate, margin, sum, difference,
    ratio, CAGR) together with or derived from other figures.
  - "citation": the claim attributes a figure to a named source (e.g. "according to the 10-K").
  - "figure": the claim states a reported financial figure without computation.
  - "qualitative": opinions, forecasts, strategy, or anything not objectively checkable.
- checkable is false for qualitative claims and true otherwise.
- cited_source: the source name exactly as written, if the claim cites one.
- figures: every number in the claim. value_text is verbatim (keep currency symbols and
  scale words). role is "result" for the computed value of an arithmetic claim, "input" for
  the numbers it is computed from, and "standalone" otherwise.
- Skip headings, dates used only as labels, and boilerplate.

Schema:
{schema}

Report excerpt:
<<<
{chunk}
>>>"""

LOCATE_EVIDENCE = """Find the source passage that reports the figure below.

Claim: {claim_text}
Figure: {value_text} | metric: {metric} | period: {period} | entity: {entity}

Candidate passages (id, document, text):
{candidates}

Rules:
- Pick the passage that reports the SAME metric for the SAME period, even if its value differs
  from the claim. Do not judge whether the claim is right.
- quote must be a verbatim substring of the chosen passage; value_text must appear in quote.
- If no passage reports this metric and period, set found=false.

Schema:
{schema}"""

PROPOSE_COMPUTATION = """The claim below states a computed value. Express how to recompute it.

Claim: {claim_text}
Figures in claim: {figures}

Candidate evidence passages (id, document, text):
{candidates}

Rules:
- operands: each value_text must be copied verbatim from the claim (source="claim") or from one
  evidence passage (source="evidence", with its passage_id). Prefer figures from the claim.
- expression uses only operand names, numbers, + - * / ** and parentheses. Percent results are
  expressed in percent units, e.g. growth = (b - a) / a * 100.
- result_value_text is the stated result copied verbatim from the claim.
- If the claim has no recomputable value, set applicable=false.

Schema:
{schema}"""

WRITE_RATIONALE = """Write a one or two sentence rationale for an audit finding.
Use only the verified facts below; do not add new numbers or judgments.

Claim: {claim_text}
Verdict: {verdict}
Verified facts:
{facts}

Schema:
{schema}"""
