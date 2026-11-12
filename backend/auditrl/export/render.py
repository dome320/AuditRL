import json
from collections import Counter
from typing import Any

import markdown as md
from sqlmodel import Session, select

from auditrl.models import AuditRun, Claim, Document, Finding, Passage, Verdict
from auditrl.review import latest_reviews


def build_export(session: Session, audit_id: int) -> dict[str, Any]:
    run = session.get(AuditRun, audit_id)
    if run is None:
        raise ValueError(f"audit {audit_id} not found")
    docs = {d.id: d for d in session.exec(select(Document).where(Document.audit_id == audit_id))}
    claims = {c.id: c for c in session.exec(select(Claim).where(Claim.audit_id == audit_id))}
    findings = session.exec(
        select(Finding).where(Finding.audit_id == audit_id).order_by(Finding.claim_id)
    ).all()
    passage_ids = {pid for f in findings for pid in f.evidence_passage_ids}
    passages = {
        p.id: p
        for p in session.exec(select(Passage).where(Passage.id.in_(passage_ids)))  # type: ignore[union-attr]
    }
    reviews = latest_reviews(session, [f.id for f in findings])

    records = []
    for f in findings:
        claim = claims[f.claim_id]
        review = reviews.get(f.id)
        final_verdict = f.verdict
        final_correction = f.correction
        if review and review.action == "edit":
            final_verdict = review.edited_verdict or f.verdict
            final_correction = review.edited_correction or f.correction
        evidence = []
        for pid, quote in zip(f.evidence_passage_ids, f.evidence_quotes, strict=False):
            p = passages.get(pid)
            doc = docs.get(p.document_id) if p else None
            evidence.append(
                {
                    "passage_id": pid,
                    "document": doc.name if doc else None,
                    "document_id": doc.id if doc else None,
                    "page": p.page if p else None,
                    "char_start": p.char_start if p else None,
                    "char_end": p.char_end if p else None,
                    "quote": quote,
                }
            )
        records.append(
            {
                "audit_id": audit_id,
                "finding_id": f.id,
                "claim_id": claim.id,
                "claim_text": claim.text,
                "claim_type": claim.claim_type,
                "report_span": [claim.span_start, claim.span_end],
                "cited_source": claim.cited_source,
                "verdict": f.verdict,
                "error_type": f.error_type,
                "rationale": f.rationale,
                "correction": f.correction,
                "confidence": f.confidence,
                "verifier": f.verifier,
                "evidence": evidence,
                "checks": f.checks,
                "review": review.model_dump(mode="json") if review else None,
                "final_verdict": final_verdict,
                "final_correction": final_correction,
            }
        )
    unchecked = [
        {"claim_id": c.id, "claim_text": c.text, "claim_type": c.claim_type}
        for c in claims.values()
        if not c.checkable
    ]
    return {
        "audit": run.model_dump(mode="json"),
        "documents": [
            {"id": d.id, "name": d.name, "kind": d.kind, "origin": d.origin, "sha256": d.sha256}
            for d in docs.values()
        ],
        "findings": records,
        "unchecked_claims": unchecked,
    }


def to_json(export: dict[str, Any]) -> str:
    return json.dumps(export, indent=2, default=str)


def to_jsonl(export: dict[str, Any]) -> str:
    return "".join(json.dumps(r, default=str) + "\n" for r in export["findings"])


def to_markdown(export: dict[str, Any]) -> str:
    run = export["audit"]
    findings = export["findings"]
    verdicts = Counter(r["final_verdict"] for r in findings)
    errors = Counter(r["error_type"] for r in findings if r["final_verdict"] == Verdict.invalid)
    report = next((d for d in export["documents"] if d["kind"] == "report"), None)
    sources = [d["name"] for d in export["documents"] if d["kind"] == "source"]
    reviewed = sum(1 for r in findings if r["review"])

    lines = [
        f"# Audit report #{run['id']}",
        "",
        f"- **Report:** {report['name'] if report else 'n/a'}",
        f"- **Sources:** {', '.join(sources) or 'none'}",
        f"- **Model:** `{run['model']}` "
        f"(prompts {run['prompt_version']}, config {run['config_hash']})",
        f"- **Status:** {run['status']}"
        + (f" in {run['latency_s']}s" if run.get("latency_s") is not None else "")
        + (f", ${run['cost_usd']:.4f}" if run.get("cost_usd") else ""),
        f"- **Claims checked:** {len(findings)} ({reviewed} human-reviewed); "
        f"{len(export['unchecked_claims'])} not checkable",
        "",
        "## Summary",
        "",
        "| Verdict | Count |",
        "|---|---|",
        *(f"| {v} | {verdicts.get(v, 0)} |" for v in ("invalid", "uncertain", "valid")),
    ]
    if errors:
        lines += ["", "| Error type | Count |", "|---|---|"]
        lines += [f"| {e} | {n} |" for e, n in errors.most_common()]

    def section(title: str, verdict: str) -> list[str]:
        rows = [r for r in findings if r["final_verdict"] == verdict]
        if not rows:
            return []
        out = ["", f"## {title}", ""]
        for i, r in enumerate(rows, 1):
            out.append(f"### {i}. {r['error_type'] or verdict}")
            out.append("")
            out.append(f"> {r['claim_text']}")
            out.append("")
            out.append(f"**Why:** {r['rationale']}")
            if r["final_correction"]:
                out.append("")
                out.append(f"**Suggested correction:** {r['final_correction']}")
            for e in r["evidence"]:
                loc = f", p. {e['page']}" if e["page"] else ""
                out.append("")
                out.append(f'- Evidence ({e["document"]}{loc}): "{e["quote"]}"')
            if r["review"]:
                rv = r["review"]
                out.append("")
                out.append(
                    f"*Reviewed by {rv['reviewer']}: {rv['action']}*"
                    + (f" - {rv['notes']}" if rv.get("notes") else "")
                )
            out.append("")
        return out

    lines += section("Errors", "invalid")
    lines += section("Needs human review (uncertain)", "uncertain")
    valid = [r for r in findings if r["final_verdict"] == "valid"]
    if valid:
        lines += ["", "## Verified claims", ""]
        lines += [f"- {r['claim_text']}" for r in valid]
    return "\n".join(lines).rstrip() + "\n"


def to_html(export: dict[str, Any]) -> str:
    body = md.markdown(to_markdown(export), extensions=["tables"])
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Audit #{export['audit']['id']}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.5;color:#1f2328}table{border-collapse:collapse}"
        "td,th{border:1px solid #d0d7de;padding:4px 10px}blockquote{margin:0;padding:.4rem 1rem;"
        "border-left:4px solid #d0d7de;background:#f6f8fa}</style></head><body>"
        f"{body}</body></html>"
    )
