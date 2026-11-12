"""End-to-end through the API using the real LiteLLM client code path with recorded responses."""

import json
import re

from fastapi.testclient import TestClient

from auditrl.api.app import create_app
from auditrl.evaluation import evaluate_dataset
from auditrl.llm import HeuristicClient, LiteLLMClient
from tests.conftest import APPLE_10K, REPO

REPORT = (
    "# Apple review\n\n"
    "Services net sales were $78.1 billion in fiscal 2023. "
    "Services grew 12.0% from $78.1 billion in fiscal 2022 to $85.2 billion in fiscal 2023.\n\n"
    "We expect a strong year."
)

EXTRACTION = {
    "claims": [
        {
            "text": "Services net sales were $78.1 billion in fiscal 2023.",
            "claim_type": "figure",
            "checkable": True,
            "figures": [
                {
                    "value_text": "$78.1 billion",
                    "metric": "Services net sales",
                    "period": "fiscal 2023",
                }
            ],
        },
        {
            # Paraphrased whitespace: alignment must still find the span.
            "text": "Services grew 12.0%  from $78.1 billion in fiscal 2022 to $85.2 billion in fiscal 2023.",
            "claim_type": "arithmetic",
            "checkable": True,
            "figures": [
                {"value_text": "12.0%", "metric": "Services growth", "role": "result"},
                {
                    "value_text": "$78.1 billion",
                    "metric": "Services net sales",
                    "period": "fiscal 2022",
                    "role": "input",
                },
                {
                    "value_text": "$85.2 billion",
                    "metric": "Services net sales",
                    "period": "fiscal 2023",
                    "role": "input",
                },
            ],
        },
        {"text": "We expect a strong year.", "claim_type": "qualitative", "checkable": False},
        {
            "text": "This sentence is not in the report at all.",
            "claim_type": "figure",
            "checkable": True,
        },
    ]
}


class RecordedTransport:
    """Replays canned model output per prompt; `locate_evidence` picks the row by label."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.first_extract = True

    def __call__(self, prompt: str, messages: list[dict[str, str]]) -> tuple[str, int, int, float]:
        self.calls.append(prompt)
        user = messages[1]["content"]
        if prompt == "extract_claims":
            if self.first_extract:
                self.first_extract = False
                return "Sorry, here you go: not json", 10, 5, 0.001
            return f"```json\n{json.dumps(EXTRACTION)}\n```", 100, 50, 0.002
        if prompt == "locate_evidence":
            metric = re.search(r"metric: (.+?) \|", user).group(1)
            label = "Services net sales" if "Services" in metric else metric
            for m in re.finditer(r"^\[(\d+)\] \([^)]*\) (.+)$", user, re.M):
                if m.group(2).startswith(label):
                    col = 0 if "2023" in user.split("Figure:")[1].split("\n")[0] else 1
                    cell = m.group(2).split(" | ")[1 + col].split(": ")[1]
                    out = {
                        "found": True,
                        "passage_id": int(m.group(1)),
                        "quote": m.group(2),
                        "value_text": cell,
                    }
                    return json.dumps(out), 50, 20, 0.001
            return json.dumps({"found": False}), 50, 5, 0.001
        if prompt == "propose_computation":
            return (
                json.dumps(
                    {
                        "operation": "growth",
                        "expression": "(b - a) / a * 100",
                        "operands": [
                            {"name": "a", "value_text": "$78.1 billion", "source": "claim"},
                            {"name": "b", "value_text": "$85.2 billion", "source": "claim"},
                        ],
                        "result_value_text": "12.0%",
                    }
                ),
                80,
                30,
                0.001,
            )
        if prompt == "write_rationale":
            return json.dumps({"rationale": "Checked against the filing."}), 40, 10, 0.0005
        raise AssertionError(prompt)


def test_api_end_to_end_with_recorded_llm(engine, settings):
    transport = RecordedTransport()
    app = create_app(engine, lambda model: LiteLLMClient(model, transport=transport), settings)
    client = TestClient(app)

    r = client.post(
        "/audits",
        data={"report_text": REPORT, "model": "recorded/model"},
        files=[("sources", ("Apple_10-K_FY2023.md", APPLE_10K.encode(), "text/markdown"))],
    )
    assert r.status_code == 202
    audit_id = r.json()["id"]

    detail = client.get(f"/audits/{audit_id}").json()
    assert detail["audit"]["status"] == "completed", detail["audit"]["error"]
    assert detail["audit"]["cost_usd"] > 0
    claims = detail["claims"]
    assert len(claims) == 3  # the unalignable claim is dropped
    text = detail["report"]["text"]
    for c in claims:
        assert text[c["span"][0] : c["span"][1]] == c["text"]

    findings = {f["claim_text"][:10]: f for f in client.get(f"/audits/{audit_id}/findings").json()}
    period = findings["Services n"]
    assert period["verdict"] == "invalid" and period["error_type"] == "wrong_period"
    assert period["correction"] == "Services net sales were $85.2 billion in fiscal 2023."
    assert period["evidence"][0]["document"] == "Apple_10-K_FY2023.md"
    growth = findings["Services g"]
    assert growth["error_type"] == "arithmetic_error"
    assert "9.1%" in growth["correction"]
    assert transport.calls.count("extract_claims") == 2  # one retry after invalid JSON

    fid = period["finding_id"]
    r = client.post(
        f"/findings/{fid}/review",
        json={"action": "edit", "edited_verdict": "uncertain", "notes": "check 10-Q"},
    )
    assert r.status_code == 200
    assert (
        client.post(
            f"/findings/{fid}/review", json={"action": "accept", "edited_correction": "x"}
        ).status_code
        == 422
    )
    assert client.post("/findings/9999/review", json={"action": "accept"}).status_code == 404

    feedback = (settings.data_dir / "feedback.jsonl").read_text().splitlines()
    assert json.loads(feedback[-1])["review"]["notes"] == "check 10-Q"

    md = client.get(f"/audits/{audit_id}/export?format=md").text
    assert "Needs human review" in md and "check 10-Q" in md
    jsonl = client.get(f"/audits/{audit_id}/export?format=jsonl").text.strip().splitlines()
    assert len(jsonl) == 2
    assert {json.loads(line)["final_verdict"] for line in jsonl} == {"uncertain", "invalid"}
    assert (
        client.get(f"/audits/{audit_id}/export?format=html")
        .headers["content-type"]
        .startswith("text/html")
    )
    assert client.get(f"/audits/{audit_id}/export?format=xml").status_code == 422


def test_api_rejects_missing_inputs(engine, settings):
    client = TestClient(create_app(engine, lambda m: HeuristicClient(), settings))
    assert client.post("/audits", data={"report_text": "x"}).status_code == 422
    files = [("sources", ("a.pptx", b"x", "application/octet-stream"))]
    assert client.post("/audits", data={"report_text": "x"}, files=files).status_code == 415


def test_curated_dataset_regression():
    result = evaluate_dataset(REPO / "data" / "curated", llm=HeuristicClient())
    s = result["summary"]
    assert s["cases"] == 10
    assert s["recall"] >= 0.9
    assert s["precision"] >= 0.9
    assert s["false_positive_rate"] == 0.0
    assert s["evidence_support"] == 1.0
