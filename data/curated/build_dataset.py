"""Generate the curated Phase 1 dataset.

Each case is an excerpt of a company's income statement (transcribed from its 10-K, in
millions) plus a short research write-up with injected, labeled errors. Run:

    python data/curated/build_dataset.py

Figures were transcribed by hand and should be spot-checked against EDGAR before this set
is used as ground truth for anything beyond smoke testing.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def table(title: str, periods: list[str], rows: list[tuple[str, ...]]) -> str:
    head = f"| | {' | '.join(periods)} |\n|---|{'---|' * len(periods)}\n"
    body = "".join(f"| {' | '.join(r)} |\n" for r in rows)
    return f"# {title}\n\nConsolidated Statements of Operations (in millions)\n\n{head}{body}"


def ok(quote: str) -> dict:
    return {"quote": quote, "verdict": "valid"}


def err(quote: str, error_type: str, correction_contains: str) -> dict:
    return {
        "quote": quote,
        "verdict": "invalid",
        "error_type": error_type,
        "correction_contains": correction_contains,
    }


CASES: list[dict] = [
    {
        "id": "apple_fy2023",
        "company": "Apple Inc.",
        "sources": {
            "Apple_10-K_FY2023.md": table(
                "Apple Inc. Form 10-K Fiscal 2023",
                ["Fiscal 2023", "Fiscal 2022"],
                [
                    ("Products net sales", "$ 298,085", "$ 316,199"),
                    ("Services net sales", "85,200", "78,129"),
                    ("Total net sales", "383,285", "394,328"),
                    ("Gross margin", "169,148", "170,782"),
                    ("Research and development", "29,915", "26,251"),
                    ("Operating income", "114,301", "119,437"),
                    ("Net income", "$ 96,995", "$ 99,803"),
                ],
            )
        },
        "claims": [
            ok("Apple reported total net sales of $383.3 billion in fiscal 2023 (Source: Apple 10-K FY2023)."),
            ok("Total net sales declined 2.8% from $394.3 billion in fiscal 2022 to $383.3 billion in fiscal 2023."),
            err(
                "Services net sales grew 11.2% from $78.1 billion in fiscal 2022 to $85.2 billion in fiscal 2023.",
                "arithmetic_error",
                "9.1%",
            ),
            ok("Gross margin was $169.1 billion in fiscal 2023."),
            err(
                "Research and development expense was $26.3 billion in fiscal 2023.",
                "wrong_period",
                "$29.9 billion",
            ),
            ok("Net income was $97.0 billion in fiscal 2023."),
        ],
        "filler": ["Management expects services to keep outgrowing hardware."],
    },
    {
        "id": "microsoft_fy2023",
        "company": "Microsoft Corporation",
        "sources": {
            "Microsoft_10-K_FY2023.md": table(
                "Microsoft Corporation Form 10-K Fiscal 2023",
                ["Fiscal 2023", "Fiscal 2022"],
                [
                    ("Total revenue", "$ 211,915", "$ 198,270"),
                    ("Gross margin", "146,052", "135,620"),
                    ("Operating income", "88,523", "83,383"),
                    ("Net income", "$ 72,361", "$ 72,738"),
                ],
            )
        },
        "claims": [
            ok("Microsoft's total revenue rose 6.9% from $198.3 billion in fiscal 2022 to $211.9 billion in fiscal 2023."),
            ok("Operating income reached $88.5 billion in fiscal 2023, an operating margin of 41.8% on revenue of $211.9 billion."),
            ok("Net income was $72.4 billion in fiscal 2023 (Source: Microsoft 10-K FY2023)."),
            err("Gross margin was $164.1 billion in fiscal 2023.", "misstated_figure", "$146.1 billion"),
            err("Operating income was $83.4 billion in fiscal 2023.", "wrong_period", "$88.5 billion"),
        ],
        "filler": ["Cloud demand remains the key driver of the investment thesis."],
    },
    {
        "id": "nvidia_fy2024",
        "company": "NVIDIA Corporation",
        "sources": {
            "NVIDIA_10-K_FY2024.md": table(
                "NVIDIA Corporation Form 10-K Fiscal 2024",
                ["Fiscal 2024", "Fiscal 2023"],
                [
                    ("Revenue", "$ 60,922", "$ 26,974"),
                    ("Data Center revenue", "47,525", "15,005"),
                    ("Gross profit", "44,301", "15,356"),
                    ("Operating income", "32,972", "4,224"),
                    ("Net income", "$ 29,760", "$ 4,368"),
                ],
            )
        },
        "claims": [
            ok("NVIDIA's revenue grew 126% from $27.0 billion in fiscal 2023 to $60.9 billion in fiscal 2024."),
            ok("Gross margin was 72.7% in fiscal 2024, on gross profit of $44.3 billion and revenue of $60.9 billion."),
            ok("Data Center revenue was $47.5 billion in fiscal 2024."),
            err("Net income rose to $29.8 million in fiscal 2024.", "unit_error", "29,760"),
            err("Operating income was $32.0 billion in fiscal 2024.", "misstated_figure", "$33.0 billion"),
        ],
        "filler": ["We believe supply constraints could ease over the coming year."],
    },
    {
        "id": "alphabet_2023",
        "company": "Alphabet Inc.",
        "sources": {
            "Alphabet_10-K_2023.md": table(
                "Alphabet Inc. Form 10-K 2023",
                ["2023", "2022"],
                [
                    ("Revenues", "$ 307,394", "$ 282,836"),
                    ("Operating income", "84,293", "74,842"),
                    ("Net income", "$ 73,795", "$ 59,972"),
                ],
            ),
            "Alphabet_10-K_2022.md": table(
                "Alphabet Inc. Form 10-K 2022",
                ["2022", "2021"],
                [
                    ("Revenues", "$ 282,836", "$ 257,637"),
                    ("Net income", "$ 59,972", "$ 76,033"),
                ],
            ),
        },
        "claims": [
            ok("Alphabet's revenues increased 8.7% from $282.8 billion in 2022 to $307.4 billion in 2023."),
            ok("Operating margin expanded to 27.4% in 2023, with operating income of $84.3 billion on revenues of $307.4 billion."),
            err(
                "Net income grew 18.9% from $60.0 billion in 2022 to $73.8 billion in 2023.",
                "arithmetic_error",
                "23.0%",
            ),
            err(
                "Net income was $73.8 billion in 2023 (Source: Alphabet 10-K 2022).",
                "citation_mismatch",
                "Alphabet 10-K 2023",
            ),
            ok("Revenues were $282.8 billion in 2022 (Source: Alphabet 10-K 2022)."),
        ],
        "filler": ["Search remains resilient despite new competition."],
    },
    {
        "id": "amazon_2023",
        "company": "Amazon.com, Inc.",
        "sources": {
            "Amazon_10-K_2023.md": table(
                "Amazon.com, Inc. Form 10-K 2023",
                ["2023", "2022"],
                [
                    ("Total net sales", "$ 574,785", "$ 513,983"),
                    ("Operating income", "36,852", "12,248"),
                    ("Net income (loss)", "$ 30,425", "$ (2,722)"),
                ],
            )
        },
        "claims": [
            ok("Amazon's total net sales rose 11.8% from $514.0 billion in 2022 to $574.8 billion in 2023."),
            ok("Operating income was $36.9 billion in 2023, up from $12.2 billion in 2022."),
            ok("Amazon posted net income of $30.4 billion in 2023 after a net loss of $2.7 billion in 2022."),
            err(
                "Operating margin was 7.4% in 2023, based on operating income of $36.9 billion and total net sales of $574.8 billion.",
                "arithmetic_error",
                "6.4%",
            ),
            ok("Total net sales were $574.8 billion in 2023 (Source: Amazon 10-K 2023)."),
        ],
        "filler": ["Margin expansion should continue as fulfillment costs normalize."],
    },
    {
        "id": "meta_2023",
        "company": "Meta Platforms, Inc.",
        "sources": {
            "Meta_10-K_2023.md": table(
                "Meta Platforms, Inc. Form 10-K 2023",
                ["2023", "2022"],
                [
                    ("Revenue", "$ 134,902", "$ 116,609"),
                    ("Costs and expenses", "88,151", "87,665"),
                    ("Income from operations", "46,751", "28,944"),
                    ("Net income", "$ 39,098", "$ 23,200"),
                ],
            )
        },
        "claims": [
            ok("Meta's revenue grew 15.7% from $116.6 billion in 2022 to $134.9 billion in 2023."),
            ok("Costs and expenses totaled $88.2 billion in 2023."),
            ok("Income from operations was $46.8 billion in 2023, a margin of 34.7% on revenue of $134.9 billion."),
            err("Net income was $23.2 billion in 2023.", "wrong_period", "$39.1 billion"),
            err("Revenue was $143.9 billion in 2023.", "misstated_figure", "$134.9 billion"),
        ],
        "filler": ["The year of efficiency reshaped the cost base."],
    },
    {
        "id": "tesla_2023",
        "company": "Tesla, Inc.",
        "sources": {
            "Tesla_10-K_2023.md": table(
                "Tesla, Inc. Form 10-K 2023",
                ["2023", "2022"],
                [
                    ("Total revenues", "$ 96,773", "$ 81,462"),
                    ("Income from operations", "8,891", "13,656"),
                    ("Net income attributable to common stockholders", "$ 14,997", "$ 12,556"),
                ],
            )
        },
        "claims": [
            ok("Tesla's total revenues increased 18.8% from $81.5 billion in 2022 to $96.8 billion in 2023."),
            ok("Income from operations fell 34.9% from $13.7 billion in 2022 to $8.9 billion in 2023."),
            err(
                "Operating margin was 12.2% in 2023, based on income from operations of $8.9 billion and total revenues of $96.8 billion.",
                "arithmetic_error",
                "9.2%",
            ),
            ok("Net income attributable to common stockholders was $15.0 billion in 2023."),
        ],
        "filler": ["Price cuts may weigh on automotive margins."],
    },
    {
        "id": "cocacola_2023",
        "company": "The Coca-Cola Company",
        "sources": {
            "Coca-Cola_10-K_2023.md": table(
                "The Coca-Cola Company Form 10-K 2023",
                ["2023", "2022"],
                [
                    ("Net operating revenues", "$ 45,754", "$ 43,004"),
                    ("Operating income", "11,311", "10,909"),
                    ("Net income attributable to shareowners", "$ 10,714", "$ 9,542"),
                ],
            ),
            "Coca-Cola_10-K_2022.md": table(
                "The Coca-Cola Company Form 10-K 2022",
                ["2022", "2021"],
                [
                    ("Net operating revenues", "$ 43,004", "$ 38,655"),
                    ("Net income attributable to shareowners", "$ 9,542", "$ 9,771"),
                ],
            ),
        },
        "claims": [
            ok("Coca-Cola's net operating revenues grew 6.4% from $43.0 billion in 2022 to $45.8 billion in 2023."),
            ok("Operating income was $11.3 billion in 2023, an operating margin of 24.7% on net operating revenues of $45.8 billion."),
            err(
                "Net income attributable to shareowners was $10.7 billion in 2023 (Source: Coca-Cola 10-K 2022).",
                "citation_mismatch",
                "Coca-Cola 10-K 2023",
            ),
            ok("Operating income rose 3.7% from $10.9 billion in 2022 to $11.3 billion in 2023."),
            err("Net operating revenues were $43.0 billion in 2023.", "wrong_period", "$45.8 billion"),
        ],
        "filler": ["Pricing power has offset volume softness."],
    },
    {
        "id": "jpmorgan_2023",
        "company": "JPMorgan Chase & Co.",
        "sources": {
            "JPMorgan_10-K_2023.md": table(
                "JPMorgan Chase & Co. Form 10-K 2023",
                ["2023", "2022"],
                [
                    ("Total net revenue", "$ 158,104", "$ 128,695"),
                    ("Noninterest expense", "87,172", "76,140"),
                    ("Net income", "$ 49,552", "$ 37,676"),
                ],
            )
        },
        "claims": [
            ok("JPMorgan's net income rose 31.5% from $37.7 billion in 2022 to $49.6 billion in 2023."),
            ok("Total net revenue was $158.1 billion in 2023."),
            err(
                "Noninterest expense was $87.2 billion in 2023, equal to 61.3% of total net revenue of $158.1 billion.",
                "arithmetic_error",
                "55.2%",
            ),
            ok("Total net revenue increased 22.9% from $128.7 billion in 2022 to $158.1 billion in 2023."),
            err("Net income was $49,552 billion in 2023.", "unit_error", "49.55"),
        ],
        "filler": ["Net interest income is likely to peak as rates fall."],
    },
    {
        "id": "walmart_fy2024",
        "company": "Walmart Inc.",
        "sources": {
            "Walmart_10-K_FY2024.md": table(
                "Walmart Inc. Form 10-K Fiscal 2024",
                ["Fiscal 2024", "Fiscal 2023"],
                [
                    ("Total revenues", "$ 648,125", "$ 611,289"),
                    ("Operating income", "27,012", "20,428"),
                    ("Consolidated net income attributable to Walmart", "$ 15,511", "$ 11,680"),
                ],
            )
        },
        "claims": [
            ok("Walmart's total revenues increased 6.0% from $611.3 billion in fiscal 2023 to $648.1 billion in fiscal 2024."),
            ok("Operating income grew 32.2% from $20.4 billion in fiscal 2023 to $27.0 billion in fiscal 2024."),
            ok("Consolidated net income attributable to Walmart was $15.5 billion in fiscal 2024."),
            err(
                "Operating margin was 5.2% in fiscal 2024, based on operating income of $27.0 billion on total revenues of $648.1 billion.",
                "arithmetic_error",
                "4.2%",
            ),
            err(
                "Total revenues were $661.1 billion in fiscal 2024 (Source: Walmart 10-K FY2024).",
                "misstated_figure",
                "$648.1 billion",
            ),
        ],
        "filler": ["E-commerce profitability is an underappreciated catalyst."],
    },
]


def build_report(case: dict) -> str:
    claims = [c["quote"] for c in case["claims"]]
    mid = len(claims) // 2
    paras = [
        f"# {case['company']}: results review",
        " ".join(claims[:mid]),
        " ".join(claims[mid:] + case["filler"]),
    ]
    return "\n\n".join(paras) + "\n"


def main() -> None:
    for case in CASES:
        case_dir = ROOT / case["id"]
        (case_dir / "sources").mkdir(parents=True, exist_ok=True)
        for name, text in case["sources"].items():
            (case_dir / "sources" / name).write_text(text)
        (case_dir / "report.md").write_text(build_report(case))
        expected = {"case_id": case["id"], "company": case["company"], "labels": case["claims"]}
        (case_dir / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {len(CASES)} cases to {ROOT}")


if __name__ == "__main__":
    main()
