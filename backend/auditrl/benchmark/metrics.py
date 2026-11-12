"""Financial metrics the benchmark covers: XBRL concepts, display labels, and report phrasing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    concepts: tuple[str, ...]
    phrases: tuple[str, ...]
    per_share: bool = False


METRICS: tuple[Metric, ...] = (
    Metric(
        "revenue",
        "Total revenue",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "RevenuesNetOfInterestExpense",
        ),
        (
            "total net sales",
            "net sales",
            "total revenues",
            "total revenue",
            "net revenues",
            "net revenue",
            "revenues",
            "revenue",
            "sales",
            "top line",
        ),
    ),
    Metric(
        "cost_of_revenue",
        "Cost of revenue",
        ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
        ("cost of revenue", "cost of revenues", "cost of sales", "cost of goods sold", "cogs"),
    ),
    Metric(
        "gross_profit",
        "Gross profit",
        ("GrossProfit",),
        ("gross profit", "gross margin"),
    ),
    Metric(
        "rnd",
        "Research and development",
        ("ResearchAndDevelopmentExpense",),
        ("research and development", "r&d"),
    ),
    Metric(
        "sga",
        "Selling, general and administrative",
        ("SellingGeneralAndAdministrativeExpense",),
        ("selling, general and administrative", "sg&a"),
    ),
    Metric(
        "operating_income",
        "Operating income",
        ("OperatingIncomeLoss",),
        ("operating income", "income from operations", "operating profit"),
    ),
    Metric(
        "net_income",
        "Net income",
        ("NetIncomeLoss", "ProfitLoss"),
        ("net income", "net earnings", "net profit", "net loss"),
    ),
    Metric(
        "eps_diluted",
        "Diluted earnings per share",
        ("EarningsPerShareDiluted",),
        ("diluted earnings per share", "diluted eps", "earnings per share", "eps"),
        per_share=True,
    ),
    Metric(
        "operating_cash_flow",
        "Net cash from operating activities",
        ("NetCashProvidedByUsedInOperatingActivities",),
        (
            "cash provided by operating activities",
            "cash from operating activities",
            "operating cash flow",
            "cash from operations",
        ),
    ),
    Metric(
        "capex",
        "Capital expenditures",
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
            "PaymentsToAcquireOtherProductiveAssets",
        ),
        ("capital expenditures", "capital expenditure", "capex"),
    ),
)

BY_KEY = {m.key: m for m in METRICS}

# Percent metrics a report may state, as (numerator, denominator) over the same fiscal year.
MARGINS: dict[str, tuple[str, str]] = {
    "gross margin": ("gross_profit", "revenue"),
    "operating margin": ("operating_income", "revenue"),
    "net margin": ("net_income", "revenue"),
    "net profit margin": ("net_income", "revenue"),
    "profit margin": ("net_income", "revenue"),
}
