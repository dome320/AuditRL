import pytest

from auditrl.intake import UnsupportedFormat, parse_bytes, parse_text
from tests.conftest import APPLE_10K


def test_markdown_tables_become_rows_with_scale_and_offsets():
    doc = parse_text("apple.md", APPLE_10K)
    rows = [p for p in doc.passages if p.table_id is not None]
    assert [r.row_label for r in rows] == [
        "Services net sales",
        "Total net sales",
        "Operating income",
        "Net income (loss)",
    ]
    total = rows[1]
    assert total.headers == ["Fiscal 2023", "Fiscal 2022"]
    assert total.cells == ["$ 383,285", "$ 394,328"]
    assert total.unit_scale == 1e6
    for p in doc.passages:
        assert doc.text[p.char_start : p.char_end] == p.text


def test_sec_html_cells_are_merged():
    html = (
        b"<html><body><p>Amounts in millions</p><table>"
        b"<tr><td></td><td>2023</td><td>2022</td></tr>"
        b"<tr><td>Revenue</td><td>$</td><td>383,285</td><td>$</td><td>394,328</td></tr>"
        b"<tr><td>Net loss</td><td>(12</td><td>)</td><td>5</td></tr>"
        b"</table></body></html>"
    )
    doc = parse_bytes("filing.html", html)
    rows = [p for p in doc.passages if p.table_id is not None]
    assert rows[0].headers == ["2023", "2022"]
    assert rows[0].cells == ["$383,285", "$394,328"]
    assert rows[1].cells == ["(12)", "5"]
    assert rows[0].unit_scale == 1e6


def test_csv_table():
    doc = parse_bytes("income (in thousands).csv", b"Line,2023,2022\nRevenue,10,9\n")
    row = doc.passages[0]
    assert row.row_label == "Revenue" and row.cells == ["10", "9"]
    assert row.unit_scale == 1e3


def test_unsupported_format():
    with pytest.raises(UnsupportedFormat):
        parse_bytes("deck.pptx", b"")


def test_sha256_is_recorded():
    doc = parse_text("a.md", "hello")
    assert len(doc.sha256) == 64
