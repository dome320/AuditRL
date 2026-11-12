"""Turn uploaded files into plain text plus locatable passages (paragraphs and table rows)."""

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from auditrl.verify.normalize import detect_scale


@dataclass
class ParsedPassage:
    char_start: int
    char_end: int
    text: str
    page: int | None = None
    table_id: int | None = None
    row_index: int | None = None
    row_label: str | None = None
    headers: list[str] | None = None
    cells: list[str] | None = None
    unit_scale: float = 1.0
    caption: str | None = None


@dataclass
class ParsedDocument:
    name: str
    content_type: str
    sha256: str
    text: str = ""
    passages: list[ParsedPassage] = field(default_factory=list)


class UnsupportedFormat(ValueError):
    pass


class _Builder:
    """Accumulates document text while recording the character offsets of each passage."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.pos = 0
        self.passages: list[ParsedPassage] = []
        self.table_count = 0
        self.recent_text = ""

    def add_paragraph(self, text: str, page: int | None = None) -> None:
        text = text.strip()
        if not text:
            return
        start = self._append(text)
        self.passages.append(ParsedPassage(start, start + len(text), text, page=page))
        self.recent_text = (self.recent_text + " " + text)[-400:]

    def add_table(
        self,
        rows: list[list[str]],
        page: int | None = None,
        caption: str | None = None,
    ) -> None:
        rows = [[c.strip() for c in r] for r in rows if any(c.strip() for c in r)]
        if len(rows) < 2:
            for r in rows:
                self.add_paragraph(" | ".join(r), page)
            return
        header_row, body = rows[0], rows[1:]
        scale = (
            detect_scale(" ".join(header_row))
            or detect_scale(caption or "")
            or detect_scale(self.recent_text)
            or 1.0
        )
        table_id = self.table_count
        self.table_count += 1
        headers = header_row[1:]
        for i, row in enumerate(body):
            label, cells = row[0], row[1:]
            if not label and not any(cells):
                continue
            pairs = [f"{h}: {c}" if h else c for h, c in zip(headers, cells, strict=False) if c]
            text = " | ".join([label, *pairs]) if pairs else label
            start = self._append(text)
            self.passages.append(
                ParsedPassage(
                    start,
                    start + len(text),
                    text,
                    page=page,
                    table_id=table_id,
                    row_index=i,
                    row_label=label,
                    headers=headers,
                    cells=cells,
                    unit_scale=scale,
                    caption=caption,
                )
            )

    def _append(self, text: str) -> int:
        if self.parts:
            self.parts.append("\n\n")
            self.pos += 2
        start = self.pos
        self.parts.append(text)
        self.pos += len(text)
        return start

    def text(self) -> str:
        return "".join(self.parts)


_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_md_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _parse_markdown(raw: str, b: _Builder) -> None:
    """Paragraphs separated by blank lines; GitHub pipe tables become table rows."""
    lines = raw.splitlines()
    para: list[str] = []
    i = 0

    def flush() -> None:
        if para:
            b.add_paragraph("\n".join(para))
            para.clear()

    while i < len(lines):
        line = lines[i]
        if (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and _MD_TABLE_SEP.match(lines[i + 1])
        ):
            flush()
            caption = b.recent_text[-200:].strip()
            rows = [_split_md_row(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_md_row(lines[i]))
                i += 1
            b.add_table(rows, caption=caption)
            continue
        if not line.strip():
            flush()
        else:
            para.append(line.rstrip())
        i += 1
    flush()


def _parse_pdf(data: bytes, b: _Builder) -> None:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for block in re.split(r"\n\s*\n", text):
            b.add_paragraph(re.sub(r"[ \t]+\n", "\n", block), page=page_no)


def _parse_html(data: bytes, b: _Builder) -> None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    body = soup.body or soup
    block_tags = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}
    for el in body.find_all(block_tags):
        if el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [
                    _clean_cell(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])
                ]
                cells = _merge_sec_cells(cells)
                if any(cells):
                    rows.append(cells)
            if rows:
                b.add_table(rows)
            continue
        if el.find_parent("table") or el.find(block_tags):
            continue
        b.add_paragraph(el.get_text(" ", strip=True))


def _clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _merge_sec_cells(cells: list[str]) -> list[str]:
    """SEC filings split '$', the number, and ')' or '%' into separate cells; glue them back."""
    if not cells:
        return []
    out: list[str] = [cells[0]]
    for c in cells[1:]:
        if not c:
            continue
        if out and c in (")", "%", ")%"):
            out[-1] += c
        elif out and out[-1] in ("$", "(", "$(", "($"):
            out[-1] += c
        else:
            out.append(c)
    return out


def _parse_csv(raw: str, b: _Builder, caption: str | None = None) -> None:
    rows = list(csv.reader(io.StringIO(raw)))
    b.add_table(rows, caption=caption)


def _parse_xlsx(data: bytes, b: _Builder) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    for ws in wb.worksheets:
        rows = [
            ["" if v is None else str(v) for v in row] for row in ws.iter_rows(values_only=True)
        ]
        b.add_paragraph(f"Sheet: {ws.title}")
        b.add_table(rows, caption=ws.title)


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_bytes(name: str, data: bytes) -> ParsedDocument:
    ext = Path(name).suffix.lower()
    b = _Builder()
    if ext in (".txt", ".md", ".markdown", ""):
        content_type = "text/markdown" if ext in (".md", ".markdown") else "text/plain"
        _parse_markdown(_decode(data), b)
    elif ext == ".pdf":
        content_type = "application/pdf"
        _parse_pdf(data, b)
    elif ext in (".html", ".htm"):
        content_type = "text/html"
        _parse_html(data, b)
    elif ext == ".csv":
        content_type = "text/csv"
        _parse_csv(_decode(data), b, caption=Path(name).stem)
    elif ext in (".xlsx", ".xlsm"):
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        _parse_xlsx(data, b)
    else:
        raise UnsupportedFormat(f"unsupported file type: {ext}")
    return ParsedDocument(
        name=name,
        content_type=content_type,
        sha256=hashlib.sha256(data).hexdigest(),
        text=b.text(),
        passages=b.passages,
    )


def parse_text(name: str, text: str) -> ParsedDocument:
    return parse_bytes(name if Path(name).suffix else f"{name}.md", text.encode())


def parse_path(path: Path) -> ParsedDocument:
    return parse_bytes(path.name, path.read_bytes())
