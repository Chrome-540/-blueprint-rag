import hashlib
import json
import os
import re
from pathlib import Path
import pymupdf  # PyMuPDF
from pydantic import BaseModel


class RawSheetRecord(BaseModel):
    version: str
    page_number: int
    sheet_id: str          # e.g., "A001", "LS101", "S101", "S202"
    canonical_id: str      # Unique key within a version
    sheet_title: str
    revision_date: str | None  # the date printed in the title block
    revisions: list[str] = []  # e.g. ["#1 04-09-26"]
    text_hash: str         # SHA-256 of normalized CAD text for instant diff checks
    image_path: str
    raw_cad_text: str


# Matches A001, LS101, S101, S202, A403, S-202, M1.01, FP101A, etc.
SHEET_ID_PATTERN = re.compile(r"^[A-Z]{1,3}[-.]?\d{2,3}(?:\.\d+)?[A-Z]?$")


DATE_PATTERN = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
REVISION_PATTERN = re.compile(r"#\d+\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")


def page_lines(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    """Every text line of the page, with its box as fractions (0..1) of the visible page."""
    w, h = page.rect.width, page.rect.height
    out = []
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        for line in block.get("lines", []):
            text = " ".join("".join(span["text"] for span in line["spans"]).split())
            if text:
                r = pymupdf.Rect(line["bbox"]) * page.rotation_matrix  # sheets are stored rotated
                r.normalize()
                out.append((text, pymupdf.Rect(r.x0 / w, r.y0 / h, r.x1 / w, r.y1 / h)))
    return out


def read_title_block(lines: list[tuple[str, pymupdf.Rect]]) -> dict:
    """
    Sheet ID, title, date and revisions, read by POSITION: the text under the labels 'SHEET:', 'NAME:',
    'REVISIONS:' and next to 'DATE:', which sit in one column. Anything not found, or found more than
    once, stays empty: no value is better than a wrong one.
    """
    def labels(name: str) -> list[tuple[str, pymupdf.Rect]]:
        return [(t, r) for t, r in lines if re.match(name + r"\s?:", t, re.IGNORECASE)]

    blocks = []
    for _, sheet in labels("SHEET"):
        def above(name: str, below: pymupdf.Rect, reach: float) -> list[tuple[str, pymupdf.Rect]]:
            """Labels in the same column, at most `reach` (of the page height) above `below`."""
            return [(t, r) for t, r in labels(name) if abs(r.x0 - sheet.x0) < 0.01 and 0 < below.y0 - r.y0 < reach]

        def inside(top: float, bottom: float) -> list[str]:
            """Text of the title block column between two heights, in reading order."""
            found = [(round((r.y0 + r.y1) / 2, 3), r.x0, t) for t, r in lines
                     if top < (r.y0 + r.y1) / 2 < bottom and r.x0 > sheet.x0 - 0.01 and r.x1 < sheet.x0 + 0.06]
            return [t for _, _, t in sorted(found)]

        names = above("NAME", sheet, 0.06)
        if len(names) != 1:
            continue  # 'SHEET:' somewhere else on the drawing
        name = names[0][1]
        ids = {t.replace(" ", "") for t in inside(sheet.y1, sheet.y1 + 0.05)}
        ids = {t for t in ids if SHEET_ID_PATTERN.fullmatch(t)}
        block = {"sheet_id": ids.pop() if len(ids) == 1 else None,
                 "sheet_title": " ".join(inside(name.y1, sheet.y0)) or None, "date": None, "revisions": []}

        dates = above("DATE", name, 0.06)
        if len(dates) == 1:
            date_text, date = dates[0]
            row = [t for t, r in lines if abs(r.y0 - date.y0) < 0.005 and date.x1 - 0.005 < r.x0 < sheet.x0 + 0.06]
            values = {v for t in [date_text, *row] for v in DATE_PATTERN.findall(t)}
            block["date"] = values.pop() if len(values) == 1 else None
            revisions = above("REVISIONS", date, 0.2)
            if len(revisions) == 1:
                block["revisions"] = [m.group(0) for t in inside(revisions[0][1].y1, date.y0)
                                      for m in REVISION_PATTERN.finditer(t)]
        blocks.append(block)
    return blocks[0] if len(blocks) == 1 else {"sheet_id": None, "sheet_title": None, "date": None, "revisions": []}


def read_pages(pdf_path: str, pages: list[int]) -> list[tuple[int, str, dict]]:
    """(page number, text, title block) of the given pages (numbers start at 1). Runs in a worker process."""
    with pymupdf.open(pdf_path) as doc:  # closed after each batch, so the PDF is not left locked
        return [(n, doc[n - 1].get_text("text").strip(), read_title_block(page_lines(doc[n - 1]))) for n in pages]


def render_pages(pdf_path: str, jobs: list[tuple[int, str]], dpi: int = 200) -> None:
    """Save (page number, image path) pages as PNG, skipping existing ones. Runs in a worker process."""
    with pymupdf.open(pdf_path) as doc:
        for n, path in jobs:
            if not Path(path).exists():
                part = path + ".part"  # renamed only when complete: never a half-written image
                doc[n - 1].get_pixmap(dpi=dpi).save(part, output="png")
                os.replace(part, path)


def build_records(texts: list[tuple[int, str, dict]], version: str, img_dir: Path) -> list[RawSheetRecord]:
    """One record per page, in page order (duplicate sheet numbers depend on that order)."""
    records: list[RawSheetRecord] = []
    seen_sheet_ids: dict[str, int] = {}
    for page_num, raw_cad_text, block in sorted(texts, key=lambda x: x[0]):
        sheet_id = block["sheet_id"] or f"UNKNOWN_P{page_num:03d}"

        # Ensure canonical_id is unique even if a consultant duplicates a sheet number
        count = seen_sheet_ids.get(sheet_id, 0) + 1
        seen_sheet_ids[sheet_id] = count
        canonical_id = sheet_id if count == 1 else f"{sheet_id}_DUP{count}"

        # Deterministic SHA-256 hash of normalized CAD text
        normalized_text = " ".join(raw_cad_text.split())
        records.append(
            RawSheetRecord(
                version=version,
                page_number=page_num,
                sheet_id=sheet_id,
                canonical_id=canonical_id,
                sheet_title=block["sheet_title"] or "UNTITLED SHEET",
                revision_date=block["date"],
                revisions=block["revisions"],
                text_hash=hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
                image_path=str(img_dir / f"p{page_num:03d}_{canonical_id}.png"),
                raw_cad_text=raw_cad_text,
            )
        )
    return records


def save_manifest(records: list[RawSheetRecord], out_path: Path) -> None:
    (out_path / "manifest.json").write_text(
        json.dumps([r.model_dump() for r in records], indent=2), encoding="utf-8"
    )


def ingest_blueprint_pdf(
    pdf_path: str,
    version: str,
    output_dir: str = "./blueprint_data",
    dpi: int = 200,
) -> list[RawSheetRecord]:
    """Single-process version for the command line; the app runs the same steps in parallel (pipeline.py)."""
    out_path = Path(output_dir) / version
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as doc:
        pages = list(range(1, doc.page_count + 1))
    records = build_records(read_pages(pdf_path, pages), version, img_dir)
    render_pages(pdf_path, [(r.page_number, r.image_path) for r in records], dpi)
    save_manifest(records, out_path)
    print(f"Saved {len(records)} pages -> {out_path / 'manifest.json'}")
    return records


if __name__ == "__main__":
    ingest_blueprint_pdf(
        pdf_path="Full Stamped Set For Construction 06-10-26 v2.pdf",
        version="v2",
    )